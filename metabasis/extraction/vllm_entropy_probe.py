"""On-device per-position entropy for the prod (vLLM) lane.

UNSTAMPED (C§8). The lane is certified for science cells only if the §3 bridge
passes; this module produces the quantity the bridge compares.

THE QUANTITY, AND IT IS THE ENGINE'S, NOT A NEW ONE.
`run_behavioral_cells.per_position_entropy_and_nll` defines it:

    lp  = log_softmax(logits.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)

over the slice `[P-1, P+n_generated-1)` -- the distributions that PRODUCED the
generated tokens, never the prompt's own positions, never padding. This module
computes exactly that, in float32, and the arithmetic is transcribed rather than
re-derived so a reader can diff the two lines.

WHAT IS ON-DEVICE AND WHY IT MATTERS. The reduction happens on the GPU and only a
SCALAR PER POSITION crosses to the host. A [batch, vocab] logits block at B=80,
V=152k is 48 MB per step; the entropy of it is 320 bytes. That is the whole point:
the byte-exact lane's probe pass costs 52.7% of a column's wall because it
re-forwards every generation twice, and shipping vocab-sized tensors to score them
would simply move that cost rather than remove it.

TWO PASSES, BECAUSE THE TWO HALVES OF THE READ ARE DIFFERENT MEASUREMENTS.

  * STEERED, AT GENERATION TIME (`mode="generation"`). The steering layer is
    attached and dosed; as vLLM computes each decode step's logits we reduce them
    to entropy in place. This is the efficiency win: it costs ONE extra kernel on
    a tensor that already exists, and it REMOVES the byte-exact lane's steered
    re-forward entirely.

  * UNSTEERED, TEACHER-FORCED (`mode="scoring"`). The generated token ids are
    replayed as a scoring request with `prompt_logprobs` set, which is what makes
    vLLM compute logits at EVERY prompt position rather than only the last. The
    steering spec is set to None for this pass, so the unsteered distribution is
    scored over exactly the tokens the steered pass produced -- the engine's own
    convention (`probe(..., steered=False)` re-forwards the same `input_ids`).

ALIGNMENT IS EXACT BY CONSTRUCTION, NOT BY ROW ARITHMETIC. The scoring pass gets
the full sequence and we slice it with that row's own `prompt_length` and
`n_generated`, exactly as the engine does. The generation-time path is validated
AGAINST the scoring path (`cross_check_generation_vs_scoring`) rather than trusted:
if vLLM's batching ever reorders rows under us, the cross-check is what catches it.

THIS IS A CROSS-STACK COMPARISON, NOT A BITWISE ONE. Different torch (2.9.1 vs
2.11.0), different attention backend (FlashInfer vs eager), fp32 GPU reduction vs
fp32 CPU. The selftest states a TOLERANCE and reports the realised deltas; a
bitwise claim here would be false and the pre-statement's §2 already says so.

NODE SAFETY: no package management, no writes outside the caller's sidecar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch

logger = logging.getLogger("metabasis.vllm_entropy_probe")

EntropyMode = Literal["generation", "scoring", "off"]


class EntropyProbeError(RuntimeError):
    """A named refusal; the lane never returns a silently misaligned array."""


def per_position_entropy_gpu(logits: torch.Tensor) -> torch.Tensor:
    """The engine's entropy, computed on-device, returning ONE SCALAR PER ROW.

    Transcribed from `run_behavioral_cells.per_position_entropy_and_nll`:
        lp = log_softmax(logits.float(), -1); ent = -(lp.exp() * lp).sum(-1)
    The `.float()` is load-bearing -- reducing in bf16 would lose the low-entropy
    tail that the dose ladder's small doses live in.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(dim=-1)


def per_position_nll_gpu(logits: torch.Tensor, target_ids: torch.Tensor
                         ) -> torch.Tensor:
    """-log p(actual token), the likelihood rung the engine files beside entropy."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -lp.gather(-1, target_ids.reshape(-1, 1).to(lp.device)).reshape(-1)


@dataclass
class EntropyCapture:
    """Ordered per-row entropy scalars for the CURRENT pass.

    Nothing vocab-sized is retained: `rows` holds float32 scalars and `counts`
    holds how many rows each `compute_logits` call contributed, which is what lets
    a caller reconstruct step structure without us guessing request identity.
    """

    mode: EntropyMode = "off"
    rows: list[np.ndarray] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    calls: int = 0
    rows_seen: int = 0

    def reset(self, mode: EntropyMode = "off") -> None:
        self.mode = mode
        self.rows.clear()
        self.counts.clear()
        self.calls = 0
        self.rows_seen = 0

    def flat(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self.rows).astype(np.float32)


def attach_entropy_capture(model: Any, capture: EntropyCapture
                           ) -> Callable[[], None]:
    """Wrap `model.compute_logits` so the reduction happens where the logits are.

    Returns a detach callable. The wrapper NEVER alters the logits it returns --
    it reads them, reduces a copy to scalars, and hands the original back
    untouched, so sampling downstream is exactly what vLLM would have done.
    """
    original = getattr(model, "compute_logits", None)
    if original is None or not callable(original):
        raise EntropyProbeError(
            "this model exposes no callable `compute_logits`; the on-device "
            "entropy path has nowhere to attach and the lane refuses to fall back "
            "to shipping vocab-sized tensors")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if capture.mode != "off" and isinstance(out, torch.Tensor) and out.dim() == 2:
            with torch.no_grad():
                ent = per_position_entropy_gpu(out)
            capture.rows.append(ent.detach().to("cpu", torch.float32).numpy())
            capture.counts.append(int(out.shape[0]))
            capture.calls += 1
            capture.rows_seen += int(out.shape[0])
        return out                                   # the ORIGINAL, untouched

    model.compute_logits = wrapped                   # type: ignore[assignment]

    def detach() -> None:
        model.compute_logits = original              # type: ignore[assignment]

    logger.info("entropy capture attached to compute_logits (on-device reduction, "
                "scalars only)")
    return detach


def slice_generated(ent_all: np.ndarray, prompt_length: int, n_generated: int,
                    *, indexing: str = "position") -> np.ndarray:
    """The generated span of a scored sequence, under an EXPLICIT index convention.

    TWO CONVENTIONS EXIST AND THEY DIFFER BY ONE. Getting this wrong is silent:
    the array still has the right length and the means still look plausible.

      indexing="position" (THE ENGINE'S) -- `ent_all[j]` is the distribution
        sitting AT position j, which predicts token j+1. The distributions that
        produced the generated tokens are therefore `[P-1, P+n_generated-1)`, which
        is exactly the slice `per_position_entropy_and_nll` takes.

      indexing="produced" (THE vLLM SCORING PASS'S) -- `ent_all[j]` is the
        distribution that PRODUCED token j, i.e. the one sitting at position j-1.
        The same generated tokens are then `[P, P+n_generated)`.

    MEASURED, NOT ASSUMED: on a matched forward (qwen2.5-3b L26, 4 sequences) the
    engine's slice agreed with the lane's array at offset +1 to max 0.163 nats and
    disagreed at offset 0 by 6.2-6.6 nats. The offset is a CONVENTION difference,
    not a numerics one, and it is named here so it can never be re-discovered as a
    tolerance question.
    """
    if n_generated < 1:
        raise EntropyProbeError("no generated positions to score")
    if indexing == "position":
        lo = prompt_length - 1
    elif indexing == "produced":
        lo = prompt_length
    else:
        raise EntropyProbeError(
            f"unknown indexing convention {indexing!r}; the two that exist are "
            "'position' (the engine's) and 'produced' (the vLLM scoring pass's)")
    hi = lo + n_generated
    if lo < 0:
        raise EntropyProbeError(f"prompt_length {prompt_length} leaves no position "
                                "before the first generated token")
    if hi > ent_all.shape[0]:
        raise EntropyProbeError(
            f"scored sequence has {ent_all.shape[0]} positions but the generated "
            f"span needs [{lo}, {hi}) under indexing={indexing!r} -- refusing to "
            "shorten the read")
    return ent_all[lo:hi].astype(np.float32)


def infer_indexing(ent_all: np.ndarray, reference: np.ndarray, prompt_length: int,
                   n_generated: int) -> dict[str, Any]:
    """Decide the convention BY VALUE against a known-good reference span.

    Used once per (stack, model) to pin the convention, never per cell: a
    convention that needs re-inferring every call is a convention nobody has
    established, and the bridge would be comparing whatever happened to fit.
    """
    out: dict[str, Any] = {}
    for name in ("position", "produced"):
        try:
            got = slice_generated(ent_all, prompt_length, n_generated,
                                  indexing=name)
        except EntropyProbeError as exc:
            out[name] = {"error": str(exc)}
            continue
        m = min(got.size, reference.size)
        d = np.abs(got[:m].astype(np.float64) - reference[:m].astype(np.float64))
        out[name] = {"max_abs_diff": float(d.max()) if d.size else None,
                     "median_abs_diff": float(np.median(d)) if d.size else None}
    scored = {k: v["max_abs_diff"] for k, v in out.items()
              if isinstance(v, dict) and v.get("max_abs_diff") is not None}
    out["inferred"] = min(scored, key=scored.get) if scored else None
    return out


def entropy_rise(mean_steered: Sequence_f, mean_unsteered: Sequence_f) -> dict:
    """§2.6's cell read: per-generation mean, THEN mean over generations, 4 dp.

    Aggregation ORDER is the engine's (`entropy_rise_from_rows`), because a mean of
    means over unequal-length generations is not the mean of the pooled positions
    and the two lanes must compute the same statistic to be comparable at all.
    """
    s = [float(x) for x in mean_steered]
    u = [float(x) for x in mean_unsteered]
    if not s or len(s) != len(u):
        raise EntropyProbeError(
            f"steered/unsteered generation counts disagree ({len(s)} vs {len(u)})")
    ms, mu = float(np.mean(s)), float(np.mean(u))
    return {"n": len(s), "mean_entropy_steered": round(ms, 4),
            "mean_entropy_unsteered": round(mu, 4),
            "entropy_rise": round(ms - mu, 4)}


def cross_check_generation_vs_scoring(gen_means: "np.ndarray",
                                      score_means: "np.ndarray",
                                      *, atol: float = 5e-3
                                      ) -> dict[str, Any]:
    """Validate the generation-time path against the teacher-forced one.

    They compute the same distribution's entropy by two routes -- one during
    incremental decoding with a KV cache, one in a single full-sequence forward --
    so they agree up to kernel and accumulation-order differences, not bitwise.
    A DISAGREEMENT BEYOND TOLERANCE MEANS ROW MISALIGNMENT, which is the failure
    this function exists to catch: vLLM may reorder rows across steps, and a
    silently permuted entropy array would corrupt every per-generation statistic
    downstream while looking entirely plausible.
    """
    a = np.asarray(gen_means, dtype=np.float64).reshape(-1)
    b = np.asarray(score_means, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return {"aligned": False, "reason": f"{a.size} vs {b.size} generations"}
    d = np.abs(a - b)
    return {"aligned": bool(np.all(d <= atol)), "n": int(a.size),
            "max_abs_diff": float(d.max()) if d.size else 0.0,
            "median_abs_diff": float(np.median(d)) if d.size else 0.0,
            "tolerance_atol": atol,
            "reading": ("the two routes to the same distribution's entropy; a "
                        "breach is row misalignment, not a numerics question")}


# `Sequence_f` kept local so the module has no typing import churn
from typing import Sequence as _Seq                                   # noqa: E402
Sequence_f = _Seq[float]


# ── selftest (desk-side, CPU, no vLLM, no weights) ────────────────────────────

def selftest() -> int:
    """Unit math against the ENGINE'S OWN entropy function where it is importable."""
    checks = 0
    fails: list[str] = []

    def ok(cond: bool, name: str) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(name)

    torch.manual_seed(20260807)
    V, T = 97, 11
    logits = torch.randn(T, V, dtype=torch.float32) * 2.3

    # -- 1. the on-device reduction equals a hand-computed entropy -------------
    ent = per_position_entropy_gpu(logits)
    p = torch.softmax(logits.float(), dim=-1)
    want = -(p * torch.log(p)).sum(dim=-1)
    ok(torch.allclose(ent, want, atol=1e-5), "entropy == -sum(p log p) per row")
    ok(ent.shape == (T,), "one scalar per row (nothing vocab-sized survives)")
    ok(float(ent.max()) <= float(np.log(V)) + 1e-4,
       "entropy is bounded by log(vocab)")
    uni = torch.zeros(1, V)
    ok(abs(float(per_position_entropy_gpu(uni)[0]) - float(np.log(V))) < 1e-5,
       "a uniform row reads exactly log(vocab)")
    spike = torch.full((1, V), -60.0); spike[0, 3] = 60.0
    ok(float(per_position_entropy_gpu(spike)[0]) < 1e-6,
       "a one-hot row reads ~0 entropy")

    # -- 2. AGAINST THE ENGINE'S OWN FUNCTION, same tokens, same slice ---------
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from metabasis.scripts.run_behavioral_cells import (
            per_position_entropy_and_nll)
        P, NG = 4, 6
        seq_logits = torch.randn(P + NG, V, dtype=torch.float32) * 1.7
        ids = torch.randint(0, V, (P + NG,), dtype=torch.long)
        e_ref, nll_ref = per_position_entropy_and_nll(seq_logits, ids, P, NG)
        e_all = per_position_entropy_gpu(seq_logits).numpy()
        e_ours = slice_generated(e_all, P, NG)
        ok(e_ours.shape == e_ref.shape, "engine slice shape reproduced")
        ok(float(np.max(np.abs(e_ours - e_ref))) < 1e-5,
           "ENTROPY MATCHES THE ENGINE'S OWN FUNCTION on a matched forward")
        pos = torch.arange(P - 1, P - 1 + NG)
        nll_ours = per_position_nll_gpu(seq_logits[pos], ids[P:P + NG]).numpy()
        ok(float(np.max(np.abs(nll_ours - nll_ref))) < 1e-5,
           "NLL matches the engine's own function")
    except ImportError as exc:                                  # pragma: no cover
        fails.append(f"engine comparison unavailable ({exc})")
        checks += 1

    # -- 3. the slice refuses rather than shortens -----------------------------
    for bad, name in (
            (lambda: slice_generated(np.zeros(5, np.float32), 4, 6),
             "refuses a scored sequence too short for the span"),
            (lambda: slice_generated(np.zeros(20, np.float32), 4, 6,
                                     indexing="nonsense"),
             "refuses an unknown indexing convention"),
            (lambda: slice_generated(np.zeros(20, np.float32), 0, 3),
             "refuses prompt_length with no preceding position"),
            (lambda: slice_generated(np.zeros(20, np.float32), 4, 0),
             "refuses zero generated positions"),
            (lambda: entropy_rise([1.0, 2.0], [1.0]),
             "refuses mismatched steered/unsteered counts")):
        try:
            bad()
            fails.append(name + " (did NOT raise)")
        except EntropyProbeError:
            pass
        checks += 1

    # -- 3b. the two index conventions differ by exactly one ------------------
    arr = np.arange(20, dtype=np.float32)
    a_pos = slice_generated(arr, 5, 4, indexing="position")
    a_prd = slice_generated(arr, 5, 4, indexing="produced")
    ok(a_pos.tolist() == [4.0, 5.0, 6.0, 7.0], "position indexing takes [P-1, P+NG)")
    ok(a_prd.tolist() == [5.0, 6.0, 7.0, 8.0], "produced indexing takes [P, P+NG)")
    inf = infer_indexing(arr, np.array([5.0, 6.0, 7.0, 8.0], np.float32), 5, 4)
    ok(inf["inferred"] == "produced", "infer_indexing picks the convention BY VALUE")
    inf2 = infer_indexing(arr, np.array([4.0, 5.0, 6.0, 7.0], np.float32), 5, 4)
    ok(inf2["inferred"] == "position", "infer_indexing picks the engine's when it fits")

    # -- 4. §2.6 aggregation ORDER (mean of means, not pooled) ----------------
    r = entropy_rise([1.0, 3.0], [0.5, 0.5])
    ok(r["mean_entropy_steered"] == 2.0 and r["entropy_rise"] == 1.5,
       "cell read = mean over generations of per-generation means, 4 dp")

    # -- 5. the capture keeps scalars only ------------------------------------
    cap = EntropyCapture()
    cap.reset("generation")

    class _Toy:
        def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
            return x

    m = _Toy()
    detach = attach_entropy_capture(m, cap)
    block = torch.randn(5, V)
    out = m.compute_logits(block)
    ok(out is block, "compute_logits returns the ORIGINAL logits untouched")
    ok(cap.rows_seen == 5 and cap.counts == [5], "capture counts rows per call")
    ok(cap.flat().shape == (5,) and cap.flat().dtype == np.float32,
       "capture holds float32 scalars, one per row")
    ok(cap.flat().nbytes == 20,
       "5 rows cost 20 bytes on the host (not 5 x vocab)")
    cap.reset("off")
    m.compute_logits(block)
    ok(cap.rows_seen == 0, "mode=off captures nothing")
    detach()
    ok(m.compute_logits(block) is block, "detach restores the original method")

    class _NoLogits:
        pass
    try:
        attach_entropy_capture(_NoLogits(), cap)
        fails.append("missing compute_logits (did NOT raise)")
    except EntropyProbeError:
        pass
    checks += 1

    # -- 6. the cross-check catches a permutation ------------------------------
    a = np.array([1.0, 2.0, 3.0, 4.0])
    ok(cross_check_generation_vs_scoring(a, a.copy())["aligned"],
       "identical arrays are aligned")
    ok(not cross_check_generation_vs_scoring(a, a[::-1].copy())["aligned"],
       "A PERMUTED array is caught (this is row misalignment, the real hazard)")
    ok(cross_check_generation_vs_scoring(a, a + 1e-4)["aligned"],
       "a within-tolerance cross-stack difference is accepted")
    ok(not cross_check_generation_vs_scoring(a, a[:2])["aligned"],
       "a length mismatch is caught")

    print(f"vllm_entropy_probe selftest checks run: {checks} "
          f"({len(fails)} failure(s))")
    for f in fails:
        print("  MISS " + f)
    return len(fails)


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
