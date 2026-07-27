"""Build any registry model's entropy-gradient target vector at a chosen site.

The generalization of `build_olmo_entropy_gradient.py` (the banked precedent) to
every model in the roster. The construction law is UNCHANGED — this module exists
so that seven wave-1 nodes plus the hub can be built by one audited recipe instead
of seven copies of it:

    entropy_gradient_L{site}
        = unit( P_[lo:hi](Sigma_L{site}) . mean_text( grad_state S_entropy ) )

where

  * `S_entropy` is the mean next-token entropy over the COMPLETION positions of a
    corpus text, teacher-forced (the corpus text is replayed verbatim; nothing is
    generated), exactly the object the state collector reduces over;
  * `grad_state` is the LEAF gradient with respect to the residual stream ENTERING
    decoder layer `site` — the same capture point as `collect_mean_states.py`
    (`forward_pre_hook` on `decoder_layers(model)[site]`), so the vector lives in
    the same raw residual space the transport maps were fit in;
  * `P_[lo:hi](Sigma)` projects onto the descending-eigenvalue band [lo, hi) of the
    site's residual covariance, captured on the SAME forward passes (it rides free:
    the entropy replay already materialises the residual at every position);
  * the aggregation is the precedent's: per-text mean over completion positions,
    then mean over texts, then band projection, then unit.

**Sign convention: + = entropy-increasing.** The band projection of a gradient has
non-negative inner product with that gradient by construction, so the unit vector
is entropy-increasing by construction; the finite-difference gate verifies it
empirically rather than trusting the algebra.

**The finite-difference gate is MANDATORY** (prereg ADDENDUM 2026-07-26-A). A
quantized or otherwise non-differentiable forward can return a plausible-looking
gradient that is really a residual-highway artefact — `autograd.grad` succeeds and
every construction-internal diagnostic passes. The only thing that catches it is
comparing the analytic directional derivative against a central finite difference
along the built direction:

    analytic:  D_a = n_pos * <mean_pos grad, v>          (chain rule, exact)
    measured:  D_fd = ( S(x + eps*v) - S(x - eps*v) ) / (2*eps)

with `eps*v` written into the residual at every completion position of the site
layer. A severed graph disagrees by orders of magnitude, not percent. The gate runs
an eps ladder (a too-small eps drowns in bf16 rounding, a too-large one leaves the
linear regime) and takes the best rung; a matched-support random band direction is
measured beside it as a direction-specificity control.

Known traps, all guarded here (REPORT-dsv3-fp8-lane-design 2026-07-26 section 5):
  * leaf-hook RE-ENTRY — the pre-hook counts its calls per forward and raises if a
    layer forward fires more than once (activation checkpointing would silently
    overwrite the leaf). Checkpointing is therefore never enabled: with
    `requires_grad_(False)` on every parameter, autograd builds a graph only from
    the leaf onward, so only layers >= site retain activations anyway.
  * `requires_grad_(False)` on everything but the leaf — no per-parameter gradient
    buffers are ever allocated (the difference between ~20 GB peak and OOM on a
    32B), and `autograd.grad(S, leaf)` never touches `.backward()`.
  * silent no-autograd forwards — `autograd.grad(..., allow_unused=False)` raises
    if the substituted leaf never reached the loss, and the FD gate catches the
    subtler case where it did reach it through the wrong path.

CPU self-test (no weights, no GPU, ~20 s):

    python -m metabasis.scripts.build_entropy_gradient --selftest

builds the vector end to end on a randomly initialised tiny Llama in float32 (where
the FD gate should agree to a fraction of a percent) and then plants a deliberately
severed gradient to prove the gate REJECTS it.

Node-side run (one card, pinned; the trunk stamp records `cuda_visible_devices`):

    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
    python -m metabasis.scripts.build_entropy_gradient \
        --model phi-4 --model-path <LOCAL_WEIGHTS_DIR> --arm-root <ARM_ROOT> \
        --site 19 --arm native --out-dir <ARM_ROOT>/staging/phi-4/vectors
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_entropy_gradient")

# ---------------------------------------------------------------- frozen constants
#: descending-eigenvalue band of the site covariance; the precedent's recipe, stamped
#: on every banked reference vector. Overridable ONLY for the CPU self-test (a 64-dim
#: toy model has no 256th eigenvector); production builds must not move it.
BAND_LO = 16
BAND_HI = 256
#: matched-support random band members, precedent seed (banked convention).
NULL_SEED = 20260714
N_RANDOM_BAND = 3
#: position budgets, matched to the precedent (60 texts x ~512 completion tokens
#: ~= the reference covariance's ~30k positions; 20 texts for the gradient).
N_SIGMA = 60
N_GRAD = 20
RIDGE_REL = 1e-3
#: finite-difference gate defaults.
FD_EPS_FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.03, 0.10, 0.30)
FD_N_PROBE = 3
FD_REL_TOL = 0.25

CANONICAL_VECTOR_KEY = "entropy_gradient_L{site}"
CANONICAL_RANDOM_KEY = "random_band{i}_L{site}"


# ---------------------------------------------------------------- error taxonomy
class EntropyGradientBuildError(RuntimeError):
    """Base class for every failure specific to this builder."""


class SiteOutOfRangeError(EntropyGradientBuildError):
    """The requested site is not a decoder-layer index of the loaded model."""


class BandOutOfRangeError(EntropyGradientBuildError):
    """The covariance band does not fit the model's hidden dimension."""


class CorpusSelectionError(EntropyGradientBuildError):
    """The corpus cannot supply the requested number of texts."""


class LeafHookError(EntropyGradientBuildError):
    """The leaf-substitution hook did not fire exactly once for a forward pass."""


class GradientPathError(EntropyGradientBuildError):
    """`autograd.grad` could not reach the substituted leaf, or returned garbage."""


class FiniteDifferenceGateNotPassed(EntropyGradientBuildError):
    """The central finite difference disagrees with the analytic gradient."""


# ---------------------------------------------------------------- typed records
class BuildRequest(BaseModel):
    """Everything that defines a build, validated before a single weight is read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: str = Field(description="bank key; keys the npz filename and the stamp")
    model_path: str = Field(description="local weights dir (node-side)")
    arm_root: Path = Field(description="collection root holding corpus/corpus_manifest.json")
    out_dir: Path
    site: int = Field(ge=0, description="decoder-layer index; residual ENTERING this layer")
    arm: str = Field(description="template arm; 'native' is the wave-1 family of record")
    n_sigma: int = Field(gt=0, default=N_SIGMA)
    n_grad: int = Field(gt=0, default=N_GRAD)
    band_lo: int = Field(ge=0, default=BAND_LO)
    band_hi: int = Field(gt=0, default=BAND_HI)
    ridge_rel: float = Field(gt=0.0, default=RIDGE_REL)
    device: str = "cuda"
    fd_eps_fractions: tuple[float, ...] = FD_EPS_FRACTIONS
    fd_n_probe: int = Field(ge=1, default=FD_N_PROBE)
    fd_rel_tol: float = Field(gt=0.0, default=FD_REL_TOL)
    entropy_chunk: int = Field(ge=0, default=0,
                               description="0 = the precedent's whole-sequence log_softmax; "
                                           ">0 chunks it (OOM fallback, same algebra)")

    @field_validator("arm")
    @classmethod
    def _known_arm(cls, v: str) -> str:
        if v not in ("native", "raw"):
            raise ValueError(f"unknown arm {v!r} (valid: native, raw)")
        return v

    @field_validator("fd_eps_fractions")
    @classmethod
    def _positive_fractions(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("the FD gate needs at least one eps rung")
        if any(f <= 0.0 for f in v):
            raise ValueError(f"eps fractions must be positive, got {v}")
        return tuple(sorted(v))

    @model_validator(mode="after")
    def _band_ordered(self) -> "BuildRequest":
        if self.band_lo >= self.band_hi:
            raise ValueError(f"band [{self.band_lo}:{self.band_hi}] is empty or reversed")
        if self.fd_n_probe > self.n_grad:
            raise ValueError(f"fd_n_probe={self.fd_n_probe} > n_grad={self.n_grad}")
        return self


class FDProbe(BaseModel):
    """One (text, eps) finite-difference measurement along the built direction."""

    model_config = ConfigDict(frozen=True)

    text_id: str
    n_positions: int
    eps: float
    #: n_pos * <per-text mean gradient, v>: the exact directional derivative of the
    #: per-text mean entropy under a simultaneous eps*v write at every completion
    #: position (the chain rule, no approximation).
    analytic_slope: float
    fd_slope: float
    entropy_base: float
    entropy_plus: float
    entropy_minus: float
    rel_error: float
    sign_agrees: bool


class FDRung(BaseModel):
    """One eps rung of the ladder, aggregated over probe texts."""

    model_config = ConfigDict(frozen=True)

    eps_fraction: float
    probes: list[FDProbe]
    median_rel_error: float
    max_rel_error: float
    all_signs_agree: bool


class FDGateResult(BaseModel):
    """The mandatory finite-difference gate (prereg ADDENDUM 2026-07-26-A)."""

    model_config = ConfigDict(frozen=True)

    criterion: str
    rel_tolerance: float
    rungs: list[FDRung]
    best_eps_fraction: float
    best_median_rel_error: float
    best_max_rel_error: float
    all_analytic_slopes_positive: bool
    all_signs_agree_at_best: bool
    #: direction-specificity control: the same FD along a matched-support RANDOM band
    #: direction. A real gradient makes the built direction the steepest in the band,
    #: so |D_fd(v)| should dominate |D_fd(random)| by a wide margin.
    control_random_fd_slopes: list[float]
    control_builtdir_fd_slopes: list[float]
    control_ratio_median: float
    direction_specific: bool
    passed: bool


class ConstructionDiagnostics(BaseModel):
    """Construction-internal health of the direction (needs no external axis bank)."""

    model_config = ConfigDict(frozen=True)

    hidden_dim: int
    n_decoder_layers: int
    n_sigma_texts: int
    n_sigma_positions: int
    n_grad_texts: int
    n_grad_positions: int
    #: how many per-text band-projected gradients point the same way as the mean
    per_text_band_sign_consistency: str
    per_text_band_pairwise_coherence: float
    #: fraction of the mean gradient's norm that survives the band projection
    band_energy_fraction: float
    mean_gradient_norm: float
    band_projected_norm: float
    vector_norm: float
    vector_dtype: str
    median_token_resid_norm: float
    sigma_ridge: float
    sigma_top_eigenvalue: float
    sigma_band_eigenvalue_lo: float
    sigma_band_eigenvalue_hi: float
    mean_entropy_nats: float


class BuildResult(BaseModel):
    """The full in-memory result; everything banked is derived from this."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    request: BuildRequest
    vector: Any                       # np.ndarray float32 [d], unit
    random_band: list[Any]            # list[np.ndarray] float32 [d], unit
    sigma_evals: Any
    sigma_evecs: Any
    sigma_mean: Any
    diagnostics: ConstructionDiagnostics
    fd_gate: FDGateResult
    sigma_text_ids: list[str]
    grad_text_ids: list[str]
    wall_seconds: float


# ---------------------------------------------------------------- text selection
def stride_indices(n_total: int, n_pick: int, half_step_offset: bool) -> list[int]:
    """The precedent's deterministic spread over a frozen, ordered corpus.

    `_sigma_gids` / `_grad_gids` in `build_olmo_entropy_gradient.py` are
    `all[::step][:n]` and `all[step//2::step][:n]`; this is the same rule expressed
    over manifest positions. The half-step offset makes the gradient texts disjoint
    from the covariance texts whenever `step > 1` (the precedent's "fresh gens").
    """
    if n_pick > n_total:
        raise CorpusSelectionError(
            f"asked for {n_pick} texts but the corpus has {n_total}")
    step = max(1, n_total // n_pick)
    start = step // 2 if half_step_offset else 0
    idx = list(range(start, n_total, step))[:n_pick]
    if len(idx) < n_pick:                                    # pragma: no cover
        raise CorpusSelectionError(
            f"stride selection yielded {len(idx)} of {n_pick} texts "
            f"(n_total={n_total}, step={step}, start={start})")
    return idx


# ---------------------------------------------------------------- the numeric core
def _mean_next_token_entropy(logits: Any, p: int, n: int, chunk: int) -> Any:
    """Mean next-token Shannon entropy (nats) over completion positions [p, n).

    `chunk == 0` is the precedent's path: one `log_softmax` over the whole sequence,
    then slice and mean. `chunk > 0` computes the same quantity in position blocks
    (fp32 summation order differs; it is an OOM fallback, recorded in the stamp).
    """
    import torch

    if chunk > 0:
        total = None
        for s in range(p, n, chunk):
            e = min(s + chunk, n)
            lp = torch.log_softmax(logits[s:e].float(), dim=-1)
            part = (-(lp.exp() * lp).sum(dim=-1)).sum()
            total = part if total is None else total + part
        assert total is not None                              # n > p is guaranteed
        return total / float(n - p)
    lp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)
    return ent[p:n].mean()


class _LeafSubstitution:
    """`forward_pre_hook` replacing the residual entering `site` with a grad leaf.

    Under `no_grad` this is a value-identical passthrough (the covariance pass);
    under `enable_grad` it is what makes `autograd.grad(S, leaf)` well defined.
    The call counter is the re-entry guard: activation checkpointing (or any other
    double-invocation of the layer's forward) would silently overwrite the leaf with
    the recompute pass's copy, and the gradient would be taken w.r.t. a tensor that
    never fed the loss.
    """

    def __init__(self, layer: Any) -> None:
        self.leaf: Any = None
        self.calls: int = 0
        self._handle = layer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module: Any, args: tuple, kwargs: dict) -> tuple:
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None:
            raise LeafHookError(
                f"{type(module).__name__}: no hidden_states in args or kwargs — "
                "the leaf-substitution hook cannot find the residual stream")
        self.calls += 1
        leaf = hs.detach().clone().requires_grad_(True)
        self.leaf = leaf
        if args:
            return (leaf,) + tuple(args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = leaf
        return args, new_kwargs

    def arm(self) -> None:
        self.calls = 0
        self.leaf = None

    def check_single_call(self, what: str) -> Any:
        if self.calls != 1:
            raise LeafHookError(
                f"{what}: the site layer's forward fired {self.calls} times in one "
                "model forward (expected exactly 1). Re-entry (activation "
                "checkpointing?) would make the leaf gradient meaningless.")
        if self.leaf is None:                                 # pragma: no cover
            raise LeafHookError(f"{what}: hook fired but captured no leaf")
        return self.leaf

    def remove(self) -> None:
        self._handle.remove()


def _forward_entropy(model: Any, ids: Any, p: int, n: int, chunk: int) -> float:
    """One no-grad forward -> scalar mean completion entropy (nats)."""
    import torch

    with torch.no_grad():
        out = model(ids, use_cache=False, return_dict=True)
        return float(_mean_next_token_entropy(out.logits[0], p, n, chunk).item())


def build_entropy_gradient(model: Any, tok: Any, entries: Sequence[dict],
                           request: BuildRequest, date_string: str,
                           device: str,
                           gradient_saboteur: Optional[Any] = None) -> BuildResult:
    """The whole construction, device- and dtype-agnostic (the self-test runs it on CPU).

    `gradient_saboteur` is a self-test hook ONLY: a callable applied to each per-text
    mean gradient before aggregation, used to plant a severed/bogus gradient and prove
    the finite-difference gate rejects it. It is never set on a real build, and its
    use is recorded in the stamp if it ever is.
    """
    import torch

    from metabasis.extraction.hooks import ResidualWriteSpec, attach_residual_write
    from metabasis.scripts.collect_mean_states import build_ids

    t0 = time.time()
    layers = _decoder_layers(model)
    n_layers = len(layers)
    site = request.site
    if not 0 <= site < n_layers:
        raise SiteOutOfRangeError(
            f"site L{site} out of range: {request.model_key} has {n_layers} decoder layers")
    hidden_dim = int(_hidden_dim(model))
    if request.band_hi > hidden_dim:
        raise BandOutOfRangeError(
            f"band [{request.band_lo}:{request.band_hi}] does not fit hidden_dim="
            f"{hidden_dim} for {request.model_key}")

    sigma_pos = stride_indices(len(entries), request.n_sigma, half_step_offset=False)
    grad_pos = stride_indices(len(entries), request.n_grad, half_step_offset=True)
    sigma_entries = [entries[i] for i in sigma_pos]
    grad_entries = [entries[i] for i in grad_pos]
    logger.info("%s L%d/%d (d=%d): %d covariance texts, %d gradient texts, arm=%s",
                request.model_key, site, n_layers, hidden_dim,
                len(sigma_entries), len(grad_entries), request.arm)

    hook = _LeafSubstitution(layers[site])
    try:
        # ── covariance pass: residual rows at the site over completion positions ──
        rows: list[np.ndarray] = []
        tok_norms: list[float] = []
        for i, e in enumerate(sigma_entries):
            ids_list, p = build_ids(tok, e, request.arm, date_string)
            n = len(ids_list)
            if n - p <= 0:
                raise CorpusSelectionError(f"{e['text_id']}: no completion positions")
            ids = torch.tensor([ids_list], dtype=torch.long, device=device)
            hook.arm()
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            leaf = hook.check_single_call(f"covariance pass, {e['text_id']}")
            r = leaf[0, p:n, :].detach().float().cpu().numpy().astype(np.float64)
            rows.append(r)
            tok_norms.append(float(np.median(np.linalg.norm(r, axis=1))))
            if (i + 1) % 20 == 0:
                logger.info("covariance %d/%d texts (%.2fs/text)", i + 1,
                            len(sigma_entries), (time.time() - t0) / (i + 1))
        R = np.concatenate(rows, axis=0)
        del rows
        mu = R.mean(0)
        Rc = R - mu
        Sigma = (Rc.T @ Rc) / (Rc.shape[0] - 1)
        del Rc
        n_sigma_positions = int(R.shape[0])
        del R
        evals, evecs = np.linalg.eigh(Sigma)                   # ascending
        ridge = float(request.ridge_rel * float(evals.mean()))
        order = np.argsort(evals)[::-1]                        # descending
        band_idx = order[request.band_lo:request.band_hi]
        Ub = evecs[:, band_idx]                                # (d, band_hi-band_lo)
        logger.info("Sigma over %d positions (%d texts); ridge %.3g; band [%d:%d]",
                    n_sigma_positions, len(sigma_entries), ridge,
                    request.band_lo, request.band_hi)

        # ── gradient pass: leaf gradient of the mean completion entropy ──
        Ge: list[np.ndarray] = []
        probe_meta: list[dict] = []
        entropies: list[float] = []
        n_grad_positions = 0
        for i, e in enumerate(grad_entries):
            ids_list, p = build_ids(tok, e, request.arm, date_string)
            n = len(ids_list)
            if n - p <= 0:
                raise CorpusSelectionError(f"{e['text_id']}: no completion positions")
            ids = torch.tensor([ids_list], dtype=torch.long, device=device)
            hook.arm()
            with torch.enable_grad():
                out = model(ids, use_cache=False, return_dict=True)
                S = _mean_next_token_entropy(out.logits[0], p, n, request.entropy_chunk)
                leaf = hook.check_single_call(f"gradient pass, {e['text_id']}")
                try:
                    (grad_leaf,) = torch.autograd.grad(S, leaf, allow_unused=False)
                except RuntimeError as exc:
                    raise GradientPathError(
                        f"{e['text_id']}: autograd.grad(S_entropy, leaf) failed ({exc}). "
                        "The substituted leaf never reached the loss — the layer's "
                        "forward ignored the hook's returned hidden_states, or the "
                        "forward has no autograd nodes at all.") from exc
                if grad_leaf is None:                          # pragma: no cover
                    raise GradientPathError(f"{e['text_id']}: gradient is None")
                g = grad_leaf[0, p:n, :].float().mean(0).cpu().numpy().astype(np.float64)
                s_value = float(S.detach().item())
            if not np.all(np.isfinite(g)):
                raise GradientPathError(
                    f"{e['text_id']}: gradient contains non-finite entries")
            if float(np.linalg.norm(g)) == 0.0:
                raise GradientPathError(
                    f"{e['text_id']}: gradient is exactly zero — no autograd path "
                    "reached the leaf through the layers above the site")
            if gradient_saboteur is not None:
                g = np.asarray(gradient_saboteur(g), dtype=np.float64)
            Ge.append(g)
            entropies.append(s_value)
            n_grad_positions += n - p
            if i < request.fd_n_probe:
                probe_meta.append({"text_id": e["text_id"], "ids": ids_list,
                                   "p": p, "n": n, "grad": g})
            logger.info("gradient %d/%d texts (|g| %.4g, S %.4f nats)",
                        i + 1, len(grad_entries), float(np.linalg.norm(g)), s_value)
    finally:
        hook.remove()

    G = np.stack(Ge)
    mean_g = G.mean(0)
    band_coeff = Ub.T @ mean_g
    v_unscaled = Ub @ band_coeff
    band_norm = float(np.linalg.norm(v_unscaled))
    if band_norm == 0.0:
        raise GradientPathError(
            "the mean gradient has zero energy inside the covariance band — "
            "no direction to build")
    vector = (v_unscaled / band_norm).astype(np.float32)
    # sign convention: + = entropy-increasing. <mean_g, v> = ||Ub^T mean_g|| >= 0 by
    # construction; assert it rather than assume the algebra held numerically.
    if float(mean_g @ vector.astype(np.float64)) <= 0.0:       # pragma: no cover
        raise EntropyGradientBuildError(
            "built direction is not entropy-increasing — the sign convention "
            "(+ = entropy-increasing) is violated; refusing to bank it")

    # ── construction-internal diagnostics ──
    bandproj = (Ub.T @ G.T).T
    bu = bandproj / np.clip(np.linalg.norm(bandproj, axis=1, keepdims=True), 1e-12, None)
    iu = np.triu_indices(len(G), k=1)
    coherence = float((bu @ bu.T)[iu].mean()) if len(G) > 1 else float("nan")
    ref = band_coeff / np.linalg.norm(band_coeff)
    per_text_sign = int(((bandproj @ ref) > 0).sum())
    mean_g_norm = float(np.linalg.norm(mean_g))
    median_resid_norm = float(np.median(tok_norms))
    band_evals = evals[band_idx]

    # ── matched-support random band members (precedent seed + recipe) ──
    rng = np.random.default_rng(NULL_SEED)
    random_band: list[np.ndarray] = []
    for _ in range(N_RANDOM_BAND):
        c = rng.standard_normal(Ub.shape[1])
        r = Ub @ c
        random_band.append((r / np.linalg.norm(r)).astype(np.float32))

    # ── the mandatory finite-difference gate ──
    fd_gate = _run_fd_gate(model=model, request=request, layers=layers,
                           vector=vector, random_direction=random_band[0],
                           probes=probe_meta, device=device,
                           median_resid_norm=median_resid_norm,
                           residual_write=(ResidualWriteSpec, attach_residual_write))

    diagnostics = ConstructionDiagnostics(
        hidden_dim=hidden_dim,
        n_decoder_layers=n_layers,
        n_sigma_texts=len(sigma_entries),
        n_sigma_positions=n_sigma_positions,
        n_grad_texts=len(grad_entries),
        n_grad_positions=n_grad_positions,
        per_text_band_sign_consistency=f"{per_text_sign}/{len(G)}",
        per_text_band_pairwise_coherence=round(coherence, 6),
        band_energy_fraction=round(band_norm / mean_g_norm, 6) if mean_g_norm else 0.0,
        mean_gradient_norm=float(mean_g_norm),
        band_projected_norm=float(band_norm),
        vector_norm=float(np.linalg.norm(vector.astype(np.float64))),
        vector_dtype=str(vector.dtype),
        median_token_resid_norm=median_resid_norm,
        sigma_ridge=ridge,
        sigma_top_eigenvalue=float(evals[order[0]]),
        sigma_band_eigenvalue_lo=float(band_evals[0]),
        sigma_band_eigenvalue_hi=float(band_evals[-1]),
        mean_entropy_nats=float(np.mean(entropies)),
    )
    return BuildResult(
        request=request, vector=vector, random_band=random_band,
        sigma_evals=evals.astype(np.float64), sigma_evecs=evecs.astype(np.float64),
        sigma_mean=mu.astype(np.float64), diagnostics=diagnostics, fd_gate=fd_gate,
        sigma_text_ids=[e["text_id"] for e in sigma_entries],
        grad_text_ids=[e["text_id"] for e in grad_entries],
        wall_seconds=round(time.time() - t0, 1))


def _run_fd_gate(*, model: Any, request: BuildRequest, layers: Any,
                 vector: np.ndarray, random_direction: np.ndarray,
                 probes: list[dict], device: str, median_resid_norm: float,
                 residual_write: tuple) -> FDGateResult:
    """Central finite difference along the built direction, over an eps ladder.

    The leaf-substitution hook is REMOVED before this runs (the caller's `finally`
    has fired), so only one pre-hook is ever on the site layer at a time — the
    residual-write hook. Every forward here is `no_grad`.
    """
    import torch

    spec_cls, attach = residual_write
    if not probes:                                             # pragma: no cover
        raise EntropyGradientBuildError("the FD gate needs at least one probe text")

    v_t = torch.tensor(vector.astype(np.float32), dtype=torch.float32)
    r_t = torch.tensor(random_direction.astype(np.float32), dtype=torch.float32)
    v64 = vector.astype(np.float64)
    r64 = random_direction.astype(np.float64)

    spec = spec_cls(layer_idx=request.site, vector=v_t, alpha=0.0,
                    start_pos=0, end_pos=None, normalize=True)
    handle = attach(model, spec)
    rungs: list[FDRung] = []
    try:
        # baseline entropies (alpha = 0 short-circuits the hook to a passthrough)
        for pr in probes:
            pr["ids_t"] = torch.tensor([pr["ids"]], dtype=torch.long, device=device)
            spec.alpha = 0.0
            spec.start_pos = pr["p"]
            pr["entropy_base"] = _forward_entropy(model, pr["ids_t"], pr["p"], pr["n"],
                                                  request.entropy_chunk)

        for frac in request.fd_eps_fractions:
            eps = float(frac * median_resid_norm)
            measured: list[FDProbe] = []
            for pr in probes:
                n_pos = pr["n"] - pr["p"]
                analytic = float(n_pos * (pr["grad"] @ v64))
                spec.vector = v_t
                spec.start_pos = pr["p"]
                spec.alpha = eps
                s_plus = _forward_entropy(model, pr["ids_t"], pr["p"], pr["n"],
                                          request.entropy_chunk)
                spec.alpha = -eps
                s_minus = _forward_entropy(model, pr["ids_t"], pr["p"], pr["n"],
                                           request.entropy_chunk)
                fd = (s_plus - s_minus) / (2.0 * eps)
                denom = abs(analytic) if abs(analytic) > 1e-12 else 1e-12
                measured.append(FDProbe(
                    text_id=pr["text_id"], n_positions=n_pos, eps=eps,
                    analytic_slope=analytic, fd_slope=fd,
                    entropy_base=pr["entropy_base"],
                    entropy_plus=s_plus, entropy_minus=s_minus,
                    rel_error=abs(fd - analytic) / denom,
                    sign_agrees=(fd > 0) == (analytic > 0)))
            errs = [m.rel_error for m in measured]
            rungs.append(FDRung(
                eps_fraction=frac, probes=measured,
                median_rel_error=float(np.median(errs)),
                max_rel_error=float(np.max(errs)),
                all_signs_agree=all(m.sign_agrees for m in measured)))
            logger.info("FD rung eps=%.4g (%.3g x median resid norm): median rel.err "
                        "%.4f, max %.4f, signs agree %s", eps, frac,
                        rungs[-1].median_rel_error, rungs[-1].max_rel_error,
                        rungs[-1].all_signs_agree)

        best = min(rungs, key=lambda r: r.median_rel_error)
        # direction-specificity control at the best rung: the same measurement along a
        # matched-support RANDOM band direction. A real gradient makes the built
        # direction the steepest in the band by a wide margin.
        eps_best = float(best.eps_fraction * median_resid_norm)
        ctrl_rand: list[float] = []
        ctrl_built: list[float] = []
        for pr in probes:
            spec.start_pos = pr["p"]
            spec.vector = r_t
            spec.alpha = eps_best
            sp = _forward_entropy(model, pr["ids_t"], pr["p"], pr["n"],
                                  request.entropy_chunk)
            spec.alpha = -eps_best
            sm = _forward_entropy(model, pr["ids_t"], pr["p"], pr["n"],
                                  request.entropy_chunk)
            ctrl_rand.append((sp - sm) / (2.0 * eps_best))
            ctrl_built.append(
                next(m.fd_slope for m in best.probes if m.text_id == pr["text_id"]))
            _ = r64                                            # kept for provenance
    finally:
        handle.remove()

    ratios = [abs(b) / max(abs(c), 1e-12) for b, c in zip(ctrl_built, ctrl_rand)]
    ratio_median = float(np.median(ratios))
    all_positive = all(m.analytic_slope > 0 for r in rungs for m in r.probes)
    passed = bool(best.median_rel_error <= request.fd_rel_tol
                  and best.all_signs_agree and all_positive)
    return FDGateResult(
        criterion=(
            "PASS iff (a) every probe text's analytic directional derivative is "
            "POSITIVE (the + = entropy-increasing sign convention), (b) at the best "
            "eps rung every probe's central finite difference has the same sign as "
            "the analytic slope, and (c) that rung's MEDIAN relative error "
            f"|D_fd - D_a|/|D_a| <= {request.fd_rel_tol}. A severed / "
            "residual-highway-only gradient disagrees by orders of magnitude."),
        rel_tolerance=request.fd_rel_tol, rungs=rungs,
        best_eps_fraction=best.eps_fraction,
        best_median_rel_error=best.median_rel_error,
        best_max_rel_error=best.max_rel_error,
        all_analytic_slopes_positive=all_positive,
        all_signs_agree_at_best=best.all_signs_agree,
        control_random_fd_slopes=[float(x) for x in ctrl_rand],
        control_builtdir_fd_slopes=[float(x) for x in ctrl_built],
        control_ratio_median=ratio_median,
        direction_specific=bool(ratio_median > 3.0),
        passed=passed)


# ---------------------------------------------------------------- model helpers
def _decoder_layers(model: Any) -> Any:
    from metabasis.extraction.hooks import decoder_layers
    return decoder_layers(model)


def _hidden_dim(model: Any) -> int:
    cfg = model.config
    dim = getattr(cfg, "hidden_size", None)
    if dim is None:
        text_cfg = getattr(cfg, "text_config", None)
        dim = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
    if dim is None:                                            # pragma: no cover
        raise EntropyGradientBuildError(
            f"cannot resolve hidden_size on {type(cfg).__name__}")
    return int(dim)


# ---------------------------------------------------------------- banking
def bank(result: BuildResult, trunk: dict, corpus_sha: str, date_string: str,
         saboteur_used: bool = False) -> dict[str, Path]:
    """Write the vector, the covariance, the FD gate and the stamp. Returns paths."""
    req = result.request
    out = req.out_dir
    out.mkdir(parents=True, exist_ok=True)
    site = req.site
    key = req.model_key

    vectors = {CANONICAL_VECTOR_KEY.format(site=site): result.vector}
    for i, r in enumerate(result.random_band, start=1):
        vectors[CANONICAL_RANDOM_KEY.format(i=i, site=site)] = r
    vec_path = out / f"entropy_gradient_{key}.npz"
    np.savez(vec_path, **vectors)

    sigma_path = out / f"sigma_L{site}_{key}.npz"
    np.savez(sigma_path, evals=result.sigma_evals, evecs=result.sigma_evecs,
             mean=result.sigma_mean, ridge=np.float64(result.diagnostics.sigma_ridge),
             n_positions=np.int64(result.diagnostics.n_sigma_positions))

    fd_path = out / f"entropy_gradient_{key}_fd_gate.json"
    fd_path.write_text(json.dumps(
        {"model": key, "site": site, "arm": req.arm,
         "PASSES_FD_GATE": result.fd_gate.passed,
         **result.fd_gate.model_dump()}, indent=1))

    stamp = {
        "STATUS": ("UNSTAMPED (prereg C section 8) — built by "
                   "build_entropy_gradient.py; scores nothing on its own"),
        "builder": "build_entropy_gradient.py",
        "model": key,
        "site": site,
        "template_arm": req.arm,
        "band": [req.band_lo, req.band_hi],
        "npz_keys": {
            "vector": CANONICAL_VECTOR_KEY.format(site=site),
            "random_band": [CANONICAL_RANDOM_KEY.format(i=i, site=site)
                            for i in range(1, N_RANDOM_BAND + 1)]},
        "sign_convention": "+ = entropy-increasing (verified by the FD gate)",
        "vector_provenance": (
            f"unit( P_[{req.band_lo}:{req.band_hi}](Sigma_L{site}) . mean_text( "
            f"grad_state S_entropy ) ); teacher-forced replay of "
            f"{result.diagnostics.n_grad_texts} frozen-corpus texts "
            f"({result.diagnostics.n_grad_positions} completion positions) on the "
            f"{req.arm} arm; band = descending-eigenvalue slice of the residual "
            f"covariance captured on the SAME forward passes over "
            f"{result.diagnostics.n_sigma_texts} texts "
            f"({result.diagnostics.n_sigma_positions} positions). SAME construction "
            "as build_olmo_entropy_gradient.py (the banked precedent)."),
        "randoms_provenance": (
            f"{N_RANDOM_BAND} matched-support randoms confined to the "
            f"[{req.band_lo}:{req.band_hi}] band eigenspace of Sigma_L{site}, "
            f"seed {NULL_SEED} (precedent recipe)"),
        "capture_convention": (
            "forward_pre_hook on decoder_layers[L] (residual ENTERING layer L), "
            "completion positions >= P, batch-1, no cache — identical to "
            "collect_mean_states.py, so the vector lives in the same raw residual "
            "space the transport maps were fit in"),
        "corpus_manifest_sha256": corpus_sha,
        "date_string": date_string,
        "arm_root": str(req.arm_root),
        "text_ids": {"covariance": result.sigma_text_ids,
                     "gradient": result.grad_text_ids},
        "selection_rule": ("deterministic stride over the frozen corpus manifest "
                           "order: covariance = [::step], gradient = [step//2::step] "
                           "(the precedent's disjoint half-step offset)"),
        "diagnostics": result.diagnostics.model_dump(),
        "fd_gate": {
            "PASSES": result.fd_gate.passed,
            "criterion": result.fd_gate.criterion,
            "best_eps_fraction": result.fd_gate.best_eps_fraction,
            "best_median_rel_error": result.fd_gate.best_median_rel_error,
            "best_max_rel_error": result.fd_gate.best_max_rel_error,
            "direction_specificity_ratio_median": result.fd_gate.control_ratio_median,
            "detail": str(fd_path.name)},
        "entropy_chunk": req.entropy_chunk,
        "ridge_rel": req.ridge_rel,
        "wall_seconds": result.wall_seconds,
        "trunk": trunk,
    }
    if saboteur_used:
        stamp["SELFTEST_SABOTEUR_APPLIED"] = True
    stamp_path = out / f"entropy_gradient_{key}_stamps.json"
    stamp_path.write_text(json.dumps(stamp, indent=1))
    return {"vector": vec_path, "sigma": sigma_path, "fd_gate": fd_path,
            "stamp": stamp_path}


# ---------------------------------------------------------------- CPU self-test
class _ToyTokenizer:
    """The minimum surface `build_ids` touches, for a weightless CPU self-test."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.chat_template = "toy"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2 + (ord(c) % (self.vocab_size - 4)) for c in text]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True,
                            **kwargs: Any) -> list[int]:
        if "date_string" in kwargs:                # exercise the TypeError fall-back path
            raise TypeError("toy template takes no date_string")
        out = [self.bos_token_id]
        for m in messages:
            out += self.encode(m["content"], add_special_tokens=False)[:8]
        return out


def _toy_corpus(n: int) -> list[dict]:
    return [{"text_id": f"T{i:03d}", "stratum": "S1", "system_prompt": "",
             "user_prompt": f"prompt {i}",
             "text": "".join(chr(40 + ((i * 7 + j * 13) % 60)) for j in range(24))}
            for i in range(n)]


def selftest() -> int:
    """End-to-end CPU verification: real algebra, tiny random model, fp32."""
    import tempfile

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    torch.manual_seed(20260727)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).to(torch.float32).eval()
    model.requires_grad_(False)
    tok = _ToyTokenizer(cfg.vocab_size)
    entries = _toy_corpus(24)

    req = BuildRequest(
        model_key="toy", model_path="(none)", arm_root=Path("."),
        out_dir=Path("."), site=2, arm="native", n_sigma=12, n_grad=6,
        band_lo=2, band_hi=32, device="cpu", fd_n_probe=3, fd_rel_tol=0.02,
        fd_eps_fractions=(0.001, 0.01, 0.05))
    res = build_entropy_gradient(model, tok, entries, req, "12 Jul 2026", "cpu")

    d = res.diagnostics
    check("vector is unit", abs(d.vector_norm - 1.0) < 1e-5, f"|v|={d.vector_norm:.8f}")
    check("vector dim == hidden_size", res.vector.shape == (64,), str(res.vector.shape))
    check("vector is float32", res.vector.dtype == np.float32, str(res.vector.dtype))
    check("3 matched-support randoms banked", len(res.random_band) == N_RANDOM_BAND)
    check("randoms are unit",
          all(abs(float(np.linalg.norm(r.astype(np.float64))) - 1.0) < 1e-5
              for r in res.random_band))
    check("randoms lie in the band (not equal to the vector)",
          all(abs(float(r.astype(np.float64) @ res.vector.astype(np.float64))) < 0.999
              for r in res.random_band))
    # Disjointness is a property of the PRODUCTION budgets (780 corpus texts, 60/20),
    # where the gradient stride 39 is a multiple of the covariance stride 13 offset by
    # 19: every gradient index is 6 mod 13, so it can never be 0 mod 13. Small toy
    # budgets need not have it, so it is asserted where it is actually load-bearing.
    check("covariance and gradient texts are disjoint at the production budgets",
          not (set(stride_indices(780, N_SIGMA, False))
               & set(stride_indices(780, N_GRAD, True))),
          f"{N_SIGMA} covariance / {N_GRAD} gradient texts of 780")
    check("band energy fraction in (0, 1]",
          0.0 < d.band_energy_fraction <= 1.0 + 1e-9,
          f"{d.band_energy_fraction:.4f}")
    check("entropy is a sane number of nats", 0.0 < d.mean_entropy_nats < 10.0,
          f"{d.mean_entropy_nats:.4f}")

    g = res.fd_gate
    check("FD gate PASSES on an exactly differentiable fp32 model", g.passed,
          f"median rel.err {g.best_median_rel_error:.2e} at eps_frac "
          f"{g.best_eps_fraction}")
    check("FD analytic slopes are all positive (+ = entropy-increasing)",
          g.all_analytic_slopes_positive)
    check("FD signs agree at the best rung", g.all_signs_agree_at_best)
    # Direction specificity: on a REAL model the built direction is the steepest in
    # the band by a wide margin (the `direction_specific` flag wants >3x). A randomly
    # initialised toy has a near-isotropic entropy surface, so only dominance itself
    # is asserted here; the ratio is reported, and the flag is a banked diagnostic,
    # never part of the FD verdict.
    check("built direction dominates a random band direction",
          g.control_ratio_median > 1.0, f"ratio {g.control_ratio_median:.2f}")
    check("eps ladder has all requested rungs",
          len(g.rungs) == len(req.fd_eps_fractions), str(len(g.rungs)))

    # Negative control: plant a severed gradient (an arbitrary fixed direction that
    # has nothing to do with the entropy surface). The gate MUST reject it.
    rng = np.random.default_rng(7)
    bogus = rng.standard_normal(64)
    bogus /= np.linalg.norm(bogus)
    try:
        res_bad = build_entropy_gradient(
            model, tok, entries, req, "12 Jul 2026", "cpu",
            gradient_saboteur=lambda _g: bogus * float(np.linalg.norm(_g)))
        check("severed-gradient control is REJECTED by the FD gate",
              not res_bad.fd_gate.passed,
              f"median rel.err {res_bad.fd_gate.best_median_rel_error:.3g}")
    except EntropyGradientBuildError as exc:
        # a sign-convention violation is also a rejection, and a louder one
        check("severed-gradient control is REJECTED (sign convention)", True, str(exc)[:80])

    # Guard coverage
    try:
        BuildRequest(model_key="x", model_path="p", arm_root=Path("."), out_dir=Path("."),
                     site=0, arm="sideways")
        check("unknown arm rejected", False)
    except Exception as exc:
        check("unknown arm rejected", "sideways" in str(exc))
    try:
        BuildRequest(model_key="x", model_path="p", arm_root=Path("."), out_dir=Path("."),
                     site=0, arm="native", band_lo=64, band_hi=16)
        check("reversed band rejected", False)
    except Exception as exc:
        check("reversed band rejected", "band" in str(exc))
    try:
        stride_indices(10, 20, False)
        check("over-sized text request rejected", False)
    except CorpusSelectionError:
        check("over-sized text request rejected", True)
    try:
        bad_req = req.model_copy(update={"site": 99})
        build_entropy_gradient(model, tok, entries, bad_req, "12 Jul 2026", "cpu")
        check("out-of-range site rejected", False)
    except SiteOutOfRangeError:
        check("out-of-range site rejected", True)
    try:
        bad_req = req.model_copy(update={"band_hi": 4096})
        build_entropy_gradient(model, tok, entries, bad_req, "12 Jul 2026", "cpu")
        check("band wider than hidden_dim rejected", False)
    except BandOutOfRangeError:
        check("band wider than hidden_dim rejected", True)
    check("stride rule matches the precedent's _sigma_gids",
          stride_indices(780, 60, False)[:4] == [0, 13, 26, 39])
    check("stride rule matches the precedent's _grad_gids",
          stride_indices(780, 20, True)[:4] == [19, 58, 97, 136])

    # Banking round-trip through the canonical keys, read back by the READOUT's loader.
    with tempfile.TemporaryDirectory() as td:
        banked_req = req.model_copy(update={"out_dir": Path(td)})
        banked = res.model_copy(update={"request": banked_req})
        paths = bank(banked, {"device": "cpu", "cuda_visible_devices": "(unset)"},
                     "0" * 64, "12 Jul 2026")
        from metabasis.scripts.read_exchange_rates import (
            load_entropy_gradient, load_random_band)
        v, spec = load_entropy_gradient(paths["vector"], "toy", 2)
        check("readout loads the canonical key entropy_gradient_L2",
              spec.key == "entropy_gradient_L2" and spec.key_convention == "canonical",
              f"{spec.key} ({spec.key_convention})")
        check("readout round-trip is bit-faithful (fp32)",
              float(np.max(np.abs(v - res.vector.astype(np.float64)))) < 1e-6)
        check("readout finds the matched-support random band",
              len(load_random_band(paths["vector"], 2)) == N_RANDOM_BAND)
        check("stamp and FD-gate artefacts written",
              paths["stamp"].exists() and paths["fd_gate"].exists()
              and paths["sigma"].exists())
        stamp = json.loads(paths["stamp"].read_text())
        check("stamp carries cuda_visible_devices",
              "cuda_visible_devices" in stamp["trunk"])
        check("stamp records the FD verdict", stamp["fd_gate"]["PASSES"] is True)

    n_miss = sum(1 for _, ok, _ in checks if not ok)
    print(json.dumps({"checks": len(checks), "misses": n_miss,
                      "failed": [n for n, ok, _ in checks if not ok]}, indent=1))
    if n_miss:
        print(f"SELFTEST-NOT-CLEAN: {n_miss} of {len(checks)} checks did not hold")
        return 1
    print(f"SELFTEST-CLEAN: {len(checks)}/{len(checks)} checks hold")
    return 0


# ---------------------------------------------------------------- CLI
def _parse_fractions(spec: str) -> tuple[float, ...]:
    try:
        return tuple(float(x) for x in spec.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"--fd-eps-fractions: cannot parse {spec!r} ({exc})") from exc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="CPU end-to-end verification (no weights, no GPU)")
    ap.add_argument("--model", help="bank key (keys the npz filename and the stamp)")
    ap.add_argument("--model-path", help="local weights dir (node-side)")
    ap.add_argument("--arm-root", type=Path,
                    help="collection root holding corpus/corpus_manifest.json")
    ap.add_argument("--out-dir", type=Path,
                    help="where the vector lands; the banking convention is "
                         "<collection>/<key>/vectors/")
    ap.add_argument("--site", type=int, help="site of record (decoder-layer index)")
    ap.add_argument("--arm", default="native", choices=("native", "raw"))
    ap.add_argument("--n-sigma", type=int, default=N_SIGMA)
    ap.add_argument("--n-grad", type=int, default=N_GRAD)
    ap.add_argument("--band-lo", type=int, default=BAND_LO)
    ap.add_argument("--band-hi", type=int, default=BAND_HI)
    ap.add_argument("--ridge-rel", type=float, default=RIDGE_REL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fd-eps-fractions", default=",".join(str(f) for f in FD_EPS_FRACTIONS),
                    help="comma list of eps / median-residual-norm ratios for the ladder")
    ap.add_argument("--fd-n-probe", type=int, default=FD_N_PROBE)
    ap.add_argument("--fd-rel-tol", type=float, default=FD_REL_TOL)
    ap.add_argument("--entropy-chunk", type=int, default=0,
                    help="OOM fallback: compute the entropy in position blocks of this "
                         "size (same algebra, different fp32 summation order)")
    ap.add_argument("--allow-fd-gate-not-passed", action="store_true",
                    help="bank the vector and exit 0 even if the FD gate does not pass "
                         "(diagnostic runs ONLY — a build without its gate does not exist)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    missing = [f"--{n.replace('_', '-')}" for n in
               ("model", "model_path", "arm_root", "out_dir", "site")
               if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")

    try:
        request = BuildRequest(
            model_key=args.model, model_path=args.model_path, arm_root=args.arm_root,
            out_dir=args.out_dir, site=args.site, arm=args.arm,
            n_sigma=args.n_sigma, n_grad=args.n_grad, band_lo=args.band_lo,
            band_hi=args.band_hi, ridge_rel=args.ridge_rel, device=args.device,
            fd_eps_fractions=_parse_fractions(args.fd_eps_fractions),
            fd_n_probe=args.fd_n_probe, fd_rel_tol=args.fd_rel_tol,
            entropy_chunk=args.entropy_chunk)
    except ValueError as exc:
        raise SystemExit(f"invalid build request: {exc}") from exc

    from metabasis.scripts.collect_mean_states import (
        VMB_CANONICAL_DATE, load_corpus, load_model_and_tok, trunk_stamp)

    try:
        entries, corpus_sha = load_corpus(request.arm_root)
    except (OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"corpus unreadable under {request.arm_root}: {exc}") from exc
    logger.info("corpus: %d texts, manifest sha256 %s", len(entries), corpus_sha[:12])

    try:
        model, tok = load_model_and_tok(request.model_path, request.device)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(
            f"cannot load {request.model_key} from {request.model_path}: "
            f"{type(exc).__name__}: {exc}") from exc
    # Freeze every parameter: backward then allocates NO per-parameter grad buffers
    # (the difference between ~20 GB peak and OOM on a 32B). The gradient still flows
    # THROUGH the frozen weights to reach the leaf; the vector is unchanged.
    model.requires_grad_(False)
    trunk = trunk_stamp(request.model_path, tok, request.device)
    logger.info("trunk: %s", json.dumps(trunk))

    try:
        result = build_entropy_gradient(model, tok, entries, request,
                                        VMB_CANONICAL_DATE, request.device)
    except EntropyGradientBuildError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"ENTROPY-GRADIENT-BUILD-INCOMPLETE model={request.model_key} "
              f"site=L{request.site} reason={type(exc).__name__}")
        return 3
    except Exception as exc:                                   # noqa: BLE001
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.error("CUDA OOM: %s", exc)
            print(f"ENTROPY-GRADIENT-BUILD-INCOMPLETE model={request.model_key} "
                  "reason=OOM — retry with --entropy-chunk 256 (same algebra, "
                  "chunked log_softmax); never shard, never touch other processes")
            return 4
        raise

    paths = bank(result, trunk, corpus_sha, VMB_CANONICAL_DATE)
    g = result.fd_gate
    d = result.diagnostics
    summary = {
        "model": request.model_key, "site": request.site, "arm": request.arm,
        "npz_key": CANONICAL_VECTOR_KEY.format(site=request.site),
        "vector_path": str(paths["vector"]),
        "vector_norm": d.vector_norm, "hidden_dim": d.hidden_dim,
        "band_energy_fraction": d.band_energy_fraction,
        "per_text_band_sign_consistency": d.per_text_band_sign_consistency,
        "per_text_band_pairwise_coherence": d.per_text_band_pairwise_coherence,
        "median_token_resid_norm": d.median_token_resid_norm,
        "mean_entropy_nats": d.mean_entropy_nats,
        "n_sigma_positions": d.n_sigma_positions,
        "n_grad_positions": d.n_grad_positions,
        "fd_gate_passed": g.passed,
        "fd_best_eps_fraction": g.best_eps_fraction,
        "fd_best_median_rel_error": g.best_median_rel_error,
        "fd_best_max_rel_error": g.best_max_rel_error,
        "fd_direction_specificity_ratio": g.control_ratio_median,
        "wall_seconds": result.wall_seconds,
        "cuda_visible_devices": trunk["cuda_visible_devices"],
    }
    print(json.dumps(summary, indent=1))
    if not g.passed:
        logger.error("FD gate did not pass: median rel.err %.4g at eps_frac %.3g "
                     "(tolerance %.3g)", g.best_median_rel_error,
                     g.best_eps_fraction, g.rel_tolerance)
        print(f"ENTROPY-GRADIENT-BUILD-INCOMPLETE model={request.model_key} "
              f"site=L{request.site} fd_gate=NOT-PASSED "
              f"median_rel_error={g.best_median_rel_error:.4g}")
        if not args.allow_fd_gate_not_passed:
            return 5
    print(f"ENTROPY-GRADIENT-BUILD-OK model={request.model_key} site=L{request.site} "
          f"arm={request.arm} fd_gate={'PASS' if g.passed else 'NOT-PASSED'} "
          f"fd_median_rel_error={g.best_median_rel_error:.4g} "
          f"card={trunk['cuda_visible_devices']} wall={result.wall_seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
