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
  * leaf-hook RE-ENTRY — the pre-hook counts its calls per forward and raises if the
    SITE layer's forward fires more than once, because a second call would replace
    the leaf with a recompute pass's copy and the gradient would then be taken with
    respect to a tensor that never fed the loss. Whole-model activation
    checkpointing therefore cannot be used, and is not: `--checkpoint-above-site`
    wraps `layers[site+1:]` ONLY (see below), and the guard is re-asserted AFTER
    the backward on every text, which is what proves the recompute stayed above
    the site.
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

    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
    python -m metabasis.scripts.build_entropy_gradient \
        --model phi-4 --model-path <LOCAL_WEIGHTS_DIR> --arm-root <ARM_ROOT> \
        --site 19 --arm native --out-dir <ARM_ROOT>/staging/phi-4/vectors

Three opt-in flags exist for nodes that cannot take the default path; each is a
no-op when unset, so every historical build is reproduced byte-for-byte without it:

  * `--max-seq-len N [--assert-n-truncated K]` — prereg ADDENDUM 2026-07-27-B. A node
    whose registry row carries `max_seq_len` (gpt2-xl: 1024) MUST build under the same
    per-text truncation it collected under, or its vector lives in a different space
    than the transport maps were fit in. The corpus-wide truncation count is re-derived
    from this node's own tokenizer BEFORE any forward pass and, with
    `--assert-n-truncated`, must reproduce the ratified number or the build refuses.
  * `--out-name STEM` — bank BESIDE an existing vector of the same model instead of
    over it (the 8bL16 construction-lineage rebuild). Moves the FILENAMES only; the npz
    KEY stays canonical, which is what the readout resolves on.
  * `--shard-across 0,1` — the collector's certified sharded loader (prereg §4; the
    sharding gate PASSED 2026-07-27 byte-identical to single-device). For a model whose
    weights plus the retained autograd graph do not fit one card. The layer activations
    of every layer >= site are retained for the leaf gradient, and eager attention is
    O(n^2) per layer, so the peak is well above the weights alone.
  * `--checkpoint-above-site` — prereg §4's "input-gradient extraction with
    checkpointing", made implementable. As written that phrase was NOT: whole-model
    checkpointing wraps the site layer too and trips the re-entry guard by
    construction, and REENTRANT checkpointing (torch's historical default) requires
    `.backward()` while this builder uses `autograd.grad` — which is the whole reason
    no per-parameter grad buffers are ever allocated. The named fix wraps
    `layers[site+1:]` in NON-REENTRANT checkpointing (`use_reentrant=False`, passed
    explicitly) for the GRADIENT PASS ONLY: layers at and below the site are never
    wrapped, so the substituted leaf and its gradient are untouched, the site
    layer's forward still fires exactly once, and the guard is RE-ASSERTED after the
    backward — the only place a recompute-driven re-entry could show up. The
    agreement of the two paths is MEASURED end to end in the selftest, never argued
    from the reasoning above. This is what unblocks the DeepSeek-V3 native target
    build, whose gradient pass runs out of memory on the retained per-layer graph
    rather than on the weights or the logits.
"""
from __future__ import annotations

import os

#  THE RULED DEFAULT (Luxia 2026-08-01): OMP_NUM_THREADS=8 everywhere. This is a
#  FLOOR, not an override — every deployed job script exports the count itself and
#  `setdefault` leaves it alone. It has to run above the numpy import because
#  OpenBLAS reads its thread count when the shared object is LOADED; setting it
#  afterwards changes the string and nothing else. The vector's bytes depend on
#  the count that was actually in effect, so `bank()` records what the THREADPOOL
#  says rather than what this line asked for (metabasis.threads).
from metabasis.threads import (RULED_OMP_NUM_THREADS, THREAD_STAMP_KEY,
                               stamp_thread_config, thread_config_stamp)

os.environ.setdefault("OMP_NUM_THREADS", str(RULED_OMP_NUM_THREADS))

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence

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

#: THE AGREEMENT TOLERANCE for `--checkpoint-above-site` (selftest block 6).
#:
#: WHY A TOLERANCE AT ALL, when the measurement is BITWISE. On CPU float32 the
#: recomputed forward runs the same kernels in the same order on the same saved
#: boundary tensors, so the built vector comes out byte-identical — measured, and
#: the selftest asserts that strictly, because a torch release that changed
#: recompute semantics should fail loudly rather than drift inside a tolerance.
#: But byte-equality is NOT guaranteed where the flag is actually FOR: a bf16
#: forward on a GPU may select a different kernel or reduction order on the second
#: pass, and the honest contract for a banked vector has to be numeric.
#:
#: WHY THIS VALUE. The vector is unit-norm float32, so a component's own
#: representation error is already ~1.2e-7 (float32 eps). 1e-6 is ~8 eps: loose
#: enough that kernel-level reassociation across a handful of layers cannot trip
#: it, tight enough to be ~5 orders of magnitude below anything that could move a
#: transported cosine at the 4-dp filing convention, and ~5 orders below the FD
#: gate's own 0.25 relative tolerance — so any disagreement big enough to matter
#: scientifically would be caught by the gate long before it approached this bound.
#: A violation here means the two paths are computing different things, which is a
#: HALT, not a rounding difference.
CHECKPOINT_AGREEMENT_ATOL = 1e-6


# ---------------------------------------------------------------- error taxonomy
class EntropyGradientBuildError(RuntimeError):
    """Base class for every failure specific to this builder."""


class SiteOutOfRangeError(EntropyGradientBuildError):
    """The requested site is not a decoder-layer index of the loaded model."""


class BandOutOfRangeError(EntropyGradientBuildError):
    """The covariance band does not fit the model's hidden dimension."""


class CorpusSelectionError(EntropyGradientBuildError):
    """The corpus cannot supply the requested number of texts."""


class TruncationAuditError(EntropyGradientBuildError):
    """The position-ceiling truncation audit did not reproduce the ratified count."""


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
    max_seq_len: Optional[int] = Field(
        default=None, ge=2,
        description="prereg ADDENDUM 2026-07-27-B position-ceiling deviation: truncate "
                    "every text to its FIRST `max_seq_len` tokens. MUST match the value "
                    "the node's registry row carries (`RosterNode.max_seq_len`), because "
                    "the addendum requires ONE truncation across collection, spot-replay, "
                    "target builds and behavioral reads — a build that truncates "
                    "differently from the collection would live in a different space "
                    "than the transport maps were fit in.")
    expect_n_truncated: Optional[int] = Field(
        default=None, ge=0,
        description="ratified corpus-wide truncation count (148 for gpt2-xl). When set, "
                    "the build re-derives the count from THIS node's tokenizer over the "
                    "WHOLE frozen corpus and refuses to run if it disagrees — the "
                    "deviation is then a verified fact of the build, not a claim.")
    out_stem: Optional[str] = Field(
        default=None, min_length=1,
        description="filename stem for the banked vector/stamp/FD-gate trio; defaults to "
                    "`entropy_gradient_<model_key>` (the convention the readout probes "
                    "first). Overridden only when a build must sit BESIDE an existing "
                    "vector of the same model — e.g. the 8bL16 construction-lineage "
                    "rebuild, which must not overwrite the legacy L16 bank.")
    checkpoint_above_site: bool = Field(
        default=False,
        description="wrap layers[site+1:] in NON-REENTRANT activation checkpointing "
                    "for the gradient pass, trading recompute for the retained "
                    "activations of every layer above the site. Layers AT and BELOW "
                    "the site are never wrapped, so the leaf and its gradient are "
                    "untouched — see `checkpoint_above_site` for why that is the whole "
                    "design and why non-reentrant is the only workable mode.")

    @property
    def stem(self) -> str:
        """The banked filename stem (never the npz KEY, which is always canonical)."""
        return self.out_stem or f"entropy_gradient_{self.model_key}"

    @field_validator("out_stem")
    @classmethod
    def _stem_is_a_bare_filename(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and ("/" in v or "\\" in v or v.endswith(".npz")):
            raise ValueError(
                f"--out-name must be a bare filename STEM (no directory, no .npz), got {v!r}")
        return v

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


class TruncationAudit(BaseModel):
    """Prereg ADDENDUM 2026-07-27-B, re-derived on this node from this tokenizer.

    Counted over the WHOLE frozen corpus (not just the build's 80 texts), because the
    ratified number the addendum names — 148 of 780 for gpt2-xl — is a corpus-wide fact,
    and re-deriving it here is what proves this build applies the SAME truncation the
    collection applied. The subset actually consumed by this build is recorded beside it.
    """

    model_config = ConfigDict(frozen=True)

    prereg: str = "ADDENDUM 2026-07-27-B"
    max_length: int
    arm: str
    n_corpus_texts: int
    n_truncated: int
    truncated_text_ids: list[str]
    n_truncated_in_build: int
    truncated_in_build_ids: list[str]
    expected_n_truncated: Optional[int] = None


def audit_truncation(tok: Any, entries: Sequence[dict], arm: str, date_string: str,
                     max_length: int, build_text_ids: Sequence[str],
                     expected: Optional[int] = None) -> TruncationAudit:
    """Which frozen-corpus texts the position ceiling actually truncates, exactly.

    Mirrors `collect_mean_states.compute_means`'s rule verbatim so the two counts are
    comparable by construction: hitting the ceiling is NECESSARY but not SUFFICIENT (a
    text whose natural length is exactly `max_length` was not truncated), so the rare
    boundary case is re-tokenized without the ceiling rather than assumed.

    Tokenizer-only: no weights, no GPU. Runs BEFORE the first forward pass so a
    mismatched deviation costs zero GPU time.
    """
    from metabasis.scripts.collect_mean_states import build_ids

    truncated: list[str] = []
    for e in entries:
        ids, _ = build_ids(tok, e, arm, date_string, max_length=max_length)
        if len(ids) == max_length:
            if len(build_ids(tok, e, arm, date_string)[0]) > max_length:
                truncated.append(e["text_id"])
    in_build = [t for t in truncated if t in set(build_text_ids)]
    audit = TruncationAudit(
        max_length=max_length, arm=arm, n_corpus_texts=len(entries),
        n_truncated=len(truncated), truncated_text_ids=truncated,
        n_truncated_in_build=len(in_build), truncated_in_build_ids=in_build,
        expected_n_truncated=expected)
    if expected is not None and len(truncated) != expected:
        raise TruncationAuditError(
            f"position-ceiling audit disagrees with the ratified deviation: this "
            f"tokenizer truncates {len(truncated)} of {len(entries)} corpus texts at "
            f"first-{max_length} on the {arm} arm, but ADDENDUM 2026-07-27-B ratified "
            f"{expected}. Either the checkpoint/tokenizer is not the collected one or "
            f"the corpus is not the frozen one — refusing to build a vector that would "
            f"not live in the collected node's space.")
    logger.info("truncation audit (first-%d, %s arm): %d/%d corpus texts truncated%s; "
                "%d of them are among this build's %d texts", max_length, arm,
                len(truncated), len(entries),
                f" (ratified {expected}: MATCH)" if expected is not None else "",
                len(in_build), len(build_text_ids))
    return audit


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
    #: how many layers above the site ran under NON-REENTRANT activation
    #: checkpointing during the gradient pass. 0 = the flag was not passed, or the
    #: site is the top layer and there was nothing above it to wrap. Recorded so a
    #: banked vector says whether it was built with the memory deviation, and a
    #: comparison between two banks can name the difference.
    n_layers_checkpointed_above_site: int = 0


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
    #: None on every node without a ruled position ceiling (the historical key set).
    truncation: Optional[TruncationAudit] = None


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


class CheckpointModeError(EntropyGradientBuildError):
    """Activation checkpointing could not be established in the required mode."""


@contextmanager
def checkpoint_above_site(layers: Any, site: int) -> Iterator[int]:
    """Wrap `layers[site+1:]` in NON-REENTRANT activation checkpointing.

    ─── WHY THIS EXISTS ────────────────────────────────────────────────────────
    The gradient pass retains the activations of every layer at or above the site
    (that is where autograd's graph lives, because `requires_grad_(False)` on the
    parameters means the graph starts at the substituted leaf and nowhere else).
    On a deep enough model with eager O(n²) attention that alone exhausts a card:
    DeepSeek-V3's entropy-gradient vector cannot be built without help, and the
    prereg §4 phrase "input-gradient extraction with checkpointing" was
    UNIMPLEMENTABLE as written — for a reason worth stating exactly, because it is
    the reason this function has the shape it has.

    ─── WHY THE OBVIOUS THING IS FORBIDDEN ─────────────────────────────────────
    Two independent obstacles, and each one alone is fatal:

    1. THE LEAF-HOOK RE-ENTRY GUARD. `_LeafSubstitution` counts the site layer's
       forward calls and raises if it fires more than once, because a second call
       would REPLACE the leaf with the recompute pass's copy and the gradient would
       then be taken with respect to a tensor that never fed the loss. Ordinary
       whole-model checkpointing (or `model.gradient_checkpointing_enable()`) wraps
       EVERY layer including the site's, so it trips that guard by construction —
       correctly. Hence `layers[site+1:]` and not `layers[:]`: layers at and below
       the site are never wrapped, the site layer's forward fires exactly once, and
       the leaf and its gradient are bit-for-bit what they were.

    2. REENTRANT CHECKPOINTING NEEDS `.backward()`. `use_reentrant=True` (the
       historical default) implements the recompute through a custom autograd
       Function that requires the backward pass to be driven by
       `Tensor.backward()`; under `torch.autograd.grad(S, leaf)` — which is what
       this builder uses, precisely so that no per-parameter grad buffers are ever
       allocated — it either raises or silently produces no gradient for inputs it
       cannot see. `use_reentrant=False` uses the saved-tensor-hooks machinery,
       composes with `autograd.grad`, and is the only mode that works here. It is
       passed EXPLICITLY rather than relied on as a default, because the default
       has changed across torch versions and a silent flip back to reentrant would
       turn a working build into a confusing failure.

    ─── WHY WRAPPING `.forward` AND NOT THE MODULE ──────────────────────────────
    This wraps each target module's `.forward` attribute, not the module itself,
    and that choice is load-bearing rather than stylistic. Forward pre-hooks fire
    in `Module.__call__`; a recompute that re-runs the captured `forward` never
    passes through `__call__`, so no pre-hook fires twice ANYWHERE in the stack —
    a second, independent guarantee beside the site-exclusion above.
    `model.gradient_checkpointing_enable()` does the opposite: it recomputes
    through the layer's `__call__`, which DOES re-invoke pre-hooks. Both behaviours
    are MEASURED in the selftest, because they differ and the difference is exactly
    what makes one approach safe and the other silently wrong.

    ─── WHAT IS PRESERVED ──────────────────────────────────────────────────────
    Checkpointing changes WHEN a forward runs, not WHAT it computes. The layers
    above the site run twice (once forward, once recomputed during the backward)
    and the recompute is fed the same inputs from the same saved boundary tensors,
    so the gradient reaching the leaf is the same gradient. `preserve_rng_state`
    is left at its default True so any stochastic op would replay identically —
    the model is in `eval()` and the builder allocates no dropout, so this is
    belt-and-braces rather than load-bearing, but a builder that relied on
    "there is no randomness here" would be one refactor from being wrong.

    The AGREEMENT is not argued from that reasoning — it is MEASURED, on a real
    end-to-end build, in the selftest.

    ─── YIELDS ─────────────────────────────────────────────────────────────────
    the number of layers wrapped. Zero is a legitimate result (a site at the top
    of the stack has nothing above it) and is NOT an error: the flag then costs
    nothing and changes nothing, which is the honest behaviour.
    """
    import torch.utils.checkpoint as _ckpt

    if not hasattr(_ckpt, "checkpoint"):             # pragma: no cover — defensive
        raise CheckpointModeError(
            "torch.utils.checkpoint.checkpoint is unavailable in this torch "
            "build; --checkpoint-above-site cannot be honoured, and the build "
            "REFUSES rather than silently running without it — an OOM that the "
            "flag was passed to avoid must not read as a model too big for the "
            "node")
    n_layers = len(layers)
    if not 0 <= site < n_layers:
        raise SiteOutOfRangeError(
            f"cannot checkpoint above site L{site}: the stack has {n_layers} "
            f"decoder layers")
    targets = list(range(site + 1, n_layers))
    originals: list[tuple[Any, Any]] = []

    def _wrap(module: Any) -> Any:
        original = module.forward

        def _checkpointed(*args: Any, **kwargs: Any) -> Any:
            #  use_reentrant=False EXPLICITLY (see the docstring): the reentrant
            #  implementation requires .backward() and this builder uses
            #  autograd.grad. The default has moved across torch versions, so it is
            #  never left to the library to decide.
            return _ckpt.checkpoint(original, *args, use_reentrant=False, **kwargs)

        return original, _checkpointed

    try:
        for index in targets:
            module = layers[index]
            original, wrapped = _wrap(module)
            originals.append((module, original))
            module.forward = wrapped                 # type: ignore[method-assign]
        if targets:
            logger.info("activation checkpointing ON for layers[%d:%d] "
                        "(%d layer(s)), NON-REENTRANT; site L%d and every layer "
                        "below it are UNWRAPPED so the leaf gradient is untouched",
                        site + 1, n_layers, len(targets), site)
        else:
            logger.info("activation checkpointing requested but site L%d is the "
                        "TOP layer — 0 layers above it, so the flag is a no-op "
                        "and the build is byte-identical to one without it", site)
        yield len(targets)
    finally:
        #  Restored on EVERY exit path. A module left carrying a wrapped forward
        #  would silently checkpoint the FD gate's no-grad passes and any later
        #  build in the same process — and the second of those would be a
        #  different object from the first without anything erroring.
        for module, original in originals:
            module.forward = original                # type: ignore[method-assign]


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

    # ADDENDUM 2026-07-27-B: audited BEFORE the first forward, so a truncation that
    # disagrees with the collection's costs no GPU time at all.
    truncation: Optional[TruncationAudit] = None
    if request.max_seq_len is not None:
        truncation = audit_truncation(
            tok, entries, request.arm, date_string, request.max_seq_len,
            [e["text_id"] for e in sigma_entries] + [e["text_id"] for e in grad_entries],
            expected=request.expect_n_truncated)

    hook = _LeafSubstitution(layers[site])
    try:
        # ── covariance pass: residual rows at the site over completion positions ──
        rows: list[np.ndarray] = []
        tok_norms: list[float] = []
        for i, e in enumerate(sigma_entries):
            ids_list, p = build_ids(tok, e, request.arm, date_string,
                                   max_length=request.max_seq_len)
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
        #  THE THREAD-SENSITIVE STEP (Luxia ruling 2026-08-01). Sigma above is
        #  bitwise stable regardless of thread count; this decomposition is not.
        #  LAPACK's tridiagonal reduction and back-transformation are blocked
        #  GEMM calls whose summation order depends on how the work is split, so
        #  eigh is bitwise-deterministic at a FIXED thread count (measured,
        #  repeat-identical) and its bytes differ ACROSS counts. Everything
        #  downstream — the band basis Ub, the projection, the unit vector — is
        #  built on these evecs, so the effective count is part of this
        #  vector's identity and `bank()` records it (metabasis.threads).
        evals, evecs = np.linalg.eigh(Sigma)                   # ascending
        ridge = float(request.ridge_rel * float(evals.mean()))
        order = np.argsort(evals)[::-1]                        # descending
        band_idx = order[request.band_lo:request.band_hi]
        Ub = evecs[:, band_idx]                                # (d, band_hi-band_lo)
        logger.info("Sigma over %d positions (%d texts); ridge %.3g; band [%d:%d]",
                    n_sigma_positions, len(sigma_entries), ridge,
                    request.band_lo, request.band_hi)

        # ── gradient pass: leaf gradient of the mean completion entropy ──
        #  ACTIVATION CHECKPOINTING, if asked for, wraps ONLY this pass. The
        #  covariance pass above runs under `no_grad`, where nothing is retained
        #  and checkpointing is pure overhead; the FD gate below runs no-grad
        #  forwards for the same reason. Scoping it here keeps every other forward
        #  in the build byte-identical to one made without the flag.
        Ge: list[np.ndarray] = []
        probe_meta: list[dict] = []
        entropies: list[float] = []
        n_grad_positions = 0
        n_checkpointed = 0
        with ExitStack() as grad_stack:
            if request.checkpoint_above_site:
                n_checkpointed = grad_stack.enter_context(
                    checkpoint_above_site(layers, site))
            for i, e in enumerate(grad_entries):
                ids_list, p = build_ids(tok, e, request.arm, date_string,
                                        max_length=request.max_seq_len)
                n = len(ids_list)
                if n - p <= 0:
                    raise CorpusSelectionError(f"{e['text_id']}: no completion positions")
                ids = torch.tensor([ids_list], dtype=torch.long, device=device)
                hook.arm()
                with torch.enable_grad():
                    out = model(ids, use_cache=False, return_dict=True)
                    S = _mean_next_token_entropy(out.logits[0], p, n,
                                                 request.entropy_chunk)
                    leaf = hook.check_single_call(f"gradient pass, {e['text_id']}")
                    try:
                        (grad_leaf,) = torch.autograd.grad(S, leaf, allow_unused=False)
                    except RuntimeError as exc:
                        raise GradientPathError(
                            f"{e['text_id']}: autograd.grad(S_entropy, leaf) failed ({exc}). "
                            "The substituted leaf never reached the loss — the layer's "
                            "forward ignored the hook's returned hidden_states, or the "
                            "forward has no autograd nodes at all.") from exc
                    #  THE RE-ENTRY GUARD, RE-ASSERTED AFTER THE BACKWARD. This is
                    #  the load-bearing check for `--checkpoint-above-site`: the
                    #  recompute happens INSIDE autograd.grad, after
                    #  check_single_call has already run, so only a second look can
                    #  prove the recompute never re-entered the site layer. It is
                    #  unconditional — the flag is not what makes a second call
                    #  possible, it is only the most likely way to cause one.
                    hook.check_single_call(
                        f"gradient pass AFTER backward, {e['text_id']} "
                        f"(checkpointed layers above site: {n_checkpointed})")
                    if grad_leaf is None:                      # pragma: no cover
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
                            i + 1, len(grad_entries), float(np.linalg.norm(g)),
                            s_value)
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
        n_layers_checkpointed_above_site=n_checkpointed,
    )
    return BuildResult(
        request=request, vector=vector, random_band=random_band,
        sigma_evals=evals.astype(np.float64), sigma_evecs=evecs.astype(np.float64),
        sigma_mean=mu.astype(np.float64), diagnostics=diagnostics, fd_gate=fd_gate,
        sigma_text_ids=[e["text_id"] for e in sigma_entries],
        grad_text_ids=[e["text_id"] for e in grad_entries],
        wall_seconds=round(time.time() - t0, 1),
        truncation=truncation)


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


# ------------------------------------------------- the peak-memory stamp
#  SELF-EVIDENCING CAPACITY. Every capacity ruling this campaign makes — the
#  x1.6 flat multiplier, the structural bound, a sole-job reservation, a
#  shard-across decision — is a claim about how much device memory a build
#  actually needs. Until now those claims were adjudicated against numbers that
#  lived in a job log beside the artifact; the artifact itself said nothing. So
#  the bank records what the allocator SAW, per device, at bank time.
#
#  WHY max_memory_allocated AND max_memory_reserved. `allocated` is what the
#  build asked for and is the number a reservation should be compared against;
#  `reserved` is what the caching allocator held from the driver and is the
#  number the CARD saw (and therefore what a co-tenant collided with). They
#  differ by the allocator's fragmentation, which is exactly the quantity a
#  margin exists to cover — so recording one without the other would leave the
#  next ruling arguing about which number the stamp meant.
#
#  MONOTONIC FROM PROCESS START, and NOT reset here. `max_memory_allocated` is a
#  high-water mark since the process began, so this covers the model load, the
#  forward/backward passes and the FD ladder together — the whole build, which
#  is the unit a capacity ruling is made in. Resetting it at any point inside
#  the build would silently narrow the window and make the stamp read LOWER
#  than the job's real peak, which is the one direction a capacity number must
#  never be wrong in.
#: The stamp key. ONE name, so a reader that wants a build's peak memory never
#: has to know which builder wrote the artifact (the THREAD_STAMP_KEY pattern).
CAPACITY_STAMP_KEY: str = "peak_device_memory"

#: The probe ran and could not measure a peak — degraded instrumentation, which
#: is NOT the same as "this stamp predates the field" (that is
#: `stamp_peak_device_memory(...) is None`). Greppable, so a log/stamp sweep
#: never has to recognise prose (the THREADS_UNRESOLVED discipline).
CAPACITY_UNMEASURED: str = "CAPACITY_UNMEASURED"

#: What a PRE-CHANGE artifact's peak memory reads as in a comparison. Every
#: vector banked before 2026-08-04 has no such field, and its peak is
#: UNRECORDED — never inferred from a job log that happens to be beside it.
CAPACITY_PRE_CHANGE_UNRECORDED: str = (
    "unrecorded (stamp predates the 2026-08-04 capacity stamp)")

GIB: float = float(1024 ** 3)


class DevicePeakMemory(BaseModel):
    """One CUDA device's high-water marks, as the allocator reports them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(
        description="the device ordinal AS THIS PROCESS SAW IT — i.e. after "
                    "CUDA_VISIBLE_DEVICES remapping, which is why that value "
                    "is recorded in the same block")
    name: Optional[str] = None
    max_memory_allocated_bytes: Optional[int] = Field(
        default=None,
        description="torch.cuda.max_memory_allocated(i) — what the build "
                    "ASKED FOR, high-water since process start. None = the "
                    "probe could not answer, a NAMED degradation, never a "
                    "silent 0")
    max_memory_reserved_bytes: Optional[int] = Field(
        default=None,
        description="torch.cuda.max_memory_reserved(i) — what the caching "
                    "allocator HELD from the driver, i.e. what a co-tenant "
                    "collided with")
    max_memory_allocated_gib: Optional[float] = None
    max_memory_reserved_gib: Optional[float] = None
    total_capacity_gib: Optional[float] = Field(
        default=None,
        description="the card's own total, so a peak can be read as a "
                    "FRACTION without a second artifact")
    note: str = ""


class PeakDeviceMemory(BaseModel):
    """The peak device memory of THIS build process, as one document.

    UNCONDITIONAL SHAPE. The same keys are present whether or not anything was
    measurable, so a no-CUDA build is a NAMED absence (`measured=False` plus a
    `note` saying why) rather than a missing field — and a missing field
    therefore means exactly one thing: the stamp predates this change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measured: bool = Field(
        default=False,
        description="did any device answer? False with a note is the M19 "
                    "degradation; False with no devices and a named reason is "
                    "the honest no-CUDA case")
    n_devices: int = 0
    devices: list[DevicePeakMemory] = Field(
        default=[], description="one row per visible CUDA device")
    peak_allocated_gib: Optional[float] = Field(
        default=None,
        description="the WORST device's allocated high-water mark — the single "
                    "number a per-card reservation is compared against")
    peak_reserved_gib: Optional[float] = None
    cuda_visible_devices: str = Field(
        default="(unset)",
        description="recorded here as well as in the trunk, because the device "
                    "ORDINALS above are meaningless without it")
    torch_version: Optional[str] = None
    probe: str = Field(
        default="", description="HOW the peaks were read, so a reader can "
                                "judge them")
    note: str = Field(
        default="",
        description="why the probe degraded, when it did. Empty on a clean "
                    "read; never empty when `measured` is False")

    @property
    def quoted(self) -> str:
        """The peak as a comparison should print it — never a bare number."""
        if self.peak_allocated_gib is None:
            return f"{CAPACITY_UNMEASURED} (0 device(s))"
        return (f"{self.peak_allocated_gib:.3f} GiB allocated / "
                f"{self.peak_reserved_gib:.3f} GiB reserved over "
                f"{self.n_devices} device(s)"
                if self.peak_reserved_gib is not None
                else f"{self.peak_allocated_gib:.3f} GiB allocated")


#: The one-line STATUS the block opens with, so a reader who has never seen the
#: field knows in one line what it is, what window it covers, and what it does
#: NOT mean.
CAPACITY_STAMP_STATUS: str = (
    "SELF-EVIDENCING CAPACITY: the per-device high-water memory marks this "
    "build actually reached, read from the allocator at bank time. "
    "max_memory_allocated/reserved are MONOTONIC FROM PROCESS START and are "
    "deliberately NOT reset anywhere in the build, so these cover the whole "
    "job (load + forwards + backwards + FD ladder) — the unit a capacity "
    "ruling is made in. DESCRIPTIVE ONLY: no gate reads it, and a build with "
    "no CUDA records a NAMED absence rather than a zero.")


def peak_device_memory() -> PeakDeviceMemory:
    """Read the per-device peaks. NEVER raises (rake M19).

    Availability-branched, with NAMED skip semantics (rake M44): no torch, no
    CUDA, and a device that would not answer are three different facts and are
    reported as three different notes — none of them as a zero, which would
    read as "this build needed no memory".
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    base = dict(cuda_visible_devices=visible)
    try:
        import torch
    except Exception as exc:                                  # noqa: BLE001 — a probe
        return PeakDeviceMemory(
            **base, probe="torch.cuda.max_memory_allocated/reserved per device",
            note=f"{CAPACITY_UNMEASURED}: torch is not importable here "
                 f"({type(exc).__name__}: {exc}) — a legitimate environment "
                 f"(the desk's CPU spine is one), not a failure")
    version = str(getattr(torch, "__version__", "unknown"))
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
    except Exception as exc:                                  # noqa: BLE001 — M19
        return PeakDeviceMemory(
            **base, torch_version=version,
            probe="torch.cuda.max_memory_allocated/reserved per device",
            note=f"{CAPACITY_UNMEASURED}: the CUDA availability probe raised "
                 f"{type(exc).__name__}: {exc}")
    if not available or count == 0:
        return PeakDeviceMemory(
            **base, torch_version=version,
            probe="torch.cuda.max_memory_allocated/reserved per device",
            note=f"{CAPACITY_UNMEASURED}: no CUDA device is visible to this "
                 f"process (torch.cuda.is_available()={available}, "
                 f"device_count={count}) — this build ran on CPU, so there is "
                 f"no device peak to record. A NAMED absence, never a 0")

    rows: list[DevicePeakMemory] = []
    for index in range(count):
        try:
            allocated = int(torch.cuda.max_memory_allocated(index))
            reserved = int(torch.cuda.max_memory_reserved(index))
        except Exception as exc:                              # noqa: BLE001 — M19
            rows.append(DevicePeakMemory(
                index=index,
                note=f"{CAPACITY_UNMEASURED}: the allocator declined "
                     f"({type(exc).__name__}: {exc})"))
            continue
        name: Optional[str] = None
        total: Optional[float] = None
        try:
            props = torch.cuda.get_device_properties(index)
            name = str(props.name)
            total = round(float(props.total_memory) / GIB, 3)
        except Exception as exc:                              # noqa: BLE001 — M19
            name = None
            total = None
            logger.debug("device %d properties unavailable: %s", index, exc)
        rows.append(DevicePeakMemory(
            index=index, name=name,
            max_memory_allocated_bytes=allocated,
            max_memory_reserved_bytes=reserved,
            max_memory_allocated_gib=round(allocated / GIB, 3),
            max_memory_reserved_gib=round(reserved / GIB, 3),
            total_capacity_gib=total))

    answered = [r for r in rows if r.max_memory_allocated_bytes is not None]
    probe = ("torch.cuda.max_memory_allocated/max_memory_reserved per visible "
             "device, read at bank time; monotonic from process start and "
             "never reset, so the window is the whole build")
    if not answered:
        return PeakDeviceMemory(
            **base, torch_version=version, n_devices=count, devices=rows,
            probe=probe,
            note=f"{CAPACITY_UNMEASURED}: {count} device(s) are visible and "
                 f"none answered the allocator — "
                 + "; ".join(f"device {r.index}: {r.note}" for r in rows))
    peak_alloc = max(float(r.max_memory_allocated_gib or 0.0) for r in answered)
    peak_res = max(float(r.max_memory_reserved_gib or 0.0) for r in answered)
    partial = ""
    if len(answered) != count:
        partial = (f"{CAPACITY_UNMEASURED}: {count - len(answered)} of {count} "
                   f"device(s) did not answer; the peaks below are over the "
                   f"{len(answered)} that did")
    return PeakDeviceMemory(
        **base, measured=True, torch_version=version, n_devices=count,
        devices=rows, peak_allocated_gib=round(peak_alloc, 3),
        peak_reserved_gib=round(peak_res, 3), probe=probe, note=partial)


def capacity_stamp(memory: Optional[PeakDeviceMemory] = None) -> dict:
    """The `peak_device_memory` block a builder writes into its stamp.

    Plain JSON-able types only, `STATUS` first, so the block reads correctly in
    a stamp opened by a human with no access to this module.
    """
    mem = memory if memory is not None else peak_device_memory()
    return {"STATUS": CAPACITY_STAMP_STATUS, **mem.model_dump(mode="json")}


def stamp_peak_device_memory(stamp: Optional[dict]) -> Optional[PeakDeviceMemory]:
    """The peak memory a stamp records — `None` for a PRE-CHANGE stamp.

    THE BACKWARD-COMPATIBLE READER, and the only one any consumer should use.
    Three states a reader must keep apart (the thread_config / M41 discipline):

        None                    the stamp PREDATES the field. The build's peak
                                is UNRECORDED and must not be inferred
        measured=False          the builder RAN and measured nothing — either
                                honestly (no CUDA) or degraded; the `note`
                                says which
        measured=True           the peaks the build actually reached

    Accepts a whole build stamp, a nested `trunk`, or the block itself, because
    all three get handed around.
    """
    if not isinstance(stamp, dict):
        return None
    block: object = stamp.get(CAPACITY_STAMP_KEY)
    if block is None and isinstance(stamp.get("trunk"), dict):
        block = stamp["trunk"].get(CAPACITY_STAMP_KEY)
    if block is None and "measured" in stamp and "cuda_visible_devices" in stamp:
        block = stamp                       # the block itself was handed in
    if not isinstance(block, dict):
        return None
    payload = {k: v for k, v in block.items() if k != "STATUS"}
    try:
        return PeakDeviceMemory(**payload)
    except Exception:                                         # noqa: BLE001 — M19
        #  A block this reader cannot parse is still EVIDENCE that the field is
        #  present, so it must not read as "pre-change". It reads as a probe
        #  that produced something unusable, which is what it is.
        return PeakDeviceMemory(
            note=f"{CAPACITY_UNMEASURED}: the stamp carries a "
                 f"{CAPACITY_STAMP_KEY} block this reader cannot parse (keys "
                 f"{sorted(str(k) for k in block)}) — recorded as an "
                 f"unmeasured probe, NOT as a pre-change absence")


def quote_peak_memory(memory: Optional[PeakDeviceMemory]) -> str:
    """How a peak is printed in a comparison. Never a bare number."""
    return (CAPACITY_PRE_CHANGE_UNRECORDED if memory is None
            else memory.quoted)


# ---------------------------------------------------------------- banking
def bank(result: BuildResult, trunk: dict, corpus_sha: str, date_string: str,
         saboteur_used: bool = False) -> dict[str, Path]:
    """Write the vector, the covariance, the FD gate and the stamp. Returns paths."""
    req = result.request
    out = req.out_dir
    out.mkdir(parents=True, exist_ok=True)
    site = req.site
    key = req.model_key
    stem = req.stem                       # == f"entropy_gradient_{key}" unless overridden

    vectors = {CANONICAL_VECTOR_KEY.format(site=site): result.vector}
    for i, r in enumerate(result.random_band, start=1):
        vectors[CANONICAL_RANDOM_KEY.format(i=i, site=site)] = r
    vec_path = out / f"{stem}.npz"
    np.savez(vec_path, **vectors)

    sigma_path = out / f"sigma_L{site}_{key}.npz"
    np.savez(sigma_path, evals=result.sigma_evals, evecs=result.sigma_evecs,
             mean=result.sigma_mean, ridge=np.float64(result.diagnostics.sigma_ridge),
             n_positions=np.int64(result.diagnostics.n_sigma_positions))

    fd_path = out / f"{stem}_fd_gate.json"
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
        # ── THE EFFECTIVE THREAD CONFIGURATION (Luxia ruling 2026-08-01) ──────
        # UNCONDITIONAL, by the same argument that made the M41 hostname
        # unconditional: the whole point is that EVERY artifact can say what
        # instrument produced it, and a conditional field would leave exactly
        # the gap the ruling was made to close. An absent key therefore means
        # ONE thing — the stamp predates 2026-08-01 — and
        # `metabasis.threads.stamp_thread_config` is the reader that keeps that
        # apart from a probe that ran and could not answer.
        #
        # WHY IT IS READ HERE AND NOT AT LOAD. This vector's bytes come off
        # `np.linalg.eigh(Sigma)` (the band basis) and the reductions built on
        # it. eigh is bitwise-deterministic at a FIXED thread count and its
        # bytes differ ACROSS counts, so the count that matters is the one in
        # force when the decomposition ran — which is this side of it. The
        # trunk stamp beside it describes the deep-learning stack at LOAD time;
        # the two are separate facts and are recorded separately.
        THREAD_STAMP_KEY: thread_config_stamp(),
        # ── THE PEAK DEVICE MEMORY THIS BUILD REACHED (2026-08-04) ───────────
        # UNCONDITIONAL, by the thread_config argument one field over: the
        # point is that every artifact can evidence the capacity ruling it was
        # built under, and a conditional field would leave exactly the gap.
        # An absent key therefore means ONE thing — the stamp predates
        # 2026-08-04 — and `stamp_peak_device_memory` is the reader that keeps
        # that apart from a probe that ran and measured nothing (no CUDA).
        #
        # WHY IT IS READ HERE. The high-water marks are monotonic from process
        # start, so bank time — after the forwards, the backwards and the FD
        # ladder, and before the process exits — is the only point that sees
        # the whole build. It is a READ of counters the allocator already
        # maintains: it allocates nothing, resets nothing, and cannot change
        # one banked number.
        CAPACITY_STAMP_KEY: capacity_stamp(),
    }
    # ADDENDUM 2026-07-27-B: recorded in EVERY stamp of a truncated node, with the same
    # key names the collector uses so a build stamp and a collection stamp can be
    # compared field-for-field. Absent on every other node, so the historical key set is
    # untouched (same discipline as the collector's `sharding` record).
    if result.truncation is not None:
        t = result.truncation
        stamp["truncation"] = f"first-{t.max_length}"
        stamp["truncation_prereg"] = t.prereg
        stamp["n_truncated"] = t.n_truncated
        stamp["truncated_text_ids"] = t.truncated_text_ids
        stamp["truncation_audit"] = t.model_dump(exclude={"truncated_text_ids"})
    # The memory deviation, recorded ONLY when it was taken — same discipline as
    # `truncation` and the collector's `sharding`, so every historical build stamp
    # keeps exactly its key set and an absent key means the deviation was not
    # taken. Its presence says the gradient pass ran with layers above the site
    # recomputed; the agreement of that path with the un-checkpointed one is a
    # property of the BUILDER, proved in its selftest, not re-proved per bank.
    if req.checkpoint_above_site:
        stamp["checkpoint_above_site"] = {
            "n_layers_checkpointed": (
                result.diagnostics.n_layers_checkpointed_above_site),
            "layers": f"[{site + 1}:{result.diagnostics.n_decoder_layers}]",
            "mode": "torch.utils.checkpoint, use_reentrant=False (NON-REENTRANT: "
                    "the reentrant implementation requires .backward() and this "
                    "builder uses autograd.grad)",
            "site_layer_unwrapped": True,
            "why": "layers AT and BELOW the site are never wrapped, so the site "
                   "layer's forward fires exactly once and the substituted leaf "
                   "and its gradient are untouched. The re-entry guard is "
                   "re-asserted AFTER the backward on every text, which is what "
                   "proves the recompute did not re-enter the site layer",
            "agreement": "gradients with and without this flag agree to the "
                         "tolerance documented in the builder's selftest; "
                         "checkpointing changes WHEN a forward runs, not WHAT it "
                         "computes",
        }
    if saboteur_used:
        stamp["SELFTEST_SABOTEUR_APPLIED"] = True
    stamp_path = out / f"{stem}_stamps.json"
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



def _stack_available() -> tuple[bool, str]:
    """Is the deep-learning stack importable here? A PROBE, never a failure.

    RAKE M44(a)/(b): a selftest count must name the configuration it was
    measured in, and a suite with an availability-branched block runs in BOTH
    configurations before a merge verdict. This builder's end-to-end block
    needs torch + transformers to instantiate a tiny Llama; the desk's own repo
    venv (numpy/scipy/pydantic, no torch) is a legitimate environment, not a
    test failure, and before this probe existed `--selftest` died there with a
    ModuleNotFoundError before running a single check.
    """
    try:
        import torch                                          # noqa: F401
        import transformers                                   # noqa: F401
    except Exception as exc:                                  # noqa: BLE001 — a probe
        return False, f"{type(exc).__name__}: {exc}"
    return True, (f"torch {__import__('torch').__version__} + transformers "
                  f"{__import__('transformers').__version__}")


def _selftest_thread_config(check) -> None:
    """The thread-config stamp contract — NO deep-learning stack needed.

    Split OUT of the stack guard deliberately (the c6b34a3 pattern): the
    backward-compatibility contract every banked artifact depends on must stay
    proven in an environment with no torch at all, because that is the
    environment a desk reads stamps in.
    """
    from metabasis.threads import (PRE_RULING_UNRECORDED,
                                   THREAD_COUNT_MISMATCH_LABEL, ThreadConfig,
                                   thread_count_mismatch)
    block = thread_config_stamp()
    check("the thread-config block is JSON-round-trippable as banked",
          json.loads(json.dumps(block)) == block,
          f"{len(block)} field(s)")
    check("the block opens with STATUS and states the ruled default",
          list(block)[0] == "STATUS" and block["ruled_default"] == 8
          == RULED_OMP_NUM_THREADS,
          f"ruled_default={block['ruled_default']}")
    hand_built = {"builder": "build_entropy_gradient.py",
                  THREAD_STAMP_KEY: block}
    read_back = stamp_thread_config(hand_built)
    check("a stamp written by this builder reads back through the "
          "backward-compatible reader",
          read_back is not None
          and read_back.ruled_default == RULED_OMP_NUM_THREADS)
    check("a PRE-RULING stamp (no thread_config key) reads as None, never as "
          "a count — the historical banks are UNRECORDED, not single-threaded "
          "by inference",
          stamp_thread_config({"builder": "build_entropy_gradient.py"}) is None)
    banked_pre = None
    rebuilt = ThreadConfig(effective_num_threads=RULED_OMP_NUM_THREADS,
                           consensus="agreed", matches_ruled_default=True)
    note = thread_count_mismatch(banked_pre, rebuilt)
    check("a v3 rebuild compared against a banked PRE-RULING vector yields a "
          "LABELED count mismatch quoting both sides, never a bare byte diff",
          note is not None and THREAD_COUNT_MISMATCH_LABEL in note
          and PRE_RULING_UNRECORDED in note and "rebuild 8" in note,
          (note or "")[:88])


def _selftest_capacity_stamp(check: Callable[..., None],
                             skip: Callable[[str], None]) -> None:
    """The peak-memory stamp contract — NO deep-learning stack needed.

    Split out for the same reason as the thread-config block: the
    backward-compatibility contract is what every future capacity ruling will
    be read through, and it must stay proven in the environment a desk reads
    stamps in, which has no torch at all.

    RAKE M44: the MEASURED half is availability-branched on CUDA and registers
    a NAMED SKIP where no device exists — a skip is a third state, never a
    failure and never a silently-passed check.
    """
    block = capacity_stamp()
    check("the peak-memory block is JSON-round-trippable as banked",
          json.loads(json.dumps(block)) == block, f"{len(block)} field(s)")
    check("the block opens with STATUS, so a reader who has never seen the "
          "field knows in one line what window it covers",
          list(block)[0] == "STATUS" and block["STATUS"] == CAPACITY_STAMP_STATUS)

    #  THE UNCONDITIONAL SHAPE, checked as a SHAPE: the measured and unmeasured
    #  documents must carry exactly the same keys, or "absent = pre-change
    #  vintage" stops being true the first time a build runs without a card.
    measured_fixture = PeakDeviceMemory(
        measured=True, n_devices=2, cuda_visible_devices="0,1",
        torch_version="fixture", probe="fixture",
        devices=[DevicePeakMemory(index=0, name="fixture-card",
                                  max_memory_allocated_bytes=3 * 1024 ** 3,
                                  max_memory_reserved_bytes=4 * 1024 ** 3,
                                  max_memory_allocated_gib=3.0,
                                  max_memory_reserved_gib=4.0,
                                  total_capacity_gib=80.0),
                 DevicePeakMemory(index=1, name="fixture-card",
                                  max_memory_allocated_bytes=1024 ** 3,
                                  max_memory_reserved_bytes=2 * 1024 ** 3,
                                  max_memory_allocated_gib=1.0,
                                  max_memory_reserved_gib=2.0,
                                  total_capacity_gib=80.0)],
        peak_allocated_gib=3.0, peak_reserved_gib=4.0)
    unmeasured_fixture = PeakDeviceMemory(
        probe="fixture", note=f"{CAPACITY_UNMEASURED}: fixture")
    check("the MEASURED and UNMEASURED blocks carry exactly the same key set — "
          "the unconditional shape 'absent = pre-change vintage' depends on",
          (list(capacity_stamp(measured_fixture))
           == list(capacity_stamp(unmeasured_fixture))),
          f"{len(capacity_stamp(unmeasured_fixture))} key(s)")
    check("an UNMEASURED block always says why (M19: a degraded or absent read "
          "is described, never silent) and never reports a 0 peak",
          unmeasured_fixture.note.startswith(CAPACITY_UNMEASURED)
          and unmeasured_fixture.peak_allocated_gib is None
          and unmeasured_fixture.devices == [],
          unmeasured_fixture.note[:60])
    check("the WORST device is the reported peak, per side — a reservation is "
          "compared against the card that got closest to it",
          measured_fixture.peak_allocated_gib == 3.0
          and measured_fixture.peak_reserved_gib == 4.0
          and "3.000 GiB allocated" in measured_fixture.quoted,
          measured_fixture.quoted)
    check("allocated and reserved are BOTH carried, per device — the gap "
          "between them is the fragmentation a margin exists to cover",
          all(d.max_memory_allocated_bytes is not None
              and d.max_memory_reserved_bytes is not None
              and d.max_memory_reserved_bytes >= d.max_memory_allocated_bytes
              for d in measured_fixture.devices))

    #  THE READER, on the three states it must keep apart.
    check("a PRE-CHANGE stamp (no peak_device_memory key) reads as None — the "
          "historical banks are UNRECORDED, never 'used no memory'",
          stamp_peak_device_memory({"builder": "build_entropy_gradient.py"})
          is None
          and stamp_peak_device_memory(None) is None
          and stamp_peak_device_memory([]) is None)          # type: ignore[arg-type]
    round_tripped = stamp_peak_device_memory(
        {"builder": "x", CAPACITY_STAMP_KEY: capacity_stamp(measured_fixture)})
    check("a stamped block round-trips through the backward-compatible reader "
          "with its peaks intact",
          round_tripped is not None and round_tripped.measured
          and round_tripped.peak_allocated_gib == 3.0
          and round_tripped.n_devices == 2,
          quote_peak_memory(round_tripped))
    nested = stamp_peak_device_memory(
        {"trunk": {CAPACITY_STAMP_KEY: capacity_stamp(measured_fixture)}})
    bare = stamp_peak_device_memory(capacity_stamp(measured_fixture))
    check("a block nested under `trunk`, and the block handed in on its own, "
          "are both found — all three shapes get handed around",
          nested is not None and nested.peak_allocated_gib == 3.0
          and bare is not None and bare.peak_allocated_gib == 3.0)
    broken = stamp_peak_device_memory({CAPACITY_STAMP_KEY: {"nonsense": 1}})
    check("an UNPARSEABLE block is an unmeasured probe, NEVER a pre-change "
          "absence — the field's presence is itself evidence",
          broken is not None and not broken.measured
          and CAPACITY_UNMEASURED in broken.note,
          (broken.note if broken else "")[:60])
    check("a PRE-CHANGE side is QUOTED as unrecorded rather than compared as a "
          "number, so a capacity ruling never rests on an inferred peak",
          quote_peak_memory(None) == CAPACITY_PRE_CHANGE_UNRECORDED)

    #  THE LIVE PROBE. It must never raise, in any configuration.
    live = peak_device_memory()
    check("the live probe returns a PeakDeviceMemory in every environment and "
          "never raises (rake M19: instrumentation cannot fail a build)",
          isinstance(live, PeakDeviceMemory)
          and (live.measured or bool(live.note)),
          f"measured={live.measured} {live.note[:48]}")
    if live.measured:
        check("with CUDA present the peaks are read per device, and the "
              "reported peak is the worst of them",
              live.n_devices >= 1 and live.devices
              and live.peak_allocated_gib is not None
              and live.peak_allocated_gib
              == max(d.max_memory_allocated_gib or 0.0 for d in live.devices),
              live.quoted)
    else:
        skip(f"NAMED SKIP — the MEASURED half of the capacity stamp: no CUDA "
             f"device answered in this configuration ({live.note[:70]}), so "
             f"only the unmeasured branch and the reader contract run here. "
             f"The measured branch runs on a node with a card, in the same "
             f"--selftest.")


def _selftest_with_stack(check: Callable[..., None]) -> None:
    """The end-to-end block: a real tiny Llama, real algebra, fp32. NEEDS torch.

    Moved here VERBATIM from `selftest()` when the M44 availability branch was
    added — not one assertion of it was rewritten, so the stack-present
    behaviour is the behaviour that was already certified.
    """
    import tempfile

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

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

    # ------- --checkpoint-above-site: THE AGREEMENT GATE (prereg §4, made real) -----
    #  A DEEPER toy than the one above, so several layers sit above the site and the
    #  recompute machinery is actually exercised: 6 layers, site 2 => layers[3:6]
    #  wrapped. The whole build runs twice and the two results are compared.
    torch.manual_seed(20260729)
    cfg_ck = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=6, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=False)
    model_ck = LlamaForCausalLM(cfg_ck).to(torch.float32).eval()
    model_ck.requires_grad_(False)
    req_ck = req.model_copy(update={"site": 2})
    plain_ck = build_entropy_gradient(model_ck, tok, entries, req_ck,
                                      "12 Jul 2026", "cpu")
    with_ck = build_entropy_gradient(
        model_ck, tok, entries,
        req_ck.model_copy(update={"checkpoint_above_site": True}),
        "12 Jul 2026", "cpu")

    n_wrapped = with_ck.diagnostics.n_layers_checkpointed_above_site
    check("the flag wraps exactly the layers ABOVE the site, and nothing else",
          n_wrapped == cfg_ck.num_hidden_layers - req_ck.site - 1 == 3
          and plain_ck.diagnostics.n_layers_checkpointed_above_site == 0,
          f"{n_wrapped} of {cfg_ck.num_hidden_layers} layers wrapped "
          f"(layers[{req_ck.site + 1}:{cfg_ck.num_hidden_layers}]); the "
          f"un-checkpointed build reports 0")
    v_plain = plain_ck.vector.astype(np.float64)
    v_ckpt = with_ck.vector.astype(np.float64)
    max_dev = float(np.max(np.abs(v_plain - v_ckpt)))
    check("THE AGREEMENT GATE: the built vector is BITWISE identical with and "
          "without checkpointing",
          np.array_equal(plain_ck.vector, with_ck.vector),
          f"max |Δ| = {max_dev:.3e} over {v_plain.size} components "
          f"(float32 eps {float(np.finfo(np.float32).eps):.3e}); on CPU fp32 the "
          f"recompute runs the same kernels in the same order on the same saved "
          f"boundary tensors, so equality is exact — asserted strictly so a torch "
          f"release that changed recompute semantics fails loudly")
    check("and it satisfies the documented numeric CONTRACT tolerance",
          max_dev <= CHECKPOINT_AGREEMENT_ATOL,
          f"max |Δ| {max_dev:.3e} <= {CHECKPOINT_AGREEMENT_ATOL:.0e} — the bound a "
          f"bf16 GPU build is held to, where the recompute may pick a different "
          f"kernel and byte-equality is not guaranteed")
    check("the two vectors are parallel to machine precision",
          abs(1.0 - float(v_plain @ v_ckpt) / (np.linalg.norm(v_plain)
                                               * np.linalg.norm(v_ckpt))) < 1e-12,
          f"1 − cos = "
          f"{abs(1.0 - float(v_plain @ v_ckpt) / (np.linalg.norm(v_plain) * np.linalg.norm(v_ckpt))):.3e} "
          f"— the direction is the object of record, so its agreement is asserted "
          f"separately from the components'")
    for field in ("mean_gradient_norm", "band_projected_norm",
                  "band_energy_fraction", "per_text_band_pairwise_coherence",
                  "mean_entropy_nats", "vector_norm", "sigma_top_eigenvalue",
                  "n_grad_positions", "n_sigma_positions"):
        a_val = getattr(plain_ck.diagnostics, field)
        b_val = getattr(with_ck.diagnostics, field)
        check(f"diagnostic `{field}` agrees exactly",
              abs(float(a_val) - float(b_val)) == 0.0,
              f"{a_val} vs {b_val}")
    check("per-text band sign consistency is unchanged",
          plain_ck.diagnostics.per_text_band_sign_consistency
          == with_ck.diagnostics.per_text_band_sign_consistency,
          plain_ck.diagnostics.per_text_band_sign_consistency)
    check("and the MANDATORY FD gate reaches the same verdict, at the same rung",
          plain_ck.fd_gate.passed == with_ck.fd_gate.passed
          and plain_ck.fd_gate.best_eps_fraction == with_ck.fd_gate.best_eps_fraction
          and abs(plain_ck.fd_gate.best_median_rel_error
                  - with_ck.fd_gate.best_median_rel_error) == 0.0,
          f"passed={with_ck.fd_gate.passed}, median rel.err "
          f"{with_ck.fd_gate.best_median_rel_error:.6e} at eps_frac "
          f"{with_ck.fd_gate.best_eps_fraction} — identical, which also shows the "
          f"wrapped forwards were RESTORED before the gate's no-grad passes ran")

    #  THE RE-ENTRY GUARD is what makes the design safe, so its two halves are
    #  demonstrated rather than described: the site layer's forward fires exactly
    #  once even under checkpointing (proved by the build above completing — the
    #  guard raises otherwise, before AND after the backward), and wrapping the
    #  SITE layer itself would trip it.
    layers_ck = _decoder_layers(model_ck)
    with checkpoint_above_site(layers_ck, req_ck.site) as wrapped_now:
        check("inside the scope, only layers above the site carry a wrapped forward",
              wrapped_now == 3
              and all(layers_ck[i].forward.__qualname__.endswith("_checkpointed")
                      for i in range(req_ck.site + 1, len(layers_ck)))
              and not any(
                  layers_ck[i].forward.__qualname__.endswith("_checkpointed")
                  for i in range(0, req_ck.site + 1)),
              f"{wrapped_now} wrapped; layers[0:{req_ck.site + 1}] untouched — the "
              f"site layer's own forward is NEVER wrapped, which is what keeps the "
              f"leaf substitution single-call")
    check("and every forward is RESTORED on scope exit",
          not any(getattr(layer.forward, "__qualname__", "").endswith("_checkpointed")
                  for layer in layers_ck),
          "a module left carrying a wrapped forward would silently checkpoint the "
          "FD gate's no-grad passes and any later build in the same process")
    with checkpoint_above_site(layers_ck, len(layers_ck) - 1) as top_wrapped:
        check("a site at the TOP of the stack wraps 0 layers and is not an error",
              top_wrapped == 0,
              "the flag then costs nothing and changes nothing, which is the "
              "honest behaviour rather than a refusal")
    try:
        with checkpoint_above_site(layers_ck, len(layers_ck)):
            check("an out-of-range site must be refused", False)
    except SiteOutOfRangeError:
        check("an out-of-range site is refused by the checkpoint scope too", True)
    #  WHY THE SITE LAYER MUST NEVER BE WRAPPED — demonstrated at unit grain on the
    #  REAL hook class, rather than asserted in a comment. Two mechanisms exist and
    #  they behave differently, which is exactly why this needs a measurement:
    #
    #    * wrapping a module's `.forward` (what `checkpoint_above_site` does) does
    #      NOT re-invoke forward pre-hooks, because pre-hooks fire in `__call__` and
    #      the recompute re-runs the captured `forward` directly. A second,
    #      independent reason the design is safe.
    #    * wrapping a module's `__call__` — which is what
    #      `model.gradient_checkpointing_enable()` does in transformers, and what
    #      prereg §4's bare phrase reads as — DOES re-invoke the pre-hook on
    #      recompute, so the leaf is replaced with the recompute pass's copy and the
    #      gradient would be taken w.r.t. a tensor that never fed the loss.
    #
    #  The second case is reproduced here on a version-proof stand-in module (a real
    #  Llama decoder layer's standalone signature moves across transformers
    #  releases; the hook and the hazard do not).
    import torch.nn as _nn
    import torch.utils.checkpoint as _ckpt_mod

    class _Passthrough(_nn.Module):
        #  `x * x` and NOT `x * 2` on purpose: non-reentrant checkpointing
        #  recomputes LAZILY, when a saved tensor is unpacked, and a multiply by a
        #  constant saves nothing — so a passthrough would never trigger a
        #  recompute and the demonstration would pass for the wrong reason.
        def forward(self, hidden_states: Any) -> Any:    # noqa: D102
            return hidden_states * hidden_states

    probe_module = _Passthrough()
    probe_hook = _LeafSubstitution(probe_module)
    try:
        probe_hook.arm()
        x_probe = torch.randn(1, 3, 4, requires_grad=True)
        out_probe = _ckpt_mod.checkpoint(probe_module.__call__, x_probe,
                                         use_reentrant=False)
        leaf_probe = probe_hook.check_single_call("probe forward")
        check("a __call__-level checkpoint leaves the guard clean BEFORE the "
              "backward", probe_hook.calls == 1,
              "which is exactly why the guard must be re-asserted afterwards — "
              "one look cannot see a recompute that has not happened yet")
        torch.autograd.grad(out_probe.sum(), leaf_probe, allow_unused=True)
        probe_hook.check_single_call("probe AFTER backward")
        check("checkpointing at __call__ level must trip the re-entry guard",
              False, f"no raise — hook fired {probe_hook.calls} time(s)")
    except LeafHookError as exc:
        check("checkpointing at __call__ level TRIPS the re-entry guard, loudly, "
              "and only the POST-BACKWARD look catches it",
              "fired 2 times" in str(exc),
              f"{str(exc)[:118]}… — which is why the flag wraps `.forward` of "
              f"layers[site+1:] and never the site module's `__call__`")
    finally:
        probe_hook.remove()

    with tempfile.TemporaryDirectory() as td_ck:
        paths_ck = bank(with_ck.model_copy(update={
            "request": req_ck.model_copy(update={"out_dir": Path(td_ck),
                                                 "checkpoint_above_site": True})}),
            {"device": "cpu", "cuda_visible_devices": "(unset)"}, "0" * 64,
            "12 Jul 2026")
        st_ck = json.loads(paths_ck["stamp"].read_text())
        check("a checkpointed build RECORDS the deviation in its stamp",
              st_ck["checkpoint_above_site"]["n_layers_checkpointed"] == 3
              and "use_reentrant=False" in st_ck["checkpoint_above_site"]["mode"]
              and st_ck["checkpoint_above_site"]["site_layer_unwrapped"] is True,
              str(st_ck["checkpoint_above_site"]["layers"]))
        paths_plain = bank(plain_ck.model_copy(update={
            "request": req_ck.model_copy(update={"out_dir": Path(td_ck),
                                                 "out_stem": "plain"})}),
            {"device": "cpu", "cuda_visible_devices": "(unset)"}, "0" * 64,
            "12 Jul 2026")
        check("and an UN-checkpointed build's stamp has no such key at all",
              "checkpoint_above_site" not in json.loads(
                  paths_plain["stamp"].read_text()),
              "an absent key means the deviation was not taken — the same "
              "discipline `truncation` and the collector's `sharding` keep, so "
              "every historical stamp is unchanged key-for-key")
    check("the request field defaults OFF, so every historical build is reproduced",
          BuildRequest(model_key="x", model_path="p", arm_root=Path("."),
                       out_dir=Path("."), site=0,
                       arm="native").checkpoint_above_site is False)

    # ---------------- ADDENDUM 2026-07-27-B: the position-ceiling truncation --------
    # Five texts whose NATIVE-arm lengths are 13/23/33/43/53 tokens under the toy
    # tokenizer (3-token prompt prefix + 10/20/30/40/50 completion tokens). At a
    # ceiling of 33 exactly two are truncated, and the third sits EXACTLY on the
    # ceiling — the boundary case the collector re-tokenizes rather than assumes,
    # and the one a naive `len(ids) == max_length` count gets wrong.
    from metabasis.scripts.collect_mean_states import build_ids as _build_ids
    trunc_entries = [{"text_id": f"X{i:03d}", "stratum": "S1", "system_prompt": "",
                      "user_prompt": f"p{i}", "text": "a" * (10 + 10 * i)}
                     for i in range(5)]
    lens = [len(_build_ids(tok, e, "native", "12 Jul 2026")[0]) for e in trunc_entries]
    check("toy truncation fixture has the intended lengths", lens == [13, 23, 33, 43, 53],
          str(lens))
    aud = audit_truncation(tok, trunc_entries, "native", "12 Jul 2026", 33,
                           [e["text_id"] for e in trunc_entries])
    check("truncation audit counts only texts that actually lost tokens",
          aud.n_truncated == 2 and aud.truncated_text_ids == ["X003", "X004"],
          f"{aud.n_truncated}: {aud.truncated_text_ids}")
    check("a text sitting EXACTLY on the ceiling is not counted as truncated",
          "X002" not in aud.truncated_text_ids)
    aud_sub = audit_truncation(tok, trunc_entries, "native", "12 Jul 2026", 33,
                               ["X000", "X004"])
    check("audit separates the corpus-wide count from this build's subset",
          aud_sub.n_truncated == 2 and aud_sub.n_truncated_in_build == 1
          and aud_sub.truncated_in_build_ids == ["X004"],
          f"corpus {aud_sub.n_truncated} / build {aud_sub.n_truncated_in_build}")
    try:
        audit_truncation(tok, trunc_entries, "native", "12 Jul 2026", 33,
                         [e["text_id"] for e in trunc_entries], expected=148)
        check("a truncation count disagreeing with the ratified one is REJECTED", False)
    except TruncationAuditError as exc:
        check("a truncation count disagreeing with the ratified one is REJECTED",
              "148" in str(exc))
    check("the ratified count is accepted when it holds",
          audit_truncation(tok, trunc_entries, "native", "12 Jul 2026", 33,
                           [], expected=2).expected_n_truncated == 2)
    # end to end: the ceiling must actually reach the forward passes, not just the audit
    trunc_corpus = _toy_corpus(24)
    for i, e in enumerate(trunc_corpus):
        e["text"] = "a" * (10 + 4 * i)                 # 13..105 native-arm tokens
    req_trunc = req.model_copy(update={"max_seq_len": 40, "n_sigma": 12, "n_grad": 6})
    res_trunc = build_entropy_gradient(model, tok, trunc_corpus, req_trunc,
                                       "12 Jul 2026", "cpu")
    cap_positions = sum(
        min(len(_build_ids(tok, e, "native", "12 Jul 2026")[0]), 40)
        - _build_ids(tok, e, "native", "12 Jul 2026")[1]
        for e in [trunc_corpus[i] for i in stride_indices(24, 6, True)])
    check("the ceiling reaches the GPU passes (gradient positions are capped)",
          res_trunc.diagnostics.n_grad_positions == cap_positions,
          f"{res_trunc.diagnostics.n_grad_positions} == {cap_positions}")
    check("a truncated build carries its audit into the result",
          res_trunc.truncation is not None and res_trunc.truncation.max_length == 40
          and res_trunc.truncation.n_truncated > 0,
          f"n_truncated={res_trunc.truncation.n_truncated if res_trunc.truncation else None}")
    check("an untruncated build carries NO truncation record (historical key set)",
          res.truncation is None)
    with tempfile.TemporaryDirectory() as td:
        paths_t = bank(res_trunc.model_copy(
            update={"request": req_trunc.model_copy(update={"out_dir": Path(td)})}),
            {"device": "cpu", "cuda_visible_devices": "(unset)"}, "0" * 64, "12 Jul 2026")
        st = json.loads(paths_t["stamp"].read_text())
        check("truncated stamp carries the collector's own key names",
              st["truncation"] == "first-40" and st["truncation_prereg"] ==
              "ADDENDUM 2026-07-27-B" and st["n_truncated"] ==
              res_trunc.truncation.n_truncated
              and len(st["truncated_text_ids"]) == st["n_truncated"],
              st["truncation"])

    # ---------------- --out-name: banking BESIDE an existing vector -----------------
    try:
        req.model_copy(update={"out_stem": "a/b"})
        BuildRequest(model_key="x", model_path="p", arm_root=Path("."), out_dir=Path("."),
                     site=0, arm="native", out_stem="dir/name")
        check("--out-name rejects a path", False)
    except Exception as exc:
        check("--out-name rejects a path", "STEM" in str(exc) or "stem" in str(exc))
    try:
        BuildRequest(model_key="x", model_path="p", arm_root=Path("."), out_dir=Path("."),
                     site=0, arm="native", out_stem="name.npz")
        check("--out-name rejects a .npz suffix", False)
    except Exception as exc:
        check("--out-name rejects a .npz suffix", "STEM" in str(exc) or "stem" in str(exc))
    check("default stem is the readout's first-probed convention",
          req.stem == "entropy_gradient_toy", req.stem)

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
        #  The ruling's field, verified on a REAL banked stamp rather than on a
        #  hand-built one: the block has to survive the json round-trip that the
        #  bank actually performs, and it has to be findable by the reader every
        #  downstream consumer uses.
        banked_threads = stamp_thread_config(stamp)
        check("stamp records the EFFECTIVE thread configuration (the ruling's "
              "instrument identity), readable by the shared reader",
              THREAD_STAMP_KEY in stamp and banked_threads is not None,
              f"{THREAD_STAMP_KEY}={stamp.get(THREAD_STAMP_KEY, {}).get('consensus')}"
              f" -> {banked_threads.quoted if banked_threads else None}")
        check("and it states the ruled default beside the measured count, so "
              "the artifact says what the convention WAS when it was built",
              banked_threads is not None
              and banked_threads.ruled_default == RULED_OMP_NUM_THREADS,
              f"ruled {RULED_OMP_NUM_THREADS}")
        #  The capacity field on a REAL banked stamp, for the same reason: it
        #  has to survive the json round-trip the bank actually performs and be
        #  findable by the reader a desk will use.
        banked_memory = stamp_peak_device_memory(stamp)
        check("stamp records the PEAK DEVICE MEMORY this build reached, "
              "readable by the backward-compatible reader",
              CAPACITY_STAMP_KEY in stamp and banked_memory is not None,
              quote_peak_memory(banked_memory))
        check("and the capacity block is present UNCONDITIONALLY — a CPU build "
              "records a NAMED absence, so an absent key can only ever mean "
              "'this stamp predates the field'",
              banked_memory is not None
              and (banked_memory.measured or CAPACITY_UNMEASURED
                   in banked_memory.note),
              (banked_memory.note if banked_memory else "")[:60])

        # The 8bL16 deconfound's shape: a SECOND vector for the same model at a site
        # the legacy bank already holds. The stem must move the files while the npz
        # KEY stays canonical — otherwise the rebuild either clobbers the legacy bank
        # (same stem) or becomes unreadable by the readout (moved key).
        stem2 = "entropy_gradient_toy_L2rebuild"
        paths2 = bank(res.model_copy(update={
            "request": banked_req.model_copy(update={"out_stem": stem2})}),
            {"device": "cpu", "cuda_visible_devices": "(unset)"}, "0" * 64, "12 Jul 2026")
        check("--out-name moves the vector/stamp/FD-gate trio",
              paths2["vector"].name == f"{stem2}.npz"
              and paths2["stamp"].name == f"{stem2}_stamps.json"
              and paths2["fd_gate"].name == f"{stem2}_fd_gate.json",
              paths2["vector"].name)
        check("--out-name does NOT overwrite the default-stem bank beside it",
              paths["vector"].exists() and paths2["vector"] != paths["vector"])
        with np.load(paths2["vector"]) as z2:
            check("--out-name keeps the CANONICAL npz key (the readout's contract)",
                  "entropy_gradient_L2" in z2.files, str(sorted(z2.files)))
        v2, spec2 = load_entropy_gradient(paths2["vector"], "toy", 2)
        check("readout loads a renamed bank by the same canonical key",
              spec2.key == "entropy_gradient_L2"
              and float(np.max(np.abs(v2 - v))) == 0.0)


def selftest() -> int:
    """End-to-end CPU verification: real algebra, tiny random model, fp32.

    NAMED CONFIGURATIONS (rake M44). One availability axis — the deep-learning
    stack — and the reported count says which side of it ran. The blocks that
    need no stack (the thread-config stamp contract) run in BOTH, so the
    backward-compatibility contract every banked artifact depends on is proved
    with no torch at all. A skip is a THIRD state, never a non-zero exit code.
    """
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    skips: list[str] = []

    def skip(name: str) -> None:
        skips.append(name)
        logger.info("SKIP %s", name)

    #  Runs in EVERY configuration — the stamp contracts need no weights.
    _selftest_thread_config(check)
    _selftest_capacity_stamp(check, skip)

    stack_ok, stack_why = _stack_available()
    if stack_ok:
        _selftest_with_stack(check)
    else:
        skip("deep-learning stack: the end-to-end build/FD-gate/checkpoint/"
             "truncation/banking blocks instantiate a tiny Llama and need "
             f"torch + transformers ({stack_why}). A venv without them is a "
             "legitimate environment (the desk's CPU spine is one), not a test "
             "failure; run this half where the stack is installed")

    n_miss = sum(1 for _, ok, _ in checks if not ok)
    print(json.dumps({"checks": len(checks), "misses": n_miss,
                      "named_skips": len(skips), "skipped": skips,
                      "configuration": ("stack(" + stack_why + ")" if stack_ok
                                        else "no stack(" + stack_why + ")"),
                      "failed": [n for n, ok, _ in checks if not ok]}, indent=1))
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
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
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="prereg ADDENDUM 2026-07-27-B position ceiling: truncate every "
                         "text to its FIRST N tokens. MUST equal the node's registry "
                         "`max_seq_len` (gpt2-xl: 1024) — the addendum requires one "
                         "truncation across every use of the node")
    ap.add_argument("--assert-n-truncated", type=int, default=None,
                    help="refuse to build unless the corpus-wide truncation audit "
                         "reproduces exactly this many truncated texts (gpt2-xl: 148)")
    ap.add_argument("--out-name", default=None,
                    help="filename STEM for the vector/stamp/FD-gate trio (no dir, no "
                         ".npz); default entropy_gradient_<model>. Use only to bank "
                         "BESIDE an existing vector of the same model (the 8bL16 "
                         "construction-lineage rebuild)")
    ap.add_argument("--shard-across", default=None,
                    help="comma list of devices (e.g. '0,1') to split the decoder stack "
                         "across, via the COLLECTOR's certified sharded loader. Same "
                         "layout the >=70B rungs collected under (prereg §4; sharding "
                         "gate PASSED 2026-07-27 — forced 8-way sharding was byte-"
                         "identical to fresh single-device). Use when the weights plus "
                         "the retained autograd graph do not fit one card")
    ap.add_argument("--entropy-chunk", type=int, default=0,
                    help="OOM fallback: compute the entropy in position blocks of this "
                         "size (same algebra, different fp32 summation order)")
    ap.add_argument("--checkpoint-above-site", action="store_true",
                    help="OOM fallback for the GRADIENT PASS: run layers[site+1:] "
                         "under NON-REENTRANT activation checkpointing "
                         "(torch.utils.checkpoint, use_reentrant=False), trading "
                         "one extra forward of those layers for their retained "
                         "activations. Layers AT and BELOW the site are NEVER "
                         "wrapped, so the substituted leaf and its gradient are "
                         "untouched and the re-entry guard still sees exactly one "
                         "site-layer call — re-asserted after the backward on "
                         "every text. Non-reentrant is not a preference: the "
                         "reentrant implementation requires .backward() and this "
                         "builder uses autograd.grad, which is what keeps "
                         "per-parameter grad buffers unallocated. Unset = the "
                         "historical path, byte for byte. Reach for this when "
                         "--entropy-chunk was not enough and the pressure is the "
                         "retained per-layer graph rather than the logits (the "
                         "DeepSeek-V3 case).")
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
            entropy_chunk=args.entropy_chunk, max_seq_len=args.max_seq_len,
            expect_n_truncated=args.assert_n_truncated, out_stem=args.out_name,
            checkpoint_above_site=args.checkpoint_above_site)
    except ValueError as exc:
        raise SystemExit(f"invalid build request: {exc}") from exc
    if args.assert_n_truncated is not None and args.max_seq_len is None:
        raise SystemExit("--assert-n-truncated is meaningless without --max-seq-len")

    from metabasis.scripts.collect_mean_states import (
        VMB_CANONICAL_DATE, DeviceMapSpecError, ShardSpec, ShardedLoadError,
        _device_key, load_corpus, load_model_and_tok, load_model_and_tok_sharded,
        trunk_stamp)

    try:
        entries, corpus_sha = load_corpus(request.arm_root)
    except (OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"corpus unreadable under {request.arm_root}: {exc}") from exc
    logger.info("corpus: %d texts, manifest sha256 %s", len(entries), corpus_sha[:12])

    # `device` is where the INPUT ids tensor is built. On the single-card path that is
    # --device; on the sharded path accelerate decides, so the loader reports it and the
    # forward passes must follow it (exactly the collector's rule).
    sharding: Optional[dict] = None
    try:
        if args.shard_across:
            spec = ShardSpec(
                shard_across=[_device_key(t) for t in args.shard_across.split(",")],
                require_multi_device=True,
                spec_echo=f"--shard-across {args.shard_across}")
            model, tok, device, sharding = load_model_and_tok_sharded(
                request.model_path, spec, (request.site,))
        else:
            model, tok = load_model_and_tok(request.model_path, request.device)
            device = request.device
    except (OSError, ValueError, RuntimeError, DeviceMapSpecError,
            ShardedLoadError) as exc:
        raise SystemExit(
            f"cannot load {request.model_key} from {request.model_path}: "
            f"{type(exc).__name__}: {exc}") from exc
    # Freeze every parameter: backward then allocates NO per-parameter grad buffers
    # (the difference between ~20 GB peak and OOM on a 32B). The gradient still flows
    # THROUGH the frozen weights to reach the leaf; the vector is unchanged.
    model.requires_grad_(False)
    trunk = trunk_stamp(request.model_path, tok, device, sharding=sharding)
    logger.info("trunk: %s", json.dumps(trunk))

    try:
        result = build_entropy_gradient(model, tok, entries, request,
                                        VMB_CANONICAL_DATE, device)
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
                  "reason=OOM — nothing was banked. Retry with --entropy-chunk 256 "
                  "(same algebra, chunked log_softmax) if the logits are the pressure; "
                  "with --checkpoint-above-site if the pressure is the RETAINED "
                  "per-layer graph above the site (one extra forward of those layers "
                  "instead of their activations; the leaf and its gradient are "
                  "untouched, agreement proved in --selftest); or with --shard-across "
                  "(the certified sharded loader) if the weights themselves exceed one "
                  "card. NEVER free memory by touching another process.")
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
    if result.truncation is not None:
        summary["truncation"] = f"first-{result.truncation.max_length}"
        summary["n_truncated"] = result.truncation.n_truncated
        summary["n_truncated_in_build"] = result.truncation.n_truncated_in_build
    if sharding is not None:
        summary["n_compute_devices"] = sharding["n_compute_devices"]
        summary["compute_devices"] = sharding["compute_devices"]
    if request.checkpoint_above_site:
        summary["n_layers_checkpointed_above_site"] = (
            d.n_layers_checkpointed_above_site)
    summary["banked_stem"] = request.stem
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
