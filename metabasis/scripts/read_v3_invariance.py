"""webtext-v3 §6 invariance reads + §7 hub race / battery / roster reads.

WHAT THIS IS
────────────────────────────────────────────────────────────────────────────
ONE metric engine, computed once, consumed by both frozen sections. §6-I1/I2's
binding clauses are *defined as* §7 eligibility agreement across contexts, so a
second implementation of "the hub score" would be two contracts wearing one
name. Everything below reads from that single engine.

**C§8 — THIS LANE COMPUTES ARITHMETIC AND DECLARES NOTHING.** Every PASS/FAIL
it emits is UNSTAMPED. Stamps are the desk's, at the first-read.

THE ESTIMAND, AND WHY IT IS NOT RE-DERIVED HERE
────────────────────────────────────────────────────────────────────────────
`â_comp(A→B | H)` is `read_composed_predictions.composed_exchange_rate` and
`â_obs(A→B)` is `read_exchange_rates.exchange_rate` — both CALLED, never
mirrored. The direction resolution for a pair fit is
`read_composed_predictions.pair_fit_probes_under` (the §8.3 construction of
record, merged `3a67578`): the wave banks ONE semi-orthogonal Procrustes object
per UNORDERED pair and both ordered slots are read from it.

This module's only genuine EXTENSIONS over the merged lane, both minimal and
both flagged in the report:

  1. **Context parameterization.** The scoring lane hardcodes the full-corpus
     fit trees and `proc_k256`. §6 needs {full, half-a, half-b} × {k32, k128,
     k256}. The half trees carry the IDENTICAL file names inside a
     `…__half{a,b}` directory, so the extension is a directory-naming function
     (`leg_dir` / `pair_dir`) and NOTHING about the arithmetic moves.
  2. **A pair fit read AS a hub leg** (§7 clause 1 / §8 step 5's all-roster
     race, whose "remaining legs are the now-fit pair maps — the same fitted
     objects, acknowledged"). A hub leg has the hub on its SOURCE side; a pair
     fit banked as `M→H` has it on the TARGET side, so it is re-oriented with
     `read_composed_predictions.adjoint_map`, the lane's own adjoint of record
     (its selftest proves `adjoint(tm).fwd == tm.rev` to machine precision).
     No new arithmetic: the identity is the module's, used as intended.

THE VINTAGE RULE IS ABSOLUTE (prereg §3.6). v2.1 numbers appear in exactly one
place in this module — `i5_cross_basis`, the §6-I5 cross-basis measurement,
which is cross-basis BY DEFINITION. Nothing else in this file may read a v2.1
path, and the emitter asserts it.

DETERMINISM. The reads artifact carries no timestamp and no environment-varying
value; every such fact is in the sidecar. `--selftest` proves a build-twice
byte-identical artifact. Permutation nulls draw from a NAMED, frozen seed.
"""
from __future__ import annotations

import os

# The thread count is instrument identity (Luxia ruling 2026-08-01). Set before
# numpy imports so the BLAS pool is born at the ruled default.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from metabasis.scripts.fit_transport_maps import TransportMap, load_transport_map
from metabasis.scripts.read_exchange_rates import (
    exchange_rate, fit_path_for, load_entropy_gradient)
import metabasis.scripts.read_composed_predictions as rcp
from metabasis.threads import thread_config_stamp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("read_v3_invariance")


# ═══════════════════════════════════════════════════════════ frozen constants
#: §7 clause 3's practical-equivalence margin, frozen in the prereg.
DELTA_PRACTICAL_EQUIVALENCE: float = 0.005
#: §7 clause 3's Holm-corrected significance level.
ALPHA_TIER: float = 0.05
#: §6-I4's Holm-corrected significance level (note: .01, NOT .05).
ALPHA_SEPARATION: float = 0.01
#: §6-I1/I2's pre-stated full-order Spearman pass.
RHO_ORDER_PASS: float = 0.8
#: §7's HL-2 support thresholds.
RHO_HL2_FULL: float = 0.6
RHO_HL2_HALF: float = 0.4
RHO_HL2_PARTIAL: float = 0.4
ALPHA_HL2_PERM: float = 0.05
#: The permutation count and NAMED seed for every max-statistic null here.
N_PERMUTATIONS: int = 20000
PERMUTATION_SEED_NAME: str = "v3-reads-race-2026-08-04/max-statistic"

#: §7's race set, as BANK KEYS (the artifact records the prose→key mapping).
RACE_SET: tuple[str, ...] = ("qwen2.5-3b-instruct", "qwen2.5-32b-instruct",
                             "3b", "8b", "gemma3-27b")
#: §7 clause 1's candidate set (race set MINUS nothing — gemma is the control
#: and is raced with the rest; clause 6 gives gemma-top-tier its own reading).
CANDIDATES: tuple[str, ...] = RACE_SET

#: The §6 grid. `proc_k32` is a LABELED CONTINUITY BESIDE (§6-I2) and is never
#: admitted to a binding clause; the engine computes it, the clause code
#: refuses it.
BINDING_RANKS: tuple[str, ...] = ("proc_k128", "proc_k256")
CONTINUITY_RANK: str = "proc_k32"
ALL_RANKS: tuple[str, ...] = (CONTINUITY_RANK,) + BINDING_RANKS

Split = Literal["full", "halfa", "halfb"]
SPLITS: tuple[Split, ...] = ("full", "halfa", "halfb")

#: §3.2's family of record on webtext-v3 (the health check passed; no demotion).
RANK_OF_RECORD: str = "proc_k256"

#: §6-I5 pins this rank, and only this one — "the only rank quotable in both
#: bases' guards; v2.1's family of record".
I5_RANK: str = "proc_k128"

#: §4's rank guard: k quotable only where k <= n_train / 1.2.
RANK_GUARD_DIVISOR: float = 1.2


class ReadsError(RuntimeError):
    """This lane cannot compute what it was asked for, and says which clause."""


class ClusterBlocker(ReadsError):
    """The computation needs an input that exists only on the cluster.

    Raised, never worked around: the brief is desk-local CPU only, and a
    silently substituted input would produce a number whose label lies.
    """


# ═══════════════════════════════════════════════════════ context / tree layout
class Context(BaseModel):
    """ONE (split, rank) cell of the §6 grid. The engine's unit of work."""
    model_config = {"frozen": True}

    split: Split
    family: str

    @property
    def key(self) -> str:
        return f"{self.split}::{self.family}"

    @property
    def binding_eligible(self) -> bool:
        """§6-I2: k32 is a labeled continuity BESIDE, never a binding clause."""
        return self.family in BINDING_RANKS

    def __str__(self) -> str:                              # pragma: no cover
        return self.key


class TreeLayout(BaseModel):
    """Where the fit objects for each split live, as verified on disk.

    ONE place names the layout, so a relocation is one edit and a wrong guess
    is a refusal rather than an UNSCORED slot that looks like a measurement.
    """
    fits_root: Path

    def leg_dir(self, ctx: Context, hub: str, hub_site: int,
                model: str, site: int, arm: str) -> Path:
        stem = f"{hub}L{hub_site}__{model}L{site}__{arm}"
        if ctx.split == "full":
            return self.fits_root/"hub-legs"/"fits-hub-legs"/stem
        return self.fits_root/"halves"/"fits-halves"/"hub-legs"/f"{stem}__{ctx.split}"

    def pair_dir(self, ctx: Context, a: str, a_site: int,
                 b: str, b_site: int, arm: str) -> Path:
        stem = f"{a}L{a_site}__{b}L{b_site}__{arm}"
        if ctx.split == "full":
            return self.fits_root/"pairs"/"fits-pairs"/stem
        return self.fits_root/"halves"/"fits-halves"/"pairs"/f"{stem}__{ctx.split}"

    def leg_path(self, ctx: Context, hub: str, hub_site: int,
                 model: str, site: int, arm: str) -> Path:
        return fit_path_for(self.leg_dir(ctx, hub, hub_site, model, site, arm),
                            hub, hub_site, model, site, arm, ctx.family)

    def pair_probes(self, ctx: Context, source: str, source_site: int,
                    target: str, target_site: int, arm: str
                    ) -> list[tuple[Path, str, str]]:
        """(path, direction, fit_pair_id) candidates, ORDER LOAD-BEARING.

        The order is `read_composed_predictions.pair_fit_probes_under`'s own —
        every slot-ordered probe first, then every reversed probe — so a slot
        whose own ordering is banked resolves exactly as the merged lane
        resolves it. Only the DIRECTORY is re-stemmed per split.
        """
        out: list[tuple[Path, str, str]] = []
        for direction, (a, a_site, b, b_site) in (
                ("fwd", (source, source_site, target, target_site)),
                ("rev", (target, target_site, source, source_site))):
            directory = self.pair_dir(ctx, a, a_site, b, b_site, arm)
            out.append((fit_path_for(directory, a, a_site, b, b_site, arm,
                                     ctx.family),
                        direction, f"{a}L{a_site}->{b}L{b_site}"))
        return out


class Banks(BaseModel):
    """Vector bank + fit trees + the sealed artifact, all resolved once."""
    model_config = {"arbitrary_types_allowed": True}

    vectors_root: Path
    layout: TreeLayout
    artifact_path: Path
    stamp_path: Optional[Path] = None


class _Cache:
    """Path-keyed map/vector cache. Cleared between contexts (memory, not speed).

    Not an lru_cache: the eviction point is a CONTEXT boundary, which is a fact
    about the work, and a size-based policy would thrash on the 265 objects one
    context needs while holding objects from a context already finished.
    """

    def __init__(self) -> None:
        self.maps: dict[str, TransportMap] = {}
        self.vectors: dict[tuple[str, int], np.ndarray] = {}
        self.n_map_loads = 0

    def tmap(self, path: Path) -> TransportMap:
        key = str(path)
        got = self.maps.get(key)
        if got is None:
            if not path.exists():
                raise ReadsError(f"fit object absent: {path}")
            got = load_transport_map(path)
            self.maps[key] = got
            self.n_map_loads += 1
        return got

    def vector(self, banks: Banks, model: str, site: int) -> np.ndarray:
        key = (model, site)
        got = self.vectors.get(key)
        if got is None:
            path = (banks.vectors_root/model
                    / f"entropy_gradient_{model}_L{site}.npz")
            if not path.exists():
                raise ReadsError(
                    f"{model} L{site}: entropy-gradient vector absent at {path} "
                    f"— the prediction was filed from it, so no observation may "
                    f"be read from anywhere else")
            got, _ = load_entropy_gradient(path, model, site)
            self.vectors[key] = got
        return got

    def clear_maps(self) -> None:
        self.maps.clear()


# ════════════════════════════════════════════════════════════ the slot basis
class BasisSlot(BaseModel):
    """One of the frozen 240 ordered slots, reduced to what the engine needs.

    Built ONLY from the sealed artifact — §7: "This set is enumerated in the
    prediction artifact BEFORE any hub leg is fit. No alternative basis is
    quotable except as a labeled beside."
    """
    model_config = {"frozen": True}

    ordinal: int
    prediction_id: str
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    pair_class: str

    @property
    def endpoints(self) -> frozenset[str]:
        return frozenset((self.source_model, self.target_model))


def basis_from_artifact(artifact: rcp.V3PredictionArtifact) -> list[BasisSlot]:
    """The fixed shared race basis, in the artifact's own ordinal order."""
    slots = [BasisSlot(ordinal=s.ordinal, prediction_id=s.prediction_id,
                       pair_id=s.pair_id, source_model=s.source_model,
                       source_site=s.source_site, target_model=s.target_model,
                       target_site=s.target_site, arm=s.arm,
                       pair_class=s.pair_class)
             for s in artifact.slots]
    slots.sort(key=lambda s: s.ordinal)
    if len(slots) != artifact.frozen_count_N:
        raise ReadsError(
            f"the sealed artifact carries {len(slots)} slots but declares "
            f"frozen_count_N={artifact.frozen_count_N}")
    return slots


# ══════════════════════════════════════════════════════════ the metric engine
class SlotError(BaseModel):
    """One slot's composed-vs-observed pairing under one hub and one context."""
    model_config = {"frozen": True}

    ordinal: int
    prediction_id: str
    arm: str
    a_comp: float
    a_obs: float
    error: float
    abs_error: float
    observed_direction: str
    observed_fit_pair_id: str


class HubContextScore(BaseModel):
    """§7 clause 2's per-hub score in ONE context, with its population."""

    hub: str
    hub_site: int
    context: str
    split: Split
    family: str
    leg_source: Literal["hub-leg-wave", "pair-fit-as-leg"]
    n_slots_realized: int
    n_slots_basis: int
    median_abs_error: float
    mean_abs_error: float
    arm_counts: dict[str, int]
    hub_basis_max_dev: float = Field(
        description="max |va_source - va_target| over the realized slots — the "
                    "two legs must share ONE hub-side PCA basis for the "
                    "two-hop to be composable; checked, never assumed")
    unrealized: list[str] = Field(
        default_factory=list,
        description="prediction ids this hub cannot quote, with the reason")
    errors: list[SlotError] = Field(default_factory=list, exclude=True)

    @property
    def by_ordinal(self) -> dict[int, SlotError]:
        return {e.ordinal: e for e in self.errors}


def _oriented_leg(cache: _Cache, banks: Banks, ctx: Context,
                  hub: str, hub_site: int, model: str, site: int, arm: str
                  ) -> tuple[TransportMap, str]:
    """The map hub→model, from a hub-leg object or a re-oriented pair fit.

    Returns (map, provenance). The hub-leg wave's object is preferred whenever
    it exists — it IS the hub leg — and the pair-fit route is what §8 step 5
    licenses for the non-race hubs ("its remaining legs are the now-fit pair
    maps — the same fitted objects, acknowledged").
    """
    leg = banks.layout.leg_path(ctx, hub, hub_site, model, site, arm)
    if leg.exists():
        return cache.tmap(leg), "hub-leg-wave"
    for path, direction, _pair_id in banks.layout.pair_probes(
            ctx, hub, hub_site, model, site, arm):
        if path.exists():
            tm = cache.tmap(path)
            if direction == "fwd":
                # banked hub→model already: the hub is on the source side.
                return tm, "pair-fit-as-leg"
            # banked model→hub: re-orient with the lane's adjoint of record.
            return rcp.adjoint_map(tm), "pair-fit-as-leg-adjoint"
    raise ReadsError(
        f"no leg {hub}L{hub_site}→{model}L{site} [{arm}] in {ctx.key}: neither "
        f"a hub-leg object nor a pair fit in either ordering is on disk")


def score_hub(banks: Banks, basis: Sequence[BasisSlot], ctx: Context,
              hub: str, hub_site: int, cache: _Cache,
              require_full_basis: bool = True) -> HubContextScore:
    """§7 clause 2 for ONE hub in ONE context, over the fixed shared basis.

    `require_full_basis=False` is the §8-step-5 all-roster mode: a non-race hub
    is an ENDPOINT of some slots and cannot quote them (its own leg would be
    the identity), and it has legs only in the arms its pairs were banked in.
    Those slots are recorded as `unrealized` WITH THEIR REASON and the hub's
    realized count is stated — never silently dropped into a smaller median.
    """
    errors: list[SlotError] = []
    unrealized: list[str] = []
    arm_counts: Counter[str] = Counter()
    max_dev = 0.0
    leg_sources: set[str] = set()

    for slot in basis:
        if hub in slot.endpoints:
            unrealized.append(f"{slot.prediction_id}: hub is an endpoint")
            continue
        try:
            tm_src, prov_a = _oriented_leg(cache, banks, ctx, hub, hub_site,
                                           slot.source_model, slot.source_site,
                                           slot.arm)
            tm_tgt, prov_b = _oriented_leg(cache, banks, ctx, hub, hub_site,
                                           slot.target_model, slot.target_site,
                                           slot.arm)
        except ReadsError as exc:
            if require_full_basis:
                raise
            unrealized.append(f"{slot.prediction_id}: {exc}")
            continue
        leg_sources.update((prov_a, prov_b))

        v_src = cache.vector(banks, slot.source_model, slot.source_site)
        v_tgt = cache.vector(banks, slot.target_model, slot.target_site)
        a_comp = rcp.composed_exchange_rate(tm_src, tm_tgt, v_src, v_tgt)

        va_a = np.asarray(tm_src.va, np.float64)
        va_b = np.asarray(tm_tgt.va, np.float64)
        if va_a.shape != va_b.shape:
            raise ReadsError(
                f"{hub} {ctx.key} {slot.prediction_id}: the two legs' hub-side "
                f"bases have shapes {va_a.shape} vs {va_b.shape} — not one hub "
                f"basis, so the two-hop is not composable")
        max_dev = max(max_dev, float(np.abs(va_a - va_b).max()))

        resolved = None
        for path, direction, fit_pair_id in banks.layout.pair_probes(
                ctx, slot.source_model, slot.source_site,
                slot.target_model, slot.target_site, slot.arm):
            if path.exists():
                resolved = (path, direction, fit_pair_id)
                break
        if resolved is None:
            if require_full_basis:
                raise ReadsError(
                    f"{ctx.key} {slot.prediction_id}: no direct pair fit in "
                    f"either ordering — â_obs has no source")
            unrealized.append(f"{slot.prediction_id}: no direct pair fit")
            continue
        path, direction, fit_pair_id = resolved
        a_obs = exchange_rate(cache.tmap(path), v_src, v_tgt,
                              direction=direction)  # type: ignore[arg-type]

        errors.append(SlotError(
            ordinal=slot.ordinal, prediction_id=slot.prediction_id,
            arm=slot.arm, a_comp=float(a_comp), a_obs=float(a_obs),
            error=float(a_comp - a_obs), abs_error=abs(float(a_comp - a_obs)),
            observed_direction=direction, observed_fit_pair_id=fit_pair_id))
        arm_counts[slot.arm] += 1

    if not errors:
        raise ReadsError(
            f"{hub} in {ctx.key}: not one slot of the frozen basis is "
            f"quotable — a median over nothing is not a score")
    abs_errors = np.array([e.abs_error for e in errors], dtype=np.float64)
    source: Literal["hub-leg-wave", "pair-fit-as-leg"] = (
        "hub-leg-wave" if leg_sources == {"hub-leg-wave"} else "pair-fit-as-leg")
    return HubContextScore(
        hub=hub, hub_site=hub_site, context=ctx.key, split=ctx.split,
        family=ctx.family, leg_source=source, n_slots_realized=len(errors),
        n_slots_basis=len(basis),
        median_abs_error=float(np.median(abs_errors)),
        mean_abs_error=float(abs_errors.mean()),
        arm_counts=dict(sorted(arm_counts.items())),
        hub_basis_max_dev=max_dev, unrealized=unrealized, errors=errors)


# ════════════════════════════════════════════════════════════════ statistics
class SignTest(BaseModel):
    """§7 clause 3(a)'s paired sign test. Two-sided, ties dropped, n stated."""

    label: str
    n_shared: int = Field(description="slots both hubs quote")
    n_ties: int
    n_effective: int = Field(description="n after ties are dropped — the n "
                                         "that is stated with every p")
    n_a_better: int = Field(description="slots where A's |e| is SMALLER")
    n_b_better: int
    p_two_sided: float
    winner: Optional[str] = None


def sign_test(label: str, a: Sequence[float], b: Sequence[float],
              name_a: str, name_b: str) -> SignTest:
    """Exact two-sided binomial sign test on paired |e|, ties DROPPED.

    Ties are dropped rather than split because §7 clause 3(a) says so
    ("two-sided, ties dropped, n stated with every p"), and the reported
    `n_effective` is the post-drop n — a p quoted against the pre-drop n would
    overstate the evidence.

    The exact binomial is used rather than a normal approximation: at these n
    (as small as the all-roster race's shared sets get) the approximation's
    error is the same order as the α it is compared against.
    """
    if len(a) != len(b):
        raise ReadsError(f"{label}: paired arrays differ in length "
                         f"({len(a)} vs {len(b)})")
    wins_a = wins_b = ties = 0
    for x, y in zip(a, b):
        if x == y:
            ties += 1
        elif x < y:
            wins_a += 1
        else:
            wins_b += 1
    n_eff = wins_a + wins_b
    if n_eff == 0:
        p = 1.0
    else:
        # Exact two-sided binomial at p=.5: symmetric, so the two-sided p is
        # 2 x the smaller tail, capped at 1. Computed from the pmf directly so
        # this file has no scipy dependency in its arithmetic of record.
        k = min(wins_a, wins_b)
        log_c = _log_binom_cdf(k, n_eff)
        p = min(1.0, 2.0 * float(np.exp(log_c)))
    winner = None
    if wins_a != wins_b:
        winner = name_a if wins_a > wins_b else name_b
    return SignTest(label=label, n_shared=len(a), n_ties=ties, n_effective=n_eff,
                    n_a_better=wins_a, n_b_better=wins_b, p_two_sided=p,
                    winner=winner)


def _log_binom_cdf(k: int, n: int) -> float:
    """log P(X <= k) for X ~ Binomial(n, .5), summed in log space.

    Log space because n reaches 240 here and 2**-240 underflows a float64 term
    long before the sum does.
    """
    from math import lgamma, log
    terms = []
    for i in range(k + 1):
        terms.append(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                     - n * log(2.0))
    m = max(terms)
    return m + float(np.log(np.exp(np.array(terms) - m).sum()))


class HolmRow(BaseModel):
    label: str
    p_raw: float
    p_holm: float
    rejected_at_alpha: bool


def holm(labelled_p: Sequence[tuple[str, float]], alpha: float
         ) -> list[HolmRow]:
    """Holm–Bonferroni step-down over a NAMED family. Monotone-enforced.

    The adjusted p is `max` over the prefix so the sequence is non-decreasing —
    without that, a later smaller raw p can produce an adjusted p below an
    earlier one, and "reject at α" would stop being a step-down rule.
    """
    order = sorted(range(len(labelled_p)), key=lambda i: labelled_p[i][1])
    m = len(labelled_p)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = labelled_p[idx][1]
        running = max(running, min(1.0, (m - rank) * raw))
        adjusted[idx] = running
    rows = [HolmRow(label=labelled_p[i][0], p_raw=labelled_p[i][1],
                    p_holm=adjusted[i], rejected_at_alpha=adjusted[i] < alpha)
            for i in range(m)]
    # Step-down: once one fails, every later (larger-p) test fails too. The
    # monotone `running` already guarantees this; asserted rather than trusted.
    seen_fail = False
    for idx in order:
        if not rows[idx].rejected_at_alpha:
            seen_fail = True
        elif seen_fail:                                    # pragma: no cover
            raise ReadsError("Holm produced a non-step-down rejection set")
    return rows


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks with ties — the Spearman convention, spelled out."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ as Pearson on average ranks (ties handled)."""
    if len(x) != len(y):
        raise ReadsError(f"spearman: lengths differ ({len(x)} vs {len(y)})")
    if len(x) < 3:
        raise ReadsError(f"spearman over n={len(x)} is not a correlation")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        raise ReadsError("spearman: a constant ordering has no correlation")
    return float((rx * ry).sum() / denom)


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     z: Sequence[float]) -> float:
    """ρ(x, y | z) — Pearson of the residuals of the RANK-transformed columns.

    The standard partial-Spearman: rank each column, then residualize x and y
    on z by least squares and correlate the residuals. §7 asks P1b to "retain
    |ρ| ≥ .4 after partialing out own-leg r²", which is exactly this with
    z = own-leg r².
    """
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.column_stack([np.ones(len(rz)), rz])
    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta
    ex, ey = resid(rx), resid(ry)
    # A control that EXPLAINS a column outright leaves a residual that is pure
    # float noise, and correlating noise with noise returns a number in
    # [-1, 1] that means nothing. Refused rather than returned.
    for residual, centred, label in ((ex, rx - rx.mean(), "x"),
                                     (ey, ry - ry.mean(), "y")):
        scale = float(np.sqrt((centred * centred).sum()))
        if scale == 0.0 or float(np.sqrt((residual * residual).sum())) < 1e-9 * scale:
            raise ReadsError(
                f"partial spearman: the control column explains {label} "
                f"completely, so the partial correlation is undefined — "
                f"refusing rather than correlating float noise")
    denom = float(np.sqrt((ex * ex).sum() * (ey * ey).sum()))
    return float((ex * ey).sum() / denom)


def _named_rng(name: str) -> np.random.Generator:
    """A generator whose seed IS its name — reproducible from the artifact."""
    word = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence(entropy=word))


class MaxStatNull(BaseModel):
    """The multiplicity-honest permutation null over a predictor SLATE.

    ONE null for the whole slate: the statistic permuted against is
    `max_j |ρ_j|`, so a slate member's p already carries the cost of every
    other member having had a chance to be the maximum. §7's clustering rule
    (P2/P6 = ONE cluster, their ρ was −.93) is applied by collapsing that
    cluster to its single best member BEFORE the maximum is taken — otherwise
    a correlated pair would inflate the null exactly as an independent pair.
    """
    n_permutations: int
    seed_name: str
    members: list[str]
    clusters: dict[str, list[str]]
    observed_abs_rho: dict[str, float]
    observed_max: float
    p_max_statistic: dict[str, float]


def max_statistic_null(quality: Sequence[float],
                       predictors: dict[str, Sequence[float]],
                       clusters: dict[str, list[str]],
                       n_perm: int = N_PERMUTATIONS,
                       seed_name: str = PERMUTATION_SEED_NAME) -> MaxStatNull:
    """Permute the QUALITY ordering; recompute every slate member's |ρ|.

    The quality column is permuted (not the predictors) so that all members see
    the same permutation — which is what makes the maximum a joint statistic.
    """
    rng = _named_rng(seed_name)
    q = np.asarray(quality, dtype=np.float64)
    observed = {name: abs(spearman(q, col)) for name, col in predictors.items()}
    # Cluster collapse: each cluster contributes its best member only.
    def cluster_max(vals: dict[str, float]) -> float:
        best = 0.0
        for members in clusters.values():
            present = [vals[m] for m in members if m in vals]
            if present:
                best = max(best, max(present))
        return best
    observed_max = cluster_max(observed)
    ge_counts = {name: 0 for name in predictors}
    for _ in range(n_perm):
        perm = rng.permutation(len(q))
        qp = q[perm]
        vals = {name: abs(spearman(qp, col)) for name, col in predictors.items()}
        null_max = cluster_max(vals)
        for name, obs in observed.items():
            if null_max >= obs:
                ge_counts[name] += 1
    # (r + 1) / (n + 1): the permutation p can never be 0, because 0 would
    # claim a certainty the resampling cannot supply.
    p = {name: (ge_counts[name] + 1) / (n_perm + 1) for name in predictors}
    return MaxStatNull(n_permutations=n_perm, seed_name=seed_name,
                       members=sorted(predictors), clusters=clusters,
                       observed_abs_rho=observed, observed_max=observed_max,
                       p_max_statistic=p)


# ══════════════════════════════════════════════════════════ §7 tier membership
class TierMembership(BaseModel):
    """§7 clause 3 in ONE context, verbatim: BOTH conditions, or no expulsion."""

    context: str
    binding_eligible: bool
    leader: str
    leader_median: float
    n_basis: int
    medians: dict[str, float]
    ranking: list[str]
    sign_tests: list[SignTest]
    holm: list[HolmRow]
    median_excess: dict[str, float]
    expelled: list[str]
    top_tier: list[str]
    practically_equivalent: list[str] = Field(
        default_factory=list,
        description="§7 clause 3's quoted phrase 'statistically "
                    "distinguishable, practically equivalent' — significant "
                    "under Holm but excess < delta, so NOT expelled")
    notes: list[str] = Field(default_factory=list)


def tier_membership(scores: dict[str, HubContextScore], ctx: Context,
                    candidates: Sequence[str] = CANDIDATES) -> TierMembership:
    """§7 clause 3 over the candidate set, on the shared basis of this context."""
    medians = {h: scores[h].median_abs_error for h in candidates}
    ranking = sorted(candidates, key=lambda h: (medians[h], h))
    leader = ranking[0]
    leader_by = scores[leader].by_ordinal

    tests: list[SignTest] = []
    for hub in ranking[1:]:
        other = scores[hub].by_ordinal
        shared = sorted(set(leader_by) & set(other))
        tests.append(sign_test(
            f"{leader} vs {hub} @ {ctx.key}",
            [leader_by[o].abs_error for o in shared],
            [other[o].abs_error for o in shared], leader, hub))
    rows = holm([(t.label, t.p_two_sided) for t in tests], ALPHA_TIER)

    excess = {h: medians[h] - medians[leader] for h in candidates}
    expelled: list[str] = []
    prac_equiv: list[str] = []
    notes: list[str] = []
    for hub, test, row in zip(ranking[1:], tests, rows):
        loses = row.rejected_at_alpha and test.winner == leader
        practical = excess[hub] >= DELTA_PRACTICAL_EQUIVALENCE
        if loses and practical:
            expelled.append(hub)
        elif loses and not practical:
            prac_equiv.append(hub)
            notes.append(
                f"{hub}: statistically distinguishable, practically equivalent "
                f"(Holm p={row.p_holm:.3g} < {ALPHA_TIER}, median excess "
                f"{excess[hub]:.5f} < delta {DELTA_PRACTICAL_EQUIVALENCE})")
        elif row.rejected_at_alpha and test.winner == hub:  # pragma: no cover
            notes.append(f"{hub}: significant AGAINST the leader — the leader "
                         f"is defined by the smallest median, so this is a "
                         f"contradiction the desk must see")
    top_tier = [h for h in ranking if h not in expelled]
    return TierMembership(
        context=ctx.key, binding_eligible=ctx.binding_eligible, leader=leader,
        leader_median=medians[leader], n_basis=scores[leader].n_slots_realized,
        medians=medians, ranking=ranking, sign_tests=tests, holm=rows,
        median_excess=excess, expelled=expelled, top_tier=top_tier,
        practically_equivalent=prac_equiv, notes=notes)


# ══════════════════════════════════════════════════ §6 clauses over the grid
class GroupingVerdict(BaseModel):
    """§7 clause 4's eligibility under ONE reading of 'the four contexts'."""

    grouping: str
    grouping_contexts: list[str]
    rationale: str
    expelled_by_context: dict[str, list[str]]
    expulsion_counts: dict[str, int]
    ineligible: list[str]
    eligible: list[str]


class InvarianceVerdicts(BaseModel):
    """§6-I1/I2 binding clauses + §7 clause 4, under BOTH defensible groupings."""

    contexts_computed: list[str]
    binding_contexts: list[str]
    continuity_contexts: list[str]
    tiers: dict[str, TierMembership]
    i1_halves_agree: bool
    i1_detail: dict[str, Any]
    i2_ranks_agree: bool
    i2_detail: dict[str, Any]
    groupings: list[GroupingVerdict]
    groupings_agree: bool
    grouping_disagreement: list[str] = Field(default_factory=list)


#: The two defensible readings of §7 clause 4's "the four contexts". The frozen
#: text says "top-tier in both halves and at both ranks (§6 I1/I2), with the
#: noise-ejection guard: ... expelled in the SAME direction in >= 2 of the four
#: contexts" — and "both halves x both ranks" is four cells, while
#: "{half-a, half-b} + {full@k128, full@k256}" is also four. Both are computed;
#: a disagreement is FLAGGED and no verdict is issued on the clause.
GROUPINGS: dict[str, tuple[str, tuple[tuple[Split, str], ...]]] = {
    "half-x-rank-2x2": (
        "the 2x2 half x rank grid — 'top-tier in both halves AND at both "
        "ranks' read as the product of the two invariance legs, which is the "
        "only reading under which every context is itself a half (I1) and a "
        "rank (I2) simultaneously",
        (("halfa", "proc_k128"), ("halfa", "proc_k256"),
         ("halfb", "proc_k128"), ("halfb", "proc_k256"))),
    "halves-plus-full-ranks": (
        "{half-a, half-b} at the rank of record + {full@k128, full@k256} — "
        "'both halves' contributes two contexts (I1) and 'both ranks' "
        "contributes two more on the full corpus (I2), which reads the two "
        "legs as separate pairs rather than as a product",
        (("halfa", RANK_OF_RECORD), ("halfb", RANK_OF_RECORD),
         ("full", "proc_k128"), ("full", "proc_k256"))),
}


def evaluate_invariance(tiers: dict[str, TierMembership],
                        candidates: Sequence[str] = CANDIDATES
                        ) -> InvarianceVerdicts:
    """§6-I1, §6-I2 and §7 clause 4 from the computed grid."""
    binding = [k for k, t in tiers.items() if t.binding_eligible]
    continuity = [k for k, t in tiers.items() if not t.binding_eligible]

    # I1: every §7 eligibility decision agrees ACROSS HALVES. The decision at
    # each rank is compared half-a vs half-b; the full corpus is not a half.
    i1_detail: dict[str, Any] = {}
    i1_ok = True
    for rank in BINDING_RANKS:
        a = tiers[f"halfa::{rank}"]
        b = tiers[f"halfb::{rank}"]
        agree = sorted(a.top_tier) == sorted(b.top_tier)
        i1_ok = i1_ok and agree
        i1_detail[rank] = {
            "halfa_top_tier": sorted(a.top_tier),
            "halfb_top_tier": sorted(b.top_tier),
            "halfa_expelled": sorted(a.expelled),
            "halfb_expelled": sorted(b.expelled),
            "agree": agree,
            "disagreeing_candidates": sorted(
                set(a.top_tier) ^ set(b.top_tier))}

    # I2: eligibility agreement ACROSS RANKS, within each split.
    i2_detail: dict[str, Any] = {}
    i2_ok = True
    for split in SPLITS:
        a = tiers[f"{split}::proc_k128"]
        b = tiers[f"{split}::proc_k256"]
        agree = sorted(a.top_tier) == sorted(b.top_tier)
        i2_ok = i2_ok and agree
        i2_detail[split] = {
            "k128_top_tier": sorted(a.top_tier),
            "k256_top_tier": sorted(b.top_tier),
            "agree": agree,
            "disagreeing_candidates": sorted(set(a.top_tier) ^ set(b.top_tier))}

    groupings: list[GroupingVerdict] = []
    for name, (rationale, cells) in GROUPINGS.items():
        keys = [f"{s}::{f}" for s, f in cells]
        expelled_by = {k: sorted(tiers[k].expelled) for k in keys}
        counts = {h: sum(1 for k in keys if h in expelled_by[k])
                  for h in candidates}
        ineligible = sorted(h for h in candidates if counts[h] >= 2)
        groupings.append(GroupingVerdict(
            grouping=name, grouping_contexts=keys, rationale=rationale,
            expelled_by_context=expelled_by, expulsion_counts=counts,
            ineligible=ineligible,
            eligible=sorted(h for h in candidates if h not in ineligible)))

    eligible_sets = {tuple(g.eligible) for g in groupings}
    agree = len(eligible_sets) == 1
    disagreement: list[str] = []
    if not agree:
        for g in groupings:
            disagreement.append(f"{g.grouping}: eligible={g.eligible}")
    return InvarianceVerdicts(
        contexts_computed=sorted(tiers), binding_contexts=sorted(binding),
        continuity_contexts=sorted(continuity), tiers=tiers,
        i1_halves_agree=i1_ok, i1_detail=i1_detail,
        i2_ranks_agree=i2_ok, i2_detail=i2_detail,
        groupings=groupings, groupings_agree=agree,
        grouping_disagreement=disagreement)


# ═══════════════════════════════════════════════════ §6-I1/I2 order besides
class OrderBeside(BaseModel):
    """§6's full-order Spearman beside, with its PRE-NAMED partial wording."""

    label: str
    population: list[str]
    n_hubs: int
    rho: float
    passes_pre_stated: bool
    wording: str


PARTIAL_WORDING = ("within-tier order is noise; tiers are invariant")


def order_beside(label: str, population: Sequence[str],
                 left: dict[str, float], right: dict[str, float],
                 binding_passes: bool) -> OrderBeside:
    """One full-order ρ, with the frozen partial-outcome sentence when it fires."""
    pop = list(population)
    rho = spearman([left[h] for h in pop], [right[h] for h in pop])
    ok = rho >= RHO_ORDER_PASS
    if ok:
        wording = (f"full-order Spearman rho = {rho:.4f} >= {RHO_ORDER_PASS} — "
                   f"the pre-stated pass, over {len(pop)} guard-quotable hubs")
    elif binding_passes:
        wording = (f"PRE-NAMED PARTIAL OUTCOME: rho = {rho:.4f} < "
                   f"{RHO_ORDER_PASS} with the binding clause passing, quoted "
                   f"as “{PARTIAL_WORDING}”")
    else:
        wording = (f"rho = {rho:.4f} < {RHO_ORDER_PASS} AND the binding clause "
                   f"does not pass — the pre-named partial outcome does NOT "
                   f"apply (it is conditioned on the binding clause passing)")
    return OrderBeside(label=label, population=pop, n_hubs=len(pop), rho=rho,
                       passes_pre_stated=ok, wording=wording)


# ══════════════════════════════════════════════ §8 step 5 all-roster race
class RosterRaceRow(BaseModel):
    hub: str
    hub_site: int
    role: str
    checkpoint_identity: str
    leg_source: str
    n_slots_realized: int
    arm_counts: dict[str, int]
    median_abs_error: float
    median_rank: int
    condorcet_wins: int
    condorcet_rank: int


class RosterRace(BaseModel):
    """§7 clause 1 / §8 step 5: EVERY core hub raced, gateless and descriptive."""

    context: str
    population: list[str]
    n_hubs: int
    rows: list[RosterRaceRow]
    median_ranking: list[str]
    condorcet_ranking: list[str]
    rank_agreement_rho: float
    pairwise_tests: list[SignTest]
    i3_statement: str
    slot_composition_note: str


I3_ORDERING_INSTABILITY = (
    "the named ordering-instability result: median and Condorcet hub rankings "
    "DISAGREE, refuting full-order (not tier) invariance")


def roster_race(scores: dict[str, HubContextScore], ctx: Context,
                roles: dict[str, str], identities: dict[str, str]
                ) -> RosterRace:
    """Median-score ranking vs Condorcet win-count ranking over the population.

    Every pairwise sign test runs on the two hubs' OWN shared slots, which is
    what makes the Condorcet column immune to the slot-composition confound the
    median column carries (the 17-hub census's finding, reused as a method).
    """
    pop = sorted(scores)
    by_ordinal = {h: scores[h].by_ordinal for h in pop}
    tests: list[SignTest] = []
    wins = {h: 0 for h in pop}
    for i, a in enumerate(pop):
        for b in pop[i + 1:]:
            shared = sorted(set(by_ordinal[a]) & set(by_ordinal[b]))
            if not shared:
                continue
            t = sign_test(f"{a} vs {b} @ {ctx.key}",
                          [by_ordinal[a][o].abs_error for o in shared],
                          [by_ordinal[b][o].abs_error for o in shared], a, b)
            tests.append(t)
            if t.n_a_better > t.n_b_better:
                wins[a] += 1
            elif t.n_b_better > t.n_a_better:
                wins[b] += 1

    median_ranking = sorted(pop, key=lambda h: (scores[h].median_abs_error, h))
    condorcet_ranking = sorted(pop, key=lambda h: (-wins[h], h))
    med_rank = {h: i + 1 for i, h in enumerate(median_ranking)}
    con_rank = {h: i + 1 for i, h in enumerate(condorcet_ranking)}
    rho = spearman([float(med_rank[h]) for h in pop],
                   [float(con_rank[h]) for h in pop])
    agree = median_ranking == condorcet_ranking
    statement = (
        f"median and Condorcet rankings AGREE exactly over {len(pop)} hubs "
        f"(Spearman rho = {rho:.4f})" if agree else
        f"{I3_ORDERING_INSTABILITY} (Spearman rho between the two rankings = "
        f"{rho:.4f}; {sum(1 for h in pop if med_rank[h] != con_rank[h])} of "
        f"{len(pop)} hubs change rank)")

    rows = [RosterRaceRow(
        hub=h, hub_site=scores[h].hub_site, role=roles.get(h, "core"),
        checkpoint_identity=identities.get(h, "instruct"),
        leg_source=scores[h].leg_source,
        n_slots_realized=scores[h].n_slots_realized,
        arm_counts=scores[h].arm_counts,
        median_abs_error=scores[h].median_abs_error,
        median_rank=med_rank[h], condorcet_wins=wins[h],
        condorcet_rank=con_rank[h]) for h in median_ranking]
    composition = (
        "SLOT-COMPOSITION CAVEAT, quoted with the median column: a hub cannot "
        "quote a slot it is an endpoint of, and a non-race hub has legs only "
        "in the arms its own pairs were banked in, so the median column is "
        "taken over DIFFERENT slot sets per hub (each row states its own n and "
        "arm split). The Condorcet column is not exposed to this: every "
        "pairwise test runs on the two hubs' shared slots only.")
    return RosterRace(context=ctx.key, population=pop, n_hubs=len(pop),
                      rows=rows, median_ranking=median_ranking,
                      condorcet_ranking=condorcet_ranking,
                      rank_agreement_rho=rho, pairwise_tests=tests,
                      i3_statement=statement,
                      slot_composition_note=composition)


class SeparationFloor(BaseModel):
    """§6-I4 (binding): every top-quartile x bottom-quartile pair, p < .01 Holm."""

    context: str
    population: list[str]
    n_hubs: int
    quartile_rule: str
    top_quartile: list[str]
    bottom_quartile: list[str]
    n_pairs: int
    tests: list[SignTest]
    holm: list[HolmRow]
    all_below_alpha: bool
    alpha: float = ALPHA_SEPARATION
    failures: list[str] = Field(default_factory=list)


def separation_floor(scores: dict[str, HubContextScore], ctx: Context,
                     ranking: Sequence[str]) -> SeparationFloor:
    """§6-I4 over the guard-quotable population at the family of record."""
    n = len(ranking)
    # Quartile rule, stated because n is not a multiple of 4: ceil(n/4) hubs at
    # each end. Ceil rather than floor so that neither quartile is empty at
    # small n, and so the two quartiles are the same size by construction.
    q = -(-n // 4)
    top = list(ranking[:q])
    bottom = list(ranking[-q:])
    if set(top) & set(bottom):
        raise ReadsError(
            f"{ctx.key}: the top and bottom quartiles overlap at n={n} "
            f"(q={q}) — I4's 'every top-quartile x bottom-quartile pair' is "
            f"not well defined here")
    by_ordinal = {h: scores[h].by_ordinal for h in ranking}
    tests: list[SignTest] = []
    for a in top:
        for b in bottom:
            shared = sorted(set(by_ordinal[a]) & set(by_ordinal[b]))
            tests.append(sign_test(
                f"{a} vs {b} @ {ctx.key}",
                [by_ordinal[a][o].abs_error for o in shared],
                [by_ordinal[b][o].abs_error for o in shared], a, b))
    rows = holm([(t.label, t.p_two_sided) for t in tests], ALPHA_SEPARATION)
    failures = [r.label for r in rows if not r.rejected_at_alpha]
    return SeparationFloor(
        context=ctx.key, population=list(ranking), n_hubs=n,
        quartile_rule=(f"ceil(n/4) = {q} hubs at each end of the median "
                       f"ranking over n={n}; ceil so the two quartiles are the "
                       f"same size and neither is empty"),
        top_quartile=top, bottom_quartile=bottom, n_pairs=len(tests),
        tests=tests, holm=rows, all_below_alpha=not failures, failures=failures)



# ══════════════════════════════ the second §6 ambiguity: what "agrees" means
class ClauseReading(BaseModel):
    """ONE reading of §6-I1/I2's binding clause, computed and labelled.

    §6-I1 says "every §7 eligibility decision agrees across halves" and §7
    clause 4 says eligibility is "top-tier in both halves and at both ranks
    ... with the noise-ejection guard: a candidate is ineligible only if it is
    expelled in the SAME direction in >= 2 of the four contexts". Those two
    sentences support two readings and they can disagree:

      STRICT   — "eligibility decision" = TOP-TIER MEMBERSHIP in a context, so
                 the per-context top-tier sets must be IDENTICAL across halves.
                 Textually the most direct, but under it §7's noise-ejection
                 guard is dead letter: the guard exists precisely to tolerate a
                 single-context expulsion, and any such expulsion makes the
                 STRICT clause fail, so the guard could never fire without
                 I1 already having failed.
      GUARDED  — "eligibility decision" = the §7 clause-4 ELIGIBILITY VERDICT,
                 computed WITHIN each half (over that half's two ranks) and
                 within each rank (over the two halves), carrying the same
                 >= 2-expulsions guard down to the 2-context group. Under this
                 reading the guard has work to do and I1/I2 ask whether the
                 guarded verdict is stable.

    BOTH ARE COMPUTED AND BOTH ARE REPORTED. This is a SECOND ambiguity beyond
    the two context-groupings the brief names, and it is FLAGGED rather than
    resolved.
    """

    reading: Literal["STRICT", "GUARDED"]
    clause: Literal["I1", "I2"]
    definition: str
    groups: dict[str, list[str]]
    verdicts_by_group: dict[str, list[str]]
    agree: bool
    disagreeing_candidates: list[str]


def _guarded_eligibility(tiers: dict[str, TierMembership], keys: Sequence[str],
                         candidates: Sequence[str]) -> list[str]:
    """§7 clause 4's guard applied to a group of contexts: >= 2 expulsions."""
    counts = {h: sum(1 for k in keys if h in tiers[k].expelled)
              for h in candidates}
    return sorted(h for h in candidates if counts[h] < 2)


def clause_readings(tiers: dict[str, TierMembership],
                    candidates: Sequence[str] = CANDIDATES
                    ) -> list[ClauseReading]:
    """I1 and I2, each under both readings."""
    out: list[ClauseReading] = []

    # ---- I1: across halves --------------------------------------------------
    strict_groups = {half: [f"{half}::{r}" for r in BINDING_RANKS]
                     for half in ("halfa", "halfb")}
    strict_verdicts = {
        f"{half}::{rank}": sorted(tiers[f"{half}::{rank}"].top_tier)
        for half in ("halfa", "halfb") for rank in BINDING_RANKS}
    per_rank_agree = all(
        strict_verdicts[f"halfa::{rank}"] == strict_verdicts[f"halfb::{rank}"]
        for rank in BINDING_RANKS)
    disagreeing = sorted({
        c for rank in BINDING_RANKS
        for c in set(strict_verdicts[f"halfa::{rank}"])
        ^ set(strict_verdicts[f"halfb::{rank}"])})
    out.append(ClauseReading(
        reading="STRICT", clause="I1",
        definition=("per-context TOP-TIER MEMBERSHIP must be identical on "
                    "half-a and half-b, at each binding rank separately"),
        groups={k: v for k, v in strict_groups.items()},
        verdicts_by_group=strict_verdicts, agree=per_rank_agree,
        disagreeing_candidates=disagreeing))

    guarded = {half: _guarded_eligibility(tiers, strict_groups[half], candidates)
               for half in ("halfa", "halfb")}
    out.append(ClauseReading(
        reading="GUARDED", clause="I1",
        definition=("§7 clause 4's ELIGIBILITY VERDICT computed within each "
                    "half over that half's two binding ranks, carrying the "
                    "noise-ejection guard (ineligible only if expelled in >= 2 "
                    "of the group's contexts), must agree across halves"),
        groups=strict_groups, verdicts_by_group=guarded,
        agree=guarded["halfa"] == guarded["halfb"],
        disagreeing_candidates=sorted(
            set(guarded["halfa"]) ^ set(guarded["halfb"]))))

    # ---- I2: across ranks ---------------------------------------------------
    rank_groups = {rank: [f"{s}::{rank}" for s in SPLITS]
                   for rank in BINDING_RANKS}
    strict_by_rank = {f"{split}::{rank}": sorted(tiers[f"{split}::{rank}"].top_tier)
                      for split in SPLITS for rank in BINDING_RANKS}
    per_split_agree = all(
        strict_by_rank[f"{split}::proc_k128"] == strict_by_rank[f"{split}::proc_k256"]
        for split in SPLITS)
    disagreeing2 = sorted({
        c for split in SPLITS
        for c in set(strict_by_rank[f"{split}::proc_k128"])
        ^ set(strict_by_rank[f"{split}::proc_k256"])})
    out.append(ClauseReading(
        reading="STRICT", clause="I2",
        definition=("per-context TOP-TIER MEMBERSHIP must be identical at "
                    "proc_k128 and proc_k256, within each split separately"),
        groups=rank_groups, verdicts_by_group=strict_by_rank,
        agree=per_split_agree, disagreeing_candidates=disagreeing2))

    guarded_rank = {rank: _guarded_eligibility(tiers, rank_groups[rank],
                                               candidates)
                    for rank in BINDING_RANKS}
    out.append(ClauseReading(
        reading="GUARDED", clause="I2",
        definition=("§7 clause 4's ELIGIBILITY VERDICT computed within each "
                    "rank over the three splits, carrying the noise-ejection "
                    "guard, must agree across ranks"),
        groups=rank_groups, verdicts_by_group=guarded_rank,
        agree=(guarded_rank["proc_k128"] == guarded_rank["proc_k256"]),
        disagreeing_candidates=sorted(set(guarded_rank["proc_k128"])
                                      ^ set(guarded_rank["proc_k256"]))))
    return out


# ═════════════════════ the I3/I4 population: "at the family of record"
def filter_scores(scores: dict[str, HubContextScore], hubs: Sequence[str],
                  arm: Optional[str] = None) -> dict[str, HubContextScore]:
    """A population restricted to `hubs`, optionally to ONE arm's slots.

    Re-aggregating rather than re-computing: the per-slot errors are already
    the engine's, so a filtered median is the SAME numbers over a stated subset
    — never a second measurement.
    """
    out: dict[str, HubContextScore] = {}
    for hub in hubs:
        score = scores[hub]
        errors = ([e for e in score.errors if e.arm == arm] if arm
                  else list(score.errors))
        if not errors:
            continue
        abs_errors = np.array([e.abs_error for e in errors], dtype=np.float64)
        out[hub] = HubContextScore(
            hub=score.hub, hub_site=score.hub_site, context=score.context,
            split=score.split, family=score.family,
            leg_source=score.leg_source, n_slots_realized=len(errors),
            n_slots_basis=score.n_slots_basis,
            median_abs_error=float(np.median(abs_errors)),
            mean_abs_error=float(abs_errors.mean()),
            arm_counts=dict(sorted(Counter(e.arm for e in errors).items())),
            hub_basis_max_dev=score.hub_basis_max_dev,
            unrealized=score.unrealized, errors=errors)
    return out


class PopulationRead(BaseModel):
    """§6-I3 + §6-I4 over ONE reading of the I3/I4 population."""

    population_reading: str
    definition: str
    n_hubs: int
    arm_restriction: Optional[str]
    race: RosterRace
    separation_floor: SeparationFloor
    diagnosis: str


#: Below this many effective pairs a two-sided sign test cannot reach p < .01
#: even on a PERFECT split (2 * 2^-n < .01 needs n >= 9), so a failure at or
#: under it is arithmetic about the shared-slot count, not about separation.
UNDERPOWERED_N: int = 9


def _floor_diagnosis(floor: SeparationFloor) -> str:
    """Why I4 landed where it did — separating the two very different causes."""
    if floor.all_below_alpha:
        return (f"All {floor.n_pairs} top-quartile x bottom-quartile pairs "
                f"clear p < {floor.alpha} after Holm.")
    failing = [(t, r) for t, r in zip(floor.tests, floor.holm)
               if not r.rejected_at_alpha]
    ns = sorted(t.n_effective for t, _ in failing)
    holms = sorted(r.p_holm for _, r in failing)
    underpowered = [t for t, _ in failing if t.n_effective <= UNDERPOWERED_N]
    head = (f"{len(failing)} of {floor.n_pairs} pairs do not clear "
            f"p < {floor.alpha} after Holm. Shared-slot counts across the "
            f"failures run {ns[0]}–{ns[-1]} (median "
            f"{int(np.median(ns))}); Holm-adjusted p runs "
            f"{holms[0]:.3g}–{holms[-1]:.3g}.")
    if underpowered:
        return head + (
            f" {len(underpowered)} of the failures are STRUCTURALLY "
            f"UNDERPOWERED: their shared set is n <= {UNDERPOWERED_N}, and a "
            f"two-sided sign test cannot reach p < {floor.alpha} at that n "
            f"even on a perfect split (2 x 2^-n alone exceeds it). That is "
            f"arithmetic about the ARM RULE — a base-checkpoint hub quotes "
            f"raw slots only, an instruct hub quotes almost all native, and "
            f"the pair shares only the handful both can reach — not evidence "
            f"about separation. The remaining "
            f"{len(failing) - len(underpowered)} failures are genuine "
            f"non-separations at usable n.")
    return head + (
        f" NONE of the failures is underpowered (every failing pair has "
        f"n > {UNDERPOWERED_N}); they are genuine non-separations, and the "
        f"smallest Holm-adjusted p among them is {holms[0]:.3g} against a "
        f"bar of {floor.alpha} — near-misses rather than collapses.")

# ═══════════════════════════════════════════════════════════ the emitter
def relativize(value: Any, repo: Path) -> Any:
    """Rewrite any absolute path under `repo` to a repo-relative one, in place.

    THE SANITIZATION RULE APPLIED AT THE SOURCE, not left to the desk sweep: an
    absolute desk path carries a username, and an artifact is the wrong place
    for one. Repo-relative is also the identity every other artifact in this
    campaign cites, so this makes the reads artifact quotable beside them.
    """
    root = str(repo)
    if isinstance(value, str):
        return value.replace(root + "/", "").replace(root, ".")
    if isinstance(value, dict):
        return {k: relativize(v, repo) for k, v in value.items()}
    if isinstance(value, list):
        return [relativize(v, repo) for v in value]
    return value


def canonical_json(payload: Any) -> str:
    """Sorted keys, fixed separators, no NaN — the byte sequence is the artifact."""
    return json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False,
                      allow_nan=False)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_deterministic(payload: dict[str, Any], out: Path,
                        sidecar_extra: Optional[dict[str, Any]] = None
                        ) -> tuple[str, Path]:
    """Write the artifact (no timestamps) + a sidecar (every varying fact)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload)
    out.write_text(text, encoding="utf-8")
    digest = sha256_of_text(text)
    sidecar = out.with_suffix(".sidecar.json")
    body: dict[str, Any] = {
        "artifact": out.name,
        "artifact_sha256": digest,
        "emitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": "metabasis/scripts/read_v3_invariance.py",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "thread_config": thread_config_stamp(),
        "attestation": ("UNSTAMPED (C section 8) — this lane computes "
                        "arithmetic and declares nothing of record. Every "
                        "PASS/FAIL inside the artifact is UNSTAMPED; stamps "
                        "are Luxia's at the first-read."),
        "determinism": ("the artifact carries no timestamp and no "
                        "environment-varying value; every such fact is here. "
                        "`--selftest` proves a build-twice byte-identical "
                        "artifact."),
    }
    if sidecar_extra:
        body.update(sidecar_extra)
    sidecar.write_text(canonical_json(body) + "\n", encoding="utf-8")
    return digest, sidecar


# ═══════════════════════════════════════════ §3.4 validity + §4 rank guard
class FitRow(BaseModel):
    """One `cp2_summary.json` record, reduced to what the guards need."""
    model_config = {"frozen": True}

    site_pair: str
    arm: str
    family: str
    n_train: int
    n_test: int
    r2: float
    valid: bool
    k_effective: int
    pca_explained_src: Optional[float] = None
    pca_explained_tgt: Optional[float] = None
    strata_carried: tuple[str, ...] = ()

    @property
    def rank_guard_ok(self) -> bool:
        """§4: k quotable only where k <= n_train / 1.2, for the split USED."""
        return self.k_effective <= self.n_train / RANK_GUARD_DIVISOR

    @property
    def rank_guard_ratio(self) -> float:
        return self.k_effective / self.n_train


_CP2_CACHE: dict[str, dict[str, FitRow]] = {}


def load_cp2(directory: Path) -> dict[str, FitRow]:
    """`cp2_summary.json`'s records, keyed by family. Cached by directory."""
    key = str(directory)
    got = _CP2_CACHE.get(key)
    if got is not None:
        return got
    path = directory/"cp2_summary.json"
    if not path.exists():
        raise ReadsError(f"no cp2_summary.json beside the fits at {directory} — "
                         f"§3.4 validity and §4's rank guard are unreadable")
    doc = json.loads(path.read_text())
    rows: dict[str, FitRow] = {}
    for rec in doc.get("records", []):
        detail = rec.get("detail") or {}
        if "k_effective" not in detail:
            continue                       # the ridge family carries no k
        rows[rec["family"]] = FitRow(
            site_pair=rec["site_pair"], arm=rec["arm"], family=rec["family"],
            n_train=int(rec["n_train"]), n_test=int(rec["n_test"]),
            r2=float(rec["r2"]), valid=bool(rec["valid"]),
            k_effective=int(detail["k_effective"]),
            pca_explained_src=detail.get("pca_explained_src"),
            pca_explained_tgt=detail.get("pca_explained_tgt"),
            strata_carried=tuple(rec.get("strata_carried", [])))
    _CP2_CACHE[key] = rows
    return rows


def _object_dir(banks: Banks, ctx: Context, hub: str, hub_site: int,
                model: str, site: int, arm: str) -> Optional[Path]:
    """The directory holding the fit object used as the leg hub→model."""
    leg = banks.layout.leg_dir(ctx, hub, hub_site, model, site, arm)
    if (leg/"cp2_summary.json").exists():
        return leg
    for a, a_s, b, b_s in ((hub, hub_site, model, site),
                           (model, site, hub, hub_site)):
        cand = banks.layout.pair_dir(ctx, a, a_s, b, b_s, arm)
        if (cand/"cp2_summary.json").exists():
            return cand
    return None


def _pair_object_dir(banks: Banks, ctx: Context, slot: BasisSlot
                     ) -> Optional[Path]:
    for a, a_s, b, b_s in (
            (slot.source_model, slot.source_site, slot.target_model,
             slot.target_site),
            (slot.target_model, slot.target_site, slot.source_model,
             slot.source_site)):
        cand = banks.layout.pair_dir(ctx, a, a_s, b, b_s, slot.arm)
        if (cand/"cp2_summary.json").exists():
            return cand
    return None


class HubGuard(BaseModel):
    """Is this hub GUARD-QUOTABLE in this context? Derived, and shown working.

    §6-I1/I4 say "all guard-quotable hubs at the family of record, enumerated
    at first fit". NO SUCH ENUMERATION WAS EVER RECORDED (checked by value: the
    sealed artifact, the hub-leg wave's meta, the pair wave's meta, the halves
    manifest and the ledger carry none). It is therefore DERIVED here from the
    two guards the prereg does define, and FLAGGED for ratification:

      §4   rank guard    — k_effective <= n_train / 1.2 on every object used
      §3.4 fit validity  — `valid` on every object used, which IS the
                           both-nulls rule AND the >= 2-of-4 stratum carry, as
                           computed by the fit lane of record

    A hub is guard-quotable iff every fit object its race score consumes (both
    legs of every realized slot, and that slot's direct pair fit) clears both.
    """

    hub: str
    context: str
    n_objects_checked: int
    n_rank_guard_violations: int
    n_invalid_fits: int
    max_rank_guard_ratio: float
    min_r2: float
    quotable: bool
    violations: list[str] = Field(default_factory=list)


def hub_guard(banks: Banks, basis: Sequence[BasisSlot], ctx: Context,
              hub: str, hub_site: int, score: HubContextScore) -> HubGuard:
    """Both guards over EVERY fit object `score` actually consumed."""
    dirs: set[Path] = set()
    quoted = {e.ordinal for e in score.errors}
    by_ordinal = {s.ordinal: s for s in basis}
    for ordinal in sorted(quoted):
        slot = by_ordinal[ordinal]
        for model, site in ((slot.source_model, slot.source_site),
                            (slot.target_model, slot.target_site)):
            found = _object_dir(banks, ctx, hub, hub_site, model, site, slot.arm)
            if found is not None:
                dirs.add(found)
        pair = _pair_object_dir(banks, ctx, slot)
        if pair is not None:
            dirs.add(pair)

    violations: list[str] = []
    n_rank = n_invalid = 0
    max_ratio = 0.0
    min_r2 = float("inf")
    for directory in sorted(dirs):
        row = load_cp2(directory).get(ctx.family)
        if row is None:
            violations.append(f"{directory.name}: no {ctx.family} record")
            n_invalid += 1
            continue
        max_ratio = max(max_ratio, row.rank_guard_ratio)
        min_r2 = min(min_r2, row.r2)
        if not row.rank_guard_ok:
            n_rank += 1
            violations.append(
                f"{directory.name}: k_eff {row.k_effective} > n_train "
                f"{row.n_train} / {RANK_GUARD_DIVISOR}")
        if not row.valid:
            n_invalid += 1
            violations.append(f"{directory.name}: fit INVALID (§3.4)")
    return HubGuard(
        hub=hub, context=ctx.key, n_objects_checked=len(dirs),
        n_rank_guard_violations=n_rank, n_invalid_fits=n_invalid,
        max_rank_guard_ratio=max_ratio,
        min_r2=(0.0 if min_r2 == float("inf") else min_r2),
        quotable=(n_rank == 0 and n_invalid == 0), violations=violations[:40])
    # (`violations` names DIRECTORY BASENAMES only, never a resolved path — see
    # the `.name` uses above — so it carries no root to sanitize.)


# ═══════════════════════════════════════════════════ the HL-2 predictor slate
#: The operational definitions of record. §7 NAMES the slate (P1a/P1b/P2/P6)
#: but defines no member; the PARENT contract (`freeze/transport-campaign`)
#: does not mention P1a/P1b/P2/P6 AT ALL — checked by value at the tag. The
#: only operational definitions in the campaign are the 17-hub census
#: pre-statement's §3, implemented in
#: `staging/s2-hub-census/glue/census_predictors.py`, which this module reuses
#: rather than re-inventing.
SLATE_DEFINITION_SOURCE = (
    "docs/planning/PRESTATEMENT-s2-hub-census-2026-07-31.md §3, implemented in "
    "staging/s2-hub-census/glue/census_predictors.py — NOT the parent "
    "contract, which does not mention the slate at all")
SLATE_DEFINITIONS: dict[str, str] = {
    "P1a": ("estimability-centrality: mean fit r-squared of H's direct maps "
            "to/from the roster. FIT-DERIVED, so §7 requires the "
            "family-excluded form on a DISJOINT fit split."),
    "P1b": ("geometric centrality: mean over M != H of the mean CCA "
            "correlation between H's and M's PCA score matrices on the TRAIN "
            "rows. §7 additionally requires a CHANCE-CORRECTED form."),
    "P2": ("compressibility: pca_explained at the rank, from the model's own "
           "train rows. BANK-ONLY."),
    "P6": ("participation ratio of the state covariance spectrum at the site, "
           "TRAIN ROWS. BANK-ONLY."),
}


class PredictorColumn(BaseModel):
    """One slate member, computed or explicitly ABSENT with its blocker."""

    name: str
    definition: str
    definition_source: str
    available: bool
    blocker: Optional[str] = None
    values: dict[str, float] = Field(default_factory=dict)
    provenance: str = ""
    circularity_guard: str = ""


def _leg_r2_values(banks: Banks, basis: Sequence[BasisSlot], ctx: Context,
                   hub: str, hub_site: int, score: HubContextScore,
                   family_of: dict[str, str], arm: str,
                   exclude_own_family: bool) -> dict[str, float]:
    quoted = {e.ordinal for e in score.errors}
    by_ordinal = {s.ordinal: s for s in basis}
    seen: dict[str, float] = {}
    for ordinal in sorted(quoted):
        slot = by_ordinal[ordinal]
        if slot.arm != arm:
            continue
        for model, site in ((slot.source_model, slot.source_site),
                            (slot.target_model, slot.target_site)):
            if exclude_own_family and family_of.get(model) == family_of.get(hub):
                continue
            directory = _object_dir(banks, ctx, hub, hub_site, model, site,
                                    slot.arm)
            if directory is None:
                continue
            row = load_cp2(directory).get(ctx.family)
            if row is not None:
                seen[str(directory)] = row.r2
    return seen


#: THE ARM RULE, APPLIED TO MODEL-INTRINSIC PREDICTORS (prereg §3.2 + §1's
#: arm-consistency rule: "base models run raw ONLY — their constants live in
#: the raw system, never native"). Both P1a and P2 are per-model constants, so
#: each is read in the model's OWN registered arm: native for an instruct
#: checkpoint, raw for a base one. This is not a compromise between the arms —
#: it is the same rule every other per-model constant in this campaign obeys,
#: and it is what makes a base hub's column comparable at all.
def _registered_arm(identities: dict[str, str], hub: str) -> str:
    return "native" if identities.get(hub, "instruct") == "instruct" else "raw"


def predictor_p1a(banks: Banks, basis: Sequence[BasisSlot], ctx: Context,
                  hubs: dict[str, int], scores: dict[str, HubContextScore],
                  family_of: dict[str, str], identities: dict[str, str],
                  exclude_own_family: bool) -> dict[str, float]:
    """P1a: mean r² over the hub's legs in the hub's OWN registered arm.

    `exclude_own_family=True` is §7's circularity cut: legs to models of the
    hub's OWN family are dropped before the mean, so a family that is merely
    numerous cannot manufacture centrality for its members.
    """
    out: dict[str, float] = {}
    for hub, hub_site in hubs.items():
        arm = _registered_arm(identities, hub)
        seen = _leg_r2_values(banks, basis, ctx, hub, hub_site, scores[hub],
                              family_of, arm, exclude_own_family)
        if not seen:
            raise ReadsError(
                f"P1a for {hub} @ {ctx.key} [{arm}]: no leg survives the "
                f"{'family-excluded ' if exclude_own_family else ''}filter")
        out[hub] = float(np.mean(sorted(seen.values())))
    return out


def predictor_p2(banks: Banks, basis: Sequence[BasisSlot], ctx: Context,
                 hubs: dict[str, int], scores: dict[str, HubContextScore],
                 identities: dict[str, str]) -> dict[str, float]:
    """P2: the hub's OWN pca_explained at the rank, in its registered arm.

    Model-AND-ARM-intrinsic: the PCA is fit on the hub's own train rows IN ONE
    ARM's state bank, so it is identical across every object of that hub in
    that arm and DIFFERENT across arms (measured: the two arms disagree by up
    to 4.2e-3, far beyond the records' 4-dp rounding — which is why this is
    read per arm rather than pooled). The within-arm constancy is ASSERTED, not
    assumed: a spread inside one arm would mean the objects were not fit
    against one PCA bank.
    """
    out: dict[str, float] = {}
    by_ordinal = {s.ordinal: s for s in basis}
    for hub, hub_site in hubs.items():
        arm = _registered_arm(identities, hub)
        quoted = {e.ordinal for e in scores[hub].errors}
        values: list[float] = []
        for ordinal in sorted(quoted):
            slot = by_ordinal[ordinal]
            if slot.arm != arm:
                continue
            for model, site in ((slot.source_model, slot.source_site),
                                (slot.target_model, slot.target_site)):
                leg = banks.layout.leg_dir(ctx, hub, hub_site, model, site,
                                           slot.arm)
                if (leg/"cp2_summary.json").exists():
                    row = load_cp2(leg).get(ctx.family)
                    if row is not None and row.pca_explained_src is not None:
                        values.append(float(row.pca_explained_src))
                    continue
                fwd = banks.layout.pair_dir(ctx, hub, hub_site, model, site,
                                            slot.arm)
                rev = banks.layout.pair_dir(ctx, model, site, hub, hub_site,
                                            slot.arm)
                if (fwd/"cp2_summary.json").exists():
                    row = load_cp2(fwd).get(ctx.family)
                    if row is not None and row.pca_explained_src is not None:
                        values.append(float(row.pca_explained_src))
                elif (rev/"cp2_summary.json").exists():
                    row = load_cp2(rev).get(ctx.family)
                    if row is not None and row.pca_explained_tgt is not None:
                        values.append(float(row.pca_explained_tgt))
        if not values:
            raise ReadsError(
                f"P2 for {hub} @ {ctx.key} [{arm}]: no pca_explained filed in "
                f"the hub's registered arm")
        spread = max(values) - min(values)
        if spread > 5e-4:
            raise ReadsError(
                f"P2 for {hub} @ {ctx.key} [{arm}]: pca_explained is "
                f"model-and-arm-intrinsic but its filed values span "
                f"{spread:.2e} WITHIN one arm across {len(values)} objects — "
                f"those objects were not fit against one PCA bank, and "
                f"averaging them would hide that")
        out[hub] = float(np.median(values))
    return out


# ═════════════════════════════════════════════ P6, from the node-built inputs
#: The artifact kind the P6-INPUTS wave emits. Anything else is refused.
P6_ARTIFACT_KIND = "webtext-v3-p6-inputs/v1"

#: The census formula, CHARACTER-FOR-CHARACTER as `census_predictors.py`
#: implements it (`participation_ratio`) and as the P6-INPUTS artifact quotes
#: its own definition. The artifact's `definition` string must CONTAIN this, or
#: the column is refused: a P6 computed under some other formula is a different
#: slate member wearing P6's name.
P6_CENSUS_FORMULA = ("lambda = svd(rows - mean, compute_uv=False)**2 / (n-1); "
                     "PR = (sum lambda)^2 / sum lambda^2")

#: The banked PR is recomputed from the banked eigenvalues and must agree to
#: this. It is a float64 re-sum of the same numbers, so the tolerance is
#: round-off, not physics.
P6_RECOMPUTE_TOL = 1e-9


class P6Inputs(BaseModel):
    """The node-built P6 column, verified against its own banked eigenvalues.

    P6 is BANK-ONLY (the hub's own train-row state spectrum) and needs the
    webtext-v3 per-text state banks, which are node-side. This artifact is the
    node's answer to that blocker: it carries, per hub, the participation ratio
    AND the full eigenvalue spectrum it was computed from — so the desk can
    re-derive the statistic rather than trust it.
    """

    path: str
    sha256: str
    artifact_kind: str
    definition: str
    corpus_manifest_sha256: str
    splits_sha256: str
    n_hubs: int
    n_train_rows: list[int]
    effective_num_threads: Optional[int] = None
    values: dict[str, float]
    sites: dict[str, int]
    arms: dict[str, str]
    recomputed_max_abs_delta: float
    norm_column_invariance_abs_delta: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


def load_p6_inputs(path: Path) -> P6Inputs:
    """Read + VERIFY the P6-INPUTS artifact. Nothing is taken on its word.

    Four checks, each of which would otherwise be an assumption:
      1. the artifact kind is the one this reader understands;
      2. its `definition` carries the census formula character-for-character;
      3. every hub's `participation_ratio` is RE-DERIVED here from that hub's
         own banked eigenvalues — the artifact does not get to assert its own
         statistic;
      4. the eigenvalue count matches the train-row count it claims.
    """
    if not path.exists():
        raise ReadsError(f"P6-INPUTS artifact absent: {path}")
    doc = json.loads(path.read_text())

    kind = doc.get("artifact")
    if kind != P6_ARTIFACT_KIND:
        raise ReadsError(
            f"{path.name}: artifact kind {kind!r} != {P6_ARTIFACT_KIND!r} — "
            f"refusing to read a P6 column out of an artifact this reader does "
            f"not understand")

    definition = str(doc.get("definition", ""))
    if P6_CENSUS_FORMULA not in definition:
        raise ReadsError(
            f"{path.name}: `definition` does not carry the census formula "
            f"character-for-character. Expected to find:\n  {P6_CENSUS_FORMULA}"
            f"\nGot:\n  {definition}\nA P6 computed under another formula is a "
            f"different slate member wearing P6's name; NOT substituted")

    values: dict[str, float] = {}
    sites: dict[str, int] = {}
    arms: dict[str, str] = {}
    worst = 0.0
    for hub in doc.get("hubs", []):
        key = str(hub["key"])
        banked = float(hub["participation_ratio"])
        lam = np.asarray(hub["eigenvalues"], dtype=np.float64)
        if lam.size != int(hub["n_eigenvalues"]):
            raise ReadsError(
                f"{path.name}: {key} banks {lam.size} eigenvalues but claims "
                f"n_eigenvalues={hub['n_eigenvalues']}")
        denominator = float((lam ** 2).sum())
        if denominator <= 0.0:
            raise ReadsError(f"{path.name}: {key} has a degenerate spectrum")
        recomputed = float((lam.sum() ** 2) / denominator)
        delta = abs(recomputed - banked)
        worst = max(worst, delta)
        if delta > P6_RECOMPUTE_TOL:
            raise ReadsError(
                f"{path.name}: {key} participation_ratio {banked!r} does not "
                f"re-derive from its OWN banked eigenvalues (recomputed "
                f"{recomputed!r}, |Δ| {delta:.3e} > {P6_RECOMPUTE_TOL:.0e}) — "
                f"the artifact's statistic and its spectrum disagree")
        values[key] = banked
        sites[key] = int(hub["site"])
        arms[key] = str(hub["arm"])

    if len(values) != int(doc.get("n_hubs", -1)):
        raise ReadsError(
            f"{path.name}: {len(values)} hub rows but n_hubs="
            f"{doc.get('n_hubs')}")

    invariance = doc.get("norm_column_invariance") or {}
    return P6Inputs(
        path=str(path), sha256=sha256_of(path), artifact_kind=kind,
        definition=definition,
        corpus_manifest_sha256=str(doc.get("corpus_manifest_sha256", "")),
        splits_sha256=str(doc.get("splits_sha256", "")),
        n_hubs=len(values),
        n_train_rows=[int(n) for n in doc.get("n_train_rows", [])],
        effective_num_threads=doc.get("effective_num_threads"),
        values=values, sites=sites, arms=arms,
        recomputed_max_abs_delta=worst,
        norm_column_invariance_abs_delta=invariance.get("abs_delta"),
        notes=[str(doc.get("p1b_status", ""))] if doc.get("p1b_status") else [])


#: §7's positive claim wording of record, VERBATIM. Quoted exactly, or not at
#: all — "No other positive phrasing is quotable."
HL2_POSITIVE_TEMPLATE = ("hub quality on webtext-v3 is predicted by "
                         "{predictor} at |ρ| = {rho}; all other slate "
                         "members quoted beside.")


class Battery(BaseModel):
    """§7's HL-2 predictor battery, strictly under the frozen support rule."""

    quality_context: str
    quality_source: str
    population: list[str]
    n_hubs: int
    quality: dict[str, float]
    quality_by_half: dict[str, dict[str, float]]
    columns: list[PredictorColumn]
    rho_full_ordering: dict[str, float]
    rho_by_half: dict[str, dict[str, float]]
    partial_rho_after_own_leg_r2: dict[str, Optional[float]]
    own_leg_r2: dict[str, float]
    max_stat: Optional[MaxStatNull]
    support_rule: str
    per_member_support: dict[str, dict[str, Any]]
    supported: bool
    supporting_members: list[str]
    positive_claim_wording: Optional[str]
    verdict_text: str
    skipped: list[dict[str, str]]
    flags: list[str]
    #: The node-built P6 column's provenance, or None when P6 stayed skipped.
    p6_inputs: Optional[P6Inputs] = None
    #: §7's clustering rule as APPLIED, member by member.
    multiplicity_clusters: dict[str, list[str]] = Field(default_factory=dict)


def build_battery(banks: Banks, basis: Sequence[BasisSlot],
                  population: Sequence[str], hub_sites: dict[str, int],
                  scores_by_ctx: dict[str, dict[str, HubContextScore]],
                  family_of: dict[str, str], identities: dict[str, str],
                  flags: list[str],
                  p6: Optional[P6Inputs] = None) -> Battery:
    """§7's HL-2 battery — computed ONLY after the race lands, guards binding.

    THE DISJOINT-SPLIT READING, stated because §7's clause is unsatisfiable on
    its face. §7 asks for "family-excluded P1a computed on a disjoint fit split
    (legs fit on the split half not used for the hub-quality measurement)". The
    hub-quality measurement of record is the FULL corpus, and neither half is
    disjoint from the full corpus — every half row is a full-corpus row. The
    only reading under which "disjoint" is true is therefore PER-HALF quality:
    quality on half-a is scored against P1a fit on half-b, and vice versa. Both
    assignments are computed and both are reported; the desk rules.

    "Full-ordering scoring only — no sub-setting, EVER" is read as the
    challenge set words it ("full-ordering scoring only, no sub-setting"): the
    ordering over the WHOLE guard-quotable hub population, never a top-k slice.
    It is not a statement about the corpus split.
    """
    pop = list(population)
    ctx_record = Context(split="full", family=RANK_OF_RECORD)
    quality = {h: scores_by_ctx[ctx_record.key][h].median_abs_error for h in pop}
    quality_halves = {
        half: {h: scores_by_ctx[f"{half}::{RANK_OF_RECORD}"][h].median_abs_error
               for h in pop}
        for half in ("halfa", "halfb")}

    columns: list[PredictorColumn] = []
    skipped: list[dict[str, str]] = []

    # ---- P1a, family-excluded, on the disjoint half ------------------------
    p1a_halves = {
        half: predictor_p1a(banks, basis,
                            Context(split=half, family=RANK_OF_RECORD),
                            {h: hub_sites[h] for h in pop},
                            scores_by_ctx[f"{half}::{RANK_OF_RECORD}"],
                            family_of, identities, exclude_own_family=True)
        for half in ("halfa", "halfb")}
    # The headline column pairs each half's QUALITY with the OTHER half's P1a;
    # reported as the mean of nothing — there is no averaging. The full-ordering
    # headline is the assignment quality(half-a) x P1a(half-b), stated, with the
    # mirror assignment quoted beside.
    columns.append(PredictorColumn(
        name="P1a-family-excluded",
        definition=SLATE_DEFINITIONS["P1a"],
        definition_source=SLATE_DEFINITION_SOURCE, available=True,
        values=p1a_halves["halfb"],
        provenance=("mean held-out r-squared over the hub's own legs at "
                    "half-b::proc_k256, own-family legs EXCLUDED; scored "
                    "against quality measured on half-a (the DISJOINT split)"),
        circularity_guard=("fit-derived, so §7 requires both the family "
                           "exclusion and the disjoint split; both applied")))

    # ---- P2, bank-only -----------------------------------------------------
    p2 = predictor_p2(banks, basis, ctx_record, {h: hub_sites[h] for h in pop},
                      scores_by_ctx[ctx_record.key], identities)
    p2_halves = {
        half: predictor_p2(banks, basis, Context(split=half,
                                                 family=RANK_OF_RECORD),
                           {h: hub_sites[h] for h in pop},
                           scores_by_ctx[f"{half}::{RANK_OF_RECORD}"],
                           identities)
        for half in ("halfa", "halfb")}
    columns.append(PredictorColumn(
        name="P2-compressibility", definition=SLATE_DEFINITIONS["P2"],
        definition_source=SLATE_DEFINITION_SOURCE, available=True, values=p2,
        provenance=("pca_explained at proc_k256 on the full corpus, read off "
                    "the hub's own filed fit records IN THE HUB'S REGISTERED "
                    "ARM (native for an instruct checkpoint, raw for a base "
                    "one — §3's arm rule, since P2 is a per-model constant) "
                    "and asserted identical across every object of that hub "
                    "in that arm"),
        circularity_guard=("BANK-ONLY: computed from the hub's own train-row "
                           "PCA and from nothing about any pair's outcome")))

    # ---- P1b and P6: NAMED, DEFINED, BUT NOT COMPUTABLE DESK-SIDE ----------
    blocker_states = (
        "the webtext-v3 per-text state banks are NOT desk-side — they exist "
        "only under the NODE-SIDE webtext-v3 arm root (checked "
        "by value: every states_*.npz beneath staging/ is v2.1-vintage). The "
        "brief is desk-local CPU only, so this member is SKIPPED with its "
        "blocker rather than improvised from a different object")
    columns.append(PredictorColumn(
        name="P1b-geometric-centrality", definition=SLATE_DEFINITIONS["P1b"],
        definition_source=SLATE_DEFINITION_SOURCE, available=False,
        blocker=(blocker_states + ". SECOND, INDEPENDENT BLOCKER: §7 asks for "
                 "a CHANCE-CORRECTED P1b and no chance correction is defined "
                 "at either freeze tag, in the census pre-statement, or "
                 "anywhere in the repo — inventing one would be improvising a "
                 "slate member's definition"),
        provenance="", circularity_guard=""))
    skipped.append({"member": "P1b-geometric-centrality",
                    "reason": "inputs absent desk-side AND the chance "
                              "correction is undefined at either tag"})
    if p6 is None:
        columns.append(PredictorColumn(
            name="P6-participation-ratio", definition=SLATE_DEFINITIONS["P6"],
            definition_source=SLATE_DEFINITION_SOURCE, available=False,
            blocker=(blocker_states + ". The sigma_L*.npz beside the v3 entropy "
                     "vectors carries an eigen-spectrum, but of a TOKEN-level "
                     "residual covariance over a 60-text stride sample — not the "
                     "train-row mean-state covariance P6 is defined over, so it is "
                     "a different object and is not substituted"),
            provenance="", circularity_guard=""))
        skipped.append({"member": "P6-participation-ratio",
                        "reason": "train-row state banks absent desk-side"})
    else:
        missing = sorted(h for h in pop if h not in p6.values)
        if missing:
            raise ReadsError(
                f"P6-INPUTS covers {p6.n_hubs} hubs but does not cover "
                f"{missing} — a slate member scored on a SUBSET of the hub "
                f"population would violate §7's full-ordering rule")
        #  the artifact's own site/arm per hub must be the site/arm of record,
        #  or the column is a different measurement wearing P6's name
        for hub in pop:
            if p6.sites[hub] != hub_sites[hub]:
                raise ReadsError(
                    f"P6-INPUTS has {hub} at L{p6.sites[hub]} but the sealed "
                    f"artifact registers it at L{hub_sites[hub]}")
            expected_arm = _registered_arm(identities, hub)
            if p6.arms[hub] != expected_arm:
                raise ReadsError(
                    f"P6-INPUTS reads {hub} in the {p6.arms[hub]} arm but its "
                    f"registered arm is {expected_arm} (§3.2) — P6 is a "
                    f"per-model constant and must be read in the model's own arm")
        columns.append(PredictorColumn(
            name="P6-participation-ratio", definition=SLATE_DEFINITIONS["P6"],
            definition_source=SLATE_DEFINITION_SOURCE, available=True,
            values={h: p6.values[h] for h in pop},
            provenance=(
                f"participation ratio of the hub's own TRAIN-ROW mean-state "
                f"covariance spectrum at its registered site, in its REGISTERED "
                f"ARM, from the node-built P6-INPUTS artifact (sha256 "
                f"{p6.sha256}); the census formula verbatim, and every hub's "
                f"statistic RE-DERIVED desk-side from its own banked "
                f"eigenvalues (max |Δ| {p6.recomputed_max_abs_delta:.3e})"),
            circularity_guard=("BANK-ONLY: computed from the hub's own train-row "
                               "state spectrum and from nothing about any "
                               "pair's outcome")))
        flags.append(
            "⚠ P6 HALF COLUMNS DO NOT EXIST: the P6-INPUTS artifact carries ONE "
            "value per hub, on the FROZEN MAIN SPLIT's train rows (n_train "
            f"{p6.n_train_rows}); it holds no per-half spectrum. §7's per-half "
            "condition is therefore evaluated for P6 as the FULL-CORPUS "
            "predictor column against EACH HALF'S QUALITY — which is NOT the "
            "mirror of P2's treatment, where the per-half pca_explained is read "
            "off each half's own fit records. The asymmetry is stated, not "
            "silently averaged; re-deriving a per-half P6 needs the node-side "
            "state banks and a second P6-INPUTS wave.")
        flags.append(
            "⚠ gemma3-27b's P6 is a 3-50x LOW OUTLIER (PR "
            f"{p6.values.get('gemma3-27b', float('nan')):.2f} against a field "
            "of 11-211; 405B tops at 210.8). It is an INPUT FACT, stated "
            "wherever P6 is quoted, and is NOT diagnosed here.")

    available = {c.name: c for c in columns if c.available}
    # §7's clustering rule: P2 and P6 count as ONE cluster in every
    # multiplicity null (their rho was -.93). The cluster is named either way;
    # with P6 restored it finally has BOTH members and the rule binds.
    clusters = {"P1a": ["P1a-family-excluded"],
                "P2/P6": [n for n in ("P2-compressibility",
                                      "P6-participation-ratio")
                          if n in available]}
    if len(clusters["P2/P6"]) > 1:
        flags.append(
            "⚠ THE P2/P6 CLUSTER RULE NOW HAS BOTH MEMBERS, AND IS "
            "MATHEMATICALLY INERT IN THIS NULL — stated because it reads as a "
            "correction and is not one. `max_statistic_null` collapses each "
            "cluster to its best member and then takes the maximum over "
            "clusters, which for a SINGLE cluster is just the maximum over its "
            "members: grouping changes no number here. That is not a defect. A "
            "max-statistic permutation null is correlation-aware BY "
            "CONSTRUCTION — two members correlated at ρ ≈ −.9 make the null's "
            "maximum barely exceed either member's own, so the multiplicity "
            "cost §7 wanted capped is already capped by the resampling rather "
            "than by a grouping constant. The rule is applied and named; the "
            "instrument is simply one that honours it for free.")

    rho_full: dict[str, float] = {}
    rho_halves: dict[str, dict[str, float]] = {"halfa": {}, "halfb": {}}
    partial: dict[str, Optional[float]] = {}
    own_leg_r2 = predictor_p1a(banks, basis, ctx_record,
                               {h: hub_sites[h] for h in pop},
                               scores_by_ctx[ctx_record.key], family_of,
                               identities, exclude_own_family=False)

    for name, column in available.items():
        if name == "P1a-family-excluded":
            # DISJOINT: quality(half-a) vs P1a(half-b), and the mirror.
            rho_full[name] = spearman([quality_halves["halfa"][h] for h in pop],
                                      [p1a_halves["halfb"][h] for h in pop])
            rho_halves["halfa"][name] = rho_full[name]
            rho_halves["halfb"][name] = spearman(
                [quality_halves["halfb"][h] for h in pop],
                [p1a_halves["halfa"][h] for h in pop])
        else:
            rho_full[name] = spearman([quality[h] for h in pop],
                                      [column.values[h] for h in pop])
            #  P2 has a per-half column (its pca_explained is read off each
            #  half's own fit records). P6 does NOT — the P6-INPUTS artifact
            #  carries one full-split value per hub — so its per-half condition
            #  is the FULL-CORPUS column against each half's quality. The
            #  asymmetry is flagged above, never hidden by an average.
            for half in ("halfa", "halfb"):
                predictor_half = (p2_halves[half] if name == "P2-compressibility"
                                  else column.values)
                rho_halves[half][name] = spearman(
                    [quality_halves[half][h] for h in pop],
                    [predictor_half[h] for h in pop])
        # §7 asks P1b to "retain |rho| >= .4 after partialing out own-leg
        # r-squared". P1b is absent, so the control is applied to every
        # computable member instead of being skipped — it is the strictly
        # harder reading, and it exposes any member that is merely own-leg
        # r-squared wearing another name. A degenerate control is recorded as
        # null, never as a number.
        try:
            partial[name] = partial_spearman(
                [quality[h] for h in pop], [column.values[h] for h in pop],
                [own_leg_r2[h] for h in pop])
        except ReadsError as exc:
            partial[name] = None
            flags.append(f"⚠ partial rho for {name} is UNDEFINED: {exc}")

    perm_columns = {name: [available[name].values[h] for h in pop]
                    for name in available}
    perm_quality = [quality[h] for h in pop]
    if "P1a-family-excluded" in perm_columns:
        # the P1a null is run against the DISJOINT quality column it is scored
        # on, so the permuted statistic is the same statistic as the observed.
        perm_quality_p1a = [quality_halves["halfa"][h] for h in pop]
        null_p1a = max_statistic_null(
            perm_quality_p1a,
            {"P1a-family-excluded": perm_columns["P1a-family-excluded"]},
            {"P1a": ["P1a-family-excluded"]},
            seed_name=PERMUTATION_SEED_NAME + "/P1a")
    else:                                                  # pragma: no cover
        null_p1a = None
    null_bank = max_statistic_null(
        perm_quality,
        {n: v for n, v in perm_columns.items() if n != "P1a-family-excluded"},
        {"P2/P6": clusters["P2/P6"]},
        seed_name=PERMUTATION_SEED_NAME + "/bank-only")

    p_values: dict[str, float] = dict(null_bank.p_max_statistic)
    if null_p1a is not None:
        p_values.update(null_p1a.p_max_statistic)

    support_rule = (
        "§7 VERBATIM: “HL-2 SUPPORTED iff ≥1 named predictor clears |ρ| ≥ .6 "
        "on the full ordering AND its max-statistic permutation p < .05 AND it "
        "holds |ρ| ≥ .4 in each I1 half independently.” P1b additionally must "
        "retain |ρ| ≥ .4 after partialing out own-leg r².")
    per_member: dict[str, dict[str, Any]] = {}
    supporting: list[str] = []
    for name in available:
        checks = {
            "abs_rho_full_ordering": abs(rho_full[name]),
            "clears_rho_0.6": abs(rho_full[name]) >= RHO_HL2_FULL,
            "p_max_statistic": p_values[name],
            "clears_p_0.05": p_values[name] < ALPHA_HL2_PERM,
            "abs_rho_halfa": abs(rho_halves["halfa"][name]),
            "abs_rho_halfb": abs(rho_halves["halfb"][name]),
            "clears_rho_0.4_both_halves": (
                abs(rho_halves["halfa"][name]) >= RHO_HL2_HALF
                and abs(rho_halves["halfb"][name]) >= RHO_HL2_HALF),
            "partial_rho_after_own_leg_r2": partial[name],
        }
        checks["SUPPORTS"] = bool(checks["clears_rho_0.6"]
                                  and checks["clears_p_0.05"]
                                  and checks["clears_rho_0.4_both_halves"])
        per_member[name] = checks
        if checks["SUPPORTS"]:
            supporting.append(name)

    supported = bool(supporting)
    if supported:
        best = max(supporting, key=lambda n: abs(rho_full[n]))
        wording = HL2_POSITIVE_TEMPLATE.format(
            predictor=best, rho=f"{abs(rho_full[best]):.3f}")
        verdict = (f"HL-2 SUPPORTED (UNSTAMPED) on {len(supporting)} slate "
                   f"member(s): {', '.join(sorted(supporting))}")
    else:
        wording = None
        verdict = ("HL-2 NOT SUPPORTED (UNSTAMPED) — no computable slate "
                   "member clears all three frozen conditions. §7's positive "
                   "claim wording is therefore NOT quotable, and no other "
                   "positive phrasing is quotable either.")

    return Battery(
        quality_context=ctx_record.key,
        quality_source=("median composed |â_comp − â_obs| over the hub's "
                        "realized slots on the frozen 240-slot basis"),
        population=pop, n_hubs=len(pop), quality=quality,
        quality_by_half=quality_halves, columns=columns,
        rho_full_ordering=rho_full, rho_by_half=rho_halves,
        partial_rho_after_own_leg_r2=partial, own_leg_r2=own_leg_r2,
        max_stat=null_bank, support_rule=support_rule,
        per_member_support=per_member, supported=supported,
        supporting_members=sorted(supporting),
        positive_claim_wording=wording, verdict_text=verdict, skipped=skipped,
        flags=flags, p6_inputs=p6, multiplicity_clusters=clusters)


# ═══════════════════════════════════════════════════════ §6-I5 (cross-basis)
#: THE ONLY v2.1 PATHS THIS MODULE MAY OPEN. Every one is a SCORED RECORD, not
#: a fit — I5 reads the v2.1 side as banked, and re-derives nothing of it.
V21_SCORED_RECORDS: tuple[str, ...] = tuple(
    f"outputs/collection/predictions/scored-predictions-batch{b}-racing-"
    f"2026-07-29.json" for b in (4, 5, 6, 7, 8, 9, 10, 11, 12))
V21_BATCH3_RECORD = "staging/e4-nulls/e4_nulls_batch3_2026-07-28.json"
#: The v2.1 corpus manifest sha every one of those records cites.
CORPUS_SHA_V21 = "5ae355bc5d130f8e9c3ae426f5e71bf2b6e99c74b95369a874bec2abcd59b5d9"


class I5Slot(BaseModel):
    model_config = {"frozen": True}

    source_model: str
    target_model: str
    arm: str
    family: str
    v21_a_obs: float
    v21_a_comp_filed: float
    v3_a_obs: float
    v3_a_comp_8b: float
    v21_record: str


class I5Measurement(BaseModel):
    """§6-I5: a MEASUREMENT with no pass state. ρ and the shift, together."""

    indexes: str
    n_slots: int
    spearman_rho: float
    median_signed_level_shift: float
    mean_signed_level_shift: float
    shift_direction: str
    statement: str


class I5Read(BaseModel):
    """The full I5 block: realized list, both indexings, overlap-clean status."""

    STATUS: str = (
        "MEASUREMENT — §6-I5 has NO PASS STATE. ρ and the median signed level "
        "shift are always quoted together; neither alone is the result.")
    vintage_rule: str = (
        "THIS BLOCK IS THE ONLY PLACE IN THIS ARTIFACT WHERE A v2.1 NUMBER "
        "APPEARS (prereg §3.6). It is cross-basis BY DEFINITION; no quantity "
        "here may be quoted beside any v3 gate or race number.")
    pinned_family: str
    v21_population_n: int
    v21_sources: list[str]
    v21_corpus_manifest_sha256: str
    v3_basis_n: int
    realized_n: int
    realized_arm_counts: dict[str, int]
    excluded_n: int
    exclusion_reasons: dict[str, int]
    realized_slots: list[I5Slot]
    measurements: list[I5Measurement]
    indexing_ambiguity: str
    overlap_clean: dict[str, Any]


def _load_v21_slots(repo: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The v2.1 scored record of record, deduped on (source, target, arm).

    Reads `slots[]` ONLY: `RECORD-CORRECTIONS-2026-07-29.md` establishes that
    the batch records' `disclosures` prose carries three stale paragraphs and
    that "every consumer of these records reads slot-level fields ... No number
    anywhere is affected". So the prose is not read at all.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rel in V21_SCORED_RECORDS:
        path = repo/rel
        if not path.exists():
            raise ReadsError(f"v2.1 scored record absent: {path}")
        doc = json.loads(path.read_text())
        for slot in doc["slots"]:
            if slot.get("cross_vintage"):
                continue
            key = (slot["source"], slot["target"], slot["arm"])
            row = out.setdefault(key, {"observed": float(slot["observed"]),
                                       "family": slot["family"],
                                       "preds": {}, "record": rel})
            if abs(row["observed"] - float(slot["observed"])) > 1e-12:
                raise ReadsError(
                    f"v2.1 record disagrees with itself on â_obs for {key}")
            row["preds"][slot["predictor"]] = float(slot["predicted"])
    path = repo/V21_BATCH3_RECORD
    if not path.exists():
        raise ReadsError(f"v2.1 batch-3 record absent: {path}")
    doc = json.loads(path.read_text())
    for slot in doc["slots"]:
        key = (slot["source_model"], slot["target_model"], slot["arm"])
        if key in out:
            raise ReadsError(f"batch-3 slot {key} duplicates a batch-4+ slot")
        out[key] = {"observed": float(slot["a_hat_observed"]),
                    "family": slot["family"],
                    "preds": {"composed": float(slot["a_comp_filed"])},
                    "record": V21_BATCH3_RECORD}
    return out


def i5_cross_basis(repo: Path, banks: Banks, basis: Sequence[BasisSlot],
                   cache: _Cache) -> I5Read:
    """§6-I5, at the pinned matched native::proc_k128, matched arm per slot."""
    v21 = _load_v21_slots(repo)
    for key, row in v21.items():
        if row["family"] != I5_RANK:
            raise ReadsError(
                f"v2.1 slot {key} is filed at {row['family']}, not the pinned "
                f"{I5_RANK} — §6-I5 pins the rank and does not convert")
    ctx = Context(split="full", family=I5_RANK)
    by_key = {(s.source_model, s.target_model, s.arm): s for s in basis}
    realized: list[I5Slot] = []
    arm_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for key in sorted(v21):
        slot = by_key.get(key)
        if slot is None:
            src, tgt, _arm = key
            why = [side for side, m in (("source", src), ("target", tgt))
                   if m in RACE_SET]
            reasons["endpoint in the §7 race set (race hubs' legs are not "
                    "scoreable pairs, §8)" if why
                    else "no counterpart in the sealed v3 240-slot basis"] += 1
            continue
        tm_src, _ = _oriented_leg(cache, banks, ctx, "8b",
                                  rcp.HUB_SITE_OF_RECORD, slot.source_model,
                                  slot.source_site, slot.arm)
        tm_tgt, _ = _oriented_leg(cache, banks, ctx, "8b",
                                  rcp.HUB_SITE_OF_RECORD, slot.target_model,
                                  slot.target_site, slot.arm)
        v_src = cache.vector(banks, slot.source_model, slot.source_site)
        v_tgt = cache.vector(banks, slot.target_model, slot.target_site)
        a_comp = rcp.composed_exchange_rate(tm_src, tm_tgt, v_src, v_tgt)
        resolved = None
        for path, direction, _pid in banks.layout.pair_probes(
                ctx, slot.source_model, slot.source_site, slot.target_model,
                slot.target_site, slot.arm):
            if path.exists():
                resolved = (path, direction)
                break
        if resolved is None:
            reasons["no v3 direct pair fit at the pinned rank"] += 1
            continue
        a_obs = exchange_rate(cache.tmap(resolved[0]), v_src, v_tgt,
                              direction=resolved[1])  # type: ignore[arg-type]
        realized.append(I5Slot(
            source_model=slot.source_model, target_model=slot.target_model,
            arm=slot.arm, family=I5_RANK, v21_a_obs=v21[key]["observed"],
            v21_a_comp_filed=v21[key]["preds"]["composed"],
            v3_a_obs=float(a_obs), v3_a_comp_8b=float(a_comp),
            v21_record=v21[key]["record"]))
        arm_counts[slot.arm] += 1

    measurements: list[I5Measurement] = []
    for label, left, right in (
            ("â_obs — the OBSERVABLE, through each basis's own direct pair fit",
             [s.v21_a_obs for s in realized], [s.v3_a_obs for s in realized]),
            ("â_comp through the 8b hub — the COMPOSED column",
             [s.v21_a_comp_filed for s in realized],
             [s.v3_a_comp_8b for s in realized])):
        shifts = [b - a for a, b in zip(left, right)]
        rho = spearman(left, right)
        median_shift = float(np.median(shifts))
        direction = ("v3 LEVELS ARE HIGHER" if median_shift > 0
                     else "v3 LEVELS ARE LOWER" if median_shift < 0
                     else "no median shift")
        measurements.append(I5Measurement(
            indexes=label, n_slots=len(realized), spearman_rho=rho,
            median_signed_level_shift=median_shift,
            mean_signed_level_shift=float(np.mean(shifts)),
            shift_direction=direction,
            statement=(f"Spearman rho = {rho:.4f} with a median signed level "
                       f"shift of {median_shift:+.4f} (v3 − v2.1) over "
                       f"n = {len(realized)} slots; {direction}. Structure "
                       f"and level are reported together and neither is "
                       f"quotable alone.")))

    ambiguity = (
        "§6-I5 does not name which quantity it indexes. Both readings are "
        "computed and labelled: â_obs (the observable §1 calls "
        "'corpus-indexed in LEVEL') and â_comp through the 8b hub (the "
        "composed column, which is what the v2.1 record filed). The "
        "REALIZED SLOT LIST IS IDENTICAL under both readings — 8b's v3 hub "
        "legs cover exactly the 16 scoreable endpoints — so the ambiguity is "
        "INERT for the denominator and touches only which pair of columns is "
        "correlated. FLAGGED, not resolved.")

    overlap_clean = {
        "STATUS": "OWED — BLOCKED ON THE CLUSTER, not computed, not simulated",
        "what_is_owed": (
            "§6-I5's beside: the same computation with the 59 shared wikitext "
            "texts' influence excluded — the legs refit on the "
            "shared-text-free train subset (n_train 920 → 861), stamps "
            "carrying the exclusion list's sha, landed under "
            "staging/webtext-v3-fits/i5-overlap-clean/ with a manifest"),
        "blocker": (
            "A refit needs the per-text webtext-v3 STATE BANKS. They are not "
            "desk-side: checked by value, every states_*.npz beneath staging/ "
            "is v2.1-vintage, and the v3 arm root the fit stamps name is "
            "NODE-SIDE only. The desk-side "
            "v3 tree carries fitted maps (va/vb/omega/scale/norms) and "
            "entropy-gradient vectors, from which no re-fit on a train subset "
            "is derivable — the Procrustes needs the paired rows, not the "
            "fitted map. The brief is desk-local CPU only, so this STOPS here "
            "rather than substituting a different computation."),
        "exclusion_set_identity": (
            "the 59 wikitext text_sha256 the freeze census enumerates; they "
            "are TRAIN-side by the §2 v2.1-overlap rule and split 30/29 "
            "across the two I1 halves (ledger, halves wave)"),
        "what_would_unblock_it": (
            "one CPU fit job on the node over the 145 legs + 120 pairs at "
            "proc_k128 with the 59 text_ids dropped from train — the fit lane "
            "of record (fit_transport_maps.run_pair_arm) already takes a "
            "row-subset; nothing new needs writing"),
        "i5_overlap_clean_dir": "staging/webtext-v3-fits/i5-overlap-clean/",
    }

    return I5Read(
        pinned_family=I5_RANK, v21_population_n=len(v21),
        v21_sources=list(V21_SCORED_RECORDS) + [V21_BATCH3_RECORD],
        v21_corpus_manifest_sha256=CORPUS_SHA_V21, v3_basis_n=len(basis),
        realized_n=len(realized), realized_arm_counts=dict(sorted(arm_counts.items())),
        excluded_n=len(v21) - len(realized),
        exclusion_reasons=dict(sorted(reasons.items())),
        realized_slots=realized, measurements=measurements,
        indexing_ambiguity=ambiguity, overlap_clean=overlap_clean)


# ═══════════════════════════════════════════════════════════ C2 and C4 reads
class C2Read(BaseModel):
    """§7's recipe-vs-range test on qwen2.5-72b. Pre-named outcomes, quoted."""

    context: str
    qwen72b_median_abs_error: float
    qwen72b_median_rank: int
    qwen72b_condorcet_rank: int
    n_hubs: int
    llama_3_1_70b_median_abs_error: float
    llama_3_1_70b_median_rank: int
    sign_test: SignTest
    site_matched_beside: dict[str, Any]
    top_tier_reference: dict[str, Any]
    outcome: str
    extremum_rider: str


def c2_read(race: RosterRace, scores: dict[str, HubContextScore],
            ctx: Context) -> C2Read:
    """qwen2.5-72b's descriptive rank + median |e|, against §7's two outcomes."""
    q, l = "qwen2.5-72b-instruct", "llama-3.1-70b-instruct"
    rows = {r.hub: r for r in race.rows}
    missing = [m for m in (q, l, "llama-3.3-70b-instruct")
               if m not in rows or m not in scores]
    if missing:
        raise ReadsError(
            f"C2 cannot be read: {missing} are not in the guard-quotable "
            f"race population, so §7's named statistic has no footing. This "
            f"is a REFUSAL, not an empty read — the desk rules on the "
            f"population before C2 is quoted")
    a, b = scores[q].by_ordinal, scores[l].by_ordinal
    shared = sorted(set(a) & set(b))
    test = sign_test(f"{q} vs {l} @ {ctx.key}",
                     [a[o].abs_error for o in shared],
                     [b[o].abs_error for o in shared], q, l)
    best = race.median_ranking[0]
    delta_to_best = (scores[q].median_abs_error
                     - scores[best].median_abs_error)
    if test.p_two_sided < ALPHA_TIER and test.winner == l:
        outcome = (
            f"PRE-NAMED OUTCOME (UNSTAMPED): degradation toward "
            f"{l}'s tier — {q} loses the paired sign test to {l} "
            f"(p = {test.p_two_sided:.3g}, n = {test.n_effective}). This is "
            f"the RANGE-ARTIFACT reading.")
    elif test.p_two_sided < ALPHA_TIER and test.winner == q:
        outcome = (
            f"PRE-NAMED OUTCOME (UNSTAMPED): {q} BEATS {l} significantly "
            f"(p = {test.p_two_sided:.3g}, n = {test.n_effective}), and its "
            f"median excess over the field's best is {delta_to_best:+.5f} "
            f"(δ = {DELTA_PRACTICAL_EQUIVALENCE}). Top-tier-equivalent at 72B "
            f"supports the RECIPE reading; the desk rules whether "
            f"{delta_to_best:+.5f} is inside the tier.")
    else:
        outcome = (
            f"NEITHER pre-named outcome fires cleanly (UNSTAMPED): {q} and "
            f"{l} are not separated by the paired sign test "
            f"(p = {test.p_two_sided:.3g}, n = {test.n_effective}), so 72B is "
            f"neither shown degraded to {l}'s tier nor shown distinct from "
            f"it. The two pre-named readings are not discriminated.")
    rider = (
        "PRE-NAMED OBSERVATION BESIDE, NEVER CONFLATED WITH THE TEST — the "
        "ledger's C2 rider verbatim (2026-08-04, the 72B scan-fit row): "
        "“72B's r² trough sits exactly on 3.1-70b's sites (37,43), its peak "
        "exactly on 3.3-70b's (58,63).” That is a statement about the 72B "
        "SITE-CURVE r², not about hub quality, and this lane computes no site "
        "curve; it is quoted as pre-named and left with the site-curve "
        "record. It does bear on the test's footing: 72B's registered site "
        "L58 COINCIDES with llama-3.3-70b's (Luxia's site ruling names this "
        "“the C2 site-for-site footing”) and does NOT coincide with "
        "llama-3.1-70b's L37, so the §7-named statistic is the "
        "SITE-MISMATCHED one and the site-matched comparison is quoted "
        "beside.")
    l33 = "llama-3.3-70b-instruct"
    c, d = scores[q].by_ordinal, scores[l33].by_ordinal
    shared33 = sorted(set(c) & set(d))
    test33 = sign_test(f"{q} vs {l33} @ {ctx.key} [SITE-MATCHED, L58 vs L58]",
                       [c[o].abs_error for o in shared33],
                       [d[o].abs_error for o in shared33], q, l33)
    beside = {
        "why": ("§7 names the statistic as “paired sign test "
                "qwen72b-vs-llama70b” and its pre-named outcomes name "
                "llama-3.1-70b, so THAT is the test of record. The "
                "site-matched sibling is quoted beside because the site "
                "ruling that registered 72B at L58 did so partly on the "
                "C2 site-for-site footing with llama-3.3-70b."),
        "comparator": l33,
        "comparator_site": scores[l33].hub_site,
        "qwen72b_site": scores[q].hub_site,
        "comparator_median_abs_error": scores[l33].median_abs_error,
        "comparator_median_rank": rows[l33].median_rank,
        "sign_test": test33.model_dump(),
    }
    return C2Read(
        context=ctx.key, qwen72b_median_abs_error=scores[q].median_abs_error,
        qwen72b_median_rank=rows[q].median_rank,
        qwen72b_condorcet_rank=rows[q].condorcet_rank, n_hubs=race.n_hubs,
        llama_3_1_70b_median_abs_error=scores[l].median_abs_error,
        llama_3_1_70b_median_rank=rows[l].median_rank, sign_test=test,
        site_matched_beside=beside,
        top_tier_reference={"field_best_hub": best,
                            "field_best_median": scores[best].median_abs_error,
                            "qwen72b_excess_over_best": delta_to_best,
                            "delta": DELTA_PRACTICAL_EQUIVALENCE},
        outcome=outcome, extremum_rider=rider)


class C4Pair(BaseModel):
    base: str
    instruct: str
    base_site: Optional[int]
    instruct_site: Optional[int]
    sites_agree: bool
    status: str
    verdict: str


class C4Read(BaseModel):
    """§7's sibling base-vs-instruct read. The F3 lesson, applied verbatim."""

    rule: str = (
        "§7 VERBATIM: base-vs-instruct hub quality and â deltas in the RAW "
        "system at the SAME registered site per pair; “a pair whose sites "
        "differ is quoted site-confounded and EXCLUDED from the C4 read (the "
        "F3 lesson)”.")
    registry_authority: str
    pairs: list[C4Pair]
    n_quotable: int
    read: str


def c4_read(sites: dict[str, int]) -> C4Read:
    """Check each sibling pair's REGISTERED sites; exclude where they differ."""
    pairs: list[C4Pair] = []
    for base, instruct in (("llama-3.1-8b-base", "8b"),
                           ("qwen2.5-7b-base", "qwen-7b")):
        bs, isite = sites.get(base), sites.get(instruct)
        agree = bs is not None and isite is not None and bs == isite
        pairs.append(C4Pair(
            base=base, instruct=instruct, base_site=bs, instruct_site=isite,
            sites_agree=agree,
            status="REGISTERED-BOTH-SIDES",
            verdict=("QUOTABLE" if agree else
                     f"SITE-CONFOUNDED, EXCLUDED — {base} is registered at "
                     f"L{bs} and {instruct} at L{isite}; the F3 lesson forbids "
                     f"reading a post-training delta across two sites")))
    pairs.append(C4Pair(
        base="olmo2-base", instruct="olmo2-7b-instruct", base_site=None,
        instruct_site=sites.get("olmo2-7b-instruct"), sites_agree=False,
        status="ABSENT-WITH-BLOCKER",
        verdict=("olmo2-base is an EXTENSION-set member (§5) that never "
                 "completed collection/registration; it is reported absent "
                 "WITH its named blocker and never silently dropped from a "
                 "quoted rate")))
    quotable = [p for p in pairs if p.verdict == "QUOTABLE"]
    if quotable:                                           # pragma: no cover
        read = (f"{len(quotable)} of 3 sibling pairs are quotable; effect "
                f"sizes and permutation nulls follow for those pairs only.")
    else:
        read = (
            "C4 IS EMPTY (UNSTAMPED): ZERO of the three pre-named sibling "
            "pairs is quotable. Both collected pairs are SITE-CONFOUNDED by "
            "the registry's own ruled sites and are EXCLUDED by §7's own F3 "
            "clause; the third is ABSENT with its named blocker. No "
            "descriptive effect size and no permutation null is computed — a "
            "number computed across two sites would be exactly the F3 "
            "mistake the clause exists to prevent. The read is the "
            "EXCLUSION, and it is a finding about the registry, not about "
            "post-training.")
    return C4Read(
        registry_authority=("metabasis.roster SITE_OF_RECORD / "
                            "read_composed_predictions.SITE_OF_RECORD + "
                            "HUB_SITE_OF_RECORD, the single authority per §5"),
        pairs=pairs, n_quotable=len(quotable), read=read)


# ═══════════════════════════════════════════ §7 clause 5 — parameter counts
#: The HF repo each race candidate's bank key names. Read from
#: `metabasis.roster.ROSTER[...].model_id` where the key is in the roster; the
#: two CARRIED keys (`3b`, `8b`) are not roster rows, and their repo ids are
#: the ones the roster's own prose names for them.
RACE_REPOS: dict[str, str] = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma3-27b": "google/gemma-3-27b-it",
}


class ParamCount(BaseModel):
    """One candidate's total parameter count, derived at a NAMED revision."""

    hub: str
    repo: str
    revision: str
    revision_provenance: str
    total_parameters: int
    total_parameters_b: float
    source: str
    dtype_breakdown: dict[str, int] = Field(default_factory=dict)
    config_fetched: bool
    config_sha256: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


def fetch_param_counts(cache_path: Optional[Path] = None) -> list[ParamCount]:
    """Total parameters per race candidate, at the revision the hub resolves to.

    ⚠ FLAGGED, NOT RESOLVED. §7 clause 5 says "config-derived at the pinned
    revision", but NO PINNED REVISION IS RECORDED for the five race candidates
    anywhere in the campaign — `metabasis.roster` carries `model_id` and
    architecture facts but no revision, and the collection stamps carry a local
    checkpoint-directory path plus a `config_sha256`, not a hub revision. What is
    recorded here is therefore the revision the repo's default branch RESOLVES
    TO at fetch time (`X-Repo-Commit` / the API's `sha`), stated as such.

    The count is read from the repo's SAFETENSORS INDEX metadata at that
    revision (the exact per-tensor parameter total), not from a re-implemented
    per-architecture formula: three of the five repos are gated and refuse
    `config.json` without a token, and a formula that silently mishandles tied
    embeddings or an MoE topology would produce a wrong ORDERING, which is the
    only thing clause 5 uses the number for.
    """
    import urllib.error
    import urllib.request

    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return [ParamCount(**row) for row in cached]

    out: list[ParamCount] = []
    for hub, repo in RACE_REPOS.items():
        notes: list[str] = []
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{repo}",
            headers={"User-Agent": "metabasis-desk-reads"})
        with urllib.request.urlopen(req, timeout=60) as response:
            info = json.loads(response.read())
        safetensors = info.get("safetensors") or {}
        total = safetensors.get("total")
        if not isinstance(total, int):
            raise ReadsError(
                f"{repo}: the hub reports no safetensors parameter total, so "
                f"no config-derived count is available at this revision")
        config_sha = None
        fetched = False
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/{repo}/resolve/{info['sha']}/config.json",
                headers={"User-Agent": "metabasis-desk-reads"})
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
            config_sha = hashlib.sha256(body).hexdigest()
            fetched = True
        except urllib.error.HTTPError as exc:
            notes.append(
                f"config.json not fetchable ({exc.code}) — the repo is gated "
                f"and no token is configured desk-side. The parameter total "
                f"still comes from the repo's own safetensors metadata AT THE "
                f"SAME REVISION, which is public")
        out.append(ParamCount(
            hub=hub, repo=repo, revision=info["sha"],
            revision_provenance=(
                "the repo's default-branch HEAD at fetch time. NO PINNED "
                "REVISION IS RECORDED IN THE CAMPAIGN FOR THIS MODEL — "
                "FLAGGED for ratification"),
            total_parameters=int(total), total_parameters_b=round(total / 1e9, 4),
            source=f"huggingface.co/api/models/{repo} :: safetensors.total",
            dtype_breakdown={k: int(v) for k, v in
                             sorted((safetensors.get("parameters") or {}).items())},
            config_fetched=fetched, config_sha256=config_sha, notes=notes))
    out.sort(key=lambda p: p.total_parameters)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            canonical_json([p.model_dump() for p in out]) + "\n",
            encoding="utf-8")
    return out


class Crowning(BaseModel):
    """§7 clause 5/6's selection — UNSTAMPED-PROPOSED, never a crowning."""

    STATUS: str = (
        "UNSTAMPED-PROPOSED. §7's selection is the desk's and Luxia's to "
        "stamp; this block reports which candidate the frozen criterion "
        "SELECTS given the arithmetic above, and declares nothing.")
    eligible_by_grouping: dict[str, list[str]]
    groupings_agree: bool
    parameter_counts: list[ParamCount]
    proposed_hub_of_record: Optional[str]
    proposed_reason: str
    fallback_fired: Optional[str]
    gemma_note: str


def propose_crowning(verdicts: InvarianceVerdicts, params: list[ParamCount],
                     tiers: dict[str, TierMembership]) -> Crowning:
    """§7 clause 5, and clause 6's NAMED fallbacks when it cannot fire."""
    by_grouping = {g.grouping: g.eligible for g in verdicts.groupings}
    by_hub = {p.hub: p for p in params}
    gemma_top = [k for k, t in tiers.items()
                 if t.binding_eligible and "gemma3-27b" in t.top_tier]
    gemma_note = (
        f"§7 clause 6: gemma3-27b is the PREDICTED-POOR CONTROL. It is "
        f"top-tier in {len(gemma_top)} of "
        f"{sum(1 for t in tiers.values() if t.binding_eligible)} binding "
        f"contexts ({', '.join(sorted(gemma_top)) or 'none'}). "
        + ("Gemma entering the tier is “a named hit on the predictor layer, "
           "quoted wherever the battery is”." if gemma_top else
           "Gemma stays out of the tier, which is the pre-stated expectation "
           "for the control."))
    if not verdicts.groupings_agree:
        return Crowning(
            eligible_by_grouping=by_grouping, groupings_agree=False,
            parameter_counts=params, proposed_hub_of_record=None,
            proposed_reason=(
                "NO PROPOSAL. The two defensible readings of §7 clause 4's "
                "'the four contexts' produce DIFFERENT eligible sets, so the "
                "selection would depend on a reading the frozen text does not "
                "fix. Flagged for desk adjudication; nothing is proposed."),
            fallback_fired=None, gemma_note=gemma_note)
    eligible = sorted(set(next(iter(by_grouping.values()))))
    if not eligible:
        return Crowning(
            eligible_by_grouping=by_grouping, groupings_agree=True,
            parameter_counts=params, proposed_hub_of_record=None,
            proposed_reason="",
            fallback_fired=(
                "§7 clause 6, NAMED FALLBACK: no eligible candidate → no "
                "crowning; hubness is per-roster-measured and the incumbent "
                "continues as convenience-with-caveat."),
            gemma_note=gemma_note)
    ranked = sorted(eligible, key=lambda h: (by_hub[h].total_parameters, h))
    winner = ranked[0]
    reason = (
        f"§7 clause 5: the SMALLEST eligible candidate by total parameter "
        f"count, config-derived at the recorded revision — "
        f"{winner} at {by_hub[winner].total_parameters:,} parameters "
        f"({by_hub[winner].total_parameters_b}B), ahead of "
        + ", ".join(f"{h} ({by_hub[h].total_parameters_b}B)"
                    for h in ranked[1:])
        + ". The config-derived number is binding, not §7's expected values; "
          "the realized ordering AGREES with the expectation "
          "(qwen2.5-3b < llama-3.2-3b < 8B < 27B < 32.8B).")
    return Crowning(
        eligible_by_grouping=by_grouping, groupings_agree=True,
        parameter_counts=params, proposed_hub_of_record=winner,
        proposed_reason=reason, fallback_fired=None, gemma_note=gemma_note)


# ═══════════════════════════════════════════════════════════ the family table
#: P1a's family exclusion needs a model→family map. TWO on-disk registries
#: already carry one and the census reconciled them; both are read rather than
#: retyped. Their union covers 18 of the 21 §5 core models — the three §5 ADDS
#: are absent from both, and are extended here BY THE REGISTRIES' OWN LINEAGE
#: CONVENTION (which places every Llama together and every Qwen together),
#: FLAGGED for ratification.
FAMILY_REGISTRIES: tuple[tuple[str, str], ...] = (
    ("staging/batch12-prep/b12_slate.py", "FAMILY_OF"),
    ("staging/chart-overlap-preview/chart_overlap_preview.py", "FAMILY_OF"),
)
FAMILY_ADDS: dict[str, str] = {
    "qwen2.5-72b-instruct": "qwen",
    "qwen2.5-7b-base": "qwen",
    "llama-3.1-8b-base": "llama",
}


def load_family_table(repo: Path) -> tuple[dict[str, str], list[str]]:
    """Reconcile the two on-disk FAMILY_OF registries; extend for the §5 adds."""
    import ast as _ast
    merged: dict[str, str] = {}
    notes: list[str] = []
    for rel, name in FAMILY_REGISTRIES:
        path = repo/rel
        if not path.exists():
            raise ReadsError(f"family registry absent: {path}")
        tree = _ast.parse(path.read_text())
        found: Optional[dict[str, str]] = None
        for node in _ast.walk(tree):
            targets = getattr(node, "targets", None) or (
                [node.target] if hasattr(node, "target") else [])
            for target in targets:
                if isinstance(target, _ast.Name) and target.id == name:
                    found = {k: v.lower() for k, v in
                             _ast.literal_eval(node.value).items()}
        if found is None:
            raise ReadsError(f"{rel} carries no {name} table")
        for key, value in found.items():
            if key in merged and merged[key] != value:
                raise ReadsError(
                    f"the two family registries DISAGREE on {key!r}: "
                    f"{merged[key]!r} vs {value!r}")
            merged[key] = value
    for key, value in FAMILY_ADDS.items():
        if key in merged:                                  # pragma: no cover
            continue
        merged[key] = value
        notes.append(
            f"{key} is absent from both on-disk family registries (a §5 add) "
            f"and is placed in {value!r} by the registries' own lineage "
            f"convention — FLAGGED for ratification")
    return merged, notes


# ═════════════════════════════════════════════════════════════ the driver
class ReadsRun(BaseModel):
    """Everything the reads artifact carries. No timestamps live here."""

    artifact: str = "webtext-v3-reads/v1"
    STATUS: str = (
        "UNSTAMPED ARITHMETIC (C section 8). This lane computes and counts; "
        "every PASS/FAIL below carries UNSTAMPED and none of it is a claim. "
        "Stamps are Luxia's at the first-read.")
    frozen_ref: str
    prereg_sha256: str
    parent_prereg_sha256: str
    artifact_path: str
    artifact_sha256: str
    stamp_verification: dict[str, Any]
    corpus_manifest_sha256: str
    splits_sha256: str
    halves_sha256: str
    manifests_verified: list[dict[str, Any]]
    family_of_record: str
    basis_n: int
    contexts: list[str]
    hub_sites: dict[str, int]
    scores: dict[str, dict[str, dict[str, Any]]]
    guards: dict[str, dict[str, Any]]
    guard_quotable_population: dict[str, Any]
    tiers: dict[str, TierMembership]
    invariance: InvarianceVerdicts
    order_besides: list[OrderBeside]
    clause_readings: list[ClauseReading]
    roster_race: RosterRace
    separation_floor: SeparationFloor
    population_reads: list[PopulationRead]
    crowning: Crowning
    battery: Battery
    c2: C2Read
    c4: C4Read
    i5: I5Read
    populations: dict[str, Any]
    flags: list[str]
    verdict_table: list[dict[str, Any]]


# ═══════════════════════════════════════════════════ manifest verification
class ManifestCheck(BaseModel):
    manifest: str
    manifest_sha256: str
    n_entries: int
    n_ok: int
    n_mismatched: int
    n_missing: int
    verified: bool
    mismatched: list[str] = Field(default_factory=list)


def verify_manifest(path: Path) -> ManifestCheck:
    """`sha256sum --check`, in process, over the manifest's own anchor dir.

    In process rather than shelled out so the RESULT is a typed object the
    artifact carries — a green line in a log is not evidence a later reader can
    check, and the brief binds this lane to verify before consuming.
    """
    anchor = path.parent
    entries: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel:
            continue
        entries.append((digest, rel))
    ok = mismatched = missing = 0
    bad: list[str] = []
    for digest, rel in entries:
        target = anchor/rel
        if not target.exists():
            missing += 1
            bad.append(f"MISSING {rel}")
            continue
        if sha256_of(target) == digest:
            ok += 1
        else:
            mismatched += 1
            bad.append(f"MISMATCH {rel}")
    return ManifestCheck(
        manifest=str(path), manifest_sha256=sha256_of(path),
        n_entries=len(entries), n_ok=ok, n_mismatched=mismatched,
        n_missing=missing, verified=(mismatched == 0 and missing == 0
                                     and len(entries) > 0),
        mismatched=bad[:20])


STAGING_MANIFESTS: tuple[str, ...] = (
    "staging/webtext-v3-fits/halves/MANIFEST-halves-fits-20260804.sha256",
    "staging/webtext-v3-fits/hub-legs/MANIFEST-hub-legs-meta-20260804.sha256",
    "staging/webtext-v3-fits/hub-legs/MANIFEST-hub-legs-meta-20260804-rawarm.sha256",
    "staging/webtext-v3-fits/pairs/MANIFEST-pairs-meta-20260804.sha256",
    "staging/webtext-v3-vectors/MANIFEST-vectors-20260804.sha256",
)


# ═════════════════════════════════════════════════════════════ the driver
def run_reads(repo: Path, out_dir: Path, verify: bool = True,
              param_cache: Optional[Path] = None,
              p6_inputs: Optional[Path] = None) -> ReadsRun:
    """The whole enactment: one engine, every §6/§7 read, one artifact."""
    flags: list[str] = []
    p6 = load_p6_inputs(p6_inputs) if p6_inputs is not None else None
    if p6 is not None:
        logger.info("P6-INPUTS: %d hubs, sha256 %s, PR re-derived max |Δ| %.3e",
                    p6.n_hubs, p6.sha256, p6.recomputed_max_abs_delta)
    staging = repo/"staging"
    banks = Banks(
        vectors_root=staging/"webtext-v3-vectors"/"vectors",
        layout=TreeLayout(fits_root=staging/"webtext-v3-fits"),
        artifact_path=(staging/"webtext-v3-predictions"
                       / "PREDICTIONS-webtext-v3-2026-08-04.json"))

    # ---- manifests, before a single fit object is consumed -----------------
    checks: list[ManifestCheck] = []
    if verify:
        for rel in STAGING_MANIFESTS:
            path = repo/rel
            if not path.exists():
                flags.append(f"⚠ staging manifest absent: {rel}")
                continue
            check = verify_manifest(path)
            checks.append(check)
            if not check.verified:
                raise ReadsError(
                    f"{rel} does NOT verify ({check.n_mismatched} mismatched, "
                    f"{check.n_missing} missing) — refusing to consume the "
                    f"tree it covers")
            logger.info("manifest OK: %s (%d/%d)", Path(rel).name,
                        check.n_ok, check.n_entries)
    covered = {c.manifest for c in checks}
    flags.append(
        "⚠ MANIFEST COVERAGE, stated not assumed: the halves manifest covers "
        "the 2650 per-half FIT FILES, but the hub-legs and pairs manifests "
        "cover META ONLY (plans, logs, run markers) — no manifest desk-side "
        "covers the FULL-CORPUS fit_*.npz in hub-legs/fits-hub-legs/ or "
        "pairs/fits-pairs/. Those objects are consumed here on the strength "
        "of the sealed prediction artifact, which names each full-corpus hub "
        "leg by path and whose â_comp this lane reproduces EXACTLY (max |Δ| = "
        "0.0 over 240 slots x 5 columns) — a byte-level parity proof that "
        "stands in for the missing manifest. FLAGGED; a full-corpus fit "
        "manifest is owed.")

    # ---- the sealed artifact, stamp FIRST ----------------------------------
    artifact, verification = rcp.load_v3_prediction_artifact(banks.artifact_path)
    basis = basis_from_artifact(artifact)
    self_desc = artifact.self_description

    hub_sites: dict[str, int] = {m["key"]: m["site_of_record"]
                                 for m in artifact.models}
    roles = {m["key"]: m["role"] for m in artifact.models}
    identities = {m["key"]: m["checkpoint_identity"] for m in artifact.models}
    non_race = sorted(set(hub_sites) - set(RACE_SET))

    family_of, family_notes = load_family_table(repo)
    flags.extend("⚠ " + note for note in family_notes)

    # ---- the grid ----------------------------------------------------------
    grid: list[Context] = (
        [Context(split="full", family=f) for f in BINDING_RANKS]
        + [Context(split=s, family=f) for s in ("halfa", "halfb")
           for f in ALL_RANKS])
    # The §6 besides are "over ALL guard-quotable hubs at the family of
    # record", and I2's beside compares k128 against k256 on the full corpus —
    # so the full corpus needs the WHOLE hub population at BOTH binding ranks,
    # not just the five candidates.
    all_hub_contexts = ({f"{s}::{RANK_OF_RECORD}" for s in SPLITS}
                        | {"full::proc_k128"})

    cache = _Cache()
    scores_by_ctx: dict[str, dict[str, HubContextScore]] = {}
    guards_by_ctx: dict[str, dict[str, HubGuard]] = {}
    for ctx in grid:
        hubs = (list(RACE_SET) + non_race if ctx.key in all_hub_contexts
                else list(RACE_SET))
        got: dict[str, HubContextScore] = {}
        guards: dict[str, HubGuard] = {}
        for hub in hubs:
            score = score_hub(banks, basis, ctx, hub, hub_sites[hub], cache,
                              require_full_basis=(hub in RACE_SET))
            got[hub] = score
            guards[hub] = hub_guard(banks, basis, ctx, hub, hub_sites[hub],
                                    score)
        scores_by_ctx[ctx.key] = got
        guards_by_ctx[ctx.key] = guards
        logger.info("context %s scored: %d hubs, %d map loads so far",
                    ctx.key, len(got), cache.n_map_loads)
        cache.clear_maps()

    # ---- §7 clause 3 per context; §6 clauses over the grid ------------------
    tiers = {ctx.key: tier_membership(scores_by_ctx[ctx.key], ctx)
             for ctx in grid}
    verdicts = evaluate_invariance(tiers)

    # ---- the guard-quotable population, DERIVED and flagged -----------------
    ctx_record = Context(split="full", family=RANK_OF_RECORD)
    record_guards = guards_by_ctx[ctx_record.key]
    quotable = sorted(h for h, g in record_guards.items() if g.quotable)
    not_quotable = sorted(h for h, g in record_guards.items() if not g.quotable)
    population = {
        "clause": ("§6-I1 beside / §6-I4 binding: “all guard-quotable hubs at "
                   "the family of record, enumerated at first fit”"),
        "enumeration_on_record": False,
        "derivation": (
            "NO SUCH ENUMERATION EXISTS. Checked by value: the sealed "
            "prediction artifact, the hub-leg wave meta, the pair-fit wave "
            "meta, the halves manifest and the ledger carry no "
            "guard-quotable-hub list, and the phrase appears only in the "
            "frozen prereg's own prose. The population is therefore DERIVED "
            "from the two guards the prereg defines — §4's rank guard "
            "(k_eff <= n_train/1.2) and §3.4's fit validity (both nulls + "
            ">= 2-of-4 stratum carry) — applied to EVERY fit object each "
            "hub's race score consumes. FLAGGED FOR RATIFICATION AT "
            "FIRST-READ."),
        "family_of_record": RANK_OF_RECORD,
        "context": ctx_record.key,
        "n_candidates_examined": len(record_guards),
        "guard_quotable": quotable,
        "n_guard_quotable": len(quotable),
        "not_guard_quotable": not_quotable,
        "per_hub_slot_counts": {
            h: {"n_slots_realized": scores_by_ctx[ctx_record.key][h].n_slots_realized,
                "arm_counts": scores_by_ctx[ctx_record.key][h].arm_counts,
                "leg_source": scores_by_ctx[ctx_record.key][h].leg_source}
            for h in sorted(scores_by_ctx[ctx_record.key])},
    }
    flags.append(
        "⚠ POPULATION DERIVED, NOT RATIFIED: §6-I1/I4's “all guard-quotable "
        "hubs at the family of record, enumerated at first fit” was never "
        "enumerated by any wave. The derivation above is deterministic and "
        "shown working, and it is the desk's to ratify at first-read.")

    # ---- §6 besides: full-order rho ---------------------------------------
    race_scores_record = scores_by_ctx[ctx_record.key]
    besides: list[OrderBeside] = []
    med = {ctx.key: {h: scores_by_ctx[ctx.key][h].median_abs_error
                     for h in scores_by_ctx[ctx.key]} for ctx in grid}
    pop_halves = sorted(
        set(quotable)
        & {h for h, g in guards_by_ctx[f"halfa::{RANK_OF_RECORD}"].items()
           if g.quotable}
        & {h for h, g in guards_by_ctx[f"halfb::{RANK_OF_RECORD}"].items()
           if g.quotable})
    besides.append(order_beside(
        f"I1 — half-a vs half-b, full order at {RANK_OF_RECORD}", pop_halves,
        med[f"halfa::{RANK_OF_RECORD}"], med[f"halfb::{RANK_OF_RECORD}"],
        verdicts.i1_halves_agree))
    pop_ranks = sorted(set(quotable) & set(med["full::proc_k128"]))
    besides.append(order_beside(
        "I2 — proc_k128 vs proc_k256, full order on the full corpus",
        pop_ranks, med["full::proc_k128"], med["full::proc_k256"],
        verdicts.i2_ranks_agree))
    for half in ("halfa", "halfb"):
        pop = sorted(set(med[f"{half}::proc_k128"]))
        besides.append(order_beside(
            f"I2 — proc_k128 vs proc_k256 on {half} (candidate set only)",
            pop, med[f"{half}::proc_k128"], med[f"{half}::proc_k256"],
            verdicts.i2_ranks_agree))
        besides.append(order_beside(
            f"I2 CONTINUITY BESIDE (k32, NEVER binding) — proc_k32 vs "
            f"{RANK_OF_RECORD} on {half}", pop, med[f"{half}::proc_k32"],
            med[f"{half}::{RANK_OF_RECORD}"], verdicts.i2_ranks_agree))

    # ---- §8 step 5 all-roster race, I3, I4 ---------------------------------
    # ⚠ THE I3/I4 POPULATION HAS A SECOND READING, and it changes the verdict.
    # §6 says "all guard-quotable hubs AT THE FAMILY OF RECORD", and §3.2's
    # family of record is native::proc_k256 for instruct<->instruct with
    # "arms are never pooled in any aggregate". A base-checkpoint hub has NO
    # native bank at all, so it is not quotable at the family of record; and a
    # median taken over a hub's native AND raw slots together IS an arm-pooled
    # aggregate. Both readings are computed and both are reported.
    native_quotable = sorted(h for h in quotable
                             if identities.get(h, "instruct") == "instruct")
    base_quotable = sorted(set(quotable) - set(native_quotable))
    race_pop = {h: race_scores_record[h] for h in quotable}
    race = roster_race(race_pop, ctx_record, roles, identities)
    floor = separation_floor(race_pop, ctx_record, race.median_ranking)

    fam_pop = filter_scores(race_scores_record, native_quotable, arm="native")
    fam_race = roster_race(fam_pop, ctx_record, roles, identities)
    fam_floor = separation_floor(fam_pop, ctx_record, fam_race.median_ranking)

    population_reads = [
        PopulationRead(
            population_reading="ALL-CORE-HUBS (arm-pooled)",
            definition=(
                "every core hub that clears both guards, each scored over its "
                "OWN realized subset of the frozen 240-slot basis with the "
                "slot's own arm — the brief's derivation. It pools native and "
                "raw slots inside one median, which is what §7 clause 2's "
                "'median on the fixed shared basis' does for the race, and it "
                "admits the three base-checkpoint hubs, whose realized sets "
                "are raw-only."),
            n_hubs=race.n_hubs, arm_restriction=None, race=race,
            separation_floor=floor, diagnosis=_floor_diagnosis(floor)),
        PopulationRead(
            population_reading="AT-THE-FAMILY-OF-RECORD (native only)",
            definition=(
                "the hubs quotable AT §3.2's family of record — native::"
                "proc_k256 — scored over NATIVE slots only. The three "
                "base-checkpoint hubs (" + ", ".join(base_quotable) + ") have "
                "no native bank and are therefore not quotable at the family "
                "of record at all; and restricting to one arm satisfies §3.2's "
                "'arms are never pooled in any aggregate' literally, which the "
                "other reading does not."),
            n_hubs=fam_race.n_hubs, arm_restriction="native", race=fam_race,
            separation_floor=fam_floor,
            diagnosis=_floor_diagnosis(fam_floor)),
    ]
    readings = clause_readings(tiers)

    # ---- §7 selection, battery, C2/C4, I5 ----------------------------------
    params = fetch_param_counts(param_cache)
    crowning = propose_crowning(verdicts, params, tiers)
    battery = build_battery(banks, basis, quotable, hub_sites, scores_by_ctx,
                            family_of, identities, [
        "⚠ SLATE PROVENANCE: §7 names P1a/P1b/P2/P6 but defines no member, and "
        "the PARENT contract at freeze/transport-campaign does not mention "
        "them at all (checked by value at the tag). The operational "
        "definitions used here are the 17-hub census pre-statement's §3, the "
        "only definitions the campaign holds. FLAGGED.",
        "⚠ DISJOINT-SPLIT READING: §7's “legs fit on the split half not used "
        "for the hub-quality measurement” is unsatisfiable against a "
        "full-corpus quality column (no half is disjoint from the full "
        "corpus). The only satisfiable reading — PER-HALF quality against the "
        "OTHER half's P1a — is used, both assignments are reported, and "
        "“full-ordering” is read as the challenge set words it: the whole hub "
        "population, never a top-k slice. FLAGGED.",
        ("⚠ TWO OF FOUR SLATE MEMBERS ARE SKIPPED WITH BLOCKERS (P1b, P6) — "
         "see battery.skipped. The battery therefore runs on 2 members, and "
         "the multiplicity null's P2/P6 cluster contains P2 alone."
         if p6 is None else
         "⚠ ONE OF FOUR SLATE MEMBERS IS SKIPPED WITH A BLOCKER (P1b) — §7 "
         "asks for a CHANCE-CORRECTED P1b and no chance correction is defined "
         "at either freeze tag, in the census pre-statement, or anywhere in "
         "the repo. NOT invented. P6 is RESTORED from the node-built "
         "P6-INPUTS artifact, so the battery runs on 3 members and the P2/P6 "
         "cluster finally holds both of its named members."),
    ], p6=p6)
    cache.clear_maps()
    c2 = c2_read(race, race_scores_record, ctx_record)
    sites = dict(rcp.SITE_OF_RECORD)
    sites["8b"] = rcp.HUB_SITE_OF_RECORD
    c4 = c4_read(sites)
    i5 = i5_cross_basis(repo, banks, basis, cache)
    cache.clear_maps()

    # ---- the verdict table -------------------------------------------------
    def row(clause: str, binding: bool, grouping: Optional[str], passes: bool,
            detail: str) -> dict[str, Any]:
        return {"clause": clause, "binding": binding,
                "context_grouping": grouping or "n/a",
                "verdict": ("PASS (UNSTAMPED)" if passes
                            else "FAIL (UNSTAMPED)"), "detail": detail}

    table: list[dict[str, Any]] = []
    for grouping in ("half-x-rank-2x2", "halves-plus-full-ranks"):
        g = next(x for x in verdicts.groupings if x.grouping == grouping)
        table.append(row(
            "§6-I1 every §7 eligibility decision agrees across halves", True,
            grouping, verdicts.i1_halves_agree,
            f"half-a vs half-b top tiers at k128 and k256: "
            f"{json.dumps(verdicts.i1_detail, sort_keys=True)}"))
        table.append(row(
            "§6-I2 eligibility agreement across ranks", True, grouping,
            verdicts.i2_ranks_agree,
            f"k128 vs k256 top tiers per split: "
            f"{json.dumps(verdicts.i2_detail, sort_keys=True)}"))
        table.append(row(
            "§7 clause 4 invariance eligibility (noise-ejection guard)", True,
            grouping, bool(g.eligible),
            f"eligible={g.eligible}; ineligible={g.ineligible}; "
            f"expulsion counts={json.dumps(g.expulsion_counts, sort_keys=True)}"))
        table.append(row(
            "§6-I4 separation floor (top x bottom quartile, Holm, all p<.01)",
            True, grouping, floor.all_below_alpha,
            f"{floor.n_pairs} pairs over n={floor.n_hubs} hubs; "
            f"failures={floor.failures}"))
    for reading in readings:
        table.append(row(
            f"§6-{reading.clause} [{reading.reading} reading] "
            f"{reading.definition[:80]}", True, "both", reading.agree,
            f"verdicts={json.dumps(reading.verdicts_by_group, sort_keys=True)}; "
            f"disagreeing={reading.disagreeing_candidates}"))
    for pread in population_reads:
        table.append(row(
            f"§6-I4 separation floor [{pread.population_reading}]", True,
            "n/a", pread.separation_floor.all_below_alpha,
            f"n_hubs={pread.n_hubs}, {pread.separation_floor.n_pairs} pairs; "
            f"{pread.diagnosis}"))
        table.append({
            "clause": f"§6-I3 aggregation [{pread.population_reading}]",
            "binding": False, "context_grouping": "n/a",
            "verdict": "DESCRIPTIVE (UNSTAMPED)",
            "detail": pread.race.i3_statement})
    table.append(row(
        "§7 clause 3 top-tier membership computed on every binding context",
        True, "both", all(t.top_tier for t in tiers.values()),
        f"contexts={sorted(tiers)}"))
    table.append({
        "clause": "§6-I3 aggregation (descriptive, stated once)",
        "binding": False, "context_grouping": "n/a",
        "verdict": "DESCRIPTIVE (UNSTAMPED)", "detail": race.i3_statement})
    table.append({
        "clause": "§6-I5 corpus-indexing measurement", "binding": False,
        "context_grouping": "n/a", "verdict": "MEASUREMENT — NO PASS STATE",
        "detail": " | ".join(m.statement for m in i5.measurements)})
    table.append({
        "clause": "§7 HL-2 predictor battery", "binding": False,
        "context_grouping": "n/a",
        "verdict": ("SUPPORTED (UNSTAMPED)" if battery.supported
                    else "NOT SUPPORTED (UNSTAMPED)"),
        "detail": battery.verdict_text})

    if floor.all_below_alpha != fam_floor.all_below_alpha:
        flags.append(
            "⚠⚠ THE I3/I4 POPULATION READING CHANGES THE §6-I4 VERDICT: "
            f"ALL-CORE-HUBS (arm-pooled, n={race.n_hubs}) gives "
            f"{'PASS' if floor.all_below_alpha else 'FAIL'} and "
            f"AT-THE-FAMILY-OF-RECORD (native only, n={fam_race.n_hubs}) gives "
            f"{'PASS' if fam_floor.all_below_alpha else 'FAIL'}. §6 says "
            "“all guard-quotable hubs at the family of record” and "
            "§3.2 says “arms are never pooled in any aggregate”; the "
            "two sentences pull toward different populations. BOTH ARE "
            "REPORTED AND NEITHER IS RESOLVED — the desk adjudicates.")
    else:
        flags.append(
            f"✓ The two I3/I4 population readings AGREE on §6-I4 "
            f"({'PASS' if floor.all_below_alpha else 'FAIL'} under both).")
    strict_i1 = next(r for r in readings
                     if r.clause == "I1" and r.reading == "STRICT")
    guarded_i1 = next(r for r in readings
                      if r.clause == "I1" and r.reading == "GUARDED")
    strict_i2 = next(r for r in readings
                     if r.clause == "I2" and r.reading == "STRICT")
    guarded_i2 = next(r for r in readings
                      if r.clause == "I2" and r.reading == "GUARDED")
    if (strict_i1.agree != guarded_i1.agree
            or strict_i2.agree != guarded_i2.agree):
        flags.append(
            "⚠⚠ A SECOND §6 AMBIGUITY, BEYOND THE TWO CONTEXT-GROUPINGS THE "
            "BRIEF NAMES, AND IT CHANGES THE I1/I2 VERDICTS: “every §7 "
            "eligibility decision agrees” can mean per-context TOP-TIER "
            "MEMBERSHIP (STRICT) or the §7 clause-4 ELIGIBILITY VERDICT with "
            "its noise-ejection guard (GUARDED). Realized: "
            f"I1 STRICT={'agree' if strict_i1.agree else 'DISAGREE'} vs "
            f"GUARDED={'agree' if guarded_i1.agree else 'DISAGREE'}; "
            f"I2 STRICT={'agree' if strict_i2.agree else 'DISAGREE'} vs "
            f"GUARDED={'agree' if guarded_i2.agree else 'DISAGREE'}. Note the "
            "textual argument: under STRICT, §7's noise-ejection guard is "
            "DEAD LETTER — it tolerates exactly the single-context expulsion "
            "that STRICT treats as a failure, so it could never fire without "
            "I1 having already failed. BOTH READINGS REPORTED, NEITHER "
            "RESOLVED — the desk adjudicates.")
    else:
        flags.append(
            "✓ The STRICT and GUARDED readings of §6-I1/I2's “eligibility "
            "decision” agree, so that ambiguity is inert here.")
    if not verdicts.groupings_agree:
        flags.append(
            "⚠⚠ THE TWO CONTEXT-GROUPINGS DISAGREE on §7 clause 4 eligibility: "
            + "; ".join(verdicts.grouping_disagreement)
            + ". NO VERDICT IS ISSUED on that clause; the desk adjudicates.")
    else:
        flags.append(
            "✓ The two defensible readings of §7 clause 4's “the four "
            "contexts” AGREE on the eligible set, so the reading is inert "
            "here and no adjudication is required.")

    halves_path = repo/"staging"/"webtext-v3-draft"/"halves.json"
    splits_path = repo/"staging"/"webtext-v3-draft"/"splits.json"

    return ReadsRun(
        frozen_ref=("freeze/webtext-v3 :: "
                    "docs/planning/PREREG-webtext-v3-2026-08-03.md §6/§7/§8"),
        prereg_sha256=self_desc["prereg"]["sha256"],
        parent_prereg_sha256=(
            "33ba8290487813a80001c4cf5b71bbb57b57004ff610aa9fcf1f651005875902"),
        artifact_path=str(banks.artifact_path.relative_to(repo)),
        artifact_sha256=verification.sha256_recomputed,
        stamp_verification=relativize(verification.model_dump(), repo),
        corpus_manifest_sha256=self_desc["corpus_manifest_sha256"],
        splits_sha256=sha256_of(splits_path),
        halves_sha256=sha256_of(halves_path),
        manifests_verified=[relativize(c.model_dump(), repo) for c in checks],
        family_of_record=RANK_OF_RECORD, basis_n=len(basis),
        contexts=[c.key for c in grid], hub_sites=hub_sites,
        scores={k: {h: s.model_dump() for h, s in v.items()}
                for k, v in scores_by_ctx.items()},
        guards={k: {h: g.model_dump() for h, g in v.items()}
                for k, v in guards_by_ctx.items()},
        guard_quotable_population=population, tiers=tiers,
        invariance=verdicts, order_besides=besides,
        clause_readings=readings, roster_race=race,
        separation_floor=floor, population_reads=population_reads,
        crowning=crowning, battery=battery,
        c2=c2, c4=c4, i5=i5,
        populations={
            "race_basis": {
                "n": len(basis),
                "source": "the sealed §8.2 prediction artifact's ordered slots",
                "arm_counts": dict(sorted(Counter(
                    s.arm for s in basis).items())),
                "pair_class_counts": dict(sorted(Counter(
                    s.pair_class for s in basis).items()))},
            "candidates": list(RACE_SET),
            "non_race_core_hubs": non_race,
            "guard_quotable_n": len(quotable)},
        flags=flags, verdict_table=table)


# ═══════════════════════════════════════════════════════════════ selftest
def _proc_map(rng: np.random.Generator, d_hub: int, d_model: int, k: int,
              scale: float = 1.0) -> TransportMap:
    """A synthetic hub→model proc map with orthonormal PCA rows and orthogonal Ω."""
    va = np.linalg.qr(rng.standard_normal((d_hub, k)))[0].T
    vb = np.linalg.qr(rng.standard_normal((d_model, k)))[0].T
    omega = np.linalg.qr(rng.standard_normal((k, k)))[0]
    return TransportMap(kind="proc", src_norm=1.0, tgt_norm=1.0,
                        va=va.astype(np.float32), vb=vb.astype(np.float32),
                        omega=omega.astype(np.float32), scale=scale)


def selftest() -> int:                                   # noqa: C901 — a checklist
    """Every claim this module makes about its own arithmetic, proved."""
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        print(("  ok   " if condition else "  FAIL ") + label)
        if not condition:
            failures.append(label)

    print("── 1. the engine, on a fixture whose pairing is KNOWN BY CONSTRUCTION")
    # A hub H and two models A, B. If the two legs are built over ONE hub basis
    # and the DIRECT pair map is the exact composition of them, then â_comp and
    # â_obs must agree to machine precision — the pairing the engine performs is
    # then verifiable rather than asserted, and any mis-pairing (swapped legs,
    # wrong direction, wrong slot) breaks it loudly.
    rng = np.random.default_rng(20260804)
    k, d_hub, d_a, d_b = 8, 32, 24, 20
    va = np.linalg.qr(rng.standard_normal((d_hub, k)))[0].T
    vb_a = np.linalg.qr(rng.standard_normal((d_a, k)))[0].T
    vb_b = np.linalg.qr(rng.standard_normal((d_b, k)))[0].T
    om_a = np.linalg.qr(rng.standard_normal((k, k)))[0]
    om_b = np.linalg.qr(rng.standard_normal((k, k)))[0]
    leg_a = TransportMap(kind="proc", src_norm=1.0, tgt_norm=1.0,
                         va=va.astype(np.float32), vb=vb_a.astype(np.float32),
                         omega=om_a.astype(np.float32), scale=1.0)
    leg_b = TransportMap(kind="proc", src_norm=1.0, tgt_norm=1.0,
                         va=va.astype(np.float32), vb=vb_b.astype(np.float32),
                         omega=om_b.astype(np.float32), scale=1.0)
    # the direct A→B map that IS the composition: A's basis on the source side,
    # B's on the target side, Ω = Ω_Aᵀ Ω_B over the shared hub basis.
    direct = TransportMap(kind="proc", src_norm=1.0, tgt_norm=1.0,
                          va=vb_a.astype(np.float32), vb=vb_b.astype(np.float32),
                          omega=(om_a.T @ om_b).astype(np.float32), scale=1.0)
    v_a = rng.standard_normal(d_a)
    v_a /= np.linalg.norm(v_a)
    v_b = rng.standard_normal(d_b)
    v_b /= np.linalg.norm(v_b)
    a_comp = rcp.composed_exchange_rate(leg_a, leg_b, v_a, v_b)
    a_obs = exchange_rate(direct, v_a, v_b, direction="fwd")
    check(abs(a_comp - a_obs) < 5e-6,
          f"composed(hub legs) == observed(the exact direct map): "
          f"|Δ| = {abs(a_comp - a_obs):.2e}")
    # ...and a MIS-pairing must break it, so the equality above has teeth: a
    # leg to a DIFFERENT target (same hub basis, different Ω) must not
    # reproduce the direct map's â.
    om_c = np.linalg.qr(rng.standard_normal((k, k)))[0]
    leg_b_wrong = TransportMap(kind="proc", src_norm=1.0, tgt_norm=1.0,
                               va=va.astype(np.float32),
                               vb=vb_b.astype(np.float32),
                               omega=om_c.astype(np.float32), scale=1.0)
    a_wrong = rcp.composed_exchange_rate(leg_a, leg_b_wrong, v_a, v_b)
    check(abs(a_wrong - a_obs) > 1e-3,
          f"the WRONG target leg breaks the identity (|Δ| = "
          f"{abs(a_wrong - a_obs):.2e}) — the fixture has teeth")
    # the adjoint re-orientation used for pair-fits-as-legs is exact. `leg_a`
    # is hub→A, so `adjoint(leg_a)` is A→hub and its REV pass is leg_a's FWD
    # pass — the identity `_oriented_leg` relies on when a pair fit is banked
    # in the model→hub ordering.
    adj = rcp.adjoint_map(leg_a)
    v_hub = rng.standard_normal(d_hub)
    v_hub /= np.linalg.norm(v_hub)
    check(float(np.abs(adj.transport(v_hub, direction="rev")
                       - leg_a.transport(v_hub, direction="fwd")).max()) < 1e-6,
          "adjoint_map(tm).rev == tm.fwd — the re-orientation of record")
    check(float(np.abs(adj.transport(v_a, direction="fwd")
                       - leg_a.transport(v_a, direction="rev")).max()) < 1e-6,
          "adjoint_map(tm).fwd == tm.rev — the other half of the identity")
    # and the whole re-orientation, end to end: composing through a pair fit
    # banked as A→hub must give the SAME â_comp as composing through hub→A.
    a_reoriented = rcp.composed_exchange_rate(
        rcp.adjoint_map(rcp.adjoint_map(leg_a)), leg_b, v_a, v_b)
    check(abs(a_reoriented - a_comp) < 5e-6,
          f"a leg re-oriented through the adjoint scores identically "
          f"(|Δ| = {abs(a_reoriented - a_comp):.2e})")

    print("── 2. the reverse-direction reading of one unordered fit object")
    a_fwd = exchange_rate(direct, v_a, v_b, direction="fwd")
    a_rev = exchange_rate(direct, v_b, v_a, direction="rev")
    check(np.isfinite(a_fwd) and np.isfinite(a_rev),
          "both orderings read out of ONE banked object (§8.3 construction)")

    print("── 3. Holm, on a fixture with a hand-computable answer")
    rows = holm([("a", 0.001), ("b", 0.008), ("c", 0.04), ("d", 0.6)], 0.05)
    by = {r.label: r for r in rows}
    check(abs(by["a"].p_holm - 0.004) < 1e-12, "Holm: 4 x .001 = .004")
    check(abs(by["b"].p_holm - 0.024) < 1e-12, "Holm: 3 x .008 = .024")
    check(abs(by["c"].p_holm - 0.08) < 1e-12, "Holm: 2 x .04 = .08")
    check(abs(by["d"].p_holm - 0.6) < 1e-12, "Holm: 1 x .6 = .6")
    check([r.rejected_at_alpha for r in rows] == [True, True, False, False],
          "Holm step-down rejects {a, b} only at alpha = .05")
    mono = holm([("x", 0.02), ("y", 0.0001), ("z", 0.019)], 0.05)
    adj = {r.label: r.p_holm for r in mono}
    check(adj["y"] <= adj["z"] <= adj["x"],
          "Holm adjusted p is MONOTONE in the raw p (the step-down property)")

    print("── 4. the sign test, on tie handling")
    t = sign_test("ties", [1.0, 1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                  "A", "B")
    check(t.n_shared == 5 and t.n_ties == 3 and t.n_effective == 2,
          f"ties DROPPED, n restated: n_shared={t.n_shared} ties={t.n_ties} "
          f"n_eff={t.n_effective}")
    check(t.n_a_better == 0 and t.n_b_better == 2 and t.winner == "B",
          "direction is by |e|, SMALLER is better")
    check(abs(t.p_two_sided - 0.5) < 1e-12,
          f"exact two-sided p at 0/2 = .5 (got {t.p_two_sided})")
    all_tied = sign_test("all ties", [1.0, 1.0], [1.0, 1.0], "A", "B")
    check(all_tied.n_effective == 0 and all_tied.p_two_sided == 1.0
          and all_tied.winner is None,
          "an all-ties comparison is p = 1 with n_eff = 0, never a division "
          "by zero and never a winner")
    t8 = sign_test("8-0", [0.0] * 8, [1.0] * 8, "A", "B")
    check(abs(t8.p_two_sided - 2.0 * 0.5 ** 8) < 1e-12,
          f"exact two-sided p at 8/0 = 2^-7 = {2.0 * 0.5 ** 8} "
          f"(got {t8.p_two_sided})")
    t240 = sign_test("240-0", [0.0] * 240, [1.0] * 240, "A", "B")
    check(t240.p_two_sided > 0.0 and np.isfinite(t240.p_two_sided),
          f"the log-space binomial does not underflow at n = 240 "
          f"(p = {t240.p_two_sided:.3e})")

    print("── 5. rank statistics")
    check(abs(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-12,
          "Spearman on a monotone map = 1")
    check(abs(spearman([1, 2, 3, 4], [10, 9, 8, 7]) + 1.0) < 1e-12,
          "Spearman on a reversal = -1")
    check(abs(rankdata([3.0, 1.0, 1.0, 2.0]) - np.array([4.0, 1.5, 1.5, 3.0])).max()
          < 1e-12, "average ranks under ties")
    rng2 = np.random.default_rng(7)
    z = rng2.standard_normal(40)
    x = rng2.standard_normal(40)
    y = rng2.standard_normal(40)
    const = np.ones(40)
    check(abs(partial_spearman(x, y, const) - spearman(x, y)) < 1e-9,
          "partial rho with a CONSTANT control == plain Spearman")
    # a shared driver: both columns are z plus small noise, so the marginal
    # correlation is high and the partial must collapse.
    xa = z + 0.05 * rng2.standard_normal(40)
    ya = z + 0.05 * rng2.standard_normal(40)
    marginal = spearman(xa, ya)
    partial_val = partial_spearman(xa, ya, z)
    check(marginal > 0.9 and abs(partial_val) < 0.4,
          f"a shared driver is absorbed: marginal rho = {marginal:.3f} -> "
          f"partial rho = {partial_val:.3f}")
    try:
        partial_spearman(x, z, z)
        degenerate_refused = False
    except ReadsError:
        degenerate_refused = True
    check(degenerate_refused,
          "partial rho(x, z | z) REFUSES — the control explains the column "
          "outright, so the partial is undefined rather than float noise")

    print("── 6. the permutation null is reproducible from its NAME")
    a = _named_rng("fixture").permutation(20)
    b = _named_rng("fixture").permutation(20)
    check(bool((a == b).all()), "the same seed name gives the same stream")
    c = _named_rng("fixture-2").permutation(20)
    check(not bool((a == c).all()), "a different name gives a different stream")

    print("── 7. the deterministic emitter")
    payload = {"b": 2, "a": [1, 2, {"z": 1, "y": 2}]}
    check(canonical_json(payload) == canonical_json(dict(reversed(list(
        payload.items())))), "key order does not change the bytes")

    print("── 8. the sanitizing relativizer")
    repo = Path("/somewhere/a-user/proj")
    got = relativize({"p": "/somewhere/a-user/proj/staging/x.json",
                      "l": ["/somewhere/a-user/proj/y", "staging/z"],
                      "n": 3, "root": "/somewhere/a-user/proj"}, repo)
    check(got["p"] == "staging/x.json", "an absolute path under the repo goes relative")
    check(got["l"] == ["y", "staging/z"], "lists are walked; relatives untouched")
    check(got["n"] == 3, "non-strings pass through")
    check(got["root"] == ".", "the bare root becomes '.'")
    check("a-user" not in json.dumps(got), "no username survives")

    print("── 9. tier membership on a constructed fixture")
    # leader clearly best; one candidate significantly worse but INSIDE delta
    # (must be quoted practically-equivalent, NOT expelled); one worse and
    # outside delta (must be expelled).
    n = 200
    base = np.linspace(0.001, 0.02, n)
    def fake(hub: str, offset: float) -> HubContextScore:
        errs = [SlotError(ordinal=i, prediction_id=f"p{i}", arm="native",
                          a_comp=0.0, a_obs=0.0, error=float(base[i] + offset),
                          abs_error=float(base[i] + offset),
                          observed_direction="fwd", observed_fit_pair_id="x")
                for i in range(n)]
        return HubContextScore(
            hub=hub, hub_site=0, context="fixture", split="full",
            family=RANK_OF_RECORD, leg_source="hub-leg-wave",
            n_slots_realized=n, n_slots_basis=n,
            median_abs_error=float(np.median([e.abs_error for e in errs])),
            mean_abs_error=float(np.mean([e.abs_error for e in errs])),
            arm_counts={"native": n}, hub_basis_max_dev=0.0, errors=errs)
    fixture = {"L": fake("L", 0.0), "E": fake("E", 0.0005),
               "X": fake("X", 0.02), "Y": fake("Y", 0.0004),
               "Z": fake("Z", 0.03)}
    tm = tier_membership(fixture, Context(split="full", family=RANK_OF_RECORD),
                         candidates=("L", "E", "X", "Y", "Z"))
    check(tm.leader == "L", "the leader is the smallest median")
    check("X" in tm.expelled and "Z" in tm.expelled,
          f"candidates beyond delta are expelled: {tm.expelled}")
    check("E" in tm.practically_equivalent and "E" not in tm.expelled,
          "a significant-but-sub-delta candidate is quoted “statistically "
          "distinguishable, practically equivalent” and NOT expelled")

    print("── 10. the population filter re-aggregates, never re-measures")
    mixed_errors = [
        SlotError(ordinal=i, prediction_id=f"p{i}",
                  arm=("native" if i < 6 else "raw"), a_comp=0.0, a_obs=0.0,
                  error=float(i), abs_error=float(i),
                  observed_direction="fwd", observed_fit_pair_id="x")
        for i in range(10)]
    mixed = HubContextScore(
        hub="H", hub_site=0, context="fixture", split="full",
        family=RANK_OF_RECORD, leg_source="hub-leg-wave", n_slots_realized=10,
        n_slots_basis=10, median_abs_error=4.5, mean_abs_error=4.5,
        arm_counts={"native": 6, "raw": 4}, hub_basis_max_dev=0.0,
        errors=mixed_errors)
    native_only = filter_scores({"H": mixed}, ["H"], arm="native")["H"]
    check(native_only.n_slots_realized == 6
          and native_only.arm_counts == {"native": 6},
          "the arm filter keeps exactly the arm's slots")
    check(abs(native_only.median_abs_error - 2.5) < 1e-12,
          f"the filtered median is the median of the KEPT slots "
          f"(got {native_only.median_abs_error})")
    check([e.abs_error for e in native_only.errors]
          == [float(i) for i in range(6)],
          "the kept per-slot errors are the SAME objects, not recomputed")
    dropped = filter_scores({"H": mixed}, ["H"], arm="ridge")
    check(dropped == {},
          "a hub with no slot in the requested arm is DROPPED from the "
          "population rather than carried with an empty median")

    print("── 11. the rank guard arithmetic")
    row = FitRow(site_pair="a->b", arm="native", family="proc_k256",
                 n_train=460, n_test=140, r2=0.5, valid=True, k_effective=256)
    check(row.rank_guard_ok, "k256 clears the guard at per-half n_train = 460")
    row2 = FitRow(site_pair="a->b", arm="native", family="proc_k512",
                  n_train=460, n_test=140, r2=0.5, valid=True, k_effective=512)
    check(not row2.rank_guard_ok,
          "k512 FAILS the guard per-half — the excluded family, as frozen")

    print("── 12. the P6-INPUTS reader, on a FIXTURE artifact")
    import tempfile

    def p6_fixture(**over: Any) -> dict[str, Any]:
        rng6 = np.random.default_rng(20260804)
        hubs = []
        for i, key in enumerate(("h1", "h2", "h3")):
            lam = np.sort(rng6.random(12) + 0.01)[::-1]
            hubs.append({
                "key": key, "site": 10 + i, "arm": "native",
                "checkpoint_identity": "instruct",
                "n_eigenvalues": int(lam.size), "n_train_rows": 12,
                "hidden_dim": 64, "eigenvalues": [float(x) for x in lam],
                "participation_ratio": float((lam.sum() ** 2) / (lam ** 2).sum()),
            })
        doc: dict[str, Any] = {
            "artifact": P6_ARTIFACT_KIND,
            "definition": f"census pre-statement §3 P6: {P6_CENSUS_FORMULA}",
            "corpus_manifest_sha256": "c0ffee", "splits_sha256": "5171175",
            "n_hubs": 3, "n_train_rows": [12], "effective_num_threads": 8,
            "hubs": hubs,
        }
        doc.update(over)
        return doc

    with tempfile.TemporaryDirectory() as td6:
        good = Path(td6)/"P6.json"
        good.write_text(json.dumps(p6_fixture()))
        loaded = load_p6_inputs(good)
        check(loaded.n_hubs == 3 and set(loaded.values) == {"h1", "h2", "h3"},
              "the P6 reader loads every hub row")
        check(loaded.recomputed_max_abs_delta <= P6_RECOMPUTE_TOL,
              f"every PR RE-DERIVES from its own banked eigenvalues "
              f"(max |Δ| {loaded.recomputed_max_abs_delta:.2e})")
        check(loaded.sites == {"h1": 10, "h2": 11, "h3": 12},
              "sites travel with the column")

        #  a PR that does not match its own spectrum must REFUSE, not round
        bad = p6_fixture()
        bad["hubs"][1]["participation_ratio"] += 1e-6
        liar = Path(td6)/"liar.json"
        liar.write_text(json.dumps(bad))
        try:
            load_p6_inputs(liar)
            check(False, "a PR contradicting its own spectrum REFUSES")
        except ReadsError:
            check(True, "a PR contradicting its own spectrum REFUSES")

        #  a different formula is a different slate member
        wrong = Path(td6)/"wrong.json"
        wrong.write_text(json.dumps(
            p6_fixture(definition="PR = trace(C)^2 / trace(C@C), whitened")))
        try:
            load_p6_inputs(wrong)
            check(False, "a NON-census formula REFUSES")
        except ReadsError:
            check(True, "a NON-census formula REFUSES")

        #  an artifact of another kind is refused outright
        alien = Path(td6)/"alien.json"
        alien.write_text(json.dumps(p6_fixture(artifact="something-else/v9")))
        try:
            load_p6_inputs(alien)
            check(False, "an artifact of another KIND refuses")
        except ReadsError:
            check(True, "an artifact of another KIND refuses")

    print("── 13. §7's P2/P6 cluster rule with BOTH members present")
    #  Two members inside ONE cluster: the null's statistic must be the maximum
    #  over both, so a member is never cheaper to clear than the pair it is
    #  clustered with. Built anti-correlated, as §7 describes them (ρ ≈ −.93).
    rng7 = np.random.default_rng(760876)
    q7 = list(rng7.random(21))
    p2_7 = [-x + 0.02 * rng7.standard_normal() for x in q7]
    p6_7 = [-v for v in p2_7]
    both = max_statistic_null(
        q7, {"P2-compressibility": p2_7, "P6-participation-ratio": p6_7},
        {"P2/P6": ["P2-compressibility", "P6-participation-ratio"]},
        n_perm=2000, seed_name="selftest/cluster/both")
    check(sorted(both.members) == ["P2-compressibility",
                                   "P6-participation-ratio"],
          "both cluster members enter the null")
    check(abs(both.observed_max - max(both.observed_abs_rho.values())) < 1e-12,
          "the cluster's observed statistic is the max over its members")
    alone = max_statistic_null(
        q7, {"P2-compressibility": p2_7}, {"P2/P6": ["P2-compressibility"]},
        n_perm=2000, seed_name="selftest/cluster/both")
    check(both.p_max_statistic["P2-compressibility"]
          >= alone.p_max_statistic["P2-compressibility"],
          "a second member in the cluster never makes P2's p SMALLER "
          f"({both.p_max_statistic['P2-compressibility']:.4f} vs "
          f"{alone.p_max_statistic['P2-compressibility']:.4f})")
    #  the grouping itself is inert in a max-statistic null; proving it here
    #  keeps the report's claim honest rather than asserted.
    split_clusters = max_statistic_null(
        q7, {"P2-compressibility": p2_7, "P6-participation-ratio": p6_7},
        {"P2": ["P2-compressibility"], "P6": ["P6-participation-ratio"]},
        n_perm=2000, seed_name="selftest/cluster/both")
    check(split_clusters.p_max_statistic == both.p_max_statistic,
          "grouping the two as ONE cluster vs TWO changes no p — the "
          "max-statistic null is correlation-aware by construction")

    print(f"\n{'ALL CHECKS PASS' if not failures else str(len(failures)) + ' FAILURE(S)'}")
    for label in failures:
        print("  FAILED:", label)
    return 0 if not failures else 1


# ═══════════════════════════════════════════════════════════════════ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=("webtext-v3 §6 invariance reads + §7 race/battery/roster "
                     "reads. UNSTAMPED ARITHMETIC — declares nothing."))
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help="repo root holding staging/ and outputs/")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the reads artifacts land "
                             "(default <repo>/staging/webtext-v3-reads)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the staging manifest checks (NOT for a "
                             "record run — the brief binds them)")
    parser.add_argument("--build-twice", action="store_true",
                        help="build the artifact twice and prove the bytes "
                             "identical")
    parser.add_argument("--p6-inputs", type=Path, default=None,
                        help="the node-built P6-INPUTS artifact. Supplied: P6 "
                             "enters the slate and the §7 P2/P6 cluster holds "
                             "both members. Omitted: P6 stays SKIPPED with its "
                             "blocker, exactly as before.")
    parser.add_argument("--battery-out", type=Path, default=None,
                        help="also write a standalone battery artifact here "
                             "(+ sidecar); the reads/race artifacts are "
                             "unaffected")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    out_dir = args.out_dir or (args.repo/"staging"/"webtext-v3-reads")
    out_dir.mkdir(parents=True, exist_ok=True)
    param_cache = out_dir/"PARAM-COUNTS-race-candidates.json"

    run = run_reads(args.repo, out_dir, verify=not args.no_verify,
                    param_cache=param_cache, p6_inputs=args.p6_inputs)
    payload = run.model_dump(mode="json")
    reads_path = out_dir/"READS-webtext-v3-2026-08-04.json"
    digest, sidecar = write_deterministic(payload, reads_path)
    logger.info("reads artifact: %s sha256 %s", reads_path, digest)

    race_payload = {
        "artifact": "webtext-v3-race/v1",
        "STATUS": run.STATUS,
        "frozen_ref": run.frozen_ref,
        "basis": run.populations["race_basis"],
        "candidates": list(RACE_SET),
        "tiers": {k: v.model_dump(mode="json") for k, v in run.tiers.items()},
        "invariance": run.invariance.model_dump(mode="json"),
        "crowning": run.crowning.model_dump(mode="json"),
        "battery": run.battery.model_dump(mode="json"),
        "clause_readings": [r.model_dump(mode="json")
                            for r in run.clause_readings],
        "roster_race": run.roster_race.model_dump(mode="json"),
        "separation_floor": run.separation_floor.model_dump(mode="json"),
        "population_reads": [r.model_dump(mode="json")
                             for r in run.population_reads],
        "c2": run.c2.model_dump(mode="json"),
        "c4": run.c4.model_dump(mode="json"),
        "guard_quotable_population": run.guard_quotable_population,
        "flags": run.flags,
    }
    race_path = out_dir/"RACE-webtext-v3-2026-08-04.json"
    race_digest, _ = write_deterministic(race_payload, race_path)
    logger.info("race artifact: %s sha256 %s", race_path, race_digest)

    i5_payload = {
        "artifact": "webtext-v3-i5-realized-slots/v1",
        "STATUS": run.i5.STATUS,
        "vintage_rule": run.i5.vintage_rule,
        "i5": run.i5.model_dump(mode="json"),
    }
    i5_path = out_dir/"I5-REALIZED-SLOTS-webtext-v3-2026-08-04.json"
    i5_digest, _ = write_deterministic(i5_payload, i5_path)
    logger.info("I5 realized-slot list: %s sha256 %s", i5_path, i5_digest)

    battery_digest = None
    if args.battery_out is not None:
        battery_payload = {
            "artifact": "webtext-v3-battery-r2/v1",
            "STATUS": run.STATUS,
            "frozen_ref": run.frozen_ref,
            "support_rule": run.battery.support_rule,
            "battery": run.battery.model_dump(mode="json"),
            "guard_quotable_population": run.guard_quotable_population,
        }
        battery_digest, _ = write_deterministic(
            relativize(battery_payload, args.repo), args.battery_out)
        logger.info("battery artifact: %s sha256 %s", args.battery_out,
                    battery_digest)

    if args.build_twice:
        # The second build runs under the IDENTICAL settings, verification
        # included: a build-twice proof that changed a flag between the two
        # passes would prove the wrong thing (the first attempt did exactly
        # that and the manifest block differed — caught, not shipped).
        second = run_reads(args.repo, out_dir, verify=not args.no_verify,
                           param_cache=param_cache, p6_inputs=args.p6_inputs)
        again = canonical_json(second.model_dump(mode="json"))
        identical = sha256_of_text(again) == digest
        print(f"BUILD-TWICE: {'BYTE-IDENTICAL' if identical else 'DIFFERENT'} "
              f"({digest} vs {sha256_of_text(again)})")
        if not identical:
            return 1

    summary: dict[str, Any] = {
        "reads_artifact": str(reads_path), "reads_sha256": digest,
        "race_artifact": str(race_path), "race_sha256": race_digest,
        "i5_artifact": str(i5_path), "i5_sha256": i5_digest,
        "sidecar": str(sidecar)}
    if battery_digest is not None:
        summary["battery_artifact"] = str(args.battery_out)
        summary["battery_sha256"] = battery_digest
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
