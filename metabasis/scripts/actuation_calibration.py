"""The actuation acceptance gate — ALIGNMENT != ACTUATION, as code (BRIEF §4).

`BRIEF-behavioral-phase-2026-07-29.md` (sha `475bc2a8…`) §4 is this module's
specification verbatim; §13 rulings 2 and 9 freeze the SSM and gpt2-xl branches.

WHY THIS GATE EXISTS, in the inherited words the brief restates:

    "A site must demonstrate actuation with a KNOWN-GOOD lever before any
     transported object is tested there... at the site that read best
     (cosine-visible), the known-good lever moved behavior not at all; at the site
     that read worse, it actuated (73.0% vs 91.2% sham, p=.045). **Cosine
     visibility at a site is not functional efficacy.**"

So this is a GATE, not a warm-up: a node that fails it does not get a
transported-write cell. The gate runs per node, at its site OF RECORD, with the
node's OWN native entropy-gradient vector against the node's OWN native random
band — the control that asks whether **the site actuates**. It is deliberately NOT
the transported band, which asks whether **the transport carries** (§5). Both bands
exist in the campaign; §4.1 keeps them distinct in name (`Rband*` native vs
`gRband*` transported) and in stamp, because conflating them is the fastest way to
make a null uninterpretable.

WHAT THIS MODULE DOES AND DOES NOT DO. It builds the §4.1 cell plan, resolves the
site from the LIVE registries, applies the §4.2 acceptance criteria, and returns a
verdict. It never scores a transported cell, never generates, and never picks a
site: **no node's site is re-chosen on behavioral evidence** (§4.2), because sites
come from curves and Luxia's rulings and a behavioral hunt would be site-fishing
against the frozen §4 ordering. `assert_no_site_fishing` is that sentence as code.

TWO LABELS THAT MUST NEVER MERGE. The brief uses two, for two different states,
and this module keeps them apart because they license different claims:

  * `SITE_UNCALIBRATED` ("SITE-UNCALIBRATED", §4.2) — the gate RAN and the site
    FAILED, at the site of record and at its registered robustness site if it has
    one. This is a RESULT ABOUT THE NODE, reported in the column table, not a gap
    in the campaign.
  * `UNCALIBRATED_SITE` ("UNCALIBRATED-SITE", §4.4 / ruling 2) — the gate could not
    be APPLIED, because the node has no known-good lever (the SSM case, when its
    calibration-only lever fails the FD gate). Every such cell is filed
    descriptively and **a floor is not claimed as a regime boundary**.

CPU self-test (no weights, no GPU, no data tree):

    python -m metabasis.scripts.actuation_calibration --selftest

RAKE M44 — THE CONFIGURATION MATRIX IS THE MERGE BAR. This selftest is verified in
all four cells of {torch, no torch} x {data tree, no data tree} and every cell
must exit 0 with either real passes or NAMED skips. Two environments differ in ways
a single run cannot see: the repo `.venv` has NO torch (numpy/pydantic/scipy only)
while `/usr/bin/python` has torch + transformers, and the data tree is gitignored so
a fresh worktree has none. A bare `import torch` at a point of use passes in one
environment and CRASHES in another, and a crash is indistinguishable from a failure
while saying less. Every torch-dependent BLOCK here therefore branches on one
availability probe and degrades to a named skip, counted in the tail so coverage is
reported per configuration rather than inferred.

Desk-side read of a live registry table (GPU-free, §4.3's preflight shape):

    python -m metabasis.scripts.actuation_calibration --who-can-calibrate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from metabasis.scripts.run_behavioral_cells import (
    BRIEF_OF_RECORD, BRIEF_SHA256, CELL_ID_TEMPLATE, DOSE_LADDER, GRADE_LINE,
    N_PER_CELL, SCORING_DOSES, BehavioralHarnessError, CellSpec,
    NativeVectorUnavailable, SiteNotOfRecord, apply_dose_ladder, baseline_cell)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("actuation_calibration")

# ---------------------------------------------------------------- frozen constants
#: §4.2(a): Spearman ρ(dose, entropy_rise) across the FULL SIGNED 6-dose ladder.
#: The floor is the SCOPE clause of the banked reverse leg: monotone across the
#: whole signed ladder, not just the positive half.
SPEARMAN_RHO_FLOOR = 0.80
#: §4.2(b): outside the node's own random band at ≥4 of 6 doses AND at both |0.3|.
#: The banked precedent scored band comparison at ≥5/6; 4/6-plus-both-extremes is
#: the same strictness with the SMALL doses, where the effect is by design nearly
#: nil, not forced to carry the verdict.
BAND_SEPARATION_MIN_DOSES = 4
BAND_SEPARATION_REQUIRED_DOSES: tuple[float, ...] = SCORING_DOSES
#: §4.2(c): the banked degeneracy guard (`trait_probe._coherence`), floor .45 of
#: record, read at the scoring dose. A "rise" produced by collapse is not actuation.
COHERENCE_FLOOR = 0.45
SCORING_DOSE = 0.3
#: §4.1: 3 members × 6 doses. The node's OWN native band, at the same site and doses.
N_BAND_MEMBERS = 3
#: §4.1's cell structure, as arithmetic rather than as a table to be retyped.
N_CALIBRATION_BASELINE_CELLS = 1
N_CALIBRATION_SIGNAL_CELLS = len(DOSE_LADDER)                       # 6
N_CALIBRATION_BAND_CELLS = N_BAND_MEMBERS * len(DOSE_LADDER)        # 18
N_CALIBRATION_CELLS = (N_CALIBRATION_BASELINE_CELLS
                       + N_CALIBRATION_SIGNAL_CELLS
                       + N_CALIBRATION_BAND_CELLS)                  # 25

#: §4.2 remedy (i): re-calibrate at the node's ROBUSTNESS SITE if it has one
#: REGISTERED — a registered site, not a new one. These two are the only registered
#: robustness sites in the campaign (gemma3-27b L38 ⋆ / L41; llama-3.1-70b L37 ⋆ /
#: L43); the values are cross-checked against `fit_transport_maps.SITES` at
#: resolution time rather than trusted here.
ROBUSTNESS_SITES: dict[str, int] = {
    "gemma3-27b": 41,
    "llama-3.1-70b-instruct": 43,
}

#: §4.2's label for a site that RAN the gate and FAILED it (both sites, if two).
SITE_UNCALIBRATED = "SITE-UNCALIBRATED"
#: §4.4 / ruling 2's label for a site where the gate could not be APPLIED at all.
#: Deliberately a DIFFERENT string from `SITE_UNCALIBRATED`: a reader must be able
#: to tell "this site was tested and does not actuate" from "this site was never
#: testable", because only the first is a result about the node.
UNCALIBRATED_SITE = "UNCALIBRATED-SITE"

#: ruling 2: the SSM's calibration-only lever is banked under a DISTINCT key, so it
#: can never be mistaken for a target vector by a reader that resolves by key.
CALIBRATION_LEVER_KEY = "calibration_lever_entropy_gradient_L{site}"
#: §4.4's FD gate on that lever — the directional finite-difference gate, verbatim:
#:   |⟨g,v⟩ − (S(+εv) − S(−εv))/2ε| / |dS| < .05
SSM_FD_GATE_TOLERANCE = 0.05
SSM_FD_GATE_CRITERION = (
    "|<g,v> - (S(+eps*v) - S(-eps*v))/(2*eps)| / |dS| < "
    f"{SSM_FD_GATE_TOLERANCE} (the directional finite-difference gate, §4.4)")
#: ruling 2's standing prohibitions on the calibration-only lever. Carried as text
#: because they are what makes the lever admissible at all, and a stamp that does
#: not say them is a stamp that lets the lever drift into a star claim.
CALIBRATION_LEVER_PROHIBITIONS = (
    "calibration instrument ONLY (ruling 2): never enters a star system, never "
    "hosts a portability coefficient, no star row, never a target vector. Banked "
    "under a distinct key so no reader that resolves by key can mistake it.")

#: ruling 9: gpt2-xl calibrates at ALL THREE built experiment sites and the
#: CALIBRATION evidence feeds Luxia's deferred site ruling. Legitimate under §4
#: because calibration is a gate on a SITE, not a read of the transported object;
#: the transported column then fires only at the ruled site.
GPT2XL_CALIBRATION_SITES: tuple[int, ...] = (7, 26, 47)
GPT2XL_KEY = "gpt2-xl"

#: §4.3 / §4.1: base-model nodes calibrate in the RAW arm only (the arm-consistency
#: rule) — the transported column into any base node rides the raw arm, so the
#: calibration must too. Instruct nodes calibrate in the arm their transported cells
#: will use. Resolved from the roster where the node is registered; this tuple is the
#: brief's own named list and is CROSS-CHECKED against the roster, never trusted over
#: it (`olmo2-7b` is a carried banked node with no roster row at all).
BASE_ARM_NODES: tuple[str, ...] = ("olmo2-7b", "pythia-6.9b", "gpt2-xl")

Verdict = Literal["PASS", "FAIL", "DEGENERATE"]


# ---------------------------------------------------------------- error taxonomy
class ActuationGateNotPassed(BehavioralHarnessError):
    """§9 item 5: FAIL or DEGENERATE at a node → no transported cells at that site.

    DEGENERATE is included deliberately: §4.2 reports it as its own state and never
    silently folds it into PASS, and a rise produced by collapse must not license a
    transported-write cell any more than no rise at all.
    """


class SiteFishingRefused(BehavioralHarnessError):
    """§4.2: a site was proposed on BEHAVIORAL evidence. Refused by construction.

    Sites come from curves and Luxia's rulings. Re-choosing one on behavioral
    evidence would front-run the frozen §4 ordering, which is the one thing the
    gate's own legitimacy rests on.
    """


class RegistryGap(BehavioralHarnessError):
    """A NAMED registry gap, not an omission — reported, never worked around.

    `olmo2-7b` (base) is the live instance: it has a banked entropy-gradient vector
    and a `SITES` grid (12, 16, 20, 24) but NO `SITE_OF_RECORD` row, so it cannot
    calibrate until its site is registered or ruled (§4.3).
    """


class CalibrationLeverRefused(BehavioralHarnessError):
    """A calibration-only lever was asked to do something ruling 2 forbids."""


# ---------------------------------------------------------------- typed records
class ActuationCriteria(BaseModel):
    """§4.2's three criterion values — the numbers a §5 stamp must carry.

    Filed as VALUES beside their verdicts, never as bare booleans: §4's stamp
    requirement is "job id + verdict + the three criterion values", so a downstream
    reader can re-apply the thresholds without re-running anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: (a) dose-ordering
    spearman_rho_dose_vs_rise: Optional[float]
    doses_present: int
    sign_flips_through_zero: bool
    dose_ordering_passes: bool
    #: (b) band separation, against the node's OWN native random band
    outside_band_doses: int
    outside_band_of: int
    outside_band_at_both_extremes: bool
    per_dose_outside_band: dict[str, bool]
    band_separation_passes: bool
    #: (c) coherence floor at the scoring dose
    coherence_at_scoring_dose: Optional[float]
    coherence_floor: float = COHERENCE_FLOOR
    coherence_passes: bool = False
    #: the thresholds this evaluation applied, so a stamp is self-describing
    rho_floor: float = SPEARMAN_RHO_FLOOR
    min_outside_band_doses: int = BAND_SEPARATION_MIN_DOSES
    scoring_dose: float = SCORING_DOSE


class ActuationCalibrationResult(BaseModel):
    """One node's §4 verdict at one site, with everything a §5 stamp names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site: int
    site_role: Literal["site_of_record", "robustness_site"]
    arm: str
    #: the Heimdall job that produced the cells. Named in every §5 stamp (§4.2's
    #: closing requirement), so a transported read points at its own gate's run.
    job_id: Optional[str]
    lever_key: str
    lever_is_calibration_only: bool
    band_family: Literal["Rband"] = "Rband"
    criteria: ActuationCriteria
    verdict: Verdict
    verdict_rationale: str
    grade: str = GRADE_LINE

    @property
    def licenses_transported_cells(self) -> bool:
        """§4.2/§9 item 5: only a PASS licenses transported-write cells here."""
        return self.verdict == "PASS"


class CalibrationCellPlan(BaseModel):
    """§4.1's cell structure for one (node, site, arm) — 25 cells, 2,000 generations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site: int
    arm: str
    lever_key: str
    band_keys: tuple[str, ...]
    baseline: CellSpec
    signal_cells: tuple[CellSpec, ...]
    band_cells: tuple[CellSpec, ...]
    alphas: dict[str, float]
    per_token_median_resid_norm: float
    n_per_cell: int = N_PER_CELL

    @property
    def cells(self) -> tuple[CellSpec, ...]:
        return (self.baseline,) + self.signal_cells + self.band_cells

    @model_validator(mode="after")
    def _matches_the_table(self) -> "CalibrationCellPlan":
        if len(self.signal_cells) != N_CALIBRATION_SIGNAL_CELLS:
            raise ValueError(
                f"§4.1 wants {N_CALIBRATION_SIGNAL_CELLS} signal cells (the signed "
                f"ladder), got {len(self.signal_cells)}")
        if len(self.band_cells) != N_CALIBRATION_BAND_CELLS:
            raise ValueError(
                f"§4.1 wants {N_CALIBRATION_BAND_CELLS} band cells "
                f"({N_BAND_MEMBERS} members × {len(DOSE_LADDER)} doses), got "
                f"{len(self.band_cells)}")
        if len(self.cells) != N_CALIBRATION_CELLS:
            raise ValueError(
                f"§4.1's calibration block is {N_CALIBRATION_CELLS} cells, got "
                f"{len(self.cells)}")
        if any(c.band_family != "Rband" for c in self.band_cells):
            raise ValueError(
                "§4.1: the calibration control is the node's OWN NATIVE random band "
                "(`Rband*`), never the transported band (`gRband*`) — the two ask "
                "different questions and conflating them makes a null uninterpretable")
        return self

    @property
    def total_generations(self) -> int:
        return len(self.cells) * self.n_per_cell


class CalibrationSiteResolution(BaseModel):
    """The site this node calibrates at, resolved from the LIVE registries.

    §4.3: "the enactor's first act is a vector-inventory preflight that RE-DERIVES
    this table from disk rather than trusting it", because vectors are landing
    nightly. This record is the resolution half; `VectorInventoryRow` is the
    from-disk half.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site_of_record: Optional[int]
    robustness_site: Optional[int]
    fixed_fit_grid: tuple[int, ...]
    registries_agree: bool
    #: gpt2-xl only (ruling 9): all three built experiment sites, no ⋆.
    deferred_multi_site: tuple[int, ...] = ()
    arm: str = "native"
    arm_rule: str = ""
    note: str = ""


class VectorInventoryRow(BaseModel):
    """§4.3's from-disk preflight row for one node — re-derived, never trusted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site: Optional[int]
    vector_path: Optional[str]
    vector_present: bool
    vector_sha256: Optional[str]
    fd_gate_present: bool
    fd_gate_passed: Optional[bool]
    corpus_vintage: Optional[str]
    corpus_vintage_is_v21: Optional[bool]
    ready: bool
    blocking_reason: Optional[str]


class SSMLeverDecision(BaseModel):
    """Ruling 2's SSM branch: a calibration-only native lever, or UNCALIBRATED-SITE.

    Ruling 2 (on the desk's recommendation): build each SSM's own native
    entropy-gradient vector at its scanned site PURELY as a calibration instrument,
    FD-gated like every other build, never entering a star system, never hosting a
    portability coefficient, no star row — with **automatic fallback to
    UNCALIBRATED-SITE labeling if the FD gate fails on either SSM**.

    Why this is the only option under which an SSM result is publishable: the SSM
    boundary probe's whole value is that "hit or floor is unambiguous", and a floor
    at an UNCALIBRATED site is not unambiguous at all — it is indistinguishable
    from a dead site. The fallback is therefore not a consolation; it is the
    labeling that keeps the ambiguous case from being read as a boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site: int
    lever_key: str
    fd_gate_criterion: str = SSM_FD_GATE_CRITERION
    fd_gate_passed: bool
    fd_gate_rel_error: Optional[float] = None
    #: True iff the lever exists AND passed — the only state in which §4's gate can
    #: be applied verbatim to this node.
    gate_applicable: bool
    label: Optional[str] = None
    prohibitions: str = CALIBRATION_LEVER_PROHIBITIONS
    rationale: str = ""

    @model_validator(mode="after")
    def _fallback_is_labeled(self) -> "SSMLeverDecision":
        if not self.gate_applicable and self.label != UNCALIBRATED_SITE:
            raise ValueError(
                f"ruling 2: an SSM whose FD gate fails must be labeled "
                f"{UNCALIBRATED_SITE!r} automatically — an unlabeled inapplicable "
                "gate is exactly the ambiguity the labeling exists to prevent")
        if self.gate_applicable and self.label is not None:
            raise ValueError(
                "a lever that PASSED its FD gate carries no UNCALIBRATED label")
        return self


class RemedyLadder(BaseModel):
    """§4.2's named remedies, in order — and the honest end of the ladder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    attempted: tuple[str, ...]
    next_action: str
    #: set once every registered site has failed
    column_label: Optional[str] = None
    licenses_transported_cells: bool = False
    note: str = ""


# ---------------------------------------------------------------- §4.2 criteria
def dose_ordering(doses: Sequence[float], rises: Sequence[float]
                  ) -> tuple[Optional[float], bool, bool]:
    """§4.2(a): (spearman_rho, sign_flips_through_zero, passes).

    ρ is computed across the FULL SIGNED ladder — not the positive half — which is
    the SCOPE clause of the banked reverse leg. "Sign flipping through zero" is
    checked as its own fact rather than folded into ρ, because a monotone curve that
    never crosses zero would satisfy a rank correlation while failing the physical
    claim (a signed dose ladder whose response never changes sign is not a dial).
    """
    if len(doses) != len(rises):
        raise ValueError(f"ragged ladder: {len(doses)} doses, {len(rises)} rises")
    present = [(d, r) for d, r in zip(doses, rises)
               if r is not None and np.isfinite(r)]
    if len(present) < 3:
        return (None, False, False)
    d = [p[0] for p in present]
    r = [p[1] for p in present]
    from scipy.stats import spearmanr
    rho = spearmanr(d, r).statistic
    rho = float(rho) if rho is not None and np.isfinite(rho) else None
    neg = [rr for dd, rr in present if dd < 0]
    pos = [rr for dd, rr in present if dd > 0]
    flips = bool(neg and pos and min(neg) < 0.0 < max(pos))
    passes = bool(rho is not None and rho >= SPEARMAN_RHO_FLOOR and flips
                  and len(present) == len(DOSE_LADDER))
    return (rho, flips, passes)


def band_separation(rises_by_dose: dict[float, float],
                    band_by_dose: dict[float, tuple[float, float]]
                    ) -> tuple[int, bool, dict[str, bool], bool]:
    """§4.2(b): (n_outside, at_both_extremes, per_dose, passes).

    "Outside" is strictly outside the node's own random band's [min, max] at that
    dose — the same band arithmetic `score_entropy_writes.band` uses, so a §4
    verdict and a §5 row are read on one convention. A dose with no band is NOT
    counted as outside: an absent control is a hole, never a pass.
    """
    per_dose: dict[str, bool] = {}
    n_outside = 0
    for dose in DOSE_LADDER:
        rise = rises_by_dose.get(dose)
        band = band_by_dose.get(dose)
        out = bool(rise is not None and band is not None
                   and np.isfinite(rise)
                   and (rise < band[0] or rise > band[1]))
        per_dose[f"{dose:+g}"] = out
        n_outside += out
    both = all(per_dose.get(f"{d:+g}", False)
               for d in BAND_SEPARATION_REQUIRED_DOSES)
    passes = bool(n_outside >= BAND_SEPARATION_MIN_DOSES and both)
    return (n_outside, both, per_dose, passes)


def evaluate_actuation(*, node_key: str, site: int, arm: str,
                       rises_by_dose: dict[float, float],
                       band_by_dose: dict[float, tuple[float, float]],
                       coherence_at_scoring_dose: Optional[float],
                       job_id: Optional[str] = None,
                       site_role: Literal["site_of_record", "robustness_site"]
                       = "site_of_record",
                       lever_key: str = "entropy_gradient",
                       lever_is_calibration_only: bool = False
                       ) -> ActuationCalibrationResult:
    """§4.2's acceptance criterion, applied — PASS / FAIL / DEGENERATE.

    THE THREE-STATE VERDICT, exactly as §4.2 defines it:
      * PASS       — (a) and (b) and (c)
      * DEGENERATE — (a) and (b) hold but (c) fails. Reported as ITS OWN STATE and
                     never silently folded into PASS: a "rise" produced by collapse
                     is not actuation.
      * FAIL       — anything else.

    `coherence_at_scoring_dose` is an INPUT, not a computation. The distinct-word
    ratio is the coherence panel's metric and lives in `capability_battery.py` (one
    source of truth for the metric); this module owns the THRESHOLD and the verdict.
    A None coherence cannot pass (c) — an unmeasured floor is not a satisfied floor.

    Mechanics only. C§8: the desk decides; this returns a verdict object with every
    input value beside it, so the desk can re-apply the criteria by hand.
    """
    rho, flips, a_ok = dose_ordering(list(DOSE_LADDER),
                                     [rises_by_dose.get(d) for d in DOSE_LADDER])
    n_out, both, per_dose, b_ok = band_separation(rises_by_dose, band_by_dose)
    c_ok = bool(coherence_at_scoring_dose is not None
                and np.isfinite(coherence_at_scoring_dose)
                and coherence_at_scoring_dose >= COHERENCE_FLOOR)
    criteria = ActuationCriteria(
        spearman_rho_dose_vs_rise=rho,
        doses_present=sum(1 for d in DOSE_LADDER
                          if rises_by_dose.get(d) is not None),
        sign_flips_through_zero=flips, dose_ordering_passes=a_ok,
        outside_band_doses=n_out, outside_band_of=len(DOSE_LADDER),
        outside_band_at_both_extremes=both, per_dose_outside_band=per_dose,
        band_separation_passes=b_ok,
        coherence_at_scoring_dose=(float(coherence_at_scoring_dose)
                                   if coherence_at_scoring_dose is not None else None),
        coherence_passes=c_ok)
    if a_ok and b_ok and c_ok:
        verdict: Verdict = "PASS"
        why = (f"(a) ρ={rho:.3f} ≥ {SPEARMAN_RHO_FLOOR} across the full signed "
               f"ladder, sign flipping through zero; (b) outside the node's own "
               f"random band at {n_out}/{len(DOSE_LADDER)} doses including both "
               f"|{SCORING_DOSE}|; (c) coherence "
               f"{coherence_at_scoring_dose:.3f} ≥ {COHERENCE_FLOOR} at "
               f"+{SCORING_DOSE}")
    elif a_ok and b_ok:
        verdict = "DEGENERATE"
        why = (f"(a) and (b) hold (ρ={rho:.3f}, {n_out}/{len(DOSE_LADDER)} outside "
               f"band, both extremes) but (c) FAILS: coherence "
               f"{coherence_at_scoring_dose if coherence_at_scoring_dose is not None else 'UNMEASURED'}"
               f" < {COHERENCE_FLOOR} at +{SCORING_DOSE}. A rise produced by "
               f"collapse is not actuation — reported as its OWN state, never "
               f"folded into PASS.")
    else:
        verdict = "FAIL"
        parts = []
        if not a_ok:
            parts.append(
                f"(a) dose-ordering FAILS (ρ="
                f"{'None' if rho is None else f'{rho:.3f}'} vs floor "
                f"{SPEARMAN_RHO_FLOOR}, sign_flips={flips}, "
                f"{criteria.doses_present}/{len(DOSE_LADDER)} doses present)")
        if not b_ok:
            parts.append(f"(b) band separation FAILS ({n_out}/{len(DOSE_LADDER)} "
                         f"outside, both extremes={both})")
        if not c_ok:
            parts.append(f"(c) coherence floor FAILS "
                         f"({coherence_at_scoring_dose} < {COHERENCE_FLOOR})")
        why = "; ".join(parts)
    logger.info("§4 verdict %s: %s L%d (%s arm) — %s", verdict, node_key, site, arm,
                why)
    return ActuationCalibrationResult(
        node_key=node_key, site=site, site_role=site_role, arm=arm, job_id=job_id,
        lever_key=lever_key, lever_is_calibration_only=lever_is_calibration_only,
        criteria=criteria, verdict=verdict, verdict_rationale=why)


def gate_transported_cells(result: ActuationCalibrationResult) -> None:
    """§9 item 5: FAIL or DEGENERATE → no transported-write cells at that site.

    A refusal, not a warning. §4.2's consequence is frozen, and an enactor never
    adjudicates a live HALT — the remedy ladder (`remedy_ladder`) is what a caller
    reaches for next, and its end is a LABEL, not an override.
    """
    if not result.licenses_transported_cells:
        raise ActuationGateNotPassed(
            f"{result.node_key} L{result.site}: actuation calibration verdict "
            f"{result.verdict} — NO transported-write cells at this site (§4.2 / §9 "
            f"item 5). {result.verdict_rationale}. Named remedies in order: (i) "
            f"re-calibrate at the node's REGISTERED robustness site if it has one; "
            f"(ii) file the node's column row as {SITE_UNCALIBRATED}, which is a "
            f"result about the node, not a gap in the campaign. A site is NEVER "
            f"re-chosen on behavioral evidence.")


def stamp_fragment(result: ActuationCalibrationResult) -> dict:
    """§4.2's closing requirement, as the object a §5 stamp embeds.

    "Every behavioral cell in §5 NAMES its site's actuation calibration in its
    stamp, by job id + verdict + the three criterion values." That is exactly these
    keys — the values, not booleans, so the thresholds are re-appliable by hand.
    """
    c = result.criteria
    return {
        "job_id": result.job_id,
        "verdict": result.verdict,
        "site": result.site,
        "site_role": result.site_role,
        "arm": result.arm,
        "lever_key": result.lever_key,
        "lever_is_calibration_only": result.lever_is_calibration_only,
        "control_band_family": result.band_family,
        "spearman_rho": c.spearman_rho_dose_vs_rise,
        "outside_band_doses": f"{c.outside_band_doses}/{c.outside_band_of}",
        "coherence_at_scoring_dose": c.coherence_at_scoring_dose,
        "thresholds": {"rho_floor": c.rho_floor,
                       "min_outside_band_doses": c.min_outside_band_doses,
                       "required_doses": list(BAND_SEPARATION_REQUIRED_DOSES),
                       "coherence_floor": c.coherence_floor,
                       "scoring_dose": c.scoring_dose},
        "rationale": result.verdict_rationale,
        "brief_of_record": BRIEF_OF_RECORD,
        "brief_sha256": BRIEF_SHA256,
    }


def remedy_ladder(node_key: str, results: Sequence[ActuationCalibrationResult]
                  ) -> RemedyLadder:
    """§4.2's named remedies, in order, with the honest end.

    (i) the node's REGISTERED robustness site (gemma3-27b L41, llama-3.1-70b L43) —
    a registered site, not a new one; (ii) if both fail, the node's row in the
    entropy-write column is filed `SITE-UNCALIBRATED` and reported as such in the
    column table, **which is a result about the node, not a gap in the campaign**.
    """
    attempted = tuple(f"L{r.site}({r.site_role}):{r.verdict}" for r in results)
    passing = [r for r in results if r.licenses_transported_cells]
    if passing:
        r = passing[0]
        return RemedyLadder(
            node_key=node_key, attempted=attempted,
            next_action=f"transported-write cells fire at L{r.site} "
                        f"({r.site_role}); the §5 stamp names this gate",
            licenses_transported_cells=True)
    tried_roles = {r.site_role for r in results}
    robustness = ROBUSTNESS_SITES.get(node_key)
    if robustness is not None and "robustness_site" not in tried_roles:
        return RemedyLadder(
            node_key=node_key, attempted=attempted,
            next_action=f"remedy (i): re-calibrate at the REGISTERED robustness "
                        f"site L{robustness} — a registered site, never a new one",
            note="no site is re-chosen on behavioral evidence (§4.2)")
    return RemedyLadder(
        node_key=node_key, attempted=attempted,
        next_action=f"remedy (ii): file this node's entropy-write column row as "
                    f"{SITE_UNCALIBRATED} and report it in the column table",
        column_label=SITE_UNCALIBRATED,
        note="a result ABOUT THE NODE, not a gap in the campaign. Distinct from "
             f"{UNCALIBRATED_SITE}, which means the gate could not be APPLIED "
             "(ruling 2's SSM fallback) rather than that the site failed it.")


# ---------------------------------------------------------------- site resolution
def _registries() -> tuple[dict[str, tuple[int, ...]], dict[str, int], int]:
    """The three LIVE registries §4.3 names, imported at call time.

    Imported inside the function, not at module import: these registries are the
    campaign's own moving parts (vectors and sites land nightly), and a module-level
    snapshot is exactly the "trusting it" §4.3 forbids.
    """
    from metabasis.scripts.fit_transport_maps import SITES
    from metabasis.scripts.read_composed_predictions import (HUB_SITE_OF_RECORD,
                                                             SITE_OF_RECORD)
    return dict(SITES), dict(SITE_OF_RECORD), int(HUB_SITE_OF_RECORD)


def resolve_calibration_arm(node_key: str) -> tuple[str, str]:
    """§4.1/§4.3's arm-consistency rule: (arm, rule).

    Base-model nodes calibrate in the RAW arm only — the transported column into a
    base node rides the raw arm, so a native-arm calibration would certify a site in
    an arm no transported cell will use. Instruct nodes calibrate in the arm their
    transported cells will use, which is `native` for the campaign's family of
    record.

    The roster is the authority where it has a row; `BASE_ARM_NODES` is the brief's
    named list and is used only for nodes the roster does not carry (`olmo2-7b` is
    a carried banked node with no roster row). Disagreement between the two is
    reported in the rule string rather than silently resolved.
    """
    try:
        from metabasis.roster import ROSTER
        node = ROSTER.get(node_key)
    except ImportError:                                   # pragma: no cover
        node = None
    named_base = node_key in BASE_ARM_NODES
    if node is not None:
        is_base = node.checkpoint_identity == "base"
        if is_base != named_base:
            logger.warning(
                "%s: roster says checkpoint_identity=%s but the brief's named "
                "base list says %s — using the ROSTER (it is verified against the "
                "checkpoint) and recording the disagreement", node_key,
                node.checkpoint_identity, named_base)
        if is_base:
            return ("raw", f"base-model node (roster checkpoint_identity=base, arms="
                           f"{list(node.arms)}): calibrates in the RAW arm ONLY "
                           "(arm-consistency rule, §4.1) — the transported column "
                           "into a base node rides the raw arm")
        return ("native", f"instruct node (roster checkpoint_identity=instruct, "
                          f"arms={list(node.arms)}): calibrates in the arm its "
                          "transported cells will use — native, the family of record")
    if named_base:
        return ("raw", "base-model node (the brief's §4.3 named list; no roster row "
                       "— a carried banked node): calibrates in the RAW arm ONLY")
    return ("native", "no roster row and not on the brief's base list: defaults to "
                      "native, the arm of record. VERIFY before firing.")


def resolve_calibration_sites(node_key: str) -> CalibrationSiteResolution:
    """Resolve the calibration site(s) from the LIVE registries (§4.3).

    Refuses three things by construction:
      * a node with NO `SITE_OF_RECORD` row (olmo2-7b): a NAMED registry gap, which
        must be registered or ruled before the node calibrates — not filled in here;
      * a site that is not on the node's FIXED FIT GRID: that is a fiat site, and a
        RETIRED site (gemma L36, 70B L17) must never resolve by default;
      * gpt2-xl's absence from `SITE_OF_RECORD`, which is DELIBERATE (its site is
        deferred) and is handled by ruling 9's three-site branch rather than treated
        as a gap.
    """
    sites, site_of_record, hub_site = _registries()
    arm, arm_rule = resolve_calibration_arm(node_key)
    grid = tuple(sites.get(node_key, ()))

    if node_key == GPT2XL_KEY:
        # ruling 9: calibrate at all three BUILT experiment sites; the calibration
        # evidence feeds Luxia's DEFERRED site ruling. Its absence from
        # SITE_OF_RECORD is the deferral, not a gap.
        from metabasis.roster import SCAN_GRIDS
        effective = tuple(SCAN_GRIDS.get(GPT2XL_KEY, ()))
        offgrid = [s for s in GPT2XL_CALIBRATION_SITES if s not in effective]
        if offgrid:
            raise SiteFishingRefused(
                f"gpt2-xl: calibration sites {offgrid} are not on the node's "
                f"effective scan grid — ruling 9 calibrates at the THREE BUILT "
                f"experiment sites, and a site the curve never visited would be a "
                f"fiat site.")
        return CalibrationSiteResolution(
            node_key=node_key, site_of_record=None, robustness_site=None,
            fixed_fit_grid=effective, registries_agree=True,
            deferred_multi_site=GPT2XL_CALIBRATION_SITES, arm=arm, arm_rule=arm_rule,
            note="ruling 9: site of record DELIBERATELY DEFERRED (Luxia "
                 "2026-07-27; L7/L26/L47 built, no ⋆). Calibrates at ALL THREE "
                 "and the CALIBRATION evidence — not an â curve — feeds the "
                 "deferred site ruling; legitimate under §4 because calibration "
                 "gates a SITE rather than reading the transported object. The "
                 "transported column then fires ONLY at the ruled site.")

    ruled = site_of_record.get(node_key)
    if ruled is None:
        raise RegistryGap(
            f"{node_key}: no `SITE_OF_RECORD` row. This is a NAMED registry gap, "
            f"not an omission — the site must be REGISTERED or RULED before the "
            f"node calibrates (§4.3). "
            + (f"Its `SITES` grid is {grid}, which is a mid-depth BAND and not a "
               f"site pick: a site of record comes from the fit's own alignment "
               f"curve or a ruling, NEVER by fiat here." if grid else
               "It has no `SITES` grid either."))
    if grid and ruled not in grid:
        raise SiteNotOfRecord(
            f"{node_key}: SITE_OF_RECORD L{ruled} is not on the fixed fit grid "
            f"{grid} — the two registries disagree, which is a HALT, not a "
            f"preference. A retired site must never resolve by default.")
    robustness = ROBUSTNESS_SITES.get(node_key)
    if robustness is not None and grid and robustness not in grid:
        raise SiteNotOfRecord(
            f"{node_key}: robustness site L{robustness} is not on the fixed fit "
            f"grid {grid} — remedy (i) re-calibrates at a REGISTERED site, so an "
            f"unregistered one is refused here rather than at fire time.")
    return CalibrationSiteResolution(
        node_key=node_key, site_of_record=ruled, robustness_site=robustness,
        fixed_fit_grid=grid, registries_agree=True, arm=arm, arm_rule=arm_rule,
        note=f"resolved from the live registries: fit_transport_maps.SITES ∧ "
             f"read_composed_predictions.SITE_OF_RECORD (hub L{hub_site}). §4.3's "
             f"vector-inventory preflight must still RE-DERIVE readiness from disk.")


def assert_no_site_fishing(node_key: str, proposed_site: int, *,
                           evidence: str) -> None:
    """§4.2's closing prohibition, as code.

    "No node's site is re-chosen on behavioral evidence — sites come from curves and
    Luxia's rulings, never from a behavioral hunt (that would be site-fishing
    against the frozen §4 ordering)."

    A proposed site is admissible only if a registry already carries it: the site of
    record, the registered robustness site, or — for gpt2-xl alone under ruling 9 —
    one of the three built experiment sites whose CALIBRATION evidence feeds the
    deferred ruling.
    """
    try:
        res = resolve_calibration_sites(node_key)
    except (RegistryGap, SiteNotOfRecord) as exc:
        raise SiteFishingRefused(
            f"{node_key}: cannot admit site L{proposed_site} — the registries do "
            f"not resolve a site for this node at all ({exc}). A site proposed "
            f"against an unresolved registry is site-fishing by definition.") from exc
    admissible = {s for s in (res.site_of_record, res.robustness_site)
                  if s is not None} | set(res.deferred_multi_site)
    if proposed_site not in admissible:
        raise SiteFishingRefused(
            f"{node_key}: site L{proposed_site} is not registered (admissible: "
            f"{sorted(admissible)}). Sites come from curves and Luxia's rulings, "
            f"never from a behavioral hunt — §4.2. Evidence offered: {evidence!r}, "
            f"which is exactly the kind of evidence that must NOT choose a site.")


def vector_inventory_row(node_key: str, site: Optional[int], *,
                         vector_path: Optional[Path],
                         corpus_sha_of_record: Optional[str] = None
                         ) -> VectorInventoryRow:
    """§4.3's from-disk readiness row for one node — re-derived, never trusted.

    "The enactor's first act is a vector-inventory preflight that RE-DERIVES this
    table from disk rather than trusting it, because vectors are landing nightly."
    So this function reads the npz and the FD-gate artifact and reports; it does not
    consult any table of who is supposed to be ready.

    Data-independent by design: an absent path is a ROW WITH A REASON, not an
    exception, because "which nodes are not ready" is the answer the preflight
    exists to produce (§9 item 3 turns it into a HALT only when a cell is actually
    being built for that node).
    """
    from metabasis.scripts.run_behavioral_cells import CORPUS_SHA_V21

    expected = corpus_sha_of_record or CORPUS_SHA_V21
    if vector_path is None or not vector_path.exists():
        return VectorInventoryRow(
            node_key=node_key, site=site,
            vector_path=str(vector_path) if vector_path else None,
            vector_present=False, vector_sha256=None, fd_gate_present=False,
            fd_gate_passed=None, corpus_vintage=None, corpus_vintage_is_v21=None,
            ready=False,
            blocking_reason="no native entropy-gradient vector on disk (§9 item 3: "
                            "absent, not FD-gated, or FD-FAIL is a HALT when a cell "
                            "is built for this node)")
    sha = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    stem = vector_path.with_suffix("")
    fd_path = Path(f"{stem}_fd_gate.json")
    stamp_path = Path(f"{stem}_stamps.json")
    fd_passed: Optional[bool] = None
    if fd_path.exists():
        try:
            fd_passed = bool(json.loads(fd_path.read_text()).get("PASSES_FD_GATE"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("%s: FD-gate artifact unreadable (%s) — treated as "
                           "ABSENT, never as a pass", node_key, exc)
    vintage: Optional[str] = None
    if stamp_path.exists():
        try:
            vintage = json.loads(stamp_path.read_text()).get(
                "corpus_manifest_sha256")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("%s: build stamp unreadable (%s)", node_key, exc)
    is_v21 = (vintage == expected) if vintage else None
    reasons = []
    if fd_passed is not True:
        reasons.append("FD gate absent or not PASSED (§9 item 3)")
    if is_v21 is not True:
        reasons.append(f"corpus vintage is not v2.1 {expected[:8]}… (§9 item 2)")
    return VectorInventoryRow(
        node_key=node_key, site=site, vector_path=str(vector_path),
        vector_present=True, vector_sha256=sha,
        fd_gate_present=fd_path.exists(), fd_gate_passed=fd_passed,
        corpus_vintage=vintage, corpus_vintage_is_v21=is_v21,
        ready=not reasons, blocking_reason="; ".join(reasons) or None)


def require_ready_vector(row: VectorInventoryRow) -> None:
    """§9 item 3 as a HALT, applied when a cell is actually being built."""
    if not row.ready:
        raise NativeVectorUnavailable(
            f"{row.node_key} L{row.site}: {row.blocking_reason} — §9 item 3. A "
            f"node's native entropy-gradient vector must be present, FD-gated and "
            f"FD-PASS at the corpus vintage of record before its calibration cells "
            f"are built.")


# ---------------------------------------------------------------- cell plan (§4.1)
def calibration_cell_plan(*, node_key: str, site: int, arm: str,
                          per_token_median_resid_norm: float,
                          lever_key: str = "entropy_gradient",
                          band_key_template: str = "Rband{i}",
                          vector_npz: Optional[str] = None,
                          vector_provenance: str = "",
                          n_per_cell: int = N_PER_CELL) -> CalibrationCellPlan:
    """§4.1's cell structure for one (node, site, arm) — 1 + 6 + 18 = 25 cells.

    The control is the node's OWN native random band, at the SAME site and the SAME
    doses (`Rband*`), per the frozen §4 text "vs its own random band". The α=0
    baseline is built by `run_behavioral_cells.baseline_cell` and is SHARED with
    §5's column, so a node pays for it once.
    """
    signal = [c for c, _ in apply_dose_ladder(
        lever_key, site, per_token_median_resid_norm=per_token_median_resid_norm,
        kind="calibration", vector_npz=vector_npz,
        vector_provenance=vector_provenance, n=n_per_cell)]
    alphas = {c.cell_id: a for c, a in apply_dose_ladder(
        lever_key, site, per_token_median_resid_norm=per_token_median_resid_norm,
        kind="calibration", vector_npz=vector_npz,
        vector_provenance=vector_provenance, n=n_per_cell)}
    band_keys = tuple(band_key_template.format(i=i)
                      for i in range(1, N_BAND_MEMBERS + 1))
    band: list[CellSpec] = []
    for key in band_keys:
        for c, a in apply_dose_ladder(
                key, site,
                per_token_median_resid_norm=per_token_median_resid_norm,
                kind="calibration_band", band_family="Rband",
                vector_npz=vector_npz,
                vector_provenance=f"{node_key} native matched-norm random band "
                                  f"member {key} at L{site}",
                n=n_per_cell):
            band.append(c)
            alphas[c.cell_id] = a
    plan = CalibrationCellPlan(
        node_key=node_key, site=site, arm=arm, lever_key=lever_key,
        band_keys=band_keys, baseline=baseline_cell(site, n=n_per_cell),
        signal_cells=tuple(signal), band_cells=tuple(band), alphas=alphas,
        per_token_median_resid_norm=per_token_median_resid_norm,
        n_per_cell=n_per_cell)
    logger.info("§4.1 plan: %s L%d (%s arm) — %d cells, %d generations "
                "(lever %s, band %s)", node_key, site, arm, len(plan.cells),
                plan.total_generations, lever_key, list(band_keys))
    return plan


def gpt2xl_cell_plans(*, per_token_median_resid_norm_by_site: dict[int, float],
                      n_per_cell: int = N_PER_CELL) -> list[CalibrationCellPlan]:
    """Ruling 9: gpt2-xl calibrates at ALL THREE built experiment sites.

    3 × 24 science cells (the α=0 baseline is shared per site), the cheapest node on
    the roster, and the CALIBRATION evidence feeds Luxia's deferred site ruling. The
    transported column fires only at the ruled site, which `assert_ruled_site_before_
    transport` enforces.
    """
    res = resolve_calibration_sites(GPT2XL_KEY)
    plans = []
    for site in res.deferred_multi_site:
        norm = per_token_median_resid_norm_by_site.get(site)
        if norm is None:
            raise ValueError(
                f"gpt2-xl L{site}: no measured per-token median residual norm — "
                "§2.5 measures it in-job per site, and a dose cannot be resolved "
                "against a missing one")
        plans.append(calibration_cell_plan(
            node_key=GPT2XL_KEY, site=site, arm=res.arm,
            per_token_median_resid_norm=norm,
            vector_provenance=f"gpt2-xl native entropy-gradient vector at L{site} "
                              "(one of the three BUILT experiment sites, no ⋆)",
            n_per_cell=n_per_cell))
    return plans


def assert_ruled_site_before_transport(node_key: str, site: int) -> None:
    """Ruling 9's second half: the transported column fires ONLY at the ruled site.

    gpt2-xl may CALIBRATE at three sites; it may not carry a transported object at a
    site Luxia has not ruled. Calibration gates a site; transport reads an object,
    and reading it at an unruled site would let the behavioral evidence pick the
    site after all — the exact thing §4.2 forbids.
    """
    _, site_of_record, _ = _registries()
    ruled = site_of_record.get(node_key)
    if ruled is None:
        raise SiteFishingRefused(
            f"{node_key}: no ruled site of record, so NO transported-write cell may "
            f"be built (attempted L{site}). Ruling 9 lets gpt2-xl CALIBRATE at all "
            f"three built sites precisely because calibration gates a site rather "
            f"than reading the transported object; the transported column waits on "
            f"Luxia's ruling.")
    if site != ruled:
        raise SiteNotOfRecord(
            f"{node_key}: transported cells fire only at the RULED site L{ruled}, "
            f"not L{site} (§9 item 4).")


# ---------------------------------------------------------------- SSM branch (§4.4)
def ssm_nodes() -> tuple[str, ...]:
    """The behavioral tier, read from the ROSTER rather than from the brief's prose.

    §4.4 and §8 name two SSMs (falcon-mamba-7b, zamba2-7b), but the roster carries
    ONE: zamba2-7b is DELIBERATELY ABSENT — hard-parked on an upstream transformers
    defect, its row waiting on Luxia's shim ruling, and in that registry "adding a
    key is what makes a node collectable, so the absence IS the block". Resolving
    membership from the roster therefore yields the nodes that can actually run and
    names the absence instead of inventing a row for it.
    """
    from metabasis.roster import BEHAVIORAL_TIER
    return tuple(n.key for n in BEHAVIORAL_TIER)


def ssm_lever_decision(*, node_key: str, site: int, fd_gate_passed: bool,
                       fd_gate_rel_error: Optional[float] = None
                       ) -> SSMLeverDecision:
    """Ruling 2: a calibration-only native lever, or the automatic UNCALIBRATED-SITE.

    The SSMs have no native entropy-gradient vector BY DESIGN (roster:
    "transported-write column only; no native target, no star row"), so §4's gate
    read literally has no lever to apply. Ruling 2 builds one PURELY as a
    calibration instrument under a distinct key, FD-gated like every other build,
    and falls back AUTOMATICALLY to `UNCALIBRATED-SITE` labeling if that gate fails.

    Why the fallback is load-bearing rather than a consolation: the SSM boundary
    probe's entire value is that "hit or floor is unambiguous", and a floor at an
    uncalibrated site is indistinguishable from a dead site. The label is what stops
    an ambiguous floor being read as a regime boundary.
    """
    key = CALIBRATION_LEVER_KEY.format(site=site)
    if fd_gate_passed:
        return SSMLeverDecision(
            node_key=node_key, site=site, lever_key=key, fd_gate_passed=True,
            fd_gate_rel_error=fd_gate_rel_error, gate_applicable=True, label=None,
            rationale="the calibration-only lever passed the directional FD gate, "
                      "so §4's acceptance criterion applies to this node VERBATIM. "
                      + CALIBRATION_LEVER_PROHIBITIONS)
    return SSMLeverDecision(
        node_key=node_key, site=site, lever_key=key, fd_gate_passed=False,
        fd_gate_rel_error=fd_gate_rel_error, gate_applicable=False,
        label=UNCALIBRATED_SITE,
        rationale="the calibration-only lever FAILED the directional FD gate "
                  "(input-gradient extraction through Mamba/hybrid blocks is "
                  "unexercised, and the FD gate is exactly the instrument that "
                  f"catches a severed graph). Ruling 2's automatic fallback: every "
                  f"cell on this node is filed {UNCALIBRATED_SITE} and reported "
                  "DESCRIPTIVELY — a floor here is NOT claimed as a regime "
                  "boundary, because a floor at an uncalibrated site is "
                  "indistinguishable from a dead site.")


def assert_lever_stays_calibration_only(lever_key: str, use: str) -> None:
    """Ruling 2's prohibitions, as code.

    A key built under `CALIBRATION_LEVER_KEY` may gate a site and nothing else: no
    star system, no portability coefficient, no star row, never a target vector.
    """
    if not lever_key.startswith("calibration_lever_"):
        return
    forbidden = ("star", "portability", "c_M", "target", "exchange_rate",
                 "transport", "atlas")
    low = use.lower()
    for word in forbidden:
        if word.lower() in low:
            raise CalibrationLeverRefused(
                f"{lever_key} is a CALIBRATION INSTRUMENT ONLY (ruling 2) and was "
                f"asked for {use!r}, which names {word!r}. "
                f"{CALIBRATION_LEVER_PROHIBITIONS}")


# ---------------------------------------------------------------- desk table
def who_can_calibrate(nodes: Optional[Sequence[str]] = None) -> list[dict]:
    """§4.3's table, RE-DERIVED from the live registries (GPU-free, desk-readable).

    Deliberately reports rows rather than raising: "who cannot calibrate and why" is
    the answer, and a table that stopped at the first gap would hide the rest.
    """
    sites, site_of_record, _ = _registries()
    keys = list(nodes) if nodes else sorted(set(sites) | set(site_of_record)
                                           | {GPT2XL_KEY} | set(ssm_nodes()))
    rows = []
    for key in keys:
        row: dict[str, Any] = {"node": key}
        try:
            res = resolve_calibration_sites(key)
            row.update(site_of_record=res.site_of_record,
                       robustness_site=res.robustness_site,
                       deferred_multi_site=list(res.deferred_multi_site) or None,
                       arm=res.arm, status="SITE RESOLVED", blocking=None)
        except (RegistryGap, SiteNotOfRecord, SiteFishingRefused) as exc:
            arm, _ = resolve_calibration_arm(key)
            row.update(site_of_record=None, robustness_site=None,
                       deferred_multi_site=None, arm=arm,
                       status=type(exc).__name__, blocking=str(exc).split(".")[0])
        if key in ssm_nodes():
            row["tier"] = ("behavioral tier (SSM): no native target by design; "
                           "ruling 2 builds a CALIBRATION-ONLY lever, FD-gated, "
                           f"auto-fallback to {UNCALIBRATED_SITE}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------- CPU self-test
def _ladder(rises: Sequence[float]) -> dict[float, float]:
    return {d: r for d, r in zip(DOSE_LADDER, rises)}


def _band(halfwidths: Sequence[float]) -> dict[float, tuple[float, float]]:
    return {d: (-h, h) for d, h in zip(DOSE_LADDER, halfwidths)}


def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, data-independent verification of the §4 gate's logic."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    def skip(name: str, why: str) -> None:
        """A NAMED skip (rake M44): a block that cannot run in THIS configuration.

        Recorded as run-and-absent rather than crashed or failed, counted in the
        tail so a sweep can report coverage per configuration, and NEVER a non-zero
        exit code — nothing was asserted and found wanting.
        """
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))
        logger.info("SKIP %s — %s", name, why)

    # ---- 1. the §4.1 cell structure ------------------------------------------
    print("== selftest 1: §4.1's cell structure (1 + 6 + 18 = 25) ==")
    plan = calibration_cell_plan(node_key="qwen2.5-3b-instruct", site=26,
                                 arm="native", per_token_median_resid_norm=10.0,
                                 vector_provenance="node native lever")
    check("the block is 1 baseline + 6 signal + 18 band = 25 cells",
          (len(plan.cells), len(plan.signal_cells), len(plan.band_cells))
          == (25, 6, 18), f"{len(plan.cells)} cells")
    check("the arithmetic matches §4.1's table, not a retyped constant",
          (N_CALIBRATION_SIGNAL_CELLS, N_CALIBRATION_BAND_CELLS,
           N_CALIBRATION_CELLS) == (6, 18, 25))
    check("2,000 generations per calibration block at n=80 (480 + 1,440 + 80)",
          plan.total_generations == 2000, str(plan.total_generations))
    check("§4.1's signal-cell total is 480 and its band total is 1,440",
          len(plan.signal_cells) * 80 == 480 and len(plan.band_cells) * 80 == 1440)
    check("the control is the node's OWN NATIVE band (Rband*), never gRband*",
          all(c.band_family == "Rband" for c in plan.band_cells)
          and all(c.vector_key.startswith("Rband") for c in plan.band_cells),
          str(plan.band_keys))
    check("3 band members × the 6-dose ladder, at the SAME site and doses",
          len(plan.band_keys) == 3
          and sorted({c.alpha_frac for c in plan.band_cells}) == sorted(DOSE_LADDER)
          and {c.site for c in plan.band_cells} == {26})
    check("the α=0 baseline is SHARED with §5's column (built once)",
          plan.baseline.is_baseline and plan.baseline.alpha_frac == 0.0
          and plan.baseline.vector_key is None)
    check("every non-baseline cell has an exact α = frac × the measured norm",
          all(abs(plan.alphas[c.cell_id] - c.alpha_frac * 10.0) < 1e-12
              for c in plan.signal_cells + plan.band_cells),
          f"α(+0.3) = {plan.alphas[plan.signal_cells[-1].cell_id]}")
    check("cell ids carry the lever key and the banked a{frac:+.2f} formatting",
          plan.signal_cells[-1].cell_id == "entropy_gradient_L26_a+0.30",
          plan.signal_cells[-1].cell_id)
    check("a plan that is not §4.1's shape is REFUSED",
          _raises(lambda: CalibrationCellPlan(
              node_key="x", site=1, arm="native", lever_key="v", band_keys=("Rband1",),
              baseline=baseline_cell(1), signal_cells=tuple(plan.signal_cells[:5]),
              band_cells=plan.band_cells, alphas={},
              per_token_median_resid_norm=1.0), ValueError))
    check("a plan controlled by the TRANSPORTED band is refused (§4.1)",
          _raises(lambda: CalibrationCellPlan(
              node_key="x", site=26, arm="native", lever_key="entropy_gradient",
              band_keys=("gRband1",), baseline=baseline_cell(26),
              signal_cells=plan.signal_cells,
              band_cells=tuple(c.model_copy(update={"band_family": "gRband"})
                               for c in plan.band_cells),
              alphas={}, per_token_median_resid_norm=10.0), ValueError))

    # ---- 2. §4.2(a) dose ordering --------------------------------------------
    print("== selftest 2: §4.2(a) dose-ordering across the full SIGNED ladder ==")
    good = [-0.62, -0.21, -0.05, 0.06, 0.24, 0.71]
    rho, flips, ok = dose_ordering(list(DOSE_LADDER), good)
    check("a clean signed ladder gives ρ = 1.0 and passes",
          rho == 1.0 and flips and ok, f"ρ={rho}")
    rho2, flips2, ok2 = dose_ordering(list(DOSE_LADDER),
                                      [0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    check("a monotone ladder that NEVER crosses zero FAILS (a) despite ρ=1",
          rho2 == 1.0 and not flips2 and not ok2,
          f"ρ={rho2} but sign never flips — a dial must change sign through zero")
    rho3, _, ok3 = dose_ordering(list(DOSE_LADDER),
                                 [-0.6, 0.3, -0.2, 0.1, -0.4, 0.7])
    check("a scrambled ladder falls below the ρ floor",
          rho3 is not None and rho3 < SPEARMAN_RHO_FLOOR and not ok3,
          f"ρ={rho3:.3f} < {SPEARMAN_RHO_FLOOR}")
    check("the ρ floor of record is .80",
          SPEARMAN_RHO_FLOOR == 0.80)
    _, _, ok4 = dose_ordering(list(DOSE_LADDER),
                              [-0.6, -0.2, None, 0.1, 0.3, 0.7])
    check("a MISSING dose cannot pass (a): the read is across the FULL ladder",
          not ok4)
    check("fewer than three present doses gives no ρ at all",
          dose_ordering([0.1, 0.3], [0.2, 0.5])[0] is None)
    check("a ragged ladder is refused, never zipped short",
          _raises(lambda: dose_ordering([0.1, 0.3], [0.2]), ValueError))

    # ---- 3. §4.2(b) band separation ------------------------------------------
    print("== selftest 3: §4.2(b) band separation (≥4/6 AND both |0.3|) ==")
    n_out, both, per_dose, ok = band_separation(
        _ladder(good), _band([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    check("a strong signal is outside the band at 4/6 doses including both extremes",
          (n_out, both, ok) == (4, True, True),
          f"{n_out}/6, both extremes={both}, per-dose {json.dumps(per_dose)}")
    # a monotone ladder (so (a) is not what fails) whose band is WIDE at +0.3 only:
    # 5 of 6 doses are outside, but one of the two required extremes is not.
    n5, both5, _, ok5 = band_separation(
        _ladder([-0.62, -0.41, -0.25, 0.31, 0.45, 0.55]),
        _band([0.1, 0.1, 0.1, 0.1, 0.1, 0.9]))
    check("5/6 outside but MISSING the +0.3 extreme FAILS (b)",
          n5 == 5 and not both5 and not ok5,
          f"{n5}/6 outside, both extremes={both5} — the count alone never carries it")
    n6, _, _, ok6 = band_separation(
        _ladder([-0.62, -0.02, -0.01, 0.01, 0.02, 0.71]),
        _band([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    check("both extremes but only 2/6 outside FAILS the ≥4/6 count",
          n6 == 2 and not ok6, f"{n6}/6 outside")
    check("the criterion of record is ≥4/6 plus both |0.3| doses",
          (BAND_SEPARATION_MIN_DOSES, tuple(BAND_SEPARATION_REQUIRED_DOSES))
          == (4, (-0.3, 0.3)))
    n7, _, _, ok7 = band_separation(_ladder(good), {})
    check("a dose with NO band is not counted as outside (a hole, never a pass)",
          n7 == 0 and not ok7)
    check("'outside' is strict: exactly on the band edge is INSIDE",
          band_separation({0.3: 0.1}, {0.3: (-0.1, 0.1)})[0] == 0)

    # ---- 4. the three-state verdict ------------------------------------------
    print("== selftest 4: PASS / FAIL / DEGENERATE (§4.2) ==")
    band6 = _band([0.1] * 6)
    passing = evaluate_actuation(
        node_key="qwen2.5-3b-instruct", site=26, arm="native",
        rises_by_dose=_ladder(good), band_by_dose=band6,
        coherence_at_scoring_dose=0.62, job_id="selftest-job")
    check("all three criteria hold → PASS, and PASS licenses transported cells",
          passing.verdict == "PASS" and passing.licenses_transported_cells)
    degen = evaluate_actuation(
        node_key="qwen2.5-3b-instruct", site=26, arm="native",
        rises_by_dose=_ladder(good), band_by_dose=band6,
        coherence_at_scoring_dose=0.31, job_id="selftest-job")
    check("(a)+(b) hold, (c) fails → DEGENERATE, reported as its OWN state",
          degen.verdict == "DEGENERATE", degen.verdict_rationale[:70])
    check("DEGENERATE does NOT license transported cells (never folded into PASS)",
          not degen.licenses_transported_cells
          and _raises(lambda: gate_transported_cells(degen), ActuationGateNotPassed))
    failing = evaluate_actuation(
        node_key="phi-4", site=19, arm="native",
        rises_by_dose=_ladder([0.01] * 6), band_by_dose=band6,
        coherence_at_scoring_dose=0.9)
    check("no ordering and no separation → FAIL", failing.verdict == "FAIL")
    check("FAIL is a HALT for transported cells (§9 item 5)",
          _raises(lambda: gate_transported_cells(failing), ActuationGateNotPassed))
    check("a PASS raises nothing", _ok(lambda: gate_transported_cells(passing)))
    unmeasured = evaluate_actuation(
        node_key="x", site=1, arm="native", rises_by_dose=_ladder(good),
        band_by_dose=band6, coherence_at_scoring_dose=None)
    check("an UNMEASURED coherence cannot satisfy (c) — DEGENERATE, not PASS",
          unmeasured.verdict == "DEGENERATE"
          and unmeasured.criteria.coherence_at_scoring_dose is None)
    check("the coherence floor of record is .45 at the +0.3 scoring dose",
          (COHERENCE_FLOOR, SCORING_DOSE) == (0.45, 0.3))
    check("coherence exactly at the floor PASSES (≥, not >)",
          evaluate_actuation(node_key="x", site=1, arm="native",
                             rises_by_dose=_ladder(good), band_by_dose=band6,
                             coherence_at_scoring_dose=0.45).verdict == "PASS")
    check("every criterion VALUE is filed, not just its boolean",
          (passing.criteria.spearman_rho_dose_vs_rise == 1.0
           and passing.criteria.outside_band_doses == 4
           and passing.criteria.coherence_at_scoring_dose == 0.62))

    # ---- 5. the §5 stamp fragment --------------------------------------------
    print("== selftest 5: what every §5 cell names about its gate ==")
    frag = stamp_fragment(passing)
    for field in ("job_id", "verdict", "spearman_rho", "outside_band_doses",
                  "coherence_at_scoring_dose"):
        check(f"the stamp fragment carries {field} (§4.2's closing requirement)",
              frag.get(field) is not None, repr(frag.get(field)))
    check("the fragment carries the THRESHOLDS, so the desk can re-apply them",
          frag["thresholds"]["rho_floor"] == SPEARMAN_RHO_FLOOR
          and frag["thresholds"]["coherence_floor"] == COHERENCE_FLOOR)
    check("the fragment names the control band family (Rband, not gRband)",
          frag["control_band_family"] == "Rband")
    check("the fragment names the brief of record and its sha",
          frag["brief_sha256"] == BRIEF_SHA256)
    check("a DEGENERATE fragment says so, and carries the failing value",
          stamp_fragment(degen)["verdict"] == "DEGENERATE"
          and stamp_fragment(degen)["coherence_at_scoring_dose"] == 0.31)
    check("the result is UNSTAMPED (C§8) — this module scores nothing on its own",
          passing.grade == GRADE_LINE)

    # ---- 6. the remedy ladder ------------------------------------------------
    print("== selftest 6: §4.2's named remedies, in order ==")
    lad = remedy_ladder("gemma3-27b", [
        evaluate_actuation(node_key="gemma3-27b", site=38, arm="native",
                           rises_by_dose=_ladder([0.01] * 6), band_by_dose=band6,
                           coherence_at_scoring_dose=0.9)])
    check("remedy (i) is the REGISTERED robustness site, named by number",
          "L41" in lad.next_action and lad.column_label is None,
          lad.next_action)
    lad2 = remedy_ladder("gemma3-27b", [
        evaluate_actuation(node_key="gemma3-27b", site=38, arm="native",
                           rises_by_dose=_ladder([0.01] * 6), band_by_dose=band6,
                           coherence_at_scoring_dose=0.9),
        evaluate_actuation(node_key="gemma3-27b", site=41, arm="native",
                           site_role="robustness_site",
                           rises_by_dose=_ladder([0.01] * 6), band_by_dose=band6,
                           coherence_at_scoring_dose=0.9)])
    check("both registered sites failing ends at SITE-UNCALIBRATED",
          lad2.column_label == SITE_UNCALIBRATED
          and not lad2.licenses_transported_cells, lad2.next_action)
    check("SITE-UNCALIBRATED is framed as a RESULT about the node",
          "not a gap in the campaign" in lad2.note)
    lad3 = remedy_ladder("phi-4", [failing])
    check("a node with NO registered robustness site goes straight to the label",
          lad3.column_label == SITE_UNCALIBRATED and "phi-4" not in ROBUSTNESS_SITES)
    lad4 = remedy_ladder("qwen2.5-3b-instruct", [failing, passing])
    check("a PASS anywhere ends the ladder and licenses the column",
          lad4.licenses_transported_cells and lad4.column_label is None)
    check("the two registered robustness sites are gemma L41 and 70B L43",
          ROBUSTNESS_SITES == {"gemma3-27b": 41, "llama-3.1-70b-instruct": 43})

    # ---- 7. the two labels must never merge ----------------------------------
    print("== selftest 7: SITE-UNCALIBRATED vs UNCALIBRATED-SITE ==")
    check("the two labels are DIFFERENT strings",
          SITE_UNCALIBRATED != UNCALIBRATED_SITE,
          f"{SITE_UNCALIBRATED!r} (gate ran, site failed) vs "
          f"{UNCALIBRATED_SITE!r} (gate not applicable)")
    check("§4.2's label is SITE-UNCALIBRATED", SITE_UNCALIBRATED == "SITE-UNCALIBRATED")
    check("§4.4/ruling 2's label is UNCALIBRATED-SITE",
          UNCALIBRATED_SITE == "UNCALIBRATED-SITE")

    # ---- 8. site resolution + no site-fishing --------------------------------
    print("== selftest 8: sites come from registries, never from behavior ==")
    res = resolve_calibration_sites("qwen2.5-3b-instruct")
    check("an instruct node resolves its site of record from the live registries",
          res.site_of_record == 26 and res.registries_agree, f"L{res.site_of_record}")
    check("the resolution names the fixed fit grid it validated against",
          26 in res.fixed_fit_grid, str(res.fixed_fit_grid))
    check("a node with a registered robustness site reports both",
          resolve_calibration_sites("gemma3-27b").robustness_site == 41)
    check("olmo2-7b (base) is a NAMED REGISTRY GAP, not an omission",
          _raises(lambda: resolve_calibration_sites("olmo2-7b"), RegistryGap))
    check("the gap message says the SITES grid is a BAND, not a site pick",
          "not a site pick" in _msg(lambda: resolve_calibration_sites("olmo2-7b")))
    check("an unknown node is a registry gap too, never a default site",
          _raises(lambda: resolve_calibration_sites("not-a-node"), RegistryGap))
    check("site-fishing on behavioral evidence is REFUSED",
          _raises(lambda: assert_no_site_fishing(
              "qwen2.5-3b-instruct", 21,
              evidence="L21 showed a bigger entropy rise in the calibration cells"),
              SiteFishingRefused))
    check("the refusal names the offered evidence as the disqualifying thing",
          "must NOT choose a site" in _msg(lambda: assert_no_site_fishing(
              "qwen2.5-3b-instruct", 21, evidence="a bigger rise")))
    check("the REGISTERED site of record is admissible",
          _ok(lambda: assert_no_site_fishing("qwen2.5-3b-instruct", 26,
                                            evidence="the ruled site")))
    check("the REGISTERED robustness site is admissible (remedy (i))",
          _ok(lambda: assert_no_site_fishing("gemma3-27b", 41,
                                            evidence="remedy (i)")))
    check("a RETIRED site never resolves (gemma L36, the 70B's L17)",
          _raises(lambda: assert_no_site_fishing("gemma3-27b", 36,
                                                evidence="it is banked"),
                  SiteFishingRefused)
          and _raises(lambda: assert_no_site_fishing("llama-3.1-70b-instruct", 17,
                                                     evidence="it is banked"),
                      SiteFishingRefused))
    check("site resolution refuses a site not on the node's fixed fit grid",
          _ok(lambda: resolve_calibration_sites("phi-4")))

    # ---- 9. the arm-consistency rule ----------------------------------------
    print("== selftest 9: base nodes calibrate in the RAW arm only (§4.1) ==")
    for key in ("pythia-6.9b", "gpt2-xl"):
        arm, rule = resolve_calibration_arm(key)
        check(f"{key} (base) calibrates raw-only", arm == "raw", rule[:60])
    for key in ("qwen2.5-3b-instruct", "phi-4", "gemma3-27b"):
        arm, _ = resolve_calibration_arm(key)
        check(f"{key} (instruct) calibrates in the arm its transported cells use",
              arm == "native")
    arm_o, rule_o = resolve_calibration_arm("olmo2-7b")
    check("olmo2-7b has no roster row and falls to the brief's named base list",
          arm_o == "raw" and "no roster row" in rule_o, rule_o[:70])
    check("the brief's named base list matches the roster where both speak",
          all(resolve_calibration_arm(k)[0] == "raw"
              for k in ("pythia-6.9b", "gpt2-xl")))

    # ---- 10. gpt2-xl's three sites (ruling 9) --------------------------------
    print("== selftest 10: gpt2-xl calibrates at all three built sites (ruling 9) ==")
    g = resolve_calibration_sites(GPT2XL_KEY)
    check("gpt2-xl resolves THREE calibration sites and no ⋆",
          tuple(g.deferred_multi_site) == (7, 26, 47) and g.site_of_record is None,
          str(g.deferred_multi_site))
    check("all three are on the node's own effective scan grid (no fiat site)",
          all(s in g.fixed_fit_grid for s in g.deferred_multi_site))
    check("gpt2-xl calibrates in the raw arm (its only arm)", g.arm == "raw")
    plans = gpt2xl_cell_plans(
        per_token_median_resid_norm_by_site={7: 8.0, 26: 9.0, 47: 11.0})
    check("three plans, 25 cells each, 6,000 generations in total",
          len(plans) == 3 and all(len(p.cells) == 25 for p in plans)
          and sum(p.total_generations for p in plans) == 6000,
          f"{sum(p.total_generations for p in plans)} generations")
    check("each plan's α resolves against ITS OWN site's measured norm",
          [p.alphas[f"entropy_gradient_L{p.site}_a+0.30"] for p in plans]
          == [0.3 * 8.0, 0.3 * 9.0, 0.3 * 11.0])
    check("a site with no measured norm cannot be planned",
          _raises(lambda: gpt2xl_cell_plans(
              per_token_median_resid_norm_by_site={7: 8.0}), ValueError))
    check("the TRANSPORTED column still waits on Luxia's ruling (ruling 9's half 2)",
          _raises(lambda: assert_ruled_site_before_transport(GPT2XL_KEY, 26),
                  SiteFishingRefused))
    check("a ruled node's transported cells must fire at the ruled site",
          _ok(lambda: assert_ruled_site_before_transport("qwen2.5-3b-instruct", 26))
          and _raises(lambda: assert_ruled_site_before_transport(
              "qwen2.5-3b-instruct", 21), SiteNotOfRecord))

    # ---- 11. the SSM branch (ruling 2) --------------------------------------
    print("== selftest 11: the SSM calibration-only lever (ruling 2) ==")
    tier = ssm_nodes()
    check("the behavioral tier is read from the ROSTER, not the brief's prose",
          tier == ("falcon-mamba-7b-instruct",), str(tier))
    check("zamba2-7b is DELIBERATELY ABSENT from the roster — the absence IS the block",
          "zamba2-7b" not in tier)
    ok_lever = ssm_lever_decision(node_key=tier[0], site=32, fd_gate_passed=True,
                                  fd_gate_rel_error=0.011)
    check("an FD-PASSED lever makes §4's gate applicable VERBATIM",
          ok_lever.gate_applicable and ok_lever.label is None,
          ok_lever.lever_key)
    check("the lever is banked under a DISTINCT key (never a target vector)",
          ok_lever.lever_key == "calibration_lever_entropy_gradient_L32")
    check("the decision carries ruling 2's prohibitions verbatim",
          "never hosts a portability coefficient" in ok_lever.prohibitions
          and "no star row" in ok_lever.prohibitions)
    bad_lever = ssm_lever_decision(node_key=tier[0], site=32, fd_gate_passed=False,
                                   fd_gate_rel_error=41.7)
    check("an FD-FAILED lever falls back AUTOMATICALLY to UNCALIBRATED-SITE",
          not bad_lever.gate_applicable and bad_lever.label == UNCALIBRATED_SITE)
    check("the fallback says a floor is NOT claimed as a regime boundary",
          "regime boundary" in bad_lever.rationale
          and "indistinguishable from a dead site" in bad_lever.rationale)
    check("an unlabeled inapplicable gate is refused (the ambiguity ruling 2 forbids)",
          _raises(lambda: SSMLeverDecision(
              node_key="x", site=1, lever_key="calibration_lever_entropy_gradient_L1",
              fd_gate_passed=False, gate_applicable=False, label=None), ValueError))
    check("a PASSED lever may not carry an UNCALIBRATED label either",
          _raises(lambda: SSMLeverDecision(
              node_key="x", site=1, lever_key="k", fd_gate_passed=True,
              gate_applicable=True, label=UNCALIBRATED_SITE), ValueError))
    check("the FD criterion of record is the directional gate at .05",
          SSM_FD_GATE_TOLERANCE == 0.05 and "2*eps" in SSM_FD_GATE_CRITERION)
    check("a calibration lever asked to host a star row is REFUSED",
          _raises(lambda: assert_lever_stays_calibration_only(
              ok_lever.lever_key, "solve the star system's portability coefficient"),
              CalibrationLeverRefused))
    check("a calibration lever asked to be a transported target is REFUSED",
          _raises(lambda: assert_lever_stays_calibration_only(
              ok_lever.lever_key, "transport this target vector to qwen-7b"),
              CalibrationLeverRefused))
    check("a calibration lever gating a site is allowed (its one job)",
          _ok(lambda: assert_lever_stays_calibration_only(
              ok_lever.lever_key, "gate the site's actuation")))
    check("a NORMAL vector key is unconstrained by ruling 2's prohibitions",
          _ok(lambda: assert_lever_stays_calibration_only(
              "entropy_gradient", "host a portability coefficient")))

    # ---- 12. the §4.3 vector-inventory preflight (from disk) ------------------
    print("== selftest 12: §4.3's preflight RE-DERIVES readiness from disk ==")
    from metabasis.scripts.run_behavioral_cells import CORPUS_SHA_V21
    absent = vector_inventory_row("phi-4", 19, vector_path=None)
    check("an absent vector is a ROW WITH A REASON, not an exception",
          not absent.ready and "no native entropy-gradient vector" in
          absent.blocking_reason)
    check("an absent vector becomes a HALT only when a cell is built (§9 item 3)",
          _raises(lambda: require_ready_vector(absent), NativeVectorUnavailable))
    with tempfile.TemporaryDirectory(prefix="vec_inv_") as td:
        d = Path(td)
        npz = d / "entropy_gradient_phi-4.npz"
        np.savez(npz, entropy_gradient_L19=np.ones(8, dtype=np.float32))
        row = vector_inventory_row("phi-4", 19, vector_path=npz)
        check("a vector with NO FD-gate artifact is not ready",
              not row.ready and "FD gate absent" in row.blocking_reason)
        (d / "entropy_gradient_phi-4_fd_gate.json").write_text(
            json.dumps({"PASSES_FD_GATE": False}))
        row = vector_inventory_row("phi-4", 19, vector_path=npz)
        check("an FD-FAIL vector is not ready (§9 item 3)",
              row.fd_gate_passed is False and not row.ready)
        (d / "entropy_gradient_phi-4_fd_gate.json").write_text(
            json.dumps({"PASSES_FD_GATE": True}))
        (d / "entropy_gradient_phi-4_stamps.json").write_text(
            json.dumps({"corpus_manifest_sha256": "1" * 64}))
        row = vector_inventory_row("phi-4", 19, vector_path=npz)
        check("a v1-vintage vector is not ready (§9 item 2)",
              not row.ready and "not v2.1" in row.blocking_reason)
        (d / "entropy_gradient_phi-4_stamps.json").write_text(
            json.dumps({"corpus_manifest_sha256": CORPUS_SHA_V21}))
        row = vector_inventory_row("phi-4", 19, vector_path=npz)
        check("FD-PASS at corpus v2.1 is READY, and its sha is re-derived from disk",
              row.ready and row.corpus_vintage_is_v21
              and row.vector_sha256 == hashlib.sha256(npz.read_bytes()).hexdigest(),
              row.vector_sha256[:16] + "…")
        check("a ready vector raises nothing", _ok(lambda: require_ready_vector(row)))
        (d / "entropy_gradient_phi-4_fd_gate.json").write_text("{not json")
        row = vector_inventory_row("phi-4", 19, vector_path=npz)
        check("an UNREADABLE FD-gate artifact is treated as ABSENT, never as a pass",
              row.fd_gate_passed is None and not row.ready)

    # ---- 13. the desk table -------------------------------------------------
    print("== selftest 13: the §4.3 who-can-calibrate table ==")
    table = who_can_calibrate()
    by_node = {r["node"]: r for r in table}
    check("the table reports ROWS, never stopping at the first gap",
          len(table) > 15 and any(r["status"] != "SITE RESOLVED" for r in table)
          and any(r["status"] == "SITE RESOLVED" for r in table),
          f"{len(table)} rows, "
          f"{sum(1 for r in table if r['status'] == 'SITE RESOLVED')} resolved")
    check("olmo2-7b appears as a RegistryGap row rather than being omitted",
          by_node["olmo2-7b"]["status"] == "RegistryGap")
    check("gpt2-xl appears with its three deferred sites",
          by_node[GPT2XL_KEY]["deferred_multi_site"] == [7, 26, 47])
    check("the SSM row names its tier and ruling 2's fallback",
          UNCALIBRATED_SITE in by_node["falcon-mamba-7b-instruct"]["tier"])
    check("a resolved instruct node reports its site and arm",
          (by_node["qwen2.5-3b-instruct"]["site_of_record"],
           by_node["qwen2.5-3b-instruct"]["arm"]) == (26, "native"))

    # ---- 14. HALT semantics --------------------------------------------------
    print("== selftest 14: every §4 refusal is a named exception ==")
    check("every §4 exception descends from BehavioralHarnessError",
          all(issubclass(c, BehavioralHarnessError) for c in (
              ActuationGateNotPassed, SiteFishingRefused, RegistryGap,
              CalibrationLeverRefused)))
    check("the gate refusal names both remedies, in order",
          "(i) re-calibrate" in _msg(lambda: gate_transported_cells(failing))
          and "(ii) file" in _msg(lambda: gate_transported_cells(failing)))
    check("the gate refusal restates that a site is never re-chosen on behavior",
          "NEVER re-chosen on behavioral evidence" in
          _msg(lambda: gate_transported_cells(failing)))

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    # RAKE M44: coverage is part of the verdict, per configuration.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if failures else 0


def _raises(fn: Callable[[], Any], exc: type[BaseException]) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                               # noqa: BLE001 — wrong class
        return False
    return False


def _ok(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except Exception:                               # noqa: BLE001
        return False


def _msg(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except Exception as exc:                        # noqa: BLE001
        return str(exc)
    return ""


# ---------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The §4 actuation acceptance gate: cell plan, site resolution, "
                    "verdict logic. Mechanics only — the desk decides (C§8).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only, data-independent verification")
    ap.add_argument("--who-can-calibrate", action="store_true",
                    help="§4.3's table, re-derived from the live registries (GPU-free)")
    ap.add_argument("--node", default=None, help="restrict the table to one node")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.who_can_calibrate:
        rows = who_can_calibrate([args.node] if args.node else None)
        print(json.dumps(rows, indent=1))
        ready = sum(1 for r in rows if r["status"] == "SITE RESOLVED")
        print(f"\n{ready}/{len(rows)} nodes resolve a calibration site. §4.3: a "
              "resolved site is NOT readiness — the vector-inventory preflight must "
              "re-derive that from disk.")
        return 0
    ap.error("nothing to do: pass --selftest or --who-can-calibrate")
    return 2                                        # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
