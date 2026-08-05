"""THE STAR BESIDE on webtext-v3 — the S1 continuation, gateless and descriptive.

WHAT THIS IS. The scalar star factorization

    â(A→B) = c_A · c_B

re-derived on the webtext-v3 basis under the PARENT contract's derivation rules
(`freeze/transport-campaign:docs/planning/PREREG-transport-campaign-2026-07-26.md`,
sha 33ba8290…, §2 estimands / §3 the star's prediction structure) and quoted ONLY
as the composed-vs-star descriptive contrast on the primary basis — v3 contract
§8's "the star beside (S1 continuation)". S1 is STAMPED ("scalar factorization
refuted at scale; hub-mediated transport demonstrated at map order"); this module
CONTINUES that record on the clean basis and CANNOT reopen it. Nothing here is a
gate. The ±.05 in-band count is reference-only and labelled NON-GATE everywhere
it appears.

NO NEW MATH. Every coefficient is produced by the banked solver lineage the
bootstrap parity gate certified — `solve_portability_coefficients.
solve_log_least_squares`, imported and called, never reimplemented. This module
is ADAPTER + SCORER + REPORTER only: it turns the sealed v3 scored record into
the lineage's own `ExchangeRateObservation` rows, hands them to the lineage's own
solver, and counts what comes back.

THE VINTAGE RULE IS ABSOLUTE. No v2.1-vintage constant enters the v3 solve —
not as an input, not as an anchor, not as a starting point. The v3 systems are
gauge-fixed by their own pair graphs (see THE ANCHOR below), so no external
constant is even admissible. The ONE legitimate v2.1 touch is the REGRESSION
PROOF (`--regression-proof`): the adapter path run on the BANKED v2.1 inputs must
reproduce the BANKED star record — the bootstrap parity-gate bar — before any v3
number is quoted. `--run` refuses to write an artifact unless that proof passes.

THE ANCHOR (the parent's rule, applied). The parent's §3 star was a pure HUB
star: bipartite, one gauge freedom (c_hub → t·c_hub, c_leaf → c_leaf/t leaves
every â unchanged), so exactly one anchor was needed and the anchor of record was
the hub constant c_8B from the inherited banked native::proc_k128 system. The v3
scoreable set is NOT a hub star: §8 defines a scoreable pair as any ordered pair
whose endpoints are both CORE roster and neither is a §7 race hub, and the 240
sealed slots realize the COMPLETE graph on 16 core models (120 unordered pairs ×
2 directions). A complete graph is non-bipartite, so — in the lineage solver's
own words — the gauge is "determined by the pair graph itself … anchors are
consistency checks here, not constraints". The v3 anchor is therefore the graph,
and it is the parent's rule that says so. The parent's own hub anchor is doubly
inapplicable: 8b-instruct is a §7 RACE HUB and not a core endpoint at all, and
its banked constant is v2.1 vintage. This module asserts nullspace_dim == 0 per
system and REFUSES to emit if a system needs an anchor it may not have.

ARMS ARE NEVER MIXED. §3.2's arm of record splits the 240: instruct↔instruct is
native::proc_k256, any base-endpoint slot is raw::proc_k256. One solve = one
(arm × family) — the arm-consistency rule (Luxia 2026-07-23) and rake 40. So the
v3 star is TWO systems, and a model that appears in both carries two constants,
each belonging to the system it was solved in. Every quoted c_M names its system.

IN-SAMPLE BY CONSTRUCTION — READ THE HEADLINE COUNT WITH THIS. There are no
v3 hub legs among core models to derive constants from (§8 step 1 fit hub legs
for the five RACE hubs only, and race-hub legs are not scoreable pairs), so the
only â available to the star are the 240 scored slots themselves. The star is
therefore FIT ON THE VERY SLOTS IT IS THEN SCORED ON, while the composed column
was filed, sha'd and desk-sealed BEFORE the first direct-pair fit ran. The
comparison is maximally generous to the star. A genuinely held-out star count is
computed beside by leave-one-unordered-pair-out over the same lineage solver
(`held_out_beside`), which is what the parent's "a system must pass its own
held-out check before hosting a constant" asks for.

Run (repo root = the DATA root; the module is imported from the worktree):
  python -m metabasis.scripts.star_beside_webtext_v3 --selftest
  python -m metabasis.scripts.star_beside_webtext_v3 --regression-proof \\
      --data-root /path/to/metabasis
  python -m metabasis.scripts.star_beside_webtext_v3 --run \\
      --data-root /path/to/metabasis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from metabasis.scripts.solve_portability_coefficients import (
    BAND_HALFWIDTH,
    ExchangeRateObservation,
    PortabilitySolution,
    solve_log_least_squares,
)
from metabasis.threads import thread_config_stamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("star_beside_webtext_v3")


# ---------------------------------------------------------------- constants
#: the sealed scored record of the v3 prediction ceremony. FULL, never a
#: SUPERSEDED-* sibling: those are earlier, wrong bytes kept only for the audit
#: trail and this lane refuses to open one by name (rake M26 at file grain).
SCORED_REL = Path("staging/webtext-v3-predictions/"
                  "SCORED-webtext-v3-2026-08-04-FULL.json")
#: the desk-sealed prediction artifact the scored record scores against.
PREDICTIONS_REL = Path("staging/webtext-v3-predictions/"
                       "PREDICTIONS-webtext-v3-2026-08-04.json")
OUT_DIR_REL = Path("staging/webtext-v3-reads/star-beside")
ARTIFACT_NAME = "STAR-BESIDE-webtext-v3-2026-08-04.json"
SIDECAR_NAME = "STAR-BESIDE-webtext-v3-2026-08-04.sidecar.json"

#: a filename this lane must never open — the superseded bytes beside the record.
FORBIDDEN_INPUT_PREFIX = "SUPERSEDED-"

#: the parent contract, read at its freeze tag. Quoted for provenance only.
PARENT_PREREG_SHA = ("33ba8290487813a80001c4cf5b71bbb57b57004ff610aa9fcf1f651005"
                     "875902")
#: the banked bootstrap star record — the parity-gate bar (LEDGER bootstrap/
#: parity-gate, PASSED 2026-07-26: "star_systems.json regenerated
#: BYTE-IDENTICAL to banked anchor (sha256 f5b8c4ee…)").
BANKED_STAR_SYSTEMS_REL = Path("outputs/battery/arms/A8_conjugation/smalls/"
                               "readouts_cpu/star_systems.json")
BANKED_STAR_SYSTEMS_SHA = ("f5b8c4eedcbcc379d55cdbbfd91a3e950b86aef73da0a01c9c56"
                           "86fee093b6b0")
#: the banked v2.1 in-lineage hub column (8 models, anchored c_8b = .8375).
BANKED_INLINEAGE_REL = Path("outputs/collection/readouts/"
                            "portability_inlineage_anchor_2026-07-27.json")

#: §8's bands, inherited verbatim. NON-GATE for the star.
PRIMARY_HALF_WIDTH = BAND_HALFWIDTH          # .05, from the lineage module
CO_PRIMARY_HALF_WIDTH = 0.025
NEAR_ZERO_ABS = 0.08

#: predictions come out of the lineage solver rounded to 4 dp; a verdict is
#: FRAGILE if this much slop could flip it. Reported, never silently absorbed.
ROUNDING_SLOP = 1e-4

#: ENACTOR-DECLARED descriptive grouping — pretrain lineage as the roster's own
#: notes record it, published WITH its membership so the desk can re-derive it or
#: overrule it. It is not a ruling and not a gate; the ONLY number that depends
#: on it is the `by_lineage_relation` breakdown that names it. Qwen2.5 and Qwen3
#: are kept APART (different pretrain runs); the pooled reading is reported too,
#: so the choice is visible rather than buried.
PRETRAIN_LINEAGE: dict[str, str] = {
    "llama-3.1-405b-instruct": "llama-3.x",
    "llama-3.1-70b-instruct": "llama-3.x",
    "llama-3.3-70b-instruct": "llama-3.x",
    "llama-3.1-8b-base": "llama-3.x",
    "qwen-7b": "qwen2.5",                 # bank key for Qwen2.5-7B-Instruct
    "qwen2.5-14b-instruct": "qwen2.5",
    "qwen2.5-72b-instruct": "qwen2.5",
    "qwen2.5-7b-base": "qwen2.5",
    "qwen3-30b-a3b": "qwen3",
    "mistral-7b-instruct-v0.3": "mistral",
    "mixtral-8x7b-instruct-v0.1": "mistral",
    "phi-3.5-mini-instruct": "phi",
    "phi-4": "phi",
    "dsv2-lite": "deepseek",
    "olmo2-7b-instruct": "olmo",
    "pythia-6.9b": "pythia",
}
#: the pooled alternative, quoted beside so the Qwen2.5/Qwen3 split is visible.
PRETRAIN_LINEAGE_POOLED_QWEN = dict(PRETRAIN_LINEAGE, **{"qwen3-30b-a3b": "qwen2.5"})

GATELESS_STATUS = (
    "UNSTAMPED · GATELESS · DESCRIPTIVE BY CONSTRUCTION (v3 contract §8, "
    "'the star beside (S1 continuation)'). This record scores NOTHING, moves no "
    "band, opens no gate and adjudicates nothing. S1 is STAMPED — 'scalar "
    "factorization refuted at scale; hub-mediated transport demonstrated at map "
    "order' (Luxia, 2026-07-29) — and this lane CONTINUES that record on the "
    "clean basis; it cannot reopen it. Every in-band count below is "
    "REFERENCE-ONLY and NON-GATE.")

ADJUDICATION = (
    "NONE — every verdict, ruling, stamp and adjudication is the desk's and "
    "Luxia's. This lane computes arithmetic and declares nothing of record.")


class StarBesideError(RuntimeError):
    """HALT. Raised, never sys.exit()ed, so a sweep survives it (rake M45)."""


# ---------------------------------------------------------------- data model
class SlotStar(BaseModel):
    """ONE sealed slot, with the star beside the composed column of record."""
    model_config = {"extra": "forbid"}

    ordinal: int
    prediction_id: str
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    family: str
    pair_class: str
    system: str
    extension_line: bool
    observed: float
    observed_direction: str

    star_predicted: float
    #: SIGNED, observed − predicted (the sealed record's own convention).
    star_error_signed: float
    star_error: float                      # = |star_error_signed|
    star_in_band_primary: bool
    star_in_band_co_primary: bool
    star_miss_side: Literal["", "high", "low"]
    star_verdict_fragile: bool

    composed_hub_of_record: str
    composed_predicted: float
    #: SIGNED, read verbatim from the sealed scored record's `error` field.
    composed_error_signed: float
    composed_error: float                  # = |composed_error_signed|
    composed_in_band_primary: bool
    composed_in_band_co_primary: bool

    concordance: Literal["both-hit", "composed-only", "star-only", "both-miss"]


class CoefficientRow(BaseModel):
    """One c_M, in the system it was solved in. Never quoted without its label."""
    model_config = {"extra": "forbid"}

    model_name: str
    system: str
    arm: str
    family: str
    c_M: float
    n_slots: int
    n_unordered_pairs: int
    median_abs_residual: float
    max_abs_residual: float
    n_out_of_band_primary: int


class SystemSummary(BaseModel):
    model_config = {"extra": "forbid"}

    system: str
    arm: str
    family: str
    n_models: int
    n_slots_fit: int
    n_unordered_pairs: int
    design_rank: int
    nullspace_dim: int
    gauge: str
    gauge_fixed: bool
    anchor_used: str
    n_free_parameters: int
    over_identification_dof: int
    rms_residual_a_hat: float
    max_abs_residual_a_hat: float
    median_abs_residual_a_hat: float
    rms_residual_log: float
    excluded: list[dict]
    solver_notes: list[str]


class HeadToHead(BaseModel):
    model_config = {"extra": "forbid"}

    n_slots: int
    star_in_band_primary: int
    composed_in_band_primary: int
    star_fraction_primary: float
    composed_fraction_primary: float
    delta_percentage_points: float
    star_in_band_co_primary: int
    composed_in_band_co_primary: int
    both_hit: int
    composed_only: int
    star_only: int
    both_miss: int
    discordant: int
    sign_flip_exponent: Optional[int]
    star_median_abs_error: float
    composed_median_abs_error: float
    star_max_abs_error: float
    composed_max_abs_error: float


# ---------------------------------------------------------------- helpers
def sha256_of_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _system_key(arm: str, family: str) -> str:
    return f"{arm}::{family}"


def _round(x: float, dp: int = 4) -> float:
    return round(float(x), dp)


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


# ---------------------------------------------------------------- ingestion
def load_scored_record(data_root: Path,
                       scored_rel: Path = SCORED_REL,
                       predictions_rel: Path = PREDICTIONS_REL) -> dict:
    """Open the sealed scored record — after verifying the desk's seal.

    Three independent checks, all BINDING here even though nothing downstream is
    a gate: (1) the named input is not a SUPERSEDED-* sibling; (2) the §8.2 desk
    stamp verifies over the prediction artifact by VALUE, recomputed here through
    the ceremony's own `verify_prediction_stamp`; (3) the scored record's own
    embedded stamp block agrees with that fresh verification AND its per-slot
    of-record predictions reproduce the sealed artifact's, slot for slot. A
    scored file that has drifted from the artifact the desk sealed is refused.
    """
    scored_path = data_root / scored_rel
    predictions_path = data_root / predictions_rel
    for p in (scored_path, predictions_path):
        if p.name.startswith(FORBIDDEN_INPUT_PREFIX):
            raise StarBesideError(
                f"REFUSING {p.name}: a SUPERSEDED-* file is earlier, wrong bytes "
                f"kept for the audit trail. This lane opens the record by name, "
                f"never a sibling that merely looks like it")
        if not p.is_file():
            raise StarBesideError(f"input absent: {p}")

    #  Imported lazily: `read_composed_predictions` is the ceremony's own
    #  machinery and a heavy module; the adapter needs only its stamp check.
    from metabasis.scripts.read_composed_predictions import (
        load_v3_prediction_artifact, verify_prediction_stamp)

    _stamp, verification = verify_prediction_stamp(predictions_path)
    if not (verification.sha_matches and verification.filename_matches):
        raise StarBesideError("stamp verification did not pass — refusing")

    try:
        scored = json.loads(scored_path.read_text())
    except (OSError, ValueError) as exc:
        raise StarBesideError(f"unreadable scored record {scored_path}: {exc}"
                              ) from exc
    if scored.get("record") != "webtext-v3-scored-record/v1":
        raise StarBesideError(
            f"{scored_path.name} is not a webtext-v3 scored record "
            f"(record={scored.get('record')!r})")

    embedded = scored.get("stamp_verification", {})
    if embedded.get("sha256_sealed") != verification.sha256_recomputed:
        raise StarBesideError(
            f"HALT — the scored record was written against sealed sha "
            f"{embedded.get('sha256_sealed')} but the artifact on disk hashes to "
            f"{verification.sha256_recomputed}. Two byte sequences and one "
            f"scoring pass is not a verification")
    if int(scored.get("frozen_count_N", -1)) != verification.frozen_count_N:
        raise StarBesideError("frozen N disagrees between record and stamp")

    artifact, _ = load_v3_prediction_artifact(predictions_path)
    sealed_of_record: dict[str, float] = {}
    for slot in artifact.slots:
        for hub, col in slot.columns.items():
            if col.of_record:
                sealed_of_record[slot.prediction_id] = float(col.a_comp)
    drift: list[str] = []
    for slot in scored["slots"]:
        pid = slot["prediction_id"]
        if pid not in sealed_of_record:
            drift.append(f"{pid}: absent from the sealed artifact")
            continue
        for col in slot["columns"]:
            if col.get("of_record"):
                if abs(float(col["a_comp_full_precision"])
                       - sealed_of_record[pid]) > 1e-12:
                    drift.append(f"{pid}: of-record a_comp drifted")
    if drift:
        raise StarBesideError(
            f"HALT — the scored record disagrees with the sealed artifact on "
            f"{len(drift)} slot(s); first: {drift[0]}")
    logger.info("scored record VERIFIED against the sealed artifact: %d slots, "
                "of-record column reproduces slot-for-slot", len(scored["slots"]))
    return scored


def observations_from_scored(scored: dict) -> dict[str, list[ExchangeRateObservation]]:
    """Sealed slots → the lineage's own observation rows, grouped by system.

    One row per SLOT (both directions of every unordered pair are separate
    sealed predictions, so both enter). The star form is symmetric, so the two
    directions share a design row and differ only in b — which is exactly where
    the directional asymmetry lands in the over-identification residual, rather
    than being averaged away out of sight.
    """
    groups: dict[str, list[ExchangeRateObservation]] = defaultdict(list)
    seen: set[str] = set()
    for slot in scored["slots"]:
        pid = str(slot["prediction_id"])
        if pid in seen:
            raise StarBesideError(f"duplicate prediction_id in the record: {pid}")
        seen.add(pid)
        obs = ExchangeRateObservation(
            source=str(slot["source_model"]),
            target=str(slot["target_model"]),
            a_hat=float(slot["observed"]),
            arm=str(slot["arm"]),
            family=str(slot["family"]),
            pair_id=pid,
            rank_forbidden=False,
            clears_null_floor=None)
        groups[_system_key(obs.arm, obs.family)].append(obs)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------- the solve
def solve_v3_systems(groups: dict[str, list[ExchangeRateObservation]]
                     ) -> dict[str, PortabilitySolution]:
    """One `solve_log_least_squares` per (arm × family). Refuses an open gauge.

    The lineage solver is called unmodified. The only thing this wrapper adds is
    the REFUSAL: a v3 system whose graph leaves a gauge freedom would need an
    anchor, the only anchors in existence are v2.1 vintage, and the vintage rule
    forbids them — so such a system is not solvable here and is not quietly
    min-normed into an artifact.
    """
    out: dict[str, PortabilitySolution] = {}
    for key, obs in groups.items():
        sol = solve_log_least_squares(obs)          # NO anchors: vintage rule
        if not sol.gauge_fixed or sol.nullspace_dim != 0:
            raise StarBesideError(
                f"HALT — system {key} is gauge-open (nullspace_dim="
                f"{sol.nullspace_dim}): {sol.gauge}. It would need an anchor, "
                f"and every anchor in existence is v2.1 vintage. The vintage "
                f"rule is absolute; refusing rather than publishing a min-norm "
                f"representative of a whole family as if it were a constant")
        if sol.excluded:
            logger.warning("system %s excluded %d observation(s) by name",
                           key, len(sol.excluded))
        out[key] = sol
    return out


def _log_rms(sol: PortabilitySolution) -> float:
    vals = [math.log(r.observed) - math.log(r.predicted)
            for r in sol.residuals if r.observed > 0 and r.predicted > 0]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def summarize_system(key: str, sol: PortabilitySolution,
                     obs: Sequence[ExchangeRateObservation]) -> SystemSummary:
    unordered = {tuple(sorted((o.source, o.target))) for o in obs}
    errs = [r.abs_error for r in sol.residuals]
    return SystemSummary(
        system=key, arm=sol.arm, family=sol.family,
        n_models=sol.n_models, n_slots_fit=sol.n_pairs_used,
        n_unordered_pairs=len(unordered),
        design_rank=sol.design_rank, nullspace_dim=sol.nullspace_dim,
        gauge=sol.gauge, gauge_fixed=sol.gauge_fixed,
        anchor_used=("NONE — the pair graph fixes the gauge (parent §3's rule: "
                     "once the graph closes, anchors are consistency checks, not "
                     "constraints). No v2.1 constant enters."),
        n_free_parameters=sol.n_models,
        over_identification_dof=len(unordered) - sol.n_models,
        rms_residual_a_hat=_round(sol.rms_residual or float("nan")),
        max_abs_residual_a_hat=_round(sol.max_abs_residual or float("nan")),
        median_abs_residual_a_hat=_round(_median(errs)),
        rms_residual_log=_round(_log_rms(sol)),
        excluded=[e.model_dump() for e in sol.excluded],
        solver_notes=list(sol.notes))


# ---------------------------------------------------------------- scoring
def score_slots(scored: dict,
                solutions: dict[str, PortabilitySolution]) -> list[SlotStar]:
    """Star beside composed, slot for slot, on the IDENTICAL sealed 240."""
    pred_by_pid: dict[str, float] = {}
    for key, sol in solutions.items():
        for r in sol.residuals:
            if r.pair_id in pred_by_pid:
                raise StarBesideError(f"pair_id {r.pair_id} in two systems")
            pred_by_pid[r.pair_id] = float(r.predicted)

    rows: list[SlotStar] = []
    for slot in scored["slots"]:
        pid = str(slot["prediction_id"])
        if pid not in pred_by_pid:
            raise StarBesideError(f"no star prediction for {pid}")
        obs = float(slot["observed"])
        star = pred_by_pid[pid]
        s_err_signed = obs - star            # the sealed record's own convention
        s_err = abs(s_err_signed)
        s_in = s_err <= PRIMARY_HALF_WIDTH
        s_in_co = s_err <= CO_PRIMARY_HALF_WIDTH
        side: Literal["", "high", "low"] = ""
        if not s_in:
            #  "miss HIGH" = the OBSERVED sits above the band, i.e. the star
            #  UNDER-predicts. This is S1/S4's stamped signature and the sense
            #  the ledger uses throughout ("all misses HIGH", "systematic
            #  under-prediction"). Named here so it cannot be read backwards.
            side = "high" if obs > star else "low"

        col = next((c for c in slot["columns"] if c.get("of_record")), None)
        if col is None:
            raise StarBesideError(f"{pid} has no of-record composed column")
        #  `error` in the sealed record is SIGNED (observed − filed prediction).
        #  Absolute values are taken here, never assumed.
        c_err_signed = float(col["error"])
        c_err = abs(c_err_signed)
        c_in = bool(col["in_band_primary"])
        c_in_co = bool(col["in_band_co_primary"])

        if s_in and c_in:
            conc = "both-hit"
        elif c_in and not s_in:
            conc = "composed-only"
        elif s_in and not c_in:
            conc = "star-only"
        else:
            conc = "both-miss"

        rows.append(SlotStar(
            ordinal=int(slot["ordinal"]), prediction_id=pid,
            pair_id=str(slot["pair_id"]),
            source_model=str(slot["source_model"]),
            source_site=int(slot["source_site"]),
            target_model=str(slot["target_model"]),
            target_site=int(slot["target_site"]),
            arm=str(slot["arm"]), family=str(slot["family"]),
            pair_class=str(slot["pair_class"]),
            system=_system_key(str(slot["arm"]), str(slot["family"])),
            extension_line=bool(slot["extension_line"]),
            observed=_round(obs, 6), observed_direction=str(slot["observed_direction"]),
            star_predicted=_round(star), star_error_signed=_round(s_err_signed),
            star_error=_round(s_err),
            star_in_band_primary=s_in, star_in_band_co_primary=s_in_co,
            star_miss_side=side,
            star_verdict_fragile=bool(
                abs(s_err - PRIMARY_HALF_WIDTH) <= ROUNDING_SLOP),
            composed_hub_of_record=str(col["hub"]),
            composed_predicted=_round(float(col["a_comp_full_precision"])),
            composed_error_signed=_round(c_err_signed),
            composed_error=_round(c_err),
            composed_in_band_primary=c_in, composed_in_band_co_primary=c_in_co,
            concordance=conc))
    return rows


def head_to_head(rows: Sequence[SlotStar]) -> HeadToHead:
    n = len(rows)
    s_in = sum(r.star_in_band_primary for r in rows)
    c_in = sum(r.composed_in_band_primary for r in rows)
    comp_only = sum(r.concordance == "composed-only" for r in rows)
    star_only = sum(r.concordance == "star-only" for r in rows)
    disc = comp_only + star_only
    #  The E3 sign-flip read, frozen at S2: under an exchangeable null every
    #  discordant slot is a coin flip, so all-one-way over d slots is 2^-d.
    #  Reported as the exponent only; the READING is the desk's.
    flip = disc if (disc > 0 and min(comp_only, star_only) == 0) else None
    return HeadToHead(
        n_slots=n,
        star_in_band_primary=s_in, composed_in_band_primary=c_in,
        star_fraction_primary=_round(s_in / n, 6),
        composed_fraction_primary=_round(c_in / n, 6),
        delta_percentage_points=_round(100.0 * (c_in - s_in) / n, 4),
        star_in_band_co_primary=sum(r.star_in_band_co_primary for r in rows),
        composed_in_band_co_primary=sum(r.composed_in_band_co_primary for r in rows),
        both_hit=sum(r.concordance == "both-hit" for r in rows),
        composed_only=comp_only, star_only=star_only,
        both_miss=sum(r.concordance == "both-miss" for r in rows),
        discordant=disc, sign_flip_exponent=flip,
        star_median_abs_error=_round(_median([r.star_error for r in rows])),
        composed_median_abs_error=_round(_median([r.composed_error for r in rows])),
        star_max_abs_error=_round(max(r.star_error for r in rows)),
        composed_max_abs_error=_round(max(r.composed_error for r in rows)))


# ---------------------------------------------------------------- structure
def _magnitude_trend(rows: Sequence[SlotStar]) -> dict:
    """Does the star COMPRESS the range? The saturation signature, measured.

    The v2.1 campaign's standing G-star-structure expectation (rake-level, and
    S4's stamped wording) was that misses cluster HIGH on large â. In-sample the
    residual is centred, so the surviving form of that claim is a TREND: signed
    error rising with observed magnitude means the star over-predicts small
    exchange rates and under-predicts large ones — a rank-one form flattening
    the spread it is asked to reproduce. Plain Pearson correlation and terciles;
    no model is fitted here.
    """
    obs = np.array([r.observed for r in rows], dtype=np.float64)
    sgn = np.array([r.star_error_signed for r in rows], dtype=np.float64)
    r_pearson = (float(np.corrcoef(obs, sgn)[0, 1])
                 if obs.size > 1 and obs.std() > 0 and sgn.std() > 0
                 else float("nan"))
    order = np.argsort(obs, kind="stable")
    thirds = np.array_split(order, 3)
    terciles = []
    for i, idx in enumerate(thirds):
        sel = [rows[j] for j in idx]
        if not sel:                       # fewer rows than terciles (selftest)
            continue
        terciles.append({
            "tercile": ["low", "mid", "high"][i],
            "n_slots": len(sel),
            "observed_range": [_round(min(r.observed for r in sel)),
                               _round(max(r.observed for r in sel))],
            "median_star_signed_error": _round(
                _median([r.star_error_signed for r in sel])),
            "n_star_miss": sum(not r.star_in_band_primary for r in sel),
            "n_miss_high": sum(r.star_miss_side == "high" for r in sel),
            "n_miss_low": sum(r.star_miss_side == "low" for r in sel),
        })
    return {
        "what": ("Pearson correlation of the star's SIGNED error "
                 "(observed − predicted) with the observed exchange rate, plus "
                 "terciles. Positive = the star under-predicts large â and "
                 "over-predicts small â, i.e. it compresses the range."),
        "pearson_r_signed_error_vs_observed": _round(r_pearson),
        "terciles": terciles,
    }


def _lineage_relation(rows: Sequence[SlotStar], rate_fn) -> dict:
    """Same-pretrain-lineage vs cross-lineage — the S1 in-family finding, re-read.

    The v2.1 hub-steelman read named it: "hub-star form wrong for in-family
    pairs". Same-lineage endpoints transport BETTER than a product of two scalars
    allows, because they share chart structure the scalar cannot carry. Grouping
    is the enactor-declared `PRETRAIN_LINEAGE` above, membership published.
    """
    out: dict = {"grouping": PRETRAIN_LINEAGE,
                 "STATUS": "ENACTOR-DECLARED descriptive grouping, membership "
                           "published; not a ruling and not a gate."}
    for name, table in (("qwen2.5_and_qwen3_separate", PRETRAIN_LINEAGE),
                        ("qwen_pooled", PRETRAIN_LINEAGE_POOLED_QWEN)):
        bucket: dict[str, list[SlotStar]] = defaultdict(list)
        unknown: set[str] = set()
        for r in rows:
            for m in (r.source_model, r.target_model):
                if m not in table:
                    unknown.add(m)
            same = table.get(r.source_model, "?A") == table.get(r.target_model, "?B")
            bucket[f"{'same' if same else 'cross'}-lineage [{r.system}]"].append(r)
            bucket["same-lineage (both arms)" if same
                   else "cross-lineage (both arms)"].append(r)
        out[name] = {"rows": rate_fn(bucket),
                     "models_absent_from_the_grouping": sorted(unknown)}
    return out


def miss_structure(rows: Sequence[SlotStar], scored: dict) -> dict:
    """Where the star's misses concentrate — S1/S4's signature, re-read on v3."""
    misses = [r for r in rows if not r.star_in_band_primary]
    hits = [r for r in rows if r.star_in_band_primary]

    def _rate(bucket: dict[str, list[SlotStar]]) -> list[dict]:
        out = []
        for k in sorted(bucket):
            v = bucket[k]
            m = [r for r in v if not r.star_in_band_primary]
            out.append({
                "label": k, "n_slots": len(v), "n_star_miss": len(m),
                "miss_rate": _round(len(m) / len(v), 4),
                "n_miss_high": sum(r.star_miss_side == "high" for r in m),
                "n_miss_low": sum(r.star_miss_side == "low" for r in m),
                "median_star_abs_error": _round(_median([r.star_error for r in v])),
                "median_star_signed_error": _round(
                    _median([r.star_error_signed for r in v])),
                "median_composed_abs_error": _round(
                    _median([r.composed_error for r in v])),
            })
        return out

    by: dict[str, dict[str, list[SlotStar]]] = {
        "by_system": defaultdict(list), "by_pair_class": defaultdict(list),
        "by_line": defaultdict(list), "by_direction": defaultdict(list),
        "by_observed_decile": defaultdict(list),
    }
    for r in rows:
        by["by_system"][r.system].append(r)
        by["by_pair_class"][r.pair_class].append(r)
        by["by_line"]["extension" if r.extension_line else "core"].append(r)
        by["by_direction"][r.observed_direction].append(r)
        by["by_observed_decile"][f"{math.floor(r.observed * 10) / 10:.1f}-"
                                 f"{math.floor(r.observed * 10) / 10 + 0.1:.1f}"
                                 ].append(r)

    per_model: dict[str, list[SlotStar]] = defaultdict(list)
    for r in rows:
        per_model[f"{r.source_model} [{r.system}]"].append(r)
        per_model[f"{r.target_model} [{r.system}]"].append(r)

    #  The symmetric star cannot represent direction: for an unordered pair
    #  whose two sealed directions differ by more than 2×.05, ONE of the two
    #  slots is out of band no matter what constant is hung on either endpoint.
    #  That is an ARITHMETIC ceiling on the star, independent of the fit.
    asym: dict[tuple[str, str], list[SlotStar]] = defaultdict(list)
    for r in rows:
        asym[tuple(sorted((r.source_model, r.target_model)))].append(r)
    gaps = []
    forced = 0
    for pair, rs in sorted(asym.items()):
        if len(rs) != 2:
            continue
        gap = abs(rs[0].observed - rs[1].observed)
        gaps.append(gap)
        if gap > 2 * PRIMARY_HALF_WIDTH:
            forced += 1

    return {
        "definition_miss_high": (
            "OBSERVED above the band, i.e. the star UNDER-predicts — the sense "
            "S1/S4 use ('misses cluster as systematic under-prediction')."),
        "n_star_miss": len(misses),
        "n_miss_high": sum(r.star_miss_side == "high" for r in misses),
        "n_miss_low": sum(r.star_miss_side == "low" for r in misses),
        "all_misses_one_sided_high": bool(
            misses and all(r.star_miss_side == "high" for r in misses)),
        "s1_signature_note": (
            "S1/S4's v2.1 signature was ALL misses HIGH — a systematic "
            "under-prediction that came from a HUB-derived, externally anchored "
            "column. The v3 star is fit IN-SAMPLE by least squares on the very "
            "slots it is scored on, so its residual is centred BY CONSTRUCTION "
            "and a pure one-sided miss set is arithmetically unavailable. The "
            "comparable reading is therefore the CONDITIONAL structure below "
            "(by observed decile / by direction), not the raw high-vs-low "
            "count."),
        "mean_star_signed_error": _round(
            float(np.mean([r.star_error_signed for r in rows]))),
        "mean_star_signed_error_note": (
            "≈0 by construction — the log-least-squares fit centres the "
            "residual. Quoted so the centring is visible rather than assumed."),
        "median_observed_at_hits": _round(_median([r.observed for r in hits])),
        "median_observed_at_misses": _round(_median([r.observed for r in misses])),
        "median_star_error_at_misses": _round(
            _median([r.star_error for r in misses])),
        "magnitude_trend": _magnitude_trend(rows),
        "by_lineage_relation": _lineage_relation(rows, _rate),
        "by_system": _rate(by["by_system"]),
        "by_pair_class": _rate(by["by_pair_class"]),
        "by_line": _rate(by["by_line"]),
        "by_direction": _rate(by["by_direction"]),
        "by_observed_decile": _rate(by["by_observed_decile"]),
        "by_model_endpoint": _rate(per_model),
        "directional_ceiling": {
            "what": ("the star form â = c_A·c_B is SYMMETRIC and the sealed set "
                     "files both directions of every unordered pair as separate "
                     "predictions. Where the two observed values differ by more "
                     "than the full band width (2 × .05), no symmetric predictor "
                     "can hold both — an arithmetic ceiling, prior to any fit."),
            "n_unordered_pairs": len(gaps),
            "median_abs_fwd_minus_rev": _round(_median(gaps)),
            "max_abs_fwd_minus_rev": _round(max(gaps)) if gaps else float("nan"),
            "n_pairs_forcing_at_least_one_miss": forced,
            "max_attainable_in_band_slots": len(rows) - forced,
        },
    }


# ------------------------------------------------- the held-out star beside
def held_out_beside(groups: dict[str, list[ExchangeRateObservation]],
                    rows: Sequence[SlotStar]) -> dict:
    """Leave-one-UNORDERED-PAIR-out, through the lineage solver's own held-out path.

    The headline star count is IN-SAMPLE (the star is fit on the very slots it is
    scored on) while the composed column was sealed before the first pair fit. To
    put the two on a comparable footing this drops BOTH directions of one
    unordered pair, re-solves on the rest with the same unmodified solver, and
    scores the two dropped slots through `held_out_checks`. No new math: the
    lineage's own `held_out=` path, run once per pair.
    """
    by_pid = {r.prediction_id: r for r in rows}
    results: list[dict] = []
    skipped: list[dict] = []
    for key, obs in groups.items():
        pairs: dict[tuple[str, str], list[ExchangeRateObservation]] = defaultdict(list)
        for o in obs:
            pairs[tuple(sorted((o.source, o.target)))].append(o)
        for pair in sorted(pairs):
            held = pairs[pair]
            keep = [o for o in obs if o.key() not in {h.key() for h in held}]
            sol = solve_log_least_squares(keep, held_out=held)
            if not sol.gauge_fixed or sol.nullspace_dim != 0:
                skipped.append({"system": key, "pair": list(pair),
                                "reason": f"dropping this pair opens the gauge "
                                          f"(nullspace {sol.nullspace_dim})"})
                continue
            missing = [h.key() for h in held
                       if h.key() not in {c.pair_id for c in sol.held_out_checks}]
            if missing:
                skipped.append({"system": key, "pair": list(pair),
                                "reason": f"endpoint left the solved set: {missing}"})
                continue
            for chk in sol.held_out_checks:
                row = by_pid[chk.pair_id]
                results.append({
                    "prediction_id": chk.pair_id, "system": key,
                    "star_predicted_heldout": _round(chk.predicted),
                    "observed": _round(chk.observed),
                    "abs_error": _round(chk.abs_error),
                    "in_band_primary": bool(chk.in_band),
                    "composed_in_band_primary": row.composed_in_band_primary,
                })
    n = len(results)
    n_in = sum(r["in_band_primary"] for r in results)
    return {
        "STATUS": "NON-GATE — descriptive beside, the parent's own held-out rule "
                  "applied to a set that has no held-out slots left to spare.",
        "construction": "leave-one-unordered-pair-out (both directions dropped "
                        "together); re-solve; score the two dropped slots.",
        "n_scored": n,
        "n_in_band_primary": n_in,
        "fraction_in_band_primary": _round(n_in / n, 6) if n else float("nan"),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "median_abs_error": _round(_median([r["abs_error"] for r in results])),
        "max_abs_error": _round(max((r["abs_error"] for r in results), default=float("nan"))),
        "rows": results,
    }


# ---------------------------------------------------------------- coefficients
def coefficient_table(solutions: dict[str, PortabilitySolution],
                      rows: Sequence[SlotStar]) -> list[CoefficientRow]:
    per_model_slots: dict[tuple[str, str], list[SlotStar]] = defaultdict(list)
    for r in rows:
        per_model_slots[(r.system, r.source_model)].append(r)
        per_model_slots[(r.system, r.target_model)].append(r)

    out: list[CoefficientRow] = []
    for key in sorted(solutions):
        sol = solutions[key]
        resid_by_pid = {r.pair_id: r.abs_error for r in sol.residuals}
        for model_name in sorted(sol.coefficients):
            mine = per_model_slots[(key, model_name)]
            errs = [resid_by_pid[r.prediction_id] for r in mine
                    if r.prediction_id in resid_by_pid]
            unordered = {tuple(sorted((r.source_model, r.target_model)))
                         for r in mine}
            out.append(CoefficientRow(
                model_name=model_name, system=key, arm=sol.arm, family=sol.family,
                c_M=sol.coefficients[model_name],
                n_slots=len(mine), n_unordered_pairs=len(unordered),
                median_abs_residual=_round(_median(errs)),
                max_abs_residual=_round(max(errs)) if errs else float("nan"),
                n_out_of_band_primary=sum(
                    1 for r in mine if not r.star_in_band_primary)))
    return out


# ---------------------------------------------------------- regression proof
def regression_proof_v21(data_root: Path) -> dict:
    """THE ONE LEGITIMATE v2.1 TOUCH. Must pass before a v3 number is quoted.

    Four checks, strongest first:

    R1  the bootstrap PARITY-GATE bar itself — `solve_star_systems` re-run
        end-to-end from the banked A8 maps and V7 vectors into a scratch
        directory, and the resulting `star_systems.json` compared BYTE-FOR-BYTE
        against the banked artifact (LEDGER bootstrap/parity-gate: sha256
        f5b8c4ee…). Nothing under outputs/ is written or touched.
    R2  THIS MODULE'S OWN ADAPTER PATH on the banked v2.1 native::proc_k128 â
        inputs: the same observation-builder shape the v3 lane uses, handed to
        the same lineage solver, must return the banked constants of record
        (c_3B .6382 · c_8B .7545 · c_Qwen .4730 · c_DSV2 .4415) and reproduce
        the banked out-of-sample check (3b→dsv2-lite predicted .2818 vs
        observed .2773, |e| .0045, in ±.05).
    R3  the banked v2.1 IN-LINEAGE hub column (8 models, anchored c_8b = .8375 —
        the anchored/bipartite branch the v3 lane never takes) reproduced
        coefficient for coefficient from its own banked â rows.
    R4  the lineage module's own selftest, run in this configuration (rake M44:
        a count that does not name its configuration vouches for nothing).
    """
    from metabasis.scripts import solve_star_systems as sss
    from metabasis.scripts import solve_portability_coefficients as spc

    banked_path = data_root / BANKED_STAR_SYSTEMS_REL
    if not banked_path.is_file():
        raise StarBesideError(f"banked star record absent: {banked_path}")
    banked_sha = sha256_of_path(banked_path)
    banked = json.loads(banked_path.read_text())

    checks: list[dict] = []

    # ---- R1: the parity-gate bar, byte for byte -------------------------
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="star_beside_paritygate_") as td:
        saved_out = sss.OUT
        try:
            sss.OUT = Path(td)
            os.chdir(data_root)
            rc = sss.main()
        finally:
            sss.OUT = saved_out
            os.chdir(cwd)
        regen = Path(td) / "star_systems.json"
        regen_sha = sha256_of_path(regen) if regen.is_file() else ""
    checks.append({
        "id": "R1",
        "what": "bootstrap parity gate — solve_star_systems regenerated from the "
                "banked A8 maps/vectors and compared byte-for-byte",
        "expected_sha256": BANKED_STAR_SYSTEMS_SHA,
        "banked_on_disk_sha256": banked_sha,
        "regenerated_sha256": regen_sha,
        "exit_code": rc,
        "PASS": bool(rc == 0 and regen_sha == BANKED_STAR_SYSTEMS_SHA
                     and banked_sha == BANKED_STAR_SYSTEMS_SHA),
    })

    # ---- R2: this module's adapter path on the banked v2.1 inputs -------
    sysblk = banked["systems"]["native::proc_k128"]
    a_in = sysblk["inputs_a_hat"]
    fit_labels = ("3b->8b", "8b->qwen-7b", "3b->qwen-7b", "8b->dsv2-lite")
    check_label = "3b->dsv2-lite"
    fit_obs = [_obs_from_label(lbl, a_in[lbl]) for lbl in fit_labels]
    ho_obs = [_obs_from_label(check_label, a_in[check_label])]
    sol = solve_log_least_squares(fit_obs, held_out=ho_obs)
    want_c = {"3b": sysblk["constants"]["c_3B"], "8b": sysblk["constants"]["c_8B"],
              "qwen-7b": sysblk["constants"]["c_Qwen"],
              "dsv2-lite": sysblk["constants"]["c_DSV2"]}
    got_c = {m: sol.coefficients.get(m) for m in want_c}
    worst = max(abs((got_c[m] or 0.0) - want_c[m]) for m in want_c)
    want_chk = sysblk["out_of_sample_check"]
    chk = sol.held_out_checks[0] if sol.held_out_checks else None
    checks.append({
        "id": "R2",
        "what": "this module's observation-builder + the lineage solver on the "
                "banked v2.1 native::proc_k128 â inputs",
        "banked_constants": want_c, "reproduced_constants": got_c,
        "worst_abs_delta_c": _round(worst, 8),
        "banked_out_of_sample": {"predicted": want_chk["predicted"],
                                 "observed": want_chk["observed"],
                                 "abs_error": want_chk["abs_error"],
                                 "within_pm_0.05": want_chk["within_pm_0.05"]},
        "reproduced_out_of_sample": (
            {"predicted": chk.predicted, "observed": chk.observed,
             "abs_error": chk.abs_error, "within_pm_0.05": chk.in_band}
            if chk else None),
        "PASS": bool(chk is not None and worst <= 1e-4
                     and abs(chk.predicted - want_chk["predicted"]) <= 1e-4
                     and abs(chk.observed - want_chk["observed"]) <= 1e-4
                     and abs(chk.abs_error - want_chk["abs_error"]) <= 1e-4
                     and chk.in_band == want_chk["within_pm_0.05"]),
    })

    # ---- R3: the banked in-lineage anchored hub column -------------------
    inl_path = data_root / BANKED_INLINEAGE_REL
    if not inl_path.is_file():
        raise StarBesideError(f"banked in-lineage column absent: {inl_path}")
    inl = json.loads(inl_path.read_text())
    anchor_model, anchor_value = "8b", float(inl["anchors"]["8b"])
    inl_obs: list[ExchangeRateObservation] = []
    for r in inl["residuals"]:
        pid = r["pair_id"]
        src_lbl, _, tgt_lbl = pid.partition("->")
        inl_obs.append(ExchangeRateObservation(
            source=_strip_site(src_lbl), target=_strip_site(tgt_lbl),
            a_hat=float(r["observed"]), arm=inl["arm"], family=inl["family"],
            pair_id=pid))
    inl_sol = solve_log_least_squares(inl_obs, anchors={anchor_model: anchor_value})
    inl_worst = max(abs(inl_sol.coefficients.get(m, float("nan")) - v)
                    for m, v in inl["coefficients"].items())
    checks.append({
        "id": "R3",
        "what": "the banked v2.1 in-lineage hub column (anchored branch) "
                "reproduced from its own banked â rows",
        "n_models": len(inl["coefficients"]),
        "banked_coefficients": inl["coefficients"],
        "reproduced_coefficients": inl_sol.coefficients,
        "worst_abs_delta_c": _round(inl_worst, 8),
        "gauge_fixed": inl_sol.gauge_fixed,
        "nullspace_dim": inl_sol.nullspace_dim,
        "PASS": bool(inl_worst <= 1e-4 and inl_sol.gauge_fixed
                     and inl_sol.nullspace_dim == int(inl["nullspace_dim"])),
    })

    # ---- R4: the lineage module's own selftest, configuration named ------
    rc4 = spc.selftest()
    checks.append({
        "id": "R4",
        "what": "solve_portability_coefficients.selftest() — the lineage's own "
                "8-part checklist, including its cross-check against the banked "
                "closed-form solver and the banked held-out value",
        "configuration": {
            "cwd": "data root (outputs/ present)",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        },
        "exit_code": rc4,
        "PASS": rc4 == 0,
    })

    return {
        "STATUS": "THE PARITY-GATE BAR. The vintage rule forbids v2.1 constants "
                  "in the v3 solve; this is the one legitimate v2.1 touch — a "
                  "REGRESSION PROOF that the adapter/solver path used for v3 "
                  "still reproduces the banked v2.1 star record exactly. No v3 "
                  "number is quoted unless every check below PASSES.",
        "banked_record": str(BANKED_STAR_SYSTEMS_REL),
        "banked_record_sha256": banked_sha,
        "checks": checks,
        "ALL_PASS": all(c["PASS"] for c in checks),
    }


def _strip_site(label: str) -> str:
    """`8bL16` → `8b`. The banked pair_ids carry the site; the model is the key."""
    head = label
    for i in range(len(label) - 1, 0, -1):
        if label[i] == "L" and label[i + 1:].isdigit():
            head = label[:i]
            break
    return head


_V21_PAIR_MODELS = {
    "3b->8b": ("3b", "8b"), "8b->qwen-7b": ("8b", "qwen-7b"),
    "3b->qwen-7b": ("3b", "qwen-7b"), "8b->dsv2-lite": ("8b", "dsv2-lite"),
    "3b->dsv2-lite": ("3b", "dsv2-lite"),
}


def _obs_from_label(label: str, a_hat: float) -> ExchangeRateObservation:
    src, tgt = _V21_PAIR_MODELS[label]
    return ExchangeRateObservation(source=src, target=tgt, a_hat=float(a_hat),
                                   arm="native", family="proc_k128", pair_id=label)


# ---------------------------------------------------------------- the artifact
def build_artifact(data_root: Path, scored: dict,
                   groups: dict[str, list[ExchangeRateObservation]],
                   solutions: dict[str, PortabilitySolution],
                   rows: list[SlotStar],
                   proof: dict) -> dict:
    """The deterministic artifact. NO timestamp, NO absolute path, NO env value."""
    summaries = [summarize_system(k, solutions[k], groups[k]).model_dump()
                 for k in sorted(solutions)]
    coeffs = [c.model_dump() for c in coefficient_table(solutions, rows)]
    h2h = head_to_head(rows)

    sol_dumps: dict[str, dict] = {}
    for k in sorted(solutions):
        d = solutions[k].model_dump()
        #  `generated` is today's date and `thread_config` reads the live
        #  threadpool: both belong in the sidecar, never in bytes that must be
        #  build-twice identical.
        d["generated"] = "(sidecar)"
        d["thread_config"] = "(sidecar)"
        sol_dumps[k] = d

    gate_free = [
        "NON-GATE: every in-band count in this record is REFERENCE-ONLY. The "
        "star opens no gate on webtext-v3; G-star-hit belongs to the v2.1 "
        "campaign and was adjudicated FAILED there (S1, stamped 2026-07-29).",
        "IN-SAMPLE BY CONSTRUCTION: the star's constants are solved on the SAME "
        "240 sealed slots the star is then scored on, because §8 fit hub legs "
        "only for the five RACE hubs and race-hub legs are not scoreable pairs — "
        "there is no core-model hub column on v3 to derive from. The composed "
        "column was filed, sha'd and desk-sealed BEFORE the first direct-pair "
        "fit. The contrast is maximally generous to the star; `held_out_beside` "
        "is the like-for-like beside.",
        "VINTAGE: no v2.1 constant enters this solve. Both systems are "
        "gauge-fixed by their own pair graphs, so no anchor is used at all; the "
        "sole v2.1 contact is the regression proof, which reproduces the banked "
        "record and feeds nothing forward.",
        "ARMS NEVER MIXED: two systems (native::proc_k256 over the 13 instruct "
        "models, raw::proc_k256 over the 42 base-endpoint pairs). A model in "
        "both carries two constants; each is quoted with its system.",
        "DIRECTION: both sealed directions of every unordered pair enter the "
        "solve as separate observations. The star form is symmetric, so the "
        "fwd/rev gap lands in the residual rather than being averaged out of "
        "sight; see `miss_structure.directional_ceiling`.",
    ]

    return {
        "record": "webtext-v3-star-beside/v1",
        "STATUS": GATELESS_STATUS,
        "adjudication": ADJUDICATION,
        "what": ("â(A→B) ≈ c_A · c_B re-derived on webtext-v3 under the parent "
                 "contract's derivation rules, quoted beside the composed "
                 "column of record on the identical sealed slots."),
        "parent_contract": {
            "path": "docs/planning/PREREG-transport-campaign-2026-07-26.md",
            "read_at": "tag freeze/transport-campaign",
            "sha256": PARENT_PREREG_SHA,
            "rules_applied": [
                "§2: c_M is derived in a NAMED (arm × family) system; one solve "
                "= one system; families and arms are never mixed (rake 40).",
                "§2/§3: the star is a POSITIVE-FACTOR model — â ≤ 0 cannot enter "
                "the log system and is excluded BY NAME, never abs()'d.",
                "§3: bands ±.05 absolute at the family of record; near-zero "
                "carve-out |predicted| < .08 scored MAGNITUDE-ONLY.",
                "§3: a system must pass its own held-out check before hosting a "
                "constant a prediction hangs on — here, `held_out_beside`.",
                "§3: the gauge. A pure hub star is bipartite and needs exactly "
                "one anchor; once the graph closes, the anchor becomes a "
                "consistency check rather than a constraint. The v3 graph "
                "closes, so the graph IS the anchor.",
            ],
        },
        "v3_contract": {
            "path": "docs/planning/PREREG-webtext-v3-2026-08-03.md",
            "sha256": scored.get("prereg", ""),
            "clause": "§8 'the star beside (S1 continuation)': the scalar star "
                      "is re-derived on webtext-v3 under the parent's derivation "
                      "rules and quoted ONLY as the composed-vs-star descriptive "
                      "contrast on the primary basis — labeled diagnostic, never "
                      "a headline, never gated.",
        },
        "s1_record_continued": {
            "stamp": "S1 STAMPED (Luxia, 2026-07-29): 'scalar factorization "
                     "refuted at scale; hub-mediated transport demonstrated at "
                     "map order'.",
            "s4_descriptive": "S4 STAMPED: 'misses cluster as systematic "
                              "under-prediction; the frame, not the "
                              "factorization, carries transport.'",
            "this_lane": "CONTINUES that record on the clean basis. It cannot "
                         "reopen it and does not try.",
        },
        "inputs": {
            "scored_record": str(SCORED_REL),
            "scored_record_sha256": sha256_of_path(data_root / SCORED_REL),
            "prediction_artifact": str(PREDICTIONS_REL),
            "prediction_artifact_sha256": scored.get("artifact_sha256", ""),
            "stamp_verification": scored.get("stamp_verification", {}),
            "corpus_manifest_sha256": scored.get("corpus_manifest_sha256", ""),
            "splits_sha256": scored.get("splits_sha256", ""),
            "family_of_record": scored.get("family_of_record", ""),
            "frozen_count_N": scored.get("frozen_count_N", 0),
            "superseded_files_never_opened": True,
        },
        "bands": {
            "primary_half_width": PRIMARY_HALF_WIDTH,
            "co_primary_half_width": CO_PRIMARY_HALF_WIDTH,
            "near_zero_carve_out_abs_predicted": NEAR_ZERO_ABS,
            "near_zero_star_slots": sum(1 for r in rows
                                        if abs(r.star_predicted) < NEAR_ZERO_ABS),
            "floor_clearing_observed_slots": sum(1 for r in rows
                                                 if abs(r.observed) >= NEAR_ZERO_ABS),
            "n_non_positive_observed": sum(1 for r in rows if r.observed <= 0.0),
            "n_star_verdict_fragile_at_rounding": sum(
                1 for r in rows if r.star_verdict_fragile),
            "note": "bands are the sealed ceremony's, inherited verbatim. "
                    "NON-GATE for the star. The near-zero carve-out never fires "
                    "here when `near_zero_star_slots` is 0.",
        },
        "regression_proof_v21": proof,
        "systems": summaries,
        "solver_solutions": sol_dumps,
        "coefficients": coeffs,
        "head_to_head": h2h.model_dump(),
        "miss_structure": miss_structure(rows, scored),
        "held_out_beside": held_out_beside(groups, rows),
        "slots": [r.model_dump() for r in rows],
        "flags": gate_free,
    }


def emit(data_root: Path, artifact: dict, out_dir: Path) -> tuple[Path, Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / ARTIFACT_NAME
    payload = json.dumps(artifact, indent=1, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    art_path.write_text(payload, encoding="utf-8")
    digest = sha256_of_path(art_path)

    sidecar = {
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": digest,
        "attestation": GATELESS_STATUS,
        "builder": "metabasis/scripts/star_beside_webtext_v3.py",
        "determinism": ("the artifact carries no timestamp, no absolute path and "
                        "no environment-varying value; every such fact is here. "
                        "`--selftest` proves a build-twice byte-identical "
                        "artifact."),
        "emitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        "solver_generated_date": datetime.now(timezone.utc).date().isoformat(),
        "thread_config": thread_config_stamp(),
    }
    side_path = out_dir / SIDECAR_NAME
    side_path.write_text(json.dumps(sidecar, indent=1, sort_keys=True,
                                    ensure_ascii=False) + "\n", encoding="utf-8")
    return art_path, side_path, digest


# ---------------------------------------------------------------- selftest
def selftest(data_root: Optional[Path] = None) -> int:            # noqa: C901
    """Checklist. Returns a failure count; never sys.exit()s (rake M45)."""
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    print("== selftest 1: synthetic exact recovery on a CLOSED (v3-shaped) graph ==")
    truth = {"a": 0.8, "b": 0.6, "c": 0.5, "d": 0.4}
    obs = []
    for i, m in enumerate(sorted(truth)):
        for n in sorted(truth):
            if m < n:
                obs.append(ExchangeRateObservation(
                    source=m, target=n, a_hat=truth[m] * truth[n],
                    arm="native", family="proc_k256", pair_id=f"{m}->{n}"))
                obs.append(ExchangeRateObservation(
                    source=n, target=m, a_hat=truth[m] * truth[n],
                    arm="native", family="proc_k256", pair_id=f"{n}->{m}"))
    sols = solve_v3_systems({"native::proc_k256": obs})
    sol = sols["native::proc_k256"]
    worst = max(abs(sol.coefficients[m] - truth[m]) for m in truth)
    check(sol.nullspace_dim == 0 and sol.gauge_fixed,
          f"a complete graph closes the gauge with NO anchor (nullspace "
          f"{sol.nullspace_dim}, gauge_fixed={sol.gauge_fixed})")
    check(worst < 5e-5, f"all constants recovered anchorlessly, worst |Δc| = {worst:.2e}")
    check(not sol.anchors, "no anchor was passed — the vintage rule holds by "
                           "construction, not by convention")

    print("== selftest 2: a gauge-OPEN system is refused, not min-normed ==")
    star_only = [ExchangeRateObservation(
        source="hub", target=m, a_hat=0.7 * c, arm="native",
        family="proc_k256", pair_id=f"hub->{m}")
        for m, c in truth.items()]
    try:
        solve_v3_systems({"native::proc_k256": star_only})
        check(False, "a bipartite hub star must be REFUSED without an anchor")
    except StarBesideError as exc:
        check("vintage" in str(exc).lower(),
              f"bipartite hub star refused, naming the vintage rule: {str(exc)[:70]}…")

    print("== selftest 3: arms are never mixed (the lineage refuses) ==")
    mixed = {"x": [ExchangeRateObservation(source="a", target="b", a_hat=0.4,
                                           arm="native", family="proc_k256",
                                           pair_id="a->b"),
                   ExchangeRateObservation(source="a", target="b", a_hat=0.3,
                                           arm="raw", family="proc_k256",
                                           pair_id="a->b-raw")]}
    try:
        solve_v3_systems(mixed)
        check(False, "mixing arms inside one solve must raise")
    except ValueError as exc:
        check(True, f"mixing arms raises: {str(exc)[:64]}…")

    print("== selftest 4: the symmetric star sees fwd/rev as residual, not noise ==")
    asym = [ExchangeRateObservation(source="a", target="b", a_hat=0.50,
                                    arm="native", family="proc_k256", pair_id="a->b"),
            ExchangeRateObservation(source="b", target="a", a_hat=0.30,
                                    arm="native", family="proc_k256", pair_id="b->a"),
            ExchangeRateObservation(source="a", target="c", a_hat=0.40,
                                    arm="native", family="proc_k256", pair_id="a->c"),
            ExchangeRateObservation(source="c", target="a", a_hat=0.40,
                                    arm="native", family="proc_k256", pair_id="c->a"),
            ExchangeRateObservation(source="b", target="c", a_hat=0.24,
                                    arm="native", family="proc_k256", pair_id="b->c"),
            ExchangeRateObservation(source="c", target="b", a_hat=0.24,
                                    arm="native", family="proc_k256", pair_id="c->b")]
    asol = solve_v3_systems({"native::proc_k256": asym})["native::proc_k256"]
    ab = {r.pair_id: r.predicted for r in asol.residuals}
    check(abs(ab["a->b"] - ab["b->a"]) < 1e-9,
          "the star predicts ONE value for both directions (symmetric by form)")
    geo = math.sqrt(0.50 * 0.30)
    check(abs(ab["a->b"] - geo) < 5e-3,
          f"and it lands at the geometric mean of the two ({ab['a->b']:.4f} vs "
          f"{geo:.4f}) — the fwd/rev gap becomes residual")

    print("== selftest 5: both-directions fit ≡ geometric-mean fit (weighting) ==")
    geo_obs = [ExchangeRateObservation(
        source="a", target="b", a_hat=math.sqrt(0.50 * 0.30), arm="native",
        family="proc_k256", pair_id="a~b"),
        ExchangeRateObservation(source="a", target="c", a_hat=0.40, arm="native",
                                family="proc_k256", pair_id="a~c"),
        ExchangeRateObservation(source="b", target="c", a_hat=0.24, arm="native",
                                family="proc_k256", pair_id="b~c")]
    gsol = solve_v3_systems({"native::proc_k256": geo_obs})["native::proc_k256"]
    worst_g = max(abs(gsol.coefficients[m] - asol.coefficients[m])
                  for m in gsol.coefficients)
    check(worst_g <= 1e-4,
          f"fitting 2N directed rows == fitting N geometric-mean rows "
          f"(worst |Δc| {worst_g:.1e}) — every pair carries exactly 2 slots, so "
          f"the doubled weight is uniform and cancels")

    print("== selftest 6: the miss-side convention is not reversible ==")
    rows = [SlotStar(
        ordinal=1, prediction_id="p", pair_id="p", source_model="a",
        source_site=1, target_model="b", target_site=2, arm="native",
        family="proc_k256", pair_class="instruct↔instruct",
        system="native::proc_k256", extension_line=False, observed=0.50,
        observed_direction="fwd", star_predicted=0.30,
        star_error_signed=0.20, star_error=0.20,
        star_in_band_primary=False, star_in_band_co_primary=False,
        star_miss_side="high", star_verdict_fragile=False,
        composed_hub_of_record="8b", composed_predicted=0.49,
        composed_error_signed=0.01, composed_error=0.01,
        composed_in_band_primary=True,
        composed_in_band_co_primary=True, concordance="composed-only")]
    ms = miss_structure(rows, {"slots": []})
    check(ms["n_miss_high"] == 1 and ms["n_miss_low"] == 0,
          "observed ABOVE the band counts as miss-HIGH = the star UNDER-predicts")
    h = head_to_head(rows)
    check(h.composed_only == 1 and h.star_only == 0 and h.sign_flip_exponent == 1,
          "head-to-head: composed-only 1, star-only 0, sign-flip exponent 1")

    print("== selftest 7: a SUPERSEDED-* input is refused by name ==")
    try:
        load_scored_record(Path("/nonexistent"),
                           scored_rel=Path("staging/SUPERSEDED-x.json"))
        check(False, "a SUPERSEDED-* name must be refused")
    except StarBesideError as exc:
        check("SUPERSEDED" in str(exc), f"refused: {str(exc)[:64]}…")

    print("== selftest 8: _strip_site recovers the model key from a banked pair_id ==")
    cases = {"8bL16": "8b", "qwen2.5-3b-instructL26": "qwen2.5-3b-instruct",
             "dsv2-liteL22": "dsv2-lite", "llama-3.1-405b-instructL99":
             "llama-3.1-405b-instruct", "3b": "3b"}
    bad = {k: _strip_site(k) for k, v in cases.items() if _strip_site(k) != v}
    check(not bad, f"site suffixes stripped correctly ({len(cases)} cases){bad or ''}")

    if data_root is not None:
        print("== selftest 9: build-twice byte-identical on the real record ==")
        try:
            scored = load_scored_record(data_root)
            groups = observations_from_scored(scored)
            sols = solve_v3_systems(groups)
            rowsr = score_slots(scored, sols)
            stub = {"STATUS": "selftest stub", "ALL_PASS": True, "checks": []}
            a1 = build_artifact(data_root, scored, groups, sols, rowsr, stub)
            sols2 = solve_v3_systems(observations_from_scored(scored))
            rows2 = score_slots(scored, sols2)
            a2 = build_artifact(data_root, scored, observations_from_scored(scored),
                                sols2, rows2, stub)
            p1 = json.dumps(a1, indent=1, sort_keys=True, ensure_ascii=False,
                            allow_nan=False)
            p2 = json.dumps(a2, indent=1, sort_keys=True, ensure_ascii=False,
                            allow_nan=False)
            check(p1 == p2, "two independent builds are byte-identical")
            check("(sidecar)" == a1["solver_solutions"][
                sorted(a1["solver_solutions"])[0]]["generated"],
                  "the date and the thread config live in the sidecar, not the "
                  "artifact")
            blob = p1
            check("/home/" not in blob and "/models/" not in blob
                  and "/mnt/" not in blob,
                  "no absolute path anywhere in the artifact bytes")
        except StarBesideError as exc:
            check(False, f"real-record build failed: {exc}")
    else:
        print("== selftest 9: SKIPPED (no --data-root; run it WITH one too, "
              "rake M44) ==")

    print(f"\nselftest: {len(failures)} failure(s) "
          f"[python {platform.python_version()} · numpy {np.__version__} · "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')} · "
          f"data_root={'yes' if data_root else 'no'}]")
    return len(failures)


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--regression-proof", action="store_true",
                    help="run the v2.1 parity-gate regression proof and stop")
    ap.add_argument("--run", action="store_true",
                    help="proof + solve + emit the artifact and sidecar")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="the repo/data root that holds staging/ and outputs/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override the artifact directory (default "
                         "<data-root>/staging/webtext-v3-reads/star-beside)")
    args = ap.parse_args(argv)

    root = args.data_root.resolve() if args.data_root else None
    if root is not None and not root.is_dir():
        logger.error("data root is not a directory: %s", root)
        return 2

    try:
        if args.selftest:
            return 1 if selftest(root) else 0

        if root is None:
            logger.error("--data-root is required for --regression-proof/--run")
            return 2

        proof = regression_proof_v21(root)
        for c in proof["checks"]:
            logger.info("REGRESSION %s %s — %s", c["id"],
                        "PASS" if c["PASS"] else "FAIL", c["what"][:72])
        if not proof["ALL_PASS"]:
            logger.error("REGRESSION PROOF FAILED — no v3 number may be quoted")
            return 1
        logger.info("REGRESSION PROOF: ALL PASS (the parity-gate bar is met)")
        if args.regression_proof and not args.run:
            return 0

        scored = load_scored_record(root)
        groups = observations_from_scored(scored)
        solutions = solve_v3_systems(groups)
        rows = score_slots(scored, solutions)
        artifact = build_artifact(root, scored, groups, solutions, rows, proof)

        out_dir = args.out_dir if args.out_dir else root / OUT_DIR_REL
        art_path, side_path, digest = emit(root, artifact, out_dir)
        logger.info("wrote %s (sha256 %s)", art_path.name, digest)
        logger.info("wrote %s", side_path.name)

        h = artifact["head_to_head"]
        logger.info("STAR %d/%d (%.1f%%) vs COMPOSED %d/%d (%.1f%%) at ±.05 "
                    "[NON-GATE] — Δ %.1f pp",
                    h["star_in_band_primary"], h["n_slots"],
                    100 * h["star_fraction_primary"],
                    h["composed_in_band_primary"], h["n_slots"],
                    100 * h["composed_fraction_primary"],
                    h["delta_percentage_points"])
        for s in artifact["systems"]:
            logger.info("SYSTEM %s: %d models, %d slots, dof %d, rms |resid| %s, "
                        "max %s", s["system"], s["n_models"], s["n_slots_fit"],
                        s["over_identification_dof"], s["rms_residual_a_hat"],
                        s["max_abs_residual_a_hat"])
        ms = artifact["miss_structure"]
        logger.info("MISSES %d — high %d / low %d; range-compression r = %s",
                    ms["n_star_miss"], ms["n_miss_high"], ms["n_miss_low"],
                    ms["magnitude_trend"]["pearson_r_signed_error_vs_observed"])
        ho = artifact["held_out_beside"]
        logger.info("HELD-OUT BESIDE %d/%d in band", ho["n_in_band_primary"],
                    ho["n_scored"])
        return 0
    except StarBesideError as exc:
        logger.error("HALT: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
