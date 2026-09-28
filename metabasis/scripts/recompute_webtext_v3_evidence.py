"""Recompute the webtext-v3 headline counts from the published evidence package.

The package in `evidence/webtext-v3/` holds the sealed prediction artifact, the
stamp that seals it, the scored record that carries one observed exchange rate
per slot, and the star-beside record. This module reads those files and nothing
else, and recomputes:

* the composed gates per hub column — in-band at ±0.05 over the 240 frozen
  slots, in-band at ±0.025 over the floor-clearing slots (|â_obs| ≥ 0.08), and
  in-band at ±0.05 over the 84-slot extension subset;
* the star beside — the scalar product model â(A→B) = c_A · c_B, fit in-sample
  on the same 240 slots and scored at ±0.05, its leave-one-unordered-pair-out
  held-out count, and the slot-by-slot discordance against the composed column
  of record.

WHAT IS RECOMPUTED AND WHAT IS READ. The filed predictions and their bands come
from the sealed artifact, after its stamp verifies. The observed value of each
slot, `observed`, is read from the scored record; everything the scored record
concluded from it (verdicts, errors, gate counts) is discarded and rebuilt here.
The scoring arithmetic is `read_composed_predictions.score_v3_column` and
`_column_gate`, and the star arithmetic is `star_beside_webtext_v3`'s
`solve_v3_systems`, `score_slots`, `head_to_head` and `held_out_beside` — all
imported and called, none restated. The rebuilt numbers are then compared with
the counts the scored record and the star-beside record hold, and any
disagreement is a failure.

The observed values themselves are cosines read through the direct pair-fit
maps (`read_exchange_rates.exchange_rate`). Those maps are several gigabytes and
are not in the package, so `observed` is an input here, not a recomputed value.

Run from the repository root:
  python -m metabasis.scripts.recompute_webtext_v3_evidence
  python -m metabasis.scripts.recompute_webtext_v3_evidence --selftest

Exit status: 0 when every sha, every recomputed count and every cross-check
agrees; 1 when any disagrees; 2 when an input is missing or unreadable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from metabasis.scripts.read_composed_predictions import (
    V3_STAMP_FILENAME,
    V3ColumnGate,
    V3PredictionArtifact,
    V3ScoredSlot,
    V3Slot,
    _column_gate,
    load_v3_prediction_artifact,
    score_v3_column,
)
from metabasis.scripts.star_beside_webtext_v3 import (
    head_to_head,
    held_out_beside,
    observations_from_scored,
    score_slots,
    solve_v3_systems,
)

logger = logging.getLogger("recompute_webtext_v3_evidence")

#: Where the package lives, relative to the repository root.
DEFAULT_EVIDENCE_REL = Path("evidence/webtext-v3")

PREDICTIONS_NAME = "PREDICTIONS-webtext-v3-2026-08-04.json"
PREDICTIONS_SIDECAR_NAME = "PREDICTIONS-webtext-v3-2026-08-04.sidecar.json"
SCORED_NAME = "SCORED-webtext-v3-2026-08-04-FULL.json"
STAR_BESIDE_NAME = "STAR-BESIDE-webtext-v3-2026-08-04.json"
STAR_BESIDE_SIDECAR_NAME = "STAR-BESIDE-webtext-v3-2026-08-04.sidecar.json"

#: The sha256 of every file in the package. A file that hashes to anything else
#: is not the file the counts were computed from, so it is refused before it is
#: parsed.
EXPECTED_SHA256: dict[str, str] = {
    PREDICTIONS_NAME:
        "de9eb699db92824443002300f98c5a8fceb0aff981d15c938887e8ca9e033502",
    V3_STAMP_FILENAME:
        "35ccd86ab1162fdec039204112a64f3704e3d0387d2e3ee288811cf6b960e64a",
    PREDICTIONS_SIDECAR_NAME:
        "8a2746d25b0b90381b916876b71982f410820ecbe3872e23a515b95ff927e243",
    SCORED_NAME:
        "6b7780035d332d5ae9a9f07b1727685e51de335f1a0f6bb8164f6aa8729cdc7f",
    STAR_BESIDE_NAME:
        "f036d3970347d3e2baa5e98ffdb359e5555ca1181075e575287f7e2c39d24e38",
    STAR_BESIDE_SIDECAR_NAME:
        "e9e626f96bd295778ca2849a2dba05e6165ec7088dd95a7f57711453f210f3a5",
}

#: The gate fields compared between the rebuilt gates and the scored record's.
GATE_FIELDS: tuple[str, ...] = (
    "n_frozen", "n_scored", "n_unscored_no_fit", "n_in_band_primary",
    "n_floor_clearing", "n_near_zero_predicted", "n_in_band_co_primary",
    "n_extension", "n_extension_scored", "n_extension_in_band_primary",
    "n_out_of_band_high", "n_out_of_band_low", "n_magnitude_only_scored",
    "g_comp_v3_meets_threshold", "g_comp_v3_tight_meets_threshold",
    "g_extension_meets_threshold", "both_core_gates_meet_thresholds",
)


class EvidenceError(RuntimeError):
    """An input is missing, unreadable, or not the file the package names."""


class ArmCount(BaseModel):
    """One hub column's in-band counts restricted to one arm of record."""
    hub: str
    arm: str
    n_slots: int
    n_in_band_primary: int
    n_floor_clearing: int
    n_in_band_co_primary: int


class StarSummary(BaseModel):
    """The star beside, recomputed. Counts are over the 240 frozen slots."""
    n_slots: int
    star_in_band_in_sample: int
    star_in_band_held_out: int
    held_out_n_scored: int
    held_out_n_skipped: int
    composed_in_band: int
    composed_only: int
    star_only: int
    both_hit: int
    both_miss: int
    system_sizes: dict[str, int]


class Recompute(BaseModel):
    """Everything the command recomputes, plus the cross-check verdicts."""
    evidence_dir: str
    file_sha256: dict[str, str]
    stamp_sealed_utc: str
    frozen_count_N: int
    of_record_hub: str
    never_before_observed: list[str]
    gates: list[V3ColumnGate]
    arm_counts: list[ArmCount]
    star: StarSummary
    mismatches: list[str]


# ---------------------------------------------------------------- inputs
def sha256_of(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evidence_dir(explicit: Optional[Path]) -> Path:
    """The package directory: `explicit`, else `./evidence/webtext-v3`, else
    the same path under the repository that holds this module.

    Raises `EvidenceError` when none of them is a directory.
    """
    candidates = ([Path(explicit)] if explicit is not None else
                  [Path.cwd() / DEFAULT_EVIDENCE_REL,
                   Path(__file__).resolve().parents[2] / DEFAULT_EVIDENCE_REL])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise EvidenceError(
        "no evidence package found; looked in "
        + ", ".join(str(c) for c in candidates)
        + ". Run from the repository root or pass --evidence-dir")


def verify_file_shas(evidence_dir: Path,
                     expected: dict[str, str] = EXPECTED_SHA256
                     ) -> dict[str, str]:
    """Hash every package file and refuse on the first mismatch or absence.

    Returns {filename: sha256}. Raises `EvidenceError` naming the file.
    """
    got: dict[str, str] = {}
    for name, want in expected.items():
        path = evidence_dir / name
        if not path.is_file():
            raise EvidenceError(f"package file absent: {path}")
        digest = sha256_of(path)
        if digest != want:
            raise EvidenceError(
                f"{path} hashes to {digest}, not {want}. It is not the file the "
                f"published counts were computed from; nothing is recomputed")
        got[name] = digest
    return got


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise EvidenceError(f"{path} is not a JSON object")
    return doc


def observations_by_slot(artifact: V3PredictionArtifact,
                         scored: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """{prediction_id: (observed â, observed_direction)} from the scored record.

    Every frozen slot must appear exactly once with a numeric `observed`, and
    its pair, arm and family must equal the sealed slot's: an observation filed
    under a different identity is not an observation of that prediction.
    Raises `EvidenceError` on any disagreement.
    """
    if scored.get("artifact_sha256") != artifact.sha256:
        raise EvidenceError(
            f"the scored record was written against artifact sha "
            f"{scored.get('artifact_sha256')!r}, not the sealed {artifact.sha256}")
    sealed = {slot.prediction_id: slot for slot in artifact.slots}
    out: dict[str, tuple[float, str]] = {}
    for row in scored.get("slots", []):
        pid = str(row.get("prediction_id"))
        slot = sealed.get(pid)
        if slot is None:
            raise EvidenceError(f"scored slot {pid} is not in the sealed artifact")
        if pid in out:
            raise EvidenceError(f"scored slot {pid} appears twice")
        for key in ("pair_id", "arm", "family", "source_model", "target_model"):
            if row.get(key) != getattr(slot, key):
                raise EvidenceError(
                    f"{pid}: scored {key}={row.get(key)!r} but the sealed slot "
                    f"has {getattr(slot, key)!r}")
        observed = row.get("observed")
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise EvidenceError(f"{pid}: observed is {observed!r}, not a number")
        out[pid] = (float(observed), str(row.get("observed_direction") or ""))
    missing = sorted(set(sealed) - set(out))
    if missing:
        raise EvidenceError(
            f"{len(missing)} sealed slot(s) have no observation; first: {missing[0]}")
    return out


# ---------------------------------------------------------------- scoring
def rescore_slot(slot: V3Slot, observed: float, direction: str,
                 never_before_observed: Sequence[str]) -> V3ScoredSlot:
    """One sealed slot scored against one observation, every column.

    The slot's fields and its `extension_line` flag are assembled the way
    `read_composed_predictions.score_v3_slot` assembles them; the per-column
    arithmetic is `score_v3_column`, called in the artifact's column order.
    """
    scored = V3ScoredSlot(
        ordinal=slot.ordinal, prediction_id=slot.prediction_id,
        pair_id=slot.pair_id, source_model=slot.source_model,
        source_site=slot.source_site, target_model=slot.target_model,
        target_site=slot.target_site, arm=slot.arm, family=slot.family,
        pair_class=slot.pair_class, filed_status=slot.status,
        observed=observed,
        observed_direction=direction if direction in ("fwd", "rev") else None,
        extension_line=bool({slot.source_model, slot.target_model}
                            & set(never_before_observed)))
    scored.columns = [score_v3_column(slot, column, observed)
                      for column in slot.columns.values()]
    return scored


def rescore(artifact: V3PredictionArtifact,
            observations: dict[str, tuple[float, str]]
            ) -> tuple[list[V3ScoredSlot], list[V3ColumnGate]]:
    """Every frozen slot rescored, then `_column_gate` per hub column."""
    never_before = list(
        artifact.extension_sub_line.never_before_observed_core_members)
    slots = [rescore_slot(slot, *observations[slot.prediction_id], never_before)
             for slot in artifact.slots]
    gates = [_column_gate(column.hub, column.of_record, slots,
                          artifact.frozen_count_N)
             for column in artifact.hub_columns]
    return slots, gates


def arm_counts(slots: Sequence[V3ScoredSlot]) -> list[ArmCount]:
    """In-band counts per (hub column, arm), from the rescored verdicts.

    The pre-registration never pools arms in an aggregate; this split shows how
    each gate's numerator divides between the native arm (instruct↔instruct
    slots) and the raw arm (slots with a base-model endpoint).
    """
    table: dict[tuple[str, str], ArmCount] = {}
    for slot in slots:
        for col in slot.columns:
            key = (col.hub, slot.arm)
            row = table.setdefault(key, ArmCount(
                hub=col.hub, arm=slot.arm, n_slots=0, n_in_band_primary=0,
                n_floor_clearing=0, n_in_band_co_primary=0))
            row.n_slots += 1
            row.n_in_band_primary += bool(col.in_band_primary)
            if col.floor_clearing:
                row.n_floor_clearing += 1
                row.n_in_band_co_primary += bool(col.in_band_co_primary)
    return list(table.values())


def star_beside(slots: Sequence[V3ScoredSlot]) -> StarSummary:
    """The star beside, through `star_beside_webtext_v3`'s own functions.

    Those functions read a scored record as plain JSON, so the rescored slots
    are handed over in that shape (`model_dump`); every star count therefore
    rests on the composed verdicts rebuilt here, not on the scored record's.
    """
    record = {"slots": [slot.model_dump() for slot in slots]}
    groups = observations_from_scored(record)
    solutions = solve_v3_systems(groups)
    rows = score_slots(record, solutions)
    h2h = head_to_head(rows)
    held = held_out_beside(groups, rows)
    return StarSummary(
        n_slots=h2h.n_slots,
        star_in_band_in_sample=h2h.star_in_band_primary,
        star_in_band_held_out=int(held["n_in_band_primary"]),
        held_out_n_scored=int(held["n_scored"]),
        held_out_n_skipped=int(held["n_skipped"]),
        composed_in_band=h2h.composed_in_band_primary,
        composed_only=h2h.composed_only, star_only=h2h.star_only,
        both_hit=h2h.both_hit, both_miss=h2h.both_miss,
        system_sizes={key: len(obs) for key, obs in groups.items()})


# ---------------------------------------------------------------- cross-checks
def compare_gates(rebuilt: Sequence[V3ColumnGate],
                  recorded: Sequence[dict[str, Any]]) -> list[str]:
    """Field-by-field disagreements between rebuilt and recorded gates."""
    problems: list[str] = []
    by_hub = {str(g.get("hub")): g for g in recorded}
    for gate in rebuilt:
        rec = by_hub.get(gate.hub)
        if rec is None:
            problems.append(f"gate {gate.hub}: absent from the scored record")
            continue
        for field in GATE_FIELDS:
            if getattr(gate, field) != rec.get(field):
                problems.append(f"gate {gate.hub}.{field}: rebuilt "
                                f"{getattr(gate, field)!r}, recorded "
                                f"{rec.get(field)!r}")
    if len(by_hub) != len(rebuilt):
        problems.append(f"the scored record holds {len(by_hub)} gates, "
                        f"rebuilt {len(rebuilt)}")
    return problems


def compare_verdicts(rebuilt: Sequence[V3ScoredSlot],
                     scored: dict[str, Any]) -> list[str]:
    """Per-column verdict and extension-flag disagreements with the record."""
    problems: list[str] = []
    recorded = {str(s["prediction_id"]): s for s in scored.get("slots", [])}
    for slot in rebuilt:
        rec = recorded[slot.prediction_id]
        if bool(rec.get("extension_line")) != slot.extension_line:
            problems.append(f"{slot.prediction_id}: extension_line differs")
        rec_cols = {str(c["hub"]): c for c in rec.get("columns", [])}
        for col in slot.columns:
            rc = rec_cols.get(col.hub, {})
            for field in ("verdict_primary", "verdict_co_primary",
                          "floor_clearing", "predicted"):
                if getattr(col, field) != rc.get(field):
                    problems.append(f"{slot.prediction_id} [{col.hub}] {field}: "
                                    f"rebuilt {getattr(col, field)!r}, "
                                    f"recorded {rc.get(field)!r}")
    return problems


def compare_star(star: StarSummary, record: dict[str, Any]) -> list[str]:
    """Disagreements with the star-beside record's head-to-head and held-out."""
    h2h = record.get("head_to_head", {})
    held = record.get("held_out_beside", {})
    pairs = (
        ("star_in_band_in_sample", h2h.get("star_in_band_primary")),
        ("composed_in_band", h2h.get("composed_in_band_primary")),
        ("composed_only", h2h.get("composed_only")),
        ("star_only", h2h.get("star_only")),
        ("both_hit", h2h.get("both_hit")),
        ("both_miss", h2h.get("both_miss")),
        ("n_slots", h2h.get("n_slots")),
        ("star_in_band_held_out", held.get("n_in_band_primary")),
        ("held_out_n_scored", held.get("n_scored")),
        ("held_out_n_skipped", held.get("n_skipped")),
    )
    return [f"star.{name}: rebuilt {getattr(star, name)!r}, recorded {want!r}"
            for name, want in pairs if getattr(star, name) != want]


# ---------------------------------------------------------------- driver
def recompute(evidence_dir: Path) -> Recompute:
    """Verify, rescore, rebuild the star, and cross-check. See the module doc."""
    shas = verify_file_shas(evidence_dir)
    artifact, verification = load_v3_prediction_artifact(
        evidence_dir / PREDICTIONS_NAME)
    stamp = _read_json(evidence_dir / V3_STAMP_FILENAME)
    mismatches: list[str] = []
    if stamp.get("sidecar_sha256") != shas[PREDICTIONS_SIDECAR_NAME]:
        mismatches.append(
            f"the stamp names sidecar sha {stamp.get('sidecar_sha256')!r}, the "
            f"sidecar hashes to {shas[PREDICTIONS_SIDECAR_NAME]}")

    scored = _read_json(evidence_dir / SCORED_NAME)
    star_record = _read_json(evidence_dir / STAR_BESIDE_NAME)
    if star_record.get("inputs", {}).get("scored_record_sha256") != shas[SCORED_NAME]:
        mismatches.append("the star-beside record names a different scored-record sha")

    observations = observations_by_slot(artifact, scored)
    slots, gates = rescore(artifact, observations)
    mismatches += compare_verdicts(slots, scored)
    mismatches += compare_gates(gates, scored.get("gates", []))
    star = star_beside(slots)
    mismatches += compare_star(star, star_record)

    return Recompute(
        evidence_dir=str(evidence_dir), file_sha256=shas,
        stamp_sealed_utc=verification.stamp_sealed_utc,
        frozen_count_N=artifact.frozen_count_N,
        of_record_hub=artifact.of_record_hub,
        never_before_observed=list(
            artifact.extension_sub_line.never_before_observed_core_members),
        gates=gates, arm_counts=arm_counts(slots), star=star,
        mismatches=mismatches)


def _pct(num: int, den: int) -> str:
    """`num/den = p%`, p rounded half-up to one decimal in exact arithmetic.

    Float formatting would print 231/240 (exactly 96.25 %) as 96.2 %, because
    the binary value of 96.25 computed through division sits just below it.
    """
    if not den:
        return f"{num}/0"
    pct = (Decimal(100 * num) / Decimal(den)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{num}/{den} = {pct}%"


def report(result: Recompute) -> str:
    """The human-readable summary the command prints."""
    lines = [
        "webtext-v3 evidence recompute",
        f"  package: {DEFAULT_EVIDENCE_REL}  (python {platform.python_version()}, "
        f"numpy {np.__version__})",
        "  file sha256 verified:",
    ]
    lines += [f"    {sha}  {name}" for name, sha in result.file_sha256.items()]
    lines += [
        f"  stamp verified: sealed_utc {result.stamp_sealed_utc}, N = "
        f"{result.frozen_count_N} slots, of-record hub {result.of_record_hub}",
        "",
        "COMPOSED GATES (rescored from the filed predictions and the observations)",
        f"  {'hub':<22} {'±0.05 of 240':<18} {'±0.025 of floor-clearing':<26} "
        f"{'near-zero predicted':<20} {'84-slot subset ±0.05'}",
    ]
    for g in result.gates:
        tag = " (of record)" if g.of_record else ""
        lines.append(
            f"  {g.hub + tag:<22} {_pct(g.n_in_band_primary, g.n_frozen):<18} "
            f"{_pct(g.n_in_band_co_primary, g.n_floor_clearing):<26} "
            f"{g.n_near_zero_predicted:<20} "
            f"{_pct(g.n_extension_in_band_primary, g.n_extension)}")
    lines += [
        f"  floor-clearing (|â_obs| ≥ 0.08) is a property of the observation: "
        f"{result.gates[0].n_floor_clearing}/{result.frozen_count_N} slots.",
        f"  the 84-slot subset = slots with either endpoint in "
        f"{', '.join(result.never_before_observed)} "
        f"(16·15 − 13·12 = 84; a subset of the 240, not additional predictions).",
        "",
        "ARM SPLIT (the same verdicts, divided by arm of record)",
    ]
    for row in result.arm_counts:
        lines.append(
            f"  {row.hub:<22} {row.arm:<7} ±0.05 "
            f"{_pct(row.n_in_band_primary, row.n_slots):<18} ±0.025 "
            f"{_pct(row.n_in_band_co_primary, row.n_floor_clearing)}")
    s = result.star
    lines += [
        "",
        "STAR BESIDE (â = c_A·c_B, one coefficient per model per arm system)",
        f"  systems: " + ", ".join(f"{k} {v} slots"
                                   for k, v in s.system_sizes.items()),
        f"  star in-sample   ±0.05: {s.star_in_band_in_sample}/{s.n_slots}",
        f"  star held out    ±0.05: {s.star_in_band_held_out}/{s.held_out_n_scored} "
        f"(leave one unordered pair out; {s.held_out_n_skipped} skipped)",
        f"  composed (of record) ±0.05: {s.composed_in_band}/{s.n_slots}",
        f"  discordance: composed-only {s.composed_only} / star-only "
        f"{s.star_only} / both-miss {s.both_miss} / both-hit {s.both_hit}",
        "",
    ]
    if result.mismatches:
        lines.append(f"CROSS-CHECK: {len(result.mismatches)} DISAGREEMENT(S)")
        lines += [f"  {m}" for m in result.mismatches[:50]]
    else:
        lines.append("CROSS-CHECK: every rebuilt verdict, gate count and star "
                     "count equals the scored record and the star-beside record.")
    return "\n".join(lines)


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    """Synthetic checks of this module's own logic, then the real package if
    present. Returns the failure count."""
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    import tempfile

    print("== 1: a file whose sha differs is refused before parsing ==")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.json"
        path.write_text("{}")
        try:
            verify_file_shas(Path(td), {"x.json": "0" * 64})
            check(False, "a wrong sha must raise")
        except EvidenceError as exc:
            check("hashes to" in str(exc), "wrong sha raises EvidenceError")
        try:
            verify_file_shas(Path(td), {"absent.json": "0" * 64})
            check(False, "an absent file must raise")
        except EvidenceError as exc:
            check("absent" in str(exc), "absent file raises EvidenceError")
        check(verify_file_shas(Path(td), {"x.json": sha256_of(path)})
              == {"x.json": sha256_of(path)}, "a matching sha passes")

    print("== 2: rescore_slot scores every column and flags the extension ==")
    from metabasis.scripts.read_composed_predictions import V3ColumnPrediction

    def col(hub: str, filed: float) -> V3ColumnPrediction:
        return V3ColumnPrediction(
            hub=hub, hub_site=1, of_record=hub == "h1", status="FILED",
            a_comp=filed, filed_a_comp=filed,
            band_primary=[round(filed - 0.05, 4), round(filed + 0.05, 4)],
            band_co_primary=[round(filed - 0.025, 4), round(filed + 0.025, 4)],
            magnitude_only=abs(filed) < 0.08)
    slot = V3Slot(ordinal=1, prediction_id="p1", pair_id="aL1->bL2",
                  source_model="a", source_site=1, target_model="b",
                  target_site=2, arm="native", arm_rule="", family="proc_k256",
                  pair_class="instruct↔instruct", status="FILED",
                  columns={"h1": col("h1", 0.3000), "h2": col("h2", 0.2600)})
    scored = rescore_slot(slot, 0.3200, "fwd", ["b"])
    by_hub = {c.hub: c for c in scored.columns}
    check(scored.extension_line, "an endpoint in never_before_observed sets extension_line")
    check(by_hub["h1"].in_band_primary and by_hub["h1"].in_band_co_primary,
          "h1: |0.32 − 0.30| = 0.02 is in band at ±0.05 and ±0.025")
    check(not by_hub["h2"].in_band_primary
          and by_hub["h2"].verdict_primary == "out-of-band-high",
          "h2: 0.32 − 0.26 = 0.06 is out of band high at ±0.05")
    check(by_hub["h2"].floor_clearing is True,
          "floor_clearing reads the observation: |0.32| ≥ 0.08")
    check(not rescore_slot(slot, 0.32, "fwd", ["z"]).extension_line,
          "no endpoint in never_before_observed leaves extension_line False")

    print("== 3: arm_counts divides verdicts by arm ==")
    raw_slot = slot.model_copy(update={"arm": "raw", "prediction_id": "p2"})
    rows = arm_counts([scored, rescore_slot(raw_slot, 0.30, "rev", [])])
    got = {(r.hub, r.arm): r.n_in_band_primary for r in rows}
    check(got == {("h1", "native"): 1, ("h2", "native"): 0,
                  ("h1", "raw"): 1, ("h2", "raw"): 1},
          f"per-(hub, arm) primary counts {got}")

    print("== 4: compare_gates reports a disagreeing field ==")
    gate = _column_gate("h1", True, [scored], 1)
    rec = gate.model_dump()
    check(compare_gates([gate], [rec]) == [], "identical gates agree")
    rec["n_in_band_primary"] = 0
    check(any("n_in_band_primary" in p for p in compare_gates([gate], [rec])),
          "a changed count is reported")

    print("== 5: percentages round half-up in exact arithmetic ==")
    check(_pct(231, 240) == "231/240 = 96.3%", f"231/240 prints {_pct(231, 240)}")
    check(_pct(225, 240) == "225/240 = 93.8%", f"225/240 prints {_pct(225, 240)}")
    check(_pct(3, 0) == "3/0", "an empty denominator prints no rate")

    print("== 6: the real package, when present ==")
    try:
        evidence_dir = resolve_evidence_dir(None)
    except EvidenceError as exc:
        print(f"  [SKIP] {exc}")
        evidence_dir = None
    if evidence_dir is not None:
        try:
            result = recompute(evidence_dir)
            check(not result.mismatches,
                  f"recompute agrees with both records ({len(result.mismatches)} "
                  f"disagreement(s))")
            of_record = next(g for g in result.gates if g.of_record)
            check(of_record.n_in_band_primary == 240
                  and of_record.n_in_band_co_primary == 225
                  and of_record.n_extension_in_band_primary == 84,
                  "of-record column: 240/240, 225 tight, 84/84")
            check(result.star.star_in_band_in_sample == 163
                  and result.star.star_in_band_held_out == 143
                  and result.star.composed_only == 77
                  and result.star.star_only == 0,
                  "star: 163 in-sample, 143 held out, discordance 77/0")
        except EvidenceError as exc:
            check(False, f"real package: {exc}")

    print(f"\nselftest: {len(failures)} failure(s)")
    return len(failures)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--evidence-dir", type=Path, default=None,
                    help=f"the package directory (default {DEFAULT_EVIDENCE_REL} "
                         f"under the current directory, then under the repository)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full recompute (every gate field, the "
                         "arm split, the star summary) to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.selftest:
        return 1 if selftest() else 0
    try:
        result = recompute(resolve_evidence_dir(args.evidence_dir))
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(report(result))
    if args.json is not None:
        args.json.write_text(result.model_dump_json(indent=1) + "\n",
                             encoding="utf-8")
    return 1 if result.mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
