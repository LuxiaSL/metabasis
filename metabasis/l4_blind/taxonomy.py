"""The cell-TYPE taxonomy and the detection-PASS population it spans.

PROPOSED, not ruled. The brief (2026-08-07 §1) says: "propose the type
taxonomy from the frozen texts (axis × cell-kind at minimum), FREEZE the
draw behind a named seed recipe, and put the taxonomy + draw rule in your
report for desk adjudication BEFORE Luxia's session." This module is that
proposal made executable; `TAXONOMY_RULE` is the sentence the desk rules on.

Where every input comes from (all desk-side, all frozen):

  * the PASS population — `staging/reading-l2-arm8/DESK-SCORE-L2-2026-08-06.json`,
    the desk's O-3 scoring (held-out AUC ≥ .65 AND strictly above all three
    band-member AUCs). Only rows with verdict == "PASS" enter.
  * the coordinates (arm / axis / site / node_key / wave / source tree) —
    `staging/reading-l2-arm8/L2-PROBE-INGREDIENTS-2026-08-06.json`, whose
    `source_tree` field is the of-record results root per (column, side)
    and already carries FLAG-F (phi-4.refusal lives in the r2b attempt dir).
  * the cell-id conventions — read off the banked cell directories and
    asserted, never assumed.

Cell-id conventions of record (asserted against the banked trees):

  | wave / side                 | dose cell id                          |
  |-----------------------------|---------------------------------------|
  | EGV, transported            | gentropy_gradient_L{site}_a{dose}     |
  | class-pilot, transported    | gcaa_{axis}_L{site}_a{dose}           |
  | class-pilot, native         | caa_{axis}_L{site}_L{site}_a{dose}    |
  | baseline (both, per column) | baseline_L{site}_a+0.00               |
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from metabasis.l4_blind.models import Axis, Dose, PassCell, Side

# ── the proposal, in one sentence ────────────────────────────────────────

TAXONOMY_RULE: Final[str] = (
    "PROPOSED cell type = (axis × side), where axis ∈ {egv, formality, "
    "language, refusal, sentiment} and side ∈ {native, transported}. The "
    "detection-PASS population realises 9 of the 10 combinations — EGV has "
    "no native type, because EGV native rows are the calibration half "
    "(FLAG-D, 'native_calibration_half_BESIDE') and are never signal. Dose "
    "SIGN (+0.30 / -0.30) is NOT a type; it is BALANCED WITHIN every type "
    "(8 pairs each way) and recorded per pair, so the desk can slice by "
    "sign post hoc without doubling the session. Model/column is NOT a "
    "type; it is a stratification key inside the type, spread as evenly as "
    "the seeded round-robin allows and recorded per pair."
)

TAXONOMY_ALTERNATIVES: Final[list[str]] = [
    "COARSER — axis only (5 types, 80 pairs, ~40 min): rejected in the "
    "proposal because it pools native with transported, and transported is "
    "the claim while native is the equivalence-BESIDE; a per-type "
    "agreement number that mixes them cannot support the O-5 bar for "
    "either.",
    "FINER — axis × side × dose-sign (18 types, 288 pairs, ~2.4 h): "
    "scientifically the cleanest (a judge good at +0.30 and bad at -0.30 "
    "would be caught), but it is a 2.4-hour blind sitting and O-5 fixes 16 "
    "pairs per type, so the cost is real. The proposal keeps sign balanced "
    "8/8 within each type as the affordable half of this.",
    "SPLIT EGV BY ARM — the EGV type mixes arm='native' instruct columns "
    "with arm='raw' base columns (llama-3.1-8b-base, pythia-6.9b, "
    "qwen2.5-7b-base), which generate very differently. The proposal keeps "
    "them one type and records `arm` per pair as a stratification key; the "
    "desk may instead rule EGV into two types (egv/transported/native-arm "
    "and egv/transported/raw-arm), which adds 16 pairs.",
]

#: The 2AFC trait each axis is judged on. PROPOSED — no desk artifact pins
#: a trait string per axis today; the frozen judge template
#: (JUDGE-TEMPLATE-2afc-2026-08-04.txt, "which text is more {trait}?")
#: leaves {trait} open. The L4 human MUST see the same trait wording the L3
#: judge sees, or the agreement number compares two different questions.
#:
#: Leak discipline: no trait string may contain its own axis key as a
#: substring (that is why sentiment is "positive in tone", not "positive in
#: sentiment"). The leak selftest asserts this.
AXIS_TRAIT: Final[dict[Axis, str]] = {
    "egv": "unpredictable in its wording",
    "formality": "formal",
    "language": "French",
    "refusal": "refusing to answer",
    "sentiment": "positive in tone",
}

#: Pairs per cell type. O-5, ruled by Luxia 2026-08-04: "16 blind pairs per
#: cell type". Must stay even so the sign balance and the order
#: counterbalance both come out exact.
PAIRS_PER_TYPE: Final[int] = 16

#: Generations per banked cell. Asserted against every cell that is drawn.
GENERATIONS_PER_CELL: Final[int] = 80

_JUDGED_SIDES: Final[frozenset[str]] = frozenset({"native", "transported"})
_DOSES: Final[tuple[Dose, ...]] = ("+0.30", "-0.30")


class TaxonomyInputs(BaseModel):
    """The two frozen desk artifacts the taxonomy reads, with their shas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    desk_score_path: str
    desk_score_sha256: str
    ingredients_path: str
    ingredients_sha256: str
    repo_root: str


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _axis_of(raw_axis: str | None) -> Axis:
    """The L2 ingredients carry `axis = null` for the whole-column EGV wave.

    That null is not missing data — it is the entropy-gradient axis, which
    predates the class split and therefore has no axis suffix on its column
    name. Naming it "egv" here is the ONE place the two vocabularies meet.
    """
    if raw_axis is None:
        return "egv"
    if raw_axis not in ("formality", "language", "refusal", "sentiment"):
        raise ValueError(f"unknown axis in the frozen ingredients: {raw_axis!r}")
    return raw_axis  # type: ignore[return-value]


def dose_cell_id(axis: Axis, side: Side, site: int, dose: Dose) -> str:
    """The banked cell id for a signal cell. Asserted on disk before use."""
    if axis == "egv":
        if side != "transported":
            raise ValueError(
                "EGV native rows are the calibration half (FLAG-D) and are "
                "not part of the judged population"
            )
        return f"gentropy_gradient_L{site}_a{dose}"
    if side == "transported":
        return f"gcaa_{axis}_L{site}_a{dose}"
    return f"caa_{axis}_L{site}_L{site}_a{dose}"


def baseline_cell_id(site: int) -> str:
    """The α=0 cell the pair's second text comes from — one per column."""
    return f"baseline_L{site}_a+0.00"


def type_key_of(axis: Axis, side: Side) -> str:
    return f"{axis}/{side}"


def load_pass_population(inputs: TaxonomyInputs) -> list[PassCell]:
    """Every detection-PASS cell, resolved to banked coordinates.

    Raises on any cell whose banked directory, baseline, or generation count
    does not check out — the draw must never silently skip a PASS cell.
    """
    root = Path(inputs.repo_root)
    score = json.loads(Path(inputs.desk_score_path).read_text())
    ingredients = json.loads(Path(inputs.ingredients_path).read_text())

    if score.get("source_sha256") is None:
        raise ValueError("DESK-SCORE-L2 carries no source_sha256 — refusing to draw")

    coords: dict[tuple[str, str], dict] = {}
    for row in ingredients["rows"]:
        coords[(row["column"], row["side"])] = row

    cells: list[PassCell] = []
    problems: list[str] = []

    for row in score["verdicts"]:
        if row["verdict"] != "PASS":
            continue
        side = row["side"]
        if side not in _JUDGED_SIDES:
            # naive_BESIDE / native_calibration_half_BESIDE never PASS, but
            # the guard is cheap and the failure mode (a BESIDE cell in the
            # gold) would be invisible downstream.
            problems.append(f"PASS row on a BESIDE side: {row['column']}/{side}")
            continue
        dose = row["dose"]
        if dose not in _DOSES:
            problems.append(f"unexpected dose {dose!r} on {row['column']}")
            continue

        meta = coords.get((row["column"], side))
        if meta is None:
            problems.append(f"no L2-ingredients coordinates for {row['column']}/{side}")
            continue

        axis = _axis_of(meta["axis"])
        site = int(meta["site"])
        results_root = root / meta["source_tree"]
        dcell = dose_cell_id(axis, side, site, dose)  # type: ignore[arg-type]
        bcell = baseline_cell_id(site)

        for cid in (dcell, bcell):
            gen = results_root / cid / "generations.jsonl"
            if not gen.is_file():
                problems.append(f"missing banked generations: {gen}")

        cells.append(
            PassCell(
                column=row["column"],
                side=side,  # type: ignore[arg-type]
                dose=dose,  # type: ignore[arg-type]
                axis=axis,
                arm=meta["arm"],
                node_key=meta["node_key"],
                site=site,
                wave=meta["wave"],
                signal_auc=float(row["signal_auc"]),
                null_band_max=float(row["null_band_max"]),
                dose_cell_id=dcell,
                baseline_cell_id=bcell,
                results_root=meta["source_tree"],
                type_key=type_key_of(axis, side),  # type: ignore[arg-type]
            )
        )

    if problems:
        raise ValueError(
            "the PASS population did not resolve cleanly; refusing to freeze a "
            "draw over a broken population:\n  - " + "\n  - ".join(problems)
        )
    if not cells:
        raise ValueError("no PASS cells found — check the DESK-SCORE artifact")
    return cells


def group_by_type(cells: list[PassCell]) -> dict[str, list[PassCell]]:
    """Canonical grouping and canonical ORDER inside each group.

    The order is (column, dose) ascending — a total order over the PASS
    population that does not depend on dict iteration, file order, or
    anything the filesystem decides. Every seeded step downstream consumes
    this order, so the draw is reproducible on any machine.
    """
    out: dict[str, list[PassCell]] = {}
    for c in cells:
        out.setdefault(c.type_key, []).append(c)
    for k in out:
        out[k] = sorted(out[k], key=lambda c: (c.column, c.dose))
    return dict(sorted(out.items()))
