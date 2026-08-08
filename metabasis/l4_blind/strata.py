"""The measured-effect layer — deck v2's whole reason for existing.

v1 drew pairs uniformly from detection-PASS cells. That was the defect.
Detection PASS is a statement about a CELL (its 80 generations separate
from the band); it says nothing about whether an INDIVIDUAL pair shows a
visible effect. Luxia's first ten pairs proved it: four were
expected-null by construction, the graded ones were .05–.13 population
shifts, and her unsures tracked our own quantitative record exactly. An
effect-blind draw cannot tell "she could not see it" from "there was
nothing to see."

v2 measures every candidate pair before drawing it.

    delta = (s_axis[steered generation] - s_axis[baseline generation])
            * sign(dose)

i.e. the on-axis classifier score moved in the DIRECTION THE DOSE
INTENDED, for that one prompt. Both generations share a prompt_id, so
the difference is the steering and not the topic.

Source: `CORRECTED-GENERATION-ROWS.jsonl` (the corrected decode of
2026-08-06) — the only desk-side artifact carrying PER-GENERATION head
scores. The banked L1 logs carry cell-level means only, which is exactly
the granularity that produced the v1 defect.

COVERAGE IS THE BINDING CONSTRAINT, and it is not a choice:

    dsv2-lite.{formality,language,refusal,sentiment}   4080 rows each
    mistral-7b-instruct-v0.3.language                  4080 rows

and nothing else. Fifteen of the twenty PASS class columns have no
per-generation scores desk-side at all, so no pair from them can be
stratified by measured effect. See `COVERAGE_HALT` — the desk must see
this, because it means v2's graded strata rest on dsv2-lite plus one
mistral column.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from metabasis.l4_blind.models import Axis, Dose, Side

#: Per-generation head scores. The corrected decode, per the desk ruling
#: that dsv2 text comes from the corrected logs.
CORRECTED_ROWS_REL: Final[str] = (
    "staging/reading-bleed/corrected-l1-dsv2/node-pass1/CORRECTED-GENERATION-ROWS.jsonl"
)

#: Stratum thresholds on the measured delta. The 0.05–0.10 gap is a
#: deliberate BUFFER: pairs landing in it join no stratum, so "moderate"
#: and "expected-null" cannot blur into each other at the boundary.
STRONG_MIN: Final[float] = 0.50
MODERATE_MIN: Final[float] = 0.10
NULL_MAX: Final[float] = 0.05

#: The langid R-2 NULL rule: s_language is undefined when P(en)+P(fr) is
#: below this. Such a generation has no measurable on-axis value and is
#: dropped from the pool — never imputed, never treated as zero.
LANGID_FLOOR: Final[float] = 0.10

#: chars/words above this means the generation decoded without spaces.
#: Validated 55/55 against the panels whose real text is banked: the
#: degenerate ones sit at 61–110 chars/word, the normal ones at most 11.8.
#: This is how degeneracy is known for generations whose text is NOT yet
#: decoded — the ratio is in the corrected rows for all 20 400.
DEGENERATE_CHARS_PER_WORD: Final[float] = 20.0

COVERAGE_HALT: Final[str] = (
    "Per-generation on-axis scores exist desk-side for FIVE columns only "
    "(dsv2-lite x 4 axes, mistral-7b-instruct-v0.3.language). The other 15 "
    "detection-PASS class columns have cell-level means only, so no pair "
    "from them can be assigned to a measured-effect stratum. v2's graded "
    "strata therefore rest on dsv2-lite plus one mistral column — and "
    "dsv2-lite is the column with the space-degenerate decode. Widening "
    "the pool needs a per-generation L1 pass on the remaining columns "
    "(the amended lane in staging/instrument-fixes/ is the instrument for "
    "it). REPORTED, not worked around."
)


class CandidatePair(BaseModel):
    """One (steered, baseline) pair with its measured on-axis effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    node_key: str
    axis: Axis
    side: Side
    arm: str
    site: int
    dose: Dose
    generation_id: int = Field(ge=0)
    dose_cell_id: str
    baseline_cell_id: str

    #: The signed, direction-corrected effect. Positive = the dose moved
    #: the on-axis score the way it intended.
    delta: float
    s_dose: float
    s_baseline: float

    #: Either panel decoded without spaces (proxy, see the constant).
    space_degenerate: bool

    stratum: str  # strong | moderate | expected_null

    @property
    def natural_id(self) -> str:
        return f"{self.axis}/{self.side}|{self.column}|{self.dose}|{self.generation_id:03d}"


def _cell_id(axis: Axis, side: Side, site: int, dose: Dose) -> str:
    if side == "transported":
        return f"gcaa_{axis}_L{site}_a{dose}"
    return f"caa_{axis}_L{site}_L{site}_a{dose}"


def load_scores(repo: Path) -> tuple[dict, dict]:
    """(column, cell_id, generation_id) -> head scores, and -> chars/words."""
    p = repo / CORRECTED_ROWS_REL
    if not p.is_file():
        raise ValueError(f"per-generation scores missing: {p}")
    scores: dict[tuple[str, str, int], dict] = {}
    shape: dict[tuple[str, str, int], float | None] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            col = f"{r['node_key']}.{r['axis']}"
            k = (col, r["cell_id"], int(r["generation_id"]))
            scores[k] = r
            w, c = r.get("words_corrected"), r.get("chars_corrected")
            shape[k] = (c / w) if (w and c) else None
    return scores, shape


def assign_stratum(axis: Axis, dose: Dose, delta: float) -> str | None:
    """Which stratum a measured delta belongs to, or None for the buffer.

    The one BY-CONSTRUCTION rule: language at -0.30 is null whatever it
    measures. The axis is one-sided — a model generating English cannot
    generate "less French" than none — so a -0.30 language pair is a
    catch trial by design, not by measurement.
    """
    if axis == "language" and dose == "-0.30":
        return "expected_null"
    if delta >= STRONG_MIN:
        return "strong"
    if MODERATE_MIN <= delta < STRONG_MIN:
        return "moderate"
    if delta < NULL_MAX:
        return "expected_null"
    return None


def build_pool(repo: Path, pass_cells: list) -> list[CandidatePair]:
    """Every PASS-cell pair that carries a measured on-axis delta.

    Composition rules (desk, 2026-08-07):
      * refusal EXCLUDED entirely — substrate-free pool, returns with v2
        of the prompt pool;
      * language graded at +0.30 ONLY; its -0.30 pairs are catch trials;
      * formality and sentiment stratified by their actual per-pair deltas.
    """
    scores, shape = load_scores(repo)
    out: list[CandidatePair] = []

    for c in pass_cells:
        if c.axis == "refusal":
            continue
        if c.axis == "egv":
            continue
        key = f"s_{c.axis}"
        sign = 1.0 if c.dose.startswith("+") else -1.0
        dcell = _cell_id(c.axis, c.side, c.site, c.dose)
        bcell = f"baseline_L{c.site}_a+0.00"
        if (c.column, dcell, 0) not in scores:
            continue  # no per-generation scores for this column — coverage halt
        for g in range(80):
            a = scores.get((c.column, dcell, g))
            b = scores.get((c.column, bcell, g))
            if a is None or b is None:
                continue
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            if c.axis == "language":
                if (a["P_en"] + a["P_fr"]) < LANGID_FLOOR:
                    continue
                if (b["P_en"] + b["P_fr"]) < LANGID_FLOOR:
                    continue
            delta = (float(va) - float(vb)) * sign
            st = assign_stratum(c.axis, c.dose, delta)
            if st is None:
                continue
            if c.axis == "language" and c.dose == "-0.30" and st != "expected_null":
                continue
            ra = shape.get((c.column, dcell, g))
            rb = shape.get((c.column, bcell, g))
            degen = bool(
                (ra is not None and ra > DEGENERATE_CHARS_PER_WORD)
                or (rb is not None and rb > DEGENERATE_CHARS_PER_WORD)
            )
            out.append(
                CandidatePair(
                    column=c.column, node_key=c.node_key, axis=c.axis, side=c.side,
                    arm=c.arm, site=c.site, dose=c.dose, generation_id=g,
                    dose_cell_id=dcell, baseline_cell_id=bcell,
                    delta=round(delta, 6), s_dose=float(va), s_baseline=float(vb),
                    space_degenerate=degen, stratum=st,
                )
            )
    return out
