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

#: The widened source (deck v2.1): a per-generation L1 pass over the 15
#: remaining detection-PASS class columns, produced with the amended
#: logger IMPORTED as the instrument of record and proven against the
#: banked L1 logs cell-by-cell, head-by-head. Optional — absent, the pool
#: is the narrow v2 one and `COVERAGE_HALT` still applies.
PER_GENERATION_ROWS_REL: Final[str] = (
    "staging/l4-tool/node-job/pulled-pergen/PER-GENERATION-L1-2026-08-07.jsonl"
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
#: This is how degeneracy is known for a generation whose text has NOT
#: been decoded yet — the ratio is in the score rows for all of them.
#:
#: RE-VALIDATED TWICE, and it is a WEAK proxy — which is why the real
#: measure runs at deck build (`bank.glued_fraction`) and why the
#: calibration anchors are hard-guarded there.
#:
#: The first calibration used 55 panels that were all either fully glued
#: or fully clean, and set this at 20.0. That let a PARTIALLY glued panel
#: through as a calibration anchor: its lines were glued
#: ("youmeanthowcantusllyteachsomeone") but separated by newlines, so it
#: scored 12.03 chars/word and a whitespace ratio of 0.083 — inside every
#: threshold then in force. Against all 424 banked panels, scored by the
#: direct measure, 24 are glued and their chars/word runs from 7.1 to
#: 696; clean panels top out at 8.14. No chars/word threshold separates
#: them cleanly. 9.0 is the conservative choice: zero false positives,
#: and it removes the worst 16 of the 24 from the candidate pool before
#: any draw sees them.
DEGENERATE_CHARS_PER_WORD: Final[float] = 9.0

#: A panel shorter than this is a stub, not an utterance, and is not
#: judgeable as one side of a 2AFC. Such generations are removed from the
#: CANDIDATE POOL rather than caught later at deck build: a pair that
#: cannot be shown is not a candidate, and discovering that after the
#: draw would leave a stratum silently short.
#:
#: This is an ELIGIBILITY rule, not an effect filter — it looks only at
#: length, never at delta, so it cannot bias the effect size a stratum
#: represents. It mirrors `bank.MIN_CHARS`, which remains as the
#: belt-and-braces check at build time.
MIN_PANEL_CHARS: Final[int] = 80

#: Generations MEASURED to be glued against their real decoded text, by
#: `bank.glued_fraction`. Evidence, not guesswork — and the answer to the
#: chars/word proxy's blind spot for PARTIAL gluing, which no
#: length-derived threshold separates.
#:
#: A glued panel is excluded from the CANDIDATE POOL, not merely from the
#: anchors: its on-axis score is a reading of mangled text, so its delta
#: is not a measurement of steering and it should not be stratified by
#: one. The list grows as text lands; anything still missed is caught by
#: the hard anchor guard at deck build.
GLUED_EXCLUSIONS_REL: Final[str] = "staging/l4-tool/GLUED-EXCLUSIONS-2026-08-07.json"

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


def load_scores(repo: Path) -> tuple[dict, dict, list[str]]:
    """(column, cell_id, generation_id) -> head scores, -> chars/words, sources.

    Two sources, same shape after normalisation:

      * the corrected dsv2 rows (5 columns), whose length fields are named
        `*_corrected` because they came from the corrected-decode pass;
      * the widened per-generation pass (15 columns), if it has landed.

    A key present in both would be a genuine conflict — the same
    generation scored twice by the same heads — so it HALTs rather than
    letting one source silently win.
    """
    scores: dict[tuple[str, str, int], dict] = {}
    shape: dict[tuple[str, str, int], tuple[float | None, int | None]] = {}
    sources: list[str] = []

    def ingest(path: Path, chars_key: str, words_key: str) -> None:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                col = r.get("column") or f"{r['node_key']}.{r['axis']}"
                k = (col, r["cell_id"], int(r["generation_id"]))
                if k in scores:
                    raise ValueError(
                        f"{k} appears in two score sources — refusing to pick "
                        f"one silently; the desk must reconcile them"
                    )
                scores[k] = r
                w, c = r.get(words_key), r.get(chars_key)
                shape[k] = ((c / w) if (w and c) else None, c)

    corrected = repo / CORRECTED_ROWS_REL
    if not corrected.is_file():
        raise ValueError(f"per-generation scores missing: {corrected}")
    ingest(corrected, "chars_corrected", "words_corrected")
    sources.append(CORRECTED_ROWS_REL)

    widened = repo / PER_GENERATION_ROWS_REL
    if widened.is_file():
        ingest(widened, "chars", "words")
        sources.append(PER_GENERATION_ROWS_REL)

    return scores, shape, sources


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
    scores, shape, _sources = load_scores(repo)
    glued: set[str] = set()
    gp = repo / GLUED_EXCLUSIONS_REL
    if gp.is_file():
        glued = set(json.loads(gp.read_text()))
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
            ra, ca = shape.get((c.column, dcell, g), (None, None))
            rb, cb = shape.get((c.column, bcell, g), (None, None))
            # eligibility: a stub panel cannot be judged, so it is not a
            # candidate. Length only — never delta.
            if (ca is not None and ca < MIN_PANEL_CHARS) or (
                    cb is not None and cb < MIN_PANEL_CHARS):
                continue
            if (f"{c.column}|{dcell}|{g}" in glued
                    or f"{c.column}|{bcell}|{g}" in glued):
                continue
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
