"""Deck v2 — a revealed calibration block, then a blind block stratified
by MEASURED effect.

Same machinery as v1 (sha256 key-sort, no PRNG, no `hash()`), same
counterbalance discipline, same sealed-map/blind-deck separation. What
changes is WHAT gets drawn: every candidate pair now carries a measured
on-axis delta (see `strata.py`), and the draw is per stratum.

    CALIBRATION BLOCK  ~3 per usable axis, REVEALED (steered side named).
                       The max-delta exemplars per (axis, side), restricted
                       to non-degenerate panels — an anchor she cannot
                       misread is the entire point. Never gold; excluded
                       from the blind block so nothing she saw labelled
                       can come back as a test.

    BLIND BLOCK        per (axis, side, stratum), seeded key-sort:
                         strong        6 per (axis, side)
                         moderate      up to 6, whatever the pool holds
                         expected_null 4 per (axis, side)   ← CATCH TRIALS
                       Sizes are PROPOSED from the real spread, not forced
                       to 16/type — the moderate stratum genuinely does not
                       exist at that size for sentiment or language.

Within a stratum the draw PREFERS non-degenerate panels: the stratum is
defined by the measured delta, so choosing readable panels inside it
cannot bias the effect size it represents. Degenerate pairs are taken
only once the clean ones run out, and any that land in the deck are
flagged (the standing rule: kept if drawn, flagged).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.draw import _ksort
from metabasis.l4_blind.models import CellTypeSpec, PairCoord
from metabasis.l4_blind.strata import (
    COVERAGE_HALT,
    CandidatePair,
    MODERATE_MIN,
    NULL_MAX,
    STRONG_MIN,
    build_pool,
)
from metabasis.l4_blind.taxonomy import AXIS_TRAIT, TaxonomyInputs, load_pass_population

DRAW_RECIPE_V2: Final[str] = "L4-MICROGOLD-DRAW|v2-stratified"

#: Per (axis, side) targets. `moderate` is a CAP, not a quota — the pool
#: decides, and the shortfall is reported rather than back-filled from a
#: neighbouring stratum (which would silently mix effect sizes).
TARGETS: Final[dict[str, int]] = {"strong": 6, "moderate": 6, "expected_null": 4}

#: Calibration exemplars per axis. Taken as: the top-delta pair from each
#: side, then the next best across both.
CALIBRATION_PER_AXIS: Final[int] = 3

STRATUM_ORDER: Final[tuple[str, ...]] = ("strong", "moderate", "expected_null")


class DrawnV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str = "l4-sealed-map/v2"
    tool_version: str = TOOL_VERSION
    grade: str = GRADE
    status: str = STATUS
    draw_recipe: str = DRAW_RECIPE_V2
    root_material: str
    deck_version: str = "v2-stratified"

    supersedes: str = (
        "deck v1 (effect-blind, 8 types x 16). Superseded 2026-08-07 after "
        "Luxia's first 10 pairs: 4 were expected-null by construction and the "
        "graded ones were .05-.13 population shifts. Her v1 verdicts are "
        "archived untouched as honest blind data."
    )
    stratum_rule: str = (
        f"delta = (s_axis[steered] - s_axis[baseline]) * sign(dose), same "
        f"prompt both sides. strong >= {STRONG_MIN}; moderate "
        f"[{MODERATE_MIN}, {STRONG_MIN}); expected_null < {NULL_MAX}; the "
        f"gap [{NULL_MAX}, {MODERATE_MIN}) is an unused buffer so the strata "
        f"cannot blur at the boundary. language@-0.30 is expected_null BY "
        f"CONSTRUCTION (one-sided axis) whatever it measures."
    )
    composition_rule: str = (
        "refusal EXCLUDED entirely (substrate-free pool; returns with pool "
        "v2). language graded at +0.30 ONLY. formality and sentiment "
        "stratified by their actual measured per-pair deltas."
    )
    coverage_halt: str = COVERAGE_HALT
    targets: dict[str, int] = Field(default_factory=lambda: dict(TARGETS))

    inputs: TaxonomyInputs
    types: list[CellTypeSpec]
    pairs: list[PairCoord]
    calibration: list[PairCoord]
    pool_census: dict[str, int]
    shortfalls: list[str]
    prompt_alignment_verified: str
    unblinding_warning: str = (
        "SEALED. Names every source and every measured delta. The serving "
        "path must never read it. The calibration block's coordinates are "
        "REVEALED to Luxia by design; everything else is not."
    )


def _pair_id(root: str, natural_id: str) -> str:
    return "L4-" + hashlib.sha256(f"{root}|pairid|{natural_id}".encode()).hexdigest()[:12]


def _to_coord(
    c: CandidatePair, *, root: str, set_label: str, stratum: str,
    dose_first: bool, ordinal_in_type: int, revealed: bool,
) -> PairCoord:
    return PairCoord(
        pair_id=_pair_id(root, c.natural_id),
        type_key=f"{c.axis}/{c.side}",
        set_label=set_label,
        axis=c.axis, side=c.side, column=c.column, node_key=c.node_key,
        arm=c.arm, site=c.site, dose=c.dose,
        dose_cell_id=c.dose_cell_id, baseline_cell_id=c.baseline_cell_id,
        generation_id=c.generation_id,
        prompt_id=f"P{c.generation_id:03d}",
        dose_first=dose_first,
        ordinal_in_type=ordinal_in_type, ordinal_global=0,
        natural_id=c.natural_id,
        stratum=stratum, measured_delta=c.delta,
        s_dose=c.s_dose, s_baseline=c.s_baseline,
        space_degenerate=c.space_degenerate, revealed=revealed,
    )


def freeze_draw_v2(inputs: TaxonomyInputs) -> DrawnV2:
    root = (
        f"{DRAW_RECIPE_V2}|{inputs.desk_score_sha256}|{inputs.ingredients_sha256}"
    )
    repo = Path(inputs.repo_root)
    pass_cells = load_pass_population(inputs)
    pool = build_pool(repo, pass_cells)
    if not pool:
        raise ValueError("empty measured-effect pool — refusing to draw")

    census: dict[str, int] = {}
    for p in pool:
        census[f"{p.axis}/{p.side}/{p.stratum}"] = census.get(
            f"{p.axis}/{p.side}/{p.stratum}", 0) + 1

    axes = sorted({p.axis for p in pool})
    sides = ("native", "transported")

    # ── calibration: max-delta, non-degenerate, per (axis, side) ──────
    calibration: list[PairCoord] = []
    used: set[str] = set()
    for axis in axes:
        picks: list[CandidatePair] = []
        for side in sides:
            cands = [
                p for p in pool
                if p.axis == axis and p.side == side
                and p.stratum == "strong" and not p.space_degenerate
            ]
            if not cands:
                continue
            # highest delta first; sha key breaks exact ties (language
            # saturates at .999, so ties are real and must not depend on
            # file order).
            cands = sorted(
                cands,
                key=lambda c: (-c.delta, hashlib.sha256(
                    f"{root}|caltie|{c.natural_id}".encode()).hexdigest()),
            )
            picks.append(cands[0])
        # fill to CALIBRATION_PER_AXIS with the next best across sides
        rest = [
            p for p in pool
            if p.axis == axis and p.stratum == "strong" and not p.space_degenerate
            and p.natural_id not in {x.natural_id for x in picks}
        ]
        rest = sorted(
            rest,
            key=lambda c: (-c.delta, hashlib.sha256(
                f"{root}|caltie|{c.natural_id}".encode()).hexdigest()),
        )
        picks.extend(rest[: max(0, CALIBRATION_PER_AXIS - len(picks))])
        for i, c in enumerate(picks):
            used.add(c.natural_id)
            calibration.append(
                _to_coord(
                    c, root=root, set_label="Calibration", stratum="calibration",
                    # revealed pairs still alternate which side is shown
                    # first, so the anchor does not teach "steered is always
                    # on the left".
                    dose_first=(i % 2 == 0),
                    ordinal_in_type=len(calibration), revealed=True,
                )
            )

    # ── blind block, per (axis, side, stratum) ────────────────────────
    labels = _set_labels_v2(axes, sides, root)
    blind: list[PairCoord] = []
    shortfalls: list[str] = []
    specs: list[CellTypeSpec] = []

    for axis in axes:
        for side in sides:
            n_type = 0
            for stratum in STRATUM_ORDER:
                want = TARGETS[stratum]
                cands = [
                    p for p in pool
                    if p.axis == axis and p.side == side and p.stratum == stratum
                    and p.natural_id not in used
                ]
                if not cands:
                    if want:
                        shortfalls.append(
                            f"{axis}/{side}/{stratum}: pool empty, wanted {want}"
                        )
                    continue
                clean = [c for c in cands if not c.space_degenerate]
                dirty = [c for c in cands if c.space_degenerate]
                mat = f"{root}|pick|{axis}|{side}|{stratum}"
                ordered = (
                    _ksort(clean, mat, lambda c: c.natural_id)
                    + _ksort(dirty, mat, lambda c: c.natural_id)
                )
                take = ordered[:want]
                if len(take) < want:
                    shortfalls.append(
                        f"{axis}/{side}/{stratum}: took {len(take)} of {want} "
                        f"(pool has {len(cands)}, {len(clean)} clean)"
                    )
                # counterbalance INSIDE the stratum: half dose-first
                flip = _ksort(take, f"{root}|flip|{axis}|{side}|{stratum}",
                              lambda c: c.natural_id)
                first = {c.natural_id for c in flip[: len(flip) // 2]}
                for c in take:
                    used.add(c.natural_id)
                    blind.append(
                        _to_coord(
                            c, root=root, set_label=labels[(axis, side)],
                            stratum=stratum, dose_first=c.natural_id in first,
                            ordinal_in_type=n_type, revealed=False,
                        )
                    )
                    n_type += 1
            drawn_here = [p for p in blind if p.axis == axis and p.side == side]
            if drawn_here:
                specs.append(
                    CellTypeSpec(
                        type_key=f"{axis}/{side}", axis=axis, side=side,
                        set_label=labels[(axis, side)],
                        n_pass_cells=len({p.column for p in pool
                                          if p.axis == axis and p.side == side}),
                        n_pass_cells_plus=len({p.column for p in pool
                                               if p.axis == axis and p.side == side
                                               and p.dose == "+0.30"}),
                        n_pass_cells_minus=len({p.column for p in pool
                                                if p.axis == axis and p.side == side
                                                and p.dose == "-0.30"}),
                        columns=sorted({p.column for p in pool
                                        if p.axis == axis and p.side == side}),
                        trait=AXIS_TRAIT[axis], pairs_drawn=len(drawn_here),
                    )
                )

    # ── global interleave (blind block only; calibration leads) ───────
    inter = _ksort(blind, f"{root}|global", lambda p: p.natural_id)
    blind_final = [p.model_copy(update={"ordinal_global": i}) for i, p in enumerate(inter)]
    cal_final = [p.model_copy(update={"ordinal_global": i}) for i, p in enumerate(calibration)]

    ids = [p.pair_id for p in blind_final + cal_final]
    if len(set(ids)) != len(ids):
        raise ValueError("pair_id collision across the v2 draw")
    nats = [p.natural_id for p in blind_final + cal_final]
    if len(set(nats)) != len(nats):
        raise ValueError("a generation was drawn twice (calibration/blind overlap?)")

    return DrawnV2(
        root_material=root, inputs=inputs, types=specs,
        pairs=blind_final, calibration=cal_final,
        pool_census=dict(sorted(census.items())), shortfalls=shortfalls,
        prompt_alignment_verified=(
            "every pair is (steered g, baseline g) at the SAME generation_id, "
            "and the banked cells carry prompt_id P000..P079 aligned 1:1 with "
            "generation_id (verified at v1 freeze against all 20 columns' "
            "generations.jsonl by value)"
        ),
    )


def _set_labels_v2(axes, sides, root: str) -> dict[tuple[str, str], str]:
    keys = [(a, s) for a in axes for s in sides]
    shuffled = _ksort(keys, f"{root}|setlabels", lambda k: f"{k[0]}/{k[1]}")
    return {k: f"Set {chr(ord('A') + i)}" for i, k in enumerate(shuffled)}
