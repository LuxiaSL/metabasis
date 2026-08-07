"""The deterministic, frozen pair draw.

THE SEED RECIPE, named and stated once (this docstring is the artifact's
own explanation; `DRAW_RECIPE` is what rides every file):

    root_material = "L4-MICROGOLD-DRAW|v1|{desk_score_sha256}|{ingredients_sha256}"

Every selection below is a KEY-SORT, never a PRNG:

    key(material, item) = sha256(f"{material}|{item}").hexdigest()

and a selection of k from a list is "sort by key ascending (ties broken on
the item's own canonical id), take the first k". This is deliberate and it
is the whole reason the draw is re-derivable:

  * `random.shuffle` / `random.sample` are NOT contract-stable across
    CPython versions — the Mersenne stream is, but the consumption pattern
    inside those helpers is an implementation detail. A key-sort is defined
    entirely by sha256 and a total order.
  * `hash()` is salted per process (M25) and is banned outright.
  * The desk can re-derive any single choice with one `sha256sum` and a
    sort, without running this code.

The four seeded steps, each with its own material so that changing one
cannot perturb another:

  1. `…|setlabels`            — permute the cell types onto opaque "Set A"…
                                labels (the blind's per-type progress).
  2. `…|cellorder|{tk}|{sign}`— order a type's PASS cells for the
                                round-robin allocation of its 8 pairs.
  3. `…|gen|{tk}|{col}|{dose}`— choose which generation ids that cell
                                contributes (without replacement).
  4. `…|flip|{tk}|{sign}`     — assign presentation order, exactly half
                                dose-first WITHIN each (type, sign).
  5. `…|order|{tk}` / `…|global` — shuffle within a type, then interleave
                                all types into one presentation sequence.

Balance guarantees, asserted in `freeze_draw` and re-asserted by the
selftests:

  * exactly PAIRS_PER_TYPE pairs per type;
  * exactly PAIRS_PER_TYPE/2 per dose sign within each type;
  * exactly PAIRS_PER_TYPE/4 dose-first within each (type, sign), so the
    counterbalance survives any post-hoc slice by sign;
  * no generation id drawn twice within a cell;
  * every pair id unique across the whole draw.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Final, Iterable, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.models import Dose, PairCoord, CellTypeSpec, PassCell
from metabasis.l4_blind.taxonomy import (
    EXCLUDED_TYPES,
    AXIS_TRAIT,
    GENERATIONS_PER_CELL,
    PAIRS_PER_TYPE,
    TAXONOMY_ALTERNATIVES,
    TAXONOMY_RULE,
    TaxonomyInputs,
    group_by_type,
    load_pass_population,
)

#: The named recipe. Rides the sealed map, the deck and the session seal.
DRAW_RECIPE: Final[str] = "L4-MICROGOLD-DRAW|v1"

#: Canonical dose order — fixed, never sorted from a set.
SIGN_ORDER: Final[tuple[Dose, ...]] = ("+0.30", "-0.30")

_T = TypeVar("_T")


def _key(material: str, item: str) -> str:
    return hashlib.sha256(f"{material}|{item}".encode("utf-8")).hexdigest()


def _ksort(items: Iterable[_T], material: str, idfn: Callable[[_T], str]) -> list[_T]:
    """Deterministic shuffle: sort by sha256 key, tie-break on canonical id."""
    return sorted(items, key=lambda x: (_key(material, idfn(x)), idfn(x)))


def _cell_id(c: PassCell) -> str:
    return f"{c.column}|{c.side}|{c.dose}"


class DrawnPairs(BaseModel):
    """The frozen draw: the sealed coordinates plus the type specs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str = Field(default="l4-sealed-map/v1")
    tool_version: str = TOOL_VERSION
    grade: str = GRADE
    status: str = STATUS
    draw_recipe: str = DRAW_RECIPE
    root_material: str
    taxonomy_rule: str = TAXONOMY_RULE
    taxonomy_alternatives: list[str] = Field(default_factory=lambda: list(TAXONOMY_ALTERNATIVES))
    pairs_per_type: int = PAIRS_PER_TYPE
    excluded_types: dict[str, str] = Field(default_factory=lambda: dict(EXCLUDED_TYPES))
    exclusion_independence: str = (
        "Exclusions are applied to the POPULATION before any seeded step, and "
        "every seeded material is keyed on a type's own type_key (plus the two "
        "input shas) — never on the roster or its length. Dropping a type "
        "therefore cannot re-draw a surviving one; only the opaque Set letters "
        "re-pack and ordinal_global renumbers, both of which are presentation. "
        "Asserted by selftest against a draw taken with the exclusion lifted."
    )
    inputs: TaxonomyInputs
    types: list[CellTypeSpec]
    pairs: list[PairCoord]
    prompt_alignment_verified: str
    unblinding_warning: str = (
        "SEALED. This file names every source. The serving path must never "
        "read it; the browser must never receive it. Unblinding is a "
        "separate desk step performed after the session seal is written."
    )


def _set_labels(type_keys: Sequence[str], root: str) -> dict[str, str]:
    """Opaque labels for the per-type progress bars.

    The judged TRAIT is necessarily visible (the L3 judge sees it too, and
    an agreement number over two different questions is meaningless), so
    trait-grouping is inferable from the page. What these labels hide is
    the split that actually carries the science: which set is `native` and
    which is `transported` on the same axis. Those two look identical on
    the page and land on different letters.
    """
    if len(type_keys) > 26:
        raise ValueError("more than 26 cell types — the Set A..Z labelling runs out")
    shuffled = _ksort(list(type_keys), f"{root}|setlabels", lambda s: s)
    return {tk: f"Set {chr(ord('A') + i)}" for i, tk in enumerate(shuffled)}


def _prompt_index(path: Path) -> dict[int, str]:
    """(generation_id -> prompt_id) for one banked cell.

    Reads only the two fields it needs; `generated_ids` is left on disk.
    """
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gid = int(row["generation_id"])
            if gid in out:
                raise ValueError(f"duplicate generation_id {gid} in {path}")
            out[gid] = str(row["prompt_id"])
    return out


def freeze_draw(inputs: TaxonomyInputs, *, apply_exclusions: bool = True) -> DrawnPairs:
    """Build the frozen draw. Pure function of the two input shas.

    Every assertion in here is a HALT, not a warning: a draw that quietly
    dropped a cell or lost its sign balance would corrupt the O-5 bar in a
    way no downstream artifact could see.
    """
    root = f"{DRAW_RECIPE}|{inputs.desk_score_sha256}|{inputs.ingredients_sha256}"
    repo = Path(inputs.repo_root)

    population = load_pass_population(inputs, apply_exclusions=apply_exclusions)
    by_type = group_by_type(population)
    labels = _set_labels(list(by_type), root)

    if PAIRS_PER_TYPE % 4 != 0:
        raise ValueError("PAIRS_PER_TYPE must be divisible by 4 for the 8/8 + 4/4 balance")
    per_sign = PAIRS_PER_TYPE // 2

    prompt_cache: dict[Path, dict[int, str]] = {}

    def prompts_for(rel_root: str, cell_id: str) -> dict[int, str]:
        p = repo / rel_root / cell_id / "generations.jsonl"
        if p not in prompt_cache:
            idx = _prompt_index(p)
            if len(idx) != GENERATIONS_PER_CELL:
                raise ValueError(
                    f"{p} has {len(idx)} generations, expected {GENERATIONS_PER_CELL}"
                )
            prompt_cache[p] = idx
        return prompt_cache[p]

    all_pairs: list[PairCoord] = []
    specs: list[CellTypeSpec] = []
    alignment_checked = 0

    for tk, cells in by_type.items():
        axis = cells[0].axis
        side = cells[0].side
        drawn: list[dict] = []

        for sign in SIGN_ORDER:
            sign_cells = [c for c in cells if c.dose == sign]
            if not sign_cells:
                raise ValueError(
                    f"type {tk} has no PASS cell at dose {sign} — the 8/8 sign "
                    f"balance is unsatisfiable; the desk must rule on this type"
                )
            order = _ksort(sign_cells, f"{root}|cellorder|{tk}|{sign}", _cell_id)
            # Round-robin over the seeded cell order: allocation is as even
            # as arithmetic allows, and WHICH cells get the remainder is
            # decided by the seed, not by the file order.
            alloc = Counter(_cell_id(order[i % len(order)]) for i in range(per_sign))

            for cell in order:
                k = alloc.get(_cell_id(cell), 0)
                if k == 0:
                    continue
                if k > GENERATIONS_PER_CELL:
                    raise ValueError(f"{_cell_id(cell)}: asked for {k} of {GENERATIONS_PER_CELL}")
                gens = _ksort(
                    range(GENERATIONS_PER_CELL),
                    f"{root}|gen|{tk}|{cell.column}|{cell.dose}",
                    lambda g: f"{g:03d}",
                )[:k]

                dose_prompts = prompts_for(cell.results_root, cell.dose_cell_id)
                base_prompts = prompts_for(cell.results_root, cell.baseline_cell_id)
                for g in gens:
                    pd, pb = dose_prompts.get(g), base_prompts.get(g)
                    if pd is None or pb is None:
                        raise ValueError(f"{_cell_id(cell)} generation {g} missing on one side")
                    if pd != pb:
                        raise ValueError(
                            f"prompt mismatch at {_cell_id(cell)} g={g}: dose {pd} "
                            f"vs baseline {pb} — the 2AFC requires the SAME prompt"
                        )
                    alignment_checked += 1
                    drawn.append(
                        {
                            "cell": cell,
                            "gen": g,
                            "prompt_id": pd,
                            "sign": sign,
                            "natural_id": f"{tk}|{cell.column}|{cell.side}|{cell.dose}|{g:03d}",
                        }
                    )

        if len(drawn) != PAIRS_PER_TYPE:
            raise ValueError(f"type {tk} drew {len(drawn)} pairs, expected {PAIRS_PER_TYPE}")

        # Presentation order: exactly a quarter of the type dose-first in
        # EACH sign, so slicing by sign later keeps the counterbalance.
        dose_first: set[str] = set()
        for sign in SIGN_ORDER:
            sub = [d for d in drawn if d["sign"] == sign]
            flip = _ksort(sub, f"{root}|flip|{tk}|{sign}", lambda d: d["natural_id"])
            for d in flip[: len(sub) // 2]:
                dose_first.add(d["natural_id"])

        ordered = _ksort(drawn, f"{root}|order|{tk}", lambda d: d["natural_id"])
        for i, d in enumerate(ordered):
            cell: PassCell = d["cell"]
            nid = d["natural_id"]
            pid = "L4-" + hashlib.sha256(f"{root}|pairid|{nid}".encode()).hexdigest()[:12]
            all_pairs.append(
                PairCoord(
                    pair_id=pid,
                    type_key=tk,
                    set_label=labels[tk],
                    axis=cell.axis,
                    side=cell.side,
                    column=cell.column,
                    node_key=cell.node_key,
                    arm=cell.arm,
                    site=cell.site,
                    dose=cell.dose,
                    dose_cell_id=cell.dose_cell_id,
                    baseline_cell_id=cell.baseline_cell_id,
                    generation_id=d["gen"],
                    prompt_id=d["prompt_id"],
                    dose_first=nid in dose_first,
                    ordinal_in_type=i,
                    ordinal_global=0,  # filled below
                    natural_id=nid,
                )
            )

        plus = [c for c in cells if c.dose == "+0.30"]
        minus = [c for c in cells if c.dose == "-0.30"]
        specs.append(
            CellTypeSpec(
                type_key=tk,
                axis=axis,
                side=side,
                set_label=labels[tk],
                n_pass_cells=len(cells),
                n_pass_cells_plus=len(plus),
                n_pass_cells_minus=len(minus),
                columns=sorted({c.column for c in cells}),
                trait=AXIS_TRAIT[axis],
                pairs_drawn=PAIRS_PER_TYPE,
            )
        )

    # Global interleave across types — the brief's "seeded shuffle across
    # types". Consecutive pairs land in different sets by construction of
    # the hash, so a run of one set cannot pace-bias the session.
    interleaved = _ksort(all_pairs, f"{root}|global", lambda p: p.natural_id)
    final = [p.model_copy(update={"ordinal_global": i}) for i, p in enumerate(interleaved)]

    ids = [p.pair_id for p in final]
    if len(set(ids)) != len(ids):
        raise ValueError("pair_id collision — widen the pair_id digest")
    nat = [p.natural_id for p in final]
    if len(set(nat)) != len(nat):
        raise ValueError("a generation was drawn twice — the without-replacement rule broke")

    return DrawnPairs(
        root_material=root,
        excluded_types=dict(EXCLUDED_TYPES) if apply_exclusions else {},
        inputs=inputs,
        types=specs,
        pairs=final,
        prompt_alignment_verified=(
            f"{alignment_checked}/{len(final)} drawn pairs verified to carry the "
            f"SAME prompt_id in the dose cell and the baseline cell (checked by "
            f"value against the banked generations.jsonl; a mismatch HALTs)"
        ),
    )


def deck_request(drawn: DrawnPairs) -> list[dict[str, object]]:
    """The exact coordinate rows a decode job must return text for.

    Two rows per pair (the dose generation and its baseline twin). This is
    the ONLY thing the desk needs to hand the node: no model, no bank, no
    GPU — the decode job already exists (`mb-l2decode-full`, tokenizers
    only, 0 GPUs) and this is a strict subset of what it already decodes.
    """
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for p in drawn.pairs:
        for cell_id, role in ((p.dose_cell_id, "dose"), (p.baseline_cell_id, "baseline")):
            key = (p.column, p.side, cell_id, p.generation_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "column": p.column,
                    "side": p.side,
                    "node_key": p.node_key,
                    "cell_id": cell_id,
                    "generation_id": p.generation_id,
                    "prompt_id": p.prompt_id,
                    "role": role,
                }
            )
    return sorted(rows, key=lambda r: (r["column"], r["side"], r["cell_id"], r["generation_id"]))
