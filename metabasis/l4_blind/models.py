"""Typed records for the L4 blind tool. Pydantic v2 throughout, all strict.

Three populations of record live here, and the type system is what keeps
them apart:

  * `PairCoord`   — the SEALED side. Names the column, side, dose, cell and
                    generation. NEVER reaches the browser.
  * `BlindPair`   — the OPEN side. Opaque pair id, opaque set label, the
                    judged trait, and two texts in presentation order.
                    This is the ONLY shape the page ever sees.
  * `VerdictRow`  — what Luxia's keystrokes append. Blind-side vocabulary
                    only ("text_1"/"text_2"/"unsure").

A `BlindPair` carries no field that could name a source; that is enforced
by `model_config = ConfigDict(extra="forbid")` plus the leak selftest.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── vocabulary ───────────────────────────────────────────────────────────

#: The five behavioral axes in the detection-PASS population. "egv" is the
#: entropy-gradient (whole-column / EGV-transported wave, `axis = null` in
#: the L2 ingredients); the other four are the class-pilot axes.
Axis = Literal["egv", "formality", "language", "refusal", "sentiment"]

#: Which side of the transport the cell sits on. The PASS population has
#: transported cells on every axis and native cells on the four class axes
#: (EGV native rows are the calibration half, BESIDE — not in this draw).
Side = Literal["native", "transported"]

#: Scoring granularity of record is per-dose (FLAG-A, L2 ruling 2).
Dose = Literal["+0.30", "-0.30"]

#: What a keystroke can record. "unsure" is its OWN state and is never
#: coerced to a pick (brief requirement 3).
Choice = Literal["text_1", "text_2", "unsure"]


class Frozen(BaseModel):
    """Base: immutable, no stray fields, no silent coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


# ── sealed side ──────────────────────────────────────────────────────────


class PairCoord(Frozen):
    """One drawn A/B pair, fully identified. SEALED — never rendered.

    The pair is (dose-cell generation g, baseline generation g) on the SAME
    prompt: the banked cells carry prompt_id P000..P079 aligned 1:1 with
    generation_id 0..79, verified desk-side before the freeze.
    """

    pair_id: str = Field(pattern=r"^L4-[0-9a-f]{12}$")
    type_key: str
    set_label: str

    axis: Axis
    side: Side
    column: str
    node_key: str
    arm: str
    site: int = Field(ge=0)
    dose: Dose

    dose_cell_id: str
    baseline_cell_id: str
    generation_id: int = Field(ge=0)
    prompt_id: str = Field(pattern=r"^P\d{3}$")

    #: Presentation order, seeded and counterbalanced within (type, sign):
    #: True => the DOSE text is shown as "Text 1".
    dose_first: bool

    #: Position of this pair inside its type (0-based, after the seeded
    #: within-type shuffle) and in the global cross-type interleave.
    ordinal_in_type: int = Field(ge=0)
    ordinal_global: int = Field(ge=0)

    # ── deck v2: the measured-effect layer ───────────────────────────
    #: strong | moderate | expected_null | calibration.
    #: `expected_null` pairs are CATCH TRIALS — scored later for
    #: unsure-honesty, never for agreement. `calibration` pairs are
    #: shown REVEALED and are never gold.
    stratum: str = "unstratified"
    #: The on-axis classifier delta, direction-corrected. None for v1.
    measured_delta: float | None = None
    s_dose: float | None = None
    s_baseline: float | None = None
    #: Either panel decoded without spaces (the dsv2 decode issue).
    space_degenerate: bool = False
    #: True for the revealed calibration block: the page NAMES the
    #: steered panel for these, so they anchor rather than test.
    revealed: bool = False

    #: The material the deterministic key-sort consumed for this pair, kept
    #: verbatim so the desk can re-derive the draw without reading code.
    natural_id: str


class CellTypeSpec(Frozen):
    """One cell TYPE — the unit the O-5 bar (≥80% agreement) applies to."""

    type_key: str
    axis: Axis
    side: Side
    set_label: str
    n_pass_cells: int = Field(ge=1)
    n_pass_cells_plus: int = Field(ge=0)
    n_pass_cells_minus: int = Field(ge=0)
    columns: list[str]
    trait: str
    pairs_drawn: int = Field(ge=0)


class PassCell(Frozen):
    """A detection-PASS cell from the L2 desk scoring, with its coordinates
    resolved against the L2 ingredients (arm / axis / site / node_key) and
    against the banked cell directories."""

    column: str
    side: Side
    dose: Dose
    axis: Axis
    arm: str
    node_key: str
    site: int
    wave: str
    signal_auc: float
    null_band_max: float
    dose_cell_id: str
    baseline_cell_id: str
    results_root: str
    type_key: str


# ── open (blind) side ────────────────────────────────────────────────────


class BlindPair(Frozen):
    """What the page sees. Nothing here can name a source.

    `set_label` is an OPAQUE letter assigned by a seeded permutation of the
    cell types, so the per-set progress bars the brief asks for do not
    disclose which set is native and which is transported.
    """

    pair_id: str = Field(pattern=r"^L4-[0-9a-f]{12}$")
    set_label: str = Field(pattern=r"^(Set [A-Z]|Calibration)$")
    trait: str
    text_1: str
    text_2: str

    #: REVEALED pairs only (the calibration block). Names which panel was
    #: steered, and by how much the on-axis score moved, so Luxia anchors
    #: on what a real effect looks like before judging anything blind.
    #: None on every blind pair — and the leak selftest asserts that the
    #: blind pairs carry no such field in the rendered page.
    revealed_steered: str | None = None
    revealed_delta: float | None = None
    revealed_direction: str | None = None

    @field_validator("text_1", "text_2")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("blind pair text is empty — a blank panel is unjudgeable")
        return v

    @model_validator(mode="after")
    def _distinct(self) -> BlindPair:
        if self.text_1 == self.text_2:
            raise ValueError(
                "blind pair texts are byte-identical — the 2AFC is undefined; "
                "the desk must rule on this pair before the session"
            )
        return self


class BlindDeck(Frozen):
    """The whole blind session, in presentation order. The ONLY file the
    server reads. Written beside — never inside — the sealed map."""

    schema_name: Literal["l4-blind-deck/v1"] = "l4-blind-deck/v1"
    tool_version: str
    grade: str
    status: str
    draw_recipe: str
    draw_sha256: str
    sealed_map_sha256: str
    bank_sha256: str
    set_labels: list[str]
    set_totals: dict[str, int]
    pairs: list[BlindPair]

    @model_validator(mode="after")
    def _totals_match(self) -> BlindDeck:
        seen: dict[str, int] = {}
        for p in self.pairs:
            seen[p.set_label] = seen.get(p.set_label, 0) + 1
        if seen != self.set_totals:
            raise ValueError(f"set_totals {self.set_totals} != observed {seen}")
        ids = [p.pair_id for p in self.pairs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate pair_id in deck")
        return self


# ── verdict stream ───────────────────────────────────────────────────────


class VerdictRow(BaseModel):
    """One appended line of the session log. Append-only: a re-judged pair
    appends a NEW row and the last row for a pair_id wins. Nothing is ever
    rewritten in place, so a kill mid-write loses at most the tail line."""

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["l4-verdict/v1"] = "l4-verdict/v1"
    tool_version: str
    grade: str
    session_id: str

    #: The DRAW sha the deck was built from — the same value the deck's
    #: `draw_sha256` and `sealed_map_sha256` carry, so a verdict line names
    #: the map that unblinds it without containing any of the map. NOT the
    #: deck FILE's sha (the session seal carries that separately, as
    #: `deck_sha256`); one name per quantity.
    draw_sha256: str

    kind: Literal["verdict", "session_note"] = "verdict"
    pair_id: str | None = None
    set_label: str | None = None
    ordinal_global: int | None = None
    choice: Choice | None = None
    note: str = ""

    #: Wall-clock, ISO-8601 UTC. `shown_at` is when the pair first became
    #: visible in this sitting; `answered_at` when the key landed.
    shown_at: str | None = None
    answered_at: str
    elapsed_ms: int | None = Field(default=None, ge=0)

    #: Which panel held which slot, in blind vocabulary. Recorded so the
    #: desk can debias position at unblinding without re-reading the deck.
    presentation: Literal["as_drawn"] = "as_drawn"

    #: True when this row supersedes an earlier verdict for the same pair.
    amends: bool = False

    @model_validator(mode="after")
    def _shape(self) -> VerdictRow:
        if self.kind == "verdict":
            if self.pair_id is None or self.choice is None:
                raise ValueError("a verdict row needs both pair_id and choice")
        else:
            if self.pair_id is not None or self.choice is not None:
                raise ValueError("a session_note row carries no pair_id/choice")
            if not self.note.strip():
                raise ValueError("an empty session note is not recorded")
        return self


class SessionSeal(Frozen):
    """The end-of-session sealed artifact: the draw recipe, the tool
    version, the input shas, and the verdict tally — sha'd on write.

    It carries counts only. It does NOT carry the sealed map, so sealing
    does not unblind: joining is the desk's separate step.
    """

    schema_name: Literal["l4-session-seal/v1"] = "l4-session-seal/v1"
    tool_version: str
    grade: str
    status: str
    session_id: str
    sealed_at: str

    draw_recipe: str
    draw_sha256: str
    sealed_map_sha256: str
    sealed_map_path: str
    deck_sha256: str
    deck_path: str
    bank_sha256: str
    verdict_log_path: str
    verdict_log_sha256: str

    n_pairs_in_deck: int = Field(ge=0)
    n_pairs_judged: int = Field(ge=0)
    n_unjudged: int = Field(ge=0)
    n_amendments: int = Field(ge=0)
    n_session_notes: int = Field(ge=0)
    n_pair_notes: int = Field(ge=0)
    choice_tally: dict[str, int]
    per_set_judged: dict[str, int]
    per_set_total: dict[str, int]
    median_seconds_per_pair: float | None = None

    unblinding: str = (
        "NOT PERFORMED HERE. Join this seal's verdict log to the sealed map "
        "(sealed_map_sha256) in a separate desk step."
    )
