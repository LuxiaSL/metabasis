"""The node-side behavioral engine — one load, all cells, batched (BRIEF §2).

The generation/injection harness of the behavioral phase. `BRIEF-behavioral-phase-
2026-07-29.md` (sha `475bc2a8…`) §2 is this module's specification verbatim and §13
holds the ten rulings that freeze its design choices; §9 is its HALT list, encoded
here as the named exceptions of `BehavioralHarnessError` rather than as warnings,
because an enactor never adjudicates a live HALT.

WHAT THIS MODULE IS. §2.1's one-load-per-node job: preflight → norm calibration →
canonical-layout determination → all calibration cells → all transported cells →
battery → entropy probe → in-job replay gate → stamps → manifest, with the model
loaded exactly once. This file owns the generation/injection/probe/replay mechanics
and the stamp. The §4 verdict logic lives in `actuation_calibration.py`; the §6
battery and coherence panel live in `capability_battery.py`; §2's CPU staging
(`build_behavioral_banks.py`) and scoring (`score_behavioral_column.py`) are
separate modules and are NOT in this batch.

THE FOUR PROPERTIES THAT MAKE THE FAST PATH LEGITIMATE, and where each is proved:

  1. **Layout invariance** (§2.2). A cell is 80 generations sharing ONE injection
     spec, so batching is legitimate — but only if a generation's token sequence is
     a function of the generation, never of the batch it rode in. Both halves of
     that are constructive here: the randomness is per-(cell, gen_id) pre-drawn
     (property 2), and the sampling step reads ONE row's logits, cast to float64
     numpy, through an inverse-CDF that touches no other row. So identical logits
     give a bitwise identical token at any B. Proved bitwise in `--selftest` at
     B ∈ {80, 40, 20, 10, 1} on a deterministic stub stepper; CHARACTERIZED (never
     asserted) on a real tiny Llama, because GPU — and, as the selftest measures,
     sometimes CPU — reductions are not batch-size invariant, which is exactly why
     §2.7 defines replay IN THE CANONICAL LAYOUT.

  2. **M5-safety by construction** (§2.3, ruling 8). Every generation's uniforms
     derive from its OWN sha256 digest over
     `{corpus_sha}|{node_key}|{arm}|L{site}|{cell_id}|{gen_id:03d}`, so a partial
     re-run of any subset of cells keeps every other cell in phase. M5's rake is a
     single stream that shifted phase when records were filtered (up to .0459 drift
     on q95s); this column WILL be re-run in parts. `hash()` appears nowhere (M25:
     PYTHONHASHSEED salts it, and the archived preview's nulls are irreproducible
     for exactly that reason).

  3. **α = 0 is bitwise no-hook** (§2.4). The hook short-circuits on `alpha == 0.0`
     and returns its args untouched, so the baseline cell with the hook attached is
     the unperturbed forward. Asserted once per node, cheaply, rather than trusted.

  4. **The in-job replay gate** (§2.7). After all cells complete, in the SAME
     process and the SAME canonical layout, K = 3 cells re-generate — selection
     deterministic from `sha256(node_key|corpus_sha)`, constrained to one signal
     cell at |0.3|, one random-band cell and one calibration cell — and both the
     token ids and the probe's float32 entropy arrays must be bitwise identical.
     Mismatch is a HALT and no cell from that node is quotable.

TWO SPEC AMBIGUITIES, RESOLVED EXPLICITLY AND REPORTED (never improvised silently):

  * **§2.6's probe batching.** The probe is specified right-padded with "per-row
    spans sliced by that row's own `prompt_length`", but the injection hook gates on
    ABSOLUTE positions with one `[seq_len]` mask per forward (`hooks.py` L181), so a
    right-padded batch of rows with DIFFERENT prompt lengths has no single valid
    mask — the steered probe would inject over prompt positions of the shorter rows.
    Resolution of necessity: probe batches group rows of EQUAL `prompt_length`
    (deterministic — ascending length, gen_id order within a length), which
    satisfies every literal §2.6 requirement and adds only the grouping. The
    realized grouping is recorded in the cell record and the stamp
    (`probe_grouping`), and `PROBE_GROUPING_READING` names it as a desk-owed
    reading rather than a frozen one. The named alternative — extending `hooks.py`
    to accept a per-row `[batch, seq]` mask — is a change to a frozen surface and
    is deliberately NOT taken here.
  * **§2.6's batch-1-vs-batch-32 sentence** both "asserts ... bit-exactness" and
    says the delta is "instrumentation and cannot fail the gate" (M19). M19 wins:
    `characterize_probe_batch_invariance` is descriptive-only and returns a record,
    and only the §2.7 replay gate — same layout, same process — can HALT.

CPU self-test (no weights, no GPU, no data tree):

    python -m metabasis.scripts.run_behavioral_cells --selftest

Node-side run (§2.1; Heimdall CLI only, the HTTP API is read-only verification):

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 \
    python -m metabasis.scripts.run_behavioral_cells --run \
        --node-key qwen2.5-3b-instruct --model-path <LOCAL_WEIGHTS_DIR> \
        --arm-root <ARM_ROOT> --work-root /models/metabasis-behavioral/<node> \
        --cells-json <STAGED_CELLS>.json --prompt-pool <POOL>.json --arm native

    python -m metabasis.scripts.run_behavioral_cells --preflight ...   # M10: a
        first-class exit-early mode, never output truncation.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_behavioral_cells")

BRIEF_OF_RECORD = "BRIEF-behavioral-phase-2026-07-29.md"
BRIEF_SHA256 = "475bc2a8ce767f70890837129644917f17094734b70bb510524be4aa0ecdcea4"

# ---------------------------------------------------------------- frozen constants
#: §2.5 / §11: the signed dose ladder is FROZEN. No extension, no interpolation, no
#: per-node tuning. The α=0 baseline is a separate cell and is deliberately NOT a
#: member of the ladder, so `len(DOSE_LADDER) == 6` is the ladder's own arity
#: everywhere (the §4.2 "4/6 doses" criterion counts against exactly this tuple).
DOSE_LADDER: tuple[float, ...] = (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3)
BASELINE_DOSE = 0.0
#: §4.2(a)/§5.5 read the ordering across the full SIGNED ladder, sign flipping
#: through zero; §4.2(b)'s "both |0.3| doses" are these two.
SCORING_DOSES: tuple[float, ...] = (-0.3, 0.3)
#: §2.2: the fixed batch-size ladder. Preflight takes the LARGEST that fits with
#: ≥15% VRAM headroom; B is then frozen in the node's stamp before any cell fires.
BATCH_LADDER: tuple[int, ...] = (80, 40, 20, 10)
VRAM_HEADROOM_FRACTION = 0.15
#: §5.1: n stays 80/cell and max_new_tokens stays 512 (§11: no deviation).
N_PER_CELL = 80
MAX_NEW_TOKENS = 512
#: §2.6's measured probe anchor (43.0 → 290.9 seq/s at len 273). The probe's realized
#: batch size is min(this, the size of the equal-prompt-length group) — see
#: `PROBE_GROUPING_READING`.
PROBE_BATCH_SIZE = 32
#: §2.3: the banked cell-id formatting, verbatim (precedent
#: `build_injection_banks.py`, parsed by `score_entropy_writes.parse_cell`).
CELL_ID_TEMPLATE = "{vector_key}_L{site}_a{frac:+.2f}"
#: §2.3: the seed-material template. Its own digest rides every stamp, so a stamp
#: names the recipe it was produced under and a changed recipe cannot pass silently.
SEED_MATERIAL_TEMPLATE = "{corpus_sha}|{node_key}|{arm}|L{site}|{cell_id}|{gen_id:03d}"
SEED_MATERIAL_TEMPLATE_DIGEST = hashlib.sha256(
    SEED_MATERIAL_TEMPLATE.encode()).hexdigest()
#: §2.7: K cells re-generate under the in-job gate, one from each required stratum.
REPLAY_GATE_K = 3
#: §2.5 / §9 item 8: measured vs banked per-token median residual norm.
NORM_DELTA_HALT_FRACTION = 0.10
#: §2.5: the fixed sample the in-job norm measurement runs over.
NORM_SAMPLE_SIZE = 64
#: §9 item 12 / Addendum B: GPT-2's learned absolute positions are a hard ceiling.
GPT2_POSITION_CEILING = 1024
#: ruling 10: the banked stage-0 pool, 20 topics × 4 strata × 1 seed.
PROMPT_POOL_TOPICS = 20
PROMPT_POOL_STRATA = 4
PROMPT_POOL_SIZE = PROMPT_POOL_TOPICS * PROMPT_POOL_STRATA
#: corpus of record (Addendum G1). Resolved from `read_composed_predictions` so the
#: campaign holds ONE literal; the brief records only the `5ae355bc…` prefix.
CORPUS_SHA_V21 = (
    "5ae355bc5d130f8e9c3ae426f5e71bf2b6e99c74b95369a874bec2abcd59b5d9")
#: ruling 3, verbatim — written into every stamp so no behavioral row carries an
#: unadjudicated §7 clause obligation.
ENVELOPE_RULING_OF_RECORD = (
    "envelope hygiene at behavioral grain (ruling 3, Luxia 2026-07-29): the §7 "
    "≥100-fresh-randoms envelope is satisfied at the COSINE/alignment read that "
    "accompanies every behavioral cell (100 seeded randoms + banked band members "
    "through the SAME map, q95 of |cos| — the read the clause was written for); the "
    "BEHAVIORAL band is 3 transported members per the banked precedent. Ruled "
    "explicitly and recorded: this is the clause's reading of record, NOT a weakened "
    "clause.")
#: §2.6's resolution of necessity — see the module docstring. Named so the desk can
#: rule differently without hunting for the assumption.
PROBE_GROUPING_READING = (
    "probe batches group rows of EQUAL prompt_length (ascending length, gen_id order "
    "within a length), because the injection hook gates on ABSOLUTE positions with "
    "one [seq_len] mask per forward (hooks.py L181) and a right-padded batch of "
    "mixed prompt lengths therefore has no single valid mask. Satisfies every "
    "literal §2.6 requirement (right padding, attention-masked, per-row spans by "
    "prompt_length, padding never in a mean) and adds only the grouping. DESK-OWED "
    "READING, recorded in every stamp; the named alternative (a per-row [batch, seq] "
    "mask in hooks.py) is a change to a frozen surface and was not taken.")
#: §11 / C§8: nothing behavioral is stamped.
GRADE_LINE = "UNSTAMPED (C§8)"
#: M10: a scheduler job carries a card index; a rogue carries the sentinel. The
#: field that caught the rogue run, so it is never allowed to be absent (§9 item 9).
CVD_UNSET_SENTINEL = "(unset)"

CellKind = Literal["baseline", "calibration", "calibration_band", "transported",
                   "transported_band", "naive", "bridge", "judged"]

#: The strata the §2.7 gate must cover, in order. Each contributes exactly one cell,
#: so K == 3 is not a tunable but the arity of this tuple.
REPLAY_GATE_STRATA: tuple[str, ...] = ("signal_at_0.3", "random_band", "calibration")


# ---------------------------------------------------------------- error taxonomy
class BehavioralHarnessError(RuntimeError):
    """Base class for every failure specific to the behavioral harness.

    Every subclass is one of the brief's §9 HALT conditions. They are exceptions
    rather than warnings on purpose: §3 and §9 both say an enactor never adjudicates
    a live HALT, and a warning is an invitation to adjudicate.
    """


class ArtifactShaMismatch(BehavioralHarnessError):
    """§9 item 1 (M4): a number-bearing artifact's sha is not the expected one."""


class CorpusVintageError(BehavioralHarnessError):
    """§9 item 2: the corpus is not v2.1, or the resolution chain is MIXED."""


class NativeVectorUnavailable(BehavioralHarnessError):
    """§9 item 3: the node's native entropy-gradient vector is absent/not FD-gated/FD-FAIL."""


class SiteNotOfRecord(BehavioralHarnessError):
    """§9 item 4: the injection site is not the registered site of record.

    A RETIRED site (gemma L36, 70B L17) must never resolve by default, so this is
    raised on a site that a bank happens to hold but the registries do not carry.
    """


class ReplayGateNotBitwise(BehavioralHarnessError):
    """§9 item 6 / §2.7: the in-job replay is not bitwise in the canonical layout."""


class NaiveTransplantRowMissing(BehavioralHarnessError):
    """§9 item 7 / §5.3: no banked naive row for a pair whose cell is being built."""


class ResidualNormDeltaError(BehavioralHarnessError):
    """§9 item 8 / §2.5: measured per-token median residual norm deviates >10%."""


class StampIncompleteError(BehavioralHarnessError):
    """§9 item 9 / §2.8: a stamp is missing a required field, or CVD is unset.

    Absence is a FIRST-CLASS reportable state (OWED, §10) — never silently "absent".
    """


class ExpectedNShortfall(BehavioralHarnessError):
    """§9 item 10 / §7: a cell or column is short of its expected N.

    M23 discipline: a count one short is a rake, not a rounding.
    """


class HookAdmissibilityError(BehavioralHarnessError):
    """§9 item 11 / §4.4: the SSM hook-admissibility preflight failed.

    The remedy — a positional-gating shim for that architecture — is a code change
    with its own gate, never something an enactor improvises mid-column.
    """


class PositionCeilingExceeded(BehavioralHarnessError):
    """§9 item 12: max(prompt_tokens) + max_new_tokens > the node's position ceiling.

    Addendum B's `truncation: first-1024` covers CORPUS texts, not behavioral
    prompts; the extension is a desk ruling, not an enactor call.
    """


class CanonicalLayoutInvalidated(BehavioralHarnessError):
    """§9 item 13: an OOM/VRAM event mid-column, or B changed after freeze.

    A mid-column B change would silently split the layout of record, so the node
    restarts from preflight instead.
    """


class LesionRecipeViolation(BehavioralHarnessError):
    """§5.4's lesion-recipe law, asserted in code.

    No object in this column is ever built by projecting out the direction known to
    produce the behavior. Re-asserted engine-side on every cell's vector provenance
    (the staging module asserts it at construction; this is the second reading, so a
    hand-staged cells-json cannot smuggle one past).
    """


class PromptPoolError(BehavioralHarnessError):
    """The behavioral prompt pool is not the banked stage-0 pool of record (ruling 10)."""


class SamplingConfigError(BehavioralHarnessError):
    """The sampling configuration is not resolvable to the config of record (ruling 4)."""


class BatchLayoutError(BehavioralHarnessError):
    """A batch layout was requested that is not on the frozen ladder (§2.2)."""


# ---------------------------------------------------------------- typed records
class SamplingConfig(BaseModel):
    """A generation sampling configuration, frozen (§2.2 / ruling 4).

    Ruling 4: **pure ancestral is the config of record**, identical across all 23
    nodes — the only cross-model-comparable choice, and the only one under which the
    entropy read stays interpretable. Exactly one exception exists: the bridge cell
    on the re-anchor pair, which runs under the NODE'S OWN `generation_config` so the
    discontinuity with the banked leg is measured rather than assumed. If the banked
    config cannot be recovered, the bridge cell is DROPPED and the discontinuity is
    NAMED (`BridgeCellDecision`), never papered over.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    do_sample: bool = True
    temperature: float = Field(gt=0.0, default=1.0)
    top_p: float = Field(gt=0.0, le=1.0, default=1.0)
    top_k: int = Field(ge=0, default=0)
    provenance: str

    @model_validator(mode="after")
    def _greedy_needs_no_uniforms(self) -> "SamplingConfig":
        if not self.do_sample and self.name != "greedy":
            raise ValueError(
                "do_sample=False is not the config of record and is only named "
                "'greedy'; ruling 4 froze pure ancestral sampling")
        return self


SAMPLING_OF_RECORD = SamplingConfig(
    name="pure-ancestral",
    do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
    provenance="ruling 4 (Luxia 2026-07-29): pure ancestral of record, identical "
               "across all 23 nodes; the only cross-model-comparable choice and the "
               "one that keeps the entropy read interpretable")


class BridgeCellDecision(BaseModel):
    """Ruling 4's bridge cell: run it, or DROP it and NAME the discontinuity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: str = Field(description="the re-anchor pair, e.g. '8b->qwen-7b'")
    banked_config_recoverable: bool
    dropped: bool
    sampling: Optional[SamplingConfig] = None
    #: Present exactly when `dropped`. Ruling 4: "the discontinuity is NAMED, not
    #: papered over" — so this is a sentence a reader can quote, not a flag.
    discontinuity_named: Optional[str] = None

    @model_validator(mode="after")
    def _drop_is_named(self) -> "BridgeCellDecision":
        if self.dropped and not self.discontinuity_named:
            raise ValueError(
                "ruling 4: a dropped bridge cell must NAME the discontinuity with "
                "the banked leg; an unnamed drop papers it over")
        if not self.dropped and self.sampling is None:
            raise ValueError("a bridge cell that runs needs its sampling config")
        return self


class CanonicalLayout(BaseModel):
    """§2.2's canonical batch layout — frozen in the node's stamp before any cell fires.

    A read produced at one layout is NEVER compared bitwise against another; the
    §2.7 replay gate is defined against this object, and §9 item 13 makes a
    mid-column change a restart-from-preflight rather than a silent split.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int
    padding_side_generation: Literal["left"] = "left"
    padding_side_probe: Literal["right"] = "right"
    dtype: str = Field(description="the node's collection regime, e.g. 'bfloat16'")
    attn_implementation: Literal["eager"] = "eager"
    max_new_tokens: int = Field(gt=0, default=MAX_NEW_TOKENS)
    use_cache_generation: bool = True
    use_cache_probe: bool = False
    probe_batch_size: int = PROBE_BATCH_SIZE
    sub_batch_rule: str = "sub_batch = gen_id // B; row = gen_id % B"
    reordering: str = ("none: no dynamic packing, no continuous batching, no "
                       "length-sorted reordering (§2.2)")
    probe_grouping_reading: str = PROBE_GROUPING_READING
    #: True once the layout is written to the node stamp. A cell that fires against
    #: an unfrozen layout is a §2.2 violation.
    frozen: bool = False
    #: M19: how B was chosen. `measured=False` means the VRAM probe was degraded and
    #: the CONSERVATIVE bound was taken (the smallest rung), never a gamble.
    measured: bool = True
    headroom_note: str = ""

    @field_validator("batch_size")
    @classmethod
    def _on_the_ladder(cls, v: int) -> int:
        if v not in BATCH_LADDER:
            raise ValueError(
                f"batch size {v} is not on the frozen ladder {BATCH_LADDER} (§2.2)")
        return v

    def sub_batch_of(self, gen_id: int) -> tuple[int, int]:
        """§2.2's deterministic, total sub-batch map: (sub_batch, row)."""
        if gen_id < 0:
            raise BatchLayoutError(f"gen_id must be non-negative, got {gen_id}")
        return divmod(gen_id, self.batch_size)


class BehavioralPrompt(BaseModel):
    """One prompt of the banked stage-0 pool (ruling 10)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str
    topic_idx: int = Field(ge=0)
    topic: str
    stratum: str
    seed: int = 0
    system_prompt: str = ""
    user_prompt: str


class PromptPool(BaseModel):
    """The behavioral prompt pool of record — 80 prompts, sha-frozen (ruling 10).

    Ruling 10: carry the banked stage-0 pool (20 topics × 4 strata × 1 seed = 80),
    rendered per-arm, sha-frozen AT HARNESS-FREEZE TIME, sha on every stamp. The
    re-anchor bridge row is only interpretable on the same pool and every prior
    behavioral leg used it, so building a fresh pool was refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Optional[str]
    sha256: str
    prompts: list[BehavioralPrompt]

    @model_validator(mode="after")
    def _pool_shape_of_record(self) -> "PromptPool":
        if len(self.prompts) != PROMPT_POOL_SIZE:
            raise ValueError(
                f"prompt pool has {len(self.prompts)} prompts, expected "
                f"{PROMPT_POOL_SIZE} ({PROMPT_POOL_TOPICS} topics × "
                f"{PROMPT_POOL_STRATA} strata × 1 seed, ruling 10)")
        ids = [p.prompt_id for p in self.prompts]
        if len(set(ids)) != len(ids):
            raise ValueError("prompt pool has duplicate prompt_ids")
        topics = {p.topic_idx for p in self.prompts}
        strata = {p.stratum for p in self.prompts}
        if len(topics) != PROMPT_POOL_TOPICS or len(strata) != PROMPT_POOL_STRATA:
            raise ValueError(
                f"prompt pool is not {PROMPT_POOL_TOPICS} topics × "
                f"{PROMPT_POOL_STRATA} strata: got {len(topics)} topics, "
                f"{len(strata)} strata {sorted(strata)}")
        return self

    def for_gen(self, gen_id: int) -> BehavioralPrompt:
        """§2.2: the map from gen_id to prompt is total and deterministic.

        n=80/cell over an 80-prompt pool, so at n=80 every cell covers the pool
        exactly once; the modulo keeps the map total for the reduced-n shape
        rehearsal (§3 step 2 runs n=8/cell) without changing which prompt any
        gen_id gets.
        """
        return self.prompts[gen_id % len(self.prompts)]


class NormConventions(BaseModel):
    """§2.5's TWO norm conventions, with provenance — only one of them sets α.

    The convention that sets α is the PER-TOKEN median residual norm at the target
    site, measured in-job. The banked mean-state median is a DIFFERENT number (3B
    L14: 12.2391 vs 12.1125) and rides beside, never used. Rake M21b's silent-
    fallback finding is why this resolution is explicit and never inherited.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    site: int
    used_for_alpha: str = "PER-TOKEN median residual norm, MEASURED in-job (§2.5)"
    measured_per_token_median: float = Field(gt=0.0)
    measured_provenance: str
    #: None on a node with no banked `a5_vectors_stamps.json` (every NEW node).
    banked_per_token_median: Optional[float] = None
    banked_provenance: Optional[str] = None
    #: The mean-STATE median convention, recorded beside and never used for α.
    banked_mean_state_median: Optional[float] = None
    banked_mean_state_provenance: Optional[str] = None
    #: Recorded, not absorbed (§2.5). >10% is a HALT, raised by `resolve_norms`.
    delta_fraction_vs_banked: Optional[float] = None


class CellSpec(BaseModel):
    """One behavioral cell: 80 generations sharing ONE injection spec (§2.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str
    kind: CellKind
    vector_key: Optional[str] = Field(
        default=None, description="npz key; None on the α=0 baseline cell")
    site: int = Field(ge=0)
    alpha_frac: float
    n: int = Field(gt=0, default=N_PER_CELL)
    #: `Rband*` = the node's OWN native random band (§4's control: does the SITE
    #: actuate). `gRband*` = the SOURCE's randoms through the SAME map (§5's
    #: control: does the TRANSPORT carry). §4.1 keeps them distinct in name and in
    #: stamp because conflating them is the fastest way to make a null
    #: uninterpretable.
    band_family: Optional[Literal["Rband", "gRband"]] = None
    vector_npz: Optional[str] = None
    vector_provenance: str = ""
    sampling: SamplingConfig = SAMPLING_OF_RECORD

    @property
    def is_null(self) -> bool:
        return self.band_family is not None

    @property
    def is_baseline(self) -> bool:
        return self.kind == "baseline"

    @model_validator(mode="after")
    def _consistent(self) -> "CellSpec":
        if self.is_baseline:
            if self.alpha_frac != BASELINE_DOSE or self.vector_key is not None:
                raise ValueError(
                    f"{self.cell_id}: the baseline cell is α=0 with no vector")
            return self
        if self.vector_key is None:
            raise ValueError(f"{self.cell_id}: a non-baseline cell needs a vector_key")
        if self.alpha_frac not in DOSE_LADDER:
            raise ValueError(
                f"{self.cell_id}: dose {self.alpha_frac} is not on the FROZEN ladder "
                f"{DOSE_LADDER} (§2.5/§11: no extension, no interpolation, no "
                "per-node tuning)")
        expected = CELL_ID_TEMPLATE.format(
            vector_key=self.vector_key, site=self.site, frac=self.alpha_frac)
        if self.cell_id != expected:
            raise ValueError(
                f"cell_id {self.cell_id!r} does not match the banked formatting "
                f"{expected!r} (§2.3; parsed by score_entropy_writes.parse_cell)")
        return self


class GenerationRecord(BaseModel):
    """One generation: its ids, its span, and the layout slot it rode in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str
    generation_id: int = Field(ge=0)
    prompt_id: str
    seed_material: str
    seed_int: int
    sub_batch: int
    row: int
    prompt_length: int = Field(gt=0, description="the row's own UNPADDED prompt length")
    padded_prompt_length: int = Field(
        gt=0, description="the sub-batch's left-padded prompt length; == the "
                          "injection spec's start_pos, one value for the whole batch")
    prompt_ids: list[int]
    generated_ids: list[int]
    finished_with_eos: bool
    n_uniforms_consumed: int

    @property
    def input_ids(self) -> list[int]:
        """The probe's `[prompt + generated]` sequence, unpadded."""
        return list(self.prompt_ids) + list(self.generated_ids)


class ProbeRow(BaseModel):
    """One generation's entropy probe result (§2.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation_id: int
    n_positions: int
    mean_entropy_steered: float
    mean_entropy_unsteered: float
    entropy_rise: float
    base_model_nll: float
    probe_group: int
    probe_group_size: int


class BatchInvarianceCharacterization(BaseModel):
    """§2.7's descriptive batch-invariance record — M19: it CANNOT fail a gate.

    GPU reductions are not batch-size-invariant; this is expected and is documented
    ONCE PER NODE so nobody later reads it as a defect. It is precisely why replay
    is defined in the canonical layout.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measured: bool
    reference_batch_size: int
    compared_batch_sizes: list[int]
    bitwise_identical: dict[str, bool]
    #: index of the first divergent token, per compared batch size; None = identical.
    first_divergent_token_index: dict[str, Optional[int]]
    note: str


class ReplayGateResult(BaseModel):
    """§2.7's blocking in-job gate: bitwise ids AND bitwise float32 entropy arrays."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int
    selection_digest: str
    selection_rule: str
    cells: list[str]
    strata: dict[str, str]
    token_id_sha256: dict[str, str]
    token_id_sha256_replay: dict[str, str]
    entropy_array_sha256: dict[str, str]
    entropy_array_sha256_replay: dict[str, str]
    passed: bool
    mismatches: list[str]


class HookAdmissibility(BaseModel):
    """§4.4's named technical preflight — HALT-gated, architecture-independent.

    The residual-write hook reads absolute positions from the `cache_position`
    kwarg. If a block does not receive it AS A KWARG the hook falls back to local-
    index slicing and RAISES BY DESIGN on incremental seq_len=1 steps rather than
    mis-injecting silently (`hooks.py` L193-199). This preflight is what turns that
    design into a budget decision instead of a mid-column surprise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    site: int
    decoder_layers_resolved: bool
    n_decoder_layers: int
    saw_cache_position: bool
    positions_injected: int
    positions_expected: int
    n_tokens_generated: int
    admissible: bool
    detail: str


# ---------------------------------------------------------------- seeding (§2.3)
def seed_material(*, corpus_sha: str, node_key: str, arm: str, site: int,
                  cell_id: str, gen_id: int) -> str:
    """§2.3's seed material, formatted from the frozen template.

    Built through `SEED_MATERIAL_TEMPLATE` rather than an f-string so the template
    whose digest rides every stamp is the SAME object the seeds are derived from —
    a stamp that names a recipe the code does not use is worse than no stamp.
    """
    return SEED_MATERIAL_TEMPLATE.format(
        corpus_sha=corpus_sha, node_key=node_key, arm=arm, site=site,
        cell_id=cell_id, gen_id=gen_id)


def seed_int(material: str) -> int:
    """§2.3: sha256 → the first 8 bytes, big-endian. NEVER `hash()` (M25)."""
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def draw_uniforms(seed: int, n: int) -> np.ndarray:
    """Ruling 8's PRE-DRAWN uniforms for one generation: float64 in [0, 1).

    Pre-drawing is the whole mechanism. Because the uniforms for (cell, gen_id) are
    materialized from that pair's own digest BEFORE any forward runs, the sampling
    step consumes a fixed tape: a partial re-run of any subset of cells keeps every
    other cell in phase (M5-safe by construction, not by discipline), and the tape
    is independent of the batch the generation rides in (the §2.2 layout-invariance
    property). Ruling 8 refused the one-stream-per-cell alternative for exactly the
    re-run case: a re-run that changes cell membership desyncs a stream.

    `n` is `max_new_tokens`: the tape is drawn to the full budget so an early EOS
    cannot shorten it and thereby couple one generation's length to another's tape.
    """
    if n <= 0:
        raise ValueError(f"uniform tape length must be positive, got {n}")
    return np.random.default_rng(seed).random(n)


def uniform_tape(*, corpus_sha: str, node_key: str, arm: str, site: int,
                 cell_id: str, gen_id: int, n: int = MAX_NEW_TOKENS
                 ) -> tuple[str, int, np.ndarray]:
    """(seed_material, seed_int, uniforms) for one generation — the whole recipe."""
    material = seed_material(corpus_sha=corpus_sha, node_key=node_key, arm=arm,
                             site=site, cell_id=cell_id, gen_id=gen_id)
    seed = seed_int(material)
    return material, seed, draw_uniforms(seed, n)


def cell_seed_root(*, corpus_sha: str, node_key: str, arm: str, site: int,
                   cell_id: str) -> str:
    """The per-cell seed ROOT that §2.8 requires in the stamp.

    A digest over the cell's material prefix (gen_id excluded): it identifies the
    cell's tape family without listing 80 seeds, and it changes if any coordinate of
    the recipe changes.
    """
    prefix = SEED_MATERIAL_TEMPLATE.rsplit("|", 1)[0].format(
        corpus_sha=corpus_sha, node_key=node_key, arm=arm, site=site,
        cell_id=cell_id)
    return hashlib.sha256(prefix.encode()).hexdigest()


# ------------------------------------------------------- the fixed sampling step
def sample_token(logits_row: Any, u: float, sampling: SamplingConfig = SAMPLING_OF_RECORD
                 ) -> int:
    """Ruling 8's FIXED sampling step: one row's logits + one pre-drawn uniform → a token.

    LAYOUT INVARIANCE LIVES HERE. The row's logits are cast to float64 numpy first,
    so every arithmetic operation of the sampling step — the temperature divide, the
    softmax, the cumulative sum, the search — runs on exactly one row's numbers in a
    fixed dtype and a fixed order. No reduction crosses rows, so identical logits
    yield a bitwise identical token at B=80 and at B=1. What is NOT invariant is the
    FORWARD that produced the logits (§2.7's characterization), which is why replay
    is defined in the canonical layout rather than across layouts.

    Inverse-CDF rather than `torch.multinomial`: multinomial consumes a global
    generator, which is the single-stream design ruling 8 refused (M5).
    """
    logits = np.asarray(logits_row, dtype=np.float64).reshape(-1)
    if logits.size == 0:
        raise ValueError("empty logits row")
    if not np.isfinite(logits).any():
        raise ValueError("logits row is entirely non-finite")
    if not sampling.do_sample:
        return int(np.argmax(logits))
    x = logits / sampling.temperature
    # top-k / top-p are 0 / 1.0 in the config of record (pure ancestral), so both
    # filters are exact no-ops there; they exist for the bridge cell, whose node
    # generation_config may carry them (ruling 4).
    if sampling.top_k:
        k = min(int(sampling.top_k), x.size)
        cut = np.partition(x, x.size - k)[x.size - k]
        x = np.where(x >= cut, x, -np.inf)
    x = x - np.max(x[np.isfinite(x)])
    p = np.exp(x)
    p[~np.isfinite(p)] = 0.0
    total = p.sum()
    if total <= 0.0 or not np.isfinite(total):
        return int(np.argmax(logits))
    p = p / total
    if sampling.top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        csum = np.cumsum(p[order])
        keep = order[:max(1, int(np.searchsorted(csum, sampling.top_p) + 1))]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0                    # guard the float32→float64 tail of the cumsum
    return int(np.searchsorted(cdf, float(u), side="right"))


# ---------------------------------------------------------------- the generation loop
class Stepper(Protocol):
    """The minimum surface the generation loop needs from a model.

    Deliberately narrow: the loop below is the load-bearing layout/sampling logic
    and it is the SAME code on the node and in the selftest. `HFStepper` implements
    this against a real HF model with a KV cache; the selftest's stub implements it
    with per-row-independent arithmetic, which is what makes bitwise layout
    invariance provable at all (a real forward's reductions are not batch-invariant,
    §2.7). One loop, two steppers — never two loops.
    """

    def prefill(self, ids: Any, attention_mask: Any) -> Any:
        """[batch, seq] ids → [batch, vocab] logits at the LAST position."""

    def step(self, next_ids: Any) -> Any:
        """[batch] just-sampled ids → [batch, vocab] logits at the new position."""

    def close(self) -> None:
        ...


def left_pad(sequences: Sequence[Sequence[int]], pad_id: int
             ) -> tuple[np.ndarray, np.ndarray, int]:
    """§2.2's left padding: (ids [b, L], attention_mask [b, L], L).

    Left padding is not a convenience — it is what makes ONE `start_pos` valid for a
    whole batch: the generated span starts at absolute position L for every row, and
    the hook gates on absolute `cache_position`.
    """
    if not sequences:
        raise ValueError("left_pad: no sequences")
    L = max(len(s) for s in sequences)
    ids = np.full((len(sequences), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(sequences), L), dtype=np.int64)
    for i, s in enumerate(sequences):
        if not s:
            raise ValueError(f"left_pad: sequence {i} is empty")
        ids[i, L - len(s):] = np.asarray(s, dtype=np.int64)
        mask[i, L - len(s):] = 1
    return ids, mask, L


def right_pad(sequences: Sequence[Sequence[int]], pad_id: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """§2.6's right padding for the probe: (ids [b, T], attention_mask [b, T])."""
    if not sequences:
        raise ValueError("right_pad: no sequences")
    T = max(len(s) for s in sequences)
    ids = np.full((len(sequences), T), pad_id, dtype=np.int64)
    mask = np.zeros((len(sequences), T), dtype=np.int64)
    for i, s in enumerate(sequences):
        ids[i, :len(s)] = np.asarray(s, dtype=np.int64)
        mask[i, :len(s)] = 1
    return ids, mask


def generate_sub_batch(
    stepper: Stepper,
    *,
    cell: CellSpec,
    gen_ids: Sequence[int],
    prompts: Sequence[BehavioralPrompt],
    prompt_ids: Sequence[Sequence[int]],
    tapes: Sequence[np.ndarray],
    layout: CanonicalLayout,
    pad_token_id: int,
    eos_token_id: Optional[int],
    corpus_sha: str,
    node_key: str,
    arm: str,
    start_pos_sink: Optional[Callable[[int], None]] = None,
) -> list[GenerationRecord]:
    """One sub-batch of a cell's generations, left-padded, uniform-tape-driven.

    `start_pos_sink` receives the sub-batch's padded prompt length L before the
    first forward, so the caller can set the live `ResidualWriteSpec.start_pos`
    (hooks.py reads it at every call, so one registered hook serves every
    sub-batch). Injection therefore covers generated positions ONLY, for every row.

    A row that emits EOS is FROZEN: it stops appending and stops consuming its tape,
    but it stays in the batch (fed the pad token) so no other row's slot moves. That
    is what keeps `sub_batch = gen_id // B; row = gen_id % B` total and deterministic
    (§2.2) — a compacting batch would make a generation's result depend on when its
    neighbours finished.
    """
    if not (len(gen_ids) == len(prompts) == len(prompt_ids) == len(tapes)):
        raise ValueError("generate_sub_batch: ragged inputs")
    ids, mask, L = left_pad(prompt_ids, pad_token_id)
    if start_pos_sink is not None:
        start_pos_sink(L)
    b = len(gen_ids)
    logits = stepper.prefill(ids, mask)
    out: list[list[int]] = [[] for _ in range(b)]
    consumed = [0] * b
    done = [False] * b
    for t in range(layout.max_new_tokens):
        next_ids = np.full((b,), pad_token_id, dtype=np.int64)
        for r in range(b):
            if done[r]:
                continue
            tape = tapes[r]
            if t >= tape.shape[0]:
                raise ValueError(
                    f"{cell.cell_id}/gen{gen_ids[r]}: uniform tape exhausted at "
                    f"step {t} (tape length {tape.shape[0]}, max_new_tokens "
                    f"{layout.max_new_tokens})")
            tok = sample_token(np.asarray(logits)[r], float(tape[t]), cell.sampling)
            consumed[r] += 1
            out[r].append(tok)
            next_ids[r] = tok
            if eos_token_id is not None and tok == eos_token_id:
                done[r] = True
        if all(done):
            break
        if t + 1 < layout.max_new_tokens:
            logits = stepper.step(next_ids)
    records = []
    for r, gid in enumerate(gen_ids):
        material, seed, _ = uniform_tape(
            corpus_sha=corpus_sha, node_key=node_key, arm=arm, site=cell.site,
            cell_id=cell.cell_id, gen_id=int(gid), n=1)
        records.append(GenerationRecord(
            cell_id=cell.cell_id, generation_id=int(gid),
            prompt_id=prompts[r].prompt_id, seed_material=material, seed_int=seed,
            sub_batch=layout.sub_batch_of(int(gid))[0],
            row=layout.sub_batch_of(int(gid))[1],
            prompt_length=len(prompt_ids[r]), padded_prompt_length=L,
            prompt_ids=list(prompt_ids[r]), generated_ids=out[r],
            finished_with_eos=done[r], n_uniforms_consumed=consumed[r]))
    return records


def generate_cell(
    stepper_factory: Callable[[], Stepper],
    *,
    cell: CellSpec,
    pool: PromptPool,
    tokenize: Callable[[BehavioralPrompt], list[int]],
    layout: CanonicalLayout,
    pad_token_id: int,
    eos_token_id: Optional[int],
    corpus_sha: str,
    node_key: str,
    arm: str,
    start_pos_sink: Optional[Callable[[int], None]] = None,
    n: Optional[int] = None,
) -> list[GenerationRecord]:
    """A whole cell, sub-batched by §2.2's deterministic total map.

    `stepper_factory` is called once per sub-batch (a KV cache belongs to one batch
    shape). The returned records are in gen_id order regardless of sub-batching, so
    a cell's output is layout-independent as an ORDERED object too, not only
    row-wise.
    """
    total = cell.n if n is None else n
    if total <= 0:
        raise ValueError(f"{cell.cell_id}: n must be positive")
    records: list[GenerationRecord] = []
    B = layout.batch_size
    for start in range(0, total, B):
        chunk = list(range(start, min(start + B, total)))
        prompts = [pool.for_gen(g) for g in chunk]
        prompt_ids = [tokenize(p) for p in prompts]
        tapes = [uniform_tape(corpus_sha=corpus_sha, node_key=node_key, arm=arm,
                              site=cell.site, cell_id=cell.cell_id, gen_id=g,
                              n=layout.max_new_tokens)[2] for g in chunk]
        stepper = stepper_factory()
        try:
            records += generate_sub_batch(
                stepper, cell=cell, gen_ids=chunk, prompts=prompts,
                prompt_ids=prompt_ids, tapes=tapes, layout=layout,
                pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                corpus_sha=corpus_sha, node_key=node_key, arm=arm,
                start_pos_sink=start_pos_sink)
        finally:
            stepper.close()
    if len(records) != total:
        raise ExpectedNShortfall(
            f"{cell.cell_id}: produced {len(records)} generations, expected {total} "
            "(§9 item 10 / M23: a count one short is a rake, not a rounding)")
    return records


# ---------------------------------------------------------------- dose application
def resolve_alpha(alpha_frac: float, per_token_median_resid_norm: float) -> float:
    """§2.5: α = frac × (PER-TOKEN median residual norm at the target site).

    NOT the median-of-mean-state norm the collection stamps carry — the two differ
    (banked example, 3B L14: 12.2391 vs 12.1125) and rake M21b's silent-fallback
    finding says the resolution must be explicit, never inherited.
    """
    if not np.isfinite(per_token_median_resid_norm) or per_token_median_resid_norm <= 0:
        raise ResidualNormDeltaError(
            f"per-token median residual norm must be finite and positive, got "
            f"{per_token_median_resid_norm!r} — refusing to resolve a dose against it")
    if alpha_frac != BASELINE_DOSE and alpha_frac not in DOSE_LADDER:
        raise ValueError(
            f"dose {alpha_frac} is not on the FROZEN ladder {DOSE_LADDER} (§2.5)")
    return float(alpha_frac) * float(per_token_median_resid_norm)


def apply_dose_ladder(vector_key: str, site: int, *,
                      per_token_median_resid_norm: float,
                      kind: CellKind,
                      band_family: Optional[str] = None,
                      vector_npz: Optional[str] = None,
                      vector_provenance: str = "",
                      sampling: SamplingConfig = SAMPLING_OF_RECORD,
                      n: int = N_PER_CELL) -> list[tuple[CellSpec, float]]:
    """The frozen ladder applied to one vector: 6 cells, exact α per cell (§2.5).

    Exactly `len(DOSE_LADDER)` cells, in ladder order, no baseline (the α=0 cell is
    SHARED between §4's calibration block and §5's column and so is built once, by
    `baseline_cell`, never per vector).
    """
    assert_no_lesion_recipe(vector_provenance, vector_key)
    out = []
    for frac in DOSE_LADDER:
        cell = CellSpec(
            cell_id=CELL_ID_TEMPLATE.format(vector_key=vector_key, site=site,
                                            frac=frac),
            kind=kind, vector_key=vector_key, site=site, alpha_frac=frac, n=n,
            band_family=band_family, vector_npz=vector_npz,
            vector_provenance=vector_provenance, sampling=sampling)
        out.append((cell, resolve_alpha(frac, per_token_median_resid_norm)))
    return out


def baseline_cell(site: int, n: int = N_PER_CELL) -> CellSpec:
    """The α=0 baseline, SHARED by §4's calibration block and §5's column (§5.1)."""
    return CellSpec(cell_id=f"baseline_L{site}_a+0.00", kind="baseline",
                    vector_key=None, site=site, alpha_frac=BASELINE_DOSE, n=n)


#: Substrings whose presence in a vector's construction provenance means the object
#: was built by projecting out the direction known to produce the behavior — the
#: lesion-recipe law's forbidden construction (§5.4). Matched case-insensitively on
#: the provenance string the staging module writes.
LESION_RECIPE_MARKERS: tuple[str, ...] = (
    "minus entropy_gradient", "- entropy_gradient", "orthogonalized against "
    "entropy_gradient", "project out entropy_gradient", "projected out "
    "entropy_gradient", "residualized on entropy_gradient",
    "entropy-orthogonalized")


def assert_no_lesion_recipe(provenance: str, vector_key: str = "") -> None:
    """§5.4's lesion-recipe law, asserted in code rather than trusted.

    No object in this column is ever built by projecting out the direction known to
    produce the behavior. The staging module asserts this at construction; this is
    the ENGINE-side second reading, so a hand-written cells-json cannot smuggle one
    in. Provenance is what is checked because it is the only thing that records the
    construction — a vector's numbers cannot say how they were made.
    """
    low = (provenance or "").lower()
    for marker in LESION_RECIPE_MARKERS:
        if marker in low:
            raise LesionRecipeViolation(
                f"{vector_key or '(vector)'}: provenance names a lesion recipe "
                f"({marker!r}) — §5.4's lesion-recipe law forbids an object built by "
                f"projecting out the direction known to produce the behavior. "
                f"Provenance: {provenance!r}")


def resolve_norms(*, site: int, measured: float, measured_provenance: str,
                  banked_per_token: Optional[float] = None,
                  banked_provenance: Optional[str] = None,
                  banked_mean_state: Optional[float] = None,
                  banked_mean_state_provenance: Optional[str] = None
                  ) -> NormConventions:
    """§2.5's both-conventions record, with the >10% HALT (§9 item 8).

    For the six carried nodes with banked a5 stamps the measured value is asserted
    against the banked one and the delta is RECORDED RATHER THAN ABSORBED; >10%
    would mean the site or the pool is not what the stamp says, which is a HALT and
    not a correction an enactor applies.
    """
    delta = None
    if banked_per_token is not None:
        if banked_per_token <= 0 or not np.isfinite(banked_per_token):
            raise ArtifactShaMismatch(
                f"L{site}: banked per-token median residual norm is "
                f"{banked_per_token!r} — a non-positive norm is a corrupt stamp, not "
                "a small delta")
        delta = float(abs(measured - banked_per_token) / banked_per_token)
        if delta > NORM_DELTA_HALT_FRACTION:
            raise ResidualNormDeltaError(
                f"L{site}: measured per-token median residual norm {measured:.4f} "
                f"deviates {delta * 100:.1f}% from the banked a5 value "
                f"{banked_per_token:.4f} (>{NORM_DELTA_HALT_FRACTION * 100:.0f}%, §9 "
                f"item 8). That means the site or the prompt pool is not what the "
                f"stamp says — HALT to the desk; an enactor does not absorb this.")
        logger.info("L%d norm conventions: measured %.4f vs banked %.4f (Δ %.2f%%) — "
                    "recorded, not absorbed", site, measured, banked_per_token,
                    delta * 100)
    return NormConventions(
        site=site, measured_per_token_median=float(measured),
        measured_provenance=measured_provenance,
        banked_per_token_median=banked_per_token,
        banked_provenance=banked_provenance,
        banked_mean_state_median=banked_mean_state,
        banked_mean_state_provenance=banked_mean_state_provenance,
        delta_fraction_vs_banked=delta)


# ---------------------------------------------------------------- entropy probe (§2.6)
def probe_groups(records: Sequence[GenerationRecord],
                 max_group: int = PROBE_BATCH_SIZE) -> list[list[int]]:
    """§2.6's probe batching under `PROBE_GROUPING_READING`.

    Rows of EQUAL `prompt_length` group together (ascending length; gen_id order
    within a length), chunked to `max_group`. Equal prompt length is what makes ONE
    absolute-position injection mask valid for the whole right-padded group, which
    is the constraint that forces the grouping at all.

    Returns groups of INDICES into `records` (not gen_ids), so a caller never has to
    re-resolve them.
    """
    if max_group <= 0:
        raise ValueError(f"max_group must be positive, got {max_group}")
    by_len: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        by_len.setdefault(r.prompt_length, []).append(i)
    groups: list[list[int]] = []
    for plen in sorted(by_len):
        idxs = sorted(by_len[plen], key=lambda i: records[i].generation_id)
        for s in range(0, len(idxs), max_group):
            groups.append(idxs[s:s + max_group])
    return groups


def per_position_entropy_and_nll(logits: Any, ids: Any, prompt_length: int,
                                 n_generated: int) -> tuple[np.ndarray, np.ndarray]:
    """§2.6's per-position next-token entropy + NLL over ONE row's generated span.

    The same algebra as `entropy_write_probe._ent_nll_over_gen` (one source of
    truth for the convention): entropy = state of the distribution; NLL =
    -log p(actual generated token), the likelihood rung filed beside.

    `logits` is that row's [T, vocab] float32; the slice is
    `[P-1, P+n_generated-1)` — the distributions that PRODUCED the generated
    tokens. Padding cannot enter, because the slice is bounded by this row's own
    `prompt_length` and its own generated count (§2.6: padding must never enter a
    mean).
    """
    import torch

    if n_generated < 1:
        raise ValueError("no generated positions to score")
    lg = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    lg = lg.float()
    lp = torch.log_softmax(lg, dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)
    pos = torch.arange(prompt_length - 1, prompt_length - 1 + n_generated)
    tgt = torch.as_tensor(ids, dtype=torch.long)[
        prompt_length:prompt_length + n_generated]
    nll = -lp[pos, tgt]
    return (ent[pos].detach().cpu().numpy().astype(np.float32),
            nll.detach().cpu().numpy().astype(np.float32))


def entropy_rise_from_rows(rows: Sequence[ProbeRow]) -> dict[str, float]:
    """§2.6's cell-level read: mean over generations, 4-dp filing convention.

    Aggregation order matches `entropy_write_probe` exactly (per-generation mean,
    then mean over generations), so a cross-harness comparison is a comparison of
    the same statistic.
    """
    if not rows:
        return {"n": 0}
    s = float(np.mean([r.mean_entropy_steered for r in rows]))
    u = float(np.mean([r.mean_entropy_unsteered for r in rows]))
    return {"n": len(rows),
            "mean_entropy_steered": round(s, 4),
            "mean_entropy_unsteered": round(u, 4),
            "entropy_rise": round(s - u, 4),
            "base_model_nll": round(
                float(np.mean([r.base_model_nll for r in rows])), 4)}


def entropy_array_digest(arrays: Sequence[np.ndarray]) -> str:
    """§2.7: sha256 over the probe's per-position float32 arrays, `.tobytes()`.

    float32 is asserted rather than cast: a silent float64 promotion would change
    the digest for a reason that has nothing to do with the model, and the gate must
    only ever fire on the model.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.asarray(a)
        if arr.dtype != np.float32:
            raise ReplayGateNotBitwise(
                f"entropy array has dtype {arr.dtype}, expected float32 — the §2.7 "
                "digest is defined over float32 .tobytes() and a promoted array "
                "would digest differently for a non-model reason")
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def token_id_digest(records: Sequence[GenerationRecord]) -> str:
    """§2.7: sha256 over the generated token-id arrays, in gen_id order."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.generation_id):
        h.update(np.asarray(r.generated_ids, dtype=np.int64).tobytes())
    return h.hexdigest()


def characterize_probe_batch_invariance(
    reference: Sequence[np.ndarray],
    comparisons: dict[int, Sequence[np.ndarray]],
    reference_batch_size: int,
    measured: bool = True,
) -> BatchInvarianceCharacterization:
    """§2.7's DESCRIPTIVE batch-invariance record. M19: it cannot fail a gate.

    §2.6 says the job "asserts" bit-exactness of a single-row probe against the
    batched row AND that the delta is instrumentation that "cannot fail the gate".
    Those cannot both hold, and M19 decides it: this function returns a record.
    Only §2.7's replay gate — same process, same canonical layout — can HALT.
    """
    bitwise: dict[str, bool] = {}
    first_div: dict[str, Optional[int]] = {}
    for b, arrays in sorted(comparisons.items()):
        key = str(b)
        if len(arrays) != len(reference):
            bitwise[key] = False
            first_div[key] = 0
            continue
        same = True
        idx: Optional[int] = None
        for ref, alt in zip(reference, arrays):
            ra, aa = np.asarray(ref), np.asarray(alt)
            if ra.shape != aa.shape:
                same, idx = False, 0
                break
            diff = np.nonzero(ra != aa)[0]
            if diff.size:
                same = False
                idx = int(diff[0]) if idx is None else min(idx, int(diff[0]))
        bitwise[key] = same
        first_div[key] = None if same else (idx if idx is not None else 0)
    return BatchInvarianceCharacterization(
        measured=measured, reference_batch_size=reference_batch_size,
        compared_batch_sizes=sorted(comparisons), bitwise_identical=bitwise,
        first_divergent_token_index=first_div,
        note="GPU reductions are not batch-size-invariant; a divergence here is "
             "EXPECTED and is documented once per node so nobody later reads it as "
             "a defect. It is precisely why §2.7 defines replay IN the canonical "
             "layout. M19: descriptive only — this record cannot fail a gate.")


# ---------------------------------------------------------------- replay gate (§2.7)
def _stratum_of(cell: CellSpec) -> Optional[str]:
    if cell.kind in ("calibration", "calibration_band"):
        return "calibration" if cell.kind == "calibration" else None
    if cell.band_family == "gRband":
        return "random_band"
    if cell.kind in ("transported", "bridge") and abs(cell.alpha_frac) == 0.3:
        return "signal_at_0.3"
    return None


def select_replay_cells(cells: Sequence[CellSpec], node_key: str, corpus_sha: str
                        ) -> tuple[list[str], dict[str, str], str]:
    """§2.7's deterministic gate-cell selection.

    "selection deterministic from `sha256(node_key|corpus_sha)`, constrained to
    include one signal cell at |0.3|, one random-band cell, and one calibration
    cell" — so K == 3 is the arity of `REPLAY_GATE_STRATA`, and each stratum
    contributes exactly one cell chosen by the digest modulo the stratum's size.
    Cells are sorted by cell_id first, so the choice depends on the cell SET and the
    digest, never on the order a cells-json happened to list them in.
    """
    digest = hashlib.sha256(f"{node_key}|{corpus_sha}".encode()).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    buckets: dict[str, list[str]] = {s: [] for s in REPLAY_GATE_STRATA}
    for c in cells:
        s = _stratum_of(c)
        if s is not None:
            buckets[s].append(c.cell_id)
    chosen: dict[str, str] = {}
    missing = []
    for i, stratum in enumerate(REPLAY_GATE_STRATA):
        options = sorted(buckets[stratum])
        if not options:
            missing.append(stratum)
            continue
        # a distinct 8-byte window per stratum, so two strata of equal size do not
        # index in lockstep
        window = int.from_bytes(
            hashlib.sha256(f"{digest}|{stratum}".encode()).digest()[:8], "big")
        chosen[stratum] = options[(seed ^ window) % len(options)]
    if missing:
        raise ExpectedNShortfall(
            f"§2.7 replay gate cannot be constituted on {node_key}: no cell for "
            f"stratum(a) {missing}. The gate is BLOCKING and its composition is "
            f"frozen ({list(REPLAY_GATE_STRATA)}), so a column that cannot supply "
            f"one of each is incomplete, not exempt.")
    return [chosen[s] for s in REPLAY_GATE_STRATA], chosen, digest


def evaluate_replay_gate(*, selection: list[str], strata: dict[str, str],
                         digest: str,
                         token_first: dict[str, str], token_replay: dict[str, str],
                         entropy_first: dict[str, str],
                         entropy_replay: dict[str, str]) -> ReplayGateResult:
    """§2.7's blocking comparison. Any mismatch → HALT; no cell is quotable."""
    mismatches = []
    for cell in selection:
        for label, a, b in (("token_ids", token_first, token_replay),
                            ("entropy_arrays", entropy_first, entropy_replay)):
            first, replay = a.get(cell), b.get(cell)
            if first is None or replay is None:
                mismatches.append(f"{cell}: {label} digest MISSING "
                                  f"(first={first!r}, replay={replay!r})")
            elif first != replay:
                mismatches.append(f"{cell}: {label} {first[:12]}… != {replay[:12]}…")
    result = ReplayGateResult(
        k=REPLAY_GATE_K, selection_digest=digest,
        selection_rule="sha256(node_key|corpus_sha); one cell per frozen stratum "
                       f"{list(REPLAY_GATE_STRATA)}, chosen by the digest modulo "
                       "the stratum's cell_id-sorted size (§2.7)",
        cells=selection, strata=strata,
        token_id_sha256=dict(token_first), token_id_sha256_replay=dict(token_replay),
        entropy_array_sha256=dict(entropy_first),
        entropy_array_sha256_replay=dict(entropy_replay),
        passed=not mismatches, mismatches=mismatches)
    if mismatches:
        raise ReplayGateNotBitwise(
            "§2.7 in-job replay gate FAILED in the canonical layout — no cell from "
            f"this node is quotable (§9 item 6). Mismatches: {mismatches}")
    return result


# ---------------------------------------------------------------- preflight
def choose_batch_size(vram_probe: Optional[Callable[[int], float]] = None,
                      ladder: Sequence[int] = BATCH_LADDER,
                      headroom: float = VRAM_HEADROOM_FRACTION
                      ) -> tuple[int, bool, str]:
    """§2.2's preflight: the largest ladder rung that fits with ≥15% headroom.

    `vram_probe(B)` returns the FRACTION of device memory a measured probe at max
    sequence length uses at batch size B. M19: every instrumentation call is
    wrapped, and a degraded probe does not gamble — it takes the CONSERVATIVE
    bound (the smallest rung) and says `measured=False`, so the consumer blocks on
    the bound rather than on a hope.
    """
    if vram_probe is None:
        return (min(ladder), False,
                "no VRAM probe supplied; M19 conservative bound = the smallest "
                f"ladder rung ({min(ladder)}), measured=false")
    for B in sorted(ladder, reverse=True):
        try:
            used = float(vram_probe(int(B)))
        except Exception as exc:                       # noqa: BLE001 — M19(a)
            logger.warning("VRAM probe at B=%d failed (%s: %s) — M19: taking the "
                           "conservative bound", B, type(exc).__name__, exc)
            return (min(ladder), False,
                    f"probe raised {type(exc).__name__} at B={B}; conservative "
                    f"bound = {min(ladder)}, measured=false")
        if not np.isfinite(used):
            return (min(ladder), False,
                    f"probe returned {used!r} at B={B}; conservative bound = "
                    f"{min(ladder)}, measured=false")
        if used <= 1.0 - headroom:
            return (int(B), True,
                    f"measured {used * 100:.1f}% of device memory at B={B} "
                    f"(≥{headroom * 100:.0f}% headroom satisfied)")
    return (min(ladder), True,
            f"no rung left ≥{headroom * 100:.0f}% headroom; smallest rung "
            f"{min(ladder)} taken and the node's fit is the binding constraint")


def freeze_layout(batch_size: int, *, dtype: str, measured: bool = True,
                  headroom_note: str = "",
                  max_new_tokens: int = MAX_NEW_TOKENS) -> CanonicalLayout:
    """§2.2: B is FROZEN in the node's stamp before any cell fires."""
    return CanonicalLayout(batch_size=batch_size, dtype=dtype,
                           max_new_tokens=max_new_tokens, frozen=True,
                           measured=measured, headroom_note=headroom_note)


def assert_position_budget(max_prompt_tokens: int, max_new_tokens: int,
                           ceiling: Optional[int], node_key: str = "") -> None:
    """§9 item 12: a node with a learned-position ceiling must fit prompt + span.

    Addendum B's `truncation: first-1024` covers CORPUS texts, not behavioral
    prompts, so an over-budget behavioral pool is a DESK RULING, not an enactor's
    truncation.
    """
    if ceiling is None:
        return
    need = int(max_prompt_tokens) + int(max_new_tokens)
    if need > int(ceiling):
        raise PositionCeilingExceeded(
            f"{node_key or 'node'}: max(prompt_tokens)={max_prompt_tokens} + "
            f"max_new_tokens={max_new_tokens} = {need} > position ceiling "
            f"{ceiling} (§9 item 12). Addendum B's first-{ceiling} truncation "
            f"covers corpus texts, NOT behavioral prompts — the extension is a "
            f"desk ruling, not an enactor call.")


def assert_alpha_zero_is_no_hook(logits_hooked_alpha0: Any,
                                 logits_unhooked: Any) -> None:
    """§2.4's cheap once-per-node correctness check.

    The hook short-circuits on `alpha == 0.0` and returns its args untouched, so the
    α=0 baseline WITH the hook attached must be bitwise identical to no hook. If it
    is not, the injection machinery is not what the stamp says it is, and every cell
    on the node is suspect.
    """
    a = np.asarray(logits_hooked_alpha0)
    b = np.asarray(logits_unhooked)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ReplayGateNotBitwise(
            "§2.4: α=0 WITH the hook attached is not bitwise identical to the "
            "unhooked forward. The hook's alpha==0.0 short-circuit is the whole "
            "reason the baseline cell is shared between §4 and §5 — HALT.")


def ssm_hook_admissibility_preflight(model: Any, *, node_key: str, site: int,
                                     vector: Any, n_tokens: int = 8,
                                     prompt_ids: Optional[Sequence[int]] = None
                                     ) -> HookAdmissibility:
    """§4.4's named technical preflight — HALT-gated, run BEFORE any SSM cell is budgeted.

    Resolve `decoder_layers` on the LOADED model, attach a spec, generate `n_tokens`
    tokens, and assert `stats["saw_cache_position"] is True` and that
    `stats["positions"]` equals expectation. On failure the SSM tier HALTs to the
    desk: the remedy (a positional-gating shim for that architecture) is a code
    change with its own gate, never something an enactor improvises mid-column.

    Not SSM-specific in the code — it is a fact about an ARCHITECTURE's kwarg
    plumbing, so it is safe (and cheap) to run on any node whose blocks are not
    already known to pass.
    """
    import torch

    from metabasis.extraction.hooks import (ResidualWriteSpec,
                                            attach_residual_write, decoder_layers)

    try:
        layers = decoder_layers(model)
    except AttributeError as exc:
        raise HookAdmissibilityError(
            f"{node_key}: decoder_layers() cannot resolve the decoder stack on "
            f"{type(model).__name__} ({exc}) — the site cannot be hooked at all. "
            "HALT to the desk (§4.4): extending decoder_layers for an architecture "
            "is a code change with its own gate.") from exc
    n_layers = len(layers)
    if not 0 <= site < n_layers:
        raise SiteNotOfRecord(
            f"{node_key}: site L{site} is not a decoder-layer index of the loaded "
            f"model ({n_layers} layers)")
    ids = torch.as_tensor([list(prompt_ids) if prompt_ids else [1, 2, 3, 4]],
                          dtype=torch.long)
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32)).to(dev)
    handle = attach_residual_write(model, ResidualWriteSpec(
        layer_idx=site, vector=vec, alpha=1.0, start_pos=int(ids.shape[1]),
        end_pos=None, normalize=True))
    detail = ""
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=int(n_tokens), do_sample=False,
                                 use_cache=True)
        stats = dict(handle.stats)
        n_new = int(out.shape[1] - ids.shape[1])
    except RuntimeError as exc:
        # hooks.py L193-199 raises BY DESIGN on an ambiguous incremental step; that
        # is the admissibility failure this preflight exists to surface early.
        raise HookAdmissibilityError(
            f"{node_key}: the residual-write hook refused an incremental step at "
            f"L{site} ({exc}). hooks.py raises rather than mis-injecting when a "
            "block does not receive `cache_position` as a kwarg. HALT to the desk "
            "(§4.4) — the remedy is a positional-gating shim with its own gate.") from exc
    finally:
        handle.remove()
    saw = bool(stats.get("saw_cache_position", False))
    positions = int(stats.get("positions", 0))
    # THE EXPECTATION, derived rather than guessed. `start_pos = prompt_length`, so
    # the PREFILL forward (positions [0, P)) injects nothing — the first generated
    # token is computed from the unperturbed prompt by construction. Each of the
    # (n_new - 1) incremental seq_len=1 forwards then carries exactly one position
    # >= P. So the last generated token is never itself fed back and never injected,
    # which is what "injecting over generated positions only" means with a KV cache.
    positions_expected = max(n_new - 1, 0)
    admissible = saw and positions == positions_expected
    detail = (f"saw_cache_position={saw}, positions={positions}, "
              f"expected={positions_expected} (prefill injects nothing; one injected "
              f"position per incremental step over {n_new} new tokens)")
    record = HookAdmissibility(
        node_key=node_key, site=site, decoder_layers_resolved=True,
        n_decoder_layers=n_layers, saw_cache_position=saw,
        positions_injected=positions, positions_expected=positions_expected,
        n_tokens_generated=n_new, admissible=admissible, detail=detail)
    if not admissible:
        raise HookAdmissibilityError(
            f"{node_key}: SSM hook-admissibility preflight FAILED at L{site} — "
            f"{detail}. §4.4/§9 item 11: the tier HALTs to the desk; a positional-"
            "gating shim for this architecture is a code change with its own gate, "
            "not an enactor improvisation mid-column.")
    logger.info("hook-admissibility preflight PASSED on %s L%d: %s",
                node_key, site, detail)
    return record


# ---------------------------------------------------------------- corpus / pool
def assert_corpus_vintage(*shas: Optional[str],
                          expected: str = CORPUS_SHA_V21) -> str:
    """§9 item 2: v2.1 everywhere in a cell's resolution chain, and never MIXED.

    Every argument is one link of the chain (corpus manifest, vector build stamp,
    state bank, map fit). A None is a HOLE, and a hole is reported as a hole — it is
    not evidence of agreement.
    """
    present = [s for s in shas if s]
    if not present:
        raise CorpusVintageError(
            "no corpus sha resolved anywhere in the chain — a cell cannot fire "
            "against an unknown corpus vintage (§9 item 2)")
    if len(shas) != len(present):
        raise CorpusVintageError(
            f"corpus vintage chain has {len(shas) - len(present)} UNRESOLVED "
            f"link(s) of {len(shas)}; an absent link is a hole, not an agreement "
            "(§9 item 2)")
    distinct = sorted(set(present))
    if len(distinct) > 1:
        raise CorpusVintageError(
            f"MIXED corpus vintage across the resolution chain: {distinct} "
            "(§9 item 2)")
    if distinct[0] != expected:
        raise CorpusVintageError(
            f"corpus vintage {distinct[0][:12]}… is not v2.1 ({expected[:12]}…) "
            "(§9 item 2)")
    return distinct[0]


def load_prompt_pool(path: Path) -> PromptPool:
    """Ruling 10's banked stage-0 pool, sha-frozen, from disk.

    Tolerant on KEY NAMES (banked pools predate this module and carry the corpus
    builder's own field names) and INTOLERANT on SHAPE: an unrecognized record
    raises rather than degrading to a partial pool, because a silently short pool
    would change which prompt every gen_id gets and thereby every cell's tokens.
    """
    if not path.exists():
        raise PromptPoolError(
            f"behavioral prompt pool not found: {path} (ruling 10: the banked "
            "stage-0 pool is the pool of record and its sha rides every stamp)")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptPoolError(f"{path}: not JSON ({exc})") from exc
    entries = doc["prompts"] if isinstance(doc, dict) and "prompts" in doc else doc
    if isinstance(doc, dict) and "entries" in doc:
        entries = doc["entries"]
    if not isinstance(entries, list):
        raise PromptPoolError(
            f"{path}: expected a list of prompts (or a dict with 'prompts'/"
            f"'entries'), got {type(entries).__name__}")
    prompts = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise PromptPoolError(f"{path}[{i}]: prompt record is not an object")
        try:
            prompts.append(BehavioralPrompt(
                prompt_id=str(e.get("prompt_id") or e.get("text_id") or f"P{i:03d}"),
                topic_idx=int(e["topic_idx"]),
                topic=str(e.get("topic", "")),
                stratum=str(e.get("stratum") or e.get("mode")),
                seed=int(e.get("seed", 0)),
                system_prompt=str(e.get("system_prompt", "")),
                user_prompt=str(e.get("user_prompt") or e["prompt"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptPoolError(
                f"{path}[{i}]: unrecognized prompt record ({type(exc).__name__}: "
                f"{exc}). Refusing a partial pool — a short pool changes which "
                "prompt every gen_id gets, and therefore every cell's tokens.") from exc
    try:
        return PromptPool(path=str(path), sha256=sha, prompts=prompts)
    except ValueError as exc:
        raise PromptPoolError(f"{path}: {exc}") from exc


def render_prompt(prompt: BehavioralPrompt, tok: Any, arm: str,
                  date_string: str = "12 Jul 2026") -> list[int]:
    """Ruling 10's "rendered per-arm": native = chat template, raw = plain.

    Mirrors `collect_mean_states.build_ids`'s arm semantics (the same
    `date_string` pin, the same TypeError fall-back for templates that take no
    `date_string`) so a behavioral prompt and a collected text are tokenized under
    the same arm rules. §5.1's frame is the BARE SYSTEM PROMPT — the standard
    entropy frame, frozen — so nothing is added here that the pool does not carry.
    """
    if arm not in ("native", "raw"):
        raise ValueError(f"unknown arm {arm!r} (valid: native, raw)")
    if arm == "raw":
        text = (f"{prompt.system_prompt}\n{prompt.user_prompt}"
                if prompt.system_prompt else prompt.user_prompt)
        return list(tok.encode(text, add_special_tokens=True))
    messages = []
    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})
    messages.append({"role": "user", "content": prompt.user_prompt})
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      date_string=date_string)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


# ---------------------------------------------------------------- stamps (§2.8)
#: §2.8's checklist, as data. Every field is non-negotiable per node, per cell; the
#: checker asserts presence AND non-nullness, because §10 makes absence a
#: first-class reportable state (OWED) rather than a silent "absent".
STAMP_REQUIRED_FIELDS: tuple[str, ...] = (
    "corpus_manifest_sha256",
    "behavioral_prompt_pool_sha256",
    "site",
    "site_cross_check",
    "arm",
    "dose_ladder",
    "alpha_value",
    "norm_conventions",
    "seed_recipe",
    "seed_material_template_digest",
    "per_cell_seed_roots",
    "cuda_visible_devices",
    "scheduler_card_index",
    "canonical_batch_layout",
    "model_config_sha256",
    "vector_npz_sha256",
    "vector_fd_gate",
    "vector_build_stamp",
    "transport_map_fit_sha256",
    "transport_map_family",
    "transport_map_arm",
    "transport_map_corpus_vintage",
    "naive_transplant_gate_row",
    "naive_transplant_verdict",
    "trunk",
    "replay_gate_digests",
    "battery_item_set_sha256",
    "actuation_calibration",
    "envelope_ruling",
    "sampling_config",
    "grade",
)

#: Fields that are legitimately null on a CALIBRATION cell — a native lever at the
#: node's own site rides no transport map and has no naive-transplant row, because
#: there is no (source, target) pair. Named explicitly so "null" is a RULED state
#: for these and only these, and the checker still refuses every other null.
CALIBRATION_NULLABLE_FIELDS: frozenset[str] = frozenset({
    "transport_map_fit_sha256", "transport_map_family", "transport_map_arm",
    "transport_map_corpus_vintage", "naive_transplant_gate_row",
    "naive_transplant_verdict",
})


def assert_stamp_complete(stamp: dict, *, cell_kind: Optional[CellKind] = None
                          ) -> None:
    """§2.8 as a checklist assertion; §9 item 9 as its HALT.

    Two ways to fail, kept apart in the message because they demand different
    responses: a MISSING/NULL field is an OWED artifact (§10), and an unset CVD is
    the M10 rogue-run tell (a scheduler job carries a card index; a rogue carries
    the sentinel).
    """
    nullable = (CALIBRATION_NULLABLE_FIELDS
                if cell_kind in ("calibration", "calibration_band", "baseline")
                else frozenset())
    missing = [f for f in STAMP_REQUIRED_FIELDS if f not in stamp]
    null = [f for f in STAMP_REQUIRED_FIELDS
            if f in stamp and stamp[f] in (None, "", [], {})
            and f not in nullable]
    if missing or null:
        raise StampIncompleteError(
            f"§2.8 stamp incomplete (§9 item 9): missing={missing} null={null}. "
            "Absence is a first-class reportable state (OWED, §10), never a silent "
            "'absent' — the desk's stamp-completeness checker reads exactly this.")
    cvd = stamp.get("cuda_visible_devices")
    if cvd == CVD_UNSET_SENTINEL or not cvd:
        raise StampIncompleteError(
            f"§9 item 9: cuda_visible_devices is {cvd!r}. M10: a SCHEDULER job "
            "carries a card index and a ROGUE carries the sentinel — this is the "
            "field that caught the rogue run, so it can never be absent.")


def build_stamp(*, cell: CellSpec, alpha: float, layout: CanonicalLayout,
                pool: PromptPool, norms: NormConventions, corpus_sha: str,
                node_key: str, arm: str, site_cross_check: dict,
                model_config_sha256: Optional[str], vector_npz_sha256: Optional[str],
                vector_fd_gate: Any, vector_build_stamp: Any,
                transport_map: Optional[dict], naive_row: Optional[dict],
                trunk: dict, replay_gate_digests: Any,
                battery_item_set_sha256: str, actuation_calibration: Any,
                per_cell_seed_roots: dict[str, str],
                scheduler_card_index: Optional[str] = None,
                extra: Optional[dict] = None) -> dict:
    """§2.8's per-cell custody stamp, built so the checklist cannot be half-met.

    Every §2.8 field is written here explicitly — nothing is `.get`-defaulted into
    existence — and `assert_stamp_complete` runs before the stamp is returned, so a
    caller can never receive a stamp that would fail the desk's own checker.
    """
    tmap = transport_map or {}
    stamp = {
        "grade": GRADE_LINE,
        "brief_of_record": BRIEF_OF_RECORD,
        "brief_sha256": BRIEF_SHA256,
        "builder": "run_behavioral_cells.py",
        "cell_id": cell.cell_id,
        "cell_kind": cell.kind,
        "vector_key": cell.vector_key,
        "band_family": cell.band_family,
        "is_null": cell.is_null,
        "n_expected": cell.n,
        "corpus_manifest_sha256": corpus_sha,
        "behavioral_prompt_pool_sha256": pool.sha256,
        "behavioral_prompt_pool_path": pool.path,
        "site": cell.site,
        "site_cross_check": site_cross_check,
        "arm": arm,
        "node_key": node_key,
        "dose_ladder": list(DOSE_LADDER),
        "alpha_frac": cell.alpha_frac,
        "alpha_value": alpha,
        "norm_conventions": norms.model_dump(),
        "seed_recipe": SEED_MATERIAL_TEMPLATE,
        "seed_material_template_digest": SEED_MATERIAL_TEMPLATE_DIGEST,
        "per_cell_seed_roots": per_cell_seed_roots,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",
                                               CVD_UNSET_SENTINEL),
        "scheduler_card_index": (
            scheduler_card_index
            or os.environ.get("HEIMDALL_CARD_INDEX")
            or os.environ.get("CUDA_VISIBLE_DEVICES", CVD_UNSET_SENTINEL)),
        "canonical_batch_layout": layout.model_dump(),
        "model_config_sha256": model_config_sha256,
        "vector_npz_sha256": vector_npz_sha256,
        "vector_fd_gate": vector_fd_gate,
        "vector_build_stamp": vector_build_stamp,
        "vector_provenance": cell.vector_provenance,
        "transport_map_fit_sha256": tmap.get("fit_sha256"),
        "transport_map_family": tmap.get("family"),
        "transport_map_arm": tmap.get("arm"),
        "transport_map_corpus_vintage": tmap.get("corpus_vintage"),
        "naive_transplant_gate_row": naive_row,
        "naive_transplant_verdict": (naive_row or {}).get("verdict"),
        "trunk": trunk,
        "replay_gate_digests": replay_gate_digests,
        "battery_item_set_sha256": battery_item_set_sha256,
        "actuation_calibration": actuation_calibration,
        "envelope_ruling": ENVELOPE_RULING_OF_RECORD,
        "probe_grouping_reading": PROBE_GROUPING_READING,
        "sampling_config": cell.sampling.model_dump(),
        "lesion_recipe_law": (
            "§5.4: no object in this column is built by projecting out the "
            "direction known to produce the behavior; asserted on the vector's "
            "construction provenance at cell build time"),
    }
    # E4.1's pre-written clause, attached at the cell's stamp for a FLAGGED pair
    # (§5.3 item 2). A flagged pair still runs and still scores.
    if naive_row and naive_row.get("verdict") == "ARTIFACT-EXPOSED":
        stamp["naive_transplant_clause"] = (
            "shared residual coordinate frame; a hit on this pair is not evidence "
            "of transport beyond frame-sharing; excluded from mechanism aggregates, "
            "retained in scored aggregates.")
    if extra:
        stamp.update(extra)
    assert_stamp_complete(stamp, cell_kind=cell.kind)
    return stamp


def assert_naive_row_banked(pair: str, gate: Optional[dict]) -> dict:
    """§5.3 item 1 / §9 item 7: no cell fires for a pair whose naive row is unbanked.

    A pair absent from the banked 66 + 36 rows HALTs to the desk for a gate
    extension BEFORE the cell is built — the gate is not something a cell can
    proceed without and then have added afterwards.
    """
    rows = (gate or {}).get("rows") or (gate or {}).get("pairs") or {}
    if isinstance(rows, list):
        rows = {r.get("pair"): r for r in rows if isinstance(r, dict)}
    row = rows.get(pair)
    if not row:
        raise NaiveTransplantRowMissing(
            f"no banked naive-transplant row for pair {pair!r} (§5.3 item 1 / §9 "
            "item 7). The banked gate is 66 candidate pairs + 36 gpt2-xl supplement "
            "rows; a new pair HALTs to the desk for a gate extension BEFORE its cell "
            "is built.")
    return row


# ---------------------------------------------------------------- HF stepper
class HFStepper:
    """`Stepper` over a real HF causal LM with a KV cache (§2.2's generation path).

    `use_cache=True` for generation (§2.2). Logits are returned as the LAST
    position's row block, float32 on CPU, because `sample_token` does its arithmetic
    in float64 numpy on one row — which is what makes the sampling step
    layout-invariant (see `sample_token`).
    """

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        import torch

        self.model = model
        self.torch = torch
        self.device = device or str(next(model.parameters()).device)
        self._past: Any = None
        self._len = 0

    def _logits(self, out: Any) -> np.ndarray:
        return out.logits[:, -1, :].detach().float().cpu().numpy()

    def prefill(self, ids: Any, attention_mask: Any) -> np.ndarray:
        t = self.torch
        i = t.as_tensor(np.asarray(ids), dtype=t.long).to(self.device)
        m = t.as_tensor(np.asarray(attention_mask), dtype=t.long).to(self.device)
        self._mask = m
        with t.no_grad():
            out = self.model(input_ids=i, attention_mask=m, use_cache=True)
        self._past = out.past_key_values
        self._len = int(i.shape[1])
        return self._logits(out)

    def step(self, next_ids: Any) -> np.ndarray:
        t = self.torch
        i = t.as_tensor(np.asarray(next_ids).reshape(-1, 1), dtype=t.long
                        ).to(self.device)
        self._mask = t.cat(
            [self._mask, t.ones((self._mask.shape[0], 1), dtype=self._mask.dtype,
                                device=self._mask.device)], dim=1)
        cache_position = t.arange(self._len, self._len + 1, device=self._mask.device)
        with t.no_grad():
            out = self.model(input_ids=i, attention_mask=self._mask,
                             past_key_values=self._past, use_cache=True,
                             cache_position=cache_position)
        self._past = out.past_key_values
        self._len += 1
        return self._logits(out)

    def close(self) -> None:
        self._past = None


# ---------------------------------------------------------------- CPU self-test
class _StubStepper:
    """A deterministic, PER-ROW-INDEPENDENT stepper — the layout-invariance witness.

    Why a stub at all, when a tiny real Llama would be more convincing: a real
    forward's reductions are NOT batch-size invariant (§2.7 says so about GPUs, and
    the selftest MEASURES it on CPU), so a real model can only CHARACTERIZE layout
    invariance, never prove it. This stub computes each row's logits from that row's
    own token history alone, in a fixed order, so it is exactly batch-invariant by
    construction — which isolates the property under test to the HARNESS's layout
    logic, where the §2.2 contract actually lives.

    The arithmetic is deliberately non-trivial (history-dependent, vocab-spread) so
    a bug that dropped or misordered a row's history would change its tokens.
    """

    def __init__(self, vocab: int = 61, *, eos_token_id: Optional[int] = None) -> None:
        self.vocab = vocab
        self.eos_token_id = eos_token_id
        self._hist: list[list[int]] = []

    def prefill(self, ids: Any, attention_mask: Any) -> np.ndarray:
        a = np.asarray(ids, dtype=np.int64)
        m = np.asarray(attention_mask, dtype=np.int64)
        self._hist = [[int(t) for t, k in zip(row, mask) if k]
                      for row, mask in zip(a, m)]
        return self._logits()

    def step(self, next_ids: Any) -> np.ndarray:
        for r, t in enumerate(np.asarray(next_ids, dtype=np.int64).reshape(-1)):
            self._hist[r].append(int(t))
        return self._logits()

    def _logits(self) -> np.ndarray:
        out = np.zeros((len(self._hist), self.vocab), dtype=np.float32)
        idx = np.arange(self.vocab, dtype=np.float64)
        for r, h in enumerate(self._hist):
            # a row's logits depend on its own history only, in a fixed order
            acc = 0.0
            for pos, tok in enumerate(h):
                acc = (acc * 1.000003 + (tok + 1) * (pos % 7 + 1)) % 9973.0
            row = np.sin(idx * (acc % 17.0 + 0.5) * 0.011) * 3.0 + np.cos(
                idx * 0.037 + acc * 1e-4) * 1.5
            out[r] = row.astype(np.float32)
        return out

    def close(self) -> None:
        self._hist = []


class _ToyTokenizer:
    """The minimum surface `render_prompt` touches, for a weightless CPU selftest."""

    def __init__(self, vocab_size: int = 61) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.chat_template = "toy"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [4 + (ord(c) % (self.vocab_size - 5)) for c in text]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def apply_chat_template(self, messages: list[dict],
                            add_generation_prompt: bool = True, **kwargs: Any
                            ) -> list[int]:
        if "date_string" in kwargs:      # exercise the TypeError fall-back path
            raise TypeError("toy template takes no date_string")
        out = [self.bos_token_id]
        for m in messages:
            # NOT truncated: the pool's prompt lengths must genuinely vary, because
            # equal lengths would hide both a left-padding bug and the §2.6
            # probe-grouping question the harness has to answer.
            out += self.encode(m["content"], add_special_tokens=False)
        if add_generation_prompt:
            out.append(3)          # a template marker, so the two arms really differ
        return out


def _toy_pool(size: int = PROMPT_POOL_SIZE) -> PromptPool:
    """A synthetic pool of the pool-of-record's SHAPE (20 topics × 4 strata).

    Data-independent by construction: the selftest must run in a checkout with no
    gitignored data tree. Prompt lengths deliberately VARY across the pool, because
    varying prompt length is what makes the §2.6 probe-grouping question real and
    what a left-padding bug would hide.
    """
    strata = ("expository", "explanatory", "argumentative", "conversational")
    prompts = []
    for t in range(PROMPT_POOL_TOPICS):
        for s, stratum in enumerate(strata):
            prompts.append(BehavioralPrompt(
                prompt_id=f"P-t{t:02d}-{stratum[:4]}", topic_idx=t,
                topic=f"topic {t}", stratum=stratum, seed=0, system_prompt="",
                user_prompt="Write about: " + ("x" * (3 + (t + 2 * s) % 11))))
    prompts = prompts[:size]
    body = json.dumps([p.model_dump() for p in prompts], sort_keys=True)
    return PromptPool(path=None,
                      sha256=hashlib.sha256(body.encode()).hexdigest(),
                      prompts=prompts)


def _toy_cells(site: int = 14) -> list[CellSpec]:
    """A miniature of the §4 + §5 cell set: enough strata for the §2.7 gate."""
    cells = [baseline_cell(site)]
    for key, kind, band in (
            ("entropy_gradient", "calibration", None),
            ("Rband1", "calibration_band", "Rband"),
            ("gentropy_gradient", "transported", None),
            ("gRband1", "transported_band", "gRband")):
        for frac in DOSE_LADDER:
            cells.append(CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key=key, site=site,
                                                frac=frac),
                kind=kind, vector_key=key, site=site, alpha_frac=frac,
                band_family=band, vector_provenance=f"toy::{key}"))
    return cells


def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, data-independent verification of every load-bearing property."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    corpus = CORPUS_SHA_V21
    node = "qwen2.5-3b-instruct"
    arm, site = "native", 26

    # ---- 1. the seed table: determinism and the M5 contract -------------------
    print("== selftest 1: seed table — same digest → same uniforms (§2.3, M25) ==")
    m1, s1, u1 = uniform_tape(corpus_sha=corpus, node_key=node, arm=arm, site=site,
                              cell_id="gentropy_gradient_L26_a+0.30", gen_id=7, n=8)
    m2, s2, u2 = uniform_tape(corpus_sha=corpus, node_key=node, arm=arm, site=site,
                              cell_id="gentropy_gradient_L26_a+0.30", gen_id=7, n=8)
    check("same seed material → identical material string", m1 == m2, m1)
    check("same material → identical seed_int", s1 == s2, str(s1))
    check("same seed → bitwise identical uniform tape",
          np.array_equal(u1, u2), f"u[0]={u1[0]:.17g}")
    check("seed_int is the sha256[:8] big-endian integer, never `hash()`",
          s1 == int.from_bytes(hashlib.sha256(m1.encode()).digest()[:8], "big"),
          str(s1))
    # M25(c) says to GREP new analysis code for `hash(` before archiving it as an
    # operationalization of record. Running the grep as an assertion means the code
    # cannot be archived without it having been run. Backtick-quoted prose mentions
    # of `hash()` are excluded by the lookbehind; a real call site is not.
    import re as _re
    hash_calls = _re.findall(r"(?<![\w.`])hash\s*\(",
                             Path(__file__).read_text())
    check("no `hash(` call site anywhere in this module (M25's grep, as an assertion)",
          not hash_calls, f"{len(hash_calls)} candidate call site(s)")
    _, sA, uA = uniform_tape(corpus_sha=corpus, node_key=node, arm=arm, site=site,
                             cell_id="gRband1_L26_a-0.03", gen_id=7, n=8)
    check("a DIFFERENT cell at the same gen_id draws a different tape",
          sA != s1 and not np.array_equal(uA, u1), f"seeds {s1} vs {sA}")
    # THE M5 CONTRACT: changed cell membership does not desync the others.
    full = _toy_cells(site)
    subset = [c for c in full if c.vector_key != "gRband1"]
    tapes_full = {c.cell_id: uniform_tape(
        corpus_sha=corpus, node_key=node, arm=arm, site=site, cell_id=c.cell_id,
        gen_id=3, n=6)[2] for c in full}
    tapes_subset = {c.cell_id: uniform_tape(
        corpus_sha=corpus, node_key=node, arm=arm, site=site, cell_id=c.cell_id,
        gen_id=3, n=6)[2] for c in subset}
    desynced = [cid for cid in tapes_subset
                if not np.array_equal(tapes_subset[cid], tapes_full[cid])]
    check("M5: dropping cells from the run does NOT desync the survivors' tapes",
          not desynced,
          f"{len(tapes_subset)} survivors of {len(tapes_full)} cells, "
          f"{len(desynced)} desynced")
    roots = {c.cell_id: cell_seed_root(corpus_sha=corpus, node_key=node, arm=arm,
                                      site=site, cell_id=c.cell_id) for c in full}
    check("per-cell seed roots are distinct (§2.8's stamp field)",
          len(set(roots.values())) == len(roots), f"{len(set(roots.values()))} roots")
    check("changing the corpus sha changes every seed root",
          all(cell_seed_root(corpus_sha="0" * 64, node_key=node, arm=arm, site=site,
                             cell_id=cid) != r for cid, r in roots.items()))

    # ---- 2. the fixed sampling step ------------------------------------------
    print("== selftest 2: the fixed sampling step (ruling 8) ==")
    logits = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    check("u→token is monotone in u (an inverse-CDF, not a shuffle)",
          [sample_token(logits, u) for u in (0.0, 0.999999)] == [0, 3],
          str([sample_token(logits, u) for u in (0.0, 0.25, 0.5, 0.9, 0.999999)]))
    check("float32 vs float64 logits row give the SAME token (the cast is explicit)",
          all(sample_token(logits, u) == sample_token(logits.astype(np.float64), u)
              for u in np.linspace(0, 0.999, 25)))
    check("a batched logits matrix samples row-wise, independent of neighbours",
          sample_token(np.stack([logits, logits * 7])[0], 0.4)
          == sample_token(logits, 0.4))
    check("top_k=1 collapses to argmax (the bridge-cell filter path)",
          all(sample_token(logits, u, SamplingConfig(
              name="k1", top_k=1, provenance="selftest")) == 2
              for u in (0.01, 0.5, 0.99)))
    check("top_p<1 truncates the tail (the bridge-cell filter path)",
          sample_token(logits, 0.999999, SamplingConfig(
              name="p", top_p=0.5, provenance="selftest")) == 2)
    check("the config of record is pure ancestral, T=1, top_p=1, top_k=0",
          (SAMPLING_OF_RECORD.do_sample, SAMPLING_OF_RECORD.temperature,
           SAMPLING_OF_RECORD.top_p, SAMPLING_OF_RECORD.top_k) == (True, 1.0, 1.0, 0),
          SAMPLING_OF_RECORD.name)

    # ---- 3. LAYOUT INVARIANCE — the §2.2 replay-gate property -----------------
    print("== selftest 3: layout invariance at B ∈ {80, 40, 20, 10, 1} (§2.2) ==")
    tok = _ToyTokenizer()
    pool = _toy_pool()
    cell = CellSpec(cell_id=CELL_ID_TEMPLATE.format(
        vector_key="gentropy_gradient", site=site, frac=0.3),
        kind="transported", vector_key="gentropy_gradient", site=site,
        alpha_frac=0.3, band_family=None, vector_provenance="toy")

    def run_at(B: int, n: int = N_PER_CELL) -> list[GenerationRecord]:
        layout = CanonicalLayout(batch_size=B, dtype="float32", max_new_tokens=12,
                                 frozen=True)
        return generate_cell(
            lambda: _StubStepper(eos_token_id=None),
            cell=cell, pool=pool,
            tokenize=lambda p: render_prompt(p, tok, arm),
            layout=layout, pad_token_id=0, eos_token_id=None,
            corpus_sha=corpus, node_key=node, arm=arm, n=n)

    ref = run_at(80)
    ref_ids = {r.generation_id: tuple(r.generated_ids) for r in ref}
    per_B = {}
    for B in (40, 20, 10):
        recs = run_at(B)
        got = {r.generation_id: tuple(r.generated_ids) for r in recs}
        per_B[B] = got == ref_ids
        check(f"a cell's 80 generations are IDENTICAL at B={B} and B=80",
              got == ref_ids,
              f"{sum(1 for g in ref_ids if got.get(g) == ref_ids[g])}/80 rows "
              f"bitwise equal; token digest {token_id_digest(recs)[:12]}…")
    check("the token-id digest is layout-invariant across the whole ladder",
          len({token_id_digest(run_at(B)) for B in BATCH_LADDER}) == 1,
          f"one digest across B ∈ {list(BATCH_LADDER)}: "
          f"{token_id_digest(ref)[:16]}…")
    check("the sub-batch map is total and deterministic (gen_id // B, gen_id % B)",
          all(CanonicalLayout(batch_size=20, dtype="f32").sub_batch_of(g)
              == (g // 20, g % 20) for g in range(80)))
    check("records come back in gen_id order regardless of sub-batching",
          [r.generation_id for r in run_at(10)] == list(range(80)))
    check("every generation records the layout slot it rode in (§2.2 stamping)",
          all(r.sub_batch == r.generation_id // 10 and r.row == r.generation_id % 10
              for r in run_at(10)))
    # left padding is what makes ONE start_pos valid for the whole batch
    seen_L: list[int] = []
    layout10 = CanonicalLayout(batch_size=10, dtype="float32", max_new_tokens=6,
                               frozen=True)
    recs10 = generate_cell(lambda: _StubStepper(), cell=cell, pool=pool,
                           tokenize=lambda p: render_prompt(p, tok, arm),
                           layout=layout10, pad_token_id=0, eos_token_id=None,
                           corpus_sha=corpus, node_key=node, arm=arm,
                           start_pos_sink=seen_L.append, n=20)
    check("left padding: one start_pos per sub-batch, == the padded prompt length",
          len(seen_L) == 2 and all(
              r.padded_prompt_length == seen_L[r.sub_batch] for r in recs10),
          f"start_pos per sub-batch {seen_L}")
    check("prompt lengths genuinely VARY in the pool (the padding is exercised)",
          len({r.prompt_length for r in recs10}) > 1,
          str(sorted({r.prompt_length for r in recs10})))
    ids, mask, L = left_pad([[5, 6, 7], [8]], 0)
    check("left_pad puts pad on the LEFT and masks it",
          ids.tolist() == [[5, 6, 7], [0, 0, 8]]
          and mask.tolist() == [[1, 1, 1], [0, 0, 1]] and L == 3)
    rids, rmask = right_pad([[5, 6, 7], [8]], 0)
    check("right_pad puts pad on the RIGHT and masks it (§2.6's probe side)",
          rids.tolist() == [[5, 6, 7], [8, 0, 0]]
          and rmask.tolist() == [[1, 1, 1], [1, 0, 0]])
    # an early EOS must not move any other row's slot
    eos_recs = generate_cell(
        lambda: _StubStepper(), cell=cell, pool=pool,
        tokenize=lambda p: render_prompt(p, tok, arm), layout=layout10,
        pad_token_id=0, eos_token_id=int(ref[0].generated_ids[0]),
        corpus_sha=corpus, node_key=node, arm=arm, n=20)
    check("an EOS row freezes without moving another row's slot",
          all(r.row == r.generation_id % 10 for r in eos_recs)
          and any(r.finished_with_eos for r in eos_recs),
          f"{sum(r.finished_with_eos for r in eos_recs)}/20 rows hit EOS")
    check("a frozen row stops consuming its uniform tape",
          all(r.n_uniforms_consumed == len(r.generated_ids) for r in eos_recs))

    # ---- 4. the entropy convention fixture -----------------------------------
    print("== selftest 4: entropy convention (§2.6) ==")
    import torch

    V, T, P = 8, 7, 4
    torch.manual_seed(20260729)
    fixture = torch.randn(T, V)
    seq = [3, 1, 4, 1, 5, 7, 2]                # every id < V, so the NLL gather is real
    ent, nll = per_position_entropy_and_nll(fixture, seq, P, T - P)
    lp = torch.log_softmax(fixture.float(), dim=-1)
    want_ent = np.array([float(-(lp[p].exp() * lp[p]).sum()) for p in (3, 4, 5)],
                        dtype=np.float32)
    want_nll = np.array([float(-lp[p, seq[p + 1]]) for p in (3, 4, 5)],
                        dtype=np.float32)
    check("entropy slice is [P-1, T-1): the distributions that PRODUCED the span",
          ent.shape == (3,) and np.allclose(ent, want_ent, atol=0, rtol=1e-6),
          f"{ent.tolist()}")
    check("NLL is the surprisal of the ACTUAL generated token (the likelihood rung)",
          np.allclose(nll, want_nll, atol=0, rtol=1e-6), f"{nll.tolist()}")
    check("probe arrays are float32 (the §2.7 digest's dtype)",
          ent.dtype == np.float32 and nll.dtype == np.float32)
    check("a zero-length generated span is refused, never averaged over nothing",
          _raises(lambda: per_position_entropy_and_nll(fixture, seq, P, 0),
                  ValueError))
    rows = [ProbeRow(generation_id=i, n_positions=3, mean_entropy_steered=2.0 + i,
                     mean_entropy_unsteered=1.0 + i, entropy_rise=1.0,
                     base_model_nll=3.0, probe_group=0, probe_group_size=2)
            for i in range(2)]
    agg = entropy_rise_from_rows(rows)
    check("cell-level rise = mean(steered) − mean(unsteered), 4 dp",
          agg["entropy_rise"] == 1.0 and agg["mean_entropy_steered"] == 2.5,
          json.dumps(agg))
    check("the aggregation order matches entropy_write_probe (per-gen mean, then mean)",
          agg["n"] == 2 and agg["base_model_nll"] == 3.0)
    d1 = entropy_array_digest([np.array([1.0, 2.0], dtype=np.float32)])
    check("entropy digests are bitwise over float32 .tobytes()",
          d1 == hashlib.sha256(
              np.array([1.0, 2.0], dtype=np.float32).tobytes()).hexdigest())
    check("a float64 entropy array is REFUSED, not silently cast",
          _raises(lambda: entropy_array_digest(
              [np.array([1.0], dtype=np.float64)]), ReplayGateNotBitwise))

    # ---- 4b. the probe grouping (§2.6's resolved ambiguity) -------------------
    print("== selftest 4b: probe grouping — one valid mask per group ==")
    groups = probe_groups(ref, max_group=32)
    check("every probe group has ONE prompt_length (the mask-validity constraint)",
          all(len({ref[i].prompt_length for i in g}) == 1 for g in groups),
          f"{len(groups)} groups, sizes {sorted({len(g) for g in groups})}")
    check("probe grouping is total: every generation is in exactly one group",
          sorted(i for g in groups for i in g) == list(range(len(ref))))
    check("group order is ascending prompt_length, gen_id order within a length",
          [ref[g[0]].prompt_length for g in groups]
          == sorted(ref[g[0]].prompt_length for g in groups)
          and all([ref[i].generation_id for i in g]
                  == sorted(ref[i].generation_id for i in g) for g in groups))
    check("max_group caps a group's size (the measured bs32 anchor)",
          all(len(g) <= 4 for g in probe_groups(ref, max_group=4)))
    inv = characterize_probe_batch_invariance(
        [np.array([1.0, 2.0], dtype=np.float32)],
        {1: [np.array([1.0, 2.5], dtype=np.float32)],
         8: [np.array([1.0, 2.0], dtype=np.float32)]}, 32)
    check("batch-invariance characterization names the first divergent index",
          inv.first_divergent_token_index["1"] == 1
          and inv.first_divergent_token_index["8"] is None,
          json.dumps(inv.first_divergent_token_index))
    check("the characterization is DESCRIPTIVE — it returns, never raises (M19)",
          inv.bitwise_identical == {"1": False, "8": True})

    # ---- 5. dose-ladder application exactness --------------------------------
    print("== selftest 5: dose ladder application (§2.5, frozen) ==")
    norm = 12.2391
    laddered = apply_dose_ladder("gentropy_gradient", site,
                                 per_token_median_resid_norm=norm,
                                 kind="transported", vector_provenance="toy")
    check("the ladder is 6 signed cells, in ladder order, no baseline",
          [c.alpha_frac for c, _ in laddered] == list(DOSE_LADDER),
          str([c.alpha_frac for c, _ in laddered]))
    check("α = frac × per-token median residual norm, exactly",
          all(a == frac * norm for (c, a), frac in zip(laddered, DOSE_LADDER)),
          f"α(+0.3) = {laddered[-1][1]!r} vs {0.3 * norm!r}")
    check("cell ids use the banked a{frac:+.2f} formatting (§2.3)",
          [c.cell_id for c, _ in laddered][:2]
          == [f"gentropy_gradient_L{site}_a-0.30",
              f"gentropy_gradient_L{site}_a-0.10"],
          laddered[0][0].cell_id)
    check("score_entropy_writes.parse_cell round-trips every laddered cell id",
          _parse_roundtrip([c.cell_id for c, _ in laddered], site))
    check("an off-ladder dose is REFUSED (no extension, no interpolation)",
          _raises(lambda: resolve_alpha(0.2, norm), ValueError)
          and _raises(lambda: CellSpec(
              cell_id=f"v_L{site}_a+0.20", kind="transported", vector_key="v",
              site=site, alpha_frac=0.2), ValueError))
    check("the baseline cell is α=0 with no vector, built once and SHARED",
          baseline_cell(site).alpha_frac == 0.0
          and baseline_cell(site).vector_key is None
          and baseline_cell(site).is_baseline)
    check("a non-positive residual norm cannot resolve a dose",
          _raises(lambda: resolve_alpha(0.3, 0.0), ResidualNormDeltaError)
          and _raises(lambda: resolve_alpha(0.3, float("nan")),
                      ResidualNormDeltaError))
    check("a mistyped cell_id is refused (the id IS the content)",
          _raises(lambda: CellSpec(cell_id="wrong", kind="transported",
                                   vector_key="v", site=site, alpha_frac=0.3),
                  ValueError))
    check("the two band families stay distinct in name (§4.1)",
          CellSpec(cell_id=f"Rband1_L{site}_a+0.30", kind="calibration_band",
                   vector_key="Rband1", site=site, alpha_frac=0.3,
                   band_family="Rband").is_null
          and CellSpec(cell_id=f"gRband1_L{site}_a+0.30", kind="transported_band",
                       vector_key="gRband1", site=site, alpha_frac=0.3,
                       band_family="gRband").band_family == "gRband")

    # ---- 6. norm conventions + the 10% HALT ----------------------------------
    print("== selftest 6: both norm conventions, delta recorded not absorbed ==")
    nc = resolve_norms(site=site, measured=12.2391, measured_provenance="in-job",
                       banked_per_token=12.2000, banked_provenance="a5 stamps",
                       banked_mean_state=12.1125,
                       banked_mean_state_provenance="collection stamps")
    check("both conventions are recorded, only the per-token one sets α",
          nc.measured_per_token_median == 12.2391
          and nc.banked_mean_state_median == 12.1125
          and "PER-TOKEN" in nc.used_for_alpha)
    check("the delta vs banked is RECORDED, not absorbed",
          abs(nc.delta_fraction_vs_banked - 0.0032) < 1e-4,
          f"Δ = {nc.delta_fraction_vs_banked * 100:.3f}%")
    check("a >10% delta is a HALT (§9 item 8)",
          _raises(lambda: resolve_norms(site=site, measured=20.0,
                                        measured_provenance="in-job",
                                        banked_per_token=12.2),
                  ResidualNormDeltaError))
    check("a NEW node with no banked a5 stamp records no delta and does not halt",
          resolve_norms(site=site, measured=9.5,
                        measured_provenance="in-job").delta_fraction_vs_banked
          is None)

    # ---- 7. replay-gate selection + verdict ----------------------------------
    print("== selftest 7: the in-job replay gate (§2.7) ==")
    sel, strata, dig = select_replay_cells(full, node, corpus)
    sel2, _, _ = select_replay_cells(list(reversed(full)), node, corpus)
    check("K=3, one cell per frozen stratum", len(sel) == 3
          and sorted(strata) == sorted(REPLAY_GATE_STRATA), json.dumps(strata))
    check("selection is deterministic from sha256(node_key|corpus_sha)",
          dig == hashlib.sha256(f"{node}|{corpus}".encode()).hexdigest(),
          dig[:16] + "…")
    check("selection is order-independent (the cell SET decides, not the json order)",
          sel == sel2, str(sel))
    check("the signal cell is at |0.3| and the band cell is a transported band",
          abs(float(strata["signal_at_0.3"].rsplit("_a", 1)[1])) == 0.3
          and strata["random_band"].startswith("gRband"))
    check("a different node selects a different gate triple (usually) but always 3",
          len(select_replay_cells(full, "phi-4", corpus)[0]) == 3)
    tfirst = {c: f"{i:064x}" for i, c in enumerate(sel)}
    efirst = {c: f"{i + 9:064x}" for i, c in enumerate(sel)}
    gate = evaluate_replay_gate(selection=sel, strata=strata, digest=dig,
                                token_first=tfirst, token_replay=dict(tfirst),
                                entropy_first=efirst, entropy_replay=dict(efirst))
    check("a bitwise replay PASSES the gate", gate.passed and not gate.mismatches)
    bad = dict(tfirst)
    bad[sel[0]] = "f" * 64
    check("a token-id mismatch HALTs (no cell from the node is quotable)",
          _raises(lambda: evaluate_replay_gate(
              selection=sel, strata=strata, digest=dig, token_first=tfirst,
              token_replay=bad, entropy_first=efirst, entropy_replay=efirst),
              ReplayGateNotBitwise))
    badE = dict(efirst)
    badE[sel[1]] = "e" * 64
    check("an ENTROPY-array mismatch HALTs too (both digests are blocking)",
          _raises(lambda: evaluate_replay_gate(
              selection=sel, strata=strata, digest=dig, token_first=tfirst,
              token_replay=tfirst, entropy_first=efirst, entropy_replay=badE),
              ReplayGateNotBitwise))
    check("a MISSING replay digest is a mismatch, never a pass by omission",
          _raises(lambda: evaluate_replay_gate(
              selection=sel, strata=strata, digest=dig, token_first=tfirst,
              token_replay={}, entropy_first=efirst, entropy_replay=efirst),
              ReplayGateNotBitwise))
    check("a cell set with no calibration cell cannot constitute the gate",
          _raises(lambda: select_replay_cells(
              [c for c in full if c.kind != "calibration"], node, corpus),
              ExpectedNShortfall))
    check("the gate's token digest is computed in gen_id order",
          token_id_digest(ref) == token_id_digest(list(reversed(ref))))

    # ---- 8. stamp completeness against §2.8 ----------------------------------
    print("== selftest 8: stamp completeness as a §2.8 checklist assertion ==")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    stamp = _toy_stamp(cell=laddered[-1][0], alpha=laddered[-1][1],
                       layout=freeze_layout(80, dtype="bfloat16",
                                            headroom_note="selftest"),
                       pool=pool, norms=nc, corpus=corpus, node=node, arm=arm,
                       roots=roots)
    check("a fully-populated stamp passes the §2.8 checklist",
          _ok(lambda: assert_stamp_complete(stamp, cell_kind="transported")),
          f"{len(STAMP_REQUIRED_FIELDS)} required fields")
    check("§2.8 checklist covers all 31 named custody items",
          len(STAMP_REQUIRED_FIELDS) == 31, str(len(STAMP_REQUIRED_FIELDS)))
    for field in ("corpus_manifest_sha256", "behavioral_prompt_pool_sha256",
                  "scheduler_card_index", "canonical_batch_layout",
                  "model_config_sha256", "replay_gate_digests",
                  "battery_item_set_sha256", "envelope_ruling",
                  "actuation_calibration", "seed_material_template_digest"):
        holed = {k: v for k, v in stamp.items() if k != field}
        check(f"a stamp missing {field} is REFUSED (OWED, never silent)",
              _raises(lambda h=holed: assert_stamp_complete(h,
                                                            cell_kind="transported"),
                      StampIncompleteError))
    nulled = dict(stamp, transport_map_fit_sha256=None)
    check("a NULL required field is refused on a transported cell",
          _raises(lambda: assert_stamp_complete(nulled, cell_kind="transported"),
                  StampIncompleteError))
    check("the same null is RULED legitimate on a calibration cell (no pair, no map)",
          _ok(lambda: assert_stamp_complete(
              dict(stamp, transport_map_fit_sha256=None,
                   transport_map_family=None, transport_map_arm=None,
                   transport_map_corpus_vintage=None,
                   naive_transplant_gate_row=None,
                   naive_transplant_verdict=None), cell_kind="calibration")))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    check("an unset CVD is a HALT (M10: the field that caught the rogue run)",
          _raises(lambda: assert_stamp_complete(
              dict(stamp, cuda_visible_devices=CVD_UNSET_SENTINEL),
              cell_kind="transported"), StampIncompleteError))
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    check("the stamp carries the envelope ruling's wording verbatim (ruling 3)",
          "3 transported members" in stamp["envelope_ruling"]
          and "≥100" in stamp["envelope_ruling"])
    check("the stamp carries the probe-grouping reading (the named §2.6 resolution)",
          "DESK-OWED READING" in stamp["probe_grouping_reading"])
    check("the stamp carries the grade line UNSTAMPED (C§8)",
          stamp["grade"] == "UNSTAMPED (C§8)")
    check("the stamp carries the brief of record and its sha",
          stamp["brief_sha256"] == BRIEF_SHA256)
    flagged = _toy_stamp(cell=laddered[-1][0], alpha=laddered[-1][1],
                         layout=freeze_layout(80, dtype="bfloat16"), pool=pool,
                         norms=nc, corpus=corpus, node=node, arm=arm, roots=roots,
                         naive_verdict="ARTIFACT-EXPOSED")
    check("E4.1's clause attaches automatically on a FLAGGED pair (§5.3 item 2)",
          "frame-sharing" in flagged.get("naive_transplant_clause", ""),
          flagged["naive_transplant_verdict"])
    check("a flagged pair still RUNS and still SCORES (the clause is not an exclusion)",
          flagged["grade"] == GRADE_LINE and "excluded from mechanism aggregates"
          in flagged["naive_transplant_clause"])

    # ---- 9. HALT semantics for the remaining §9 conditions --------------------
    print("== selftest 9: §9 HALT conditions are named exceptions, not warnings ==")
    check("every HALT class descends from BehavioralHarnessError",
          all(issubclass(c, BehavioralHarnessError) for c in (
              ArtifactShaMismatch, CorpusVintageError, NativeVectorUnavailable,
              SiteNotOfRecord, ReplayGateNotBitwise, NaiveTransplantRowMissing,
              ResidualNormDeltaError, StampIncompleteError, ExpectedNShortfall,
              HookAdmissibilityError, PositionCeilingExceeded,
              CanonicalLayoutInvalidated, LesionRecipeViolation)))
    check("corpus vintage: v2.1 across the chain passes",
          assert_corpus_vintage(corpus, corpus, corpus) == corpus)
    check("a MIXED chain is a HALT (§9 item 2)",
          _raises(lambda: assert_corpus_vintage(corpus, "1" * 64),
                  CorpusVintageError))
    check("a HOLE in the chain is reported as a hole, not as agreement",
          _raises(lambda: assert_corpus_vintage(corpus, None), CorpusVintageError))
    check("a non-v2.1 vintage is a HALT",
          _raises(lambda: assert_corpus_vintage("2" * 64), CorpusVintageError))
    check("gpt2-xl's position ceiling: 1024 - 512 leaves 512 prompt tokens",
          _ok(lambda: assert_position_budget(512, 512, GPT2_POSITION_CEILING))
          and _raises(lambda: assert_position_budget(513, 512,
                                                     GPT2_POSITION_CEILING),
                      PositionCeilingExceeded))
    check("a node with no ceiling is unconstrained (the historical key set)",
          _ok(lambda: assert_position_budget(10_000, 512, None)))
    check("α=0-is-no-hook is asserted bitwise, not approximately",
          _ok(lambda: assert_alpha_zero_is_no_hook(
              np.array([1.0, 2.0], dtype=np.float32),
              np.array([1.0, 2.0], dtype=np.float32)))
          and _raises(lambda: assert_alpha_zero_is_no_hook(
              np.array([1.0, 2.0], dtype=np.float32),
              np.array([1.0, 2.0001], dtype=np.float32)),
              ReplayGateNotBitwise))
    check("the lesion-recipe law refuses an orthogonalized construction (§5.4)",
          _raises(lambda: assert_no_lesion_recipe(
              "unit(Vrep entropy-orthogonalized against entropy_gradient_L26)",
              "gVrep_perp"), LesionRecipeViolation))
    check("a clean provenance passes the lesion-recipe law",
          _ok(lambda: assert_no_lesion_recipe(
              "unit(g · entropy_gradient_L16) via proc_k128", "gentropy_gradient")))
    check("a naive-transplant row must be banked before a cell is built (§9 item 7)",
          _raises(lambda: assert_naive_row_banked("8b->falcon-mamba-7b", {"rows": {}}),
                  NaiveTransplantRowMissing)
          and assert_naive_row_banked(
              "8b->qwen-7b",
              {"rows": [{"pair": "8b->qwen-7b", "verdict": "CLEAR"}]})["verdict"]
          == "CLEAR")
    check("an expected-N shortfall is a HALT, not a rounding (M23)",
          _raises(lambda: generate_cell(
              lambda: _StubStepper(), cell=cell, pool=pool,
              tokenize=lambda p: render_prompt(p, tok, arm),
              layout=CanonicalLayout(batch_size=10, dtype="f32",
                                     max_new_tokens=2, frozen=True),
              pad_token_id=0, eos_token_id=None, corpus_sha=corpus,
              node_key=node, arm=arm, n=0), ValueError))
    check("an off-ladder batch size is refused (§2.2's fixed ladder)",
          _raises(lambda: CanonicalLayout(batch_size=64, dtype="f32"), ValueError))
    check("a layout is not FROZEN until it is frozen (§2.2)",
          CanonicalLayout(batch_size=80, dtype="f32").frozen is False
          and freeze_layout(80, dtype="f32").frozen is True)

    # ---- 10. preflight (M10 / M19) -------------------------------------------
    print("== selftest 10: preflight is exit-early and M19-safe ==")
    B, measured, note = choose_batch_size(lambda b: 0.10 * (b / 10.0))
    check("preflight takes the LARGEST rung with ≥15% headroom",
          (B, measured) == (80, True), f"B={B}: {note}")
    B2, m2, n2 = choose_batch_size(lambda b: 0.30 * (b / 10.0))
    check("a tighter node steps down the ladder rather than off it",
          B2 in BATCH_LADDER and B2 < 80, f"B={B2}: {n2}")
    B3, m3, n3 = choose_batch_size(lambda b: 1.0 / 0)
    check("a RAISING probe takes the conservative bound, measured=false (M19)",
          (B3, m3) == (min(BATCH_LADDER), False), n3)
    B4, m4, _ = choose_batch_size(lambda b: float("nan"))
    check("a non-finite probe reading also blocks on the bound (M19)",
          (B4, m4) == (min(BATCH_LADDER), False))
    B5, m5, _ = choose_batch_size(None)
    check("no probe at all = the conservative bound, never a guess",
          (B5, m5) == (min(BATCH_LADDER), False))
    check("a layout that cannot fit any rung still names the smallest, measured",
          choose_batch_size(lambda b: 0.99)[0] == min(BATCH_LADDER))

    # ---- 11. prompt pool (ruling 10) -----------------------------------------
    print("== selftest 11: the prompt pool of record (ruling 10) ==")
    check("the pool of record is 20 topics × 4 strata = 80",
          len(pool.prompts) == 80 and PROMPT_POOL_SIZE == 80)
    check("gen_id → prompt is total and deterministic",
          pool.for_gen(0).prompt_id == pool.prompts[0].prompt_id
          and pool.for_gen(80).prompt_id == pool.prompts[0].prompt_id)
    check("a short pool is REFUSED (a partial pool changes every cell's tokens)",
          _raises(lambda: PromptPool(path=None, sha256="x",
                                     prompts=pool.prompts[:79]), ValueError))
    check("a pool that is not 20×4 is refused even at the right size",
          _raises(lambda: PromptPool(
              path=None, sha256="x",
              prompts=[p.model_copy(update={"topic_idx": 0}) for p in pool.prompts]),
              ValueError))
    with tempfile.TemporaryDirectory(prefix="behav_pool_") as td:
        p = Path(td) / "pool.json"
        p.write_text(json.dumps([q.model_dump() for q in pool.prompts]))
        loaded = load_prompt_pool(p)
        check("a pool loads from disk with its sha frozen",
              len(loaded.prompts) == 80
              and loaded.sha256 == hashlib.sha256(p.read_bytes()).hexdigest(),
              loaded.sha256[:16] + "…")
        check("the loader accepts the banked {'entries': [...]} shape too",
              len(load_prompt_pool(_write(Path(td) / "e.json", {
                  "entries": [q.model_dump() for q in pool.prompts]})).prompts) == 80)
        check("a missing pool is a named refusal, not a silent empty pool",
              _raises(lambda: load_prompt_pool(Path(td) / "nope.json"),
                      PromptPoolError))
        check("an unrecognized prompt record is refused, never partially loaded",
              _raises(lambda: load_prompt_pool(_write(Path(td) / "bad.json",
                                                      [{"nope": 1}])),
                      PromptPoolError))
    check("native arm renders through the chat template (date_string fall-back)",
          render_prompt(pool.prompts[0], tok, "native")[0] == tok.bos_token_id)
    check("raw arm renders plain, and the two arms differ",
          render_prompt(pool.prompts[0], tok, "raw")
          != render_prompt(pool.prompts[0], tok, "native"))
    check("an unknown arm is refused",
          _raises(lambda: render_prompt(pool.prompts[0], tok, "sideways"),
                  ValueError))

    # ---- 12. ruling 4's bridge cell ------------------------------------------
    print("== selftest 12: the bridge cell (ruling 4) ==")
    ran = BridgeCellDecision(pair="8b->qwen-7b", banked_config_recoverable=True,
                             dropped=False,
                             sampling=SamplingConfig(
                                 name="node-generation_config", temperature=0.7,
                                 top_p=0.9, provenance="node generation_config"))
    check("a recoverable banked config runs the bridge cell under it",
          ran.sampling.temperature == 0.7 and not ran.dropped)
    dropped = BridgeCellDecision(
        pair="8b->qwen-7b", banked_config_recoverable=False, dropped=True,
        discontinuity_named="the banked leg's sampling config is not recoverable "
                            "from its metadata; the bridge cell is DROPPED and the "
                            "harness/vintage discontinuity is named, not papered over")
    check("an unrecoverable config DROPS the cell and NAMES the discontinuity",
          dropped.dropped and "not papered over" in dropped.discontinuity_named)
    check("an unnamed drop is refused (ruling 4's whole point)",
          _raises(lambda: BridgeCellDecision(
              pair="p", banked_config_recoverable=False, dropped=True), ValueError))
    check("a bridge cell that runs without a sampling config is refused",
          _raises(lambda: BridgeCellDecision(
              pair="p", banked_config_recoverable=True, dropped=False), ValueError))

    # ---- 13. real-model characterization (availability-branched, M19) ---------
    print("== selftest 13: real-forward characterization (never a gate) ==")
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(20260729)
        cfg = LlamaConfig(vocab_size=61, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=128,
                          tie_word_embeddings=False)
        real = LlamaForCausalLM(cfg).to(torch.float32).eval()
        real.requires_grad_(False)
        small = CanonicalLayout(batch_size=10, dtype="float32", max_new_tokens=5,
                               frozen=True)
        r10 = generate_cell(lambda: HFStepper(real, "cpu"), cell=cell, pool=pool,
                            tokenize=lambda p: render_prompt(p, tok, arm),
                            layout=small, pad_token_id=0, eos_token_id=None,
                            corpus_sha=corpus, node_key=node, arm=arm, n=20)
        small20 = CanonicalLayout(batch_size=20, dtype="float32", max_new_tokens=5,
                                  frozen=True)
        r20 = generate_cell(lambda: HFStepper(real, "cpu"), cell=cell, pool=pool,
                            tokenize=lambda p: render_prompt(p, tok, arm),
                            layout=small20, pad_token_id=0, eos_token_id=None,
                            corpus_sha=corpus, node_key=node, arm=arm, n=20)
        check("the REAL HF path runs through the same loop and stepper protocol",
              len(r10) == 20 and all(len(r.generated_ids) == 5 for r in r10),
              f"20 generations × 5 tokens through HFStepper")
        same = token_id_digest(r10) == token_id_digest(r20)
        first_div = None
        if not same:
            for a, b in zip(sorted(r10, key=lambda x: x.generation_id),
                            sorted(r20, key=lambda x: x.generation_id)):
                for i, (x, y) in enumerate(zip(a.generated_ids, b.generated_ids)):
                    if x != y:
                        first_div = i if first_div is None else min(first_div, i)
                        break
        # M19: DESCRIPTIVE. A real forward's reductions are not batch-invariant, so
        # this can only be characterized. The stub-stepper proof above is what
        # asserts the harness's own §2.2 contract.
        check("real-forward B=10 vs B=20 CHARACTERIZED (M19: cannot fail a gate)",
              True,
              f"bitwise={same}"
              + ("" if same else f", first divergent token index {first_div} — "
                                 "EXPECTED; §2.7 defines replay IN the canonical "
                                 "layout for exactly this reason"))
        # the α=0 short-circuit, on the real hook
        from metabasis.extraction.hooks import (ResidualWriteSpec,
                                                attach_residual_write)
        ids0 = torch.as_tensor([render_prompt(pool.prompts[0], tok, arm)],
                               dtype=torch.long)
        with torch.no_grad():
            plain = real(ids0, use_cache=False).logits.detach().numpy()
        h = attach_residual_write(real, ResidualWriteSpec(
            layer_idx=1, vector=torch.randn(32), alpha=0.0,
            start_pos=int(ids0.shape[1]), end_pos=None, normalize=True))
        try:
            with torch.no_grad():
                hooked = real(ids0, use_cache=False).logits.detach().numpy()
        finally:
            h.remove()
        check("§2.4 on the REAL hook: α=0 with the hook attached is bitwise no-hook",
              _ok(lambda: assert_alpha_zero_is_no_hook(hooked, plain)),
              "the baseline cell is shared between §4 and §5 because of this")
        adm = ssm_hook_admissibility_preflight(
            real, node_key="toy-llama", site=1, vector=np.ones(32, dtype=np.float32),
            n_tokens=4, prompt_ids=render_prompt(pool.prompts[0], tok, arm))
        check("hook-admissibility preflight PASSES on a cache_position architecture",
              adm.admissible and adm.saw_cache_position
              and adm.positions_injected == adm.positions_expected, adm.detail)
        check("preflight refuses a site that is not a decoder-layer index",
              _raises(lambda: ssm_hook_admissibility_preflight(
                  real, node_key="toy-llama", site=99,
                  vector=np.ones(32, dtype=np.float32)), SiteNotOfRecord))
    except ImportError as exc:                      # availability branch
        check("real-forward characterization SKIPPED (transformers unavailable)",
              True, f"{exc} — the stub-stepper proofs above are the gate")

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    print(f"selftest checks run: {len(checks)}")
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


def _write(path: Path, doc: Any) -> Path:
    path.write_text(json.dumps(doc))
    return path


def _parse_roundtrip(cell_ids: Sequence[str], site: int) -> bool:
    from metabasis.scripts.score_entropy_writes import parse_cell
    for cid in cell_ids:
        vname, s, frac = parse_cell(cid)
        if s != site or CELL_ID_TEMPLATE.format(
                vector_key=vname, site=s, frac=frac) != cid:
            return False
    return True


def _toy_stamp(*, cell: CellSpec, alpha: float, layout: CanonicalLayout,
               pool: PromptPool, norms: NormConventions, corpus: str, node: str,
               arm: str, roots: dict[str, str],
               naive_verdict: str = "CLEAR") -> dict:
    return build_stamp(
        cell=cell, alpha=alpha, layout=layout, pool=pool, norms=norms,
        corpus_sha=corpus, node_key=node, arm=arm,
        site_cross_check={"SITES": [26], "SITE_OF_RECORD": 26, "agrees": True},
        model_config_sha256="a" * 64, vector_npz_sha256="b" * 64,
        vector_fd_gate={"PASSES": True, "best_median_rel_error": 0.01},
        vector_build_stamp={"builder": "build_entropy_gradient.py"},
        transport_map={"fit_sha256": "c" * 64, "family": "proc_k128",
                       "arm": "native", "corpus_vintage": corpus},
        naive_row={"pair": "8b->qwen2.5-3b-instruct", "verdict": naive_verdict,
                   "bare_cos": 0.04, "q95": 0.09},
        trunk={"transformers": "5.3.0", "torch": "2.9.0", "hostname": "selftest",
               "cuda_device_name": "cpu", "driver": "n/a"},
        replay_gate_digests={"token_ids": "d" * 64, "entropy": "e" * 64},
        battery_item_set_sha256="f" * 64,
        actuation_calibration={"job_id": "selftest", "verdict": "PASS",
                               "spearman_rho": 0.94, "outside_band_doses": "5/6",
                               "coherence_at_scoring_dose": 0.62},
        per_cell_seed_roots=roots)


# ---------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The node-side behavioral engine (BRIEF §2). Heimdall CLI only; "
                    "the coordinator HTTP API is read-only verification.")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only, data-independent verification (no GPU, no weights)")
    ap.add_argument("--preflight", action="store_true",
                    help="M10: a first-class exit-early preflight mode, NEVER output "
                         "truncation. Resolves the pool, the site, the layout and "
                         "the hook admissibility, banks nothing, fires no cell.")
    ap.add_argument("--run", action="store_true",
                    help="the full §2.1 one-load-per-node job")
    ap.add_argument("--node-key", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--arm-root", type=Path, default=None)
    ap.add_argument("--work-root", type=Path, default=None,
                    help="node-side work root; all new node-side data lives under "
                         "/models/metabasis-behavioral/<node> (standing rule 2026-07-29)")
    ap.add_argument("--cells-json", type=Path, default=None,
                    help="staged cell specs (build_behavioral_banks.py output)")
    ap.add_argument("--prompt-pool", type=Path, default=None)
    ap.add_argument("--arm", choices=("native", "raw"), default="native")
    ap.add_argument("--site", type=int, default=None)
    ap.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.preflight or args.run:
        # The GPU job body is staged for the §3 step-2 certification enactment; this
        # module's step-1 deliverable is the mechanics + selftests (§3 step 1: "no
        # science cell fires"). Refusing loudly beats a half-wired run.
        missing = [f for f in ("node_key", "model_path", "prompt_pool")
                   if getattr(args, f) is None]
        if missing:
            ap.error(f"--{'/--'.join(m.replace('_', '-') for m in missing)} required")
        pool = load_prompt_pool(args.prompt_pool)
        logger.info("prompt pool of record: %d prompts, sha %s",
                    len(pool.prompts), pool.sha256)
        logger.info("dose ladder (FROZEN): %s; n/cell %d; max_new_tokens %d",
                    list(DOSE_LADDER), args.n_per_cell, MAX_NEW_TOKENS)
        logger.info("sampling of record: %s", SAMPLING_OF_RECORD.model_dump())
        B, measured, note = choose_batch_size(None)
        logger.info("canonical layout preflight: B=%d measured=%s (%s)",
                    B, measured, note)
        if args.run:
            ap.error(
                "--run is not wired in this build: BRIEF §3 step 1 is 'modules + "
                "selftests only; NO science cell fires', and the §3 step-2 "
                "certification run on the 3B is a separate later enactment against "
                "the FROZEN harness. Use --preflight, or --selftest.")
        return 0
    ap.error("nothing to do: pass --selftest, --preflight or --run")
    return 2                                        # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
