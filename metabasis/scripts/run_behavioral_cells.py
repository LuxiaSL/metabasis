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
battery and coherence panel live in `capability_battery.py`; §2's CPU staging is
`build_behavioral_banks.py` (whose `CellsDocument` is this module's `--run` input) and
scoring is `score_behavioral_column.py`, which is NOT in this batch — the desk scores
and this module never self-scores (C§8).

THE JOB IS WIRED, AND ITS ORCHESTRATION IS PROVED WITHOUT A GPU. `run_column` is the
§2.1 job; it is written against the narrow `NodeRuntime` surface so the SAME
orchestration runs under `HFNodeRuntime` on a node and under a numpy-only stub in
`--selftest`. That is what keeps the fire ORDER (calibration before transported,
because §4 is a gate), the per-cell §2.8 stamps, the completeness guard and the replay
gate proven in a configuration with no deep-learning stack — while the torch block
additionally drives the REAL runtime end to end on a tiny CPU Llama, so the hook, the
two-forward probe, the in-job norm measurement and the battery-under-injection are
exercised against a genuine forward rather than described.

BASIS-AGNOSTIC AT RUN TIME. `CORPUS_SHA_V21` is this module's DEFAULT expectation and
nothing more: `run_column` takes `corpus_sha_of_record` as a parameter and the CLI
defaults it to the cells document's own basis, so re-pointing the harness at another
corpus/basis is a staging act, never an edit here.

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

     TWO KERNELS, ONE STEP (2026-08-01). The per-generation python step is also the
     harness's hot path, so a VECTORIZED kernel computes the same step for a whole
     sub-batch at once (`sample_tokens`). `sample_token` remains in the module as the
     path of reference and the definition of the step; the vectorized kernel may not
     draw a token at a (batch, vocab) shape until it has been certified BYTE-identical
     to the reference at that shape, on that machine, in that process
     (`assert_batched_sampler_identity`). Every subcase whose batched spelling would
     be a different computation rather than a faster one — greedy, top-k, top-p, any
     degenerate row — is REFUSED back to the reference rather than approximated.

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

Node-side run (§2.1; Heimdall CLI only, the HTTP API is read-only verification):

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 \
    python -m metabasis.scripts.run_behavioral_cells --run \
        --node-key qwen2.5-3b-instruct --model-path <LOCAL_WEIGHTS_DIR> \
        --arm-root <ARM_ROOT> --work-root <NODE_DATA_ROOT>/metabasis-behavioral/<node> \
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
from typing import (Any, Callable, Collection, Literal, Optional, Protocol,
                    Sequence)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

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

#: The band families that ARE the null of record — the two §4.1 keeps distinct, and the
#: only two any gate population may contain.
GATE_BAND_FAMILIES: tuple[str, ...] = ("Rband", "gRband")
#: The BESIDE families (Luxia's B4 ruling, 2026-08-04, on the staging wave's blocker B4):
#: a Σ-shaped band is a STRICTER null quoted BESIDE the null of record, on the two
#: DESIGNATED cell tuples only (pre-statement §2 / open word O-2). Widening
#: `CellSpec.band_family` to admit it is the whole of that authorization — it does NOT
#: make a Σ cell a gate input. Every gate population in this file filters on
#: `GATE_BAND_FAMILIES` rather than on `is_null`, precisely so "never the null of
#: record, never inside any gate" is mechanical rather than a habit; the actuation
#: criteria in `actuation_calibration` filter narrower still (`Rband` only).
BESIDE_BAND_FAMILIES: tuple[str, ...] = ("SigmaBand",)
BAND_FAMILIES: tuple[str, ...] = GATE_BAND_FAMILIES + BESIDE_BAND_FAMILIES

#: The strata the §2.7 gate must cover, in order. Each contributes exactly one cell,
#: so K == 3 is not a tunable but the arity of this tuple.
REPLAY_GATE_STRATA: tuple[str, ...] = ("signal_at_0.3", "random_band", "calibration")

#: B-1, RULED (Luxia, 2026-08-05 ~03:00, session 12), quoted verbatim from the ledger
#: row: "**B-1 = the DESK READING RULED**: for a column with no transported cells the
#: §2.7 replay strata map to the column's own work-types — signal role = native EGV at
#: |0.3| · band role = the node's own Rband · calibration role = a small-dose cell;
#: dated interpretation, quoted in the gate code + a pre-statement addendum; the gate's
#: intent (one of each work-type, deterministic selection, bitwise replay) is
#: preserved."
#:
#: What the ruling is NOT: a second gate, a second stratum set, or a weaker gate.
#: `REPLAY_GATE_STRATA` and K == 3 are untouched, so a calibration-only column's gate
#: record has exactly the same shape and the same blocking force as a transported
#: column's; only WHICH cells fill the three roles changes, and only on a column that
#: has no transported cells at all. A column that cannot fill a mapped role still
#: raises `ExpectedNShortfall` — "incomplete, not exempt" is unchanged.
REPLAY_ROLE_MAPPING_OF_RECORD = "of_record"
REPLAY_ROLE_MAPPING_CALIBRATION_ONLY = "calibration_only"
#: The dose magnitude the |0.3| signal role reads at (`SCORING_DOSES` as a magnitude).
SCORING_DOSE_MAGNITUDE = 0.3
#: The enactor's reading of "a small-dose cell", recorded so the desk can rule
#: differently without hunting for the assumption: the calibration ROLE is filled by a
#: native-lever cell at a NON-SCORING, non-zero dose — |0.03| or |0.10| on the frozen
#: ladder. The named alternative (the ladder's SMALLEST magnitude, |0.03|, only) is
#: narrower and was NOT taken, because the role's job is "the lever below the scoring
#: dose" and both sub-scoring rungs are that; the choice is presence-independent
#: either way, so the selection stays deterministic from the digest alone.
CALIBRATION_ONLY_ROLE_READING = (
    "B-1 ruled (Luxia 2026-08-05, session 12): on a column with NO transported cells "
    "the three frozen §2.7 roles map to the column's own work-types — signal role = "
    "the native EGV cell at |0.3| · band role = the node's own Rband · calibration "
    "role = a small-dose native-lever cell (non-scoring, non-zero: |0.03| or |0.10| "
    "on the frozen ladder). Deterministic selection, bitwise replay and the blocking "
    "HALT are unchanged; a transported column never reaches this mapping.")
#: The kinds whose presence makes a column "transported" for B-1's trigger. `naive`
#: is in the set BY CONSTRUCTION: a naive cell carries a SOURCE object into a target
#: (ruling 5), so a column holding one is not a calibration-only column. `judged` is
#: in the set conservatively — a judged cell's provenance is not decidable from its
#: kind, and the conservative outcome is the of-record mapping's HALT (a report to the
#: desk), never a silently substituted role.
TRANSPORTED_CELL_KINDS: tuple[str, ...] = ("transported", "transported_band", "bridge",
                                           "naive", "judged")


# ---------------------------------------------------------------- error taxonomy
class BesideCellInGatePopulation(RuntimeError):
    """A BESIDE cell (Σ-shaped band) reached a population a gate reads.

    Not a §9 HALT of the brief's numbering — it is the mechanical expression of the
    B4 ruling's own condition ("the engine's gates must never read them as gate
    inputs"). It is an exception rather than a filter-and-continue because silently
    dropping a cell from a gate population is how a gate quietly changes arity.
    """


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


class VectorizedSamplerNotIdentical(BehavioralHarnessError):
    """The batched sampling kernel is not byte-identical to ruling 8's reference step.

    Not a tolerance and not a warning. The vectorized kernel exists ONLY as a faster
    spelling of `sample_token`; a platform on which it spells something else must not
    produce one token of record, so this fires BEFORE the first token of a sub-batch
    shape is drawn (`assert_batched_sampler_identity`) and stops the run for the desk.
    """


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
    #: §2.7's batch-invariance characterization asks for a cell re-run "at B=8 and
    #: B=1" — NEITHER of which is on §2.2's frozen ladder {80, 40, 20, 10}. The two
    #: clauses are both in the brief and both meant: the ladder governs the LAYOUT OF
    #: RECORD, the characterization deliberately steps OFF it to measure what stepping
    #: off costs. This flag is the named resolution: an off-ladder size is admissible
    #: ONLY on a characterization layout, and a characterization layout can never be
    #: FROZEN — so no cell of record can ever be produced at one.
    characterization_only: bool = False

    @model_validator(mode="after")
    def _ladder_governs_the_layout_of_record(self) -> "CanonicalLayout":
        if self.characterization_only:
            if self.frozen:
                raise ValueError(
                    "a characterization layout is never FROZEN: §2.7 defines replay IN "
                    "the canonical layout, and a frozen off-ladder layout would be a "
                    "second layout of record (§9 item 13's silent split)")
            if self.batch_size <= 0:
                raise ValueError("batch size must be positive")
            return self
        if self.batch_size not in BATCH_LADDER:
            raise ValueError(
                f"batch size {self.batch_size} is not on the frozen ladder "
                f"{BATCH_LADDER} (§2.2). §2.7's B=8/B=1 characterization is the one "
                "exception and must set `characterization_only=True`.")
        return self

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
    #: `SigmaBand` is admitted as a BESIDE family ONLY (B4, Luxia 2026-08-04): it
    #: schedules and generates like any other band cell and it is never a gate input.
    #: See `is_beside` / `is_null_of_record` and `BESIDE_BAND_FAMILIES`.
    band_family: Optional[Literal["Rband", "gRband", "SigmaBand"]] = None
    vector_npz: Optional[str] = None
    vector_provenance: str = ""
    sampling: SamplingConfig = SAMPLING_OF_RECORD

    @property
    def is_null(self) -> bool:
        """True for ANY band cell — a Σ-beside cell is a null draw, just not THE null.

        Kept deliberately broad so no consumer can mistake a Σ cell for signal (the
        capability battery's dose × metric table splits on exactly this, then splits
        the band again on `is_beside`). Gate populations use `is_null_of_record`.
        """
        return self.band_family is not None

    @property
    def is_beside(self) -> bool:
        """True iff this cell's band family is a BESIDE family (never a gate input)."""
        return self.band_family in BESIDE_BAND_FAMILIES

    @property
    def is_null_of_record(self) -> bool:
        """True iff this cell is a band of record (`Rband`/`gRband`) — the gate test."""
        return self.band_family in GATE_BAND_FAMILIES

    @property
    def is_baseline(self) -> bool:
        return self.kind == "baseline"

    @model_validator(mode="after")
    def _consistent(self) -> "CellSpec":
        if self.is_beside and self.kind not in ("calibration_band",
                                                "transported_band"):
            raise ValueError(
                f"{self.cell_id}: a BESIDE band cell (band_family="
                f"{self.band_family!r}) must be staged as a band cell "
                "('calibration_band' beside an Rband, 'transported_band' beside a "
                f"gRband), not as {self.kind!r} — a beside that wore a signal kind "
                "would be pooled as signal by every consumer that splits on kind")
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
    #: B-1 (ruled 2026-08-05): which role mapping filled the three frozen strata —
    #: `of_record` (the transported column's) or `calibration_only`. Defaulted so a
    #: record written before the ruling reads back as what it was.
    role_mapping: str = REPLAY_ROLE_MAPPING_OF_RECORD
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

    THE PATH OF REFERENCE. This function is the DEFINITION of the sampling step and is
    deliberately unchanged by the 2026-08-01 vectorization: `sample_tokens` computes
    the same step for a whole sub-batch and is certified byte-identical against THIS
    code before it may draw a token, and every subcase the batched kernel does not own
    lands back here. Change this body and the batched kernel becomes wrong by
    definition — which is the intended coupling.
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


# ------------------------------------------------- the batched sampling step (2026-08-01)
class SamplerKernel(BaseModel):
    """WHICH spelling of ruling 8's fixed sampling step draws the tokens.

    Two kernels, ONE step. `sample_token` above is the reference and stays in the
    module unchanged; `vectorized` computes the SAME step for a whole sub-batch in one
    set of numpy calls. The kernel is a stamped fact, never an optimization detail,
    because a reader who cannot tell which code drew a token cannot reproduce it —
    even when the two are proven to draw the same one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["reference", "vectorized"]
    provenance: str


SAMPLER_KERNEL_REFERENCE = SamplerKernel(
    name="reference",
    provenance="ruling 8's per-generation `sample_token`, one python call per row per "
               "step. The definition of the sampling step; kept as the path of "
               "reference so the vectorized kernel is measured against code rather "
               "than against a description of code.")
SAMPLER_KERNEL_VECTORIZED = SamplerKernel(
    name="vectorized",
    provenance="the batched kernel (Luxia 2026-08-01): the SAME float64 "
               "temperature→max-shift→exp→sum→normalize→cumsum→inverse-CDF, computed "
               "for a whole sub-batch at once. Byte-identical to `sample_token` by "
               "construction (no reduction crosses a row; every op is either exactly "
               "rounded or an exact selection) and certified so at each (batch, vocab) "
               "shape before that shape draws a token.")
#: The kernel of record. Both `generate_sub_batch` and the stamp read THIS object, so a
#: stamp can never name a kernel other than the one that ran.
SAMPLER_KERNEL_OF_RECORD = SAMPLER_KERNEL_VECTORIZED

#: The working set one tile of the batched kernel aims at, in bytes: a core's private
#: L2, which is where the reference's one-row-at-a-time chain already lived. MEASURED
#: on the desk CPU (8-core Zen3 class, 512 KiB L2 per core) across vocab 1000 / 50257
#: / 128256 — 512 KiB was at or within noise of the best tile at every one, while the
#: unbounded spelling (one [80, V] array per intermediate) came in at 0.75× at vocab
#: 128256, i.e. SLOWER than the reference. At vocab 128256 this is one row; at vocab
#: 1000 it is 64. A PERFORMANCE knob and nothing else: the tile size cannot move a
#: token (a row's table is a function of that row alone), and the per-shape
#: certification runs against whatever tiling it produces.
SAMPLER_TILE_BYTES = 512 << 10
#: The adversarial logit families the certification runs over — the DEGENERATE
#: distributions named as much as the realistic ones, because a softmax that agrees on
#: a plausible logits row and disagrees on a delta is not an identity. `all_nonfinite`
#: is deliberately outside both tuples: it asserts an EXCEPTION rather than a token,
#: which is a code property and shape-independent, so the selftest owns it.
CERTIFICATION_FAMILIES: tuple[str, ...] = (
    "normal", "wide", "one_hot", "uniform", "tiny_spread", "denormal_adjacent",
    "extreme_mixed")
#: What the run-time per-shape gate uses. Four families rather than seven because the
#: gate runs inside a job: these four span the float regimes that could plausibly
#: separate a SIMD lane from a scalar tail, and the selftest runs all seven.
CERTIFICATION_FAMILIES_RUNTIME: tuple[str, ...] = (
    "normal", "wide", "one_hot", "denormal_adjacent")
CERTIFICATION_SEED = 20260801

#: Certificates already earned in this process, keyed by (shape, sampling, families).
#: A certificate is per-PROCESS because what it certifies — this numpy build's loop
#: selection at this shape — is a property of the loaded binary, not of the job.
_SAMPLER_CERTIFICATES: dict[tuple, str] = {}


def vectorized_sampling_refusal(sampling: SamplingConfig) -> str:
    """"" if the batched kernel handles this config; else the NAMED reason it will not.

    REFUSAL, never approximation. The kernel implements exactly ruling 4's config of
    record (pure ancestral: sample, T free, no top-k, no top-p). Greedy and the
    top-k/top-p filters are `sample_token`'s branches — the battery's and the bridge
    cell's, both off the 80×512 hot path — and each one contains an operation whose
    batched spelling would be a DIFFERENT computation, not a faster one (`np.partition`
    picks a different pivot at another length; the stable argsort's tie order and the
    renormalized `mask.sum()` are length-dependent). Those subcases therefore go to the
    reference path and stay bit-exact by identity rather than by argument.
    """
    if not sampling.do_sample:
        return ("do_sample=False (greedy): `sample_token`'s argmax branch — the §6 "
                "battery's format probe, never a science cell")
    if sampling.top_k:
        return (f"top_k={sampling.top_k}: the np.partition/-inf filter is ruling 4's "
                "bridge-cell exception, not the config of record")
    if sampling.top_p < 1.0:
        return (f"top_p={sampling.top_p}: the stable-argsort renormalization is ruling "
                "4's bridge-cell exception, not the config of record")
    return ""


def sampler_tile_rows(vocab: int) -> int:
    """How many rows the batched kernel computes at once, at this vocab.

    THE MEASURED REASON THIS EXISTS. The obvious spelling — one [80, 128256] array per
    intermediate — measured 0.30× against the per-row reference, i.e. 3.3× SLOWER, and
    0.75× even after the redundant passes were removed. Each float64 intermediate is
    82 MB there, so the chain streams half a gigabyte per step and every pass starts
    from DRAM. Batching does not make elementwise float64 work cheaper; it amortizes
    per-call overhead, and at vocab 128256 that overhead is already noise. So the
    kernel batches by a TILE sized to stay cache-resident: the amortization is kept
    and the working set of a per-row chain is kept with it.

    A tile is rows, never columns: a row's table must remain a function of that row
    alone (§2.2), so a tile boundary can only ever fall between generations.
    """
    return max(1, int(SAMPLER_TILE_BYTES // max(1, int(vocab) * 8)))


def _tile_cdf(tile: np.ndarray, temperature: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """One tile of rows → ([T, V] float64 inverse-CDF table, [T] bool 'refused').

    THE IDENTITY ARGUMENT, op by op, against `sample_token`'s body. Every step below
    is either exactly rounded, an exact selection, or a provable no-op — the three
    ways an operation can be moved or dropped without moving a bit:

      * `x / T` when T == 1.0 is SKIPPED. `a / 1.0 == a` exactly for every double
        including ±0, ±inf, NaN and subnormals, so the reference's divide is the
        identity function and spelling it costs a full pass for nothing. (T != 1.0 is
        not the config of record, and there the divide is spelled.)
      * the row max is taken in the tile's OWN dtype (float32 out of both steppers).
        Max is an exact SELECTION and widening is order-preserving and injective, so
        the float32 max widened IS the float64 max — one pass over 4-byte elements
        instead of 8-byte ones.
      * `np.subtract(tile, m, dtype=np.float64)` fuses the widening cast into the
        max-shift: the ufunc casts to float64 and subtracts in float64, which is what
        `logits.astype(f64) - m` does, in one pass instead of two.
      * THE ALL-FINITE FAST PATH. If every row's plain max is finite then no row holds
        +inf or NaN (either would BE the max), so (a) the reference's finite-filtered
        max equals the plain max, and (b) every shifted value is ≤ 0 and finite, so
        `np.exp` returns (0, 1] and the reference's `p[~isfinite(p)] = 0.0` assigns to
        nothing. A no-op is not spelled. Rows holding -inf are still on this path:
        -inf shifts to -inf and exponentiates to exactly 0.0, in both kernels.
      * `np.exp` — the one transcendental, and the one op whose bits could in
        principle depend on an element's POSITION (a SIMD lane vs a scalar tail).
        Argued nowhere: certified per shape on the running machine before that shape
        draws a token (`assert_batched_sampler_identity`, gate 1).
      * `p.sum(axis=1)` — numpy's pairwise summation blocks by the length of the
        contiguous run being reduced, and both spellings reduce one contiguous run of
        exactly V float64 with stride 8, so the summation TREE is the same tree.
        Certified per shape all the same.
      * `np.cumsum(..., axis=1)` — `add.accumulate` is sequential in index order by
        definition, so the row's partial sums are the row's partial sums. Written
        `out=` into its own input, which changes where the bytes land and not what
        they are.
      * the draw itself stays in the caller — `np.searchsorted(cdf_row, u,
        side="right")`, the SAME call the reference makes. A batched comparison-count
        (`(cdf <= u).sum()`) would agree only while the cdf is nondecreasing, and
        `cdf[-1] = 1.0` can (legally, rarely) break that.

    No reduction crosses a row, so a row's table is a function of that row alone — the
    §2.2 layout-invariance property ruling 8 bought is preserved, not traded away.
    """
    src = tile if temperature == 1.0 else np.asarray(
        tile, dtype=np.float64) / temperature
    m = src.max(axis=1)
    if np.isfinite(m).all():                              # the fast path
        x = np.subtract(src, m[:, None], dtype=np.float64)
        np.exp(x, out=x)
        total = x.sum(axis=1)
        refused = ~(np.isfinite(total) & (total > 0.0))
        np.divide(x, np.where(refused, 1.0, total)[:, None], out=x)
        np.cumsum(x, axis=1, out=x)
        x[:, -1] = 1.0               # the same float32→float64 tail guard, per row
        return x, refused
    # THE GENERAL PATH: ±inf or NaN is present somewhere in the tile, so every guard
    # `sample_token` carries is spelled out, in its order.
    x = np.asarray(src, dtype=np.float64)
    finite = np.isfinite(x)
    has_finite = finite.any(axis=1)
    mm = np.max(x, axis=1, where=finite, initial=-np.inf)
    p = x - mm[:, None]
    np.exp(p, out=p)
    p[~np.isfinite(p)] = 0.0
    total = p.sum(axis=1)
    ok = has_finite & np.isfinite(total) & (total > 0.0)
    # A refused row's divisor is neutralized so its arithmetic cannot raise or warn on
    # the way past; its TOKEN is recomputed by `sample_token`, never read from here.
    np.divide(p, np.where(ok, total, 1.0)[:, None], out=p)
    np.cumsum(p, axis=1, out=p)
    p[:, -1] = 1.0
    return p, ~ok


def _batched_cdf(block: np.ndarray, sampling: SamplingConfig
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The whole block's inverse-CDF table, tile by tile.

    Materializing [b, V] is exactly what the hot path does NOT do (see
    `_sample_tokens_vectorized`); this exists for the certifier and the selftest,
    which need every row's table at once in order to compare it.
    """
    if block.ndim != 2:
        raise ValueError(f"_batched_cdf: expected [batch, vocab], got {block.shape}")
    b, v = int(block.shape[0]), int(block.shape[1])
    cdf = np.empty((b, v), dtype=np.float64)
    refused = np.zeros(b, dtype=bool)
    step = sampler_tile_rows(v)
    for s in range(0, b, step):
        e = min(s + step, b)
        cdf[s:e], refused[s:e] = _tile_cdf(block[s:e], sampling.temperature)
    return cdf, refused


def _sample_tokens_vectorized(block: np.ndarray, *, us: Sequence[float],
                              rows: Sequence[int], sampling: SamplingConfig
                              ) -> list[int]:
    """The batched draw, WITHOUT the certification gate (the certifier's own entry).

    The tile's table is consumed while it is still in cache and never accumulated
    into a [batch, vocab] result. A tile holding no active row is not computed at
    all — which costs a frozen row's neighbours nothing and moves no row's slot,
    because the tiling is a fixed function of the vocab and never of who is done.

    Rows the kernel does not own are not approximated: they are handed to
    `sample_token`, which either returns its argmax fall-back or raises its own
    ValueError — so a degenerate row behaves exactly as it always has, including in
    how it fails.
    """
    b, v = int(block.shape[0]), int(block.shape[1])
    step = sampler_tile_rows(v)
    at = {int(r): i for i, r in enumerate(rows)}
    out: list[int] = [0] * len(rows)
    for s in range(0, b, step):
        e = min(s + step, b)
        needed = [r for r in at if s <= r < e]
        if not needed:
            continue
        cdf, refused = _tile_cdf(block[s:e], sampling.temperature)
        for r in needed:
            i = at[r]
            u = float(us[i])
            out[i] = (sample_token(block[r], u, sampling) if refused[r - s]
                      else int(np.searchsorted(cdf[r - s], u, side="right")))
    return out


def sample_tokens(logits_block: Any, *, us: Sequence[float],
                  rows: Optional[Sequence[int]] = None,
                  sampling: SamplingConfig = SAMPLING_OF_RECORD,
                  kernel: SamplerKernel = SAMPLER_KERNEL_OF_RECORD) -> list[int]:
    """One step of a sub-batch: [b, V] logits + one uniform per ACTIVE row → tokens.

    `rows` are indices into the block (the rows not yet frozen by EOS) and `us` are
    their tape values, in the same order; the returned tokens are in that order too.
    A frozen row keeps its SLOT in the batch (§2.2) — the tiling is a function of the
    vocab alone and never of who has finished — but a tile holding no active row is
    not computed, so freezing costs its neighbours nothing.

    The vectorized kernel cannot draw a token at a (batch, vocab) shape until that
    shape has been certified against the reference ON THIS MACHINE, in this process.
    That is the structural guarantee: not "we tested it once", but "this shape is
    refused until it has been proven here".
    """
    block = np.asarray(logits_block)
    if block.ndim != 2:
        raise ValueError(f"sample_tokens: expected [batch, vocab] logits, got shape "
                         f"{block.shape}")
    idx = list(range(int(block.shape[0]))) if rows is None else [int(r) for r in rows]
    if len(idx) != len(us):
        raise ValueError(f"sample_tokens: {len(idx)} rows but {len(us)} uniforms")
    if not idx:
        return []
    if int(block.shape[1]) == 0:
        raise ValueError("empty logits row")
    if kernel.name == "reference" or vectorized_sampling_refusal(sampling):
        return [sample_token(block[r], float(u), sampling) for r, u in zip(idx, us)]
    assert_batched_sampler_identity(batch=int(block.shape[0]),
                                    vocab=int(block.shape[1]), sampling=sampling)
    return _sample_tokens_vectorized(block, us=us, rows=idx, sampling=sampling)


def certification_logits(batch: int, vocab: int, family: str,
                         rng: np.random.Generator) -> np.ndarray:
    """One adversarial [batch, vocab] float32 logits block, by NAMED float regime.

    float32 because that is what both steppers hand the sampler; the interesting
    arithmetic is what happens after the exact widening to float64.
    """
    if family == "normal":                       # a realistic logits row
        return (rng.standard_normal((batch, vocab)) * 3.0).astype(np.float32)
    if family == "wide":                         # peaked: most of the mass in a few ids
        return (rng.standard_normal((batch, vocab)) * 30.0).astype(np.float32)
    if family == "one_hot":                      # a delta — p is exactly {0, 1}
        blk = np.full((batch, vocab), -1e30, dtype=np.float32)
        blk[np.arange(batch), rng.integers(0, vocab, batch)] = 1e30
        return blk
    if family == "uniform":                      # the flat distribution, p == 1/V
        return np.zeros((batch, vocab), dtype=np.float32)
    if family == "tiny_spread":                  # flat to within an ulp of flat
        return (rng.standard_normal((batch, vocab)) * 1e-12).astype(np.float32)
    if family == "denormal_adjacent":
        # x - max lands in [-745, -708], so exp() returns SUBNORMALS and zeros — the
        # regime where a flush-to-zero difference between a SIMD lane and a scalar
        # tail would show up if there were one.
        blk = rng.uniform(-745.0, -708.0, size=(batch, vocab)).astype(np.float32)
        blk[:, 0] = 0.0
        return blk
    if family == "extreme_mixed":                # ±inf and NaN beside finite mass
        blk = (rng.standard_normal((batch, vocab)) * 3.0).astype(np.float32)
        if vocab >= 4:
            blk[:, 1] = np.inf
            blk[:, 2] = -np.inf
            blk[0, 3] = np.nan
        return blk
    if family == "all_nonfinite":                # every row refused; both paths RAISE
        return np.full((batch, vocab), np.nan, dtype=np.float32)
    raise ValueError(f"unknown certification family {family!r}")


def _certification_uniforms(cdf: np.ndarray, n_u: int,
                            rng: np.random.Generator) -> list[np.ndarray]:
    """Uniforms that sit exactly ON the inverse-CDF's breakpoints, plus random ones.

    A one-ulp disagreement between the two paths is invisible to a random u and
    certain to be visible to a u that equals a breakpoint, because `side="right"` is
    decided there. So the certification aims at the breakpoints.
    """
    b, v = cdf.shape
    draws: list[np.ndarray] = []
    for j in range(max(1, n_u)):
        if j == 0:
            k = rng.integers(0, v, b)
            u = cdf[np.arange(b), k]                       # exactly a breakpoint
        elif j == 1:
            k = rng.integers(0, v, b)
            u = np.nextafter(cdf[np.arange(b), k], -np.inf)  # one ulp below one
        else:
            u = rng.random(b)
        draws.append(np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0 - 2.0 ** -53))
    return draws


def _row_cdf(row: np.ndarray, sampling: SamplingConfig) -> np.ndarray:
    """`sample_token`'s arithmetic for ONE row, spelled out and stopping at the table.

    A spelling of the reference, not the reference: gate 1 uses it to compare whole
    tables (which `sample_token` does not return), and gate 2 anchors the same rows to
    `sample_token` itself. Kept beside `sample_token` in the file for the obvious
    reason — if that body ever changes, this is the second place to change, and the
    identity gate fails loudly until it is.
    """
    x = np.asarray(row, dtype=np.float64).reshape(-1) / sampling.temperature
    m = np.max(x[np.isfinite(x)])
    p = np.exp(x - m)
    p[~np.isfinite(p)] = 0.0
    cdf = np.cumsum(p / p.sum())
    cdf[-1] = 1.0
    return cdf


def _first_divergent_op(row: np.ndarray, tile: np.ndarray, at: int,
                        sampling: SamplingConfig) -> str:
    """NAME the operation whose bits moved. Diagnostics only — reached on failure."""
    xr = np.asarray(row, dtype=np.float64).reshape(-1) / sampling.temperature
    xt = np.asarray(tile, dtype=np.float64) / sampling.temperature
    mr = np.max(xr[np.isfinite(xr)])
    mt = np.max(xt, axis=1, where=np.isfinite(xt), initial=-np.inf)
    if np.float64(mr).tobytes() != np.float64(mt[at]).tobytes():
        return "the row max (an exact selection — this should be impossible)"
    sr = xr - mr
    st = (xt - mt[:, None])[at]
    if sr.tobytes() != st.tobytes():
        return "the max-shift (an exactly rounded subtract)"
    pr, pt = np.exp(sr), np.exp(st)
    if pr.tobytes() != pt.tobytes():
        return "np.exp (a SIMD lane vs a scalar tail — the position-dependent one)"
    if np.float64(pr.sum()).tobytes() != np.float64(pt.sum()).tobytes():
        return "np.add.reduce (the pairwise summation tree)"
    if np.cumsum(pr / pr.sum()).tobytes() != np.cumsum(pt / pt.sum()).tobytes():
        return "np.add.accumulate (the sequential prefix sum)"
    return "the inverse-CDF table (no single primitive reproduced it)"


def assert_batched_sampler_identity(
        *, batch: int, vocab: int, sampling: SamplingConfig = SAMPLING_OF_RECORD,
        families: Sequence[str] = CERTIFICATION_FAMILIES_RUNTIME, n_u: int = 2,
        seed: int = CERTIFICATION_SEED, force: bool = False) -> str:
    """Certify the batched kernel against `sample_token` AT THIS SHAPE, or HALT.

    Two gates, because they can fail for different reasons and only one of them is
    about this module's code:

      GATE 1 — THE WHOLE TABLE. For this exact (batch, vocab), the kernel's
        inverse-CDF table for a row is byte-identical to the row-wise spelling of
        `sample_token`'s arithmetic. Comparing the whole TABLE rather than sampled
        tokens is what makes this gate complete rather than lucky: equal cdf bytes
        means equal tokens for EVERY uniform in [0, 1), not for the ones that were
        tried. It is also the only step of the identity argument a machine can take
        away — `np.exp` is a polynomial in a SIMD lane and a libm call in a scalar
        tail, and nothing but a comparison on the running binary settles it. On a
        mismatch the intermediates are recomputed to NAME the operation that moved.
      GATE 2 — THE ANCHOR. The kernel's token equals the token drawn by the actual
        `sample_token`, at uniforms sitting exactly on the table's breakpoints. Gate 1
        compares against a spelling of the reference; gate 2 compares against the
        reference itself, so neither gate is checking its own homework.

    Returns a one-line certificate (also cached per process and per shape) so a run
    can say which shapes it proved rather than that it "checked sampling".
    """
    key = (int(batch), int(vocab), sampling.do_sample, float(sampling.temperature),
           float(sampling.top_p), int(sampling.top_k), tuple(families), int(n_u))
    cached = _SAMPLER_CERTIFICATES.get(key)
    if cached is not None and not force:
        return cached
    refusal = vectorized_sampling_refusal(sampling)
    if refusal:
        raise VectorizedSamplerNotIdentical(
            f"the batched kernel does not own this sampling config — {refusal}. It "
            "must never be certified for one it refuses; the caller dispatches such "
            "configs to `sample_token`.")
    if batch <= 0 or vocab <= 0:
        raise VectorizedSamplerNotIdentical(
            f"nothing to certify at shape ({batch}, {vocab})")
    rng = np.random.default_rng(seed)
    rows_checked = 0
    draws_checked = 0
    for family in families:
        blk32 = certification_logits(batch, vocab, family, rng)
        # ---- GATE 1: the kernel's table == the row-wise table, byte for byte -----
        cdf, refused = _batched_cdf(blk32, sampling)
        for r in range(batch):
            if refused[r]:
                continue
            row_cdf = _row_cdf(blk32[r], sampling)
            if row_cdf.tobytes() == cdf[r].tobytes():
                rows_checked += 1
                continue
            step = sampler_tile_rows(vocab)
            s_ = (r // step) * step
            moved = _first_divergent_op(blk32[r], blk32[s_:min(s_ + step, batch)],
                                        r - s_, sampling)
            raise VectorizedSamplerNotIdentical(
                f"GATE 1 at shape ({batch}, {vocab}), family {family!r}, row {r}: "
                f"{moved} is NOT byte-identical between the row-wise and tiled "
                "spellings on this numpy build. The batched kernel is refused here — "
                "ruling 8's step is defined by `sample_token`, and a faster spelling "
                "that rounds differently is a different experiment. Run with "
                "SAMPLER_KERNEL_REFERENCE and file the platform.")
        # ---- GATE 2: the kernel's token == `sample_token`'s token ----------------
        # The first uniform set goes through `_sample_tokens_vectorized` end to end;
        # the rest reuse the table gate 1 already built, because recomputing an
        # identical [batch, vocab] cdf per uniform set costs seconds at vocab 128256
        # and proves nothing the first pass did not.
        keep = [r for r in range(batch) if not refused[r]]
        for j, us in enumerate(_certification_uniforms(cdf, n_u, rng)):
            if j == 0:
                got = _sample_tokens_vectorized(
                    blk32, us=[float(us[r]) for r in keep], rows=keep,
                    sampling=sampling)
            else:
                got = [int(np.searchsorted(cdf[r], float(us[r]), side="right"))
                       for r in keep]
            want = [sample_token(blk32[r], float(us[r]), sampling) for r in keep]
            if got != want:
                bad = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
                raise VectorizedSamplerNotIdentical(
                    f"GATE 2 at shape ({batch}, {vocab}), family {family!r}: the "
                    f"batched kernel drew {got[bad]} where `sample_token` drew "
                    f"{want[bad]} for row {keep[bad]} at u={float(us[keep[bad]])!r}. "
                    "A mismatch is a FAILURE, never a tolerance.")
            draws_checked += len(keep)
    cert = (f"batched sampler certified at (batch={batch}, vocab={vocab}) vs "
            f"`sample_token`: {rows_checked} row(s) byte-identical through "
            f"max/shift/exp/sum/normalize/cumsum, {draws_checked} draw(s) "
            f"token-identical, families {list(families)}, numpy {np.__version__}")
    _SAMPLER_CERTIFICATES[key] = cert
    logger.info("%s", cert)
    return cert


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
    sampler_kernel: SamplerKernel = SAMPLER_KERNEL_OF_RECORD,
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

    `sampler_kernel` chooses which spelling of ruling 8's step draws the tokens. The
    two are certified byte-identical at the sub-batch's own (batch, vocab) shape
    before the first token is drawn, so this is a COST switch and never a semantic
    one — which is exactly why the certification is a gate and not a test: the switch
    is only ever allowed to be free.
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
        # Tape exhaustion is checked in ROW ORDER before any token is drawn, so the
        # same (row, step) raises under either kernel — a batched draw must not
        # change WHICH generation is named by a failure.
        active: list[int] = []
        for r in range(b):
            if done[r]:
                continue
            tape = tapes[r]
            if t >= tape.shape[0]:
                raise ValueError(
                    f"{cell.cell_id}/gen{gen_ids[r]}: uniform tape exhausted at "
                    f"step {t} (tape length {tape.shape[0]}, max_new_tokens "
                    f"{layout.max_new_tokens})")
            active.append(r)
        drawn = sample_tokens(logits, us=[float(tapes[r][t]) for r in active],
                              rows=active, sampling=cell.sampling,
                              kernel=sampler_kernel)
        for r, tok in zip(active, drawn):
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
    sampler_kernel: SamplerKernel = SAMPLER_KERNEL_OF_RECORD,
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
                start_pos_sink=start_pos_sink, sampler_kernel=sampler_kernel)
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
def _stratum_of(cell: CellSpec, *, calibration_only: bool = False) -> Optional[str]:
    # B4: a BESIDE cell holds NO stratum. Written as the first test rather than left
    # to fall through the family comparisons below, so that adding a stratum later
    # cannot accidentally admit one. It is the first test under BOTH role mappings.
    if cell.is_beside:
        return None
    if calibration_only:
        return _stratum_of_calibration_only(cell)
    if cell.kind in ("calibration", "calibration_band"):
        return "calibration" if cell.kind == "calibration" else None
    if cell.band_family == "gRband":
        return "random_band"
    if cell.kind in ("transported", "bridge") and abs(cell.alpha_frac) == 0.3:
        return "signal_at_0.3"
    return None


def _stratum_of_calibration_only(cell: CellSpec) -> Optional[str]:
    """B-1's RULED role mapping (`CALIBRATION_ONLY_ROLE_READING`), for one cell.

    Reached ONLY through `_stratum_of(..., calibration_only=True)`, which
    `select_replay_cells` engages only for a column that has no transported cells AND
    cannot constitute the gate under the mapping of record — so a transported column's
    selection is byte-identical to its selection before this ruling was implemented.
    """
    if cell.band_family == "Rband":              # band role: the node's OWN band
        return "random_band"
    if cell.kind == "calibration":               # the native lever, split by dose
        if abs(cell.alpha_frac) == SCORING_DOSE_MAGNITUDE:
            return "signal_at_0.3"               # signal role: native EGV at |0.3|
        if 0.0 < abs(cell.alpha_frac) < SCORING_DOSE_MAGNITUDE:
            return "calibration"                 # calibration role: a small-dose cell
    return None


def _bucket_by_stratum(cells: Sequence[CellSpec], *, calibration_only: bool
                       ) -> dict[str, list[str]]:
    """The per-stratum cell_id buckets under one role mapping (never sorted here)."""
    buckets: dict[str, list[str]] = {s: [] for s in REPLAY_GATE_STRATA}
    for c in cells:
        s = _stratum_of(c, calibration_only=calibration_only)
        if s is not None:
            buckets[s].append(c.cell_id)
    return buckets


def column_has_transported_cells(cells: Sequence[CellSpec]) -> bool:
    """True iff ANY cell in the set is transported-family (B-1's trigger, negated)."""
    return any(c.kind in TRANSPORTED_CELL_KINDS or c.band_family == "gRband"
               for c in cells)


def replay_gate_role_mapping(cells: Sequence[CellSpec]) -> str:
    """Which §2.7 role mapping this cell SET falls under (B-1, ruled 2026-08-05).

    The mapping of record wins whenever it CAN constitute the gate, so every column
    that worked before this ruling selects exactly what it selected before — the
    calibration-only mapping is unreachable for them by construction, not by care.
    A column that is short a stratum AND holds transported cells also keeps the
    mapping of record, so its shortfall still HALTs instead of being re-roled.
    """
    of_record = _bucket_by_stratum(cells, calibration_only=False)
    if all(of_record[s] for s in REPLAY_GATE_STRATA):
        return REPLAY_ROLE_MAPPING_OF_RECORD
    if column_has_transported_cells(cells):
        return REPLAY_ROLE_MAPPING_OF_RECORD
    return REPLAY_ROLE_MAPPING_CALIBRATION_ONLY


def select_replay_cells(cells: Sequence[CellSpec], node_key: str, corpus_sha: str
                        ) -> tuple[list[str], dict[str, str], str]:
    """§2.7's deterministic gate-cell selection.

    "selection deterministic from `sha256(node_key|corpus_sha)`, constrained to
    include one signal cell at |0.3|, one random-band cell, and one calibration
    cell" — so K == 3 is the arity of `REPLAY_GATE_STRATA`, and each stratum
    contributes exactly one cell chosen by the digest modulo the stratum's size.
    Cells are sorted by cell_id first, so the choice depends on the cell SET and the
    digest, never on the order a cells-json happened to list them in.

    B-1 (ruled 2026-08-05): a column with NO transported cells fills the same three
    frozen roles from its own work-types (`CALIBRATION_ONLY_ROLE_READING`). The
    mapping is chosen by `replay_gate_role_mapping`, which prefers the mapping of
    record whenever it can constitute the gate — so nothing about a transported
    column's selection, digest or strata changes.
    """
    digest = hashlib.sha256(f"{node_key}|{corpus_sha}".encode()).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    mapping = replay_gate_role_mapping(cells)
    buckets = _bucket_by_stratum(
        cells, calibration_only=(mapping == REPLAY_ROLE_MAPPING_CALIBRATION_ONLY))
    beside_ids = {c.cell_id for c in cells if c.is_beside}
    leaked = sorted(beside_ids.intersection(
        cid for ids in buckets.values() for cid in ids))
    if leaked:                                                    # pragma: no cover
        raise BesideCellInGatePopulation(
            f"{node_key}: BESIDE cell(s) {leaked} entered a §2.7 replay-gate stratum. "
            "B4 admits Σ-shaped bands as besides ONLY; a beside inside the blocking "
            "gate would make the gate's population depend on a designation.")
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
            f"one of each is incomplete, not exempt. "
            f"[role mapping: {mapping}"
            + (f" — {CALIBRATION_ONLY_ROLE_READING}]"
               if mapping == REPLAY_ROLE_MAPPING_CALIBRATION_ONLY else "]"))
    return [chosen[s] for s in REPLAY_GATE_STRATA], chosen, digest


def evaluate_replay_gate(*, selection: list[str], strata: dict[str, str],
                         digest: str,
                         token_first: dict[str, str], token_replay: dict[str, str],
                         entropy_first: dict[str, str],
                         entropy_replay: dict[str, str],
                         role_mapping: str = REPLAY_ROLE_MAPPING_OF_RECORD
                         ) -> ReplayGateResult:
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
                       "the stratum's cell_id-sorted size (§2.7)"
                       + ("" if role_mapping == REPLAY_ROLE_MAPPING_OF_RECORD
                          else f" — ROLE MAPPING: {CALIBRATION_ONLY_ROLE_READING}"),
        role_mapping=role_mapping,
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
    `date_string`, and the same MAPPING unwrap) so a behavioral prompt and a
    collected text are tokenized under the same arm rules. §5.1's frame is the
    BARE SYSTEM PROMPT — the standard entropy frame, frozen — so nothing is added
    here that the pool does not carry.

    ⚠ THE MAPPING UNWRAP IS NOT COSMETIC. `apply_chat_template` returns a bare id
    list for some tokenizers and a `BatchEncoding` for others — on transformers
    5.3.0 the Llama-3.2-3B tokenizer returns the mapping and the Qwen2.5-3B one
    returns the list, so the two shapes appear on ONE roster at ONE version.
    Iterating the mapping yields its KEYS, and `int("input_ids")` is where that
    surfaced (3B certification, 2026-08-05). `build_ids` — the collection lane of
    record — has always carried
        `list(res["input_ids"] if hasattr(res, "keys") else res)`
    and this function's own docstring claimed to mirror it while omitting exactly
    that line. The unwrap is restored here in the ancestor's form, so a behavioral
    prompt and a collected text tokenize identically on every tokenizer in the
    roster rather than on the subset that happens to return a list.
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
    if hasattr(ids, "keys"):                      # BatchEncoding / dict-like
        ids = ids["input_ids"]
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

#: B-2, RULED (Luxia, 2026-08-05 ~03:00, session 12), quoted verbatim from the ledger
#: row: "**B-2 = ADOPTED**: `naive` joins the nullable-kind set for EXACTLY the four
#: transport_map_* stamp fields; the naive gate row/verdict fields stay required; a
#: selftest must prove a naive cell missing its gate row still HALTs."
#:
#: WHY the four and only the four: a naive cell is the ruling-5 null — the SOURCE's
#: object dropped into the target's site with NO map applied — so there is no fit sha,
#: no family, no arm and no map vintage to name, by construction. The naive-transplant
#: gate row and its verdict are the opposite case: a naive cell is exactly the cell
#: that gate was written for (§5.3 item 1 / §9 item 7), so they stay REQUIRED and a
#: naive cell that lost them still HALTs.
NAIVE_NULLABLE_FIELDS: frozenset[str] = frozenset({
    "transport_map_fit_sha256", "transport_map_family", "transport_map_arm",
    "transport_map_corpus_vintage",
})
#: The nullable-kind table, read by `assert_stamp_complete`. Every other kind gets the
#: empty set — "null" stays a RULED state for named (kind, field) pairs and nothing
#: else, which is the property the whole checklist rests on.
NULLABLE_FIELDS_BY_KIND: dict[str, frozenset[str]] = {
    "calibration": CALIBRATION_NULLABLE_FIELDS,
    "calibration_band": CALIBRATION_NULLABLE_FIELDS,
    "baseline": CALIBRATION_NULLABLE_FIELDS,
    "naive": NAIVE_NULLABLE_FIELDS,
}


def assert_stamp_complete(stamp: dict, *, cell_kind: Optional[CellKind] = None
                          ) -> None:
    """§2.8 as a checklist assertion; §9 item 9 as its HALT.

    Two ways to fail, kept apart in the message because they demand different
    responses: a MISSING/NULL field is an OWED artifact (§10), and an unset CVD is
    the M10 rogue-run tell (a scheduler job carries a card index; a rogue carries
    the sentinel).

    A null is legitimate only for a (kind, field) pair named in
    `NULLABLE_FIELDS_BY_KIND` — the calibration half's six (no pair, so no map and no
    naive row) and, since B-2 was ruled 2026-08-05, a `naive` cell's four
    `transport_map_*` fields and NOTHING else. A MISSING field is refused for every
    kind: `nullable` widens what may be null, never what may be absent.
    """
    nullable = NULLABLE_FIELDS_BY_KIND.get(cell_kind or "", frozenset())
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
                sampler_kernel: SamplerKernel = SAMPLER_KERNEL_OF_RECORD,
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
        # B4, written into every stamp so a beside can never be read back as the null
        # of record by anything downstream that only has the stamp.
        "is_null_of_record": cell.is_null_of_record,
        "is_beside_only": cell.is_beside,
        "beside_discipline": (
            "BESIDE families " + ", ".join(BESIDE_BAND_FAMILIES) + " are a stricter "
            "null quoted BESIDE the null of record on DESIGNATED cell tuples only "
            "(pre-statement §2 / O-2; Luxia's B4 ruling 2026-08-04). A beside cell "
            "schedules and generates like any other cell and enters NO gate: not the "
            "§2.7 replay gate, not the §4 actuation criteria, not the §7 expected-N "
            "of record."),
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
        # WHICH code drew the tokens. The two kernels are certified byte-identical
        # before either draws one, so this can never explain a difference in the
        # numbers — which is precisely why it has to be on the record rather than
        # inferred from a commit date.
        "sampler_kernel": sampler_kernel.model_dump(),
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
    position's row block, float32 on CPU, because the sampling step does its
    arithmetic in float64 numpy on one row — which is what makes it layout-invariant
    (see `sample_token`).
    """

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        import torch

        self.model = model
        self.torch = torch
        self.device = device or str(next(model.parameters()).device)
        self._past: Any = None
        self._len = 0

    def _logits(self, out: Any) -> np.ndarray:
        """[batch, vocab] float32 numpy at the LAST position — one D2H per step.

        The widening to float32 happens AFTER the transfer, not before. On a bf16 or
        fp16 node that halves the per-step device→host traffic (at B=80, V=128256:
        41 MB → 20 MB, ×512 steps × 25 cells), and it cannot move a bit: widening a
        bf16/fp16 to float32 is exact in IEEE-754, so the numbers the sampler sees
        are the same numbers either way. On an fp32 model the two orders are the same
        call. `.contiguous()` before `.cpu()` keeps the copy one packed block rather
        than a strided gather across the prefill's [batch, seq, vocab] logits.
        """
        return (out.logits[:, -1, :].detach().contiguous().cpu()
                .to(self.torch.float32).numpy())

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


# ---------------------------------------------------------------- the column job (§2.1)
#: §2.1's fire order, as data. Calibration runs BEFORE anything transported because §4
#: is a GATE and not a warm-up: a node whose site fails it gets no transported-write
#: cell there, and a column that fired the transported half first would have spent the
#: budget before the gate could refuse it.
CELL_KIND_ORDER: tuple[CellKind, ...] = (
    "baseline", "calibration", "calibration_band", "transported", "transported_band",
    "naive", "bridge", "judged")

#: §2.7's descriptive batch-invariance sizes. Both are OFF the frozen ladder, on
#: purpose — see `CanonicalLayout.characterization_only`.
CHARACTERIZATION_BATCH_SIZES: tuple[int, ...] = (8, 1)

#: How a cell's entropy digest is composed (§2.7 says "the probe's per-position
#: float32 entropy arrays" without fixing an order; an unnamed order would make the
#: digest unreproducible by anyone but this file).
ENTROPY_DIGEST_ORDER = (
    "sha256 over float32 .tobytes() of: every generation's STEERED per-position entropy "
    "array in gen_id order, followed by every generation's UNSTEERED array in gen_id "
    "order. Both halves are in the digest because the rise is their difference and a "
    "gate on only one half could pass while the read moved.")

#: §6(1)/§6(2): the battery's own injection span. The brief fixes the hook, the α and
#: the site but not the SPAN, and the span is what makes the read a capability-under-
#: dose rather than a model fact — so it is named here rather than assumed.
BATTERY_INJECTION_SPAN = (
    "the battery injects over the SCORED span only — the continuation positions of a "
    "likelihood item, the generated positions of a format item — mirroring §2.4's "
    "'generated positions only' for the column's own cells. A battery injected over "
    "the item's prompt would measure a different intervention from the one the cell's "
    "entropy read measures.")

#: The battery's format probe is GREEDY. §6 wants the battery deterministic ("no
#: sampling noise on top of the effect being measured") and only the likelihood half is
#: deterministic by construction, so the generated half is pinned rather than sampled.
BATTERY_SAMPLING = SamplingConfig(
    name="greedy", do_sample=False, provenance=(
        "§6: the battery must add no sampling variance to the effect being measured; "
        "the likelihood half is exactly deterministic by construction and the format "
        "half is pinned greedy to match. NOT the column's sampling of record (ruling "
        "4's pure ancestral), and never used for a science cell."))


class NodeRuntime(Protocol):
    """Everything the column job needs from a loaded node — and nothing more.

    The orchestration (`run_column`) is the part that has to be right and the part no
    GPU can be spared to test, so it is written against this narrow surface: the SAME
    orchestration runs under `HFNodeRuntime` on a node and under a numpy-only stub in
    `--selftest`, which is how the §2.1 order, the stamps, the completeness guard and
    the replay gate stay proven in a configuration with no deep-learning stack at all.
    """

    def trunk(self) -> dict:
        """§2.8's trunk facts: transformers/torch versions, node, GPU model, driver."""

    def model_config_sha256(self) -> Optional[str]:
        """§2.8's M9 drift anchor."""

    def tokenize(self, prompt: BehavioralPrompt) -> list[int]:
        ...

    def pad_token_id(self) -> int:
        ...

    def eos_token_id(self) -> Optional[int]:
        ...

    def position_ceiling(self) -> Optional[int]:
        """A learned-position ceiling (gpt2-xl: 1024), or None."""

    def vram_probe(self, batch_size: int) -> float:
        """Fraction of device memory used at `batch_size` and max sequence length."""

    def measure_per_token_median_resid_norm(
            self, *, site: int, prompt_ids: Sequence[Sequence[int]]) -> float:
        """§2.5's in-job measurement — the convention that sets α."""

    def begin_cell(self, *, cell: CellSpec, alpha: float) -> None:
        """Attach/point the injection for one cell (one hook, mutated per cell)."""

    def start_pos_sink(self, padded_prompt_length: int) -> None:
        """Receive each sub-batch's padded prompt length (the live `start_pos`)."""

    def stepper(self) -> Stepper:
        ...

    def probe(self, records: Sequence[GenerationRecord], *, steered: bool
              ) -> list[tuple[np.ndarray, np.ndarray]]:
        """(entropy, nll) float32 arrays per record, right-padded and masked (§2.6)."""

    def decode(self, ids: Sequence[int]) -> str:
        ...

    def likelihood_nll(self) -> Callable[[str, str], float]:
        """NLL(prompt, continuation) UNDER the current cell's injection (§6(1))."""

    def format_probe_texts(self) -> dict[str, str]:
        """item_id → generated text under the current cell's injection (§6(2))."""

    def alpha_zero_equivalence(self) -> Optional[tuple[Any, Any]]:
        """(hooked-at-α0 logits, unhooked logits), or None if not measurable."""

    def hook_admissibility(self, site: int) -> Optional[HookAdmissibility]:
        """§4.4's preflight, or None on an architecture already known to pass."""

    def end_cell(self) -> None:
        ...

    def close(self) -> None:
        ...


class CellOutcome(BaseModel):
    """One completed cell: its read, its digests, and where its raw sits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str
    kind: CellKind
    alpha_frac: float
    alpha: float
    n: int
    tokens_generated: int
    elapsed_s: float
    entropy: dict
    probe_rows: list[ProbeRow]
    token_id_sha256: str
    entropy_array_sha256: str
    capability: Optional[dict] = None
    coherence: Optional[dict] = None
    stamp: dict = Field(default_factory=dict)
    raw_paths: dict[str, str] = Field(default_factory=dict)


class ColumnResult(BaseModel):
    """The whole node job: layout, norms, cells, gate, table, custody (§2.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    arm: str
    site: int
    corpus_manifest_sha256: str
    label: str = ""
    layout: CanonicalLayout
    norms: NormConventions
    site_cross_check: dict
    hook_admissibility: Optional[HookAdmissibility] = None
    alpha_zero_is_no_hook: Optional[bool] = None
    cells: list[CellOutcome]
    replay_gate: ReplayGateResult
    batch_invariance: Optional[BatchInvarianceCharacterization] = None
    fresh_process_replay: dict = Field(default_factory=dict)
    capability_table: Optional[dict] = None
    n_generations: int
    work_root: Optional[str] = None
    manifest: dict = Field(default_factory=dict)
    grade: str = GRADE_LINE
    brief_of_record: str = BRIEF_OF_RECORD
    brief_sha256: str = BRIEF_SHA256


def trunk_stamp(runtime: NodeRuntime) -> dict:
    """§2.8's trunk facts with M19 framing: a degraded read is described, not dropped."""
    try:
        trunk = dict(runtime.trunk())
    except Exception as exc:                             # noqa: BLE001 — M19(a)
        logger.warning("trunk facts unavailable (%s: %s) — recorded as degraded",
                       type(exc).__name__, exc)
        return {"measured": False,
                "note": f"trunk instrumentation degraded ({type(exc).__name__}: {exc}); "
                        "M19: the consumer blocks on this bound rather than on a hope"}
    trunk.setdefault("measured", True)
    return trunk


def site_cross_check(node_key: str, site: int) -> dict:
    """§2.8's `SITES` ∧ `SITE_OF_RECORD` cross-check, re-derived at run time.

    Delegates to the staging module so the two halves of the campaign cannot disagree
    about what "the site of record" means. Imported inside the function because the
    staging module imports THIS one (the engine owns the primitives, the staging module
    owns the composition), and because §4.3 forbids trusting a module-level snapshot of
    registries that change nightly.
    """
    from metabasis.scripts.build_behavioral_banks import site_cross_check as _cross
    return _cross(node_key, site)


def order_cells(cells: Sequence[CellSpec]) -> list[CellSpec]:
    """§2.1's fire order: calibration first, then transported, naive last of the reads."""
    rank = {k: i for i, k in enumerate(CELL_KIND_ORDER)}
    return sorted(cells, key=lambda c: (rank.get(c.kind, len(rank)),
                                        c.alpha_frac, c.cell_id))


def assert_column_complete(outcomes: Sequence[CellOutcome], expected_cells: int,
                           *, n_per_cell: int, node_key: str = "",
                           beside_cell_ids: Collection[str] = ()) -> None:
    """§7's completeness guard / §9 item 10 — a count one short is a rake (M23).

    `beside_cell_ids` names the BESIDE cells (Σ-shaped bands, B4) among the planned
    set. Expected-N is then asserted TWICE — on the whole column and on the column
    MINUS its besides — so a Σ cell can never stand in for a missing cell of record.
    Without the second assertion a designated row could lose a gRband dose and still
    count complete, which is exactly the substitution "never a gate input" forbids.
    """
    if len(outcomes) != expected_cells:
        raise ExpectedNShortfall(
            f"{node_key or 'column'}: {len(outcomes)} cells completed, {expected_cells} "
            "planned (§9 item 10 / M23: a count one short is a rake, not a rounding)")
    beside = set(beside_cell_ids)
    completed = {o.cell_id for o in outcomes}
    missing_beside = sorted(beside - completed)
    if missing_beside:
        raise ExpectedNShortfall(
            f"{node_key or 'column'}: BESIDE cell(s) {missing_beside} were planned "
            "and did not complete, yet the total came out right — which means a cell "
            "of record ran in a beside's place. B4: the two populations are counted "
            "separately precisely so that substitution cannot pass.")
    # A cell that STAMPED itself a beside but was never planned as one is the same
    # substitution from the other side, and the stamp is the only place the fact
    # survives into `CellOutcome`.
    undeclared = sorted(o.cell_id for o in outcomes
                        if (o.stamp or {}).get("is_beside_only") is True
                        and o.cell_id not in beside)
    if undeclared:
        raise ExpectedNShortfall(
            f"{node_key or 'column'}: cell(s) {undeclared} stamped themselves BESIDE "
            "but are not in the planned beside set — an undeclared beside inside the "
            "expected-N of record (B4).")
    of_record, expected_of_record = len(outcomes) - len(beside), (
        expected_cells - len(beside))
    if of_record != expected_of_record:                           # pragma: no cover
        raise ExpectedNShortfall(
            f"{node_key or 'column'}: {of_record} cells OF RECORD completed, "
            f"{expected_of_record} planned ({len(beside)} BESIDE cells declared). "
            "B4: a Σ-beside cell never counts toward the expected-N of the column "
            "of record.")
    short = [o.cell_id for o in outcomes if o.n != n_per_cell]
    if short:
        raise ExpectedNShortfall(
            f"{node_key or 'column'}: cell(s) {short} are short of n={n_per_cell} "
            "(§7's expected-N completeness guard)")


def cell_entropy_arrays(rows: Sequence[GenerationRecord],
                        steered: dict[int, np.ndarray],
                        unsteered: dict[int, np.ndarray]) -> list[np.ndarray]:
    """The digest's array sequence, in the order `ENTROPY_DIGEST_ORDER` names."""
    order = sorted(r.generation_id for r in rows)
    return ([steered[g] for g in order] + [unsteered[g] for g in order])


def probe_cell(runtime: NodeRuntime, records: Sequence[GenerationRecord], *,
               layout: CanonicalLayout
               ) -> tuple[list[ProbeRow], dict[int, np.ndarray],
                          dict[int, np.ndarray]]:
    """§2.6's batched two-forward probe over one cell, under `PROBE_GROUPING_READING`.

    Steered and unsteered run over the SAME token ids by construction (the records'
    `input_ids`), so the rise is a difference of two reads of one text, never of two
    texts. Padding cannot enter a mean: each row's span is sliced by its own
    `prompt_length` and its own generated count, inside a group where every row's
    prompt length is equal.
    """
    groups = probe_groups(records, max_group=layout.probe_batch_size)
    rows: list[ProbeRow] = []
    ent_s: dict[int, np.ndarray] = {}
    ent_u: dict[int, np.ndarray] = {}
    for gi, group in enumerate(groups):
        subset = [records[i] for i in group]
        steered = runtime.probe(subset, steered=True)
        unsteered = runtime.probe(subset, steered=False)
        if not (len(steered) == len(unsteered) == len(subset)):
            raise BehavioralHarnessError(
                f"probe returned {len(steered)}/{len(unsteered)} rows for a group of "
                f"{len(subset)} — a probe that drops rows would silently shorten a "
                "cell's read (§9 item 10)")
        for rec, (es, _), (eu, nu) in zip(subset, steered, unsteered):
            es = np.asarray(es, dtype=np.float32)
            eu = np.asarray(eu, dtype=np.float32)
            if es.shape != eu.shape or es.size == 0:
                raise BehavioralHarnessError(
                    f"{rec.cell_id}/gen{rec.generation_id}: steered/unsteered entropy "
                    f"arrays disagree in shape ({es.shape} vs {eu.shape}) or are "
                    "empty — the rise is their difference and cannot be formed")
            ent_s[rec.generation_id] = es
            ent_u[rec.generation_id] = eu
            rows.append(ProbeRow(
                generation_id=rec.generation_id, n_positions=int(es.size),
                mean_entropy_steered=float(np.mean(es)),
                mean_entropy_unsteered=float(np.mean(eu)),
                entropy_rise=float(np.mean(es) - np.mean(eu)),
                base_model_nll=float(np.mean(np.asarray(nu, dtype=np.float32))),
                probe_group=gi, probe_group_size=len(subset)))
    rows.sort(key=lambda r: r.generation_id)
    return rows, ent_s, ent_u


def run_cell(runtime: NodeRuntime, cell: CellSpec, *, alpha: float, pool: PromptPool,
             layout: CanonicalLayout, corpus_sha: str, node_key: str, arm: str,
             n: int, with_battery: bool = True,
             baseline_block: Optional[Any] = None) -> tuple[
                 list[GenerationRecord], list[ProbeRow], dict[int, np.ndarray],
                 dict[int, np.ndarray], Optional[Any], Optional[Any], float, int]:
    """One cell end to end: generate → probe → battery → panel.

    Returns everything the caller needs to file the cell and to replay it, rather than
    filing here: the replay gate re-runs this exact function and compares digests, so
    it must be free of side effects on disk.
    """
    import time

    from metabasis.scripts.capability_battery import (LIKELIHOOD_SUBTESTS,
                                                      capability_block,
                                                      coherence_panel,
                                                      score_format_probe,
                                                      score_likelihood_subtest)

    t0 = time.monotonic()
    runtime.begin_cell(cell=cell, alpha=alpha)
    try:
        records = generate_cell(
            runtime.stepper, cell=cell, pool=pool, tokenize=runtime.tokenize,
            layout=layout, pad_token_id=runtime.pad_token_id(),
            eos_token_id=runtime.eos_token_id(), corpus_sha=corpus_sha,
            node_key=node_key, arm=arm, start_pos_sink=runtime.start_pos_sink, n=n)
        rows, ent_s, ent_u = probe_cell(runtime, records, layout=layout)
        block = None
        panel = None
        if with_battery:
            # §6 rides EVERY dose cell, including the bands — that is the point: the
            # band rows at the same dose are the dose-matched capability floor.
            texts = [runtime.decode(r.generated_ids) for r in records]
            panel = coherence_panel(texts, [r.finished_with_eos for r in records])
            nll_of = runtime.likelihood_nll()
            likelihood = [score_likelihood_subtest(s, nll_of)
                          for s in LIKELIHOOD_SUBTESTS]
            fmt = score_format_probe(runtime.format_probe_texts())
            block = capability_block(cell=cell, likelihood=likelihood,
                                     format_result=fmt, panel=panel,
                                     baseline=baseline_block)
    finally:
        runtime.end_cell()
    elapsed = time.monotonic() - t0
    tokens = sum(len(r.generated_ids) for r in records)
    # M22: "the job is running" is not evidence of progress — the per-cell line IS the
    # progress record, and it carries rc/tokens/elapsed exactly as §8 requires.
    logger.info("CELL %s rc=0 tokens_generated=%d elapsed=%.1fs n=%d",
                cell.cell_id, tokens, elapsed, len(records))
    return records, rows, ent_s, ent_u, block, panel, elapsed, tokens


def characterize_layout_invariance(runtime: NodeRuntime, cell: CellSpec, *,
                                   alpha: float, pool: PromptPool,
                                   layout: CanonicalLayout, corpus_sha: str,
                                   node_key: str, arm: str, n: int
                                   ) -> BatchInvarianceCharacterization:
    """§2.7's DESCRIPTIVE B=8/B=1 re-run. M19: it cannot fail a gate.

    GPU reductions are not batch-size invariant; a divergence here is EXPECTED and is
    documented once per node so nobody later reads it as a defect. Any failure of the
    instrumentation itself degrades to `measured=false` rather than taking the column
    down with it — the characterization is not allowed to be the thing that stops a
    job whose gate it cannot fail.
    """
    reference: list[np.ndarray] = []
    comparisons: dict[int, list[np.ndarray]] = {}
    measured = True
    try:
        ref, *_ = run_cell(runtime, cell, alpha=alpha, pool=pool, layout=layout,
                           corpus_sha=corpus_sha, node_key=node_key, arm=arm, n=n,
                           with_battery=False)
        reference = [np.asarray(r.generated_ids, dtype=np.int64)
                     for r in sorted(ref, key=lambda x: x.generation_id)]
        for B in CHARACTERIZATION_BATCH_SIZES:
            alt_layout = layout.model_copy(update={
                "batch_size": B, "frozen": False, "characterization_only": True})
            alt, *_ = run_cell(runtime, cell, alpha=alpha, pool=pool,
                               layout=alt_layout, corpus_sha=corpus_sha,
                               node_key=node_key, arm=arm, n=n, with_battery=False)
            comparisons[B] = [np.asarray(r.generated_ids, dtype=np.int64)
                              for r in sorted(alt, key=lambda x: x.generation_id)]
    except Exception as exc:                             # noqa: BLE001 — M19(a)
        logger.warning("batch-invariance characterization degraded (%s: %s) — "
                       "recorded as measured=false; it cannot fail a gate (M19)",
                       type(exc).__name__, exc)
        measured = False
    return characterize_probe_batch_invariance(
        reference, {b: v for b, v in comparisons.items()}, layout.batch_size,
        measured=measured)


def run_column(runtime: NodeRuntime, *, doc: Any, pool: PromptPool,
               work_root: Optional[Path] = None,
               corpus_sha_of_record: str = CORPUS_SHA_V21,
               n_per_cell: Optional[int] = None,
               actuation_calibration_stamp: Optional[dict] = None,
               characterize: bool = True, write: bool = True,
               scheduler_card_index: Optional[str] = None) -> ColumnResult:
    """§2.1's one-load-per-node job, in order, with every §9 HALT live.

    preflight → norm calibration → canonical-layout determination → all calibration
    cells → all transported cells → battery → entropy probe → in-job replay gate →
    stamps → manifest. The model is loaded exactly once, by the caller, and this
    function never loads or reloads it.

    `doc` is a `build_behavioral_banks.CellsDocument`. Everything basis-shaped — the
    corpus sha, the vector paths, the map's vintage — arrives through it, so this
    function is as basis-agnostic as the document it is handed;
    `corpus_sha_of_record` is the expectation the chain is asserted AGAINST and is a
    parameter, not a constant read from this module.
    """
    node_key, arm, site = doc.node_key, doc.arm, doc.site
    n = int(n_per_cell or doc.n_per_cell)

    # ---- preflight (M10: a first-class phase, never output truncation) --------
    corpus_sha = assert_corpus_vintage(
        *[v for v in doc.vintage_links() if v is not None],
        expected=corpus_sha_of_record) if any(
            v is not None for v in doc.vintage_links()) else assert_corpus_vintage(
                doc.corpus_manifest_sha256, expected=corpus_sha_of_record)
    cross = site_cross_check(node_key, site)
    if pool.sha256 and doc.prompt_pool_sha256 and pool.sha256 != doc.prompt_pool_sha256:
        raise ArtifactShaMismatch(
            f"prompt pool sha {pool.sha256[:12]}… does not match the staged document's "
            f"{doc.prompt_pool_sha256[:12]}… (M4). The pool of record is sha-frozen "
            "(ruling 10) and a cell generated against another pool is another read.")
    prompt_ids = [runtime.tokenize(p) for p in pool.prompts]
    assert_position_budget(max(len(i) for i in prompt_ids), doc.max_new_tokens,
                           runtime.position_ceiling(), node_key)
    admissibility = runtime.hook_admissibility(site)
    B, measured, note = choose_batch_size(runtime.vram_probe)
    trunk = trunk_stamp(runtime)
    layout = freeze_layout(B, dtype=str(trunk.get("dtype", "bfloat16")),
                           measured=measured, headroom_note=note,
                           max_new_tokens=doc.max_new_tokens)
    alpha0_ok: Optional[bool] = None
    equivalence = runtime.alpha_zero_equivalence()
    if equivalence is not None:
        assert_alpha_zero_is_no_hook(*equivalence)          # raises on mismatch
        alpha0_ok = True

    # ---- norm calibration (§2.5) ---------------------------------------------
    sample = prompt_ids[:NORM_SAMPLE_SIZE]
    measured_norm = float(runtime.measure_per_token_median_resid_norm(
        site=site, prompt_ids=sample))
    norms = resolve_norms(
        site=site, measured=measured_norm,
        measured_provenance=(
            f"MEASURED in-job over a fixed {len(sample)}-prompt sample of the "
            f"behavioral pool (sha {pool.sha256[:12]}…), per-token residual norms at "
            f"L{site}, median (§2.5)"),
        banked_per_token=doc.banked_norms.get("per_token_median"),
        banked_provenance=doc.banked_norm_provenance or None,
        banked_mean_state=doc.banked_norms.get("mean_state_median"),
        banked_mean_state_provenance=(
            "the collection stamps' median-of-MEAN-STATE norm — a DIFFERENT convention, "
            "recorded beside and never used for α (§2.5, rake M21b)"))

    # ---- the cells, in §2.1's order ------------------------------------------
    specs = order_cells(doc.cell_specs())
    seed_roots = {c.cell_id: cell_seed_root(corpus_sha=corpus_sha, node_key=node_key,
                                            arm=arm, site=site, cell_id=c.cell_id)
                  for c in specs}
    outcomes: list[CellOutcome] = []
    raw: dict[str, tuple[list[GenerationRecord], dict, dict]] = {}
    blocks = []
    baseline_block = None
    token_first: dict[str, str] = {}
    entropy_first: dict[str, str] = {}
    for spec in specs:
        alpha = resolve_alpha(spec.alpha_frac, norms.measured_per_token_median)
        (records, rows, ent_s, ent_u, block, panel, elapsed,
         tokens) = run_cell(runtime, spec, alpha=alpha, pool=pool, layout=layout,
                            corpus_sha=corpus_sha, node_key=node_key, arm=arm, n=n,
                            baseline_block=baseline_block)
        if block is not None:
            blocks.append(block)
            if spec.is_baseline and baseline_block is None:
                baseline_block = block
        tdigest = token_id_digest(records)
        edigest = entropy_array_digest(cell_entropy_arrays(records, ent_s, ent_u))
        token_first[spec.cell_id] = tdigest
        entropy_first[spec.cell_id] = edigest
        raw[spec.cell_id] = (records, ent_s, ent_u)
        staged = _staged_cell(doc, spec.cell_id)
        stamp = build_stamp(
            cell=spec, alpha=alpha, layout=layout, pool=pool, norms=norms,
            corpus_sha=corpus_sha, node_key=node_key, arm=arm,
            site_cross_check=cross,
            model_config_sha256=runtime.model_config_sha256(),
            vector_npz_sha256=(staged.vector_sha256 if staged else None)
            or doc.vectors_npz_sha256,
            vector_fd_gate=(staged.fd_gate if staged else None)
            or doc.bank_stamp.get("vector_fd_gate"),
            vector_build_stamp=(staged.build_stamp if staged else None)
            or doc.bank_stamp.get("vector_build_stamp"),
            transport_map=(staged.transport_map if staged else None),
            naive_row=(staged.naive_row if staged else None),
            trunk=trunk,
            replay_gate_digests={"token_ids": tdigest, "entropy_arrays": edigest,
                                 "order": ENTROPY_DIGEST_ORDER},
            battery_item_set_sha256=doc.battery_item_set_sha256,
            actuation_calibration=actuation_calibration_stamp or {
                "verdict": "OWED",
                "note": "§4.2 requires every §5 cell to NAME its site's actuation "
                        "calibration by job id + verdict + the three criterion values; "
                        "this column ran before that verdict was filed, which is an "
                        "OWED state (§10), never a silent absence."},
            per_cell_seed_roots={spec.cell_id: seed_roots[spec.cell_id]},
            scheduler_card_index=scheduler_card_index,
            extra={"label": doc.label or None,
                   "battery_injection_span": BATTERY_INJECTION_SPAN,
                   "entropy_digest_order": ENTROPY_DIGEST_ORDER})
        outcomes.append(CellOutcome(
            cell_id=spec.cell_id, kind=spec.kind, alpha_frac=spec.alpha_frac,
            alpha=alpha, n=len(records), tokens_generated=tokens, elapsed_s=elapsed,
            entropy=entropy_rise_from_rows(rows), probe_rows=rows,
            token_id_sha256=tdigest, entropy_array_sha256=edigest,
            capability=block.model_dump() if block is not None else None,
            coherence=panel.model_dump() if panel is not None else None,
            stamp=stamp))

    assert_column_complete(outcomes, len(specs), n_per_cell=n, node_key=node_key,
                           beside_cell_ids=[c.cell_id for c in specs if c.is_beside])

    # ---- the in-job replay gate (§2.7) ---------------------------------------
    selection, strata, digest = select_replay_cells(specs, node_key, corpus_sha)
    role_mapping = replay_gate_role_mapping(specs)
    token_replay: dict[str, str] = {}
    entropy_replay: dict[str, str] = {}
    by_id = {c.cell_id: c for c in specs}
    for cell_id in selection:
        spec = by_id[cell_id]
        alpha = resolve_alpha(spec.alpha_frac, norms.measured_per_token_median)
        rrecords, _, rs, ru, *_ = run_cell(
            runtime, spec, alpha=alpha, pool=pool, layout=layout,
            corpus_sha=corpus_sha, node_key=node_key, arm=arm, n=n,
            with_battery=False)
        token_replay[cell_id] = token_id_digest(rrecords)
        entropy_replay[cell_id] = entropy_array_digest(
            cell_entropy_arrays(rrecords, rs, ru))
    gate = evaluate_replay_gate(
        selection=selection, strata=strata, digest=digest,
        token_first={c: token_first[c] for c in selection},
        token_replay=token_replay,
        entropy_first={c: entropy_first[c] for c in selection},
        entropy_replay=entropy_replay, role_mapping=role_mapping)

    # ---- descriptive instrumentation (M19 — never a gate) --------------------
    invariance = None
    if characterize:
        target = by_id[strata["signal_at_0.3"]]
        invariance = characterize_layout_invariance(
            runtime, target, alpha=resolve_alpha(
                target.alpha_frac, norms.measured_per_token_median),
            pool=pool, layout=layout, corpus_sha=corpus_sha, node_key=node_key,
            arm=arm, n=min(n, layout.batch_size))

    table = None
    if blocks:
        from metabasis.scripts.capability_battery import dose_metric_table
        try:
            table = dose_metric_table(blocks)
        except Exception as exc:                         # noqa: BLE001 — M19(a)
            logger.warning("capability table not formed (%s: %s) — the per-cell blocks "
                           "are filed regardless", type(exc).__name__, exc)
            table = {"measured": False, "note": f"{type(exc).__name__}: {exc}"}

    result = ColumnResult(
        node_key=node_key, arm=arm, site=site, corpus_manifest_sha256=corpus_sha,
        label=doc.label, layout=layout, norms=norms, site_cross_check=cross,
        hook_admissibility=admissibility, alpha_zero_is_no_hook=alpha0_ok,
        cells=outcomes, replay_gate=gate, batch_invariance=invariance,
        fresh_process_replay={
            "state": "OWED",
            "recipe": "§2.7 descriptive: re-run ONE cell in a NEW process, same "
                      "canonical layout, CUBLAS_WORKSPACE_CONFIG=:4096:8 and "
                      "torch.use_deterministic_algorithms(True, warn_only=True), then "
                      "compare this result's token_id_sha256/entropy_array_sha256 for "
                      "that cell. Filed as a separate invocation by design: a job "
                      "cannot fork a fresh process of itself and still be the "
                      "one-load-per-node job (§2.1). M19: descriptive, never a gate."},
        capability_table=table, n_generations=sum(o.n for o in outcomes),
        work_root=str(work_root) if work_root else None)

    if write and work_root is not None:
        result = _write_column(result, raw=raw, work_root=Path(work_root))
    return result


def _staged_cell(doc: Any, cell_id: str) -> Optional[Any]:
    try:
        return doc.by_id(cell_id)
    except KeyError:                                          # pragma: no cover
        return None


def _write_column(result: ColumnResult, *, raw: dict, work_root: Path) -> ColumnResult:
    """Bank the column: raw beside every claim, then a manifest over everything.

    Raw generations and per-position entropy arrays are written per cell so a desk
    recompute has the same inputs the filed numbers came from (§10's recipes), and the
    manifest's shas are what make M4 checkable without re-running anything.
    """
    cells_dir = work_root / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    updated: list[CellOutcome] = []
    for outcome in result.cells:
        d = cells_dir / outcome.cell_id
        d.mkdir(parents=True, exist_ok=True)
        records, ent_s, ent_u = raw[outcome.cell_id]
        gen_path = d / "generations.jsonl"
        gen_path.write_text("".join(
            json.dumps(r.model_dump(), sort_keys=True) + "\n"
            for r in sorted(records, key=lambda x: x.generation_id)))
        ent_path = d / "entropy.npz"
        np.savez(ent_path,
                 **{f"steered_{g:04d}": a for g, a in sorted(ent_s.items())},
                 **{f"unsteered_{g:04d}": a for g, a in sorted(ent_u.items())})
        stamp_path = d / "stamp.json"
        stamp_path.write_text(json.dumps(outcome.stamp, indent=1, sort_keys=True,
                                         default=str))
        probe_path = d / "probe_rows.json"
        probe_path.write_text(json.dumps(
            [r.model_dump() for r in outcome.probe_rows], indent=1, sort_keys=True))
        paths = {"generations": str(gen_path), "entropy": str(ent_path),
                 "stamp": str(stamp_path), "probe_rows": str(probe_path)}
        if outcome.capability is not None:
            cap_path = d / "capability.json"
            cap_path.write_text(json.dumps(outcome.capability, indent=1, sort_keys=True,
                                           default=str))
            paths["capability"] = str(cap_path)
        updated.append(outcome.model_copy(update={"raw_paths": paths}))
    result = result.model_copy(update={"cells": updated})
    column_path = work_root / "column_result.json"
    column_path.write_text(json.dumps(json.loads(result.model_dump_json()), indent=1,
                                      sort_keys=True))
    manifest = {
        "grade": GRADE_LINE, "engine": "run_behavioral_cells.py",
        "brief_of_record": BRIEF_OF_RECORD, "brief_sha256": BRIEF_SHA256,
        "node_key": result.node_key, "arm": result.arm, "site": result.site,
        "label": result.label or None,
        "corpus_manifest_sha256": result.corpus_manifest_sha256,
        "replay_gate_passed": result.replay_gate.passed,
        "artifacts": {},
    }
    for path in sorted(work_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][str(path.relative_to(work_root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    manifest_path = work_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    logger.info("column banked: %d cells, %d generations → %s (manifest %s)",
                len(result.cells), result.n_generations, work_root, manifest_path)
    return result.model_copy(update={"manifest": manifest})


# ---------------------------------------------------------------- the HF runtime
class HFNodeRuntime:
    """`NodeRuntime` over a real HF causal LM — the node-side implementation.

    ONE hook is registered for the whole column and its `alpha`/`start_pos` are mutated
    per cell and per sub-batch (`hooks.py` reads both at every call), so the injection
    machinery is identical across cells by construction rather than by care. The α=0
    baseline runs WITH the hook attached, because the short-circuit is what makes it
    bitwise no-hook (§2.4) and that is the property the shared baseline rests on.
    """

    def __init__(self, model: Any, tokenizer: Any, *, node_key: str, arm: str,
                 site: int, vectors: dict[str, np.ndarray],
                 device: Optional[str] = None, dtype_name: str = "bfloat16",
                 position_ceiling: Optional[int] = None,
                 date_string: str = "12 Jul 2026") -> None:
        import torch

        self.torch = torch
        self.model = model
        self.tok = tokenizer
        self.node_key = node_key
        self.arm = arm
        self.site = site
        self.vectors = dict(vectors)
        self.dtype_name = dtype_name
        self._ceiling = position_ceiling
        self._date_string = date_string
        self.device = device or str(next(model.parameters()).device)
        self._handle: Any = None
        self._alpha = 0.0
        self._cell: Optional[CellSpec] = None
        self._hidden = int(
            getattr(model.config, "hidden_size", 0)
            or getattr(getattr(model.config, "text_config", None), "hidden_size"))

    # -- custody ------------------------------------------------------------
    def trunk(self) -> dict:
        import platform

        torch = self.torch
        out: dict[str, Any] = {
            "torch": torch.__version__, "python": platform.python_version(),
            "hostname": platform.node(), "dtype": self.dtype_name,
            "device": self.device, "measured": True}
        try:
            import transformers
            out["transformers"] = transformers.__version__
        except ImportError:                                   # pragma: no cover
            out["transformers"] = None
        if torch.cuda.is_available():
            try:
                out["gpu"] = torch.cuda.get_device_name(0)
                out["driver"] = getattr(torch.version, "cuda", None)
            except Exception as exc:                     # noqa: BLE001 — M19(a)
                out["gpu"] = f"(unavailable: {type(exc).__name__})"
                out["measured"] = False
        return out

    def model_config_sha256(self) -> Optional[str]:
        try:
            doc = self.model.config.to_dict()
        except Exception as exc:                         # noqa: BLE001 — M19(a)
            logger.warning("model config unavailable (%s) — M9 anchor recorded as "
                           "absent, never as agreement", exc)
            return None
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()

    def position_ceiling(self) -> Optional[int]:
        if self._ceiling is not None:
            return self._ceiling
        cfg = self.model.config
        for attr in ("n_positions", "max_position_embeddings"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    # -- tokenization -------------------------------------------------------
    def tokenize(self, prompt: BehavioralPrompt) -> list[int]:
        return render_prompt(prompt, self.tok, self.arm, self._date_string)

    def pad_token_id(self) -> int:
        pad = getattr(self.tok, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tok, "eos_token_id", None)
        if pad is None:
            raise BehavioralHarnessError(
                f"{self.node_key}: the tokenizer has neither a pad nor an eos token, "
                "so left padding has no filler — a column cannot be batched")
        return int(pad)

    def eos_token_id(self) -> Optional[int]:
        eos = getattr(self.tok, "eos_token_id", None)
        return None if eos is None else int(eos)

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return str(self.tok.decode(list(ids), skip_special_tokens=True))
        except TypeError:                                     # toy/limited tokenizers
            return str(self.tok.decode(list(ids)))

    # -- the injection ------------------------------------------------------
    def _vector_for(self, cell: CellSpec) -> Any:
        torch = self.torch
        if cell.vector_key is None:
            # §2.4: the baseline runs WITH the hook attached; α=0 short-circuits, so
            # the direction is never read. A named placeholder beats a None branch
            # that would make the baseline a different code path from every other cell.
            v = np.ones(self._hidden, dtype=np.float32)
            return torch.as_tensor(v / np.linalg.norm(v))
        if cell.vector_key not in self.vectors:
            raise NativeVectorUnavailable(
                f"{cell.cell_id}: vector key {cell.vector_key!r} is not in the staged "
                f"bank (keys: {sorted(self.vectors)}) — §9 item 3")
        v = np.asarray(self.vectors[cell.vector_key], dtype=np.float32).reshape(-1)
        if v.size != self._hidden:
            raise BehavioralHarnessError(
                f"{cell.cell_id}: vector dim {v.size} != model hidden {self._hidden}")
        return torch.as_tensor(v)

    def begin_cell(self, *, cell: CellSpec, alpha: float) -> None:
        from metabasis.extraction.hooks import (ResidualWriteSpec,
                                                attach_residual_write)
        assert_no_lesion_recipe(cell.vector_provenance, cell.vector_key or "")
        self.end_cell()
        self._cell = cell
        self._alpha = float(alpha)
        vec = self._vector_for(cell).to(self.device)
        self._handle = attach_residual_write(self.model, ResidualWriteSpec(
            layer_idx=self.site, vector=vec, alpha=float(alpha), start_pos=0,
            end_pos=None, normalize=True))

    def start_pos_sink(self, padded_prompt_length: int) -> None:
        if self._handle is not None:
            self._handle.spec.start_pos = int(padded_prompt_length)

    def end_cell(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def close(self) -> None:
        self.end_cell()

    def stepper(self) -> Stepper:
        return HFStepper(self.model, self.device)

    # -- measurements -------------------------------------------------------
    def vram_probe(self, batch_size: int) -> float:
        """Fraction of device memory a probe at max sequence length uses at `B`.

        Raises on a CPU-only device rather than returning a plausible number: M19's
        rule is that a degraded probe takes the CONSERVATIVE bound, and
        `choose_batch_size` implements exactly that when this raises.
        """
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device: VRAM headroom is unmeasurable here")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        seq = int(self._ceiling or 0) or 1024
        ids = torch.ones((int(batch_size), min(seq, 512)), dtype=torch.long,
                         device=self.device)
        with torch.no_grad():
            self.model(input_ids=ids, use_cache=True)
        after_free, _ = torch.cuda.mem_get_info()
        del ids
        torch.cuda.empty_cache()
        return float((total - after_free) / total)

    def measure_per_token_median_resid_norm(
            self, *, site: int, prompt_ids: Sequence[Sequence[int]]) -> float:
        """§2.5: one batched teacher-forced forward; per-token residual norms; median.

        The hook that carries the write is the same hook whose INPUT is measured here —
        `decoder_layers(model)[site]`'s incoming hidden states — so α is resolved
        against the very quantity it perturbs.
        """
        from metabasis.extraction.hooks import decoder_layers

        torch = self.torch
        if not prompt_ids:
            raise ResidualNormDeltaError("no prompts to measure the residual norm over")
        ids, mask = right_pad(list(prompt_ids), self.pad_token_id())
        captured: list[Any] = []

        def _capture(_module: Any, args: Any, kwargs: Any) -> None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            captured.append(hidden.detach().float())

        layer = decoder_layers(self.model)[site]
        handle = layer.register_forward_pre_hook(_capture, with_kwargs=True)
        try:
            with torch.no_grad():
                self.model(
                    input_ids=torch.as_tensor(ids, dtype=torch.long).to(self.device),
                    attention_mask=torch.as_tensor(mask, dtype=torch.long
                                                   ).to(self.device),
                    use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise ResidualNormDeltaError(
                f"L{site}: no hidden states captured — the site cannot be measured, so "
                "no dose can be resolved against it (§2.5)")
        hidden = captured[0]
        norms = hidden.norm(dim=-1).cpu().numpy()
        keep = np.asarray(mask, dtype=bool)
        vals = norms[keep]
        if vals.size == 0:
            raise ResidualNormDeltaError(f"L{site}: every position was padding")
        return float(np.median(vals))

    def alpha_zero_equivalence(self) -> Optional[tuple[Any, Any]]:
        """§2.4's once-per-node check, measured on this node's own first prompt."""
        from metabasis.extraction.hooks import (ResidualWriteSpec,
                                                attach_residual_write)
        torch = self.torch
        ids = torch.ones((1, 8), dtype=torch.long, device=self.device)
        with torch.no_grad():
            plain = self.model(input_ids=ids, use_cache=False).logits.detach().float(
            ).cpu().numpy()
        vec = torch.as_tensor(np.ones(self._hidden, dtype=np.float32)).to(self.device)
        handle = attach_residual_write(self.model, ResidualWriteSpec(
            layer_idx=self.site, vector=vec, alpha=0.0, start_pos=0, end_pos=None,
            normalize=True))
        try:
            with torch.no_grad():
                hooked = self.model(input_ids=ids, use_cache=False
                                    ).logits.detach().float().cpu().numpy()
        finally:
            handle.remove()
        return hooked, plain

    def hook_admissibility(self, site: int) -> Optional[HookAdmissibility]:
        return ssm_hook_admissibility_preflight(
            self.model, node_key=self.node_key, site=site,
            vector=np.ones(self._hidden, dtype=np.float32), n_tokens=4)

    # -- the probe (§2.6) ---------------------------------------------------
    def probe(self, records: Sequence[GenerationRecord], *, steered: bool
              ) -> list[tuple[np.ndarray, np.ndarray]]:
        torch = self.torch
        if not records:
            return []
        lengths = {r.prompt_length for r in records}
        if len(lengths) != 1:
            raise BehavioralHarnessError(
                f"probe group mixes prompt lengths {sorted(lengths)} — one absolute-"
                "position mask cannot be valid for all of them "
                f"({PROBE_GROUPING_READING})")
        prompt_len = lengths.pop()
        seqs = [r.input_ids for r in records]
        ids, mask = right_pad(seqs, self.pad_token_id())
        prev_alpha = None
        if self._handle is not None:
            prev_alpha = self._handle.spec.alpha
            self._handle.spec.alpha = float(self._alpha) if steered else 0.0
            self._handle.spec.start_pos = int(prompt_len)
        try:
            with torch.no_grad():
                out = self.model(
                    input_ids=torch.as_tensor(ids, dtype=torch.long).to(self.device),
                    attention_mask=torch.as_tensor(mask, dtype=torch.long
                                                   ).to(self.device),
                    use_cache=False)
            logits = out.logits.detach().float().cpu()
        finally:
            if self._handle is not None and prev_alpha is not None:
                self._handle.spec.alpha = prev_alpha
        rows = []
        for i, rec in enumerate(records):
            rows.append(per_position_entropy_and_nll(
                logits[i], rec.input_ids, rec.prompt_length,
                len(rec.generated_ids)))
        return rows

    # -- the battery (§6) ---------------------------------------------------
    def likelihood_nll(self) -> Callable[[str, str], float]:
        torch = self.torch

        def nll_of(prompt: str, continuation: str) -> float:
            p_ids = list(self.tok.encode(prompt, add_special_tokens=True))
            c_ids = list(self.tok.encode(continuation, add_special_tokens=False))
            if not c_ids:
                raise BehavioralHarnessError(
                    f"battery item continuation {continuation!r} tokenizes to nothing")
            ids = p_ids + c_ids
            if self._handle is not None:
                self._handle.spec.start_pos = len(p_ids)   # BATTERY_INJECTION_SPAN
            with torch.no_grad():
                logits = self.model(
                    input_ids=torch.as_tensor([ids], dtype=torch.long).to(self.device),
                    use_cache=False).logits.detach().float().cpu()[0]
            _, nll = per_position_entropy_and_nll(logits, ids, len(p_ids), len(c_ids))
            return float(np.sum(nll))

        return nll_of

    def format_probe_texts(self) -> dict[str, str]:
        from metabasis.scripts.capability_battery import FORMAT_ITEMS

        out: dict[str, str] = {}
        for item in FORMAT_ITEMS:
            ids = list(self.tok.encode(item.prompt, add_special_tokens=True))
            if self._handle is not None:
                self._handle.spec.start_pos = len(ids)
            stepper = self.stepper()
            try:
                logits = stepper.prefill(np.asarray([ids], dtype=np.int64),
                                         np.ones((1, len(ids)), dtype=np.int64))
                generated: list[int] = []
                for _ in range(item.max_new_tokens):
                    tok = sample_token(np.asarray(logits)[0], 0.0, BATTERY_SAMPLING)
                    generated.append(tok)
                    if self.eos_token_id() is not None and tok == self.eos_token_id():
                        break
                    logits = stepper.step(np.asarray([tok], dtype=np.int64))
            finally:
                stepper.close()
            out[item.item_id] = self.decode(generated)
        return out


def load_vectors(npz_path: Optional[Path]) -> dict[str, np.ndarray]:
    """The staged vector bank, by key — every failure named (§9 item 3)."""
    if npz_path is None:
        return {}
    path = Path(npz_path)
    if not path.exists():
        raise NativeVectorUnavailable(
            f"staged vector bank absent: {path} (§9 item 3). The cells document names "
            "it, so its absence is a HALT and not an empty bank.")
    try:
        with np.load(path) as z:
            return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    except (OSError, ValueError, EOFError) as exc:
        raise NativeVectorUnavailable(
            f"{path}: unreadable vector bank ({type(exc).__name__}: {exc})") from exc


def preflight_report(doc: Any, pool: PromptPool, *,
                     runtime: Optional[NodeRuntime] = None,
                     corpus_sha_of_record: str = CORPUS_SHA_V21) -> dict:
    """M10's first-class exit-early preflight: resolve everything, fire nothing.

    Split deliberately into a WEIGHTLESS half (vintage chain, pool, site cross-check,
    cell arithmetic, replay-gate constitution, seed roots) and a LOADED half (position
    budget, VRAM ladder, hook admissibility, the α=0 identity, the measured norm). The
    weightless half is exactly what the desk can re-run itself, and the loaded half
    names what it could not check when no model is present — never silently omitting it.
    """
    specs = order_cells(doc.cell_specs())
    kinds: dict[str, int] = {}
    for spec in specs:
        kinds[spec.kind] = kinds.get(spec.kind, 0) + 1
    report: dict[str, Any] = {
        "grade": GRADE_LINE, "node_key": doc.node_key, "arm": doc.arm,
        "site": doc.site, "label": doc.label or None,
        "n_cells": len(specs), "cells_by_kind": kinds,
        "n_per_cell": doc.n_per_cell,
        "n_generations": len(specs) * doc.n_per_cell,
        "dose_ladder": list(DOSE_LADDER),
        "sampling_of_record": SAMPLING_OF_RECORD.model_dump(),
        "prompt_pool": {"sha256": pool.sha256, "n": len(pool.prompts),
                        "matches_document": (doc.prompt_pool_sha256 is None
                                             or doc.prompt_pool_sha256 == pool.sha256)},
        "vintage_chain": dict(doc.vintage_chain),
        "loaded_checks": "NOT RUN (no model handed to the preflight)",
    }
    links = [v for v in doc.vintage_links() if v is not None]
    report["corpus_manifest_sha256"] = assert_corpus_vintage(
        *(links or [doc.corpus_manifest_sha256]), expected=corpus_sha_of_record)
    report["site_cross_check"] = site_cross_check(doc.node_key, doc.site)
    selection, strata, digest = select_replay_cells(
        specs, doc.node_key, report["corpus_manifest_sha256"])
    _mapping = replay_gate_role_mapping(specs)
    report["replay_gate"] = {"k": REPLAY_GATE_K, "cells": selection,
                             "strata": strata, "selection_digest": digest,
                             "constituted": True, "role_mapping": _mapping,
                             **({"role_mapping_reading": CALIBRATION_ONLY_ROLE_READING}
                                if _mapping == REPLAY_ROLE_MAPPING_CALIBRATION_ONLY
                                else {})}
    report["seed_roots_sample"] = {
        spec.cell_id: cell_seed_root(
            corpus_sha=report["corpus_manifest_sha256"], node_key=doc.node_key,
            arm=doc.arm, site=doc.site, cell_id=spec.cell_id)
        for spec in specs[:3]}
    if runtime is not None:
        prompt_ids = [runtime.tokenize(p) for p in pool.prompts]
        longest = max(len(i) for i in prompt_ids)
        assert_position_budget(longest, doc.max_new_tokens,
                               runtime.position_ceiling(), doc.node_key)
        B, measured, note = choose_batch_size(runtime.vram_probe)
        admissibility = runtime.hook_admissibility(doc.site)
        equivalence = runtime.alpha_zero_equivalence()
        if equivalence is not None:
            assert_alpha_zero_is_no_hook(*equivalence)
        norm = runtime.measure_per_token_median_resid_norm(
            site=doc.site, prompt_ids=prompt_ids[:NORM_SAMPLE_SIZE])
        report["loaded_checks"] = {
            "max_prompt_tokens": longest,
            "position_budget_ok": True,
            "canonical_batch_size": B, "batch_size_measured": measured,
            "headroom_note": note,
            "hook_admissible": None if admissibility is None
            else admissibility.admissible,
            "alpha_zero_is_no_hook": equivalence is not None,
            "measured_per_token_median_resid_norm": norm,
            "trunk": trunk_stamp(runtime),
            "model_config_sha256": runtime.model_config_sha256(),
        }
    return report


def load_model_and_tokenizer(model_path: str, *, dtype_name: str = "bfloat16"
                             ) -> tuple[Any, Any, str]:
    """§2.1's ONE load, with §2.2's frozen attention/dtype choices applied at it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"                                  # §2.2
    common = {"attn_implementation": "eager",
              "device_map": "auto" if torch.cuda.is_available() else None}
    # The dtype kwarg was RENAMED across the transformers versions this campaign spans
    # (`torch_dtype` in 4.5x, `dtype` in 5.x). Trying one and falling back is the only
    # version-robust form: silently loading at the WRONG dtype would change every
    # number in the column while the stamp still said bf16 (M9's drift surface).
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **common)
    except TypeError:
        logger.info("this transformers build takes `torch_dtype`, not `dtype` — "
                    "loading at %s through the older kwarg", dtype_name)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                     **common)
    actual = str(next(model.parameters()).dtype).replace("torch.", "")
    if actual != dtype_name:
        raise BehavioralHarnessError(
            f"requested dtype {dtype_name} but the loaded parameters are {actual} — "
            "the collection regime is a stamped fact (§2.8) and a silent downgrade "
            "would make every number in this column incomparable to the bank")
    model.eval()
    model.requires_grad_(False)
    return model, tok, dtype_name


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

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Enough of a decode for the coherence panel and the format probe."""
        return " ".join(f"w{int(i) % 17}" for i in ids)

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


class _StubRuntime:
    """A weightless `NodeRuntime` — the orchestration proof (rake M44).

    The §2.1 ORDER, the per-cell stamps, the completeness guard, the replay gate and
    the banking are the parts of this module that no GPU can be spared to test and that
    a mis-wire would corrupt silently. They are therefore proved here against a runtime
    whose every reading is a deterministic function of its inputs: the same
    `run_column` that drives a 405B drives this, so the orchestration stays proven in a
    configuration with no deep-learning stack at all. What it CANNOT prove — that a
    real forward's numbers are right — is exactly what the torch block below
    characterizes on a real (tiny) model.
    """

    def __init__(self, pool: PromptPool, tok: Any, arm: str, site: int, *,
                 hidden: int = 8, norm: float = 12.2391) -> None:
        self.pool, self.tok, self.arm, self.site = pool, tok, arm, site
        self.hidden, self._norm = hidden, norm
        self.alpha = 0.0
        self.cell: Optional[CellSpec] = None
        self.order: list[str] = []
        self.start_positions: list[int] = []

    def trunk(self) -> dict:
        return {"torch": None, "transformers": None, "hostname": "selftest",
                "dtype": "float32", "device": "cpu", "measured": True}

    def model_config_sha256(self) -> Optional[str]:
        return "a" * 64

    def tokenize(self, prompt: BehavioralPrompt) -> list[int]:
        return render_prompt(prompt, self.tok, self.arm)

    def pad_token_id(self) -> int:
        return 0

    def eos_token_id(self) -> Optional[int]:
        return None

    def position_ceiling(self) -> Optional[int]:
        return None

    def vram_probe(self, batch_size: int) -> float:
        return 0.10 * (batch_size / 10.0)

    def measure_per_token_median_resid_norm(self, *, site: int,
                                            prompt_ids: Sequence[Sequence[int]]
                                            ) -> float:
        if not prompt_ids:
            raise ResidualNormDeltaError("no prompts to measure over")
        return self._norm

    def begin_cell(self, *, cell: CellSpec, alpha: float) -> None:
        assert_no_lesion_recipe(cell.vector_provenance, cell.vector_key or "")
        self.cell, self.alpha = cell, float(alpha)
        self.order.append(cell.cell_id)

    def start_pos_sink(self, padded_prompt_length: int) -> None:
        self.start_positions.append(int(padded_prompt_length))

    def end_cell(self) -> None:
        self.cell = None

    def stepper(self) -> Stepper:
        return _StubStepper(eos_token_id=None)

    def probe(self, records: Sequence[GenerationRecord], *, steered: bool
              ) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for r in records:
            base = np.array([abs(t) % 7 + 1 for t in r.generated_ids],
                            dtype=np.float32) / 7.0
            # the steered read differs from the unsteered one by the cell's own α, so
            # a rise is a function of the dose and the digest covers both halves
            ent = base + (np.float32(self.alpha) * np.float32(0.01) if steered else 0.0)
            nll = base * np.float32(2.0)
            out.append((ent.astype(np.float32), nll.astype(np.float32)))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"w{int(i) % 23}" for i in ids)

    def likelihood_nll(self) -> Callable[[str, str], float]:
        def nll_of(prompt: str, continuation: str) -> float:
            # deterministic, and mildly α-sensitive so a capability delta is non-zero
            h = int(hashlib.sha256((prompt + "|" + continuation).encode()
                                   ).hexdigest()[:8], 16)
            return float((h % 1000) / 100.0) + abs(self.alpha) * 0.001
        return nll_of

    def format_probe_texts(self) -> dict[str, str]:
        from metabasis.scripts.capability_battery import FORMAT_ITEMS
        return {item.item_id: ("blue" if i % 2 == 0 else "well, it depends")
                for i, item in enumerate(FORMAT_ITEMS)}

    def alpha_zero_equivalence(self) -> Optional[tuple[Any, Any]]:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        return arr, arr.copy()

    def hook_admissibility(self, site: int) -> Optional[HookAdmissibility]:
        return None

    def close(self) -> None:
        self.cell = None


def _toy_document(pool: PromptPool, *, node: str, arm: str, site: int, corpus: str,
                  cells: Sequence[CellSpec], n_per_cell: int,
                  max_new_tokens: int) -> Any:
    """A `CellsDocument` built in memory — the staging→engine contract, exercised."""
    from metabasis.scripts.build_behavioral_banks import CellsDocument, StagedCell
    from metabasis.scripts.capability_battery import BATTERY_ITEM_SET_SHA256

    tmap = {"fit_sha256": "c" * 64, "family": "proc_k128", "arm": arm,
            "corpus_vintage": corpus}
    naive = {"pair": "hub->target", "verdict": "CLEAR", "bare_cos": 0.03, "q95": 0.09}
    staged = []
    for c in cells:
        transported = c.kind in ("transported", "transported_band", "naive", "bridge",
                                 "judged")
        staged.append(StagedCell(
            spec=c.model_copy(update={"n": n_per_cell}),
            transport_map=tmap if transported else None,
            naive_row=naive if transported else None,
            vector_sha256="b" * 64,
            fd_gate={"PASSES_FD_GATE": True, "median_rel_error": 0.011},
            build_stamp={"corpus_manifest_sha256": corpus, "builder": "selftest"}))
    return CellsDocument(
        node_key=node, arm=arm, site=site, source_key="hub", pair="hub->target",
        corpus_manifest_sha256=corpus, prompt_pool_sha256=pool.sha256,
        battery_item_set_sha256=BATTERY_ITEM_SET_SHA256, vectors_npz=None,
        vectors_npz_sha256="b" * 64, n_per_cell=n_per_cell,
        max_new_tokens=max_new_tokens, label="SELFTEST, NOT A READ",
        vintage_chain={"corpus_manifest": corpus, "transport_map": corpus},
        banked_norms={"per_token_median": None, "mean_state_median": 12.1125},
        cells=tuple(staged),
        bank_stamp={"vector_fd_gate": {"PASSES_FD_GATE": True},
                    "vector_build_stamp": {"builder": "selftest"}})


def _compact_cells(site: int) -> list[CellSpec]:
    """The smallest cell set that can constitute the §2.7 gate (all three strata)."""
    cells = [baseline_cell(site)]
    for key, kind, band in (("entropy_gradient", "calibration", None),
                            ("gRband1", "transported_band", "gRband"),
                            ("gentropy_gradient", "transported", None)):
        for frac in SCORING_DOSES:
            cells.append(CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key=key, site=site, frac=frac),
                kind=kind, vector_key=key, site=site, alpha_frac=frac,
                band_family=band, vector_provenance=f"toy::{key}"))
    return cells


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


def _calibration_only_cells(site: int = 26, *, with_beside: bool = True
                            ) -> list[CellSpec]:
    """A miniature of the CALIBRATION-ONLY column (B-1's case): no transported cell.

    Mirrors the staged qwen2.5-3b-instruct certification column in shape — the native
    lever over the full ladder, one Rband member over the full ladder, and (optionally)
    the designated Σ-beside cells at the scoring doses — which is exactly the column
    that could not constitute the §2.7 gate before B-1 was ruled.
    """
    cells = [baseline_cell(site)]
    for key, kind, band in (("entropy_gradient", "calibration", None),
                            ("Rband1", "calibration_band", "Rband")):
        for frac in DOSE_LADDER:
            cells.append(CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key=key, site=site, frac=frac),
                kind=kind, vector_key=key, site=site, alpha_frac=frac,
                band_family=band, vector_provenance=f"toy::{key}"))
    if with_beside:
        for frac in SCORING_DOSES:
            cells.append(CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key="SigmaBand1", site=site,
                                                frac=frac),
                kind="calibration_band", vector_key="SigmaBand1", site=site,
                alpha_frac=frac, band_family="SigmaBand",
                vector_provenance="toy::SigmaBand1"))
    return cells


def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, data-independent verification of every load-bearing property."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    def skip(name: str, why: str) -> None:
        """A NAMED skip: a block that cannot run here, recorded as run-and-absent.

        RAKE M44: a selftest that CRASHES in a configuration is indistinguishable
        from a selftest that FAILS, and neither says which. A named skip is a
        third state — the check did not run, the reason is on the record, and the
        suite's exit code stays 0 because nothing FAILED. The skip is counted in
        the tail so a sweep can report coverage per configuration rather than
        inferring it.
        """
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))
        logger.info("SKIP %s — %s", name, why)

    # THE AVAILABILITY PROBE (rake M44). torch is absent from the desk's own repo
    # .venv, so every torch-dependent BLOCK below is branched on this one probe
    # rather than on a bare import at its point of use. The properties that make
    # the harness legitimate — the seed table, the layout invariance, the dose
    # ladder, the stamp checklist, the replay-gate logic — are numpy-only BY
    # DESIGN and must stay provable in a configuration with no deep-learning
    # stack at all; only the entropy fixture and the real-forward
    # characterization genuinely need torch.
    try:
        import torch
        _torch: Any = torch
        _no_torch = ""
    except ImportError as exc:                       # the desk .venv configuration
        _torch = None
        _no_torch = f"torch unavailable ({exc})"

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

    # ---- 2b. THE IDENTITY GATE: the batched kernel IS `sample_token` ----------
    # The whole warrant for the 2026-08-01 vectorization. Any mismatch below is a
    # FAILURE and never a tolerance: the two kernels are the same step or the fast
    # one does not run.
    print("== selftest 2b: the batched kernel is byte-identical to ruling 8's step ==")
    # (batch, vocab, families). The full seven families run everywhere except the
    # 80 × 128256 corner, where a run of the whole set costs ~20 s of a suite that
    # has to pass in four configurations; the four run there are the ones that could
    # plausibly separate a SIMD lane from a scalar tail, and the same corner is
    # re-certified at run time inside every job anyway.
    gate_shapes: tuple[tuple[int, int, tuple[str, ...]], ...] = (
        (1, 2, CERTIFICATION_FAMILIES),              # the smallest vocab there is
        (1, 61, CERTIFICATION_FAMILIES),
        (3, 61, CERTIFICATION_FAMILIES),
        (80, 61, CERTIFICATION_FAMILIES),            # toy vocab, layout of record
        (1, 1000, CERTIFICATION_FAMILIES),
        (3, 1000, CERTIFICATION_FAMILIES),
        (80, 1000, CERTIFICATION_FAMILIES),
        (3, 32000, CERTIFICATION_FAMILIES),          # gpt2-xl / qwen-ish
        (1, 128256, CERTIFICATION_FAMILIES),         # llama-3-ish, one row
        (80, 128256, CERTIFICATION_FAMILIES_RUNTIME))    # …and the real corner
    certified: list[str] = []
    gate_failure = ""
    for gB, gV, gFam in gate_shapes:
        try:
            certified.append(assert_batched_sampler_identity(
                batch=gB, vocab=gV, families=gFam, n_u=3, force=True))
        except VectorizedSamplerNotIdentical as exc:
            gate_failure = f"shape ({gB}, {gV}): {exc}"
            break
    check("both kernels agree BYTE for byte at every gate shape "
          "(batch ∈ {1, 3, 80} × vocab ∈ {2, 61, 1000, 32000, 128256})",
          not gate_failure and len(certified) == len(gate_shapes),
          gate_failure or (
              f"{len(certified)} shapes certified through max/shift/exp/sum/"
              f"normalize/cumsum and token draws; families "
              f"{list(CERTIFICATION_FAMILIES)}; numpy {np.__version__}"))
    check("the degenerate distributions are IN the gate, not beside it",
          all(f in CERTIFICATION_FAMILIES
              for f in ("one_hot", "uniform", "denormal_adjacent", "tiny_spread",
                        "extreme_mixed")),
          "one-hot (p is exactly {0,1}), uniform (p == 1/V), denormal-adjacent "
          "(exp returns subnormals), tiny-spread, ±inf/NaN mixed")

    # EVERY breakpoint of the inverse CDF, not a sample of uniforms: the two step
    # functions agree at each breakpoint and one ulp below it, so they agree for
    # every u in [0, 1) — a proof over the whole tape, not over the tape's draws.
    gate_rng = np.random.default_rng(CERTIFICATION_SEED)
    bp_blk = certification_logits(3, 61, "normal", gate_rng)
    bp_cdf, _ = _batched_cdf(np.ascontiguousarray(bp_blk, dtype=np.float64),
                             SAMPLING_OF_RECORD)
    bp_bad: list[tuple[int, int, float]] = []
    bp_probes = 0
    for r_ in range(bp_blk.shape[0]):
        for k_ in range(bp_blk.shape[1]):
            for u_ in (float(bp_cdf[r_, k_]),
                       float(np.nextafter(bp_cdf[r_, k_], -np.inf))):
                u_ = min(max(u_, 0.0), 1.0 - 2.0 ** -53)
                bp_probes += 1
                if (sample_tokens(bp_blk, us=[u_], rows=[r_])[0]
                        != sample_token(bp_blk[r_], u_, SAMPLING_OF_RECORD)):
                    bp_bad.append((r_, k_, u_))
    check("EVERY inverse-CDF breakpoint (and one ulp below it) draws the same token",
          not bp_bad, f"{bp_probes} breakpoint probes over 3×61, {len(bp_bad)} "
                      "disagreement(s)")
    # …and the probes have TEETH: a kernel that got the boundary rule wrong by one
    # index is caught by exactly these uniforms. A gate nothing can fail is not one.
    teeth = sum(1 for r_ in range(bp_blk.shape[0]) for k_ in range(bp_blk.shape[1])
                if int(np.searchsorted(bp_cdf[r_], float(bp_cdf[r_, k_]),
                                       side="left"))
                != sample_token(bp_blk[r_], float(bp_cdf[r_, k_]),
                                SAMPLING_OF_RECORD))
    check("the breakpoint probes have TEETH: a side='left' kernel is caught by them",
          teeth > 0, f"{teeth} of {bp_blk.shape[0] * bp_blk.shape[1]} breakpoints "
                     "separate side='right' from side='left'")

    # DEGENERATE ROWS ARE REFUSED, NOT APPROXIMATED — including how they fail.
    nonfinite = certification_logits(2, 8, "all_nonfinite", gate_rng)
    check("an all-non-finite row raises ValueError under BOTH kernels",
          _raises(lambda: sample_token(nonfinite[0], 0.5), ValueError)
          and _raises(lambda: sample_tokens(nonfinite, us=[0.5, 0.5]), ValueError),
          "the batched kernel hands the row back to `sample_token`, so a degenerate "
          "row fails in exactly the way it always has")
    hot = SamplingConfig(name="hot", temperature=1e-300, provenance="selftest")
    overflow = np.array([[1e30, 2e30, 3e30], [1.0, 2.0, 3.0]], dtype=np.float32)
    check("a row that overflows to ±inf under the temperature divide is refused "
          "identically",
          _raises(lambda: sample_token(overflow[0], 0.5, hot), ValueError)
          and _raises(lambda: sample_tokens(overflow, us=[0.5, 0.5], sampling=hot),
                      ValueError),
          "both paths raise; the batched kernel never invents a token for a row the "
          "reference refuses")

    # THE CONFIGS THE KERNEL DOES NOT OWN go to the reference, and the certifier
    # refuses to certify them at all.
    refused_cfgs = (
        SamplingConfig(name="greedy", do_sample=False, provenance="selftest"),
        SamplingConfig(name="k1", top_k=1, provenance="selftest"),
        SamplingConfig(name="p", top_p=0.5, provenance="selftest"))
    cfg_blk = certification_logits(4, 61, "normal", gate_rng)
    cfg_us = [0.01, 0.4, 0.77, 0.999]
    check("greedy / top-k / top-p dispatch to the reference and are NAMED refusals",
          all(bool(vectorized_sampling_refusal(c))
              and sample_tokens(cfg_blk, us=cfg_us, sampling=c)
              == [sample_token(cfg_blk[r_], cfg_us[r_], c) for r_ in range(4)]
              and _raises(lambda c=c: assert_batched_sampler_identity(
                  batch=4, vocab=61, sampling=c), VectorizedSamplerNotIdentical)
              for c in refused_cfgs),
          "; ".join(vectorized_sampling_refusal(c).split(":")[0]
                    for c in refused_cfgs))
    check("the vectorized kernel OWNS the config of record (no refusal)",
          vectorized_sampling_refusal(SAMPLING_OF_RECORD) == "",
          SAMPLING_OF_RECORD.name)
    check("a shape is certified once per process and the certificate is quotable",
          assert_batched_sampler_identity(batch=3, vocab=61)
          == assert_batched_sampler_identity(batch=3, vocab=61)
          and "byte-identical" in assert_batched_sampler_identity(batch=3, vocab=61),
          assert_batched_sampler_identity(batch=3, vocab=61))
    check("the kernel of record is the vectorized one and both are named objects",
          SAMPLER_KERNEL_OF_RECORD.name == "vectorized"
          and SAMPLER_KERNEL_REFERENCE.name == "reference"
          and _raises(lambda: SamplerKernel(name="approximate", provenance="no"),
                      ValueError),
          f"kernel of record: {SAMPLER_KERNEL_OF_RECORD.name}")

    # ---- 3. LAYOUT INVARIANCE — the §2.2 replay-gate property -----------------
    print("== selftest 3: layout invariance at B ∈ {80, 40, 20, 10, 1} (§2.2) ==")
    tok = _ToyTokenizer()
    pool = _toy_pool()
    cell = CellSpec(cell_id=CELL_ID_TEMPLATE.format(
        vector_key="gentropy_gradient", site=site, frac=0.3),
        kind="transported", vector_key="gentropy_gradient", site=site,
        alpha_frac=0.3, band_family=None, vector_provenance="toy")

    def run_at(B: int, n: int = N_PER_CELL,
               kernel: SamplerKernel = SAMPLER_KERNEL_OF_RECORD
               ) -> list[GenerationRecord]:
        layout = CanonicalLayout(batch_size=B, dtype="float32", max_new_tokens=12,
                                 frozen=True)
        return generate_cell(
            lambda: _StubStepper(eos_token_id=None),
            cell=cell, pool=pool,
            tokenize=lambda p: render_prompt(p, tok, arm),
            layout=layout, pad_token_id=0, eos_token_id=None,
            corpus_sha=corpus, node_key=node, arm=arm, n=n,
            sampler_kernel=kernel)

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
    # THE END-TO-END HALF OF THE IDENTITY GATE (numpy-only, so it holds in the
    # no-torch configuration): the same 80 generations, the same tapes, both kernels.
    kernel_digests = {B: (token_id_digest(run_at(B, kernel=SAMPLER_KERNEL_REFERENCE)),
                          token_id_digest(run_at(B, kernel=SAMPLER_KERNEL_VECTORIZED)))
                      for B in BATCH_LADDER}
    check("a whole cell is token-identical under both kernels at every B on the ladder",
          all(a == b for a, b in kernel_digests.values())
          and len({d for pair in kernel_digests.values() for d in pair}) == 1,
          f"one digest across B ∈ {list(BATCH_LADDER)} × both kernels: "
          f"{kernel_digests[BATCH_LADDER[0]][0][:16]}…")
    eos_ref = generate_cell(
        lambda: _StubStepper(), cell=cell, pool=pool,
        tokenize=lambda p: render_prompt(p, tok, arm), layout=layout10,
        pad_token_id=0, eos_token_id=int(ref[0].generated_ids[0]),
        corpus_sha=corpus, node_key=node, arm=arm, n=20,
        sampler_kernel=SAMPLER_KERNEL_REFERENCE)
    check("EOS-frozen rows draw identically under both kernels (the active subset "
          "shrinks, the block does not)",
          [tuple(r.generated_ids) for r in eos_ref]
          == [tuple(r.generated_ids) for r in eos_recs]
          and [r.n_uniforms_consumed for r in eos_ref]
          == [r.n_uniforms_consumed for r in eos_recs],
          f"{sum(r.finished_with_eos for r in eos_ref)}/20 rows frozen; the batched "
          "kernel keeps computing the whole block so no row's slot moves")

    # ---- 4. the entropy convention fixture -----------------------------------
    print("== selftest 4: entropy convention (§2.6) ==")
    # The FIXTURE half needs torch (`per_position_entropy_and_nll` is a torch
    # kernel); the aggregation and digest half below is numpy-only and always runs.
    if _torch is None:
        skip("entropy convention fixture (§2.6): per-position entropy/NLL against a "
             "hand-computed log_softmax", _no_torch)
    else:
        V, T, P = 8, 7, 4
        _torch.manual_seed(20260729)
        fixture = _torch.randn(T, V)
        seq = [3, 1, 4, 1, 5, 7, 2]            # every id < V, so the NLL gather is real
        ent, nll = per_position_entropy_and_nll(fixture, seq, P, T - P)
        lp = _torch.log_softmax(fixture.float(), dim=-1)
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
    # the banked parser lives behind `entropy_write_probe`, which imports torch, so
    # this one check inherits the availability branch rather than the whole block.
    if _torch is None:
        skip("score_entropy_writes.parse_cell round-trips every laddered cell id",
             f"{_no_torch} — the banked parser is reached through "
             "entropy_write_probe, which imports torch")
    else:
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

    # ---- 5b. the Σ-beside BESIDE family (B4, Luxia 2026-08-04) ----------------
    print("== selftest 5b: SigmaBand is a BESIDE family, never a gate input (B4) ==")
    sigma_cal = CellSpec(cell_id=f"SigmaBand1_L{site}_a+0.30", kind="calibration_band",
                         vector_key="SigmaBand1", site=site, alpha_frac=0.3,
                         band_family="SigmaBand", vector_provenance="toy::SigmaBand1")
    sigma_tr = CellSpec(cell_id=f"SigmaBand2_L{site}_a-0.30", kind="transported_band",
                        vector_key="SigmaBand2", site=site, alpha_frac=-0.3,
                        band_family="SigmaBand", vector_provenance="toy::SigmaBand2")
    check("(a) a SigmaBand cell CONSTRUCTS like any cell — banked id formatting, "
          "frozen ladder, n of record",
          sigma_cal.cell_id == CELL_ID_TEMPLATE.format(
              vector_key="SigmaBand1", site=site, frac=0.3)
          and sigma_cal.alpha_frac in DOSE_LADDER and sigma_cal.n == N_PER_CELL)
    check("(a) …and it SCHEDULES like any cell: the §2.1 fire order ranks it with its "
          "own band kind, and it takes a dose exactly like the band it sits beside",
          order_cells([sigma_tr, sigma_cal, baseline_cell(site)])[0].cell_id
          == baseline_cell(site).cell_id
          and order_cells([sigma_tr, sigma_cal])[0].cell_id == sigma_cal.cell_id
          and resolve_alpha(sigma_cal.alpha_frac, norm) == 0.3 * norm)
    check("(a) …and it generates from the SAME seed table as any other cell (the "
          "beside is a different DRAW, never a different mechanism)",
          seed_int(seed_material(corpus_sha="c" * 64, node_key="n", arm="native",
                                 site=site, cell_id=sigma_cal.cell_id, gen_id=3))
          == seed_int(seed_material(corpus_sha="c" * 64, node_key="n", arm="native",
                                    site=site, cell_id=sigma_cal.cell_id, gen_id=3)))
    check("a beside is `is_null` (a null DRAW) but NOT `is_null_of_record`",
          sigma_cal.is_null and sigma_cal.is_beside
          and not sigma_cal.is_null_of_record
          and CellSpec(cell_id=f"gRband1_L{site}_a+0.30", kind="transported_band",
                       vector_key="gRband1", site=site, alpha_frac=0.3,
                       band_family="gRband").is_null_of_record)
    check("(b) a beside holds NO replay-gate stratum, on either side of the column",
          _stratum_of(sigma_cal) is None and _stratum_of(sigma_tr) is None)
    with_beside = list(full) + [sigma_cal, sigma_tr]
    sel_b, strata_b, _ = select_replay_cells(with_beside, node, corpus)
    sel_plain, strata_plain, _ = select_replay_cells(full, node, corpus)
    check("(b) adding besides does not change the §2.7 gate's selection AT ALL",
          sel_b == sel_plain and strata_b == strata_plain, str(sel_b))
    from metabasis.scripts.actuation_calibration import (
        N_CALIBRATION_BAND_CELLS, CalibrationCellPlan, calibration_cell_plan)
    _plan = calibration_cell_plan(node_key=node, site=site, arm="native",
                                  per_token_median_resid_norm=norm)
    _swapped = [c.model_dump() for c in _plan.band_cells]
    _swapped[0] = sigma_cal.model_dump()          # dict, not the object: the engine runs
    _plan_fields = {**_plan.model_dump(), "band_cells": _swapped}  # as __main__ here
    check("(b) the §4 ACTUATION criteria refuse a beside in the band population — the "
          "18-cell arity holds, so the FAMILY test is the one that fires",
          len(_swapped) == N_CALIBRATION_BAND_CELLS
          and _raises(lambda: CalibrationCellPlan(**_plan_fields), ValueError)
          and all(c.band_family == "Rband" for c in _plan.band_cells))
    check("(c) a beside outside a band KIND is refused by CellSpec itself — a beside "
          "wearing a signal kind would be pooled as signal",
          _raises(lambda: CellSpec(
              cell_id=f"SigmaBand1_L{site}_a+0.30", kind="transported",
              vector_key="SigmaBand1", site=site, alpha_frac=0.3,
              band_family="SigmaBand"), ValueError)
          and _raises(lambda: CellSpec(
              cell_id=f"SigmaBand1_L{site}_a+0.30", kind="calibration",
              vector_key="SigmaBand1", site=site, alpha_frac=0.3,
              band_family="SigmaBand"), ValueError))
    check("(c) an unknown band family is still refused (the literal is widened, "
          "not opened)",
          _raises(lambda: CellSpec(
              cell_id=f"Wband1_L{site}_a+0.30", kind="calibration_band",
              vector_key="Wband1", site=site, alpha_frac=0.3,
              band_family="Wband"), ValueError))
    check("the family tuples partition: GATE ∪ BESIDE == BAND_FAMILIES, disjoint",
          set(GATE_BAND_FAMILIES) | set(BESIDE_BAND_FAMILIES) == set(BAND_FAMILIES)
          and not set(GATE_BAND_FAMILIES) & set(BESIDE_BAND_FAMILIES))

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

    # ---- 7b. B-1: the calibration-only ROLE MAPPING (RULED 2026-08-05) ---------
    print("== selftest 7b: B-1's calibration-only replay-gate role mapping ==")
    # The BYTE ASSERT. Both constants were read off the run of the code BEFORE B-1 was
    # implemented (/tmp/claude-output/gate-baseline-BEFORE-*.log, case
    # "toy26|qwen2.5-3b-instruct"), so this check fails the moment the ruling's
    # implementation perturbs a transported column's gate by so much as one byte.
    B1_PRE_RULING_SELECTION = ("gentropy_gradient_L26_a+0.30",
                               "gRband1_L26_a-0.10",
                               "entropy_gradient_L26_a+0.03")
    B1_PRE_RULING_SHA256 = (
        "4115092eff9d6ca1adca17ea8dc952dac546918cbf2d8841a9f58281f1c761c9")
    _sel_blob = json.dumps([sel, strata, dig], sort_keys=True)
    check("(B-1) a TRANSPORTED column's gate selection is BYTE-IDENTICAL to its "
          "pre-ruling selection (frozen triple + strata + digest)",
          tuple(sel) == B1_PRE_RULING_SELECTION
          and hashlib.sha256(_sel_blob.encode()).hexdigest() == B1_PRE_RULING_SHA256,
          hashlib.sha256(_sel_blob.encode()).hexdigest()[:16] + "…")
    check("(B-1) …and a transported column never reaches the calibration-only "
          "mapping — the mapping OF RECORD wins whenever it can constitute the gate",
          replay_gate_role_mapping(full) == REPLAY_ROLE_MAPPING_OF_RECORD
          and replay_gate_role_mapping(with_beside) == REPLAY_ROLE_MAPPING_OF_RECORD
          and replay_gate_role_mapping(_compact_cells(site))
          == REPLAY_ROLE_MAPPING_OF_RECORD
          and column_has_transported_cells(full))
    cal_only = _calibration_only_cells(site)
    check("(B-1) the calibration-only column HAS no transported cell, and falls under "
          "the ruled mapping",
          not column_has_transported_cells(cal_only)
          and replay_gate_role_mapping(cal_only)
          == REPLAY_ROLE_MAPPING_CALIBRATION_ONLY)
    csel, cstrata, cdig = select_replay_cells(cal_only, node, corpus)
    csel2, cstrata2, _ = select_replay_cells(list(reversed(cal_only)), node, corpus)
    check("(B-1) it constitutes the gate: K=3, one cell per FROZEN stratum (the "
          "stratum set and K are untouched by the ruling)",
          len(csel) == 3 and sorted(cstrata) == sorted(REPLAY_GATE_STRATA)
          and REPLAY_GATE_K == 3, json.dumps(cstrata))
    check("(B-1) signal role = the native EGV cell at |0.3| (ruled)",
          cstrata["signal_at_0.3"].startswith("entropy_gradient")
          and abs(float(cstrata["signal_at_0.3"].rsplit("_a", 1)[1]))
          == SCORING_DOSE_MAGNITUDE, cstrata["signal_at_0.3"])
    check("(B-1) band role = the node's OWN Rband (never a gRband, never a beside)",
          cstrata["random_band"].startswith("Rband"), cstrata["random_band"])
    check("(B-1) calibration role = a SMALL-DOSE cell (native lever, non-scoring, "
          "non-zero)",
          cstrata["calibration"].startswith("entropy_gradient")
          and 0.0 < abs(float(cstrata["calibration"].rsplit("_a", 1)[1]))
          < SCORING_DOSE_MAGNITUDE, cstrata["calibration"])
    check("(B-1) the three roles are filled by three DISTINCT cells",
          len(set(csel)) == 3, str(csel))
    check("(B-1) selection is deterministic and order-independent, from the same "
          "sha256(node_key|corpus_sha) as any other column",
          csel == csel2 and cstrata == cstrata2
          and cdig == hashlib.sha256(f"{node}|{corpus}".encode()).hexdigest(),
          str(csel))
    check("(B-1) a Σ-BESIDE cell still enters NO role — the B4 rule holds under the "
          "new mapping (dropping the besides changes nothing)",
          select_replay_cells(_calibration_only_cells(site, with_beside=False),
                              node, corpus)[1] == cstrata
          and all(_stratum_of(c, calibration_only=True) is None
                  for c in cal_only if c.is_beside))
    check("(B-1) a calibration-only column with NO Rband still HALTs "
          "(ExpectedNShortfall — incomplete, not exempt)",
          _raises(lambda: select_replay_cells(
              [c for c in cal_only if c.band_family != "Rband"], node, corpus),
              ExpectedNShortfall))
    check("(B-1) a calibration-only column with no SMALL-DOSE cell still HALTs",
          _raises(lambda: select_replay_cells(
              [c for c in cal_only
               if c.kind != "calibration" or abs(c.alpha_frac)
               == SCORING_DOSE_MAGNITUDE], node, corpus),
              ExpectedNShortfall))
    check("(B-1) a calibration-only column with no |0.3| lever cell still HALTs",
          _raises(lambda: select_replay_cells(
              [c for c in cal_only
               if c.kind != "calibration" or abs(c.alpha_frac)
               != SCORING_DOSE_MAGNITUDE], node, corpus),
              ExpectedNShortfall))
    check("(B-1) a column that HOLDS transported cells but is short a stratum is NOT "
          "re-roled — it keeps the mapping of record and HALTs as before",
          replay_gate_role_mapping([c for c in full if c.kind != "calibration"])
          == REPLAY_ROLE_MAPPING_OF_RECORD
          and _raises(lambda: select_replay_cells(
              [c for c in full if c.kind != "calibration"], node, corpus),
              ExpectedNShortfall))
    check("(B-1) a naive cell makes a column transported-family (ruling 5: it carries "
          "a SOURCE object), so a naive-bearing column is never calibration-only",
          column_has_transported_cells(cal_only + [CellSpec(
              cell_id=CELL_ID_TEMPLATE.format(vector_key="naive_entropy_gradient",
                                              site=site, frac=0.3),
              kind="naive", vector_key="naive_entropy_gradient", site=site,
              alpha_frac=0.3, vector_provenance="toy::naive")])
          and "naive" in TRANSPORTED_CELL_KINDS)
    cgate = evaluate_replay_gate(
        selection=csel, strata=cstrata, digest=cdig,
        token_first={c: f"{i:064x}" for i, c in enumerate(csel)},
        token_replay={c: f"{i:064x}" for i, c in enumerate(csel)},
        entropy_first={c: f"{i + 5:064x}" for i, c in enumerate(csel)},
        entropy_replay={c: f"{i + 5:064x}" for i, c in enumerate(csel)},
        role_mapping=REPLAY_ROLE_MAPPING_CALIBRATION_ONLY)
    check("(B-1) the gate RECORD names which role mapping filled the strata, and "
          "quotes the dated ruling's reading",
          cgate.role_mapping == REPLAY_ROLE_MAPPING_CALIBRATION_ONLY
          and "2026-08-05" in cgate.selection_rule
          and CALIBRATION_ONLY_ROLE_READING in cgate.selection_rule
          and cgate.passed)
    check("(B-1) a transported column's gate record still reads `of_record` and its "
          "selection_rule is UNCHANGED (no ruling text leaks onto it)",
          gate.role_mapping == REPLAY_ROLE_MAPPING_OF_RECORD
          and "2026-08-05" not in gate.selection_rule
          and gate.selection_rule.endswith("(§2.7)"))
    check("(B-1) a calibration-only column's gate is just as BLOCKING — a token-id "
          "mismatch on it HALTs like any other",
          _raises(lambda: evaluate_replay_gate(
              selection=csel, strata=cstrata, digest=cdig,
              token_first={c: f"{i:064x}" for i, c in enumerate(csel)},
              token_replay={c: "f" * 64 for c in csel},
              entropy_first={c: f"{i + 5:064x}" for i, c in enumerate(csel)},
              entropy_replay={c: f"{i + 5:064x}" for i, c in enumerate(csel)},
              role_mapping=REPLAY_ROLE_MAPPING_CALIBRATION_ONLY),
              ReplayGateNotBitwise))

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
    check("the stamp NAMES which sampling kernel drew the tokens",
          stamp["sampler_kernel"]["name"] == SAMPLER_KERNEL_OF_RECORD.name
          and bool(stamp["sampler_kernel"]["provenance"]),
          f"{stamp['sampler_kernel']['name']} — the two are certified byte-identical, "
          "so this can never explain a difference in the numbers, which is exactly "
          "why it is recorded rather than inferred")
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

    # ---- 8b. B-2: the naive cell's four nullable map fields (RULED 2026-08-05) --
    print("== selftest 8b: B-2's naive-cell stamp nullability ==")
    naive_stamp = dict(stamp, transport_map_fit_sha256=None, transport_map_family=None,
                       transport_map_arm=None, transport_map_corpus_vintage=None)
    check("(B-2) a NAIVE cell with all four transport_map_* fields null passes the "
          "§2.8 checklist (ruling 5: it rides no map BY CONSTRUCTION)",
          _ok(lambda: assert_stamp_complete(naive_stamp, cell_kind="naive")),
          "transport_map_fit_sha256/family/arm/corpus_vintage")
    check("(B-2) the naive nullable set is EXACTLY the four transport_map_* fields",
          NAIVE_NULLABLE_FIELDS == frozenset({
              "transport_map_fit_sha256", "transport_map_family",
              "transport_map_arm", "transport_map_corpus_vintage"})
          and all(f in STAMP_REQUIRED_FIELDS for f in NAIVE_NULLABLE_FIELDS)
          and "naive_transplant_gate_row" not in NAIVE_NULLABLE_FIELDS
          and "naive_transplant_verdict" not in NAIVE_NULLABLE_FIELDS,
          str(sorted(NAIVE_NULLABLE_FIELDS)))
    check("(B-2) a naive cell whose gate ROW is null still HALTs (the gate row is what "
          "a naive cell is FOR — §5.3 item 1 / §9 item 7)",
          _raises(lambda: assert_stamp_complete(
              dict(naive_stamp, naive_transplant_gate_row=None), cell_kind="naive"),
              StampIncompleteError))
    check("(B-2) a naive cell whose gate VERDICT is null still HALTs",
          _raises(lambda: assert_stamp_complete(
              dict(naive_stamp, naive_transplant_verdict=None), cell_kind="naive"),
              StampIncompleteError))
    for _f in ("naive_transplant_gate_row", "naive_transplant_verdict"):
        _holed = {k: v for k, v in naive_stamp.items() if k != _f}
        check(f"(B-2) a naive cell MISSING {_f} still HALTs (absence is never a pass)",
              _raises(lambda h=_holed: assert_stamp_complete(h, cell_kind="naive"),
                      StampIncompleteError))
    check("(B-2) a naive cell null on any OTHER required field still HALTs — the "
          "widening is four fields, not a kind-wide exemption",
          all(_raises(lambda f=f: assert_stamp_complete(
              dict(naive_stamp, **{f: None}), cell_kind="naive"),
              StampIncompleteError)
              for f in ("model_config_sha256", "vector_npz_sha256",
                        "replay_gate_digests", "canonical_batch_layout",
                        "battery_item_set_sha256")))
    check("(B-2) non-naive kinds are UNCHANGED: a transported cell's null map field "
          "is still refused, and an unnamed kind gets no nullable set at all",
          _raises(lambda: assert_stamp_complete(naive_stamp, cell_kind="transported"),
                  StampIncompleteError)
          and _raises(lambda: assert_stamp_complete(naive_stamp,
                                                    cell_kind="transported_band"),
                      StampIncompleteError)
          and _raises(lambda: assert_stamp_complete(naive_stamp, cell_kind="bridge"),
                      StampIncompleteError)
          and _raises(lambda: assert_stamp_complete(naive_stamp, cell_kind=None),
                      StampIncompleteError))
    check("(B-2) the nullable-kind table names exactly the ruled kinds, and the "
          "calibration half's set is untouched by the ruling",
          set(NULLABLE_FIELDS_BY_KIND) == {"calibration", "calibration_band",
                                           "baseline", "naive"}
          and NULLABLE_FIELDS_BY_KIND["calibration"] == CALIBRATION_NULLABLE_FIELDS
          and NULLABLE_FIELDS_BY_KIND["naive"] == NAIVE_NULLABLE_FIELDS
          and NAIVE_NULLABLE_FIELDS < CALIBRATION_NULLABLE_FIELDS)
    check("(B-2) a naive cell's stamp BUILDS end-to-end through `build_stamp` with a "
          "banked gate row and no map (the path the column actually takes)",
          _ok(lambda: build_stamp(
              cell=CellSpec(
                  cell_id=CELL_ID_TEMPLATE.format(
                      vector_key="naive_entropy_gradient", site=site, frac=0.3),
                  kind="naive", vector_key="naive_entropy_gradient", site=site,
                  alpha_frac=0.3, vector_provenance="toy::naive transplant"),
              alpha=0.3 * norm,
              layout=freeze_layout(80, dtype="bfloat16", headroom_note="selftest"),
              pool=pool, norms=nc, corpus_sha=corpus, node_key=node, arm=arm,
              site_cross_check={"SITES": [site], "SITE_OF_RECORD": site,
                                "agrees": True},
              model_config_sha256="c" * 64, vector_npz_sha256="d" * 64,
              vector_fd_gate={"PASSES": True},
              vector_build_stamp={"builder": "selftest"},
              transport_map=None,
              naive_row={"pair": "8b->qwen2.5-3b-instruct", "verdict": "CLEAR",
                         "bare_cos": 0.04, "q95": 0.09},
              trunk={"transformers": "5.3.0", "hostname": "selftest"},
              replay_gate_digests={"token_ids": "d" * 64, "entropy": "e" * 64},
              battery_item_set_sha256="f" * 64,
              actuation_calibration={"job_id": "selftest", "verdict": "PASS"},
              per_cell_seed_roots=roots)),
          "transport_map=None + a banked naive row")
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

    # The BatchEncoding shape. `apply_chat_template` returns a bare list for some
    # tokenizers and a mapping for others — both appear on this roster at one
    # transformers version — and iterating the mapping yields its KEYS, which is
    # how `int("input_ids")` reached a live node (3B certification, 2026-08-05).
    # The stub below returns the mapping and nothing else changes, so this check
    # fails the moment the unwrap is removed again.
    class _MappingTemplateTokenizer(_ToyTokenizer):
        def apply_chat_template(self, messages: list[dict],
                                add_generation_prompt: bool = True,
                                **kwargs: Any) -> Any:
            ids = _ToyTokenizer.apply_chat_template(
                self, messages, add_generation_prompt=add_generation_prompt,
                **kwargs)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    _mtok = _MappingTemplateTokenizer()
    check("a chat template returning a BatchEncoding is unwrapped, not iterated "
          "(the collection lane's own `res['input_ids'] if hasattr(res,'keys')`)",
          render_prompt(pool.prompts[0], _mtok, "native")
          == render_prompt(pool.prompts[0], tok, "native"))

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
    if _torch is None:
        skip("real-forward characterization: the REAL HF path, the batch-invariance\n         characterization, §2.4 on the real hook, and the hook-admissibility preflight",
             f"{_no_torch} — the stub-stepper proofs above are the GATE; this "
             "block only ever CHARACTERIZED (M19), so its absence costs no "
             "assertion")
    else:
      try:
          from transformers import LlamaConfig, LlamaForCausalLM

          _torch.manual_seed(20260729)
          cfg = LlamaConfig(vocab_size=61, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=128,
                            tie_word_embeddings=False)
          real = LlamaForCausalLM(cfg).to(_torch.float32).eval()
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
          # THE IDENTITY GATE, END TO END ON A REAL FORWARD (the torch half of
          # selftest 2b). Same model, same layout, same tapes, both kernels — so the
          # gate covers real logits (fp32 out of a real lm_head, not synthesized)
          # arriving through `HFStepper._logits`, whose D2H order changed with the
          # vectorization. Bitwise or nothing: unlike the B=10-vs-B=20
          # characterization below, this one CAN fail, because nothing about it is
          # allowed to depend on a reduction order.
          r10_ref = generate_cell(lambda: HFStepper(real, "cpu"), cell=cell, pool=pool,
                                  tokenize=lambda p: render_prompt(p, tok, arm),
                                  layout=small, pad_token_id=0, eos_token_id=None,
                                  corpus_sha=corpus, node_key=node, arm=arm, n=20,
                                  sampler_kernel=SAMPLER_KERNEL_REFERENCE)
          check("toy-Llama end to end: both kernels draw the SAME 100 tokens "
                "(a real forward's logits, not synthetic ones)",
                token_id_digest(r10_ref) == token_id_digest(r10)
                and [r.generated_ids for r in sorted(
                    r10_ref, key=lambda x: x.generation_id)]
                == [r.generated_ids for r in sorted(
                    r10, key=lambda x: x.generation_id)],
                f"20 generations × 5 tokens, digest {token_id_digest(r10)[:16]}… "
                f"under both kernels; vocab {cfg.vocab_size}, B=10")
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
          ids0 = _torch.as_tensor([render_prompt(pool.prompts[0], tok, arm)],
                                 dtype=_torch.long)
          with _torch.no_grad():
              plain = real(ids0, use_cache=False).logits.detach().numpy()
          h = attach_residual_write(real, ResidualWriteSpec(
              layer_idx=1, vector=_torch.randn(32), alpha=0.0,
              start_pos=int(ids0.shape[1]), end_pos=None, normalize=True))
          try:
              with _torch.no_grad():
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
          # ---- the REAL runtime, end to end ---------------------------------
          # What the stub runtime proves about §2.1's ORDER and custody, this proves
          # about the torch plumbing: one hook mutated across cells, the two-forward
          # probe, the battery under injection, the in-job norm measurement and the
          # replay gate, all against a genuine (tiny) forward.
          hidden = int(cfg.hidden_size)
          basis_vectors = {k: np.eye(hidden, dtype=np.float32)[i]
                           for i, k in enumerate(("entropy_gradient",
                                                  "gentropy_gradient", "gRband1"))}
          # a node key the registries do NOT carry, on purpose: block 14 exercises
          # a REGISTERED node (the §9-item-4 cross-check must agree there), and this
          # block exercises the torch plumbing on a model that is not any node.
          toy_node = "toy-llama"
          rt = HFNodeRuntime(real, tok, node_key=toy_node, arm=arm, site=1,
                             vectors=basis_vectors, device="cpu",
                             dtype_name="float32")
          cdoc = _toy_document(pool, node=toy_node, arm=arm, site=1, corpus=corpus,
                               cells=_compact_cells(1), n_per_cell=2,
                               max_new_tokens=3)
          with tempfile.TemporaryDirectory(prefix="behav_real_") as rtd:
              real_column = run_column(rt, doc=cdoc, pool=pool, work_root=Path(rtd),
                                       corpus_sha_of_record=corpus,
                                       characterize=False,
                                       scheduler_card_index="cpu-selftest")
              banked_manifest = dict(real_column.manifest)
          check("the REAL runtime runs the whole §2.1 job through one loaded model",
                len(real_column.cells) == 7 and real_column.n_generations == 14,
                f"B={real_column.layout.batch_size}, "
                f"measured norm={real_column.norms.measured_per_token_median:.4f}")
          check("the §2.7 replay gate is bitwise on a REAL forward, same layout",
                real_column.replay_gate.passed
                and not real_column.replay_gate.mismatches,
                json.dumps(real_column.replay_gate.strata))
          check("§2.5's in-job per-token median residual norm is measured, positive",
                real_column.norms.measured_per_token_median > 0
                and "MEASURED in-job" in real_column.norms.measured_provenance)
          check("§2.4 holds inside the job: α=0 with the hook attached is no-hook",
                real_column.alpha_zero_is_no_hook is True)
          check("the α=0 baseline's rise is EXACTLY zero on the real model",
                abs(next(c for c in real_column.cells
                         if c.kind == "baseline").entropy["entropy_rise"]) == 0.0)
          check("a dosed cell's steered and unsteered probes genuinely differ",
                any(c.entropy["entropy_rise"] != 0.0 for c in real_column.cells
                    if c.alpha_frac != 0.0),
                str({c.cell_id: c.entropy["entropy_rise"]
                     for c in real_column.cells if c.alpha_frac == 0.3}))
          check("the battery scored under injection on the real model (§6)",
                all(c.capability is not None for c in real_column.cells)
                and all(set(c.capability["likelihood_rates"])
                        == {"factual", "arithmetic", "syntactic_agreement"}
                        for c in real_column.cells))
          check("every real-run stamp passes the §2.8 checklist",
                all(_ok(lambda s=c.stamp, k=c.kind:
                        assert_stamp_complete(s, cell_kind=k))
                    for c in real_column.cells))
          check("the real run banked a manifest over every artifact it wrote",
                len(banked_manifest["artifacts"]) >= 4 * len(real_column.cells)
                and banked_manifest["replay_gate_passed"] is True,
                f"{len(banked_manifest['artifacts'])} artifacts")
      except ImportError as exc:                    # transformers availability
        skip("real-forward characterization (transformers unavailable)",
             f"{exc} — the stub-stepper proofs above are the gate")

    # ---- 14. THE COLUMN JOB, end to end, weightless (§2.1) --------------------
    print("== selftest 14: the §2.1 column job, orchestrated on a stub runtime ==")
    stub = _StubRuntime(pool, tok, arm, site)
    doc = _toy_document(pool, node=node, arm=arm, site=site, corpus=corpus,
                        cells=_toy_cells(site), n_per_cell=4, max_new_tokens=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    with tempfile.TemporaryDirectory(prefix="behav_column_") as td:
        work = Path(td)
        column = run_column(stub, doc=doc, pool=pool, work_root=work,
                            corpus_sha_of_record=corpus, characterize=True,
                            scheduler_card_index="3")
        check("the column runs every staged cell exactly once",
              len(column.cells) == len(doc.cells) == 25
              and len({c.cell_id for c in column.cells}) == 25,
              f"{len(column.cells)} cells × n={doc.n_per_cell} = "
              f"{column.n_generations} generations")
        fired = [c.cell_id for c in column.cells]
        kinds = [c.kind for c in column.cells]
        check("§2.1's order fires calibration BEFORE anything transported (§4 is a GATE)",
              max(i for i, k in enumerate(kinds)
                  if k in ("calibration", "calibration_band"))
              < min(i for i, k in enumerate(kinds)
                    if k in ("transported", "transported_band")),
              f"first: {fired[0]}, last: {fired[-1]}")
        check("the runtime saw the cells in exactly the filed order",
              stub.order[:len(fired)] == fired)
        check("α is resolved from the MEASURED per-token median, per cell (§2.5)",
              all(abs(c.alpha - c.alpha_frac * 12.2391) < 1e-12 for c in column.cells)
              and column.norms.measured_per_token_median == 12.2391,
              f"α(+0.3) = {[c.alpha for c in column.cells if c.alpha_frac == 0.3][0]}")
        check("the banked mean-STATE norm rides beside and never sets α",
              column.norms.banked_mean_state_median == 12.1125
              and "PER-TOKEN" in column.norms.used_for_alpha)
        check("the canonical layout is FROZEN before any cell fires (§2.2)",
              column.layout.frozen and column.layout.batch_size in BATCH_LADDER
              and column.layout.measured, f"B={column.layout.batch_size}")
        check("the §2.7 replay gate PASSED bitwise in the canonical layout",
              column.replay_gate.passed and len(column.replay_gate.cells) == 3
              and not column.replay_gate.mismatches,
              json.dumps(column.replay_gate.strata))
        check("the gate's three strata are the frozen ones",
              sorted(column.replay_gate.strata) == sorted(REPLAY_GATE_STRATA))
        check("every generation's entropy rise is a difference of two reads of ONE text",
              all(len(c.probe_rows) == c.n for c in column.cells)
              and all(r.n_positions > 0 for c in column.cells for r in c.probe_rows))
        base_cell = next(c for c in column.cells if c.kind == "baseline")
        dosed = next(c for c in column.cells
                     if c.kind == "transported" and c.alpha_frac == 0.3)
        check("the α=0 baseline shows no rise and a dosed cell does (dose response)",
              abs(base_cell.entropy["entropy_rise"]) < 1e-9
              and dosed.entropy["entropy_rise"] > 0.0,
              f"baseline {base_cell.entropy['entropy_rise']} vs +0.3 "
              f"{dosed.entropy['entropy_rise']}")
        check("§6's battery rode EVERY cell, bands included (the dose-matched floor)",
              all(c.capability is not None for c in column.cells)
              and all(c.coherence is not None for c in column.cells))
        check("the battery's deltas are formed against the α=0 cell, which has none",
              base_cell.capability["baseline_available"] is False
              and dosed.capability["baseline_available"] is True
              and set(dosed.capability["delta_vs_alpha0"]) >= {"factual", "format"})
        check("the capability table carries the random band beside every dose (§6)",
              column.capability_table is not None
              and set(column.capability_table.get("by_dose", {}))
              >= {"+0.3", "-0.3"},
              str(sorted(column.capability_table.get("by_dose", {}))))
        check("§2.7's B=8/B=1 characterization is DESCRIPTIVE and recorded (M19)",
              column.batch_invariance is not None
              and column.batch_invariance.compared_batch_sizes == [1, 8],
              json.dumps(column.batch_invariance.bitwise_identical))
        check("the off-ladder characterization sizes never became a layout of record",
              column.layout.characterization_only is False and column.layout.frozen)
        check("the fresh-process replay is filed as OWED with its recipe (§2.7, M19)",
              column.fresh_process_replay["state"] == "OWED"
              and "CUBLAS_WORKSPACE_CONFIG" in column.fresh_process_replay["recipe"])
        check("every cell's stamp passes the §2.8 checklist",
              all(_ok(lambda s=c.stamp, k=c.kind: assert_stamp_complete(s, cell_kind=k))
                  for c in column.cells))
        check("every stamp names the entropy-digest ORDER (so the gate is reproducible)",
              all("gen_id order" in c.stamp["entropy_digest_order"]
                  for c in column.cells))
        check("every stamp names the battery's injection span",
              all("SCORED span" in c.stamp["battery_injection_span"]
                  for c in column.cells))
        check("the actuation calibration is named on every cell — OWED, never absent",
              all(c.stamp["actuation_calibration"]["verdict"] == "OWED"
                  for c in column.cells))
        check("raw sits next to every claim: generations + entropy + probe rows banked",
              all(Path(c.raw_paths["generations"]).exists()
                  and Path(c.raw_paths["entropy"]).exists()
                  and Path(c.raw_paths["probe_rows"]).exists()
                  for c in column.cells))
        one = column.cells[0]
        check("a banked generations file has exactly n lines",
              len(Path(one.raw_paths["generations"]).read_text().splitlines())
              == one.n)
        check("the manifest sha's every artifact the job wrote (M4-checkable)",
              len(column.manifest["artifacts"]) >= 4 * len(column.cells)
              and all(len(v) == 64 for v in column.manifest["artifacts"].values()),
              f"{len(column.manifest['artifacts'])} artifacts")
        check("the column result banks the replay verdict in its manifest",
              column.manifest["replay_gate_passed"] is True)
        check("the rehearsal label rides the column and every stamp",
              column.label == "SELFTEST, NOT A READ"
              and all(c.stamp["label"] == "SELFTEST, NOT A READ"
                      for c in column.cells))
        check("an expected-N shortfall in the column is a HALT (§9 item 10 / M23)",
              _raises(lambda: assert_column_complete(column.cells, 26, n_per_cell=4),
                      ExpectedNShortfall)
              and _raises(lambda: assert_column_complete(column.cells,
                                                         len(column.cells),
                                                         n_per_cell=80),
                          ExpectedNShortfall))
        # B4: expected-N is a gate, so a beside may not fill a hole in it, from either
        # side — a planned beside that did not run (the total is right because a cell
        # of record ran in its place), or a cell that stamped itself a beside without
        # being planned as one.
        _smuggled = column.cells[0].model_copy(update={
            "stamp": {**column.cells[0].stamp, "is_beside_only": True}})
        check("(b) expected-N refuses to let a Σ-beside stand in for a cell of record",
              _ok(lambda: assert_column_complete(
                  column.cells, len(column.cells), n_per_cell=4, beside_cell_ids=()))
              and _raises(lambda: assert_column_complete(
                  column.cells, len(column.cells), n_per_cell=4,
                  beside_cell_ids=[f"SigmaBand1_L{site}_a+0.30"]), ExpectedNShortfall)
              and _raises(lambda: assert_column_complete(
                  [_smuggled] + list(column.cells[1:]), len(column.cells),
                  n_per_cell=4, beside_cell_ids=()), ExpectedNShortfall))
        check("a pool whose sha disagrees with the document is a HALT (M4)",
              _raises(lambda: run_column(
                  _StubRuntime(pool, tok, arm, site), doc=doc.model_copy(
                      update={"prompt_pool_sha256": "9" * 64}),
                  pool=pool, work_root=None, corpus_sha_of_record=corpus,
                  characterize=False, write=False), ArtifactShaMismatch))
        check("a non-of-record basis in the chain is a HALT (§9 item 2)",
              _raises(lambda: run_column(
                  _StubRuntime(pool, tok, arm, site), doc=doc, pool=pool,
                  work_root=None, corpus_sha_of_record="7" * 64, characterize=False,
                  write=False), CorpusVintageError),
              "the basis is a PARAMETER — the engine hardcodes none for a run")
        # M10: the preflight is a first-class mode and resolves without firing a cell
        weightless = preflight_report(doc, pool, corpus_sha_of_record=corpus)
        loaded = preflight_report(doc, pool, runtime=_StubRuntime(pool, tok, arm, site),
                                  corpus_sha_of_record=corpus)
        check("the weightless preflight constitutes the replay gate before any spend",
              weightless["replay_gate"]["constituted"]
              and len(weightless["replay_gate"]["cells"]) == 3)
        check("the weightless preflight NAMES what it could not check",
              weightless["loaded_checks"] == "NOT RUN (no model handed to the "
                                             "preflight)")
        check("the loaded preflight reports the layout, the norm and the trunk",
              loaded["loaded_checks"]["canonical_batch_size"] in BATCH_LADDER
              and loaded["loaded_checks"]["measured_per_token_median_resid_norm"] > 0)
        check("the preflight's cell arithmetic matches the document",
              weightless["n_cells"] == 25 and weightless["n_generations"] == 100)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # ---- 15. the characterization layout (the §2.2/§2.7 clause conflict) -------
    print("== selftest 15: off-ladder characterization layouts are fenced ==")
    for B in CHARACTERIZATION_BATCH_SIZES:
        check(f"B={B} is admissible ONLY as a characterization layout",
              _raises(lambda b=B: CanonicalLayout(batch_size=b, dtype="f32"),
                      ValueError)
              and CanonicalLayout(batch_size=B, dtype="f32",
                                  characterization_only=True).batch_size == B)
    check("a characterization layout can NEVER be frozen (no second layout of record)",
          _raises(lambda: CanonicalLayout(batch_size=8, dtype="f32", frozen=True,
                                          characterization_only=True), ValueError))
    check("freeze_layout still produces a FROZEN on-ladder layout of record",
          freeze_layout(80, dtype="bfloat16").frozen
          and not freeze_layout(80, dtype="bfloat16").characterization_only)

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    # RAKE M44: the tail states coverage as well as the verdict, because a suite
    # that ran 120 of 127 checks in this configuration and one that ran all 127 are
    # different facts, and only the first needs the other configuration to close it.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    # A named skip is NOT a failure: nothing was asserted and found wanting. rc=0.
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


# ---------------------------------------------------------------- the sampler bench
def bench_sampler(*, batch: int = N_PER_CELL, vocab: int = 128256, steps: int = 8,
                  seed: int = CERTIFICATION_SEED) -> dict:
    """CPU cost of the two kernels over SYNTHETIC logits at the layout of record.

    What this is: the sampling step alone, isolated from the forward, because the
    sampling step is the only thing the 2026-08-01 vectorization touched. What this is
    NOT: a throughput number. The node-side seq/s certification measures a real
    forward on a real card and is a separate job — a ratio measured here does not
    become a column's speed-up, it bounds the part of the column that was slow for
    this reason.

    Identity is re-asserted inside the bench (the two kernels must draw the same
    tokens on the very blocks that were timed), so a bench run can never report a
    speed-up that was bought by drawing something else.
    """
    import time

    if steps <= 0 or batch <= 0 or vocab <= 0:
        raise ValueError(f"bench_sampler: nonsense shape ({batch}, {vocab}) × {steps}")
    rng = np.random.default_rng(seed)
    # Two blocks rotating: 80 × 128256 float32 is 41 MB, and what is being timed is
    # arithmetic, not allocation.
    blocks = [(rng.standard_normal((batch, vocab)) * 3.0).astype(np.float32)
              for _ in range(2)]
    us = [float(u) for u in rng.random(batch)]
    rows = list(range(batch))
    cert = assert_batched_sampler_identity(batch=batch, vocab=vocab)  # warm + gate

    def ref_step(blk: np.ndarray) -> list[int]:
        return [sample_token(blk[r], us[r], SAMPLING_OF_RECORD) for r in rows]

    def vec_step(blk: np.ndarray) -> list[int]:
        return sample_tokens(blk, us=us, rows=rows, sampling=SAMPLING_OF_RECORD,
                             kernel=SAMPLER_KERNEL_VECTORIZED)

    # THE PROTOCOL, and why it is not a stopwatch around two loops. Measured the naive
    # way — all of A, then all of B, once — the SAME code read 1.04× and 2.9× on two
    # runs: a laptop ramps its clock under sustained load and glibc adapts its mmap
    # threshold to what has already been freed, so whichever kernel runs LAST tends to
    # win. CPU time rather than wall clock (the machine is shared), warm-up discarded,
    # the order alternated every rep, and the median reported with the interquartile
    # range so a reader sees the spread rather than trusting a mean.
    want = [ref_step(b) for b in blocks]      # per block: two blocks, two answers
    for i in range(min(6, steps)):
        ref_step(blocks[i % 2])
        vec_step(blocks[i % 2])
    ref_t: list[float] = []
    vec_t: list[float] = []
    identical = True
    for i in range(steps):
        blk = blocks[i % 2]
        order = (ref_step, vec_step) if i % 2 == 0 else (vec_step, ref_step)
        for fn in order:
            t0 = time.process_time()
            got = fn(blk)
            dt = time.process_time() - t0
            (ref_t if fn is ref_step else vec_t).append(dt)
            identical = identical and got == want[i % 2]
    if not identical:
        raise VectorizedSamplerNotIdentical(
            "the benched blocks did not draw identical tokens — a speed-up bought by "
            "drawing something else is not a speed-up (this is why the bench asserts)")

    def _stat(xs: list[float]) -> tuple[float, float, float]:
        a = np.sort(np.asarray(xs, dtype=np.float64))
        return (float(np.median(a)), float(a[len(a) // 4]),
                float(a[(3 * len(a)) // 4]))

    ref_med, ref_lo, ref_hi = _stat(ref_t)
    vec_med, vec_lo, vec_hi = _stat(vec_t)
    return {
        "shape": {"batch": batch, "vocab": vocab, "steps": steps,
                  "tile_rows": sampler_tile_rows(vocab),
                  "tile_bytes": SAMPLER_TILE_BYTES},
        "certificate": cert,
        "reference_s_per_step": ref_med,
        "reference_iqr_s": [ref_lo, ref_hi],
        "vectorized_s_per_step": vec_med,
        "vectorized_iqr_s": [vec_lo, vec_hi],
        "reference_ms_per_draw": ref_med / batch * 1e3,
        "vectorized_ms_per_draw": vec_med / batch * 1e3,
        "speedup_x": (ref_med / vec_med) if vec_med > 0 else None,
        "tokens_identical": True,
        "n_draws_compared": 2 * steps * batch,
        "numpy": np.__version__,
        "reading": (
            "CPU time for the sampling step alone, synthetic logits, one process, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, median of {steps} "
            "order-alternated reps after warm-up. The node-side throughput "
            "certification is a separate job and is NOT this number."),
    }


# ---------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The node-side behavioral engine (BRIEF §2). Heimdall CLI only; "
                    "the coordinator HTTP API is read-only verification.")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only, data-independent verification (no GPU, no weights)")
    ap.add_argument("--bench-sampler", action="store_true",
                    help="CPU cost of the reference vs vectorized sampling kernels "
                         "over synthetic logits (identity re-asserted on the benched "
                         "blocks). Not a node throughput number.")
    ap.add_argument("--bench-batch", type=int, default=N_PER_CELL)
    ap.add_argument("--bench-vocab", type=int, default=128256)
    ap.add_argument("--bench-steps", type=int, default=8)
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
                         "<NODE_DATA_ROOT>/metabasis-behavioral/<node> (standing rule 2026-07-29)")
    ap.add_argument("--cells-json", type=Path, default=None,
                    help="staged cell specs (build_behavioral_banks.py output)")
    ap.add_argument("--prompt-pool", type=Path, default=None)
    ap.add_argument("--arm", choices=("native", "raw"), default=None,
                    help="cross-checked against the cells document; the document wins "
                         "and a disagreement is a refusal, never a silent override")
    ap.add_argument("--site", type=int, default=None,
                    help="cross-checked against the cells document (see --arm)")
    ap.add_argument("--n-per-cell", type=int, default=None,
                    help=f"override n/cell (frozen at {N_PER_CELL}; a reduced n is a "
                         "REHEARSAL and the document must carry a label saying so)")
    ap.add_argument("--corpus-sha-of-record", default=None,
                    help="the basis the vintage chain is asserted against. Defaults to "
                         "the document's own corpus sha, so this module hardcodes no "
                         "basis for a run — the open v2.1/v3 ruling lives here.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--scheduler-card-index", default=None,
                    help="M10: a scheduler job carries a card index; a rogue carries "
                         "the sentinel, and §9 item 9 refuses the stamp without it")
    ap.add_argument("--no-characterize", action="store_true",
                    help="skip §2.7's DESCRIPTIVE B=8/B=1 characterization (M19: it "
                         "can never fail a gate, so skipping it costs no assertion)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.bench_sampler:
        try:
            print(json.dumps(bench_sampler(batch=args.bench_batch,
                                           vocab=args.bench_vocab,
                                           steps=args.bench_steps), indent=1))
        except BehavioralHarnessError as exc:
            logger.error("HALT (%s): %s", type(exc).__name__, exc)
            return 2
        return 0
    if not (args.preflight or args.run):
        ap.error("nothing to do: pass --selftest, --bench-sampler, --preflight "
                 "or --run")

    from metabasis.scripts.build_behavioral_banks import load_cells_document

    missing = [f for f in ("cells_json", "prompt_pool")
               if getattr(args, f) is None]
    if args.run and args.model_path is None:
        missing.append("model_path")
    if missing:
        ap.error(f"--{' / --'.join(m.replace('_', '-') for m in missing)} required")

    try:
        doc = load_cells_document(args.cells_json)
        pool = load_prompt_pool(args.prompt_pool)
    except BehavioralHarnessError as exc:
        logger.error("HALT: %s", exc)
        return 2

    for name, given in (("node_key", args.node_key), ("arm", args.arm),
                        ("site", args.site)):
        if given is not None and getattr(doc, name) != given:
            logger.error(
                "HALT: --%s %r disagrees with the cells document's %r. The document is "
                "the staged column and a CLI override would run one column under "
                "another column's stamp.", name.replace('_', '-'), given,
                getattr(doc, name))
            return 2
    basis = args.corpus_sha_of_record or doc.corpus_manifest_sha256
    logger.info("cells document: %s L%d (%s arm), %d cells, basis %s%s",
                doc.node_key, doc.site, doc.arm, len(doc.cells), basis[:12],
                f" — LABEL: {doc.label}" if doc.label else "")
    logger.info("prompt pool of record: %d prompts, sha %s",
                len(pool.prompts), pool.sha256)
    logger.info("dose ladder (FROZEN): %s; n/cell %d; max_new_tokens %d",
                list(DOSE_LADDER), args.n_per_cell or doc.n_per_cell,
                doc.max_new_tokens)
    logger.info("sampling of record: %s", SAMPLING_OF_RECORD.model_dump())

    runtime: Optional[NodeRuntime] = None
    try:
        if args.model_path is not None:
            model, tok, dtype_name = load_model_and_tokenizer(
                args.model_path, dtype_name=args.dtype)
            runtime = HFNodeRuntime(
                model, tok, node_key=doc.node_key, arm=doc.arm, site=doc.site,
                vectors=load_vectors(Path(doc.vectors_npz) if doc.vectors_npz
                                     else None),
                dtype_name=dtype_name)
        if args.preflight:
            report = preflight_report(doc, pool, runtime=runtime,
                                      corpus_sha_of_record=basis)
            print(json.dumps(report, indent=1, default=str))
            return 0
        if runtime is None:                                   # pragma: no cover
            ap.error("--run needs --model-path")
        if args.work_root is None:
            ap.error("--run needs --work-root (all new node-side data lives under "
                     "<NODE_DATA_ROOT>/metabasis-behavioral/<node>, standing rule 2026-07-29)")
        result = run_column(
            runtime, doc=doc, pool=pool, work_root=args.work_root,
            corpus_sha_of_record=basis, n_per_cell=args.n_per_cell,
            characterize=not args.no_characterize,
            scheduler_card_index=args.scheduler_card_index)
        print(json.dumps({
            "node_key": result.node_key, "site": result.site, "arm": result.arm,
            "label": result.label or None,
            "n_cells": len(result.cells), "n_generations": result.n_generations,
            "canonical_batch_size": result.layout.batch_size,
            "measured_per_token_median_resid_norm":
                result.norms.measured_per_token_median,
            "replay_gate_passed": result.replay_gate.passed,
            "replay_gate_cells": result.replay_gate.cells,
            "work_root": result.work_root, "grade": result.grade}, indent=1))
        return 0
    except BehavioralHarnessError as exc:
        # §3/§9: an enactor never adjudicates a live HALT. The job stops, says which
        # condition fired, and leaves the adjudication to the desk.
        logger.error("HALT (%s): %s", type(exc).__name__, exc)
        return 2
    except (OSError, ImportError) as exc:
        logger.error("environment failure (%s): %s", type(exc).__name__, exc)
        return 3
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:                     # noqa: BLE001 — M19(a)
                logger.warning("runtime close failed (%s: %s)",
                               type(exc).__name__, exc)


if __name__ == "__main__":
    sys.exit(main())
