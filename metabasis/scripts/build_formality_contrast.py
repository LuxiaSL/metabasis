"""Build a model's FORMALITY-CONTRAST target vector (the second vector class).

The transport campaign's whole result column rides on ONE vector class (the
entropy-gradient target). This builder supplies a SECOND, structurally unrelated
class so the column can be read twice: a textbook CAA (contrastive-activation-
addition) difference-of-means over freshly SAMPLED generations under two opposed
system prompts.

    formality_contrast_L{site}
        = unit( mean_pairs( mean_pos h_formal[site] - mean_pos h_informal[site] ) )

over 20 topics x 2 user templates = 40 matched pairs, where each pair generates
once under a maximally FORMAL system prompt and once under a maximally CASUAL one
from the SAME (topic, template) user message and the SAME RNG seed, and `h[site]`
is the residual stream ENTERING decoder layer `site`, meaned over the GENERATED
positions only.

PROVENANCE — this is a LIFT, not a new recipe. The construction is taken verbatim
in behavior from the anamnesis pipeline:

    anamnesis: pipeline/anamnesis/scripts/vmb_a5_build_vectors.py
        build_v1               L90-128   (the recipe)
        _chat_ids              L56-62    (the chat-template call)
        _mean_resid_at_sites   L65-75    (the extraction)
        DATE_STRING / FORMAL_SYS / INFORMAL_SYS   L46-52  (the constants)

authorized by `docs/desk/briefs/BRIEF-class-probe-v1-2026-07-29.md` section 1,
sha256 568dcb610cc6c9df38fafca597ab3af16fb5609eda6f00b0a2010778b72d19c7. The
anamnesis builder was NEVER lifted before (the 2026-07-29 census corrected the
canon's premise); metabasis carried only banked *vectors* of this class, topping
out at 27B.

WHAT IS THE SAME, AND IS LOAD-BEARING (a fresh build must be comparable with the
banked ones, so none of this is a preference):

  * the 20 topics and their ORDER (`corpus/formality_topics_v1.json`, copied from
    the anamnesis prompt set with its source sha) — the topic index drives the seed;
  * the 2 user templates and the 2 system prompts, verbatim;
  * `date_string` pinned to the campaign canonical date;
  * `add_generation_prompt=True`;
  * `do_sample=True, top_p=0.9, max_new_tokens=160`, temperature per the model's
    preset of record;
  * **THE MATCHED-RNG DESIGN**: the seed `100000 + 10*topic_index + template_index`
    is set BEFORE EACH CONDITION, so the formal and the informal generation of one
    pair sample from an IDENTICAL random stream. This is what makes the difference a
    contrast rather than two independent draws, and it is proved rather than asserted:
    the RNG state is digested at each seeding and the per-pair digests are compared,
    banked in the stamp (`seed_table[*].matched_rng`) and checked by the selftest;
  * a pair whose generation returns fewer than 8 new tokens is SKIPPED with a
    warning (the recipe's guard) — but see the HALT below;
  * the aggregation: per-condition mean over generated positions, difference,
    mean over pairs, then unit-normalize. `raw_norm` (the pre-normalization norm)
    is banked, as in the precedent's stamp.

WHAT IS DELIBERATELY DIFFERENT, AND WHY (each one census-justified):

  * **Loader.** The anamnesis original does `.to("cuda")`, which cannot place a
    405B. The single-card path here is the collector's `load_model_and_tok`; the
    multi-card path is the collector's CERTIFIED sharded loader
    (`load_model_and_tok_sharded` + `build_even_layer_device_map`), imported rather
    than forked, so the sharding layout is the same object the >=70B rungs collected
    under and lands in the trunk stamp the same way. Inputs go on the loader's
    reported input-embedding device, never `.to(device)` on a dispatched model.
  * **No `--stage0-run` / `--a2-root`.** Those were anamnesis-banked corpora feeding
    the sibling V3/norms stages, which are NOT lifted (no V3, no norms builder).
    Site norms for a metabasis model live in its banked collection stamp
    (`median_token_resid_norms`), which is a read, not a build.
  * **HALT, not warn, on a short pair.** The recipe skipped and carried on with
    fewer pairs; a vector built from 39 pairs is a different object from one built
    from 40 and is not comparable with the banked builds. `n_pairs_effective == 40`
    is asserted at the end and raises `PairCountError`. `n_topics == 20` is asserted
    when the manifest loads (the 8-pairs-instead-of-40 rake: the anamnesis prompt
    file's `topics` key is a DICT, and iterating it yielded 4 set names).
  * **Atomic banking.** Nothing is written until the whole build has succeeded, and
    then every file is written to a temp name and `os.replace`d, with rollback of
    everything already placed if any later write fails. The 405B build is a
    preemptable job: a partial bank is worse than no bank.
  * **All requested sites come out of ONE extraction pass.** `--sites 99,107` costs
    exactly what `--sites 99` costs; the extra site is free.

EXTRACTION CONVENTION. `output_hidden_states=True` and `hidden_states[s]` is the
residual stream ENTERING decoder layer `s` — the SAME capture point as
`collect_mean_states.py`'s `forward_pre_hook` on `decoder_layers(model)[s]`, so the
vector lives in the raw residual space the transport maps were fit in. That is not
an argument, it is a MEASUREMENT: `--selftest` runs one forward with the collector's
`SiteCapture` attached AND `output_hidden_states=True` and asserts the two are
BITWISE equal at every site. (`hidden_states` has `n_layers + 1` entries; only
`s < n_layers` is the "entering layer s" convention, which is why the site range
check exists.)

KNOWN CAVEAT, CARRIED IN THE STAMP: **sharded-generation reproducibility is NOT
certified.** The prereg §4 sharding gate (PASSED 2026-07-27) compared FORWARD passes
and found them byte-identical to single-device. This builder GENERATES, and a
sampled decode reads the RNG once per step through kernels whose selection can
depend on placement. The seeding is deterministic and recorded; bitwise equality of
a sharded generation against a single-device one is not claimed anywhere. Every
stamp says so (`generation_reproducibility`), and the shard spec is recorded beside it.

CPU self-test (no weights, no GPU, no data tree):

    python -m metabasis.scripts.build_formality_contrast --selftest

The topic manifest is tracked CODE, so the manifest/seed/aggregation/stamp/atomicity
blocks run anywhere the CPU spine runs. The blocks that need a model (the
hidden-state-convention proof, the end-to-end 40-pair build, the assertion-fires
controls) are SKIPPED with a named line when torch/transformers are absent, so a
CPU-spine sweep gets a clean partial count instead of a traceback.

Node-side runs (one card for the controls; sharded for the 405B):

    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \\
    python -m metabasis.scripts.build_formality_contrast \\
        --model 8b --model-path <LOCAL_WEIGHTS_DIR> --sites 16,14 \\
        --out-dir <VECTORS_DIR>

    OMP_NUM_THREADS=1 python -m metabasis.scripts.build_formality_contrast \\
        --model llama-3.1-405b-instruct --model-path <LOCAL_WEIGHTS_DIR> \\
        --sites 99,107 --shard-across 8 --temperature 0.6 \\
        --eos-token-ids 128001,128008,128009 --out-dir <VECTORS_DIR>

The 405B needs `--temperature` / `--eos-token-ids` explicitly: it has NO row in
`metabasis.config.MODEL_PRESETS`, and this builder refuses to guess a sampling
config (see `resolve_sampling`).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_formality_contrast")


# ---------------------------------------------------------------- frozen recipe
#: The anamnesis source of record for every constant and every line of the loop
#: below. Banked verbatim in every stamp so a reader can diff the recipe.
RECIPE_SOURCE = (
    "anamnesis pipeline/anamnesis/scripts/vmb_a5_build_vectors.py::build_v1 "
    "L90-128 (helpers _chat_ids L56-62, _mean_resid_at_sites L65-75, constants "
    "L46-52)")
RECIPE_BRIEF = (
    "BRIEF-class-probe-v1-2026-07-29.md section 1, sha256 "
    "568dcb610cc6c9df38fafca597ab3af16fb5609eda6f00b0a2010778b72d19c7")

#: The two user templates, verbatim (order load-bearing: the template index is
#: part of the seed).
TEMPLATES: tuple[str, ...] = ("Write about {topic}.",
                             "Explain {topic} to a beginner.")

#: The two system prompts, verbatim from the anamnesis constants block.
FORMAL_SYS = ("You are an extremely formal assistant. Respond with maximal formality: "
              "precise, ceremonious, professional register; no contractions, no "
              "colloquialisms, no humor.")
INFORMAL_SYS = ("You are a super casual assistant. Keep it loose and chatty — use slang, "
                "contractions, and casual asides, like you're texting a friend.")

#: sha256 of each system prompt as it stands in the anamnesis source (verified by
#: exec-ing that file's constants block, 2026-07-29, rather than by eyeballing the
#: copy). Pinned so "verbatim" is a CHECK a selftest can fail on — a re-wrapped line
#: or a smart-quote substitution would build a silently different contrast — and
#: banked in every stamp so a reader can verify the prompts without this file.
FORMAL_SYS_SHA256 = "988cd7cf46c3b76b1f4ad87ff627775bc018f94b39eb47bca1005538bfed0a40"
INFORMAL_SYS_SHA256 = "cdad622b753e31ce0fb94f8397613825b14e577037687d23af5a504753e7787c"

#: (condition name, system prompt) in the order the recipe generates them. The
#: FORMAL condition is the MINUEND: the sign convention is
#: "+ = more formal / less casual".
CONDITIONS: tuple[tuple[str, str], ...] = (("formal", FORMAL_SYS),
                                           ("informal", INFORMAL_SYS))
SIGN_CONVENTION = "+ = formal minus informal (more formal / less casual)"

#: Sampling, verbatim. `temperature` is per-model and is resolved, never defaulted.
DO_SAMPLE = True
TOP_P = 0.9
MAX_NEW_TOKENS = 160
#: A generation shorter than this makes the pair unusable (the recipe's guard).
MIN_NEW_TOKENS = 8

#: `100000 + 10*topic_index + template_index`, verbatim. With 20 topics and 2
#: templates the 40 seeds are distinct and none collides with a neighbour's
#: (10 > n_templates), which the selftest checks rather than assumes.
SEED_BASE = 100_000
SEED_TOPIC_STRIDE = 10

N_TOPICS_REQUIRED = 20
N_PAIRS_REQUIRED = N_TOPICS_REQUIRED * len(TEMPLATES)   # == 40

#: The tracked copy of the anamnesis topic set. CODE, not data: it is in the repo,
#: so every selftest that consumes it runs in a checkout with no data tree.
TOPIC_MANIFEST_NAME = "formality_topics_v1.json"

#: npz key. The bank key is CANONICAL and never carries the filename's model tag,
#: which is the convention every metabasis readout resolves on. It is deliberately
#: NOT the anamnesis bank key (`V1_L{site}`): the naming authority forbids V#-codes
#: in anything new. The equivalence is recorded in the stamp as provenance
#: (`anamnesis_bank_key_equivalent`) so the fresh-vs-banked anchor read knows what
#: to compare against.
CANONICAL_VECTOR_KEY = "formality_contrast_L{site}"
BANK_STEM = "formality_contrast_{model}_L{site}"
ANAMNESIS_BANK_KEY = "V1_L{site}"

#: Recorded on EVERY stamp, not conditionally. Unlike the collector's `sharding` /
#: `truncation` deviations there is no historical key set to protect here (this is
#: the first stamp this builder has ever written), and the caveat is a property of
#: the RECIPE — it generates — rather than of one run's configuration.
GENERATION_REPRODUCIBILITY = (
    "SAMPLED DECODE. Seeding is deterministic and fully recorded (see seed_table), "
    "but bitwise reproducibility of the generations is NOT certified. The prereg §4 "
    "sharding gate (PASSED 2026-07-27, byte-identical to single-device) covered "
    "FORWARD passes only; a sampled generation reads the RNG once per step through "
    "kernels whose selection can depend on device placement, and no gate has ever "
    "compared a sharded generation against a single-device one. Two builds of this "
    "recipe are documented NON-INTERCHANGEABLE independently of sharding (the "
    "anamnesis V1->V1b swap collapsed the portability coefficient to ~0): a fresh "
    "build is a NEW OBJECT, and the fresh-vs-banked anchor cosine is what sizes it.")

TOPIC_LEAKAGE_LIMIT = (
    "TOPIC LEAKAGE (named on every artifact): the 20 contrast topics ARE the frozen "
    "transport corpus's topic universe (set_a + set_b), so the transport maps were "
    "fit on text from the same topics this contrast is built from. The read is "
    "DESCRIPTIVE. Leakage inflates neither null (nulls are random-unit), but it is "
    "never omitted.")

NO_BUILD_GATE_LIMIT = (
    "NO FD-GATE ANALOGUE EXISTS FOR A CAA OBJECT. Build acceptance is the stamp's "
    "completeness + the n_pairs assertion + the anchor cosines read afterwards. The "
    "on-manifold covariance screen is NOT part of this builder (banked-Sigma "
    "availability varies); if it is wanted it is a separate cheap read.")

#: Every key `bank()` must have written. Asserted BEFORE the first byte lands, so a
#: stamp cannot go out incomplete; the selftest checks the same constant, which is
#: why the checklist is one object rather than two lists that could drift.
REQUIRED_STAMP_FIELDS: tuple[str, ...] = (
    "STATUS", "builder", "model", "site", "sites", "npz_keys",
    "anamnesis_bank_key_equivalent", "sign_convention",
    "recipe_provenance", "vector_provenance", "capture_convention",
    "topic_manifest", "n_pairs_expected", "n_pairs_effective", "n_pairs_skipped",
    "skipped_pairs", "sampling", "seed_table", "matched_rng_holds",
    "raw_norm", "unit_norm", "vector_dtype", "hidden_dim", "n_decoder_layers",
    "diagnostics", "date_string", "wall_seconds",
    "generation_reproducibility", "sharded_generation_certified",
    "named_limits", "trunk",
)


# ---------------------------------------------------------------- error taxonomy
class FormalityContrastBuildError(RuntimeError):
    """Base class for every failure specific to this builder. HALT semantics:
    raised, never swallowed, and nothing is banked when one escapes."""


class TopicManifestError(FormalityContrastBuildError):
    """The topic manifest is missing, malformed, or does not carry exactly 20 topics."""


class PairCountError(FormalityContrastBuildError):
    """Fewer than 40 usable pairs — the vector would not be comparable. HALT."""


class SiteOutOfRangeError(FormalityContrastBuildError):
    """A requested site is not a decoder-layer index of the loaded model."""


class HiddenStateConventionError(FormalityContrastBuildError):
    """`hidden_states` cannot supply the "entering layer s" tensor for a site."""


class UnknownModelError(FormalityContrastBuildError):
    """The model key is in neither metabasis registry (roster / MODEL_PRESETS)."""


class SamplingUnresolvedError(FormalityContrastBuildError):
    """The sampling config cannot be resolved and this builder will not guess it."""


class DtypeRegimeError(FormalityContrastBuildError):
    """The model's preset dtype is not the bf16 the shared loaders pin."""


class ChatTemplateError(FormalityContrastBuildError):
    """The checkpoint cannot take a system-role chat template — the class needs one."""


class BankCollisionError(FormalityContrastBuildError):
    """A target file already exists in the vectors dir. Census first, never clobber."""


class StampIncompleteError(FormalityContrastBuildError):
    """A stamp was assembled without a field `REQUIRED_STAMP_FIELDS` names."""


# ---------------------------------------------------------------- topic manifest
class TopicManifest(BaseModel):
    """The 20 contrast topics, their order, and the sha of the file they came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    sha256: str
    n_topics: int
    topics: tuple[str, ...]
    topic_sets_in_order: tuple[str, ...]
    source: str
    source_sha256: str

    @field_validator("topics")
    @classmethod
    def _exactly_twenty_distinct(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # THE 8-PAIRS-INSTEAD-OF-40 RAKE, as a type constraint. The anamnesis prompt
        # file's top-level `topics` is a DICT of set names; iterating it yielded 4
        # keys and built the vector from 8 pairs instead of 40 (caught 2026-07-13).
        if len(v) != N_TOPICS_REQUIRED:
            raise ValueError(
                f"the formality contrast needs exactly {N_TOPICS_REQUIRED} topics, "
                f"got {len(v)} ({list(v)[:4]}...) — if this is 4, the loader iterated "
                f"the topic-SET names instead of the topics")
        if len(set(v)) != len(v):
            raise ValueError("duplicate topic in the manifest: the 40 pairs would not "
                             "be 40 distinct (topic, template) cells")
        return v

    @property
    def stamp(self) -> dict[str, Any]:
        return {"path": self.path.name, "sha256": self.sha256,
                "n_topics": self.n_topics,
                "topic_sets_in_order": list(self.topic_sets_in_order),
                "topics": list(self.topics),
                "source": self.source, "source_sha256": self.source_sha256}


def topic_manifest_candidates(explicit: Optional[Path] = None) -> list[Path]:
    """Where the tracked topic manifest is looked for, in order.

    The package-relative probe comes before the cwd one so a run from any directory
    finds the manifest that ships with the code it is running (the cwd-dependence
    lesson of selftest 22, commit dcbe7d7). `--topics` always wins.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]                    # metabasis/scripts/x.py -> repo root
    out: list[Path] = []
    if explicit is not None:
        out.append(Path(explicit))
    out.append(repo_root / "corpus" / TOPIC_MANIFEST_NAME)
    out.append(Path.cwd() / "corpus" / TOPIC_MANIFEST_NAME)
    return out


def load_topic_manifest(explicit: Optional[Path] = None) -> TopicManifest:
    """Read + validate the topic manifest. Raises `TopicManifestError` on anything."""
    probed = topic_manifest_candidates(explicit)
    path = next((p for p in probed if p.is_file()), None)
    if path is None:
        raise TopicManifestError(
            f"{TOPIC_MANIFEST_NAME} not found; probed "
            f"{[str(p) for p in probed]}. It is tracked code (corpus/) — pass "
            f"--topics if the deploy moved it.")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TopicManifestError(f"{path}: not readable as JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise TopicManifestError(f"{path}: expected a JSON object, got "
                                 f"{type(doc).__name__}")
    try:
        sets_in_order = tuple(str(s) for s in doc["topic_sets_in_order"])
        topics = tuple(str(t) for t in doc["topics"])
        provenance = doc["provenance"]
        source = str(provenance["source"])
        source_sha = str(provenance["source_sha256"])
    except (KeyError, TypeError) as exc:
        raise TopicManifestError(
            f"{path}: missing or malformed key ({exc}); expected "
            "topic_sets_in_order / topics / provenance.{source,source_sha256}") from exc
    # The flattened list must BE the concatenation of the named sets in the named
    # order — otherwise the topic index (hence every seed) means something else than
    # the banked builds' index did.
    if "topic_sets" in doc:
        try:
            rebuilt = tuple(str(t) for s in sets_in_order for t in doc["topic_sets"][s])
        except (KeyError, TypeError) as exc:
            raise TopicManifestError(f"{path}: topic_sets does not cover "
                                     f"topic_sets_in_order ({exc})") from exc
        if rebuilt != topics:
            raise TopicManifestError(
                f"{path}: `topics` is not the concatenation of {list(sets_in_order)} "
                "in order — the topic index would not match the banked builds' index")
    try:
        return TopicManifest(
            path=path, sha256=hashlib.sha256(raw).hexdigest(), n_topics=len(topics),
            topics=topics, topic_sets_in_order=sets_in_order,
            source=source, source_sha256=source_sha)
    except ValueError as exc:
        raise TopicManifestError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------- sampling config
class SamplingResolution(BaseModel):
    """The sampling config, with WHERE EACH FIELD CAME FROM recorded beside it.

    `metabasis.config.MODEL_PRESETS` carries an explicit warning that some of its
    temperatures are transformers' 1.0 default rather than a checkpoint's native
    value, and that the desk should pin real temperatures before a GENERATION probe
    uses them. This builder IS a generation probe, so the source of the number is
    part of the record, and a key with no preset at all (the 405B) is a refusal
    rather than a guess.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: str
    registries: tuple[str, ...]
    temperature: float = Field(gt=0.0)
    temperature_source: str
    eos_token_ids: tuple[int, ...]
    eos_source: str
    torch_dtype: str
    dtype_source: str
    preset_hidden_dim: Optional[int] = None
    checkpoint_identity: Optional[str] = None

    @field_validator("eos_token_ids")
    @classmethod
    def _nonempty(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("eos_token_ids must not be empty (it also supplies the "
                             "pad id fall-back, exactly as the recipe does)")
        return v

    @property
    def stamp(self) -> dict[str, Any]:
        """The RESOLUTION only. The values a run can move (`max_new_tokens`, `top_p`)
        live on the request and are merged in by `sampling_stamp` — recording the
        module constants here would have a stamp claim 160 for a run that passed 12,
        which is the whole class of defect stamps exist to prevent."""
        return {"do_sample": DO_SAMPLE, "temperature": self.temperature,
                "temperature_source": self.temperature_source,
                "min_new_tokens_accepted": MIN_NEW_TOKENS,
                "eos_token_ids": list(self.eos_token_ids),
                "eos_source": self.eos_source,
                "torch_dtype": self.torch_dtype, "dtype_source": self.dtype_source,
                "registries": list(self.registries),
                "checkpoint_identity": self.checkpoint_identity}


def known_model_keys() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(roster keys, preset keys) — the two metabasis registries, never anamnesis."""
    from metabasis.config import MODEL_PRESETS
    from metabasis.roster import ROSTER
    return tuple(sorted(ROSTER)), tuple(sorted(MODEL_PRESETS))


def resolve_sampling(model_key: str, temperature: Optional[float] = None,
                     eos_token_ids: Optional[Sequence[int]] = None
                     ) -> SamplingResolution:
    """Resolve temperature / eos / dtype for `model_key`, or refuse loudly.

    Order per field: an explicit CLI value, then `metabasis.config.MODEL_PRESETS`,
    then a refusal. The roster is consulted for the key's existence and for
    `checkpoint_identity` (a base checkpoint cannot take a system-role template and
    is out of this class by construction) but carries no sampling fields.
    """
    from metabasis.config import MODEL_PRESETS
    from metabasis.roster import ROSTER

    roster_keys, preset_keys = known_model_keys()
    node = ROSTER.get(model_key)
    preset = MODEL_PRESETS.get(model_key)
    registries = tuple(n for n, hit in (("roster", node is not None),
                                        ("MODEL_PRESETS", preset is not None)) if hit)
    if not registries:
        raise UnknownModelError(
            f"unknown model key {model_key!r}. metabasis.roster.ROSTER: "
            f"{list(roster_keys)}; metabasis.config.MODEL_PRESETS: "
            f"{list(preset_keys)}. Model keys come from the metabasis registries, "
            "never from the anamnesis presets.")
    if node is not None and node.checkpoint_identity == "base":
        raise ChatTemplateError(
            f"{model_key} is a BASE checkpoint (roster row {node.roster_row}). The "
            "formality contrast is a SYSTEM-PROMPT contrast, so a base model is out "
            "of this vector class by construction (the precedent: olmo2-base has no "
            "banked vector of this class either). Not a bug to work around.")

    if temperature is not None:
        temp, temp_src = float(temperature), "explicit --temperature"
    elif preset is not None:
        temp = float(preset.temperature)
        temp_src = (f"metabasis.config.MODEL_PRESETS[{model_key!r}].temperature "
                    f"(identical to the anamnesis preset that built the banked "
                    f"vector of this class)")
    else:
        raise SamplingUnresolvedError(
            f"no temperature for {model_key!r}: it has no row in "
            f"metabasis.config.MODEL_PRESETS ({list(preset_keys)}), and this builder "
            "will not guess a sampling temperature — the vector is built FROM the "
            "samples. Pass --temperature explicitly with a desk-ruled value (the "
            "roster row's notes record the checkpoint's own generation_config "
            "temperature; the ruling that promotes prose to a build parameter is "
            "Luxia's, not this module's).")

    if eos_token_ids is not None:
        eos, eos_src = tuple(int(e) for e in eos_token_ids), "explicit --eos-token-ids"
    elif preset is not None:
        eos = tuple(int(e) for e in preset.eos_token_ids)
        eos_src = f"metabasis.config.MODEL_PRESETS[{model_key!r}].eos_token_ids"
    else:
        raise SamplingUnresolvedError(
            f"no eos_token_ids for {model_key!r} (no MODEL_PRESETS row). Pass "
            "--eos-token-ids explicitly: the stop condition decides how long every "
            "generation is, so it cannot be defaulted either.")

    if preset is not None:
        dtype, dtype_src = str(preset.torch_dtype), "MODEL_PRESETS.torch_dtype"
    else:
        dtype, dtype_src = "bfloat16", "the shared loaders' pin (no preset row)"
    if dtype != "bfloat16":
        # The collector's loaders pin bf16. Forwarding a float16-preset model through
        # them would build the vector in a different dtype regime than its preset
        # names, which is exactly the silent-wrong-answer shape. Surface it.
        raise DtypeRegimeError(
            f"{model_key}: MODEL_PRESETS says torch_dtype={dtype!r}, but "
            "collect_mean_states' loaders (single-card and sharded alike) pin "
            "bfloat16. Building through them would put this vector in a different "
            "dtype regime than the preset names. Needs a desk ruling, not a cast.")

    return SamplingResolution(
        model_key=model_key, registries=registries,
        temperature=temp, temperature_source=temp_src,
        eos_token_ids=eos, eos_source=eos_src,
        torch_dtype=dtype, dtype_source=dtype_src,
        preset_hidden_dim=int(preset.hidden_dim) if preset is not None else None,
        # Only the roster carries a VERIFIED checkpoint identity. MODEL_PRESETS has no
        # such field, so for a preset-only key the honest value is "not established
        # here" — never a plausible "instruct". The load-time guard that actually
        # protects this vector class is the tokenizer's chat template (see `main`),
        # which is evidence rather than a registry claim.
        checkpoint_identity=(node.checkpoint_identity if node is not None else
                             "UNVERIFIED (no roster row; MODEL_PRESETS carries no "
                             "checkpoint_identity field — the chat-template guard at "
                             "load time is what enforces the class requirement)"))


# ---------------------------------------------------------------- typed records
class BuildRequest(BaseModel):
    """Everything that defines a build, validated before a single weight is read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: str = Field(description="bank key; keys the filenames and the stamp")
    model_path: str = Field(description="local weights dir (node-side)")
    out_dir: Path
    sites: tuple[int, ...] = Field(
        description="decoder-layer indices; residual ENTERING each layer. ALL of them "
                    "come out of ONE extraction pass, so extra sites are free.")
    max_new_tokens: int = Field(gt=0, default=MAX_NEW_TOKENS)
    top_p: float = Field(gt=0.0, le=1.0, default=TOP_P)
    device: str = "cuda"
    allow_overwrite: bool = Field(
        default=False,
        description="bank over an existing file of the same name. OFF by default: the "
                    "ops rule is a bank census BEFORE any write into an existing "
                    "vectors dir, and no vector of this class exists in the v2.1 banks, "
                    "so a collision means something unexpected is there.")

    @field_validator("sites")
    @classmethod
    def _sites_sane(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("at least one site is required")
        if any(s < 0 for s in v):
            raise ValueError(f"sites must be non-negative decoder-layer indices, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate site in {v}")
        return v

    def stem(self, site: int) -> str:
        return BANK_STEM.format(model=self.model_key, site=site)


class ConditionRecord(BaseModel):
    """One generation of one pair — its seed, its RNG state, its length."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["formal", "informal"]
    prompt_len: int
    n_new_tokens: int
    seed: int
    #: sha256 of `torch.get_rng_state()` immediately AFTER seeding and BEFORE the
    #: generate call — i.e. of the stream the generation will consume.
    rng_state_sha256: str


class PairRecord(BaseModel):
    """One (topic, template) cell: its seed, both conditions, whether the RNG matched."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_index: int
    topic: str
    template_index: int
    template: str
    user_prompt: str
    seed: int
    conditions: tuple[ConditionRecord, ...] = ()
    matched_rng: bool = False
    accepted: bool = False
    skip_reason: Optional[str] = None

    @property
    def stamp(self) -> dict[str, Any]:
        return {"topic_index": self.topic_index, "topic": self.topic,
                "template_index": self.template_index, "seed": self.seed,
                "matched_rng": self.matched_rng, "accepted": self.accepted,
                "n_new_tokens": {c.condition: c.n_new_tokens for c in self.conditions},
                "prompt_len": {c.condition: c.prompt_len for c in self.conditions},
                "rng_state_sha256": {c.condition: c.rng_state_sha256[:16]
                                     for c in self.conditions},
                **({"skip_reason": self.skip_reason}
                   if self.skip_reason is not None else {})}


class SiteVector(BaseModel):
    """The built direction at one site, with the aggregation's own numbers beside it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    site: int
    vector: Any                       # np.ndarray float32 [d], unit
    raw_norm: float
    unit_norm: float
    n_pairs: int
    #: how many per-pair differences point the same way as the mean (mirrors the
    #: entropy-gradient builder's diagnostic so the two classes are comparable)
    per_pair_sign_consistency: str
    per_pair_pairwise_coherence: float

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {"per_pair_sign_consistency": self.per_pair_sign_consistency,
                "per_pair_pairwise_coherence": self.per_pair_pairwise_coherence,
                "raw_norm": self.raw_norm, "unit_norm": self.unit_norm,
                "n_pairs": self.n_pairs}


class BuildResult(BaseModel):
    """The full in-memory result; everything banked is derived from this."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request: BuildRequest
    manifest: TopicManifest
    sampling: SamplingResolution
    site_vectors: tuple[SiteVector, ...]
    pairs: tuple[PairRecord, ...]
    n_pairs_effective: int
    n_pairs_skipped: int
    hidden_dim: int
    n_decoder_layers: int
    date_string: str
    wall_seconds: float
    sharding: Optional[dict] = None

    @property
    def matched_rng_holds(self) -> bool:
        """The matched-RNG contract over every ACCEPTED pair — a banked fact."""
        return all(p.matched_rng for p in self.pairs if p.accepted)

    def vector_at(self, site: int) -> SiteVector:
        for sv in self.site_vectors:
            if sv.site == site:
                return sv
        raise KeyError(f"no vector at site L{site}")


# ---------------------------------------------------------------- seeding
def pair_seed(topic_index: int, template_index: int) -> int:
    """`100000 + 10*topic_index + template_index`, verbatim from the recipe."""
    return SEED_BASE + SEED_TOPIC_STRIDE * topic_index + template_index


def seed_pair_rng(seed: int) -> str:
    """Seed torch (CPU + every CUDA device) and digest the resulting stream state.

    Both calls are the recipe's, in the recipe's order. The digest is what turns the
    matched-RNG DESIGN into a matched-RNG MEASUREMENT: it is taken after seeding and
    before generating, so two conditions of one pair having the same digest proves
    they sampled from the same stream.
    """
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- prompt + extract
def chat_ids(tok: Any, user: str, system: Optional[str], date_string: str) -> Any:
    """The recipe's chat-template call, verbatim (L56-62).

    `date_string` is passed UNCONDITIONALLY, as the recipe does. That is deliberate
    and it is verified, not hoped: on transformers 5.3.0 the kwarg is INERT on a
    template that does not reference it (the rendered text is identical), so there is
    no TypeError branch to take. `collect_mean_states.build_ids` carries such a branch
    for older template handling; taking it here would silently produce different token
    ids from the banked build of this class and invalidate the fresh-vs-banked anchor,
    so a TypeError is surfaced as `ChatTemplateError` instead of worked around.
    """
    import torch

    msgs: list[dict[str, str]] = [{"role": "user", "content": user}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    try:
        res = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", date_string=date_string)
    except TypeError as exc:
        raise ChatTemplateError(
            f"this tokenizer's chat template rejected date_string={date_string!r} "
            f"({exc}). The recipe passes it unconditionally; dropping it would change "
            "the token ids and make this vector incomparable with the banked builds "
            "of its class. HALT — needs a desk ruling, not a fall-back.") from exc
    ids = res if isinstance(res, torch.Tensor) else res["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ChatTemplateError(
            f"expected batch-1 ids of shape [1, n], got {tuple(ids.shape)}")
    return ids


def mean_resid_at_sites(model: Any, ids: Any, prompt_len: int,
                        sites: Sequence[int], device: str,
                        n_layers: Optional[int] = None) -> dict[int, np.ndarray]:
    """Mean residual ENTERING each `site`, over the GENERATED positions. Recipe L65-75.

    `hidden_states[s]` is that residual for every `s < n_layers` — the same tensor the
    collector's `forward_pre_hook` on `decoder_layers(model)[s]` captures, proved
    bitwise in `--selftest`. `hidden_states` has `n_layers + 1` entries and the LAST
    one is the final normed state, NOT "entering layer n_layers", which is why the
    range check refuses it rather than banking a differently-defined object.

    `device` is where the ids tensor is built. On the sharded path that is the
    loader's reported input-embedding device — NEVER `next(model.parameters()).device`
    (meta for offloaded weights) and never `.to()` on a dispatched model.
    """
    import torch

    with torch.no_grad():
        out = model(ids.to(device), use_cache=False,
                    output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    limit = len(hs) - 1 if n_layers is None else n_layers
    res: dict[int, np.ndarray] = {}
    for s in sites:
        if not 0 <= s < limit:
            raise HiddenStateConventionError(
                f"site L{s}: hidden_states carries {len(hs)} entries for {limit} "
                f"decoder layers, so only 0 <= s < {limit} is the "
                '"residual entering layer s" convention (entry '
                f"{limit} is the final normed state)")
        h = hs[s][0, prompt_len:]
        if h.shape[0] == 0:
            raise FormalityContrastBuildError(
                f"site L{s}: no generated positions to mean over "
                f"(prompt_len={prompt_len}, seq_len={hs[s].shape[1]})")
        res[s] = h.float().mean(dim=0).cpu().numpy()
    return res


def resolve_hidden_dim(model: Any, preset_hidden_dim: Optional[int] = None) -> int:
    """`config.hidden_size`, then `config.text_config.hidden_size`, then the preset.

    The wrapper-config rake: Gemma3Config nests the text dims under `text_config` and
    declares no top-level `hidden_size`, so a build that trusts the top level gets
    `None` and silently mis-shapes. The preset carries the canonical number as the
    last resort (the anamnesis original's `or preset.hidden_dim`).
    """
    cfg = getattr(model, "config", None)
    dim = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if dim is None and cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        dim = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
    if dim is None:
        dim = preset_hidden_dim
    if dim is None:
        raise FormalityContrastBuildError(
            f"cannot resolve hidden_size on {type(cfg).__name__} and no preset "
            "hidden_dim is available — refusing to build a vector of unknown width")
    return int(dim)


# ---------------------------------------------------------------- the build
def build_formality_contrast(model: Any, tok: Any, manifest: TopicManifest,
                             request: BuildRequest, sampling: SamplingResolution,
                             device: str, date_string: str,
                             sharding: Optional[dict] = None) -> BuildResult:
    """The lifted recipe. One (topic, template) cell per iteration, two conditions each.

    The loop is the anamnesis `build_v1` loop in behavior, statement for statement:
    tokenize -> seed -> generate -> length guard -> extract, formal then informal,
    difference per site, mean over pairs, unit-normalize. What is added is RECORDING
    (the seed table with its RNG digests, the per-site diagnostics) and the terminal
    `n_pairs_effective == 40` assertion; what is removed is nothing.

    `sampling` is resolved by the caller BEFORE any weight is read, so a model with no
    preset row (the 405B, which needs an explicit temperature) is refused for free.
    """
    import torch
    from metabasis.extraction.hooks import decoder_layers

    if sampling.model_key != request.model_key:
        raise FormalityContrastBuildError(
            f"sampling resolution is for {sampling.model_key!r} but the request is "
            f"for {request.model_key!r} — refusing to build with another model's "
            "sampling config")
    t0 = time.time()
    layers = decoder_layers(model)
    n_layers = len(layers)
    bad = [s for s in request.sites if not 0 <= s < n_layers]
    if bad:
        raise SiteOutOfRangeError(
            f"sites {bad} are not decoder-layer indices of {request.model_key}: it "
            f"has {n_layers} layers, so valid sites are 0..{n_layers - 1}")
    hidden_dim = resolve_hidden_dim(model, sampling.preset_hidden_dim)

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = sampling.eos_token_ids[0]          # the recipe's fall-back, verbatim
    sites = tuple(request.sites)
    diffs: dict[int, list[np.ndarray]] = {s: [] for s in sites}
    pairs: list[PairRecord] = []
    n_pairs = 0

    for ti, topic in enumerate(manifest.topics):
        for tmpl_i, tmpl in enumerate(TEMPLATES):
            user = tmpl.format(topic=topic)
            seed = pair_seed(ti, tmpl_i)
            pair_means: dict[str, dict[int, np.ndarray]] = {}
            conditions: list[ConditionRecord] = []
            skip_reason: Optional[str] = None
            for cond, sys_p in CONDITIONS:
                ids = chat_ids(tok, user, sys_p, date_string).to(device)
                digest = seed_pair_rng(seed)
                with torch.no_grad():
                    seq = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=request.max_new_tokens, do_sample=DO_SAMPLE,
                        temperature=float(sampling.temperature), top_p=request.top_p,
                        eos_token_id=list(sampling.eos_token_ids), pad_token_id=pad_id)
                n_new = int(seq.shape[1] - ids.shape[1])
                conditions.append(ConditionRecord(
                    condition=cond, prompt_len=int(ids.shape[1]), n_new_tokens=n_new,
                    seed=seed, rng_state_sha256=digest))
                if n_new < MIN_NEW_TOKENS:
                    skip_reason = (f"{cond} generation returned {n_new} new tokens "
                                   f"(< {MIN_NEW_TOKENS})")
                    logger.warning("pair t%d tmpl%d (%s): short gen (%d new tokens), "
                                   "skipping pair", ti, tmpl_i, cond, n_new)
                    break
                pair_means[cond] = mean_resid_at_sites(
                    model, seq, int(ids.shape[1]), sites, device, n_layers)
            digests = {c.condition: c.rng_state_sha256 for c in conditions}
            matched = (len(digests) == len(CONDITIONS)
                       and len(set(digests.values())) == 1
                       and all(c.seed == seed for c in conditions))
            if skip_reason is not None:
                pairs.append(PairRecord(
                    topic_index=ti, topic=topic, template_index=tmpl_i, template=tmpl,
                    user_prompt=user, seed=seed, conditions=tuple(conditions),
                    matched_rng=matched, accepted=False, skip_reason=skip_reason))
                continue
            for s in sites:
                diffs[s].append(pair_means["formal"][s] - pair_means["informal"][s])
            n_pairs += 1
            pairs.append(PairRecord(
                topic_index=ti, topic=topic, template_index=tmpl_i, template=tmpl,
                user_prompt=user, seed=seed, conditions=tuple(conditions),
                matched_rng=matched, accepted=True))
            if n_pairs % 10 == 0:
                logger.info("%d/%d pairs (%.1fs/pair)", n_pairs, N_PAIRS_REQUIRED,
                            (time.time() - t0) / n_pairs)

    n_skipped = sum(1 for p in pairs if not p.accepted)
    if n_pairs != N_PAIRS_REQUIRED:
        raise PairCountError(
            f"{request.model_key}: {n_pairs} usable pairs, expected exactly "
            f"{N_PAIRS_REQUIRED} ({n_skipped} skipped: "
            f"{[p.skip_reason for p in pairs if not p.accepted][:4]}). A vector built "
            "from a different number of pairs is a different object and is not "
            "comparable with the banked builds of this class — NOTHING has been "
            "written. HALT and surface it; do not re-run with a lower bar.")

    site_vectors = tuple(aggregate_site(s, diffs[s]) for s in sites)
    for sv in site_vectors:
        logger.info("L%d: |v|=%.8f raw_norm=%.4f sign=%s coherence=%.4f",
                    sv.site, sv.unit_norm, sv.raw_norm,
                    sv.per_pair_sign_consistency, sv.per_pair_pairwise_coherence)
    result = BuildResult(
        request=request, manifest=manifest, sampling=sampling,
        site_vectors=site_vectors, pairs=tuple(pairs), n_pairs_effective=n_pairs,
        n_pairs_skipped=n_skipped, hidden_dim=hidden_dim, n_decoder_layers=n_layers,
        date_string=date_string, wall_seconds=round(time.time() - t0, 1),
        sharding=sharding)
    for sv in site_vectors:
        if sv.vector.shape != (hidden_dim,):
            raise FormalityContrastBuildError(
                f"L{sv.site}: built vector has shape {sv.vector.shape}, expected "
                f"({hidden_dim},) — the residual width and the config disagree")
    if not result.matched_rng_holds:
        offenders = [(p.topic_index, p.template_index) for p in result.pairs
                     if p.accepted and not p.matched_rng]
        raise FormalityContrastBuildError(
            "the matched-RNG contract does NOT hold on every accepted pair: the "
            "formal and informal generations of some cell sampled from different "
            "streams, so their difference is not a controlled contrast. Offending "
            f"(topic, template) cells: {offenders[:4]} of {len(offenders)}")
    return result


def aggregate_site(site: int, diffs: Sequence[np.ndarray]) -> SiteVector:
    """The CAA aggregation: mean of per-pair differences, unit-normalized.

    Verbatim arithmetic (`np.mean` over the float32 per-pair differences, then divide
    by the norm, then float32) — the accumulation dtype is part of the banked object,
    so it is not "improved" here. `raw_norm` is the pre-normalization norm, exactly
    the number the precedent's stamp carried. The diagnostics are computed in float64
    and touch nothing that is banked.
    """
    if not diffs:
        raise FormalityContrastBuildError(f"L{site}: no pair differences to aggregate")
    v = np.mean(diffs, axis=0)
    raw_norm = float(np.linalg.norm(v))
    if raw_norm == 0.0:
        raise FormalityContrastBuildError(
            f"L{site}: the mean formal-minus-informal difference is exactly zero — "
            "no direction to build")
    vector = (v / np.linalg.norm(v)).astype(np.float32)

    D = np.stack([d.astype(np.float64) for d in diffs])
    unit = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    iu = np.triu_indices(len(D), k=1)
    coherence = float((unit @ unit.T)[iu].mean()) if len(D) > 1 else float("nan")
    ref = vector.astype(np.float64)
    n_positive = int(((D @ ref) > 0).sum())
    return SiteVector(
        site=site, vector=vector, raw_norm=raw_norm,
        unit_norm=float(np.linalg.norm(vector.astype(np.float64))),
        n_pairs=len(diffs),
        per_pair_sign_consistency=f"{n_positive}/{len(D)}",
        per_pair_pairwise_coherence=round(coherence, 6))


# ---------------------------------------------------------------- banking
def _recipe_provenance() -> dict[str, Any]:
    """The lift's provenance block: source path + line range, the authorizing brief,
    and the shas of the two system prompts so the contrast is verifiable from the
    stamp alone."""
    return {"source": RECIPE_SOURCE,
            "authorized_by": RECIPE_BRIEF,
            "lifted_on": "2026-07-29",
            "templates": list(TEMPLATES),
            "system_prompt_sha256": {"formal": FORMAL_SYS_SHA256,
                                     "informal": INFORMAL_SYS_SHA256}}


def sampling_stamp(sampling: SamplingResolution, request: BuildRequest) -> dict[str, Any]:
    """The resolution merged with the values THIS RUN actually used.

    `max_new_tokens` / `top_p` are recorded as used, with the recipe's values beside
    them and an explicit `recipe_faithful` verdict — a vector built at other values is
    not comparable with the banked builds of this class, and the stamp has to say so
    rather than leave a reader to notice.
    """
    faithful = (request.max_new_tokens == MAX_NEW_TOKENS and request.top_p == TOP_P)
    return {**sampling.stamp,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "recipe_max_new_tokens": MAX_NEW_TOKENS,
            "recipe_top_p": TOP_P,
            "recipe_faithful": faithful}


def site_stamp(result: BuildResult, site: int, trunk: dict) -> dict[str, Any]:
    """The stamp for one site. Every field `REQUIRED_STAMP_FIELDS` names, or it raises."""
    req = result.request
    sv = result.vector_at(site)
    stamp: dict[str, Any] = {
        "STATUS": ("UNSTAMPED (prereg C section 8) — built by "
                   "build_formality_contrast.py; scores nothing on its own"),
        "builder": "build_formality_contrast.py",
        "model": req.model_key,
        "site": site,
        "sites": list(req.sites),
        "npz_keys": {"vector": CANONICAL_VECTOR_KEY.format(site=site)},
        "anamnesis_bank_key_equivalent": ANAMNESIS_BANK_KEY.format(site=site),
        "sign_convention": SIGN_CONVENTION,
        "recipe_provenance": _recipe_provenance(),
        "vector_provenance": (
            f"unit( mean over {result.n_pairs_effective} matched pairs of "
            f"( mean over GENERATED positions of h_formal[L{site}] - the same for "
            f"h_informal[L{site}] ) ); {result.manifest.n_topics} topics x "
            f"{len(TEMPLATES)} user templates x 2 opposed system prompts, sampled "
            f"fresh at temperature {result.sampling.temperature} / top_p {req.top_p} / "
            f"max_new_tokens {req.max_new_tokens}, with the RNG re-seeded per "
            "(topic, template) BEFORE EACH CONDITION so the two conditions of a pair "
            "sample from an identical stream."),
        "capture_convention": (
            "output_hidden_states=True, hidden_states[L] = the residual stream "
            "ENTERING decoder layer L over the GENERATED positions (index > prompt "
            "length), batch-1, no cache, fp32 mean. VERIFIED BITWISE IDENTICAL to "
            "collect_mean_states.py's forward_pre_hook on decoder_layers(model)[L] "
            "(the builder's --selftest asserts it), so this vector lives in the same "
            "raw residual space the transport maps were fit in."),
        "topic_manifest": result.manifest.stamp,
        "n_pairs_expected": N_PAIRS_REQUIRED,
        "n_pairs_effective": result.n_pairs_effective,
        "n_pairs_skipped": result.n_pairs_skipped,
        "skipped_pairs": [p.stamp for p in result.pairs if not p.accepted],
        "sampling": sampling_stamp(result.sampling, req),
        "seed_table": [p.stamp for p in result.pairs],
        "matched_rng_holds": result.matched_rng_holds,
        "raw_norm": sv.raw_norm,
        "unit_norm": sv.unit_norm,
        "vector_dtype": str(sv.vector.dtype),
        "hidden_dim": result.hidden_dim,
        "n_decoder_layers": result.n_decoder_layers,
        "diagnostics": sv.diagnostics,
        "date_string": result.date_string,
        "wall_seconds": result.wall_seconds,
        "generation_reproducibility": GENERATION_REPRODUCIBILITY,
        "sharded_generation_certified": False,
        "named_limits": {"topic_leakage": TOPIC_LEAKAGE_LIMIT,
                         "no_build_gate": NO_BUILD_GATE_LIMIT},
        "trunk": trunk,
    }
    # The shard spec, only on the sharded path — same discipline as the collector's
    # `sharding` key, and the caveat above is what makes its presence meaningful.
    if result.sharding is not None:
        stamp["sharding"] = result.sharding
    missing = [k for k in REQUIRED_STAMP_FIELDS if k not in stamp]
    if missing:                                                # pragma: no cover
        raise StampIncompleteError(
            f"stamp for {req.model_key} L{site} is missing {missing} — refusing to "
            "bank a vector whose provenance record is incomplete")
    return stamp


def planned_paths(request: BuildRequest,
                  sites: Sequence[int]) -> dict[int, dict[str, Path]]:
    """Every file a build of `sites` will write. Pure path arithmetic — no I/O."""
    return {s: {"vector": request.out_dir / f"{request.stem(s)}.npz",
                "stamp": request.out_dir / f"{request.stem(s)}_stamps.json"}
            for s in sites}


def census_targets(request: BuildRequest, sites: Sequence[int]) -> list[Path]:
    """The bank census: which target files already exist. Raises on a collision.

    Called TWICE by design. Once in `main` BEFORE the weights are loaded — the ops rule
    is a census before any write into an existing vectors dir, and discovering a
    collision after a three-hour 405B build would be discovering it far too late — and
    once inside `bank`, which is the guard that actually holds when this function is
    not the caller's entry point.
    """
    existing = [p for paths in planned_paths(request, sites).values()
                for p in paths.values() if p.exists()]
    if existing and not request.allow_overwrite:
        raise BankCollisionError(
            f"{len(existing)} target file(s) already exist in {request.out_dir} and "
            f"--allow-overwrite is not set: {[p.name for p in existing][:4]}. No "
            "vector of this class has ever been banked in a v2.1 vectors dir, so this "
            "is a census failure, not a re-run — HALT and find out what is there.")
    return existing


def bank(result: BuildResult, trunk: dict,
         saboteur: Optional[Callable[[int], None]] = None
         ) -> dict[int, dict[str, Path]]:
    """Write every site's npz + stamp ATOMICALLY, or leave the directory untouched.

    Two properties, both asserted by the selftest:

      * NOTHING is written until the whole build has succeeded (this function is
        called once, after `build_formality_contrast` returns), so a preempted 405B
        job banks no partial vector;
      * each file is written to a temp name and `os.replace`d — atomic within a
        filesystem — and if ANY write fails, every file this call already placed is
        removed before the exception propagates. A half-banked site set is as bad as
        a partial file.

    `saboteur` is selftest-only: called with each site index just before that site's
    files are placed, so the rollback path can be exercised without a real failure.
    """
    req = result.request
    out = req.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # BANK CENSUS (ops rule) over the WHOLE slate before placing anything, so a
    # collision on the last site cannot follow several successful writes. `main` has
    # already run this before loading the weights; this is the guard that holds for any
    # other caller.
    sites = [sv.site for sv in result.site_vectors]
    census_targets(req, sites)
    planned = planned_paths(req, sites)

    placed: list[Path] = []
    tmp_suffix = f".tmp-{os.getpid()}"
    try:
        for sv in result.site_vectors:
            if saboteur is not None:
                saboteur(sv.site)
            stamp = site_stamp(result, sv.site, trunk)
            vec_path, stamp_path = planned[sv.site]["vector"], planned[sv.site]["stamp"]
            vec_tmp = vec_path.with_name(vec_path.name + tmp_suffix)
            stamp_tmp = stamp_path.with_name(stamp_path.name + tmp_suffix)
            try:
                # np.savez APPENDS `.npz` to a filename that does not already end in
                # it — `foo.npz.tmp-123` would become `foo.npz.tmp-123.npz` and the
                # os.replace below would then fail on a missing file. Handing it an
                # open FILE OBJECT writes exactly where we said.
                with open(vec_tmp, "wb") as fh:
                    np.savez(fh, **{CANONICAL_VECTOR_KEY.format(site=sv.site):
                                    sv.vector})
                stamp_tmp.write_text(json.dumps(stamp, indent=1, ensure_ascii=False))
                os.replace(vec_tmp, vec_path)
                placed.append(vec_path)
                os.replace(stamp_tmp, stamp_path)
                placed.append(stamp_path)
            finally:
                for tmp in (vec_tmp, stamp_tmp):
                    if tmp.exists():
                        tmp.unlink()
    except BaseException:
        for p in placed:
            if p.exists():
                p.unlink()
        logger.error("bank failed; rolled back %d file(s) — the vectors dir is as it "
                     "was", len(placed))
        raise
    for site, paths in planned.items():
        logger.info("banked L%d -> %s + %s", site, paths["vector"].name,
                    paths["stamp"].name)
    return planned


# ---------------------------------------------------------------- CPU self-test
class _ToyChatTokenizer:
    """The minimum tokenizer surface this builder touches, with a system role.

    Deliberately NOT a real tokenizer: the point of the CPU selftest is the recipe's
    control flow and arithmetic, and a hand-rolled tokenizer makes the token ids (and
    therefore the generation lengths) a fact of the test rather than of a download.
    """

    def __init__(self, vocab_size: int) -> None:
        import torch
        self._torch = torch
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.pad_token_id = None
        self.chat_template = "toy-with-system-role"
        self.date_strings_seen: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2 + (ord(c) % (self.vocab_size - 4)) for c in text]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    #: per-message character budget. It MUST be long enough that the two system
    #: prompts render to DIFFERENT ids: at 6 chars both are "You ar", the two
    #: conditions become the same prompt, the matched RNG makes the two generations
    #: identical, and every per-pair difference is exactly zero — which the builder
    #: correctly refuses to normalize. 48 chars separates them ("You are an extremely
    #: formal assistant. Respond w" vs "You are a super casual assistant. Keep it
    #: loose "), and the selftest asserts the separation rather than trusting it.
    MSG_CHARS = 48

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True,
                            return_tensors: Optional[str] = None,
                            **kwargs: Any) -> Any:
        if "date_string" in kwargs:
            self.date_strings_seen.append(str(kwargs["date_string"]))
        out = [self.bos_token_id]
        for m in messages:
            out += self.encode(m["content"], add_special_tokens=False)[:self.MSG_CHARS]
        if add_generation_prompt:
            out.append(3)
        if return_tensors == "pt":
            return self._torch.tensor([out], dtype=self._torch.long)
        return out


class _DateFreeTokenizer(_ToyChatTokenizer):
    """A tokenizer whose template REJECTS `date_string` (the HALT path)."""

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True,
                            return_tensors: Optional[str] = None,
                            **kwargs: Any) -> Any:
        if "date_string" in kwargs:
            raise TypeError("toy date-free template takes no date_string")
        return super().apply_chat_template(messages, add_generation_prompt,
                                           return_tensors)


def _toy_diffs(n: int, d: int, seed: int = 20260729) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(d).astype(np.float32) for _ in range(n)]


def _synthetic_result(out_dir: Path, sites: Sequence[int] = (2, 3),
                      hidden_dim: int = 8) -> BuildResult:
    """A BuildResult with no model behind it — for the banking/stamp/atomicity blocks."""
    manifest = load_topic_manifest()
    req = BuildRequest(model_key="toy", model_path="(none)", out_dir=out_dir,
                       sites=tuple(sites), device="cpu")
    sampling = SamplingResolution(
        model_key="toy", registries=("selftest",), temperature=0.7,
        temperature_source="selftest", eos_token_ids=(7,), eos_source="selftest",
        torch_dtype="bfloat16", dtype_source="selftest", preset_hidden_dim=hidden_dim,
        checkpoint_identity="instruct")
    pairs = tuple(
        PairRecord(topic_index=ti, topic=t, template_index=tm, template=TEMPLATES[tm],
                   user_prompt=TEMPLATES[tm].format(topic=t), seed=pair_seed(ti, tm),
                   conditions=tuple(
                       ConditionRecord(condition=c, prompt_len=11, n_new_tokens=16,
                                       seed=pair_seed(ti, tm), rng_state_sha256="0" * 64)
                       for c in ("formal", "informal")),
                   matched_rng=True, accepted=True)
        for ti, t in enumerate(manifest.topics) for tm in range(len(TEMPLATES)))
    site_vectors = tuple(
        aggregate_site(s, _toy_diffs(N_PAIRS_REQUIRED, hidden_dim, seed=1000 + s))
        for s in sites)
    return BuildResult(
        request=req, manifest=manifest, sampling=sampling, site_vectors=site_vectors,
        pairs=pairs, n_pairs_effective=N_PAIRS_REQUIRED, n_pairs_skipped=0,
        hidden_dim=hidden_dim, n_decoder_layers=max(sites) + 2,
        date_string="12 Jul 2026", wall_seconds=1.0)


def _torch_available() -> tuple[bool, str]:
    try:
        import torch                                            # noqa: F401
        import transformers                                     # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def selftest() -> int:
    """CPU verification. The model-free blocks run anywhere; the rest is torch-gated."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []
    skipped: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    # ---------------- 1. the topic manifest (tracked CODE, so data-independent) ----
    manifest = load_topic_manifest()
    check("topic manifest resolves package-relative (cwd-independent)",
          manifest.path.is_file(), str(manifest.path.name))
    check(f"manifest carries exactly {N_TOPICS_REQUIRED} topics",
          manifest.n_topics == N_TOPICS_REQUIRED, str(manifest.n_topics))
    check("manifest topics are distinct", len(set(manifest.topics)) == N_TOPICS_REQUIRED)
    check("manifest order is set_a then set_b (the seed index depends on it)",
          manifest.topic_sets_in_order == ("set_a", "set_b"),
          str(manifest.topic_sets_in_order))
    check("manifest sha256 is recorded and stable",
          manifest.sha256 == hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
          manifest.sha256[:16])
    check("manifest records its anamnesis source sha",
          len(manifest.source_sha256) == 64 and "prompt_sets" in manifest.source,
          manifest.source_sha256[:16])
    check(f"{N_PAIRS_REQUIRED} pairs = {N_TOPICS_REQUIRED} topics x "
          f"{len(TEMPLATES)} templates",
          N_PAIRS_REQUIRED == manifest.n_topics * len(TEMPLATES))

    # the n_topics assertion FIRES (the 8-pairs-instead-of-40 rake)
    with tempfile.TemporaryDirectory() as td:
        for label, doc in (
                ("19 topics", {"topic_sets_in_order": ["set_a"],
                               "topics": list(manifest.topics[:19]),
                               "provenance": {"source": "prompt_sets.json",
                                              "source_sha256": "0" * 64}}),
                ("4 topic-SET names", {"topic_sets_in_order": ["set_a"],
                                       "topics": ["set_a", "set_b", "set_c", "set_d"],
                                       "provenance": {"source": "prompt_sets.json",
                                                      "source_sha256": "0" * 64}}),
                ("a duplicated topic",
                 {"topic_sets_in_order": ["set_a"],
                  "topics": list(manifest.topics[:19]) + [manifest.topics[0]],
                  "provenance": {"source": "prompt_sets.json",
                                 "source_sha256": "0" * 64}})):
            p = Path(td) / f"bad-{abs(hash(label))}.json"
            p.write_text(json.dumps(doc))
            try:
                load_topic_manifest(p)
                check(f"n_topics/duplicate assertion fires on {label}", False)
            except TopicManifestError as exc:
                check(f"n_topics/duplicate assertion fires on {label}", True,
                      str(exc).split(":")[-1].strip()[:60])
        # order coupling: `topics` that is not the concatenation of the named sets
        p = Path(td) / "reordered.json"
        p.write_text(json.dumps({
            "topic_sets_in_order": ["set_a", "set_b"],
            "topic_sets": {"set_a": list(manifest.topics[:10]),
                           "set_b": list(manifest.topics[10:])},
            "topics": list(reversed(manifest.topics)),
            "provenance": {"source": "prompt_sets.json", "source_sha256": "0" * 64}}))
        try:
            load_topic_manifest(p)
            check("manifest refuses a `topics` list that reorders its sets", False)
        except TopicManifestError:
            check("manifest refuses a `topics` list that reorders its sets", True)

    # ---------------- 2. the seed table arithmetic --------------------------------
    seeds = {(ti, tm): pair_seed(ti, tm)
             for ti in range(N_TOPICS_REQUIRED) for tm in range(len(TEMPLATES))}
    check("seed formula is 100000 + 10*topic + template, verbatim",
          all(s == 100000 + 10 * ti + tm for (ti, tm), s in seeds.items()))
    check(f"all {N_PAIRS_REQUIRED} seeds are distinct",
          len(set(seeds.values())) == N_PAIRS_REQUIRED, str(len(set(seeds.values()))))
    check("the topic stride cannot collide with a template index",
          SEED_TOPIC_STRIDE > len(TEMPLATES),
          f"stride {SEED_TOPIC_STRIDE} > {len(TEMPLATES)} templates")
    check("seed range is contiguous-per-topic",
          min(seeds.values()) == 100000 and max(seeds.values()) == 100191,
          f"{min(seeds.values())}..{max(seeds.values())}")

    # ---------------- 3. the CAA aggregation arithmetic on a fixture --------------
    # A hand-computable case: two pairs whose mean is (2, 0, 0) before normalization.
    fx = [np.array([3.0, 1.0, 0.0], dtype=np.float32),
          np.array([1.0, -1.0, 0.0], dtype=np.float32)]
    sv = aggregate_site(5, fx)
    check("difference-of-means is the plain mean over pairs",
          np.allclose(sv.vector, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
          str(sv.vector))
    check("raw_norm is the PRE-normalization norm", abs(sv.raw_norm - 2.0) < 1e-6,
          f"{sv.raw_norm:.6f}")
    check("banked vector is unit", abs(sv.unit_norm - 1.0) < 1e-6, f"{sv.unit_norm:.8f}")
    check("banked vector is float32", sv.vector.dtype == np.float32, str(sv.vector.dtype))
    check("n_pairs is recorded", sv.n_pairs == 2, str(sv.n_pairs))
    check("sign consistency counts pairs agreeing with the mean",
          sv.per_pair_sign_consistency == "2/2", sv.per_pair_sign_consistency)
    cos = float(np.dot(fx[0] / np.linalg.norm(fx[0]), fx[1] / np.linalg.norm(fx[1])))
    check("pairwise coherence is the mean pairwise cosine of the per-pair diffs",
          abs(sv.per_pair_pairwise_coherence - round(cos, 6)) < 1e-6,
          f"{sv.per_pair_pairwise_coherence} vs {round(cos, 6)}")
    sv40 = aggregate_site(5, _toy_diffs(N_PAIRS_REQUIRED, 32))
    check("40-pair aggregation is unit and records 40 pairs",
          abs(sv40.unit_norm - 1.0) < 1e-6 and sv40.n_pairs == N_PAIRS_REQUIRED)
    try:
        aggregate_site(5, [np.array([1.0, 0.0], np.float32),
                           np.array([-1.0, 0.0], np.float32)])
        check("a zero mean difference is refused, not divided by", False)
    except FormalityContrastBuildError:
        check("a zero mean difference is refused, not divided by", True)
    try:
        aggregate_site(5, [])
        check("an empty pair set is refused", False)
    except FormalityContrastBuildError:
        check("an empty pair set is refused", True)

    # ---------------- 4. sampling resolution (registries, refusals) ---------------
    roster_keys, preset_keys = known_model_keys()
    res_8b = resolve_sampling("8b")
    check("8b resolves temperature from MODEL_PRESETS",
          res_8b.temperature == 0.6 and "MODEL_PRESETS" in res_8b.temperature_source,
          f"T={res_8b.temperature}")
    check("8b resolves eos from MODEL_PRESETS",
          res_8b.eos_token_ids == (128001, 128008, 128009), str(res_8b.eos_token_ids))
    check("qwen-7b resolves from MODEL_PRESETS",
          resolve_sampling("qwen-7b").temperature == 0.7)
    check("gemma3-27b is in BOTH metabasis registries",
          resolve_sampling("gemma3-27b").registries == ("roster", "MODEL_PRESETS"),
          str(resolve_sampling("gemma3-27b").registries))
    check("gemma3-27b carries a preset hidden_dim for the wrapper-config fall-back",
          resolve_sampling("gemma3-27b").preset_hidden_dim == 5376)
    try:
        resolve_sampling("llama-3.1-405b-instruct")
        check("405B refuses to guess a temperature (no MODEL_PRESETS row)", False)
    except SamplingUnresolvedError as exc:
        check("405B refuses to guess a temperature (no MODEL_PRESETS row)", True,
              "MODEL_PRESETS" in str(exc))
    r405 = resolve_sampling("llama-3.1-405b-instruct", temperature=0.6,
                            eos_token_ids=(128001, 128008, 128009))
    check("405B resolves with explicit CLI values, source recorded",
          r405.temperature == 0.6 and r405.temperature_source == "explicit --temperature"
          and r405.registries == ("roster",), r405.temperature_source)
    check("405B dtype falls back to the loaders' bf16 pin",
          r405.torch_dtype == "bfloat16" and "no preset row" in r405.dtype_source)
    try:
        resolve_sampling("not-a-model")
        check("an unknown model key is refused with both registries listed", False)
    except UnknownModelError as exc:
        check("an unknown model key is refused with both registries listed",
              "roster" in str(exc) and "MODEL_PRESETS" in str(exc))
    try:
        resolve_sampling("3b")
        check("a float16-preset model is refused, not silently cast to bf16", False)
    except DtypeRegimeError as exc:
        check("a float16-preset model is refused, not silently cast to bf16",
              "float16" in str(exc))
    from metabasis.roster import ROSTER
    base_keys = [k for k in roster_keys
                 if ROSTER[k].checkpoint_identity == "base"]
    if base_keys:
        try:
            resolve_sampling(base_keys[0])
            check("a BASE checkpoint is refused (the class needs a system role)", False)
        except ChatTemplateError:
            check("a BASE checkpoint is refused (the class needs a system role)", True,
                  base_keys[0])
    check("the build slate's control models all resolve",
          all(resolve_sampling(k).temperature > 0
              for k in ("8b", "qwen-7b", "gemma3-27b")))
    check("preset registry is metabasis', not anamnesis'",
          "8b" in preset_keys and "gemma3-27b" in roster_keys)

    # ---------------- 5. stamp completeness + banking round-trip ------------------
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "vectors"
        result = _synthetic_result(out)
        trunk = {"device": "cpu", "cuda_visible_devices": "(unset)", "hostname": "toy"}
        paths = bank(result, trunk)
        check("one npz + one stamp per requested site", len(paths) == 2
              and all(set(v) == {"vector", "stamp"} for v in paths.values()))
        for site in (2, 3):
            check(f"L{site} filenames follow formality_contrast_<model>_L<site>",
                  paths[site]["vector"].name == f"formality_contrast_toy_L{site}.npz"
                  and paths[site]["stamp"].name
                  == f"formality_contrast_toy_L{site}_stamps.json",
                  paths[site]["vector"].name)
            with np.load(paths[site]["vector"]) as z:
                check(f"L{site} npz carries the canonical key",
                      list(z.files) == [f"formality_contrast_L{site}"], str(z.files))
                check(f"L{site} banked vector round-trips bit-exactly",
                      np.array_equal(z[f"formality_contrast_L{site}"],
                                     result.vector_at(site).vector))
            stamp = json.loads(paths[site]["stamp"].read_text())
            missing = [k for k in REQUIRED_STAMP_FIELDS if k not in stamp]
            check(f"L{site} stamp carries every required field", not missing,
                  f"{len(REQUIRED_STAMP_FIELDS)} fields, missing {missing}")
            check(f"L{site} stamp records the topic-manifest sha",
                  stamp["topic_manifest"]["sha256"] == manifest.sha256)
            check(f"L{site} stamp records the recipe provenance (path + lines + brief)",
                  "L90-128" in stamp["recipe_provenance"]["source"]
                  and "568dcb61" in stamp["recipe_provenance"]["authorized_by"])
            check(f"L{site} stamp carries the full {N_PAIRS_REQUIRED}-row seed table",
                  len(stamp["seed_table"]) == N_PAIRS_REQUIRED
                  and stamp["seed_table"][0]["seed"] == 100000
                  and stamp["seed_table"][-1]["seed"] == 100191,
                  f"{len(stamp['seed_table'])} rows")
            check(f"L{site} stamp records n_pairs_effective and the expectation",
                  stamp["n_pairs_effective"] == N_PAIRS_REQUIRED
                  and stamp["n_pairs_expected"] == N_PAIRS_REQUIRED
                  and stamp["n_pairs_skipped"] == 0)
            check(f"L{site} stamp records unit_norm AND raw_norm",
                  abs(stamp["unit_norm"] - 1.0) < 1e-6 and stamp["raw_norm"] > 0.0,
                  f"raw {stamp['raw_norm']:.4f}")
            check(f"L{site} stamp carries the sharded-generation caveat as a field",
                  stamp["sharded_generation_certified"] is False
                  and "NOT certified" in stamp["generation_reproducibility"])
            check(f"L{site} stamp names the topic-leakage limit",
                  "TOPIC LEAKAGE" in stamp["named_limits"]["topic_leakage"])
            check(f"L{site} stamp carries the trunk (cuda facts + host)",
                  "cuda_visible_devices" in stamp["trunk"])
            check(f"L{site} stamp records the sampling config",
                  stamp["sampling"]["top_p"] == TOP_P
                  and stamp["sampling"]["max_new_tokens"] == MAX_NEW_TOKENS
                  and stamp["sampling"]["do_sample"] is True
                  and stamp["sampling"]["recipe_faithful"] is True)
            check(f"L{site} stamp cross-references the anamnesis bank key",
                  stamp["anamnesis_bank_key_equivalent"] == f"V1_L{site}")
            check(f"L{site} stamp records the matched-RNG verdict",
                  stamp["matched_rng_holds"] is True)
            check(f"L{site} stamp records the site LIST (all-sites-from-one-forward)",
                  stamp["sites"] == [2, 3], str(stamp["sites"]))
        check("no sharding key on the single-card path",
              "sharding" not in json.loads(paths[2]["stamp"].read_text()))
        # the sharded path DOES record the spec beside the caveat
        sharded = result.model_copy(update={
            "sharding": {"mode": "sharded", "spec": "--shard-across 8",
                         "n_compute_devices": 8}})
        st = site_stamp(sharded, 2, trunk)
        check("the sharded path records the shard spec in the stamp",
              st["sharding"]["n_compute_devices"] == 8
              and st["sharded_generation_certified"] is False)

        # A DEVIATING run must be recorded as deviating. The stamp used to report the
        # module CONSTANT for max_new_tokens/top_p regardless of what the run passed,
        # which would have a 12-token build claim 160 — caught by the CLI smoke, and
        # this is the regression check.
        dev_req = result.request.model_copy(update={"max_new_tokens": 12, "top_p": 0.5})
        dev = sampling_stamp(result.sampling, dev_req)
        check("the stamp records the max_new_tokens/top_p the RUN used, not the "
              "module default",
              dev["max_new_tokens"] == 12 and dev["top_p"] == 0.5
              and dev["recipe_max_new_tokens"] == MAX_NEW_TOKENS
              and dev["recipe_top_p"] == TOP_P
              and dev["recipe_faithful"] is False,
              json.dumps({k: dev[k] for k in ("max_new_tokens", "top_p",
                                              "recipe_faithful")}))
        check("a recipe-faithful run is stamped recipe_faithful",
              sampling_stamp(result.sampling, result.request)["recipe_faithful"] is True)
        check("checkpoint_identity is never a plausible guess for a preset-only key",
              "UNVERIFIED" in (resolve_sampling("8b").checkpoint_identity or ""),
              str(resolve_sampling("8b").checkpoint_identity)[:40])
        check("a roster key carries its VERIFIED checkpoint identity",
              resolve_sampling("gemma3-27b").checkpoint_identity == "instruct")

        # a second bank into the same dir is a CENSUS FAILURE, not a re-run
        try:
            bank(result, trunk)
            check("an existing target file is a HALT (bank census)", False)
        except BankCollisionError as exc:
            check("an existing target file is a HALT (bank census)",
                  "census" in str(exc))
        check("the refused re-bank changed nothing",
              all(p.exists() for v in paths.values() for p in v.values()))
        # the census is a PREFLIGHT too: it must fire from paths alone, with no build
        # and no weights, which is what keeps a collision from costing a 405B window.
        try:
            census_targets(result.request, [2, 3])
            check("census_targets refuses from PATHS ALONE (the preflight)", False)
        except BankCollisionError:
            check("census_targets refuses from PATHS ALONE (the preflight)", True)
        check("census_targets is clean on an untouched dir",
              census_targets(result.request.model_copy(
                  update={"out_dir": out / "fresh"}), [2, 3]) == [])
        check("--allow-overwrite is the named valve, off by default",
              result.request.allow_overwrite is False
              and census_targets(result.request.model_copy(
                  update={"allow_overwrite": True}), [2, 3]) != [])
        check("planned_paths does no I/O and names both files per site",
              set(planned_paths(result.request, [7])[7]) == {"vector", "stamp"}
              and planned_paths(result.request, [7])[7]["vector"].name
              == "formality_contrast_toy_L7.npz")

    # ---------------- 6. atomicity: a failure leaves NO output files --------------
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "vectors"
        result = _synthetic_result(out, sites=(1, 2, 3))
        trunk = {"device": "cpu", "cuda_visible_devices": "(unset)"}

        def fail_on_third(site: int) -> None:
            if site == 3:
                raise RuntimeError("simulated preemption between sites")

        try:
            bank(result, trunk, saboteur=fail_on_third)
            check("a mid-bank failure propagates", False)
        except RuntimeError as exc:
            check("a mid-bank failure propagates", "simulated" in str(exc))
        leftovers = sorted(p.name for p in out.iterdir()) if out.exists() else []
        check("a mid-bank failure leaves NO output files (full rollback)",
              leftovers == [], str(leftovers))

        def fail_immediately(site: int) -> None:
            raise RuntimeError("simulated preemption before the first write")

        try:
            bank(result, trunk, saboteur=fail_immediately)
        except RuntimeError:
            pass
        check("a failure before the first write leaves the dir empty",
              (sorted(p.name for p in out.iterdir()) if out.exists() else []) == [])
        # and the happy path in the same dir still works afterwards
        paths = bank(result, trunk)
        check("banking succeeds after a rolled-back attempt", len(paths) == 3
              and all(p.exists() for v in paths.values() for p in v.values()))
        check("no temp files survive a successful bank",
              not [p.name for p in out.iterdir() if ".tmp-" in p.name],
              str(sorted(p.name for p in out.iterdir()))[:80])

    # ---------------- 7. constants are the anamnesis constants -------------------
    check("2 user templates, verbatim",
          TEMPLATES == ("Write about {topic}.", "Explain {topic} to a beginner."))
    check("the formal system prompt is byte-identical to the anamnesis constant",
          hashlib.sha256(FORMAL_SYS.encode()).hexdigest() == FORMAL_SYS_SHA256,
          f"len {len(FORMAL_SYS)}, sha {FORMAL_SYS_SHA256[:12]}")
    check("the informal system prompt is byte-identical to the anamnesis constant",
          hashlib.sha256(INFORMAL_SYS.encode()).hexdigest() == INFORMAL_SYS_SHA256,
          f"len {len(INFORMAL_SYS)}, sha {INFORMAL_SYS_SHA256[:12]}")
    check("both system prompt shas are recorded in the stamp's recipe provenance",
          FORMAL_SYS_SHA256 in json.dumps(_recipe_provenance())
          and INFORMAL_SYS_SHA256 in json.dumps(_recipe_provenance()))
    check("formal is the minuend (sign convention)",
          CONDITIONS[0][0] == "formal" and CONDITIONS[1][0] == "informal")
    check("sampling constants are the recipe's",
          (DO_SAMPLE, TOP_P, MAX_NEW_TOKENS, MIN_NEW_TOKENS) == (True, 0.9, 160, 8))
    from metabasis.scripts.collect_mean_states import VMB_CANONICAL_DATE
    check("date_string is the campaign canonical date (== the anamnesis pin)",
          VMB_CANONICAL_DATE == "12 Jul 2026", VMB_CANONICAL_DATE)

    # ---------------- 8. model-dependent blocks (torch-gated) -------------------
    ok, why = _torch_available()
    if not ok:
        skipped.append(f"model-dependent blocks (torch/transformers absent: {why})")
        logger.warning("SKIP model-dependent selftest blocks — %s", why)
    else:
        checks.extend(_selftest_with_model())

    n_miss = sum(1 for _, ok_, _ in checks if not ok_)
    print(json.dumps({"checks": len(checks), "misses": n_miss,
                      "skipped_blocks": skipped,
                      "failed": [n for n, ok_, _ in checks if not ok_]}, indent=1))
    if n_miss:
        print(f"SELFTEST-NOT-CLEAN: {n_miss} of {len(checks)} checks did not hold")
        return 1
    suffix = f" ({len(skipped)} block(s) skipped)" if skipped else ""
    print(f"SELFTEST-CLEAN: {len(checks)}/{len(checks)} checks hold{suffix}")
    return 0


def _selftest_with_model() -> list[tuple[str, bool, str]]:
    """The blocks that need a model: the convention proof and the end-to-end build."""
    import tempfile

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from metabasis.scripts.collect_mean_states import SiteCapture

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    torch.manual_seed(20260729)
    n_layers, hidden = 4, 16
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=32,
                      num_hidden_layers=n_layers, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=512,
                      tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).to(torch.float32).eval()
    model.requires_grad_(False)

    # ---- THE MANDATED PROOF: hidden_states[s] == the collector's pre-hook capture --
    sites_all = tuple(range(n_layers))
    ids = torch.randint(4, cfg.vocab_size, (1, 23))
    cap = SiteCapture(model, sites_all)
    try:
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True,
                        return_dict=True)
        deltas: dict[int, float] = {}
        bitwise: dict[int, bool] = {}
        for s in sites_all:
            hs, hk = out.hidden_states[s], cap.grab[s]
            bitwise[s] = bool(torch.equal(hs, hk))
            deltas[s] = float((hs.double() - hk.double()).abs().max())
    finally:
        cap.close()
    check("hidden_states has n_layers + 1 entries",
          len(out.hidden_states) == n_layers + 1, str(len(out.hidden_states)))
    check("CONVENTION PROOF: hidden_states[s] is BITWISE the pre-hook capture at "
          "every site", all(bitwise.values()),
          f"max|delta| = {max(deltas.values()):.3e} over sites {list(sites_all)}")
    check("CONVENTION PROOF: the means over generated positions agree exactly",
          all(np.array_equal(
              out.hidden_states[s][0, 7:].float().mean(dim=0).numpy(),
              cap.grab[s][0, 7:].float().mean(dim=0).numpy()) for s in sites_all),
          "prompt_len=7")
    # The proof again on the ACTUAL production code path: `mean_resid_at_sites` (which
    # is what the build calls) against the collector's hook driven over the same
    # sequence. Comparing the raw tensors is necessary; comparing the reduced objects
    # the two builders actually bank is the claim that matters.
    prompt_len = 9
    cap = SiteCapture(model, sites_all)
    try:
        with torch.no_grad():
            model(ids, use_cache=False, return_dict=True)
        hook_means = {s: cap.grab[s][0, prompt_len:, :].float().mean(dim=0).cpu().numpy()
                      for s in sites_all}
    finally:
        cap.close()
    hs_means = mean_resid_at_sites(model, ids, prompt_len, sites_all, "cpu", n_layers)
    check("CONVENTION PROOF: mean_resid_at_sites == the collector's reduced object, "
          "bitwise",
          all(np.array_equal(hs_means[s], hook_means[s]) for s in sites_all),
          "max|delta| = "
          f"{max(float(np.abs(hs_means[s] - hook_means[s]).max()) for s in sites_all):.3e}")

    # and in the PRODUCTION dtype: the node forwards in bf16, and a convention that
    # only held in fp32 would prove nothing about a banked vector.
    bf16 = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    bf16.load_state_dict({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    bf16.requires_grad_(False)
    cap = SiteCapture(bf16, sites_all)
    try:
        with torch.no_grad():
            out16 = bf16(ids, use_cache=False, output_hidden_states=True,
                         return_dict=True)
        bf16_bitwise = all(torch.equal(out16.hidden_states[s], cap.grab[s])
                           for s in sites_all)
    finally:
        cap.close()
    check("CONVENTION PROOF: the equivalence also holds BITWISE in bfloat16 (the "
          "production forward dtype)", bf16_bitwise, "sites " + str(list(sites_all)))

    # and the range guard: the LAST hidden_states entry is NOT "entering layer n"
    try:
        mean_resid_at_sites(model, ids, 7, (n_layers,), "cpu", n_layers)
        check("a site == n_layers is refused (final normed state is a different object)",
              False)
    except HiddenStateConventionError:
        check("a site == n_layers is refused (final normed state is a different object)",
              True, f"n_layers={n_layers}")

    # ---- the end-to-end 40-pair build -----------------------------------------
    manifest = load_topic_manifest()
    tok = _ToyChatTokenizer(cfg.vocab_size)
    sampling = SamplingResolution(
        model_key="toy-llama", registries=("selftest",), temperature=0.7,
        # eos deliberately OUT OF VOCAB so every generation runs to max_new_tokens:
        # the recipe's length guard is exercised separately and deterministically.
        temperature_source="selftest", eos_token_ids=(cfg.vocab_size + 7,),
        eos_source="selftest", torch_dtype="bfloat16", dtype_source="selftest",
        preset_hidden_dim=hidden, checkpoint_identity="instruct")
    req = BuildRequest(model_key="toy-llama", model_path="(none)",
                       out_dir=Path("."), sites=(1, 2, 3), max_new_tokens=10,
                       device="cpu")
    result = build_formality_contrast(model, tok, manifest, req, sampling, "cpu",
                                     "12 Jul 2026")
    #: snapshot BEFORE the extra probe calls below, so the "one template call per
    #: condition" count is the BUILD's count and nothing else's.
    template_calls_in_build = list(tok.date_strings_seen)
    check("end-to-end build accepts exactly 40 pairs",
          result.n_pairs_effective == N_PAIRS_REQUIRED and result.n_pairs_skipped == 0,
          f"{result.n_pairs_effective} pairs")
    check("all requested sites come out of ONE extraction pass per condition",
          tuple(sv.site for sv in result.site_vectors) == (1, 2, 3))
    check("every site vector is unit and the right width",
          all(abs(sv.unit_norm - 1.0) < 1e-6 and sv.vector.shape == (hidden,)
              for sv in result.site_vectors),
          str([round(sv.unit_norm, 8) for sv in result.site_vectors]))
    check("every site records a positive raw_norm",
          all(sv.raw_norm > 0.0 for sv in result.site_vectors),
          str([round(sv.raw_norm, 4) for sv in result.site_vectors]))
    check("sites differ from one another (the extraction is per-site, not broadcast)",
          len({sv.vector.tobytes() for sv in result.site_vectors}) == 3)

    # ---- THE MATCHED-RNG CONTRACT, measured -----------------------------------
    check("MATCHED RNG: every pair's two conditions consumed the identical stream",
          result.matched_rng_holds and all(
              len({c.rng_state_sha256 for c in p.conditions}) == 1
              for p in result.pairs),
          f"{len(result.pairs)} pairs, 1 distinct RNG state each")
    check("MATCHED RNG: both conditions of a pair carry the pair's seed",
          all(all(c.seed == p.seed for c in p.conditions) for p in result.pairs))
    check("MATCHED RNG: distinct cells got DISTINCT streams",
          len({p.conditions[0].rng_state_sha256 for p in result.pairs})
          == N_PAIRS_REQUIRED,
          f"{len({p.conditions[0].rng_state_sha256 for p in result.pairs})} distinct")
    check("the seed table is the recipe's formula, cell for cell",
          all(p.seed == pair_seed(p.topic_index, p.template_index)
              for p in result.pairs))
    # The two conditions must reach the model as DIFFERENT prompts, or the matched RNG
    # makes their generations identical and every difference is exactly zero. That is
    # a real failure mode (it happened while writing this selftest, with a tokenizer
    # that truncated both system prompts to "You ar"), so it is asserted directly.
    f_ids = chat_ids(tok, "Write about X.", FORMAL_SYS, "12 Jul 2026")
    i_ids = chat_ids(tok, "Write about X.", INFORMAL_SYS, "12 Jul 2026")
    check("the two opposed system prompts reach the model as different token ids",
          f_ids.shape != i_ids.shape or not bool(torch.equal(f_ids, i_ids)),
          f"{tuple(f_ids.shape)} vs {tuple(i_ids.shape)}")
    check("no site vector came out degenerate (a zero difference would be refused)",
          all(sv.raw_norm > 0.0 for sv in result.site_vectors))
    check("date_string reached the chat template on every one of the 80 calls",
          set(template_calls_in_build) == {"12 Jul 2026"}
          and len(template_calls_in_build) == 2 * N_PAIRS_REQUIRED,
          f"{len(template_calls_in_build)} calls")

    # ---- the n_pairs assertion FIRES, and nothing banks ------------------------
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "vectors"
        short = BuildRequest(model_key="toy-llama", model_path="(none)", out_dir=out,
                             sites=(1, 2), max_new_tokens=4, device="cpu")
        try:
            build_formality_contrast(model, tok, manifest, short, sampling, "cpu",
                                     "12 Jul 2026")
            check("PairCountError fires when generations are too short", False)
        except PairCountError as exc:
            check("PairCountError fires when generations are too short", True,
                  f"{'40' in str(exc)} message names the expectation")
        check("a HALTed build wrote NO output files (nothing partial banks)",
              not out.exists() or not list(out.iterdir()),
              str(sorted(p.name for p in out.iterdir())) if out.exists() else "(no dir)")

    # ---- site range + wrapper-config hidden_dim + date-free template HALT ------
    bad = BuildRequest(model_key="toy-llama", model_path="(none)", out_dir=Path("."),
                       sites=(n_layers,), max_new_tokens=10, device="cpu")
    try:
        build_formality_contrast(model, tok, manifest, bad, sampling, "cpu",
                                 "12 Jul 2026")
        check("a site beyond the decoder stack is refused before any generation", False)
    except SiteOutOfRangeError as exc:
        check("a site beyond the decoder stack is refused before any generation",
              str(n_layers) in str(exc), f"{n_layers} layers")

    class _WrapperCfg:
        def __init__(self, inner_dim: int) -> None:
            self.text_config = type("TextCfg", (), {"hidden_size": inner_dim})()

    class _StubModel:
        def __init__(self, cfg_obj: Any) -> None:
            self.config = cfg_obj

    check("hidden_dim resolves through a wrapper config's text_config (the gemma rake)",
          resolve_hidden_dim(_StubModel(_WrapperCfg(5376))) == 5376)
    check("hidden_dim falls back to the preset when the config declares none",
          resolve_hidden_dim(_StubModel(type("Bare", (), {})()), 4096) == 4096)
    try:
        resolve_hidden_dim(_StubModel(type("Bare", (), {})()), None)
        check("an unresolvable hidden_dim is refused, not guessed", False)
    except FormalityContrastBuildError:
        check("an unresolvable hidden_dim is refused, not guessed", True)
    check("hidden_dim prefers the real config over the preset",
          resolve_hidden_dim(model, 99999) == hidden, str(hidden))

    df = BuildRequest(model_key="toy-llama", model_path="(none)", out_dir=Path("."),
                      sites=(1,), max_new_tokens=10, device="cpu")
    try:
        build_formality_contrast(model, _DateFreeTokenizer(cfg.vocab_size), manifest,
                                 df, sampling, "cpu", "12 Jul 2026")
        check("a template that rejects date_string HALTs (no silent fall-back)", False)
    except ChatTemplateError as exc:
        check("a template that rejects date_string HALTs (no silent fall-back)",
              "date_string" in str(exc))

    # ---- the loaders this builder reuses are the collector's, not forks --------
    # The certified sharded path (prereg §4, gate PASSED 2026-07-27) certifies the
    # COLLECTOR's loader; a fork of it would certify nothing. `main` reaches these by
    # `from ... import`, so the check that matters is that the names it imports are the
    # collector's objects and that this module defines no same-named shadow.
    import metabasis.scripts.build_formality_contrast as me
    import metabasis.scripts.collect_mean_states as cms
    for name in ("load_model_and_tok", "load_model_and_tok_sharded",
                 "build_even_layer_device_map", "trunk_stamp", "ShardSpec",
                 "parse_shard_across", "parse_max_memory", "VMB_CANONICAL_DATE"):
        check(f"{name} is reused from the collector, not forked here",
              hasattr(cms, name) and not hasattr(me, name),
              "imported inside main()")
    check("the certified sharded loader is reachable and is the collector's",
          callable(cms.load_model_and_tok_sharded)
          and callable(cms.build_even_layer_device_map))
    return checks


# ---------------------------------------------------------------- CLI
def _parse_sites(spec: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in spec.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"--sites: cannot parse {spec!r} ({exc})") from exc


def _parse_eos(spec: Optional[str]) -> Optional[tuple[int, ...]]:
    if spec is None:
        return None
    try:
        return tuple(int(x) for x in spec.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"--eos-token-ids: cannot parse {spec!r} ({exc})") from exc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="CPU verification (no weights, no GPU, no data tree)")
    ap.add_argument("--model", help="bank key from the metabasis registries "
                                    "(roster / MODEL_PRESETS)")
    ap.add_argument("--model-path", help="local weights dir (node-side)")
    ap.add_argument("--sites", help="comma list of decoder-layer indices; ALL of them "
                                    "come out of one extraction pass")
    ap.add_argument("--out-dir", type=Path,
                    help="the model's vectors dir; files are "
                         "formality_contrast_<model>_L<site>.npz + _stamps.json")
    ap.add_argument("--topics", type=Path, default=None,
                    help=f"topic manifest (default: the tracked corpus/"
                         f"{TOPIC_MANIFEST_NAME})")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature. Required for a model with no "
                         "MODEL_PRESETS row (the 405B) — this builder will not guess it")
    ap.add_argument("--eos-token-ids", default=None,
                    help="comma list of stop ids; also supplies the pad-id fall-back. "
                         "Required for a model with no MODEL_PRESETS row")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help=f"the recipe's value is {MAX_NEW_TOKENS}; moving it makes the "
                         f"vector incomparable with the banked builds")
    ap.add_argument("--top-p", type=float, default=TOP_P,
                    help=f"the recipe's value is {TOP_P} (see --max-new-tokens)")
    ap.add_argument("--device", default="cuda", help="single-card path only")
    ap.add_argument("--shard-across", default=None,
                    help="comma list of devices ('0,1') or a count ('8') to split the "
                         "decoder stack across, via the COLLECTOR's certified sharded "
                         "loader (prereg §4; sharding gate PASSED 2026-07-27 for "
                         "FORWARDS — generation reproducibility is NOT certified and "
                         "the stamp says so). The 405B path.")
    ap.add_argument("--max-memory", default=None,
                    help="'0=170GiB,cpu=0GiB' or @file.json — the usual way to forbid "
                         "a silent CPU spill")
    ap.add_argument("--offload-folder", default=None)
    ap.add_argument("--allow-offload", action="store_true",
                    help="accept a weights spill to cpu/disk. STANDING DESK RULING: "
                         "never on the 405B")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="bank over existing files of the same name. Off by default: "
                         "the ops rule is a bank census first, and no vector of this "
                         "class exists in any v2.1 bank")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    missing = [f"--{n.replace('_', '-')}" for n in
               ("model", "model_path", "sites", "out_dir")
               if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")

    # Everything that can be refused WITHOUT reading a weight is refused first: the
    # topic manifest, the sampling resolution, the request shape. A 405B load is ~20
    # minutes; a typo should not cost it.
    try:
        manifest = load_topic_manifest(args.topics)
        sampling = resolve_sampling(args.model, args.temperature,
                                    _parse_eos(args.eos_token_ids))
        request = BuildRequest(
            model_key=args.model, model_path=args.model_path, out_dir=args.out_dir,
            sites=_parse_sites(args.sites), max_new_tokens=args.max_new_tokens,
            top_p=args.top_p, device=args.device,
            allow_overwrite=args.allow_overwrite)
        # THE BANK CENSUS, BEFORE THE WEIGHTS. The 405B build is a whole-node window;
        # finding a collision after it would waste the window and teach nothing.
        census_targets(request, request.sites)
    except (FormalityContrastBuildError, ValueError) as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc
    logger.info("bank census: %d target file(s) planned under %s, none present",
                2 * len(request.sites), request.out_dir)
    logger.info("topic manifest: %s (sha %s), %d topics", manifest.path.name,
                manifest.sha256[:12], manifest.n_topics)
    logger.info("sampling: %s", json.dumps(sampling.stamp))
    if args.max_new_tokens != MAX_NEW_TOKENS or args.top_p != TOP_P:
        logger.warning("SAMPLING DEVIATION: max_new_tokens=%d top_p=%s (recipe: %d / "
                       "%s) — the built vector is NOT comparable with the banked "
                       "builds of this class", args.max_new_tokens, args.top_p,
                       MAX_NEW_TOKENS, TOP_P)

    from metabasis.scripts.collect_mean_states import (
        VMB_CANONICAL_DATE, DeviceMapSpecError, ShardSpec, ShardedLoadError,
        load_model_and_tok, load_model_and_tok_sharded, parse_max_memory,
        parse_shard_across, trunk_stamp)

    sharding: Optional[dict] = None
    try:
        if args.shard_across:
            spec = ShardSpec(
                shard_across=parse_shard_across(args.shard_across),
                max_memory=(parse_max_memory(args.max_memory)
                            if args.max_memory else None),
                offload_folder=args.offload_folder,
                allow_offload=args.allow_offload,
                require_multi_device=True,
                spec_echo=f"--shard-across {args.shard_across}")
            model, tok, device, sharding = load_model_and_tok_sharded(
                request.model_path, spec, request.sites)
        else:
            model, tok = load_model_and_tok(request.model_path, request.device)
            device = request.device
    except (OSError, ValueError, RuntimeError, DeviceMapSpecError,
            ShardedLoadError) as exc:
        raise SystemExit(
            f"cannot load {request.model_key} from {request.model_path}: "
            f"{type(exc).__name__}: {exc}") from exc
    # No gradients are ever taken here, but freezing costs nothing and keeps a stray
    # autograd graph from being retained across 80 generations.
    model.requires_grad_(False)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(
            f"ChatTemplateError: {request.model_key}'s tokenizer carries no chat "
            "template, so it cannot take the system-role contrast this vector class "
            "is built from. Base checkpoints are out of the class by construction.")
    trunk = trunk_stamp(request.model_path, tok, device, sharding=sharding)
    logger.info("trunk: %s", json.dumps(trunk)[:300])

    try:
        result = build_formality_contrast(model, tok, manifest, request, sampling,
                                         device, VMB_CANONICAL_DATE, sharding)
    except FormalityContrastBuildError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"FORMALITY-CONTRAST-BUILD-INCOMPLETE model={request.model_key} "
              f"sites={list(request.sites)} reason={type(exc).__name__} — nothing "
              "was banked")
        return 3
    except Exception as exc:                                       # noqa: BLE001
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.error("CUDA OOM: %s", exc)
            print(f"FORMALITY-CONTRAST-BUILD-INCOMPLETE model={request.model_key} "
                  "reason=OOM — nothing was banked. The dose and the sites are NOT "
                  "negotiable in-enactment: report, do not improvise a smaller "
                  "config. The only sanctioned lever is placement "
                  "(--shard-across / --max-memory), and NEVER free memory by "
                  "touching another process.")
            return 4
        raise

    try:
        paths = bank(result, trunk)
    except BankCollisionError as exc:
        logger.error("%s", exc)
        print(f"FORMALITY-CONTRAST-BUILD-INCOMPLETE model={request.model_key} "
              f"reason=BankCollisionError — nothing was banked")
        return 5

    summary = {
        "model": request.model_key,
        "sites": list(request.sites),
        "npz_keys": {f"L{s}": CANONICAL_VECTOR_KEY.format(site=s)
                     for s in request.sites},
        "paths": {f"L{s}": str(paths[s]["vector"]) for s in request.sites},
        "n_pairs_effective": result.n_pairs_effective,
        "n_pairs_skipped": result.n_pairs_skipped,
        "matched_rng_holds": result.matched_rng_holds,
        "topic_manifest_sha256": result.manifest.sha256,
        "hidden_dim": result.hidden_dim,
        "n_decoder_layers": result.n_decoder_layers,
        "temperature": result.sampling.temperature,
        "temperature_source": result.sampling.temperature_source,
        "per_site": {f"L{sv.site}": {"raw_norm": sv.raw_norm,
                                     "unit_norm": sv.unit_norm,
                                     "sign_consistency": sv.per_pair_sign_consistency,
                                     "coherence": sv.per_pair_pairwise_coherence}
                     for sv in result.site_vectors},
        "wall_seconds": result.wall_seconds,
        "cuda_visible_devices": trunk["cuda_visible_devices"],
        "sharded_generation_certified": False,
    }
    if sharding is not None:
        summary["n_compute_devices"] = sharding["n_compute_devices"]
        summary["compute_devices"] = sharding["compute_devices"]
    print(json.dumps(summary, indent=1))
    print(f"FORMALITY-CONTRAST-BUILD-OK model={request.model_key} "
          f"sites={','.join('L' + str(s) for s in request.sites)} "
          f"n_pairs={result.n_pairs_effective}/{N_PAIRS_REQUIRED} "
          f"matched_rng={result.matched_rng_holds} "
          f"card={trunk['cuda_visible_devices']} wall={result.wall_seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
