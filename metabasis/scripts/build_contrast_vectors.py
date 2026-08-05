"""Build TEXT-CONTRAST class vectors — the CAA construction of record + the repeng row.

The class-pilot's two constructions live in ONE module because they consume the SAME
contrast set and differ only in how a set of paired residual states becomes a
direction. Reading them side by side is the point: the method row is method
DIVERSITY on shared contrast data, not a second experiment.

    caa_<axis>_L<site>       = unit( mean_pairs(h+) - mean_pairs(h-) )
    repengpca_<axis>_L<site> = sign-fixed first principal component of the
                               PAIRED DIFFERENCES h+ - h-, centered (sklearn's
                               PCA centers; a reimplementation that forgets to
                               is a different object, so the centering is
                               asserted here rather than assumed)

RULINGS THIS MODULE IS BUILT AGAINST (all ledgered 2026-08-04/05, session 12)

  * **Construction of record: TEXT-CONTRAST CAA** — "mean states over contrastive
    raw text pairs — no system prompt needed" → base models get native
    comparators and the instruct-only constraint dissolves for
    language/sentiment/formality. Refusal's base-absence is a fact about the
    behavior, stated, not engineered around.
  * **FRESH public contrast sets published in-repo with shas** — which kills the
    V1 topic-leakage caveat. The banked V1 formality vectors
    (`build_formality_contrast.py`) are quotable only as fresh-vs-banked cosine
    ANCHORS; this module never consumes them.
  * **repeng METHOD ROW on SENTIMENT**, construction verbatim, **single-site
    application NAMED as a variant with expected attenuation** — because repeng
    publishes a MULTI-SITE actuator and our harness writes at ONE site. The stamp
    says so in a field, not in a footnote (`application_scope`).
  * **SCOPE GUARD** — the behavioral slate is frozen at {EGV column + 4 needle
    axes + repeng row + L1 panel + §10 tier contrast}. `AXES_OF_RECORD` is that
    guard as arithmetic: a fifth axis cannot be spelled here at all.

THE REFUSAL THAT MAKES THIS SAFE TO RUN OVERNIGHT
-------------------------------------------------
**This module REFUSES to build from a contrast set whose sha256 is not RULED in a
pin file.** It mirrors `build_behavioral_banks.py`'s band-recipe guard exactly in
shape: there, the refusal lifts for one ruled recipe string and no other; here, it
lifts for one ruled sha and no other. A `PROPOSED` row is refused BY STATUS with
its own message, because on 2026-08-05 every draft set is PROPOSED and the set
freeze is Luxia's morning word — so tonight's drafts provably cannot become a
vector, no matter what a later command line asks for. Six independent ways to be
refused, each with its own exception class:

    no pin file · unrecognized pin schema · set sha absent from the pins ·
    pin row still PROPOSED · pin row's axis ≠ the set's axis ·
    a RULED row with no ruler or no date on it

WHAT THIS MODULE DOES NOT DO. It does not score, does not transport (transport is
`fit_transport_maps` + the readouts), does not judge, and never self-stamps: every
stamp carries `UNSTAMPED (C§8)`. It does not build a lesion recipe — the
constructions are a CLOSED set (§5.4's law, same treatment as the staging module).
It refuses to hand a MULTI-SITE family to anything that transports: multi-site
transport is a NAMED FUTURE ARC with its own pre-registration (Luxia, 2026-08-04),
and the multi-site build here exists ONLY for the source-side decomposition
control (native multi-site vs native single-site-at-record, no transport, which
attributes a weak transported repeng row to application cost vs transport cost).

TWO LANES, ONE OF WHICH NEEDS NO GPU
------------------------------------
  * **construction lane (CPU, pure numpy):** paired states in → direction out.
    Everything gate-bearing lives here, so the whole contract is selftestable on
    synthetic states with no weights, no torch, and no data tree.
  * **extraction lane (`--extract`, GPU, node):** contrast set + model →
    paired states. It does NOT re-implement state capture: it builds
    collector-shaped entries and calls `collect_mean_states.compute_means`
    VERBATIM, so "the vector lives in the residual space the transport maps were
    fit in" is code reuse rather than an argument. The native arm wraps each text
    in the chat template under ONE axis-neutral carrier prompt, identical on both
    sides of every pair, so the carrier cancels in the difference; the raw arm is
    the bare text. Both arms where the model has both.

CPU self-test (no weights, no GPU, no data tree, no torch, no sklearn):

    python -m metabasis.scripts.build_contrast_vectors --selftest

RAKE M44 — THE CONFIGURATION MATRIX IS THE MERGE BAR. Every fixture is built in a
`TemporaryDirectory` and nothing resolves a banked path relative to cwd (the
dcbe7d7 pattern), so the {data tree, no data tree} axis is vacuous and asserted
vacuous. The axes that are NOT vacuous are {torch present, absent} — the
extraction lane's convention proof — and {sklearn present, absent} — the
independent cross-check of the PCA. Each degrades to a NAMED skip carrying its
triggering condition, counted in the tail.

RAKE M45 — `selftest()` RETURNS an int and never `sys.exit()`s, so an all-module
sweep records a result instead of dying at this file. M25 — no `hash()`, no
unseeded RNG; in fact neither construction draws a random number at all, which is
why build-twice here is an assertion about linear algebra rather than about seeds.

Typical use (repo root):

    python -m metabasis.scripts.build_contrast_vectors --example-pins > pins.json
    python -m metabasis.scripts.build_contrast_vectors --verify-set SET.json
    python -m metabasis.scripts.build_contrast_vectors \\
        --set SET.json --pins pins.json --states STATES.npz \\
        --construction caa --sites 26 --out-dir <VECTORS_DIR>
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")     # the 2026-08-01 ruling, before numpy

import argparse                                                       # noqa: E402
import hashlib                                                        # noqa: E402
import json                                                           # noqa: E402
import logging                                                        # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402
from typing import Any, Callable, Literal, Optional, Sequence         # noqa: E402

import numpy as np                                                    # noqa: E402
from pydantic import (BaseModel, ConfigDict, Field, field_validator,  # noqa: E402
                      model_validator)

from metabasis.threads import (RULED_OMP_NUM_THREADS, ThreadConfig,   # noqa: E402
                               effective_thread_config,
                               thread_config_stamp)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_contrast_vectors")

# ---------------------------------------------------------------- contracts

#: The C§8 grade every stamp carries. Nothing here self-scores.
GRADE_LINE = "UNSTAMPED (C§8)"

#: The contrast-set document contract. A document this module does not recognize is
#: refused rather than guessed at — a silently-changed schema is a vector built
#: against terms nobody agreed to.
CONTRAST_SET_SCHEMA_VERSION = "contrast-set/1"

#: The pin-file contract. Same reasoning, one file over: the pin file is the ONLY
#: thing standing between a draft and a banked object, so its shape is pinned too.
CONTRAST_SET_PIN_SCHEMA_VERSION = "contrast-set-pins/1"

#: The vector-bundle contract (what `--build` writes beside the npz).
CONTRAST_VECTOR_SCHEMA_VERSION = "contrast-vectors/1"

#: **THE SCOPE GUARD AS ARITHMETIC** (Luxia, 2026-08-04, in writing): the behavioral
#: phase slate is FROZEN at four needle axes. An axis outside this tuple is refused
#: by name at every entry point, so a fifth axis needs a dated ruling and a code
#: change — it cannot arrive through a spec field.
AXES_OF_RECORD: tuple[str, ...] = ("language", "sentiment", "formality", "refusal")

#: Which axes have an induce-only framing (Luxia: refusal is induce-only — the
#: vector induces refusal; the suppression direction is a jailbreak object and is
#: out of scope). Recorded on the set and asserted against the pin row.
INDUCE_ONLY_AXES: tuple[str, ...] = ("refusal",)

#: The construction of record and its method-row sibling, as pinned strings. A
#: recipe whose `construction_of_record` is not one of these is refused — the same
#: mechanical move `build_behavioral_banks.RandomBandRecipe` makes for the band.
CAA_CONSTRUCTION_OF_RECORD = "text-contrast-caa/difference-of-means/ledger-2026-08-04"
REPENG_CONSTRUCTION_OF_RECORD = "repeng-pca-diff/single-site-application/ledger-2026-08-04"

CAA_CONSTRUCTION_TEXT = (
    "unit( mean over pairs of (positive mean-residual-state at the site) - mean over "
    "pairs of (negative mean-residual-state at the site) ). States are per-text fp32 "
    "means over the text's own token positions, captured by "
    "collect_mean_states.compute_means (forward_pre_hook on decoder_layers[site]) — "
    "the residual stream ENTERING decoder layer `site`, the space the transport maps "
    "were fit in. Banked UNIT; the pre-normalization norm is banked beside as "
    "raw_norm (the build_formality_contrast / build_injection_banks convention).")

REPENG_CONSTRUCTION_TEXT = (
    "The published repeng `pca_diff` reading: stack the pair states interleaved "
    "[pos_0, neg_0, pos_1, neg_1, ...]; take train = h[::2] - h[1::2] (the PAIRED "
    "differences); fit a 1-component PCA to `train` — which CENTERS `train` by its "
    "own mean, so this is NOT the mean difference and a reimplementation that skips "
    "the centering builds a different object; take the component; then fix the sign "
    "by projecting the full stack onto it and requiring the positive member of a "
    "pair to project LARGER than its negative on a majority of pairs. Components are "
    "unit by construction (SVD right-singular vector).")

#: **The single-site NAME (Luxia's ruling, verbatim in force).** repeng publishes a
#: multi-site actuator: one direction per layer, all attached at once. Our harness
#: writes at ONE site. That is a VARIANT, it is named as one in every stamp, and the
#: expected consequence — attenuation relative to the published application — is
#: pre-stated so a weak row cannot later be read as a transport failure.
REPENG_APPLICATION_SCOPE = (
    "SINGLE-SITE APPLICATION — a NAMED VARIANT of the published method. repeng "
    "trains and attaches one direction PER LAYER across a band of layers; the "
    "metabasis behavioral harness writes at the node's single site of record. "
    "ATTENUATION RELATIVE TO THE PUBLISHED APPLICATION IS EXPECTED AND PRE-STATED. "
    "The source-side decomposition control (native multi-site vs native "
    "single-site-at-record on the SOURCE, no transport) is what attributes a weak "
    "transported repeng row to application cost rather than transport cost.")

#: Multi-site families exist for that control and for nothing else. Transport of a
#: multi-site family is a NAMED FUTURE ARC with its own pre-registration; this
#: module refuses to label one as transportable so the arc cannot start by accident.
MULTI_SITE_SCOPE = (
    "SOURCE-SIDE CONTROL ONLY — no transport. Multi-site transport is a named "
    "future arc (Luxia, 2026-08-04) with its own pre-registration; depth-indexed "
    "map families do not exist in this campaign and no map here carries one.")

#: The banked key templates. Descriptive by construction — a reader who has only the
#: key knows the construction, the axis and the site.
CAA_KEY_TEMPLATE = "caa_{axis}_L{site}"
REPENG_KEY_TEMPLATE = "repengpca_{axis}_L{site}"

#: The native arm's carrier prompt. Text-contrast CAA needs NO system prompt (the
#: ruling), but a chat template needs a user turn, so the text rides as the
#: completion under ONE axis-neutral carrier that is IDENTICAL on both sides of
#: every pair — therefore it cancels in the difference. Recorded in every stamp so
#: the cancellation argument is checkable rather than trusted.
NATIVE_CARRIER_PROMPT = "Write about: general knowledge"

#: `date_string` pin for chat templates that take one (the collector's convention);
#: a template that renders today's date would make the build unreproducible.
DATE_STRING_PIN = "26 Jul 2026"

#: The closed set of constructions. Anything outside it is refused BY NAME, which is
#: what makes §5.4's lesion-recipe law mechanical rather than a habit: a lesion
#: recipe cannot be spelled here at all.
ADMISSIBLE_CONSTRUCTIONS: tuple[str, ...] = ("caa", "repeng_pca")

#: Substrings that name a lesion recipe (§5.4). Same list the staging module refuses.
LESION_MARKERS: tuple[str, ...] = (
    "project_out", "projectout", "orthogonalize", "residualize", "lesion",
    "subtract entropy", "entropy_gradient", "perp", "ablate")


# ---------------------------------------------------------------- error taxonomy
class ContrastBuildError(RuntimeError):
    """Base class for every refusal in this module. Each subclass is a HALT."""


class AxisNotOfRecord(ContrastBuildError):
    """The scope guard: the slate is frozen at four axes and this is not one."""


class ContrastSetError(ContrastBuildError):
    """The contrast-set document is unreadable, incomplete, or self-inconsistent."""


class PinFileError(ContrastBuildError):
    """The pin file is missing, unreadable, or speaks a contract we do not."""


class ContrastSetNotPinned(ContrastBuildError):
    """The set's sha256 is not RULED in the pin file — the refusal that matters.

    This is the class-pilot's analogue of the band-recipe guard: a vector is a
    claim about a specific corpus of contrast text, so the corpus is identified by
    digest and the digest is ruled, or nothing is built.
    """


class ContrastSetNotRuled(ContrastSetNotPinned):
    """The sha is IN the pin file but its row is still PROPOSED.

    Kept distinct from `ContrastSetNotPinned` because the two failures mean
    different things to a reader: "I have never heard of this set" versus "this is
    the set you meant, and nobody has ruled it yet." On 2026-08-05 every draft set
    is in the second state by design.
    """


class ConstructionRefused(ContrastBuildError):
    """§5.4's law, or a construction outside the closed set."""


class DimensionMismatch(ContrastBuildError):
    """States do not agree with each other or with the set they claim to come from."""


class PairCountError(ContrastBuildError):
    """The states do not carry the number of pairs the set declares."""


class DegenerateDirection(ContrastBuildError):
    """The construction produced a zero/non-finite direction instead of a vector."""


class StatesError(ContrastBuildError):
    """The states bundle is unreadable or does not carry what it claims."""


# ---------------------------------------------------------------- small utilities
def sha256_file(path: Path) -> str:
    """Full digest, never padded (rake M40)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unit(v: np.ndarray, *, what: str = "direction") -> np.ndarray:
    """Unit-normalize, refusing a degenerate direction rather than emitting NaN."""
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise DegenerateDirection(f"{what}: non-finite entries — refusing to normalize")
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise DegenerateDirection(
            f"{what}: norm {n!r} — the two sides of the contrast produced the same "
            "mean state, so there is no direction here. A zero vector normalized is "
            "NaN, and a NaN banked is a silent hole in a column.")
    return (v / n).astype(np.float32)


def assert_axis_of_record(axis: str, *, what: str = "axis") -> str:
    """The scope guard, at every entry point."""
    if axis not in AXES_OF_RECORD:
        raise AxisNotOfRecord(
            f"{what}={axis!r} is not one of the four ruled needle axes "
            f"{AXES_OF_RECORD}. The behavioral slate is FROZEN (Luxia, 2026-08-04, "
            "in writing): no further axes without a dated ruling stating the "
            "evidentiary purpose. REFUSED.")
    return axis


def assert_construction_admissible(construction: str, *, key: str = "") -> str:
    """The closed set + §5.4's lesion-recipe law, at the construction site."""
    low = construction.lower()
    for marker in LESION_MARKERS:
        if marker in low:
            raise ConstructionRefused(
                f"{key or construction!r}: §5.4's lesion-recipe law forbids building "
                f"any object whose construction contains {marker!r}. REFUSED at "
                "construction, which is where the law binds.")
    if construction not in ADMISSIBLE_CONSTRUCTIONS:
        raise ConstructionRefused(
            f"{key or ''}construction {construction!r} is outside the closed set "
            f"{ADMISSIBLE_CONSTRUCTIONS}. This module builds two constructions and "
            "refuses every other by name.")
    return construction


def _read_json(path: Path, *, what: str, exc: type[ContrastBuildError]) -> dict:
    if not path.exists():
        raise exc(f"no {what} at {path}")
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc_:
        raise exc(f"{path}: unreadable {what} "
                  f"({type(exc_).__name__}: {exc_})") from exc_
    if not isinstance(doc, dict):
        raise exc(f"{path}: a {what} is a JSON object")
    return doc


# ---------------------------------------------------------------- contrast sets
class SourcePin(BaseModel):
    """One public source a contrast set was derived from, pinned by revision.

    A source without a revision is not a pin: an upstream repo can be force-pushed,
    and then "we used dataset X" names nothing reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    config: Optional[str] = None
    split: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    year: Optional[int] = None

    @field_validator("revision")
    @classmethod
    def _is_a_commit(cls, v: str) -> str:
        if v.lower() in {"main", "master", "head"}:
            raise ValueError(
                f"revision={v!r} is a branch, not a pin — a branch moves and the "
                "set stops being reproducible the day it does")
        return v


class ContrastPair(BaseModel):
    """One contrast pair: the positive side, the negative side, and their digests.

    Digests are carried per side so a set can be published as a RECONSTRUCT recipe
    plus per-text shas (the `corpus/webtext-v3` golden path) without ever shipping
    a body — and so a body that drifts is caught text by text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str = Field(min_length=1)
    positive_text: str = Field(min_length=1)
    negative_text: str = Field(min_length=1)
    positive_sha256: Optional[str] = None
    negative_sha256: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def verified_shas(self) -> tuple[str, str]:
        """Recompute both digests; raise if a declared one disagrees."""
        p, n = sha256_text(self.positive_text), sha256_text(self.negative_text)
        for got, declared, side in ((p, self.positive_sha256, "positive"),
                                    (n, self.negative_sha256, "negative")):
            if declared is not None and declared != got:
                raise ContrastSetError(
                    f"{self.pair_id}/{side}: declared sha256 {declared[:12]}… but the "
                    f"body digests to {got[:12]}… — the body drifted from its manifest")
        return p, n


class ContrastSet(BaseModel):
    """A DRAFT or RULED contrast set: pinned public sources, a selection rule, pairs.

    `selection_rule` is prose ON PURPOSE and is load-bearing prose: it is the only
    thing that lets a reader rebuild the set from the pins without reading code, and
    it is what Luxia reads when she rules the set. A set without one is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["contrast-set/1"] = CONTRAST_SET_SCHEMA_VERSION
    set_id: str = Field(min_length=1)
    axis: str
    positive_class: str = Field(min_length=1)
    negative_class: str = Field(min_length=1)
    induce_only: bool = False
    sources: list[SourcePin] = Field(min_length=1)
    selection_rule: str = Field(min_length=16)
    license_note: str = Field(min_length=1)
    parallel: bool = Field(
        description="True when the two sides are content-matched (a translation, a "
                    "minimally-edited twin); False for a class contrast, where the "
                    "two sides are different texts and content is a confound the "
                    "readout must name.")
    n_pairs: int = Field(gt=0)
    pairs: list[ContrastPair] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @field_validator("axis")
    @classmethod
    def _axis_of_record(cls, v: str) -> str:
        return assert_axis_of_record(v, what="set.axis")

    @model_validator(mode="after")
    def _coherent(self) -> "ContrastSet":
        if len(self.pairs) != self.n_pairs:
            raise ValueError(
                f"{self.set_id}: declares n_pairs={self.n_pairs} but carries "
                f"{len(self.pairs)} pairs — a vector built from a different count is "
                "a different object (the 8-pairs-instead-of-40 rake)")
        ids = [p.pair_id for p in self.pairs]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"{self.set_id}: duplicate pair_id(s) {dupes[:5]}")
        if self.axis in INDUCE_ONLY_AXES and not self.induce_only:
            raise ValueError(
                f"{self.set_id}: axis {self.axis!r} is INDUCE-ONLY by ruling "
                "(Luxia, 2026-08-04) and the set must say so in a field")
        return self

    def pair_manifest(self) -> list[dict[str, str]]:
        """Per-pair digests — the publishable half of the set (no bodies)."""
        out = []
        for p in self.pairs:
            ps, ns = p.verified_shas()
            out.append({"pair_id": p.pair_id, "positive_sha256": ps,
                        "negative_sha256": ns})
        return out


def load_contrast_set(path: Path) -> ContrastSet:
    """Read a contrast set, refusing an unrecognized contract by name."""
    doc = _read_json(path, what="contrast set", exc=ContrastSetError)
    version = doc.get("schema_version")
    if version != CONTRAST_SET_SCHEMA_VERSION:
        raise ContrastSetError(
            f"{path}: contrast-set schema {version!r}, this build speaks "
            f"{CONTRAST_SET_SCHEMA_VERSION!r}. Refusing a contract nobody agreed to.")
    try:
        return ContrastSet(**doc)
    except (TypeError, ValueError) as exc:
        raise ContrastSetError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------- the pin guard
class ContrastSetPinRow(BaseModel):
    """One row of the ruled pin file: a sha, an axis, and who ruled it when."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_id: str = Field(min_length=1)
    axis: str
    sha256: str = Field(min_length=64, max_length=64)
    status: Literal["RULED", "PROPOSED"]
    ruled_by: Optional[str] = None
    ruled_on: Optional[str] = None
    note: Optional[str] = None

    @field_validator("sha256")
    @classmethod
    def _is_a_digest(cls, v: str) -> str:
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"sha256={v!r} is not 64 hex characters (M40: never "
                             "pad, never abbreviate a digest in a pin)")
        return v.lower()

    @field_validator("axis")
    @classmethod
    def _axis_of_record(cls, v: str) -> str:
        return assert_axis_of_record(v, what="pin.axis")


class ContrastSetPinFile(BaseModel):
    """The ruled pin file. The ONLY thing standing between a draft and a vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["contrast-set-pins/1"] = CONTRAST_SET_PIN_SCHEMA_VERSION
    rulings: list[ContrastSetPinRow] = Field(default_factory=list)
    note: Optional[str] = None

    @model_validator(mode="after")
    def _no_duplicate_shas(self) -> "ContrastSetPinFile":
        shas = [r.sha256 for r in self.rulings]
        if len(set(shas)) != len(shas):
            dupes = sorted({s for s in shas if shas.count(s) > 1})
            raise ValueError(
                f"pin file carries the same sha twice ({[d[:12] for d in dupes]}) — "
                "one digest, one ruling, or 'which row applied' has no answer")
        return self


def load_pin_file(path: Optional[Path]) -> ContrastSetPinFile:
    """Read the pin file. `None` is itself a refusal — there is no default path.

    A default would be the whole guard's failure mode: the point is that building
    requires someone to have written a ruling down and to have said where it is.
    """
    if path is None:
        raise PinFileError(
            "no --pins given. This module builds ONLY from a contrast set whose "
            "sha256 is RULED in a pin file, and there is deliberately no default pin "
            "path: a default is how a guard becomes a formality.")
    doc = _read_json(path, what="pin file", exc=PinFileError)
    version = doc.get("schema_version")
    if version != CONTRAST_SET_PIN_SCHEMA_VERSION:
        raise PinFileError(
            f"{path}: pin schema {version!r}, this build speaks "
            f"{CONTRAST_SET_PIN_SCHEMA_VERSION!r}. REFUSED.")
    try:
        return ContrastSetPinFile(**doc)
    except (TypeError, ValueError) as exc:
        raise PinFileError(f"{path}: {exc}") from exc


def assert_contrast_set_pinned(set_path: Path, pins_path: Optional[Path], *,
                               contrast_set: Optional[ContrastSet] = None
                               ) -> tuple[str, ContrastSetPinRow]:
    """THE GUARD. Returns (set sha256, the ruling row) or raises, six ways.

    Mirrors `build_behavioral_banks.RandomBandRecipe._is_the_recipe_of_record`: there
    a refusal lifts for exactly one ruled recipe string; here it lifts for exactly
    one ruled digest. Everything else — including a set this desk drafted an hour
    ago — is refused, and the PROPOSED case is refused with its own message because
    that is the case that will actually happen.
    """
    pins = load_pin_file(pins_path)
    if not set_path.exists():
        raise ContrastSetError(f"no contrast set at {set_path}")
    digest = sha256_file(set_path)
    cset = contrast_set if contrast_set is not None else load_contrast_set(set_path)

    row = next((r for r in pins.rulings if r.sha256 == digest), None)
    if row is None:
        known = ", ".join(f"{r.set_id}:{r.sha256[:12]}…({r.status})"
                          for r in pins.rulings) or "(none)"
        raise ContrastSetNotPinned(
            f"{set_path} digests to {digest} and that sha is NOT in {pins_path}. "
            f"Pinned shas: {known}. A contrast vector is a claim about a specific "
            "corpus of contrast text; the corpus is identified by digest, and an "
            "unpinned digest is refused. REFUSED.")
    if row.status != "RULED":
        raise ContrastSetNotRuled(
            f"{set_path} ({row.set_id}, sha {digest[:12]}…) is pinned but its status "
            f"is {row.status!r}. The set freeze is LUXIA'S WORD — a drafted set is "
            "evidence for a decision, never an input to a build. REFUSED.")
    if row.axis != cset.axis:
        raise ContrastSetNotPinned(
            f"{set_path}: the set declares axis {cset.axis!r} but its ruling row "
            f"{row.set_id!r} was ruled for axis {row.axis!r}. A set cannot change "
            "axis after it was ruled. REFUSED.")
    if not (row.ruled_by and row.ruled_on):
        raise ContrastSetNotPinned(
            f"{set_path}: ruling row {row.set_id!r} is marked RULED with "
            f"ruled_by={row.ruled_by!r} ruled_on={row.ruled_on!r}. A ruling with no "
            "ruler and no date is a status field, not a ruling. REFUSED.")
    logger.info("pin guard PASSED: %s sha %s ruled by %s on %s",
                row.set_id, digest[:12] + "…", row.ruled_by, row.ruled_on)
    return digest, row


# ---------------------------------------------------------------- recipes
class CAARecipe(BaseModel):
    """The construction of record, pinned to its ruled string."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    construction: Literal["caa"] = "caa"
    construction_of_record: str = CAA_CONSTRUCTION_OF_RECORD
    construction_text: str = CAA_CONSTRUCTION_TEXT

    @model_validator(mode="after")
    def _is_the_construction_of_record(self) -> "CAARecipe":
        if self.construction_of_record != CAA_CONSTRUCTION_OF_RECORD:
            raise ValueError(
                f"construction_of_record={self.construction_of_record!r} is not the "
                f"ruled string {CAA_CONSTRUCTION_OF_RECORD!r}. This module builds ONE "
                "CAA construction and refuses every other by name.")
        if self.construction_text != CAA_CONSTRUCTION_TEXT:
            raise ValueError(
                "construction_text differs from the ruled construction — the "
                "construction is the object, so its description cannot drift from it")
        return self


class RepengPCARecipe(BaseModel):
    """The method row, pinned — including the single-site NAME the ruling requires."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    construction: Literal["repeng_pca"] = "repeng_pca"
    construction_of_record: str = REPENG_CONSTRUCTION_OF_RECORD
    construction_text: str = REPENG_CONSTRUCTION_TEXT
    application_scope: str = REPENG_APPLICATION_SCOPE
    center_before_pca: bool = True

    @model_validator(mode="after")
    def _is_the_construction_of_record(self) -> "RepengPCARecipe":
        if self.construction_of_record != REPENG_CONSTRUCTION_OF_RECORD:
            raise ValueError(
                f"construction_of_record={self.construction_of_record!r} is not the "
                f"ruled string {REPENG_CONSTRUCTION_OF_RECORD!r}")
        if self.construction_text != REPENG_CONSTRUCTION_TEXT:
            raise ValueError("construction_text differs from the ruled construction")
        if self.application_scope != REPENG_APPLICATION_SCOPE:
            raise ValueError(
                "application_scope differs from the ruled naming. The single-site "
                "application is a NAMED variant with pre-stated attenuation; a stamp "
                "that softens the name would let a weak row read as transport failure")
        if not self.center_before_pca:
            raise ValueError(
                "center_before_pca=False is not the published method: sklearn's PCA "
                "centers, so a component fitted on uncentered differences is a "
                "different object. Refusing to call it the repeng row.")
        return self


Recipe = CAARecipe | RepengPCARecipe


# ---------------------------------------------------------------- paired states
class PairedStates(BaseModel):
    """Per-site paired mean residual states — the construction lane's whole input.

    Held as a typed object (rather than a bare dict of arrays) so that "these states
    came from this set, this node, this arm" is carried WITH the numbers and can be
    asserted against the set before a direction exists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid",
                              arbitrary_types_allowed=True)

    axis: str
    set_id: str
    set_sha256: Optional[str] = None
    node_key: str = Field(min_length=1)
    arm: Literal["native", "raw"]
    dim: int = Field(gt=0)
    n_pairs: int = Field(gt=0)
    pair_ids: tuple[str, ...] = ()
    positive: dict[int, np.ndarray]
    negative: dict[int, np.ndarray]

    @field_validator("axis")
    @classmethod
    def _axis_of_record(cls, v: str) -> str:
        return assert_axis_of_record(v, what="states.axis")

    @model_validator(mode="after")
    def _shapes_agree(self) -> "PairedStates":
        if set(self.positive) != set(self.negative):
            raise ValueError(
                f"positive sites {sorted(self.positive)} != negative sites "
                f"{sorted(self.negative)} — a half-populated site is a hole")
        if not self.positive:
            raise ValueError("no sites in the states bundle")
        for site, arr in list(self.positive.items()) + list(self.negative.items()):
            if arr.ndim != 2:
                raise ValueError(f"L{site}: states must be 2-D [n_pairs, dim], got "
                                 f"shape {arr.shape}")
            if arr.shape != (self.n_pairs, self.dim):
                raise ValueError(
                    f"L{site}: states are {arr.shape} but the bundle declares "
                    f"({self.n_pairs}, {self.dim})")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"L{site}: non-finite states — refusing to build a "
                                 "direction out of NaN")
        if self.pair_ids and len(self.pair_ids) != self.n_pairs:
            raise ValueError(f"{len(self.pair_ids)} pair_ids for {self.n_pairs} pairs")
        return self

    @property
    def sites(self) -> tuple[int, ...]:
        return tuple(sorted(self.positive))


def assert_states_match_set(states: PairedStates, cset: ContrastSet,
                            set_sha256: Optional[str] = None) -> None:
    """The states and the set must be the same experiment, checked by value."""
    if states.axis != cset.axis:
        raise ContrastSetError(
            f"states carry axis {states.axis!r} but the set is {cset.axis!r}")
    if states.set_id != cset.set_id:
        raise ContrastSetError(
            f"states carry set_id {states.set_id!r} but the set is {cset.set_id!r}")
    if states.n_pairs != cset.n_pairs:
        raise PairCountError(
            f"states carry {states.n_pairs} pairs, the set declares {cset.n_pairs}. "
            "A vector built from a different pair count is a different object; the "
            "counts are asserted, never reconciled.")
    if set_sha256 is not None and states.set_sha256 not in (None, set_sha256):
        raise ContrastSetError(
            f"states were extracted from set sha {states.set_sha256[:12]}… but the "
            f"set on disk digests to {set_sha256[:12]}… — the set changed under the "
            "states. REFUSED.")
    if states.pair_ids:
        want = tuple(p.pair_id for p in cset.pairs)
        if states.pair_ids != want:
            raise ContrastSetError(
                "states' pair_ids are not the set's pair_ids in order — the pairing "
                "is the construction, so its order is load-bearing")


def load_paired_states(path: Path) -> PairedStates:
    """Read a paired-states npz written by the extraction lane (or a fixture)."""
    if not path.exists():
        raise StatesError(f"no paired-states bundle at {path}")
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise StatesError(f"{path}: unreadable states bundle "
                          f"({type(exc).__name__}: {exc})") from exc
    with z:
        keys = set(z.files)
        if "meta" not in keys:
            raise StatesError(
                f"{path}: no `meta` array — a states bundle carries its own identity "
                f"(axis/set_id/node/arm), not just numbers. Found: {sorted(keys)}")
        try:
            meta = json.loads(str(z["meta"].item()))
        except (ValueError, AttributeError) as exc:
            raise StatesError(f"{path}: `meta` is not a JSON string") from exc
        pos, neg = {}, {}
        for k in sorted(keys):
            if k.startswith("pos_L"):
                pos[int(k[5:])] = np.asarray(z[k], dtype=np.float64)
            elif k.startswith("neg_L"):
                neg[int(k[5:])] = np.asarray(z[k], dtype=np.float64)
        if not pos:
            raise StatesError(f"{path}: no pos_L*/neg_L* arrays")
        pair_ids = tuple(str(x) for x in z["pair_ids"]) if "pair_ids" in keys else ()
    try:
        return PairedStates(positive=pos, negative=neg, pair_ids=pair_ids, **meta)
    except (TypeError, ValueError) as exc:
        raise StatesError(f"{path}: {exc}") from exc


def save_paired_states(path: Path, states: PairedStates) -> Path:
    """Write a states bundle (used by the extraction lane and by the selftest)."""
    meta = {"axis": states.axis, "set_id": states.set_id,
            "set_sha256": states.set_sha256, "node_key": states.node_key,
            "arm": states.arm, "dim": states.dim, "n_pairs": states.n_pairs}
    arrays: dict[str, Any] = {"meta": np.array(json.dumps(meta, sort_keys=True))}
    if states.pair_ids:
        arrays["pair_ids"] = np.array(states.pair_ids)
    for s in states.sites:
        arrays[f"pos_L{s}"] = states.positive[s].astype(np.float32)
        arrays[f"neg_L{s}"] = states.negative[s].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------- constructions
class SiteVector(BaseModel):
    """One built direction at one site, with the statistics that describe it."""

    model_config = ConfigDict(frozen=True, extra="forbid",
                              arbitrary_types_allowed=True)

    site: int
    key: str
    construction: Literal["caa", "repeng_pca"]
    vector: np.ndarray
    raw_norm: float
    n_pairs: int
    dim: int
    stats: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _is_a_unit_direction(self) -> "SiteVector":
        v = np.asarray(self.vector)
        if v.ndim != 1 or v.shape[0] != self.dim:
            raise ValueError(f"L{self.site}: vector shape {v.shape}, dim {self.dim}")
        n = float(np.linalg.norm(v.astype(np.float64)))
        if not np.isfinite(n) or abs(n - 1.0) > 1e-5:
            raise ValueError(
                f"L{self.site}: banked vectors are UNIT (the write hook re-normalizes "
                f"at attach; orientation is the object) — this one has norm {n!r}")
        return self


def project_onto_direction(H: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """repeng's own projection: `(H @ direction) / ||direction||`, lifted verbatim.

    Named as a function rather than inlined because the SIGN RULE is defined in
    terms of it, and a sign convention that lives inside a loop is a sign convention
    nobody can check.
    """
    mag = float(np.linalg.norm(direction))
    if not np.isfinite(mag) or mag <= 0.0:
        raise DegenerateDirection("projection onto a zero/non-finite direction")
    return (H @ direction) / mag


def _svd_flip_sign(component: np.ndarray) -> np.ndarray:
    """sklearn's `svd_flip` convention: largest-|value| entry made positive.

    Applied so this implementation agrees with `sklearn.decomposition.PCA` component
    for component (the selftest cross-checks it when sklearn is importable). It is
    then OVERRIDDEN by the repeng sign rule, which is the actual determinant — so
    the banked vector's orientation never depends on this convention. Both facts are
    recorded rather than trusted.
    """
    i = int(np.argmax(np.abs(component)))
    return component if component[i] >= 0 else -component


def caa_direction(states: PairedStates, site: int, *,
                  recipe: Optional[CAARecipe] = None) -> SiteVector:
    """The construction of record: unit(mean(positive) - mean(negative)) at `site`.

    Deterministic with no RNG anywhere, which is why the build-twice property here
    is a statement about linear algebra rather than about a seed.
    """
    recipe = recipe or CAARecipe()
    assert_construction_admissible(recipe.construction, key=f"L{site} ")
    if site not in states.positive:
        raise DimensionMismatch(f"no states at L{site}; have {states.sites}")
    pos = states.positive[site].astype(np.float64)
    neg = states.negative[site].astype(np.float64)
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    raw_norm = float(np.linalg.norm(diff))
    v = unit(diff, what=f"caa L{site}")

    per_pair = pos - neg
    pp_norms = np.linalg.norm(per_pair, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pp_cos = np.where(pp_norms > 0, (per_pair @ v.astype(np.float64)) / pp_norms,
                          np.nan)
    finite = pp_cos[np.isfinite(pp_cos)]
    stats = {
        "raw_norm": raw_norm,
        "mean_per_pair_diff_norm": float(pp_norms.mean()),
        "per_pair_cos_mean": float(finite.mean()) if finite.size else None,
        "per_pair_cos_min": float(finite.min()) if finite.size else None,
        "per_pair_cos_frac_positive": (float((finite > 0).mean())
                                       if finite.size else None),
        "coherence_note": (
            "per_pair_cos_* are DESCRIPTIVE build diagnostics (the V1 precedent's "
            "`coherence`), never a gate: no FD-gate analogue exists for a CAA "
            "object (BRIEF-class-probe-v1 §4)."),
    }
    return SiteVector(site=site, key=CAA_KEY_TEMPLATE.format(axis=states.axis,
                                                             site=site),
                      construction="caa", vector=v, raw_norm=raw_norm,
                      n_pairs=states.n_pairs, dim=states.dim, stats=stats)


def repeng_pca_direction(states: PairedStates, site: int, *,
                         recipe: Optional[RepengPCARecipe] = None) -> SiteVector:
    """The published repeng `pca_diff` construction, single-site application NAMED.

    Faithful to the method, step for step:

      1. interleave the pair states `[pos_0, neg_0, pos_1, neg_1, …]` — repeng's own
         layout, because every later step indexes `[::2]` / `[1::2]`;
      2. `train = h[::2] - h[1::2]` — the paired differences;
      3. one-component PCA of `train`, WHICH CENTERS `train` (sklearn does; so do
         we, explicitly, because the centering is exactly what makes this a
         different object from the CAA mean);
      4. sign: project the full interleaved stack onto the component and flip if the
         positive member projects SMALLER than its negative on more pairs than not.

    A single-component PCA is already unit-norm, so no renormalization is applied —
    `unit()` runs only as the degeneracy guard.
    """
    recipe = recipe or RepengPCARecipe()
    assert_construction_admissible(recipe.construction, key=f"L{site} ")
    if site not in states.positive:
        raise DimensionMismatch(f"no states at L{site}; have {states.sites}")
    pos = states.positive[site].astype(np.float64)
    neg = states.negative[site].astype(np.float64)
    n = states.n_pairs

    # 1. repeng's interleaved layout.
    h = np.empty((2 * n, states.dim), dtype=np.float64)
    h[0::2] = pos
    h[1::2] = neg

    # 2. the paired differences.
    train = h[0::2] - h[1::2]

    # 3. one-component PCA — centering included, and named.
    mean_ = train.mean(axis=0)
    centered = train - mean_ if recipe.center_before_pca else train
    if n < 2:
        raise PairCountError(
            "the repeng construction fits a PCA to the paired differences; with a "
            "single pair the centered matrix is identically zero and the component "
            "is undefined. n_pairs >= 2 is structural, not a preference.")
    try:
        _u, sv, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise DegenerateDirection(f"L{site}: SVD did not converge ({exc})") from exc
    component = _svd_flip_sign(vt[0])
    v64 = np.asarray(component, dtype=np.float64)
    v64 = unit(v64, what=f"repeng L{site}").astype(np.float64)

    # 4. the sign rule, on the FULL stack (repeng projects `h`, not `train`).
    projected = project_onto_direction(h, v64)
    pos_proj, neg_proj = projected[0::2], projected[1::2]
    positive_smaller = float(np.mean(pos_proj < neg_proj))
    positive_larger = float(np.mean(pos_proj > neg_proj))
    flipped = positive_smaller > positive_larger
    if flipped:
        v64 = -v64
    sign_tie = positive_smaller == positive_larger

    total_var = float((sv ** 2).sum())
    stats = {
        "explained_variance_ratio_pc1": (float(sv[0] ** 2 / total_var)
                                         if total_var > 0 else None),
        "singular_values_head": [float(x) for x in sv[:3]],
        "sign_flipped": bool(flipped),
        "sign_tie": bool(sign_tie),
        "positive_larger_fraction": positive_larger,
        "positive_smaller_fraction": positive_smaller,
        "train_mean_norm": float(np.linalg.norm(mean_)),
        "centered": bool(recipe.center_before_pca),
        "cos_to_caa_mean_direction": float(
            v64 @ (mean_ / np.linalg.norm(mean_))) if np.linalg.norm(mean_) > 0
            else None,
        "application_scope": recipe.application_scope,
        "sign_tie_note": (
            "a tie leaves the svd_flip orientation standing — repeng's own rule "
            "flips only on a strict majority, and that is preserved rather than "
            "improved on"),
    }
    return SiteVector(
        site=site, key=REPENG_KEY_TEMPLATE.format(axis=states.axis, site=site),
        construction="repeng_pca", vector=v64.astype(np.float32),
        raw_norm=float(sv[0]), n_pairs=n, dim=states.dim, stats=stats)


CONSTRUCTORS: dict[str, Callable[..., SiteVector]] = {
    "caa": caa_direction,
    "repeng_pca": repeng_pca_direction,
}


def build_site_vectors(states: PairedStates, construction: str, *,
                       sites: Optional[Sequence[int]] = None
                       ) -> dict[int, SiteVector]:
    """Build one direction per requested site. Multi-site = the CONTROL family."""
    assert_construction_admissible(construction)
    want = tuple(sites) if sites is not None else states.sites
    missing = [s for s in want if s not in states.positive]
    if missing:
        raise DimensionMismatch(
            f"asked for sites {list(want)} but the states carry {list(states.sites)} "
            f"(missing {missing})")
    fn = CONSTRUCTORS[construction]
    return {s: fn(states, s) for s in want}


def orthogonality_stats(vectors: dict[int, SiteVector]) -> dict[str, Any]:
    """DESCRIPTIVE pairwise |cos| across sites — reported, never a gate."""
    sites = sorted(vectors)
    pairs = []
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            c = float(np.asarray(vectors[a].vector, dtype=np.float64)
                      @ np.asarray(vectors[b].vector, dtype=np.float64))
            pairs.append({"sites": [a, b], "cos": c, "abs_cos": abs(c)})
    dim = vectors[sites[0]].dim if sites else 0
    return {
        "pairs": pairs,
        "max_abs_cos": max((p["abs_cos"] for p in pairs), default=None),
        "expected_abs_cos_random": (float(np.sqrt(2 / (np.pi * dim)))
                                    if dim > 0 else None),
        "note": "DESCRIPTIVE. Cross-site cosines of one construction are a depth "
                "correspondence statistic, not an acceptance criterion.",
    }


def cross_construction_cosines(a: dict[int, SiteVector],
                               b: dict[int, SiteVector]) -> dict[str, Any]:
    """cos(CAA, repeng) per site — the method row's headline descriptive number."""
    common = sorted(set(a) & set(b))
    rows = [{"site": s,
             "cos": float(np.asarray(a[s].vector, dtype=np.float64)
                          @ np.asarray(b[s].vector, dtype=np.float64))}
            for s in common]
    return {"rows": rows,
            "note": "cos(construction-of-record, method-row) per site. DESCRIPTIVE: "
                    "two constructions on ONE contrast set agreeing is evidence "
                    "about the set, not about transport."}


# ---------------------------------------------------------------- stamps + banking
class BuildResult(BaseModel):
    """What a build produced, as a typed object (never a bare dict of paths)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_id: str
    axis: str
    set_sha256: str
    construction: str
    node_key: str
    arm: str
    sites: list[int]
    keys: list[str]
    n_pairs: int
    dim: int
    wrote: list[str] = Field(default_factory=list)
    grade: str = GRADE_LINE


def build_stamp(*, cset: ContrastSet, set_sha256: str, pin_row: ContrastSetPinRow,
                states: PairedStates, vectors: dict[int, SiteVector],
                recipe: CAARecipe | RepengPCARecipe,
                brief_sha256: Optional[str] = None,
                thread_config: Optional[ThreadConfig] = None) -> dict[str, Any]:
    """The full stamp. Everything a reader needs to disbelieve the vector."""
    tc = thread_config or effective_thread_config()
    sites = sorted(vectors)
    multi = len(sites) > 1
    return {
        "schema_version": CONTRAST_VECTOR_SCHEMA_VERSION,
        "grade": GRADE_LINE,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "construction": recipe.construction,
        "construction_of_record": recipe.construction_of_record,
        "construction_text": recipe.construction_text,
        "application_scope": getattr(recipe, "application_scope", None),
        "multi_site_family": multi,
        "multi_site_scope": MULTI_SITE_SCOPE if multi else None,
        "transportable": not multi,
        "transportable_note": (
            MULTI_SITE_SCOPE if multi else
            "single-site object: transportable through a banked map like any other "
            "vector of this campaign"),
        "axis": cset.axis,
        "induce_only": cset.induce_only,
        "set": {
            "set_id": cset.set_id,
            "sha256": set_sha256,
            "n_pairs": cset.n_pairs,
            "positive_class": cset.positive_class,
            "negative_class": cset.negative_class,
            "parallel": cset.parallel,
            "selection_rule": cset.selection_rule,
            "license_note": cset.license_note,
            "sources": [s.model_dump() for s in cset.sources],
            "notes": cset.notes,
        },
        "ruling": pin_row.model_dump(),
        "extraction": {
            "node_key": states.node_key,
            "arm": states.arm,
            "dim": states.dim,
            "n_pairs": states.n_pairs,
            "sites": sites,
            "state_convention": (
                "residual stream ENTERING decoder layer `site`, per-text fp32 mean "
                "over the text's own token positions — collect_mean_states."
                "compute_means, called verbatim (not re-implemented)"),
            "native_carrier_prompt": NATIVE_CARRIER_PROMPT,
            "native_carrier_note": (
                "identical on both sides of every pair, so it cancels in the "
                "difference; recorded so the cancellation is checkable"),
            "date_string_pin": DATE_STRING_PIN,
        },
        "vectors": {
            v.key: {"site": v.site, "raw_norm": v.raw_norm, "dim": v.dim,
                    "n_pairs": v.n_pairs, "stats": v.stats}
            for v in vectors.values()
        },
        "orthogonality": orthogonality_stats(vectors),
        "thread_config": thread_config_stamp(tc),
        "ruled_omp_num_threads": RULED_OMP_NUM_THREADS,
        "brief_sha256": brief_sha256,
        "determinism": (
            "Neither construction draws a random number: CAA is a difference of "
            "means, the method row is an SVD with a deterministic sign rule. "
            "Build-twice is therefore a bitwise property of the linear algebra at a "
            "fixed thread count, and the thread count is stamped above."),
        "what_this_is_not": [
            "not scored (C§8: nothing here self-scores)",
            "not calibrated (the per-class natural-lever calibration is a separate, "
            "later act on the target's OWN vector)",
            "not transported (transport is fit_transport_maps + the readouts)",
            "no FD-gate analogue exists for a CAA object — build acceptance is stamp "
            "completeness + the pair-count assertion + the anchor cosines",
        ],
    }


def bank(result_dir: Path, *, cset: ContrastSet, set_sha256: str,
         pin_row: ContrastSetPinRow, states: PairedStates,
         vectors: dict[int, SiteVector],
         recipe: CAARecipe | RepengPCARecipe,
         write: bool = True) -> BuildResult:
    """Write `<key>.npz` + `<construction>_<axis>_stamps.json`, atomically.

    Nothing is written until every artifact has been produced, and each file is
    written to a temp name and `os.replace`d — a partial bank is worse than no bank
    (the 405B lesson, `build_formality_contrast` §atomic banking).
    """
    stamp = build_stamp(cset=cset, set_sha256=set_sha256, pin_row=pin_row,
                        states=states, vectors=vectors, recipe=recipe)
    wrote: list[str] = []
    if write:
        result_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        try:
            arrays = {v.key: np.asarray(v.vector, dtype=np.float32)
                      for v in vectors.values()}
            npz_final = result_dir / (f"{recipe.construction}_{cset.axis}_"
                                      f"{states.node_key}_{states.arm}.npz")
            # `np.savez(path)` APPENDS `.npz` when the name does not end in it, so
            # a `.tmp` suffix would silently write somewhere else and the atomic
            # rename would then fail on a file that was never created. Writing
            # through an open handle takes numpy's naming out of the loop.
            npz_tmp = npz_final.with_name(npz_final.name + ".tmp")
            with open(npz_tmp, "wb") as fh:
                np.savez(fh, **arrays)
            staged.append((npz_tmp, npz_final))

            stamp_final = npz_final.with_name(npz_final.stem + "_stamps.json")
            stamp_tmp = stamp_final.with_name(stamp_final.name + ".tmp")
            stamp_tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True,
                                            default=str))
            staged.append((stamp_tmp, stamp_final))

            for tmp, final in staged:
                os.replace(tmp, final)
                wrote.append(str(final))
        except Exception:
            for tmp, _final in staged:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:                                   # pragma: no cover
                    pass
            raise
    return BuildResult(
        set_id=cset.set_id, axis=cset.axis, set_sha256=set_sha256,
        construction=recipe.construction, node_key=states.node_key,
        arm=states.arm, sites=sorted(vectors),
        keys=[v.key for v in vectors.values()], n_pairs=states.n_pairs,
        dim=states.dim, wrote=wrote)


def build_from_disk(set_path: Path, pins_path: Optional[Path], states_path: Path, *,
                    construction: str, sites: Optional[Sequence[int]] = None,
                    out_dir: Optional[Path] = None,
                    write: bool = True) -> tuple[BuildResult, dict[int, SiteVector]]:
    """The whole build, guard first. This is the only path `--build` takes."""
    assert_construction_admissible(construction)
    cset = load_contrast_set(set_path)
    set_sha, pin_row = assert_contrast_set_pinned(set_path, pins_path,
                                                  contrast_set=cset)
    states = load_paired_states(states_path)
    assert_states_match_set(states, cset, set_sha)
    vectors = build_site_vectors(states, construction, sites=sites)
    recipe: CAARecipe | RepengPCARecipe = (
        CAARecipe() if construction == "caa" else RepengPCARecipe())
    result = bank(out_dir or states_path.parent, cset=cset, set_sha256=set_sha,
                  pin_row=pin_row, states=states, vectors=vectors, recipe=recipe,
                  write=write)
    return result, vectors


# ---------------------------------------------------------------- extraction lane
def contrast_entries(cset: ContrastSet, side: Literal["positive", "negative"],
                     *, carrier_prompt: str = NATIVE_CARRIER_PROMPT) -> list[dict]:
    """Collector-shaped entries for one side of every pair, in the set's own order.

    Shaped for `collect_mean_states.build_ids`, which reads `text_id`, `text`,
    `user_prompt` and `system_prompt`. The carrier prompt is identical on both sides
    and the system prompt is EMPTY (the ruling: text-contrast CAA needs no system
    prompt), so the native arm's template contribution is common-mode.
    """
    out = []
    for p in cset.pairs:
        text = p.positive_text if side == "positive" else p.negative_text
        out.append({"text_id": f"{cset.set_id}:{p.pair_id}:{side}",
                    "text": text, "system_prompt": "",
                    "user_prompt": carrier_prompt})
    return out


def extract_paired_states(model: Any, tok: Any, cset: ContrastSet, *,
                          sites: Sequence[int], arm: Literal["native", "raw"],
                          node_key: str, device: str,
                          set_sha256: Optional[str] = None,
                          carrier_prompt: str = NATIVE_CARRIER_PROMPT,
                          max_length: Optional[int] = None) -> PairedStates:
    """GPU lane: contrast set + model -> paired mean residual states.

    Delegates to `collect_mean_states.compute_means` VERBATIM. That is the whole
    design: the capture point, the fp32 mean, the position convention and the
    truncation deviation are the collector's, not a second implementation that might
    drift from it.
    """
    from metabasis.scripts.collect_mean_states import compute_means

    site_tuple = tuple(int(s) for s in sites)
    pos_means, _pn, _sl, pos_trunc = compute_means(
        model, tok, contrast_entries(cset, "positive", carrier_prompt=carrier_prompt),
        site_tuple, arm, DATE_STRING_PIN, device, max_length=max_length)
    neg_means, _nn, _sl2, neg_trunc = compute_means(
        model, tok, contrast_entries(cset, "negative", carrier_prompt=carrier_prompt),
        site_tuple, arm, DATE_STRING_PIN, device, max_length=max_length)
    if pos_trunc or neg_trunc:
        logger.warning("position ceiling truncated %d positive / %d negative texts",
                       len(pos_trunc), len(neg_trunc))
    dim = int(next(iter(pos_means.values())).shape[1])
    return PairedStates(
        axis=cset.axis, set_id=cset.set_id, set_sha256=set_sha256,
        node_key=node_key, arm=arm, dim=dim, n_pairs=cset.n_pairs,
        pair_ids=tuple(p.pair_id for p in cset.pairs),
        positive={s: np.asarray(v, dtype=np.float64) for s, v in pos_means.items()},
        negative={s: np.asarray(v, dtype=np.float64) for s, v in neg_means.items()})


# ---------------------------------------------------------------- example pin file
def example_pin_file() -> dict:
    """A pin-file template. Deliberately PROPOSED, so copying it builds nothing."""
    return {
        "schema_version": CONTRAST_SET_PIN_SCHEMA_VERSION,
        "note": ("Rows are RULED only by Luxia, with a date. A copied template is "
                 "PROPOSED and this module refuses to build from it — which is the "
                 "template working, not the template failing."),
        "rulings": [
            {"set_id": "<the set's own set_id>",
             "axis": "<one of " + "/".join(AXES_OF_RECORD) + ">",
             "sha256": "<the 64-hex sha256 OF THE SET FILE, full, never padded>",
             "status": "PROPOSED",
             "ruled_by": None,
             "ruled_on": None,
             "note": "drafted <date> by <who>; awaiting the set freeze"},
        ],
    }


# ---------------------------------------------------------------- CPU self-test
def _synthetic_states(n_pairs: int = 24, dim: int = 32, sites: Sequence[int] = (7, 9),
                      *, axis: str = "sentiment", seed: int = 20260805,
                      signal: float = 1.0, spread: float = 0.4) -> PairedStates:
    """Synthetic paired states with a PLANTED direction, so truth is known.

    `pos = base + signal * d + noise`, `neg = base + noise'` with a per-site `d`, so
    both constructions have a right answer to be checked against instead of only
    being checked against each other.
    """
    rng = np.random.default_rng(seed)                # M25: seeded, never hash()
    base = rng.standard_normal((n_pairs, dim))
    pos, neg = {}, {}
    for k, s in enumerate(sites):
        d = rng.standard_normal(dim)
        d /= np.linalg.norm(d)
        scale = 1.0 + 0.5 * rng.standard_normal(n_pairs)[:, None]
        pos[s] = base + signal * scale * d + spread * rng.standard_normal((n_pairs, dim))
        neg[s] = base + spread * rng.standard_normal((n_pairs, dim))
        pos[s] = pos[s] + 0.1 * k
        neg[s] = neg[s] + 0.1 * k
    return PairedStates(axis=axis, set_id="synthetic-set", set_sha256=None,
                        node_key="toy-node", arm="native", dim=dim,
                        n_pairs=n_pairs,
                        pair_ids=tuple(f"p{i:03d}" for i in range(n_pairs)),
                        positive=pos, negative=neg)


def _planted_states(direction: np.ndarray, n_pairs: int = 16, site: int = 5,
                    *, noise: float = 0.0, seed: int = 7) -> PairedStates:
    """States whose CAA answer is exactly `unit(direction)` (noise=0)."""
    rng = np.random.default_rng(seed)
    dim = direction.shape[0]
    base = rng.standard_normal((n_pairs, dim))
    pos = base + direction + noise * rng.standard_normal((n_pairs, dim))
    neg = base + noise * rng.standard_normal((n_pairs, dim))
    return PairedStates(axis="formality", set_id="planted", node_key="toy",
                        arm="raw", dim=dim, n_pairs=n_pairs,
                        pair_ids=tuple(f"q{i:03d}" for i in range(n_pairs)),
                        positive={site: pos}, negative={site: neg})


def _toy_set(axis: str = "sentiment", n_pairs: int = 24, *,
             set_id: str = "toy-set", induce_only: bool = False) -> ContrastSet:
    pairs = [ContrastPair(pair_id=f"p{i:03d}",
                          positive_text=f"a positive example number {i}",
                          negative_text=f"a negative example number {i}")
             for i in range(n_pairs)]
    pairs = [p.model_copy(update={"positive_sha256": sha256_text(p.positive_text),
                                  "negative_sha256": sha256_text(p.negative_text)})
             for p in pairs]
    return ContrastSet(
        set_id=set_id, axis=axis, positive_class="pos", negative_class="neg",
        induce_only=induce_only,
        sources=[SourcePin(dataset="toy/dataset", revision="0" * 40,
                           license="CC0-1.0", year=2019)],
        selection_rule="the first n rows in index order, for the selftest only",
        license_note="synthetic fixture, no third-party text",
        parallel=True, n_pairs=n_pairs, pairs=pairs)


def _write_set(path: Path, cset: ContrastSet) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(cset.model_dump_json()), indent=1,
                               sort_keys=True))
    return path


def _write_pins(path: Path, rows: list[dict], *,
                schema: str = CONTRAST_SET_PIN_SCHEMA_VERSION) -> Path:
    path.write_text(json.dumps({"schema_version": schema, "rulings": rows}, indent=1))
    return path


def _raises(fn: Callable[[], Any], exc: Any) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                                   # noqa: BLE001 — wrong class
        return False
    return False


def _ok(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _sklearn_available() -> tuple[bool, str]:
    try:
        import sklearn                                            # noqa: F401
        from sklearn.decomposition import PCA                     # noqa: F401
        return True, sklearn.__version__
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _torch_available() -> tuple[bool, str]:
    try:
        import torch                                              # noqa: F401
        return True, torch.__version__
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, data-independent, torch-free verification of the whole contract."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    def skip(name: str, why: str) -> None:
        """A NAMED skip (rake M44): a third state, distinct from pass and fail."""
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))
        logger.info("SKIP %s — %s", name, why)

    with tempfile.TemporaryDirectory(prefix="contrast_vectors_") as td:
        root = Path(td)

        # ---- 1. the scope guard is arithmetic, not prose ----------------------
        print("== selftest 1: the SCOPE GUARD (four axes, frozen) ==")
        check("the four ruled axes are exactly the roster",
              AXES_OF_RECORD == ("language", "sentiment", "formality", "refusal"),
              str(AXES_OF_RECORD))
        for good in AXES_OF_RECORD:
            check(f"a ruled axis passes: {good!r}", _ok(lambda a=good:
                                                        assert_axis_of_record(a)))
        for bad in ("toxicity", "sycophancy", "temperature", "mode", "truthfulness"):
            check(f"an off-slate axis is REFUSED: {bad!r}",
                  _raises(lambda b=bad: assert_axis_of_record(b), AxisNotOfRecord))
        check("a set cannot smuggle an off-slate axis through a field",
              _raises(lambda: _toy_set(axis="toxicity"), Exception))
        check("refusal is INDUCE-ONLY and a set must say so",
              _raises(lambda: _toy_set(axis="refusal", induce_only=False), Exception)
              and _ok(lambda: _toy_set(axis="refusal", induce_only=True)))

        # ---- 2. the closed construction set + §5.4's law ----------------------
        print("== selftest 2: constructions are a CLOSED set (§5.4's law) ==")
        check("both admissible constructions pass",
              all(_ok(lambda c=c: assert_construction_admissible(c))
                  for c in ADMISSIBLE_CONSTRUCTIONS))
        for bad in ("project_out entropy_gradient", "orthogonalize", "residualize",
                    "lesion", "Vrep_perp", "ablate the gradient"):
            check(f"a lesion-recipe construction is REFUSED: {bad!r}",
                  _raises(lambda b=bad: assert_construction_admissible(b, key="v"),
                          ConstructionRefused))
        check("an unknown-but-innocent construction is refused as OUT OF SET",
              _raises(lambda: assert_construction_admissible("mean_shift"),
                      ConstructionRefused))
        check("unit() refuses a degenerate direction rather than emitting NaN",
              _raises(lambda: unit(np.zeros(4)), DegenerateDirection)
              and _raises(lambda: unit(np.array([np.nan, 1.0])), DegenerateDirection))

        # ---- 3. THE PIN GUARD: six independent refusals -----------------------
        print("== selftest 3: THE PIN GUARD — six ways to be refused ==")
        cset = _toy_set()
        set_path = _write_set(root / "set.json", cset)
        set_sha = sha256_file(set_path)
        check("a set file digests to a full 64-hex sha (M40: never padded)",
              len(set_sha) == 64 and all(c in "0123456789abcdef" for c in set_sha),
              set_sha[:12] + "…")

        check("(1) NO pin file at all is a refusal — there is no default path",
              _raises(lambda: assert_contrast_set_pinned(set_path, None),
                      PinFileError))
        check("(1b) a pin path that does not exist is a refusal",
              _raises(lambda: assert_contrast_set_pinned(set_path,
                                                         root / "nope.json"),
                      PinFileError))
        bad_schema = root / "pins_badschema.json"
        _write_pins(bad_schema, [], schema="contrast-set-pins/99")
        check("(2) an unrecognized pin SCHEMA is refused by name",
              _raises(lambda: assert_contrast_set_pinned(set_path, bad_schema),
                      PinFileError))
        empty = _write_pins(root / "pins_empty.json", [])
        check("(3) a sha absent from the pins is refused (the unpinned-set case)",
              _raises(lambda: assert_contrast_set_pinned(set_path, empty),
                      ContrastSetNotPinned))
        proposed = _write_pins(root / "pins_proposed.json", [
            {"set_id": cset.set_id, "axis": cset.axis, "sha256": set_sha,
             "status": "PROPOSED", "ruled_by": None, "ruled_on": None}])
        check("(4) a PROPOSED row is refused BY STATUS — the 2026-08-05 case",
              _raises(lambda: assert_contrast_set_pinned(set_path, proposed),
                      ContrastSetNotRuled),
              "the set freeze is Luxia's word; a drafted set cannot become a vector")
        check("    …and ContrastSetNotRuled is a ContrastSetNotPinned (one catch)",
              issubclass(ContrastSetNotRuled, ContrastSetNotPinned))
        wrong_axis = _write_pins(root / "pins_axis.json", [
            {"set_id": cset.set_id, "axis": "formality", "sha256": set_sha,
             "status": "RULED", "ruled_by": "Luxia", "ruled_on": "2026-08-05"}])
        check("(5) a ruling for a DIFFERENT axis is refused",
              _raises(lambda: assert_contrast_set_pinned(set_path, wrong_axis),
                      ContrastSetNotPinned))
        no_ruler = _write_pins(root / "pins_noruler.json", [
            {"set_id": cset.set_id, "axis": cset.axis, "sha256": set_sha,
             "status": "RULED", "ruled_by": None, "ruled_on": None}])
        check("(6) RULED with no ruler and no date is refused",
              _raises(lambda: assert_contrast_set_pinned(set_path, no_ruler),
                      ContrastSetNotPinned))
        dupe = root / "pins_dupe.json"
        _write_pins(dupe, [
            {"set_id": "a", "axis": "sentiment", "sha256": set_sha,
             "status": "RULED", "ruled_by": "Luxia", "ruled_on": "2026-08-05"},
            {"set_id": "b", "axis": "sentiment", "sha256": set_sha,
             "status": "PROPOSED"}])
        check("(6b) the same sha ruled twice is refused (which row applied?)",
              _raises(lambda: assert_contrast_set_pinned(set_path, dupe),
                      PinFileError))
        short_sha = root / "pins_short.json"
        _write_pins(short_sha, [
            {"set_id": "a", "axis": "sentiment", "sha256": set_sha[:12],
             "status": "RULED", "ruled_by": "Luxia", "ruled_on": "2026-08-05"}])
        check("(6c) an ABBREVIATED sha in a pin is refused (M40)",
              _raises(lambda: assert_contrast_set_pinned(set_path, short_sha),
                      PinFileError))

        ruled = _write_pins(root / "pins_ruled.json", [
            {"set_id": cset.set_id, "axis": cset.axis, "sha256": set_sha,
             "status": "RULED", "ruled_by": "Luxia", "ruled_on": "2026-08-05",
             "note": "selftest fixture"}])
        got_sha, row = assert_contrast_set_pinned(set_path, ruled)
        check("a RULED, dated, axis-matching row PASSES the guard",
              got_sha == set_sha and row.status == "RULED"
              and row.ruled_by == "Luxia")
        # the guard is about the BYTES, not the name
        tampered = root / "set_tampered.json"
        tampered.write_text(set_path.read_text() + "\n")
        check("one appended byte moves the sha and the guard REFUSES the same set",
              sha256_file(tampered) != set_sha
              and _raises(lambda: assert_contrast_set_pinned(tampered, ruled),
                          ContrastSetNotPinned),
              "the ruling is over bytes, not over a name")
        check("the example pin file is PROPOSED, so copying it builds NOTHING",
              all(r["status"] == "PROPOSED"
                  for r in example_pin_file()["rulings"]))

        # ---- 4. the contrast-set document contract ---------------------------
        print("== selftest 4: the contrast-set document contract ==")
        check("an unrecognized set schema is refused by name",
              _raises(lambda: load_contrast_set(
                  _write_set_raw(root / "s2.json",
                                 {"schema_version": "contrast-set/99"})),
                      ContrastSetError))
        check("n_pairs that disagrees with len(pairs) is refused (the 8-vs-40 rake)",
              _raises(lambda: ContrastSet(**{**json.loads(cset.model_dump_json()),
                                             "n_pairs": 3}), Exception))
        check("duplicate pair_ids are refused",
              _raises(lambda: ContrastSet(
                  **{**json.loads(cset.model_dump_json()),
                     "pairs": [json.loads(cset.pairs[0].model_dump_json())] * 2,
                     "n_pairs": 2}), Exception))
        check("a source pinned to a BRANCH is refused (a branch is not a pin)",
              _raises(lambda: SourcePin(dataset="d", revision="main"), Exception))
        check("per-pair digests recompute and a drifted body is caught",
              len(cset.pair_manifest()) == cset.n_pairs
              and _raises(lambda: ContrastPair(
                  pair_id="x", positive_text="a", negative_text="b",
                  positive_sha256="0" * 64).verified_shas(), ContrastSetError))

        # ---- 5. CAA: the construction of record ------------------------------
        print("== selftest 5: CAA — planted truth, unit, build-twice ==")
        d = np.zeros(24)
        d[3] = 2.0
        planted = _planted_states(d, noise=0.0)
        v = caa_direction(planted, 5)
        check("CAA recovers a PLANTED direction exactly at zero noise",
              float(np.asarray(v.vector, dtype=np.float64) @ unit(d).astype(np.float64))
              > 0.999999,
              f"cos = {float(np.asarray(v.vector, np.float64) @ unit(d).astype(np.float64)):.9f}")
        check("the banked vector is UNIT and raw_norm is banked beside it",
              abs(float(np.linalg.norm(np.asarray(v.vector, np.float64))) - 1.0) < 1e-6
              and abs(v.raw_norm - 2.0) < 1e-9, f"raw_norm={v.raw_norm:.6f}")
        check("the key is descriptive: caa_<axis>_L<site>",
              v.key == "caa_formality_L5", v.key)
        st = _synthetic_states()
        a1 = build_site_vectors(st, "caa")
        a2 = build_site_vectors(st, "caa")
        check("CAA is BUILD-TWICE BITWISE (no RNG anywhere in the construction)",
              all(a1[s].vector.tobytes() == a2[s].vector.tobytes() for s in a1),
              "byte equality, not allclose")
        check("swapping the two sides negates the CAA direction exactly",
              np.allclose(
                  np.asarray(caa_direction(PairedStates(
                      **{**{k: v for k, v in st.model_dump().items()
                            if k not in ("positive", "negative")},
                         "positive": st.negative, "negative": st.positive}),
                      st.sites[0]).vector, dtype=np.float64),
                  -np.asarray(a1[st.sites[0]].vector, dtype=np.float64), atol=1e-6))
        same = PairedStates(axis="sentiment", set_id="s", node_key="n", arm="raw",
                            dim=4, n_pairs=3,
                            positive={1: np.ones((3, 4))},
                            negative={1: np.ones((3, 4))})
        check("identical sides produce a REFUSAL, never a NaN vector",
              _raises(lambda: caa_direction(same, 1), DegenerateDirection))
        check("per-pair coherence statistics are present and DESCRIPTIVE",
              a1[st.sites[0]].stats["per_pair_cos_mean"] is not None
              and "never a gate" in a1[st.sites[0]].stats["coherence_note"],
              f"per_pair_cos_mean={a1[st.sites[0]].stats['per_pair_cos_mean']:.4f}, "
              f"frac_positive="
              f"{a1[st.sites[0]].stats['per_pair_cos_frac_positive']:.3f}")

        # ---- 6. repeng: faithful to the published method ---------------------
        print("== selftest 6: the repeng method row — faithful, and DISTINCT ==")
        r1 = build_site_vectors(st, "repeng_pca")
        r2 = build_site_vectors(st, "repeng_pca")
        check("the method row is BUILD-TWICE BITWISE",
              all(r1[s].vector.tobytes() == r2[s].vector.tobytes() for s in r1))
        check("the key names the construction: repengpca_<axis>_L<site>",
              r1[st.sites[0]].key == f"repengpca_sentiment_L{st.sites[0]}")
        site0 = st.sites[0]
        # the layout claim, checked directly
        h = np.empty((2 * st.n_pairs, st.dim))
        h[0::2] = st.positive[site0]
        h[1::2] = st.negative[site0]
        train = h[0::2] - h[1::2]
        check("the interleaved layout gives train == positive - negative",
              np.allclose(train, st.positive[site0] - st.negative[site0]))
        # centering is what makes it a different object
        mean_ = train.mean(axis=0)
        _u, _s, vt_c = np.linalg.svd(train - mean_, full_matrices=False)
        _u2, _s2, vt_u = np.linalg.svd(train, full_matrices=False)
        cos_cu = abs(float(unit(vt_c[0]).astype(np.float64)
                           @ unit(vt_u[0]).astype(np.float64)))
        check("CENTERING CHANGES THE COMPONENT — so it is not optional",
              cos_cu < 0.9999,
              f"|cos(centered PC1, uncentered PC1)| = {cos_cu:.6f}; a "
              "reimplementation that skips sklearn's centering builds a different "
              "object")
        check("a recipe that turns the centering off is REFUSED by name",
              _raises(lambda: RepengPCARecipe(center_before_pca=False), Exception))
        rv = r1[site0]
        cos_to_mean = abs(float(np.asarray(rv.vector, np.float64)
                                @ unit(mean_).astype(np.float64)))
        check("the method row is a DISTINCT object from the CAA row (same set)",
              cos_to_mean < 0.999,
              f"|cos(repeng PC1, CAA mean direction)| = {cos_to_mean:.6f} — method "
              "diversity on shared contrast data is the point")
        check("cos(CAA, repeng) is reported per site as a DESCRIPTIVE row",
              len(cross_construction_cosines(a1, r1)["rows"]) == len(a1),
              ", ".join(f"L{r['site']}:{r['cos']:+.4f}"
                        for r in cross_construction_cosines(a1, r1)["rows"]))
        # the sign rule
        proj = project_onto_direction(h, np.asarray(rv.vector, dtype=np.float64))
        frac_larger = float(np.mean(proj[0::2] > proj[1::2]))
        check("the SIGN RULE holds: positives project larger on a majority of pairs",
              frac_larger >= 0.5, f"positive-larger fraction = {frac_larger:.3f}")
        swapped = PairedStates(**{**{k: v for k, v in st.model_dump().items()
                                     if k not in ("positive", "negative")},
                                  "positive": st.negative, "negative": st.positive})
        rs = repeng_pca_direction(swapped, site0)
        cos_swap = float(np.asarray(rs.vector, np.float64)
                         @ np.asarray(rv.vector, np.float64))
        check("swapping the sides flips the method row's sign (the rule bites)",
              cos_swap < -0.999, f"cos = {cos_swap:+.6f}")
        check("the vector is unit and explained-variance is reported",
              abs(float(np.linalg.norm(np.asarray(rv.vector, np.float64))) - 1.0)
              < 1e-6 and rv.stats["explained_variance_ratio_pc1"] is not None,
              f"PC1 explains {rv.stats['explained_variance_ratio_pc1']:.4f} of the "
              "paired-difference variance")
        check("n_pairs < 2 is a structural refusal for the PCA construction",
              _raises(lambda: repeng_pca_direction(
                  PairedStates(axis="sentiment", set_id="s", node_key="n",
                               arm="raw", dim=4, n_pairs=1,
                               positive={1: np.ones((1, 4))},
                               negative={1: np.zeros((1, 4))}), 1), PairCountError))
        sk_ok, sk_note = _sklearn_available()
        if not sk_ok:
            skip("sklearn cross-check of the PCA component", sk_note)
        else:                                                     # pragma: no cover
            from sklearn.decomposition import PCA
            comp = PCA(n_components=1, whiten=False).fit(train).components_[0]
            ours = _svd_flip_sign(np.linalg.svd(train - mean_,
                                                full_matrices=False)[2][0])
            check("our PCA agrees with sklearn's component (sign convention incl.)",
                  float(abs(unit(comp).astype(np.float64)
                            @ unit(ours).astype(np.float64))) > 0.999999,
                  f"sklearn {sk_note}")

        # ---- 7. states/set agreement ------------------------------------------
        print("== selftest 7: the states and the set must be one experiment ==")
        st_set = _toy_set(n_pairs=st.n_pairs, set_id=st.set_id)
        check("matching states+set pass",
              _ok(lambda: assert_states_match_set(st, st_set)))
        check("an axis disagreement is refused",
              _raises(lambda: assert_states_match_set(
                  st, _toy_set(axis="formality", n_pairs=st.n_pairs,
                               set_id=st.set_id)), ContrastSetError))
        check("a SET-ID disagreement is refused (these are not one experiment)",
              _raises(lambda: assert_states_match_set(st, _toy_set(
                  n_pairs=st.n_pairs, set_id="some-other-set")), ContrastSetError))
        check("a PAIR-COUNT disagreement is refused, never reconciled",
              _raises(lambda: assert_states_match_set(
                  st, _toy_set(n_pairs=st.n_pairs - 1, set_id=st.set_id)),
                  PairCountError))
        check("states extracted from a DIFFERENT set sha are refused",
              _raises(lambda: assert_states_match_set(
                  st.model_copy(update={"set_sha256": "a" * 64}), st_set, "b" * 64),
                  ContrastSetError))
        check("a pair_id ORDER change is refused (the pairing is the construction)",
              _raises(lambda: assert_states_match_set(
                  st.model_copy(update={"pair_ids": tuple(
                      reversed(st.pair_ids))}), st_set), ContrastSetError))
        check("a ragged / non-finite states bundle is refused at construction",
              _raises(lambda: PairedStates(
                  axis="sentiment", set_id="s", node_key="n", arm="raw", dim=4,
                  n_pairs=2, positive={1: np.full((2, 4), np.nan)},
                  negative={1: np.zeros((2, 4))}), Exception)
              and _raises(lambda: PairedStates(
                  axis="sentiment", set_id="s", node_key="n", arm="raw", dim=4,
                  n_pairs=2, positive={1: np.zeros((2, 4))},
                  negative={2: np.zeros((2, 4))}), Exception))
        check("asking for a site the states do not carry is refused",
              _raises(lambda: build_site_vectors(st, "caa", sites=[99]),
                      DimensionMismatch))

        # ---- 8. round-trip + the end-to-end build (guard first) ---------------
        print("== selftest 8: round-trip, banking, and the end-to-end build ==")
        sp = save_paired_states(root / "states.npz",
                                st.model_copy(update={"set_id": st_set.set_id,
                                                      "set_sha256": None}))
        back = load_paired_states(sp)
        check("a states bundle round-trips (identity + numbers)",
              back.axis == st.axis and back.n_pairs == st.n_pairs
              and back.sites == st.sites
              and all(np.allclose(back.positive[s], st.positive[s].astype(np.float32))
                      for s in st.sites))
        check("a states file with no `meta` is refused (numbers without identity)",
              _raises(lambda: load_paired_states(
                  _npz_no_meta(root / "bad_states.npz")), StatesError))

        set2 = _write_set(root / "set2.json", st_set)
        sha2 = sha256_file(set2)
        pins_prop2 = _write_pins(root / "pins2_proposed.json", [
            {"set_id": st_set.set_id, "axis": st_set.axis, "sha256": sha2,
             "status": "PROPOSED"}])
        check("END TO END: --build against a PROPOSED set writes NOTHING",
              _raises(lambda: build_from_disk(set2, pins_prop2, sp,
                                              construction="caa",
                                              out_dir=root / "out_never"),
                      ContrastSetNotRuled)
              and not (root / "out_never").exists(),
              "the overnight hard line, proved mechanically")
        pins_ruled2 = _write_pins(root / "pins2_ruled.json", [
            {"set_id": st_set.set_id, "axis": st_set.axis, "sha256": sha2,
             "status": "RULED", "ruled_by": "Luxia", "ruled_on": "2026-08-05"}])
        res, vecs = build_from_disk(set2, pins_ruled2, sp, construction="caa",
                                    out_dir=root / "out")
        check("END TO END: a RULED set builds and banks", len(res.wrote) == 2
              and all(Path(p).exists() for p in res.wrote),
              ", ".join(Path(p).name for p in res.wrote))
        stamp_path = next(Path(p) for p in res.wrote if p.endswith("_stamps.json"))
        stamp = json.loads(stamp_path.read_text())
        for field in ("grade", "construction_of_record", "set", "ruling",
                      "extraction", "vectors", "orthogonality", "thread_config",
                      "determinism", "transportable"):
            check(f"the stamp carries `{field}`", field in stamp)
        check("the stamp is UNSTAMPED (C§8) — this module never self-scores",
              stamp["grade"] == "UNSTAMPED (C§8)")
        check("the stamp carries the SET's full sha and the RULING that let it build",
              stamp["set"]["sha256"] == sha2
              and stamp["ruling"]["ruled_by"] == "Luxia")
        check("the stamp records the native carrier prompt (so cancellation is "
              "checkable)",
              stamp["extraction"]["native_carrier_prompt"] == NATIVE_CARRIER_PROMPT)
        check("the stamp records the ruled thread count",
              stamp["ruled_omp_num_threads"] == 8
              and os.environ.get("OMP_NUM_THREADS") == "8",
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
        res2, _ = build_from_disk(set2, pins_ruled2, sp, construction="caa",
                                  out_dir=root / "out2")
        npz1 = next(Path(p) for p in res.wrote if p.endswith(".npz"))
        npz2 = next(Path(p) for p in res2.wrote if p.endswith(".npz"))
        with np.load(npz1) as z1, np.load(npz2) as z2:
            same_bytes = (sorted(z1.files) == sorted(z2.files)
                          and all(z1[k].tobytes() == z2[k].tobytes()
                                  for k in z1.files))
        check("BUILD-TWICE on disk: two independent builds are byte-identical",
              same_bytes, "npz payloads compared byte for byte")
        check("a dry build writes nothing",
              build_from_disk(set2, pins_ruled2, sp, construction="caa",
                              out_dir=root / "out3", write=False)[0].wrote == []
              and not (root / "out3").exists())

        # ---- 9. multi-site: the CONTROL family, and its scope guard -----------
        print("== selftest 9: the multi-site family is a CONTROL, never transport ==")
        multi = build_site_vectors(st, "repeng_pca")
        stamp_multi = build_stamp(cset=st_set, set_sha256=sha2,
                                  pin_row=ContrastSetPinRow(
                                      set_id=st_set.set_id, axis=st_set.axis,
                                      sha256=sha2, status="RULED",
                                      ruled_by="Luxia", ruled_on="2026-08-05"),
                                  states=st, vectors=multi,
                                  recipe=RepengPCARecipe())
        check("a multi-site build is stamped multi_site_family=True",
              stamp_multi["multi_site_family"] is True and len(multi) > 1)
        check("…and is stamped NOT transportable, naming the future arc",
              stamp_multi["transportable"] is False
              and "future arc" in stamp_multi["multi_site_scope"])
        single = build_site_vectors(st, "repeng_pca", sites=[st.sites[0]])
        stamp_single = build_stamp(cset=st_set, set_sha256=sha2,
                                   pin_row=ContrastSetPinRow(
                                       set_id=st_set.set_id, axis=st_set.axis,
                                       sha256=sha2, status="RULED",
                                       ruled_by="Luxia", ruled_on="2026-08-05"),
                                   states=st, vectors=single,
                                   recipe=RepengPCARecipe())
        check("a single-site build IS transportable",
              stamp_single["transportable"] is True)
        check("the single-site APPLICATION is named in the stamp, with attenuation",
              "SINGLE-SITE APPLICATION" in stamp_single["application_scope"]
              and "ATTENUATION" in stamp_single["application_scope"])
        check("the CAA stamp carries no application_scope (it has no such variant)",
              build_stamp(cset=st_set, set_sha256=sha2,
                          pin_row=ContrastSetPinRow(
                              set_id=st_set.set_id, axis=st_set.axis, sha256=sha2,
                              status="RULED", ruled_by="L", ruled_on="2026-08-05"),
                          states=st, vectors=single | {},
                          recipe=CAARecipe())["application_scope"] is None)
        og = orthogonality_stats(multi)
        check("cross-site orthogonality is reported and DESCRIPTIVE",
              og["max_abs_cos"] is not None
              and "not an acceptance criterion" in og["note"],
              f"max |cos| across sites = {og['max_abs_cos']:.4f} "
              f"(E|cos| at d={st.dim} ~ {og['expected_abs_cos_random']:.4f})")

        # ---- 10. recipes are pinned like the band recipe ----------------------
        print("== selftest 10: the recipes pin to their ruled strings ==")
        check("a CAA recipe with another construction string is REFUSED",
              _raises(lambda: CAARecipe(
                  construction_of_record="something-else/2026"), Exception))
        check("a CAA recipe whose TEXT drifts from the ruled text is REFUSED",
              _raises(lambda: CAARecipe(construction_text="a difference of means"),
                      Exception))
        check("a repeng recipe with another construction string is REFUSED",
              _raises(lambda: RepengPCARecipe(
                  construction_of_record="repeng/2026"), Exception))
        check("a repeng recipe that SOFTENS the single-site naming is REFUSED",
              _raises(lambda: RepengPCARecipe(
                  application_scope="applied at one site"), Exception))

        # ---- 11. the extraction lane (named skip without torch) ---------------
        print("== selftest 11: the extraction lane ==")
        ents_p = contrast_entries(st_set, "positive")
        ents_n = contrast_entries(st_set, "negative")
        check("entries are collector-shaped (text/text_id/user_prompt/system_prompt)",
              all(set(e) == {"text_id", "text", "system_prompt", "user_prompt"}
                  for e in ents_p))
        check("the carrier prompt is IDENTICAL on both sides (so it cancels)",
              {e["user_prompt"] for e in ents_p + ents_n} == {NATIVE_CARRIER_PROMPT}
              and {e["system_prompt"] for e in ents_p + ents_n} == {""},
              "no system prompt — the text-contrast ruling")
        check("entries follow the set's own pair order",
              [e["text_id"].split(":")[1] for e in ents_p]
              == [p.pair_id for p in st_set.pairs])
        torch_ok, torch_note = _torch_available()
        if not torch_ok:
            skip("extraction-lane forward against collect_mean_states", torch_note)
        else:                                                     # pragma: no cover
            from metabasis.scripts.collect_mean_states import compute_means as _cm
            check("compute_means is importable and is what the lane calls",
                  callable(_cm), f"torch {torch_note}")

        # ---- 12. M44's configuration matrix, asserted vacuous where it is -----
        print("== selftest 12: the configuration matrix (M44) ==")
        check("every fixture lived in a TemporaryDirectory (data-tree axis vacuous)",
              str(root).startswith(tempfile.gettempdir()), str(root))
        check("this module imports no torch and no sklearn at module scope",
              "torch" not in sys.modules or torch_ok,
              "the construction lane is CPU-only by construction")

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if failures else 0


def _write_set_raw(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc))
    return path


def _npz_no_meta(path: Path) -> Path:
    np.savez(path, pos_L1=np.zeros((2, 3)), neg_L1=np.ones((2, 3)))
    return path


# ---------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build text-contrast class vectors (CAA of record + the repeng "
                    "method row). REFUSES to build from a contrast set whose sha256 "
                    "is not RULED in a pin file.")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only, data-independent, torch-free verification")
    ap.add_argument("--example-pins", action="store_true",
                    help="print a pin-file template (PROPOSED, so it builds nothing)")
    ap.add_argument("--verify-set", type=Path, default=None, metavar="SET",
                    help="load a contrast set, verify every per-pair digest, and "
                         "print its own sha256 + a census. Builds nothing.")
    ap.add_argument("--set", type=Path, default=None, help="the contrast set (JSON)")
    ap.add_argument("--pins", type=Path, default=None,
                    help="the RULED pin file — required for --build, no default")
    ap.add_argument("--states", type=Path, default=None,
                    help="paired-states npz (from the extraction lane)")
    ap.add_argument("--construction", choices=ADMISSIBLE_CONSTRUCTIONS, default="caa")
    ap.add_argument("--sites", type=str, default=None,
                    help="comma-separated sites; default = every site in the states. "
                         "More than one site makes a SOURCE-SIDE CONTROL family, "
                         "which is stamped NOT transportable.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--build", action="store_true",
                    help="build and bank (guard first)")
    ap.add_argument("--dry-run", action="store_true", help="plan without writing")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.example_pins:
        print(json.dumps(example_pin_file(), indent=1))
        return 0
    if args.verify_set is not None:
        try:
            cset = load_contrast_set(args.verify_set)
            manifest = cset.pair_manifest()
        except ContrastBuildError as exc:
            logger.error("HALT: %s", exc)
            return 2
        print(json.dumps({
            "set_id": cset.set_id, "axis": cset.axis, "n_pairs": cset.n_pairs,
            "parallel": cset.parallel, "induce_only": cset.induce_only,
            "sha256": sha256_file(args.verify_set),
            "sources": [s.model_dump() for s in cset.sources],
            "pair_digests_verified": len(manifest),
            "license_note": cset.license_note,
            "selection_rule": cset.selection_rule,
        }, indent=1))
        return 0
    if not args.build:
        ap.error("nothing to do: pass --selftest, --example-pins, --verify-set "
                 "or --build")
    for flag, val in (("--set", args.set), ("--states", args.states)):
        if val is None:
            ap.error(f"{flag} is required for --build")

    sites = ([int(s) for s in args.sites.split(",")] if args.sites else None)
    try:
        result, _vectors = build_from_disk(
            args.set, args.pins, args.states, construction=args.construction,
            sites=sites, out_dir=args.out_dir, write=not args.dry_run)
    except ContrastBuildError as exc:
        logger.error("HALT: %s", exc)
        return 2
    print(json.dumps(json.loads(result.model_dump_json()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
