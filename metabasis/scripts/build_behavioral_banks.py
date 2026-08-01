"""CPU staging for the behavioral column — transported objects, bands, cells, stamps.

`BRIEF-behavioral-phase-2026-07-29.md` (sha `475bc2a8…`) §2 names three modules; this
is the first of them: "CPU staging: transported objects, random bands, cell specs,
stamps. (Successor to `build_injection_banks.py` / `build_reverse_banks.py`, which are
the templates of record for construction and stamping.)" The engine
(`run_behavioral_cells.py`) consumes what this module emits; the scorer
(`score_behavioral_column.py`) consumes what the engine produces.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT

  * It is the CPU half: it resolves artifacts FROM DISK, verifies their shas and their
    corpus vintage, constructs the transported objects and the random bands, plans the
    51-cell column, and writes one `cells.json` + one bank stamp + one manifest.
  * It fires nothing, loads no weights, and imports no deep-learning stack. Every
    number it writes is either a sha, a dose fraction, or a vector coordinate.
  * **It never self-scores** (C§8) and everything it stamps is `UNSTAMPED (C§8)`.

BASIS-AGNOSTIC BY CONSTRUCTION — the standing constraint on this build. The corpus /
basis identity is an open Luxia ruling (v2.1 today, v3 upcoming), so **no corpus sha,
no vector path, no fit path and no state path is hardcoded here**. Everything arrives
through a `BankSpec` (a JSON file), and `corpus_sha_of_record` is a SPEC FIELD whose
only default is the campaign's current constant, overridable per build and per CLI
invocation. Re-pointing this module at another basis is editing a spec, never editing
this file. The one thing that is frozen is the SHAPE of the column (§4.1 + §5.1's cell
table, the signed ladder, n/cell) — that is the brief, not the basis.

THE BANDS THAT DO NOT EXIST YET. The census (`--census`) is the first-class product of
this module precisely because the v2.1 random bands (`Rband*` native, `gRband*`
transported) have never been built for any node: it reports, per required artifact,
present/absent + sha + how many of the 51 cells its absence kills, and it NEVER raises
on absence — "which objects are owed" is the answer a preflight exists to produce
(§4.3's from-disk re-derivation, and the certification HALT's census). `--build`
refuses on a non-READY census rather than improvising. A band that must be constructed
rather than consumed has a NAMED deterministic recipe (`RandomBandRecipe`,
sha256-seeded per member, M25-safe) that is only ever run when a spec asks for it in
writing.

CPU self-test (no weights, no GPU, no data tree, no torch):

    python -m metabasis.scripts.build_behavioral_banks --selftest

RAKE M44 — THE CONFIGURATION MATRIX IS THE MERGE BAR. Every reported count names the
configuration it was measured in (cwd, data-tree presence, venv). This module has no
torch dependency at all — by design, since §10's verification recipes are desk acts and
the desk's own repo `.venv` carries numpy/pydantic/scipy and nothing else — and its
selftest is data-independent (temp dirs and synthetic artifacts throughout, never a
cwd-relative banked path, the dcbe7d7 pattern). It therefore expects the SAME count in
all four cells of {torch, no torch} × {data tree, no data tree}, and a named skip
appears only if the live registries cannot be imported at all.

RAKE M45 — this module's selftest raises rather than `sys.exit()`s on an internal
refusal, and `main` returns an int, so an all-module sweep records a result instead of
dying at this file.

Typical use (repo root; the spec lives beside the bank it builds, off-repo):

    python -m metabasis.scripts.build_behavioral_banks --example-spec > spec.json
    python -m metabasis.scripts.build_behavioral_banks --spec spec.json --census
    python -m metabasis.scripts.build_behavioral_banks --spec spec.json --build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from metabasis.scripts.run_behavioral_cells import (
    ArtifactShaMismatch, BASELINE_DOSE, BRIEF_OF_RECORD, BRIEF_SHA256,
    BehavioralHarnessError, CELL_ID_TEMPLATE, CORPUS_SHA_V21, CellKind, CellSpec,
    CorpusVintageError, DOSE_LADDER, ENVELOPE_RULING_OF_RECORD, ExpectedNShortfall,
    GRADE_LINE, LesionRecipeViolation, MAX_NEW_TOKENS, N_PER_CELL,
    NaiveTransplantRowMissing, NativeVectorUnavailable, SCORING_DOSES,
    SEED_MATERIAL_TEMPLATE, SEED_MATERIAL_TEMPLATE_DIGEST, SamplingConfig,
    SiteNotOfRecord, apply_dose_ladder, assert_corpus_vintage, assert_naive_row_banked,
    assert_no_lesion_recipe, baseline_cell, seed_int)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_behavioral_banks")

#: The cells-document contract version. The engine refuses a document it does not
#: recognize rather than guessing at a field, so this string is load-bearing.
CELLS_SCHEMA_VERSION = "behavioral-cells/1"

#: §4.1 + §5.1's cell table, as arithmetic rather than prose. A planned full column
#: must equal this or the plan is short (§9 item 10 / M23: one short is a rake).
N_BAND_MEMBERS = 3
N_CALIBRATION_SIGNAL_CELLS = len(DOSE_LADDER)                    # 6
N_CALIBRATION_BAND_CELLS = N_BAND_MEMBERS * len(DOSE_LADDER)     # 18
N_BASELINE_CELLS = 1
N_TRANSPORTED_SIGNAL_CELLS = len(DOSE_LADDER)                    # 6
N_TRANSPORTED_BAND_CELLS = N_BAND_MEMBERS * len(DOSE_LADDER)     # 18
N_NAIVE_CELLS = len(SCORING_DOSES)                               # 2 (ruling 5)
N_FULL_COLUMN_CELLS = (N_BASELINE_CELLS + N_CALIBRATION_SIGNAL_CELLS
                       + N_CALIBRATION_BAND_CELLS + N_TRANSPORTED_SIGNAL_CELLS
                       + N_TRANSPORTED_BAND_CELLS + N_NAIVE_CELLS)          # 51

#: Ruling 5: the behavioral naive-transplant null fires at ±0.3 ONLY (~4% of a pair's
#: budget), and the object injected is the coordinate-identified RAW source vector —
#: no map. Named here so the construction is a constant, not a habit.
NAIVE_KEY_PREFIX = "naive"
NAIVE_CONSTRUCTION = (
    "zero-pad/truncate coordinate identification of the RAW source vector (NO transport "
    "map), unit-normalized; ruling 5's behavioral null asks whether the two spaces "
    "already share a frame, in the same currency as the claim")

#: §2.5/§5.1: the staged document carries dose FRACTIONS. α of record is resolved
#: IN-JOB from the measured per-token median residual norm, so any α this module writes
#: is provisional and labeled as such — rake M21b's silent-fallback finding says the
#: resolution must be explicit and never inherited.
PROVISIONAL_ALPHA_NOTE = (
    "PROVISIONAL ONLY. α of record = frac × the PER-TOKEN median residual norm MEASURED "
    "IN-JOB at the site (§2.5); the engine re-resolves every α after its in-job norm "
    "measurement and records both conventions in the stamp. Any α in this document was "
    "computed from a BANKED norm (or from the unit placeholder 1.0 when none is banked) "
    "purely so a reader can sanity-check magnitudes — it never sets a dose.")

#: The band construction of record when a band must be BUILT rather than consumed.
#: Bands are banked UNIT (the write hook re-normalizes at attach), so "matched-norm" is
#: realized at attach time by the SAME α the signal cell uses — the construction only
#: has to supply an orientation, and it supplies it from a digest so it is reproducible
#: from the recipe string alone (M25: never `hash()`, never an unseeded RNG).
RANDOM_BAND_SEED_TEMPLATE = "{basis_sha}|{owner}|L{site}|{family}|member{index:02d}"
RANDOM_BAND_CONSTRUCTION = (
    "unit(standard normal drawn from np.random.default_rng(sha256(recipe)[:8])) at the "
    "owner's hidden dimension; banked UNIT because the residual-write hook re-normalizes "
    "at attach, so matched-norm is realized by the cell's own α and only ORIENTATION is "
    "load-bearing (banked convention, build_injection_banks.py stamp)")


# ---------------------------------------------------------------- error taxonomy
class BankStagingError(BehavioralHarnessError):
    """Base class for staging-side refusals. Every one is a §9 HALT or a §5.4 law."""


class SpecError(BankStagingError):
    """The build spec is unreadable, incomplete, or internally inconsistent."""


class CensusNotReady(BankStagingError):
    """`--build` was asked for while the from-disk census still lists owed artifacts.

    The census is the product, not the obstacle: an absent object is reported as OWED
    with the count of cells it kills, and building around it would bank a column whose
    controls do not exist.
    """


class ConstructionRefused(BankStagingError):
    """§5.4's lesion-recipe law, or a construction this module will not perform.

    "The staging module asserts that no transported object's construction contains a
    subtraction of a target-side entropy-gradient component, and refuses to build one."
    """


class DimensionMismatch(BankStagingError):
    """A vector does not fit the space it is being carried into or out of."""


# ---------------------------------------------------------------- spec types
class VectorRef(BaseModel):
    """One banked vector: where it lives, which key inside it, and its gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="npz key inside `npz`")
    npz: Path
    #: expected sha of the npz. Verified when present (M4); an absent expectation is
    #: recorded as UNVERIFIED in the census rather than treated as agreement.
    sha256: Optional[str] = None
    #: `*_fd_gate.json` and `*_stamps.json` beside the npz, per the banked convention.
    fd_gate: Optional[Path] = None
    build_stamp: Optional[Path] = None
    provenance: str = ""

    @model_validator(mode="after")
    def _sha_is_a_sha(self) -> "VectorRef":
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError(f"{self.key}: sha256 must be 64 hex chars")
        return self


class RandomBandRecipe(BaseModel):
    """A deterministic, sha256-seeded random band — used only when none is banked.

    The v2.1 bands do not exist anywhere (the certification census found zero rband
    files in the node tree; the v1 anamnesis `load_axes` banks were the precedent and
    have no v2.1 equivalent). This recipe is what a band WOULD be built from, written
    so the members are re-derivable from the recipe string alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = Field(min_length=1, description="whose space the band lives in")
    site: int = Field(ge=0)
    dim: int = Field(gt=0, description="hidden dimension of `owner` at `site`")
    n_members: int = Field(gt=0, default=N_BAND_MEMBERS)
    #: the basis the band is anchored to — a corpus/manifest sha, so a band built at
    #: one basis can never be silently reused at another.
    basis_sha: str = Field(min_length=8)
    construction: str = RANDOM_BAND_CONSTRUCTION


class BandRef(BaseModel):
    """A random band: `Rband*` (native control) or `gRband*` (transported control).

    §4.1 keeps the two distinct in name and in stamp: `Rband*` asks whether the SITE
    actuates, `gRband*` asks whether the TRANSPORT carries. Conflating them is the
    fastest way to make a null uninterpretable, so the family is a typed field and the
    key prefix is derived from it rather than typed twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["Rband", "gRband"]
    #: banked members, in member order. Empty = the band is OWED (or must be built).
    members: tuple[VectorRef, ...] = ()
    n_members_required: int = N_BAND_MEMBERS
    #: present only when the spec authorizes CONSTRUCTING the band in writing.
    recipe: Optional[RandomBandRecipe] = None

    @model_validator(mode="after")
    def _band_is_resolvable_or_owed(self) -> "BandRef":
        if self.members and len(self.members) != self.n_members_required:
            raise ValueError(
                f"{self.family}: {len(self.members)} banked members, expected "
                f"{self.n_members_required} (§4.1/§5.1's 3-member band; ruling 3 froze "
                "the behavioral band at 3 transported members)")
        if self.recipe is not None and self.recipe.n_members != self.n_members_required:
            raise ValueError(
                f"{self.family}: recipe builds {self.recipe.n_members} members but the "
                f"band is {self.n_members_required}")
        return self

    @property
    def resolvable(self) -> bool:
        return bool(self.members) or self.recipe is not None


class TransportMapRef(BaseModel):
    """The banked map a transported object rides, with its family, arm and vintage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fit: Path
    family: str = Field(min_length=1, description="e.g. proc_k128 (ruling 1)")
    arm: Literal["native", "raw"]
    direction: Literal["fwd", "rev"] = "fwd"
    sha256: Optional[str] = None
    #: the corpus vintage the MAP was fit at (§2.8 requires it on every stamp).
    corpus_vintage: Optional[str] = None


class BankSpec(BaseModel):
    """Everything one (node, arm, site) column is staged from — the whole basis.

    Nothing outside this object decides what gets built. That is the basis-agnostic
    contract: a v3 column is this file plus a different spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str = Field(min_length=1, description="the TARGET node")
    arm: Literal["native", "raw"]
    site: int = Field(ge=0, description="the target's site of record")
    #: the SOURCE of the transported half (the hub, under ruling 6's slate).
    source_key: Optional[str] = None
    source_site: Optional[int] = None
    #: the pair string the naive-transplant gate is keyed by, e.g. "8b->qwen-7b".
    pair: Optional[str] = None

    corpus_manifest: Optional[Path] = None
    #: the basis of record for THIS build. Defaults to the campaign's current corpus
    #: constant and is overridable — the open v2.1/v3 ruling lives exactly here.
    corpus_sha_of_record: str = CORPUS_SHA_V21
    prompt_pool: Optional[Path] = None

    #: §4's lever: the node's OWN native entropy-gradient vector at its own site.
    native_vector: Optional[VectorRef] = None
    native_band: BandRef = BandRef(family="Rband")
    #: §5's signal: the SOURCE's entropy-gradient vector, carried through the map.
    source_vector: Optional[VectorRef] = None
    source_band: BandRef = BandRef(family="gRband")
    transport_map: Optional[TransportMapRef] = None
    naive_gate: Optional[Path] = None

    include_calibration: bool = True
    include_transported: bool = True
    include_naive: bool = True
    n_per_cell: int = Field(gt=0, default=N_PER_CELL)
    max_new_tokens: int = Field(gt=0, default=MAX_NEW_TOKENS)
    #: a label every artifact of a non-of-record run carries (the certification's
    #: `SHAPE-REHEARSAL, NOT A READ`). Empty = an ordinary column.
    label: str = ""
    out_dir: Path = Path("staging/behavioral-banks")
    #: banked norm conventions, recorded and passed through to the engine (§2.5). The
    #: engine MEASURES its own and records the delta rather than absorbing it.
    banked_per_token_median_resid_norm: Optional[float] = None
    banked_mean_state_median_resid_norm: Optional[float] = None
    banked_norm_provenance: str = ""

    @model_validator(mode="after")
    def _halves_have_what_they_need(self) -> "BankSpec":
        if self.include_transported and self.source_key is None:
            raise ValueError(
                "the transported half needs a `source_key` (§5.2's slate names the "
                "source; a transported object with no named source is unstampable)")
        if self.include_naive and not self.include_transported:
            raise ValueError(
                "ruling 5's naive null is a control FOR the transported half; staging "
                "it alone would bank a null with nothing to falsify")
        if self.n_per_cell > 0 and self.label == "" and self.n_per_cell != N_PER_CELL:
            raise ValueError(
                f"n/cell is frozen at {N_PER_CELL} (§5.1/§11); a reduced n is a "
                "rehearsal and must carry a `label` saying so")
        return self

    @property
    def transported_prefix(self) -> str:
        """The banked `g`-prefix convention for a transported object."""
        return "g"


# ---------------------------------------------------------------- census types
CensusRole = Literal["corpus", "prompt_pool", "native_vector", "native_band",
                     "source_vector", "source_band", "transport_map", "naive_gate"]


class ArtifactRow(BaseModel):
    """One from-disk row of §4.3's re-derived preflight — never trusted, always read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: CensusRole
    path: Optional[str]
    present: bool
    sha256: Optional[str] = None
    expected_sha256: Optional[str] = None
    sha_verified: Optional[bool] = None
    corpus_vintage: Optional[str] = None
    fd_gate_passed: Optional[bool] = None
    #: how many of the planned cells this row's absence kills — the ledger's own
    #: framing ("kills 42/51 cells"), computed rather than narrated.
    cells_at_risk: int = 0
    ready: bool = False
    blocking_reason: Optional[str] = None
    constructible: bool = False


class BankCensus(BaseModel):
    """The from-disk readiness table. Reports; does not raise (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str
    arm: str
    site: int
    source_key: Optional[str]
    corpus_sha_of_record: str
    planned_cells: int
    rows: tuple[ArtifactRow, ...]
    label: str = ""
    grade: str = GRADE_LINE

    @property
    def ready(self) -> bool:
        return all(r.ready for r in self.rows)

    @property
    def owed(self) -> tuple[ArtifactRow, ...]:
        return tuple(r for r in self.rows if not r.ready)

    @property
    def cells_at_risk(self) -> int:
        """Cells killed by at least one owed row, without double counting."""
        return min(self.planned_cells, sum(r.cells_at_risk for r in self.owed))

    def as_table(self) -> str:
        lines = [f"census: {self.node_key} L{self.site} ({self.arm} arm), source "
                 f"{self.source_key or '—'}, basis {self.corpus_sha_of_record[:12]}…",
                 f"  planned cells: {self.planned_cells}; "
                 f"READY={self.ready}; cells at risk: {self.cells_at_risk}"]
        for r in self.rows:
            state = "READY" if r.ready else ("CONSTRUCTIBLE" if r.constructible
                                             else "OWED")
            lines.append(
                f"  [{state:13s}] {r.name:28s} {r.path or '(no path in spec)'}"
                + (f"  sha {r.sha256[:12]}…" if r.sha256 else "")
                + (f"  kills {r.cells_at_risk} cell(s)" if not r.ready else "")
                + (f"  — {r.blocking_reason}" if r.blocking_reason else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------- cells document
class StagedCell(BaseModel):
    """One planned cell plus the per-cell custody the engine needs to stamp it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: CellSpec
    #: §2.5: provisional only. See PROVISIONAL_ALPHA_NOTE.
    provisional_alpha: Optional[float] = None

    @field_validator("spec", mode="before")
    @classmethod
    def _accept_a_cellspec_from_either_import_path(cls, v: Any) -> Any:
        """Normalize a `CellSpec` that came from the OTHER copy of the engine module.

        The engine is run as `python -m metabasis.scripts.run_behavioral_cells`, which
        makes it `__main__`, while every importer (this module, capability_battery)
        sees `metabasis.scripts.run_behavioral_cells` — two distinct classes compiled
        from one source. Pydantic rejects the foreign instance by identity, which would
        turn a correct call into an unreadable validation error inside the node job.
        The contract here is about the cell's SHAPE, so a shaped object is normalized
        through `model_dump` and revalidated; anything else falls through to pydantic's
        own message.
        """
        if isinstance(v, CellSpec) or isinstance(v, dict):
            return v
        dump = getattr(v, "model_dump", None)
        return dump() if callable(dump) else v
    transport_map: Optional[dict] = None
    naive_row: Optional[dict] = None
    vector_sha256: Optional[str] = None
    fd_gate: Optional[dict] = None
    build_stamp: Optional[dict] = None


class CellsDocument(BaseModel):
    """The staging→engine contract: what `--run` consumes.

    Written by `build_banks`, read by `run_behavioral_cells.load_cells_document`. The
    schema version is checked on load, because a silently-changed field would be a
    column built against a contract nobody agreed to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["behavioral-cells/1"] = CELLS_SCHEMA_VERSION
    node_key: str
    arm: Literal["native", "raw"]
    site: int
    source_key: Optional[str] = None
    pair: Optional[str] = None
    corpus_manifest_sha256: str
    prompt_pool_sha256: Optional[str] = None
    battery_item_set_sha256: str
    vectors_npz: Optional[str] = None
    vectors_npz_sha256: Optional[str] = None
    n_per_cell: int = N_PER_CELL
    max_new_tokens: int = MAX_NEW_TOKENS
    label: str = ""
    #: every link of the vintage chain the engine re-asserts (§9 item 2). A None is a
    #: HOLE and is reported as one — never as agreement.
    vintage_chain: dict[str, Optional[str]] = Field(default_factory=dict)
    banked_norms: dict[str, Optional[float]] = Field(default_factory=dict)
    banked_norm_provenance: str = ""
    cells: tuple[StagedCell, ...]
    bank_stamp: dict = Field(default_factory=dict)
    provisional_alpha_note: str = PROVISIONAL_ALPHA_NOTE
    grade: str = GRADE_LINE

    @model_validator(mode="after")
    def _cells_are_a_set(self) -> "CellsDocument":
        ids = [c.spec.cell_id for c in self.cells]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate cell_id(s) in the document: {dupes}")
        offsite = sorted({c.spec.site for c in self.cells} - {self.site})
        if offsite:
            raise ValueError(
                f"cells at site(s) {offsite} in a document for L{self.site} — one "
                "document is one (node, arm, site) column (§2.1's one load per node)")
        return self

    def cell_specs(self) -> list[CellSpec]:
        return [c.spec for c in self.cells]

    def by_id(self, cell_id: str) -> StagedCell:
        for c in self.cells:
            if c.spec.cell_id == cell_id:
                return c
        raise KeyError(cell_id)

    def vintage_links(self) -> tuple[Optional[str], ...]:
        return tuple(self.vintage_chain.values())


# ---------------------------------------------------------------- small utilities
def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, streamed (banks are large)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize, refusing a degenerate vector rather than emitting NaN."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n == 0.0:
        raise ConstructionRefused(
            f"degenerate vector (norm={n!r}) — a zero or non-finite direction cannot "
            "be banked; the write hook re-normalizes at attach and would divide by it")
    return a / n


def _read_json(path: Optional[Path], *, what: str) -> Optional[dict]:
    """Read a JSON sidecar, degrading to None with a WARNING rather than crashing.

    M19: a degraded read is described, never silent — and never upgraded to a pass.
    """
    if path is None or not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("%s: %s unreadable (%s) — treated as ABSENT, never as a pass",
                       what, path, exc)
        return None
    return doc if isinstance(doc, dict) else {"_document": doc}


def load_npz_vector(ref: VectorRef) -> np.ndarray:
    """Load one banked vector by key, with every failure named.

    Tolerant on nothing: a missing file, a missing key and a non-1-D array are three
    different defects and each says which it is, because "the vector is wrong" is not
    an actionable message at 3 a.m. on a node.
    """
    if not ref.npz.exists():
        raise NativeVectorUnavailable(
            f"{ref.key}: no vector npz at {ref.npz} (§9 item 3)")
    try:
        with np.load(ref.npz) as z:
            if ref.key not in z.files:
                raise NativeVectorUnavailable(
                    f"{ref.key}: key absent from {ref.npz} (keys: {sorted(z.files)}) — "
                    "a near-miss key would silently rotate every α (rake M12)")
            arr = np.asarray(z[ref.key], dtype=np.float64)
    except (OSError, ValueError, EOFError) as exc:
        raise NativeVectorUnavailable(
            f"{ref.key}: {ref.npz} is not a readable npz ({type(exc).__name__}: "
            f"{exc})") from exc
    arr = arr.reshape(-1) if arr.ndim > 1 and 1 in arr.shape else arr
    if arr.ndim != 1:
        raise DimensionMismatch(
            f"{ref.key}: expected a 1-D direction, got shape {arr.shape}")
    return arr


def verify_sha(path: Path, expected: Optional[str], *, what: str) -> tuple[str, Optional[bool]]:
    """(actual, verified?) — M4: a mismatch on a number-bearing artifact is a HALT."""
    actual = sha256_file(path)
    if expected is None:
        return actual, None
    if actual != expected:
        raise ArtifactShaMismatch(
            f"{what}: {path} sha {actual[:12]}… != expected {expected[:12]}… (§9 item "
            "1 / M4). A sha mismatch on a number-bearing artifact is a HALT and an "
            "enactor never adjudicates a live HALT.")
    return actual, True


# ---------------------------------------------------------------- constructions
#: The operations this module will perform on a vector, as a closed set. Anything
#: outside it is refused by name, which is what makes §5.4's law mechanical rather
#: than a habit: a lesion recipe cannot be spelled here at all.
Construction = Literal["transport", "coordinate_identify", "random_band", "identity"]

FORBIDDEN_CONSTRUCTIONS: tuple[str, ...] = (
    "project_out", "projection", "orthogonalize", "orthogonalise", "residualize",
    "residualise", "subtract", "lesion", "ablate", "perp",
)


def assert_construction_admissible(construction: str, *, key: str = "") -> None:
    """§5.4's lesion-recipe law at the construction site, not only in prose.

    "The staging module asserts that no transported object's construction contains a
    subtraction of a target-side entropy-gradient component, and refuses to build one."
    Two readings are enforced: the OPERATION must be one of the closed admissible set,
    and the NAME must not spell a forbidden one — the second catches a construction
    smuggled in through a spec's free text.
    """
    low = str(construction).lower()
    for bad in FORBIDDEN_CONSTRUCTIONS:
        if bad in low:
            raise LesionRecipeViolation(
                f"{key or '(object)'}: construction {construction!r} names {bad!r} — "
                "§5.4's lesion-recipe law forbids building any object in this column "
                "by projecting out the direction known to produce the behavior. "
                "REFUSED at construction, which is where the law binds.")
    if low not in ("transport", "coordinate_identify", "random_band", "identity"):
        raise ConstructionRefused(
            f"{key or '(object)'}: {construction!r} is not an admissible staging "
            f"construction (admissible: transport, coordinate_identify, random_band, "
            f"identity). This module builds a closed set of objects on purpose.")


def transport_vector(transport: Callable[[np.ndarray], np.ndarray], v: np.ndarray, *,
                     key: str = "") -> tuple[np.ndarray, float]:
    """Carry one direction across and unit-normalize: (unit vector, pre-unit norm).

    The pre-unit norm is attenuation-informative and is banked beside (the
    `transported_magnitudes_pre_unit` field of the banked stamp convention), never used
    for α: vectors are banked UNIT because the hook re-normalizes at attach, so only
    ORIENTATION is load-bearing.
    """
    assert_construction_admissible("transport", key=key)
    out = np.asarray(transport(np.asarray(v, dtype=np.float64)), dtype=np.float64)
    out = out.reshape(-1)
    n = float(np.linalg.norm(out))
    if not np.isfinite(n) or n == 0.0:
        raise ConstructionRefused(
            f"{key or '(object)'}: degenerate transported vector (norm={n!r}) — the "
            "map annihilated the direction, which is a fact about the fit and a HALT "
            "here, not a small number to normalize away")
    return unit(out), n


def coordinate_identify(v: np.ndarray, target_dim: int, *, key: str = "") -> np.ndarray:
    """Ruling 5's naive transplant: zero-pad / truncate, NO map, then unit.

    This is the null the column's own headline has to beat: if the transported object
    only works because the two spaces already share a residual coordinate frame, this
    object works too. Truncation and padding are both recorded in the stamp because
    which one happened is a fact about the pair, not an implementation detail.
    """
    assert_construction_admissible("coordinate_identify", key=key)
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if target_dim <= 0:
        raise DimensionMismatch(f"{key}: target dimension must be positive")
    if a.size == target_dim:
        return unit(a)
    if a.size > target_dim:
        return unit(a[:target_dim])
    out = np.zeros(target_dim, dtype=np.float64)
    out[:a.size] = a
    return unit(out)


def random_band_member(recipe: RandomBandRecipe, index: int) -> np.ndarray:
    """One member of a constructed random band — reproducible from the recipe string.

    M25: the seed is a sha256 digest of a NAMED material string, never `hash()` and
    never an unseeded default RNG, so the band is re-derivable years later from the
    stamp alone.
    """
    assert_construction_admissible("random_band", key=f"{recipe.owner}/member{index}")
    if not 1 <= index <= recipe.n_members:
        raise ValueError(
            f"band member index {index} out of range 1..{recipe.n_members}")
    material = RANDOM_BAND_SEED_TEMPLATE.format(
        basis_sha=recipe.basis_sha, owner=recipe.owner, site=recipe.site,
        family="band", index=index)
    rng = np.random.default_rng(seed_int(material))
    return unit(rng.standard_normal(recipe.dim))


def build_random_band(recipe: RandomBandRecipe, family: Literal["Rband", "gRband"]
                      ) -> dict[str, np.ndarray]:
    """The whole band, keyed `{family}{i}` in member order (§4.1's naming)."""
    return {f"{family}{i}": random_band_member(recipe, i)
            for i in range(1, recipe.n_members + 1)}


# ---------------------------------------------------------------- the census (§4.3)
def _vector_row(name: str, role: CensusRole, ref: Optional[VectorRef], *,
                cells_at_risk: int, corpus_sha_of_record: str,
                require_fd_gate: bool = True) -> ArtifactRow:
    """One vector's from-disk row: present? sha? FD-gated? at the basis of record?"""
    if ref is None:
        return ArtifactRow(
            name=name, role=role, path=None, present=False, cells_at_risk=cells_at_risk,
            ready=False,
            blocking_reason="not named in the spec (§9 item 3: a node's vector must be "
                            "present, FD-gated and FD-PASS at the basis of record)")
    if not ref.npz.exists():
        return ArtifactRow(
            name=name, role=role, path=str(ref.npz), present=False,
            expected_sha256=ref.sha256, cells_at_risk=cells_at_risk, ready=False,
            blocking_reason="absent from disk")
    try:
        sha, verified = verify_sha(ref.npz, ref.sha256, what=name)
    except ArtifactShaMismatch as exc:
        return ArtifactRow(
            name=name, role=role, path=str(ref.npz), present=True,
            sha256=sha256_file(ref.npz), expected_sha256=ref.sha256,
            sha_verified=False, cells_at_risk=cells_at_risk, ready=False,
            blocking_reason=str(exc))
    fd = _read_json(ref.fd_gate, what=f"{name} FD gate")
    fd_passed = None if fd is None else bool(
        fd.get("PASSES_FD_GATE", fd.get("PASSES", False)))
    stamp = _read_json(ref.build_stamp, what=f"{name} build stamp")
    vintage = None if stamp is None else stamp.get("corpus_manifest_sha256")
    reasons = []
    if require_fd_gate and fd_passed is not True:
        reasons.append("FD gate absent or not PASSED (§9 item 3)")
    if vintage is None:
        reasons.append("build stamp carries no corpus vintage (§9 item 2: a hole is a "
                       "hole, not an agreement)")
    elif vintage != corpus_sha_of_record:
        reasons.append(f"built at basis {vintage[:12]}…, not the basis of record "
                       f"{corpus_sha_of_record[:12]}… (§9 item 2)")
    return ArtifactRow(
        name=name, role=role, path=str(ref.npz), present=True, sha256=sha,
        expected_sha256=ref.sha256, sha_verified=verified, corpus_vintage=vintage,
        fd_gate_passed=fd_passed, cells_at_risk=0 if not reasons else cells_at_risk,
        ready=not reasons, blocking_reason="; ".join(reasons) or None)


def _band_row(name: str, role: CensusRole, band: BandRef, *, cells_at_risk: int,
              corpus_sha_of_record: str) -> ArtifactRow:
    """A band's row — the one the certification census had to report as ZERO EXISTING."""
    if band.members:
        rows = [_vector_row(f"{name}[{m.key}]", role, m, cells_at_risk=0,
                            corpus_sha_of_record=corpus_sha_of_record,
                            require_fd_gate=False) for m in band.members]
        blocking = [r.blocking_reason for r in rows if not r.ready]
        return ArtifactRow(
            name=name, role=role,
            path=str(band.members[0].npz), present=all(r.present for r in rows),
            sha256=rows[0].sha256, cells_at_risk=0 if not blocking else cells_at_risk,
            ready=not blocking, blocking_reason="; ".join(b for b in blocking if b) or None)
    if band.recipe is not None:
        return ArtifactRow(
            name=name, role=role, path=None, present=False, cells_at_risk=cells_at_risk,
            ready=False, constructible=True,
            blocking_reason=f"no banked members; CONSTRUCTIBLE from the named recipe "
                            f"({band.recipe.n_members} members at dim "
                            f"{band.recipe.dim}, basis "
                            f"{band.recipe.basis_sha[:12]}…) — pass --construct-bands "
                            "to build it, which is a bank-content act and not the "
                            "default")
    return ArtifactRow(
        name=name, role=role, path=None, present=False, cells_at_risk=cells_at_risk,
        ready=False,
        blocking_reason=f"no banked {band.family} members and no construction recipe. "
                        "The v2.1 random bands have never been built for any node "
                        "(the v1 load_axes banks were the precedent and have no v2.1 "
                        "equivalent) — this is the OWED object, reported, not "
                        "improvised.")


def census(spec: BankSpec) -> BankCensus:
    """§4.3's first act: RE-DERIVE readiness from disk, and report rather than raise.

    "The enactor's first act is a vector-inventory preflight that re-derives this table
    from disk rather than trusting it, because vectors are landing nightly." A census
    that stopped at the first gap would hide the rest, so every row is computed and the
    verdict is a property of the table.
    """
    planned = planned_cell_count(spec)
    rows: list[ArtifactRow] = []

    if spec.corpus_manifest is None:
        rows.append(ArtifactRow(
            name="corpus manifest", role="corpus", path=None, present=False,
            cells_at_risk=planned, ready=False,
            blocking_reason="no corpus manifest in the spec — every cell's resolution "
                            "chain starts here (§9 item 2)"))
    elif not spec.corpus_manifest.exists():
        rows.append(ArtifactRow(
            name="corpus manifest", role="corpus", path=str(spec.corpus_manifest),
            present=False, cells_at_risk=planned, ready=False,
            blocking_reason="absent from disk"))
    else:
        sha = sha256_file(spec.corpus_manifest)
        ok = sha == spec.corpus_sha_of_record
        rows.append(ArtifactRow(
            name="corpus manifest", role="corpus", path=str(spec.corpus_manifest),
            present=True, sha256=sha, expected_sha256=spec.corpus_sha_of_record,
            sha_verified=ok, corpus_vintage=sha,
            cells_at_risk=0 if ok else planned, ready=ok,
            blocking_reason=None if ok else
            f"basis mismatch: on disk {sha[:12]}…, of record "
            f"{spec.corpus_sha_of_record[:12]}… (§9 item 2)"))

    if spec.prompt_pool is not None:
        present = spec.prompt_pool.exists()
        rows.append(ArtifactRow(
            name="behavioral prompt pool", role="prompt_pool",
            path=str(spec.prompt_pool), present=present,
            sha256=sha256_file(spec.prompt_pool) if present else None,
            cells_at_risk=0 if present else planned, ready=present,
            blocking_reason=None if present else "absent from disk (ruling 10: the "
                                                 "banked stage-0 pool is the pool of "
                                                 "record and its sha rides every stamp)"))

    if spec.include_calibration:
        rows.append(_vector_row(
            "native entropy-gradient lever", "native_vector", spec.native_vector,
            cells_at_risk=N_CALIBRATION_SIGNAL_CELLS,
            corpus_sha_of_record=spec.corpus_sha_of_record))
        rows.append(_band_row(
            "native random band (Rband*)", "native_band", spec.native_band,
            cells_at_risk=N_CALIBRATION_BAND_CELLS,
            corpus_sha_of_record=spec.corpus_sha_of_record))

    if spec.include_transported:
        rows.append(_vector_row(
            "source entropy-gradient vector", "source_vector", spec.source_vector,
            cells_at_risk=N_TRANSPORTED_SIGNAL_CELLS + N_NAIVE_CELLS,
            corpus_sha_of_record=spec.corpus_sha_of_record))
        rows.append(_band_row(
            "source random band (→ gRband*)", "source_band", spec.source_band,
            cells_at_risk=N_TRANSPORTED_BAND_CELLS,
            corpus_sha_of_record=spec.corpus_sha_of_record))
        tm = spec.transport_map
        if tm is None or not tm.fit.exists():
            rows.append(ArtifactRow(
                name="transport map fit", role="transport_map",
                path=str(tm.fit) if tm else None, present=False,
                cells_at_risk=N_TRANSPORTED_SIGNAL_CELLS + N_TRANSPORTED_BAND_CELLS,
                ready=False,
                blocking_reason="absent" if tm else "not named in the spec"))
        else:
            try:
                sha, verified = verify_sha(tm.fit, tm.sha256, what="transport map fit")
                blocking = None
                if tm.corpus_vintage is None:
                    blocking = ("the map's corpus vintage is unrecorded — §2.8 requires "
                                "the corpus vintage OF THE MAP on every stamp")
                elif tm.corpus_vintage != spec.corpus_sha_of_record:
                    blocking = (f"map fit at basis {tm.corpus_vintage[:12]}…, not the "
                                f"basis of record {spec.corpus_sha_of_record[:12]}… "
                                "(§9 item 2: MIXED vintage)")
                rows.append(ArtifactRow(
                    name="transport map fit", role="transport_map", path=str(tm.fit),
                    present=True, sha256=sha, expected_sha256=tm.sha256,
                    sha_verified=verified, corpus_vintage=tm.corpus_vintage,
                    cells_at_risk=0 if not blocking else
                    N_TRANSPORTED_SIGNAL_CELLS + N_TRANSPORTED_BAND_CELLS,
                    ready=not blocking, blocking_reason=blocking))
            except ArtifactShaMismatch as exc:
                rows.append(ArtifactRow(
                    name="transport map fit", role="transport_map", path=str(tm.fit),
                    present=True, sha256=sha256_file(tm.fit), expected_sha256=tm.sha256,
                    sha_verified=False,
                    cells_at_risk=N_TRANSPORTED_SIGNAL_CELLS + N_TRANSPORTED_BAND_CELLS,
                    ready=False, blocking_reason=str(exc)))

    if spec.include_naive:
        gate = _read_json(spec.naive_gate, what="naive-transplant gate")
        row_present = gate is not None
        blocking = None
        if not row_present:
            blocking = ("no banked naive-transplant gate file (§5.3 item 1 / §9 item "
                        "7: no behavioral cell fires for a pair whose naive row is not "
                        "banked)")
        elif spec.pair:
            try:
                assert_naive_row_banked(spec.pair, gate)
            except NaiveTransplantRowMissing as exc:
                blocking = str(exc)
        else:
            blocking = ("the spec names no `pair`, so the banked gate cannot be keyed "
                        "— a naive cell with an unnamed pair is unstampable")
        rows.append(ArtifactRow(
            name="naive-transplant gate row", role="naive_gate",
            path=str(spec.naive_gate) if spec.naive_gate else None,
            present=row_present,
            sha256=(sha256_file(spec.naive_gate)
                    if spec.naive_gate and spec.naive_gate.exists() else None),
            cells_at_risk=0 if not blocking else N_NAIVE_CELLS,
            ready=not blocking, blocking_reason=blocking))

    return BankCensus(
        node_key=spec.node_key, arm=spec.arm, site=spec.site,
        source_key=spec.source_key, corpus_sha_of_record=spec.corpus_sha_of_record,
        planned_cells=planned, rows=tuple(rows), label=spec.label)


# ---------------------------------------------------------------- the plan (§4.1/§5.1)
def planned_cell_count(spec: BankSpec) -> int:
    """The cell arithmetic this spec plans — computed, never narrated."""
    n = N_BASELINE_CELLS
    if spec.include_calibration:
        n += N_CALIBRATION_SIGNAL_CELLS + N_CALIBRATION_BAND_CELLS
    if spec.include_transported:
        n += N_TRANSPORTED_SIGNAL_CELLS + N_TRANSPORTED_BAND_CELLS
    if spec.include_naive:
        n += N_NAIVE_CELLS
    return n


def _ladder_cells(vector_key: str, site: int, *, kind: CellKind,
                  band_family: Optional[Literal["Rband", "gRband"]],
                  provenance: str, npz: Optional[str], n: int,
                  norm: float) -> list[tuple[CellSpec, float]]:
    return apply_dose_ladder(
        vector_key, site, per_token_median_resid_norm=norm, kind=kind,
        band_family=band_family, vector_npz=npz, vector_provenance=provenance, n=n)


def plan_cells(spec: BankSpec, *, vector_npz: Optional[str] = None,
               transport_map_stamp: Optional[dict] = None,
               naive_row: Optional[dict] = None,
               vector_sha256: Optional[str] = None,
               provisional_norm: Optional[float] = None) -> list[StagedCell]:
    """§4.1 + §5.1's cell table for one column, in fire order.

    Order is calibration-then-transported on purpose: §4's gate is a GATE, and a node
    whose site fails it gets no transported cell at that site, so the engine runs the
    calibration block first and the desk can read a verdict from a partial column.

    Every α here is PROVISIONAL (see `PROVISIONAL_ALPHA_NOTE`): the engine measures the
    per-token median residual norm in-job and re-resolves. When no banked norm exists —
    the ordinary case for a new node — the unit placeholder 1.0 is used so the numbers
    in the document are transparently fractions rather than plausible-looking doses.
    """
    norm = provisional_norm if provisional_norm and provisional_norm > 0 else 1.0
    n = spec.n_per_cell
    cells: list[StagedCell] = [StagedCell(spec=baseline_cell(spec.site, n=n))]

    if spec.include_calibration:
        lever_key = spec.native_vector.key if spec.native_vector else "entropy_gradient"
        lever_prov = (spec.native_vector.provenance if spec.native_vector else "") or (
            f"{spec.node_key} native entropy-gradient vector at L{spec.site} "
            f"(§4's known-good lever, FD-gated, basis "
            f"{spec.corpus_sha_of_record[:12]}…)")
        assert_no_lesion_recipe(lever_prov, lever_key)
        for c, a in _ladder_cells(lever_key, spec.site, kind="calibration",
                                  band_family=None, provenance=lever_prov,
                                  npz=vector_npz, n=n, norm=norm):
            cells.append(StagedCell(spec=c, provisional_alpha=a,
                                    vector_sha256=vector_sha256))
        for i in range(1, N_BAND_MEMBERS + 1):
            key = f"Rband{i}"
            prov = (f"{spec.node_key} OWN native matched-norm random band member {key} "
                    f"at L{spec.site} — §4.1's control asks whether the SITE actuates, "
                    "never whether the transport carries")
            for c, a in _ladder_cells(key, spec.site, kind="calibration_band",
                                      band_family="Rband", provenance=prov,
                                      npz=vector_npz, n=n, norm=norm):
                cells.append(StagedCell(spec=c, provisional_alpha=a,
                                        vector_sha256=vector_sha256))

    if spec.include_transported:
        src = spec.source_key or "source"
        signal_key = f"{spec.transported_prefix}entropy_gradient"
        fam = spec.transport_map.family if spec.transport_map else "(map)"
        prov = (f"unit({fam} · {src} entropy-gradient at L"
                f"{spec.source_site if spec.source_site is not None else '?'}) → "
                f"{spec.node_key} L{spec.site}; sign-anchored BEFORE transport "
                "(standing rule); banked UNIT (the hook re-normalizes at attach)")
        assert_no_lesion_recipe(prov, signal_key)
        for c, a in _ladder_cells(signal_key, spec.site, kind="transported",
                                  band_family=None, provenance=prov, npz=vector_npz,
                                  n=n, norm=norm):
            cells.append(StagedCell(spec=c, provisional_alpha=a,
                                    transport_map=transport_map_stamp,
                                    naive_row=naive_row, vector_sha256=vector_sha256))
        for i in range(1, N_BAND_MEMBERS + 1):
            key = f"{spec.transported_prefix}Rband{i}"
            bprov = (f"{src} banked random-band member {i} through the SAME {fam} map "
                     f"→ {spec.node_key} L{spec.site} — the control travels the same "
                     "road as the signal (§5.1); ruling 3 froze the behavioral band at "
                     f"{N_BAND_MEMBERS} transported members")
            for c, a in _ladder_cells(key, spec.site, kind="transported_band",
                                      band_family="gRband", provenance=bprov,
                                      npz=vector_npz, n=n, norm=norm):
                cells.append(StagedCell(spec=c, provisional_alpha=a,
                                        transport_map=transport_map_stamp,
                                        naive_row=naive_row,
                                        vector_sha256=vector_sha256))

    if spec.include_naive:
        key = f"{NAIVE_KEY_PREFIX}_entropy_gradient"
        nprov = (f"{NAIVE_CONSTRUCTION}; {spec.source_key or 'source'} → "
                 f"{spec.node_key} L{spec.site}, pair {spec.pair or '(unnamed)'}")
        assert_no_lesion_recipe(nprov, key)
        for frac in SCORING_DOSES:
            cell = CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key=key, site=spec.site,
                                                frac=frac),
                kind="naive", vector_key=key, site=spec.site, alpha_frac=frac, n=n,
                band_family=None, vector_npz=vector_npz, vector_provenance=nprov,
                sampling=SamplingConfig(
                    name="pure-ancestral", provenance="ruling 4: the config of record "
                                                      "applies to the naive null too"))
            cells.append(StagedCell(spec=cell, provisional_alpha=frac * norm,
                                    transport_map=None, naive_row=naive_row,
                                    vector_sha256=vector_sha256))

    expected = planned_cell_count(spec)
    if len(cells) != expected:
        raise ExpectedNShortfall(
            f"{spec.node_key}: planned {len(cells)} cells, the §4.1+§5.1 table says "
            f"{expected} (§9 item 10 / M23: a count one short is a rake, not a "
            "rounding)")
    return cells


# ---------------------------------------------------------------- site cross-check
def site_cross_check(node_key: str, site: int) -> dict:
    """§2.8's `SITES`/`SITE_OF_RECORD` cross-check, from the LIVE registries.

    Reports rather than raises for an unknown node (the registries do not carry every
    key), but RAISES when they carry the node and disagree with the site being staged:
    a retired site (gemma L36, 70B L17) must never resolve by default (§9 item 4).
    """
    try:
        from metabasis.scripts.fit_transport_maps import SITES
        from metabasis.scripts.read_composed_predictions import SITE_OF_RECORD
    except ImportError as exc:                                # pragma: no cover
        return {"agrees": None, "note": f"registries unavailable ({exc}) — the site "
                                        "cross-check is OWED, never assumed"}
    grid = tuple(dict(SITES).get(node_key, ()))
    ruled = dict(SITE_OF_RECORD).get(node_key)
    out = {"SITES": list(grid) or None, "SITE_OF_RECORD": ruled, "staged_site": site}
    if ruled is not None and site != ruled:
        raise SiteNotOfRecord(
            f"{node_key}: staging cells at L{site} but SITE_OF_RECORD is L{ruled} "
            "(§9 item 4). A retired site must never resolve by default.")
    if grid and site not in grid:
        raise SiteNotOfRecord(
            f"{node_key}: L{site} is not on the fixed fit grid {grid} — that is a fiat "
            "site (§9 item 4).")
    out["agrees"] = True
    return out


# ---------------------------------------------------------------- the build
class BuildResult(BaseModel):
    """What one `--build` produced, with every artifact's sha (M4/M43)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    out_dir: str
    vectors_npz: Optional[str]
    vectors_npz_sha256: Optional[str]
    cells_json: str
    cells_json_sha256: str
    bank_stamp_json: str
    manifest_json: str
    n_cells: int
    n_vectors: int
    n_generations: int
    census_ready: bool
    label: str = ""
    grade: str = GRADE_LINE


def build_banks(spec: BankSpec, *, construct_bands: bool = False,
                write: bool = True) -> tuple[BuildResult, CellsDocument]:
    """Stage one column: verify, construct, plan, stamp, write.

    Refuses on a non-READY census (`CensusNotReady`) unless every owed row is a band
    the spec explicitly authorized CONSTRUCTING and `construct_bands` is set — which is
    the one place a bank CONTENT decision is taken, and it is taken in writing, twice
    (a recipe in the spec plus a flag on the command line).
    """
    from metabasis.scripts.capability_battery import BATTERY_ITEM_SET_SHA256

    cen = census(spec)
    unresolved = [r for r in cen.owed
                  if not (r.constructible and construct_bands)]
    if unresolved:
        raise CensusNotReady(
            f"{spec.node_key}: {len(unresolved)} owed artifact(s) — refusing to build "
            f"a column whose objects do not exist.\n{cen.as_table()}")

    cross = site_cross_check(spec.node_key, spec.site)
    vectors: dict[str, np.ndarray] = {}
    magnitudes: dict[str, float] = {}
    constructions: dict[str, str] = {}

    # --- §4's native half -----------------------------------------------------
    if spec.include_calibration:
        if spec.native_vector is None:                        # pragma: no cover
            raise SpecError("the calibration half needs a native lever")
        lever = load_npz_vector(spec.native_vector)
        vectors[spec.native_vector.key] = unit(lever).astype(np.float32)
        constructions[spec.native_vector.key] = "identity (banked lever, unit)"
        band = spec.native_band
        if band.members:
            for i, m in enumerate(band.members, start=1):
                v = load_npz_vector(m)
                if v.size != lever.size:
                    raise DimensionMismatch(
                        f"{m.key}: dim {v.size} != the node's lever dim {lever.size}")
                vectors[f"Rband{i}"] = unit(v).astype(np.float32)
                constructions[f"Rband{i}"] = f"identity (banked {m.key}, unit)"
        elif band.recipe is not None and construct_bands:
            for key, v in build_random_band(band.recipe, "Rband").items():
                if v.size != lever.size:
                    raise DimensionMismatch(
                        f"{key}: recipe dim {v.size} != lever dim {lever.size}")
                vectors[key] = v.astype(np.float32)
                constructions[key] = RANDOM_BAND_CONSTRUCTION

    # --- §5's transported half ------------------------------------------------
    tm_stamp: Optional[dict] = None
    if spec.include_transported:
        if spec.source_vector is None or spec.transport_map is None:  # pragma: no cover
            raise SpecError("the transported half needs a source vector and a map")
        from metabasis.scripts.fit_transport_maps import load_transport_map

        tmap = load_transport_map(spec.transport_map.fit)
        direction = spec.transport_map.direction

        def carry(v: np.ndarray) -> np.ndarray:
            return tmap.transport(v, direction=direction)

        src = load_npz_vector(spec.source_vector)
        gkey = f"{spec.transported_prefix}entropy_gradient"
        gv, mag = transport_vector(carry, src, key=gkey)
        vectors[gkey] = gv.astype(np.float32)
        magnitudes[gkey] = round(mag, 6)
        constructions[gkey] = (f"transport({spec.transport_map.family}, "
                               f"{direction}) then unit")
        target_dim = gv.size
        sband = spec.source_band
        if sband.members:
            for i, m in enumerate(sband.members, start=1):
                v = load_npz_vector(m)
                if v.size != src.size:
                    raise DimensionMismatch(
                        f"{m.key}: dim {v.size} != the source vector's {src.size}")
                key = f"{spec.transported_prefix}Rband{i}"
                gvi, magi = transport_vector(carry, v, key=key)
                vectors[key] = gvi.astype(np.float32)
                magnitudes[key] = round(magi, 6)
                constructions[key] = constructions[gkey]
        elif sband.recipe is not None and construct_bands:
            for i, (_, v) in enumerate(
                    sorted(build_random_band(sband.recipe, "Rband").items()), start=1):
                key = f"{spec.transported_prefix}Rband{i}"
                gvi, magi = transport_vector(carry, v, key=key)
                vectors[key] = gvi.astype(np.float32)
                magnitudes[key] = round(magi, 6)
                constructions[key] = ("random_band (source-side, constructed) then "
                                      + constructions[gkey])
        if spec.include_naive:
            nkey = f"{NAIVE_KEY_PREFIX}_entropy_gradient"
            vectors[nkey] = coordinate_identify(src, target_dim, key=nkey
                                                ).astype(np.float32)
            constructions[nkey] = NAIVE_CONSTRUCTION
        tm_stamp = {
            "fit_path": str(spec.transport_map.fit),
            "fit_sha256": sha256_file(spec.transport_map.fit),
            "family": spec.transport_map.family,
            "arm": spec.transport_map.arm,
            "direction": direction,
            "corpus_vintage": spec.transport_map.corpus_vintage,
        }

    naive_row: Optional[dict] = None
    if spec.include_naive:
        gate = _read_json(spec.naive_gate, what="naive-transplant gate")
        naive_row = assert_naive_row_banked(spec.pair or "", gate or {})

    provisional_norm = spec.banked_per_token_median_resid_norm
    out_dir = Path(spec.out_dir)
    npz_path = out_dir / f"behavioral_vectors_{spec.node_key}_L{spec.site}_{spec.arm}.npz"
    cells = plan_cells(
        spec, vector_npz=str(npz_path) if vectors else None,
        transport_map_stamp=tm_stamp, naive_row=naive_row,
        provisional_norm=provisional_norm)

    vintage_chain: dict[str, Optional[str]] = {
        "corpus_manifest": (sha256_file(spec.corpus_manifest)
                            if spec.corpus_manifest and spec.corpus_manifest.exists()
                            else None),
        "native_vector_build": _vintage_of(spec.native_vector),
        "source_vector_build": _vintage_of(spec.source_vector),
        "transport_map": spec.transport_map.corpus_vintage if spec.transport_map
        else None,
    }
    present_chain = {k: v for k, v in vintage_chain.items() if v is not None}
    if present_chain:
        assert_corpus_vintage(*present_chain.values(),
                              expected=spec.corpus_sha_of_record)

    pool_sha = (sha256_file(spec.prompt_pool)
                if spec.prompt_pool and spec.prompt_pool.exists() else None)
    stamp = bank_stamp(spec, cells=cells, vectors=vectors, magnitudes=magnitudes,
                       constructions=constructions, transport_map=tm_stamp,
                       naive_row=naive_row, site_cross_check=cross,
                       prompt_pool_sha256=pool_sha,
                       battery_item_set_sha256=BATTERY_ITEM_SET_SHA256)

    doc = CellsDocument(
        node_key=spec.node_key, arm=spec.arm, site=spec.site,
        source_key=spec.source_key, pair=spec.pair,
        corpus_manifest_sha256=(vintage_chain["corpus_manifest"]
                                or spec.corpus_sha_of_record),
        prompt_pool_sha256=pool_sha,
        battery_item_set_sha256=BATTERY_ITEM_SET_SHA256,
        vectors_npz=str(npz_path) if vectors else None,
        n_per_cell=spec.n_per_cell, max_new_tokens=spec.max_new_tokens,
        label=spec.label, vintage_chain=vintage_chain,
        banked_norms={
            "per_token_median": spec.banked_per_token_median_resid_norm,
            "mean_state_median": spec.banked_mean_state_median_resid_norm},
        banked_norm_provenance=spec.banked_norm_provenance,
        cells=tuple(cells), bank_stamp=stamp)

    if not write:
        return (BuildResult(
            out_dir=str(out_dir), vectors_npz=None, vectors_npz_sha256=None,
            cells_json="(not written)", cells_json_sha256="", bank_stamp_json="",
            manifest_json="", n_cells=len(cells), n_vectors=len(vectors),
            n_generations=len(cells) * spec.n_per_cell, census_ready=cen.ready,
            label=spec.label), doc)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_sha = None
    if vectors:
        np.savez(npz_path, **vectors)
        npz_sha = sha256_file(npz_path)
    # the npz's own sha can only exist once the npz does; the stamp and the document
    # both carry it, and they carry the SAME one because it is computed once here.
    stamp["vector_npz_sha256"] = npz_sha
    doc = doc.model_copy(update={"vectors_npz_sha256": npz_sha,
                                 "bank_stamp": stamp})
    cells_path = out_dir / f"cells_{spec.node_key}_L{spec.site}_{spec.arm}.json"
    cells_path.write_text(json.dumps(json.loads(doc.model_dump_json()), indent=1,
                                     sort_keys=True))
    stamp_path = out_dir / f"bank_stamp_{spec.node_key}_L{spec.site}_{spec.arm}.json"
    stamp_path.write_text(json.dumps(stamp, indent=1, sort_keys=True, default=str))
    manifest = {
        "grade": GRADE_LINE, "builder": "build_behavioral_banks.py",
        "brief_of_record": BRIEF_OF_RECORD, "brief_sha256": BRIEF_SHA256,
        "label": spec.label or None,
        "artifacts": {p.name: sha256_file(p)
                      for p in sorted(out_dir.iterdir()) if p.is_file()
                      and p.name != "manifest.json"},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    logger.info("staged %s L%d (%s): %d cells, %d vectors, %d generations → %s",
                spec.node_key, spec.site, spec.arm, len(cells), len(vectors),
                len(cells) * spec.n_per_cell, out_dir)
    return (BuildResult(
        out_dir=str(out_dir), vectors_npz=str(npz_path) if vectors else None,
        vectors_npz_sha256=npz_sha, cells_json=str(cells_path),
        cells_json_sha256=sha256_file(cells_path), bank_stamp_json=str(stamp_path),
        manifest_json=str(manifest_path), n_cells=len(cells), n_vectors=len(vectors),
        n_generations=len(cells) * spec.n_per_cell, census_ready=cen.ready,
        label=spec.label), doc)


def _vintage_of(ref: Optional[VectorRef]) -> Optional[str]:
    stamp = _read_json(ref.build_stamp, what="build stamp") if ref else None
    return None if stamp is None else stamp.get("corpus_manifest_sha256")


# ---------------------------------------------------------------- the bank stamp
#: The §2.8 fields this CPU module can populate. The rest are IN-JOB facts (CVD, the
#: canonical layout, the measured norms, the replay digests) and are filled by the
#: engine's `build_stamp`. Partitioning them explicitly is what lets the desk's
#: completeness checker distinguish "not staged" from "not yet run".
STAGED_STAMP_FIELDS: tuple[str, ...] = (
    "corpus_manifest_sha256", "behavioral_prompt_pool_sha256", "site",
    "site_cross_check", "arm", "dose_ladder", "seed_recipe",
    "seed_material_template_digest", "vector_npz_sha256", "vector_fd_gate",
    "vector_build_stamp", "transport_map_fit_sha256", "transport_map_family",
    "transport_map_arm", "transport_map_corpus_vintage", "naive_transplant_gate_row",
    "naive_transplant_verdict", "battery_item_set_sha256", "envelope_ruling",
    "sampling_config", "grade",
)
IN_JOB_STAMP_FIELDS: tuple[str, ...] = (
    "alpha_value", "norm_conventions", "per_cell_seed_roots", "cuda_visible_devices",
    "scheduler_card_index", "canonical_batch_layout", "model_config_sha256", "trunk",
    "replay_gate_digests", "actuation_calibration",
)


def bank_stamp(spec: BankSpec, *, cells: Sequence[StagedCell],
               vectors: dict[str, np.ndarray], magnitudes: dict[str, float],
               constructions: dict[str, str], transport_map: Optional[dict],
               naive_row: Optional[dict], site_cross_check: dict,
               prompt_pool_sha256: Optional[str],
               battery_item_set_sha256: str) -> dict:
    """The staging half of §2.8's custody, as one document.

    Everything here is a fact this module established from disk or constructed itself.
    Nothing is defaulted into existence: an unknown is written as `None` and shows up
    in the desk's OWED column (§10), which is the point.
    """
    fd = _read_json(spec.native_vector.fd_gate if spec.native_vector else None,
                    what="native FD gate")
    build = _read_json(spec.native_vector.build_stamp if spec.native_vector else None,
                       what="native build stamp")
    return {
        "grade": GRADE_LINE,
        "label": spec.label or None,
        "builder": "build_behavioral_banks.py",
        "brief_of_record": BRIEF_OF_RECORD,
        "brief_sha256": BRIEF_SHA256,
        "node_key": spec.node_key,
        "source_key": spec.source_key,
        "pair": spec.pair,
        "arm": spec.arm,
        "site": spec.site,
        "site_cross_check": site_cross_check,
        "corpus_manifest_sha256": (sha256_file(spec.corpus_manifest)
                                   if spec.corpus_manifest
                                   and spec.corpus_manifest.exists() else None),
        "corpus_sha_of_record": spec.corpus_sha_of_record,
        "basis_note": "the basis of record is a SPEC FIELD, not a module constant — "
                      "this bank is re-pointable at another corpus/basis by editing "
                      "the spec, never this module",
        "behavioral_prompt_pool_sha256": prompt_pool_sha256,
        "dose_ladder": list(DOSE_LADDER),
        "baseline_dose": BASELINE_DOSE,
        "n_per_cell": spec.n_per_cell,
        "max_new_tokens": spec.max_new_tokens,
        "n_cells": len(cells),
        "n_generations": len(cells) * spec.n_per_cell,
        "seed_recipe": SEED_MATERIAL_TEMPLATE,
        "seed_material_template_digest": SEED_MATERIAL_TEMPLATE_DIGEST,
        "vector_npz_sha256": None,          # filled by the writer once the npz exists
        "vector_keys": sorted(vectors),
        "vector_constructions": constructions,
        "transported_magnitudes_pre_unit": magnitudes or None,
        "vectors_stored": "UNIT — the residual-write hook re-normalizes at attach, so "
                          "only ORIENTATION is load-bearing; sign-anchoring happens "
                          "BEFORE transport, always (standing rule)",
        "vector_fd_gate": fd,
        "vector_build_stamp": build,
        "transport_map_fit_sha256": (transport_map or {}).get("fit_sha256"),
        "transport_map_family": (transport_map or {}).get("family"),
        "transport_map_arm": (transport_map or {}).get("arm"),
        "transport_map_corpus_vintage": (transport_map or {}).get("corpus_vintage"),
        "naive_transplant_gate_row": naive_row,
        "naive_transplant_verdict": (naive_row or {}).get("verdict"),
        "naive_construction": NAIVE_CONSTRUCTION if spec.include_naive else None,
        "battery_item_set_sha256": battery_item_set_sha256,
        "envelope_ruling": ENVELOPE_RULING_OF_RECORD,
        "sampling_config": {"of_record": "ruling 4: pure ancestral (do_sample=True, "
                                         "T=1.0, top_p=1.0, top_k=0), identical across "
                                         "all nodes"},
        "band_families": {
            "Rband": "the node's OWN native matched-norm random band — §4's control: "
                     "does the SITE actuate",
            "gRband": "the SOURCE's randoms through the SAME map — §5's control: does "
                      "the TRANSPORT carry",
            "why_distinct": "conflating them is the fastest way to make a null "
                            "uninterpretable (§4.1)",
        },
        "lesion_recipe_law": "§5.4: asserted at CONSTRUCTION — the admissible set is "
                             "closed (transport, coordinate_identify, random_band, "
                             "identity) and every forbidden name is refused",
        "provisional_alpha_note": PROVISIONAL_ALPHA_NOTE,
        "stamp_field_partition": {"staged_here": list(STAGED_STAMP_FIELDS),
                                  "filled_in_job": list(IN_JOB_STAMP_FIELDS)},
    }


# ---------------------------------------------------------------- example spec
def example_spec() -> dict:
    """A spec template with every field named and NO basis baked in.

    Emitted rather than documented so the shape cannot drift from the model, and
    deliberately full of obvious placeholders: a spec that ran as-is would be a bank
    built from nothing.
    """
    return {
        "node_key": "<TARGET-NODE-KEY>",
        "arm": "native",
        "site": 0,
        "source_key": "<SOURCE-NODE-KEY (the hub, under ruling 6)>",
        "source_site": 0,
        "pair": "<source>-><target>",
        "corpus_manifest": "<PATH>/corpus_manifest.json",
        "corpus_sha_of_record": "<THE BASIS OF RECORD — Luxia's open v2.1/v3 ruling>",
        "prompt_pool": "<PATH>/behavioral_prompt_pool.json",
        "native_vector": {
            "key": "entropy_gradient", "npz": "<PATH>/entropy_gradient_<node>_L<site>.npz",
            "sha256": None, "fd_gate": "<PATH>/..._fd_gate.json",
            "build_stamp": "<PATH>/..._stamps.json",
            "provenance": "the node's own FD-gated entropy-gradient vector"},
        "native_band": {
            "family": "Rband", "members": [],
            "recipe": {"owner": "<TARGET-NODE-KEY>", "site": 0, "dim": 0,
                       "n_members": N_BAND_MEMBERS,
                       "basis_sha": "<THE BASIS OF RECORD>"}},
        "source_vector": {
            "key": "entropy_gradient", "npz": "<PATH>/entropy_gradient_<hub>_L<site>.npz",
            "fd_gate": "<PATH>/..._fd_gate.json",
            "build_stamp": "<PATH>/..._stamps.json"},
        "source_band": {"family": "gRband", "members": []},
        "transport_map": {"fit": "<PATH>/fit_<src>__<tgt>_<arm>_proc_k128.npz",
                          "family": "proc_k128", "arm": "native", "direction": "fwd",
                          "corpus_vintage": "<THE BASIS OF RECORD>"},
        "naive_gate": "<PATH>/naive_transplant_gate.json",
        "n_per_cell": N_PER_CELL,
        "label": "",
        "out_dir": "<PATH>/staging/behavioral-banks/<node>",
        "banked_per_token_median_resid_norm": None,
        "banked_norm_provenance": "a5_vectors_stamps.json, if the node has one; a NEW "
                                  "node has none and §2.5 measures in-job",
    }


def load_spec(path: Path) -> BankSpec:
    """Read a spec, with every failure mode named rather than traced."""
    if not path.exists():
        raise SpecError(f"no build spec at {path}")
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SpecError(f"{path}: unreadable spec ({type(exc).__name__}: {exc})") from exc
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: a spec is a JSON object, got {type(doc).__name__}")
    try:
        return BankSpec(**doc)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------- CPU self-test
def _toy_vector_npz(dirpath: Path, key: str, dim: int, *, seed: int,
                    vintage: Optional[str], fd_pass: bool = True) -> VectorRef:
    """A synthetic banked vector + its two sidecars — data-independent (M44(c))."""
    rng = np.random.default_rng(seed)
    npz = dirpath / f"{key}.npz"
    np.savez(npz, **{key: rng.standard_normal(dim).astype(np.float32)})
    fd = dirpath / f"{key}_fd_gate.json"
    fd.write_text(json.dumps({"PASSES_FD_GATE": fd_pass, "median_rel_error": 0.01}))
    stamp = dirpath / f"{key}_stamps.json"
    stamp.write_text(json.dumps({"corpus_manifest_sha256": vintage,
                                 "builder": "selftest"}))
    return VectorRef(key=key, npz=npz, sha256=sha256_file(npz), fd_gate=fd,
                     build_stamp=stamp, provenance=f"selftest::{key}")


def _toy_map_npz(dirpath: Path, d_src: int, d_tgt: int, *, seed: int) -> Path:
    """A tiny real `fit_transport_maps` Procrustes bank — loaded by the real loader."""
    rng = np.random.default_rng(seed)
    k = min(d_src, d_tgt)
    va = np.linalg.qr(rng.standard_normal((d_src, k)))[0].T          # [k, d_src]
    vb = np.linalg.qr(rng.standard_normal((d_tgt, k)))[0].T          # [k, d_tgt]
    omega = np.linalg.qr(rng.standard_normal((k, k)))[0]
    path = dirpath / "fit_selftest_proc_k.npz"
    np.savez(path, kind=np.array("proc"), src_norm=np.array(1.0),
             tgt_norm=np.array(1.0), scale=np.array(1.0),
             va=va.astype(np.float32), vb=vb.astype(np.float32),
             omega=omega.astype(np.float32))
    return path


def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, data-independent, torch-free verification of the staging contract."""
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

    basis = "b" * 64          # a SYNTHETIC basis: this module hardcodes none
    other_basis = "c" * 64

    with tempfile.TemporaryDirectory(prefix="behav_banks_") as td:
        root = Path(td)
        d_src, d_tgt = 12, 10

        # ---- 1. constructions: the closed set and §5.4's law ------------------
        print("== selftest 1: constructions are a CLOSED set (§5.4's law) ==")
        check("the admissible constructions pass",
              all(_ok(lambda c=c: assert_construction_admissible(c))
                  for c in ("transport", "coordinate_identify", "random_band",
                            "identity")))
        for bad in ("project_out entropy_gradient", "orthogonalize", "residualize",
                    "lesion", "subtract entropy_gradient", "Vrep_perp"):
            check(f"a lesion-recipe construction is REFUSED at construction: {bad!r}",
                  _raises(lambda b=bad: assert_construction_admissible(b, key="v"),
                          LesionRecipeViolation))
        check("an unknown-but-innocent construction is refused as OUT OF SET",
              _raises(lambda: assert_construction_admissible("interpolate", key="v"),
                      ConstructionRefused))
        check("unit() refuses a degenerate direction rather than emitting NaN",
              _raises(lambda: unit(np.zeros(4)), ConstructionRefused)
              and _raises(lambda: unit(np.array([np.nan, 1.0])), ConstructionRefused))

        # ---- 2. the random band recipe (M25-safe, reproducible) ---------------
        print("== selftest 2: the constructed random band (M25) ==")
        recipe = RandomBandRecipe(owner="toy-node", site=7, dim=d_tgt, basis_sha=basis)
        band1 = build_random_band(recipe, "Rband")
        band2 = build_random_band(recipe, "Rband")
        check("a band is 3 members keyed Rband1..3 (§4.1's naming)",
              sorted(band1) == ["Rband1", "Rband2", "Rband3"])
        check("the band is bitwise reproducible from the recipe alone",
              all(np.array_equal(band1[k], band2[k]) for k in band1))
        check("members are distinct directions, not one vector three times",
              len({v.tobytes() for v in band1.values()}) == 3)
        check("every member is UNIT (the hook re-normalizes; orientation is the object)",
              all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-12 for v in band1.values()))
        other = build_random_band(
            recipe.model_copy(update={"basis_sha": other_basis}), "Rband")
        check("a DIFFERENT basis gives a different band (no silent cross-basis reuse)",
              not any(np.array_equal(band1[k], other[k]) for k in band1))
        check("gRband naming is the transported family, same recipe machinery",
              sorted(build_random_band(recipe, "gRband")) == ["gRband1", "gRband2",
                                                              "gRband3"])
        check("an out-of-range member index is refused",
              _raises(lambda: random_band_member(recipe, 0), ValueError)
              and _raises(lambda: random_band_member(recipe, 4), ValueError))

        # ---- 3. coordinate identification (ruling 5's naive null) -------------
        print("== selftest 3: the naive transplant's coordinate identification ==")
        v = np.arange(1.0, d_src + 1.0)
        padded = coordinate_identify(v, d_src + 4, key="naive")
        trunc = coordinate_identify(v, d_src - 4, key="naive")
        check("zero-pad keeps the leading coordinates and pads the tail with zeros",
              padded.size == d_src + 4 and np.allclose(padded[d_src:], 0.0)
              and np.allclose(padded[:d_src] / padded[0], v / v[0]))
        check("truncation keeps the leading coordinates",
              trunc.size == d_src - 4
              and np.allclose(trunc / trunc[0], v[:d_src - 4] / v[0]))
        check("the identified object is UNIT",
              abs(float(np.linalg.norm(padded)) - 1.0) < 1e-12)
        check("equal dimensions are the identity (up to unit)",
              np.allclose(coordinate_identify(v, d_src, key="n"), unit(v)))
        check("a non-positive target dimension is refused",
              _raises(lambda: coordinate_identify(v, 0), DimensionMismatch))

        # ---- 4. transport through a REAL banked map loader --------------------
        print("== selftest 4: transport rides the real fit_transport_maps loader ==")
        from metabasis.scripts.fit_transport_maps import load_transport_map
        map_path = _toy_map_npz(root, d_src, d_tgt, seed=4)
        tmap = load_transport_map(map_path)
        gv, mag = transport_vector(lambda x: tmap.transport(x), v, key="gv")
        check("a transported object is UNIT and its pre-unit magnitude is banked beside",
              abs(float(np.linalg.norm(gv)) - 1.0) < 1e-12 and mag > 0.0,
              f"pre-unit ‖g·v‖ = {mag:.6f}")
        check("transport lands in the TARGET dimension", gv.size == d_tgt)
        check("a map that annihilates the direction is a HALT, not a small number",
              _raises(lambda: transport_vector(lambda x: np.zeros(d_tgt), v, key="gv"),
                      ConstructionRefused))

        # ---- 5. the census reports rather than raises -------------------------
        print("== selftest 5: the census reports OWED artifacts (§4.3) ==")
        corpus = root / "corpus_manifest.json"
        corpus.write_text(json.dumps({"texts": []}))
        corpus_sha = sha256_file(corpus)
        native = _toy_vector_npz(root, "entropy_gradient", d_tgt, seed=5,
                                 vintage=corpus_sha)
        source = _toy_vector_npz(root, "source_entropy_gradient", d_src, seed=6,
                                 vintage=corpus_sha)
        gate = root / "naive_gate.json"
        gate.write_text(json.dumps({"rows": [{"pair": "src->tgt", "verdict": "CLEAR",
                                              "bare_cos": 0.02, "q95": 0.09}]}))
        spec = BankSpec(
            node_key="toy-node", arm="native", site=7, source_key="toy-hub",
            source_site=3, pair="src->tgt", corpus_manifest=corpus,
            corpus_sha_of_record=corpus_sha,
            native_vector=native, source_vector=source,
            transport_map=TransportMapRef(fit=map_path, family="proc_k128",
                                          arm="native", corpus_vintage=corpus_sha),
            naive_gate=gate, out_dir=root / "out")
        cen = census(spec)
        band_rows = [r for r in cen.rows if r.role in ("native_band", "source_band")]
        check("a spec with NO bands is NOT ready, and says which rows are owed",
              not cen.ready and len(cen.owed) == 2
              and all(r.role in ("native_band", "source_band") for r in cen.owed),
              f"owed: {[r.name for r in cen.owed]}")
        check("the census counts the cells an absent band kills (the ledger's framing)",
              cen.cells_at_risk == N_CALIBRATION_BAND_CELLS + N_TRANSPORTED_BAND_CELLS,
              f"{cen.cells_at_risk} of {cen.planned_cells} cells at risk")
        check("the missing v2.1 bands are named as the OWED object, not improvised",
              all("no banked" in (r.blocking_reason or "") for r in band_rows))
        check("the census never raises on absence — it returns a table",
              isinstance(cen.as_table(), str) and "OWED" in cen.as_table())
        check("a full column plans exactly 51 cells (§4.1 24+1 + §5.1 24 + 2)",
              planned_cell_count(spec) == N_FULL_COLUMN_CELLS == 51,
              f"{planned_cell_count(spec)} cells")
        # a band the spec authorizes CONSTRUCTING is a third state
        spec_c = spec.model_copy(update={
            "native_band": BandRef(family="Rband", recipe=RandomBandRecipe(
                owner="toy-node", site=7, dim=d_tgt, basis_sha=corpus_sha)),
            "source_band": BandRef(family="gRband", recipe=RandomBandRecipe(
                owner="toy-hub", site=3, dim=d_src, basis_sha=corpus_sha))})
        cen_c = census(spec_c)
        check("a CONSTRUCTIBLE band is a third census state, still not READY",
              not cen_c.ready and all(r.constructible for r in cen_c.owed),
              "CONSTRUCTIBLE")
        # sha and vintage failures
        bad_sha = spec.model_copy(update={
            "native_vector": native.model_copy(update={"sha256": "0" * 64})})
        check("a sha mismatch is reported as a blocking row (M4), never skipped",
              not census(bad_sha).ready
              and any("sha" in (r.blocking_reason or "").lower()
                      for r in census(bad_sha).owed))
        bad_basis = spec.model_copy(update={"corpus_sha_of_record": other_basis})
        check("a basis mismatch blocks EVERY cell (§9 item 2)",
              any(r.role == "corpus" and not r.ready for r in census(bad_basis).rows))
        (root / "nofd").mkdir(exist_ok=True)
        no_fd = _toy_vector_npz(root / "nofd", "entropy_gradient", d_tgt, seed=7,
                                vintage=corpus_sha, fd_pass=False)
        cen_nofd = census(spec.model_copy(update={"native_vector": no_fd}))
        check("an FD-FAIL vector is blocking (§9 item 3)",
              not cen_nofd.ready
              and any("FD gate" in (r.blocking_reason or "") for r in cen_nofd.owed))
        (root / "novintage").mkdir(exist_ok=True)
        wrong_basis = _toy_vector_npz(root / "novintage", "entropy_gradient", d_tgt,
                                      seed=8, vintage=other_basis)
        cen_basis = census(spec.model_copy(update={"native_vector": wrong_basis}))
        check("a vector built at ANOTHER basis is blocking (§9 item 2)",
              not cen_basis.ready
              and any("basis" in (r.blocking_reason or "") for r in cen_basis.owed),
              "the v2.1/v3 boundary is exactly this row")
        check("an unbanked naive row blocks the naive cells (§5.3 item 1 / §9 item 7)",
              any(r.role == "naive_gate" and not r.ready
                  for r in census(spec.model_copy(
                      update={"pair": "src->never-banked"})).rows))

        # ---- 6. the plan (§4.1 + §5.1's table) --------------------------------
        print("== selftest 6: the cell plan is the brief's own arithmetic ==")
        cells = plan_cells(spec)
        kinds = [c.spec.kind for c in cells]
        check("the plan is 51 cells", len(cells) == 51, str(len(cells)))
        check("1 baseline + 6 calibration + 18 Rband + 6 transported + 18 gRband + 2 naive",
              (kinds.count("baseline"), kinds.count("calibration"),
               kinds.count("calibration_band"), kinds.count("transported"),
               kinds.count("transported_band"), kinds.count("naive"))
              == (1, 6, 18, 6, 18, 2),
              str({k: kinds.count(k) for k in dict.fromkeys(kinds)}))
        check("calibration cells come BEFORE transported cells (§4 is a GATE)",
              max(i for i, k in enumerate(kinds)
                  if k in ("calibration", "calibration_band"))
              < min(i for i, k in enumerate(kinds)
                    if k in ("transported", "transported_band")))
        check("the native band is `Rband*` and the transported band is `gRband*`",
              {c.spec.band_family for c in cells if c.spec.kind == "calibration_band"}
              == {"Rband"}
              and {c.spec.band_family for c in cells
                   if c.spec.kind == "transported_band"} == {"gRband"})
        check("every non-baseline cell sits on the FROZEN signed ladder",
              {c.spec.alpha_frac for c in cells if not c.spec.is_baseline}
              == set(DOSE_LADDER))
        check("the naive null fires at ±0.3 ONLY (ruling 5)",
              sorted(c.spec.alpha_frac for c in cells if c.spec.kind == "naive")
              == sorted(SCORING_DOSES))
        check("cell ids use the banked a{frac:+.2f} formatting (§2.3)",
              all(c.spec.cell_id.endswith(f"a{c.spec.alpha_frac:+.2f}")
                  for c in cells if not c.spec.is_baseline))
        check("α in the document is PROVISIONAL and defaults to the unit placeholder",
              all(c.provisional_alpha == c.spec.alpha_frac for c in cells
                  if c.provisional_alpha is not None),
              "unit placeholder 1.0 — the engine measures in-job (§2.5)")
        with_norm = plan_cells(spec, provisional_norm=12.2391)
        check("a banked norm gives provisional α = frac × norm, still provisional",
              all(abs(c.provisional_alpha - c.spec.alpha_frac * 12.2391) < 1e-12
                  for c in with_norm if c.provisional_alpha is not None)
              and "never sets a dose" in PROVISIONAL_ALPHA_NOTE)
        check("the calibration-only plan is 25 cells and names no transport",
              len(plan_cells(spec.model_copy(update={
                  "include_transported": False, "include_naive": False}))) == 25)
        # M19(c) / M23: the shortfall guard is EXERCISED, not merely written. The
        # table's arity is injected out from under the planner, which is the only way
        # a plan can disagree with its own arithmetic.
        check("a plan short of the table's arity is an ExpectedNShortfall (M23)",
              _raises(_shortfall_probe, ExpectedNShortfall),
              "table arity injected; the planner refuses rather than rounds")

        # ---- 7. the cells document contract -----------------------------------
        print("== selftest 7: the staging→engine document contract ==")
        res, doc = build_banks(spec_c, construct_bands=True, write=True)
        check("build refuses while a band is merely OWED (not constructible)",
              _raises(lambda: build_banks(spec, construct_bands=True, write=False),
                      CensusNotReady))
        check("build refuses a CONSTRUCTIBLE band without the explicit flag",
              _raises(lambda: build_banks(spec_c, construct_bands=False, write=False),
                      CensusNotReady))
        check("a built column writes cells + stamp + npz + manifest, all sha'd",
              all(Path(p).exists() for p in (res.cells_json, res.bank_stamp_json,
                                             res.manifest_json, res.vectors_npz or ""))
              and len(res.cells_json_sha256) == 64,
              f"{res.n_cells} cells, {res.n_vectors} vectors, "
              f"{res.n_generations} generations")
        check("the document round-trips through JSON with its schema version",
              load_cells_document(Path(res.cells_json)).schema_version
              == CELLS_SCHEMA_VERSION)
        loaded = load_cells_document(Path(res.cells_json))
        check("the document carries 51 cells and one site",
              len(loaded.cells) == 51 and {c.spec.site for c in loaded.cells} == {7})
        check("the vintage chain is explicit, with holes visible as None",
              set(loaded.vintage_chain) == {"corpus_manifest", "native_vector_build",
                                            "source_vector_build", "transport_map"})
        check("the banked npz holds every planned vector key",
              _npz_keys(Path(res.vectors_npz)) ==
              {"entropy_gradient", "Rband1", "Rband2", "Rband3", "gentropy_gradient",
               "gRband1", "gRband2", "gRband3", "naive_entropy_gradient"},
              str(sorted(_npz_keys(Path(res.vectors_npz)))))
        check("a duplicate cell_id is refused by the document itself",
              _raises(lambda: CellsDocument(
                  node_key="n", arm="native", site=7,
                  corpus_manifest_sha256=corpus_sha, battery_item_set_sha256="x",
                  cells=(doc.cells[1], doc.cells[1])), ValueError))
        check("a document mixing sites is refused (one document = one column)",
              _raises(lambda: CellsDocument(
                  node_key="n", arm="native", site=8,
                  corpus_manifest_sha256=corpus_sha, battery_item_set_sha256="x",
                  cells=(doc.cells[0],)), ValueError))
        check("an unknown schema version is refused on load",
              _raises(lambda: load_cells_document(
                  _write(root / "bad_schema.json",
                         {**json.loads(Path(res.cells_json).read_text()),
                          "schema_version": "behavioral-cells/999"})), SpecError))

        # ---- 8. the stamp ------------------------------------------------------
        print("== selftest 8: the bank stamp's staging half of §2.8 ==")
        stamp = loaded.bank_stamp
        check("the stamp carries the grade line UNSTAMPED (C§8)",
              stamp["grade"] == GRADE_LINE)
        check("the stamp names the basis as a SPEC field, not a module constant",
              stamp["corpus_sha_of_record"] == corpus_sha
              and corpus_sha != CORPUS_SHA_V21
              and "re-pointable" in stamp["basis_note"],
              "the module built a column at a SYNTHETIC basis — no v2.1 anywhere")
        check("the stamp carries the seed recipe and its template digest",
              stamp["seed_recipe"] == SEED_MATERIAL_TEMPLATE
              and stamp["seed_material_template_digest"] == SEED_MATERIAL_TEMPLATE_DIGEST)
        check("the stamp carries the envelope ruling verbatim (ruling 3)",
              "3 transported members" in stamp["envelope_ruling"])
        check("the stamp distinguishes the two band families in words",
              "does the SITE actuate" in stamp["band_families"]["Rband"]
              and "does the TRANSPORT carry" in stamp["band_families"]["gRband"])
        check("the stamp records EVERY vector's construction",
              set(stamp["vector_constructions"]) == _npz_keys(Path(res.vectors_npz)))
        check("the stamp records the transported pre-unit magnitudes beside",
              set(stamp["transported_magnitudes_pre_unit"])
              >= {"gentropy_gradient", "gRband1"})
        check("the stamp carries the FD gate and the build stamp it read from disk",
              stamp["vector_fd_gate"]["PASSES_FD_GATE"] is True
              and stamp["vector_build_stamp"]["corpus_manifest_sha256"] == corpus_sha)
        check("the stamp carries the naive gate row and its verdict",
              stamp["naive_transplant_verdict"] == "CLEAR")
        check("the stamp carries the battery item-set sha (§2.8)",
              len(stamp["battery_item_set_sha256"]) == 64)
        staged, in_job = set(STAGED_STAMP_FIELDS), set(IN_JOB_STAMP_FIELDS)
        from metabasis.scripts.run_behavioral_cells import STAMP_REQUIRED_FIELDS
        check("staged ∪ in-job == §2.8's 31 required fields, and they are DISJOINT",
              staged | in_job == set(STAMP_REQUIRED_FIELDS) and not (staged & in_job),
              f"{len(staged)} staged + {len(in_job)} in-job = "
              f"{len(STAMP_REQUIRED_FIELDS)}")
        check("every staged §2.8 field is actually present in the stamp",
              all(f in stamp for f in STAGED_STAMP_FIELDS),
              str(sorted(f for f in STAGED_STAMP_FIELDS if f not in stamp)))

        # ---- 9. refusals that protect the column -------------------------------
        print("== selftest 9: staging-side HALTs ==")
        check("a spec whose transported half has no source is refused",
              _raises(lambda: BankSpec(node_key="n", arm="native", site=1), ValueError))
        check("a reduced n/cell without a label is refused (§5.1's frozen n)",
              _raises(lambda: BankSpec(node_key="n", arm="native", site=1,
                                       include_transported=False, include_naive=False,
                                       n_per_cell=8), ValueError))
        check("a reduced n/cell WITH a rehearsal label is allowed",
              BankSpec(node_key="n", arm="native", site=1, include_transported=False,
                       include_naive=False, n_per_cell=8,
                       label="SHAPE-REHEARSAL, NOT A READ").n_per_cell == 8)
        check("a 2-member band is refused (ruling 3 froze the band at 3)",
              _raises(lambda: BandRef(
                  family="gRband",
                  members=(source, source)), ValueError))
        check("a lesion-recipe provenance is refused while PLANNING, too",
              _raises(lambda: plan_cells(spec.model_copy(update={
                  "native_vector": native.model_copy(update={
                      "provenance": "unit(Vrep entropy-orthogonalized against "
                                    "entropy_gradient_L7)"})})),
                  LesionRecipeViolation))
        check("a missing npz key is a named refusal, not a traceback",
              _raises(lambda: load_npz_vector(native.model_copy(
                  update={"key": "nope"})), NativeVectorUnavailable))
        check("a missing spec file is a named refusal",
              _raises(lambda: load_spec(root / "nope.json"), SpecError))
        check("a malformed spec is a named refusal",
              _raises(lambda: load_spec(_write(root / "bad.json", {"node_key": 1})),
                      SpecError))
        check("the example spec round-trips into a BankSpec shape check",
              isinstance(example_spec(), dict)
              and example_spec()["corpus_sha_of_record"].startswith("<"),
              "the template bakes in NO basis")
        check("a MIXED vintage chain is a HALT, and a HOLE is reported as a hole",
              _raises(lambda: assert_corpus_vintage(corpus_sha, other_basis,
                                                    expected=corpus_sha),
                      CorpusVintageError)
              and _raises(lambda: assert_corpus_vintage(corpus_sha, None,
                                                        expected=corpus_sha),
                          CorpusVintageError),
              "§9 item 2, asserted against the SPEC's basis — not against v2.1")

        # ---- 10. the site cross-check against the LIVE registries --------------
        print("== selftest 10: site cross-check reads the live registries ==")
        try:
            from metabasis.scripts.fit_transport_maps import SITES as _SITES
            from metabasis.scripts.read_composed_predictions import (
                SITE_OF_RECORD as _SOR)
            registries_ok = True
        except ImportError as exc:                            # pragma: no cover
            registries_ok = False
            skip("site cross-check against the live registries",
                 f"registries unimportable ({exc})")
        if registries_ok:
            known = next((k for k in _SOR if k in _SITES), None)
            check("an unknown node cross-checks to a reported (not assumed) state",
                  site_cross_check("no-such-node", 3)["agrees"] is True)
            if known is None:                                 # pragma: no cover
                skip("cross-check on a registered node", "no node in BOTH registries")
            else:
                check(f"a registered node agrees with its ruled site ({known})",
                      site_cross_check(known, _SOR[known])["agrees"] is True,
                      f"L{_SOR[known]}")
                check("a site that is NOT the ruled site is a HALT (§9 item 4)",
                      _raises(lambda: site_cross_check(known, _SOR[known] + 1),
                              SiteNotOfRecord),
                      "a retired site must never resolve by default")

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    # RAKE M44: the tail states coverage as well as the verdict.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if failures else 0


def _shortfall_probe() -> None:
    """Exercise §9 item 10's guard by injecting a table arity the planner cannot meet.

    The guard can only fire when the plan and the table disagree, and they cannot
    disagree by accident — so the arity is injected and restored. Test scaffolding, run
    from the selftest only.
    """
    global N_CALIBRATION_BAND_CELLS
    original = N_CALIBRATION_BAND_CELLS
    try:
        N_CALIBRATION_BAND_CELLS = original + 1
        plan_cells(BankSpec(node_key="probe", arm="native", site=1,
                            include_transported=False, include_naive=False))
    finally:
        N_CALIBRATION_BAND_CELLS = original


def _npz_keys(path: Path) -> set:
    with np.load(path) as z:
        return set(z.files)


def _write(path: Path, doc: Any) -> Path:
    path.write_text(json.dumps(doc, default=str))
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


# ---------------------------------------------------------------- document loading
def load_cells_document(path: Path) -> CellsDocument:
    """Read a cells document, refusing an unrecognized contract by name."""
    if not path.exists():
        raise SpecError(f"no cells document at {path}")
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SpecError(f"{path}: unreadable cells document "
                        f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: a cells document is a JSON object")
    version = doc.get("schema_version")
    if version != CELLS_SCHEMA_VERSION:
        raise SpecError(
            f"{path}: cells document schema {version!r}, this build speaks "
            f"{CELLS_SCHEMA_VERSION!r}. A silently-changed contract is a column built "
            "against terms nobody agreed to — refusing.")
    try:
        return CellsDocument(**doc)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="CPU staging for the behavioral column (BRIEF §2). Builds no "
                    "bank contents without an explicit spec and flag; the basis is a "
                    "SPEC field, never a module constant.")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only, data-independent, torch-free verification")
    ap.add_argument("--example-spec", action="store_true",
                    help="print a spec template (no basis baked in) and exit")
    ap.add_argument("--spec", type=Path, default=None, help="the build spec (JSON)")
    ap.add_argument("--census", action="store_true",
                    help="M10: a first-class exit-early readiness census, NEVER output "
                         "truncation. Reports owed artifacts; builds nothing.")
    ap.add_argument("--build", action="store_true",
                    help="stage the column (refuses on a non-READY census)")
    ap.add_argument("--construct-bands", action="store_true",
                    help="build the random band(s) from the spec's named recipe — a "
                         "BANK-CONTENT act, required in writing twice")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override the spec's out_dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and stamp without writing anything")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.example_spec:
        print(json.dumps(example_spec(), indent=1))
        return 0
    if not (args.census or args.build):
        ap.error("nothing to do: pass --selftest, --example-spec, --census or --build")
    if args.spec is None:
        ap.error("--spec is required for --census/--build")

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        logger.error("%s", exc)
        return 2
    if args.out_dir is not None:
        spec = spec.model_copy(update={"out_dir": args.out_dir})

    try:
        cen = census(spec)
    except BehavioralHarnessError as exc:                     # a HALT during census
        logger.error("census HALT: %s", exc)
        return 2
    print(cen.as_table())
    if args.census and not args.build:
        return 0 if cen.ready else 1

    try:
        result, _ = build_banks(spec, construct_bands=args.construct_bands,
                                write=not args.dry_run)
    except BehavioralHarnessError as exc:
        logger.error("HALT: %s", exc)
        return 2
    print(json.dumps(json.loads(result.model_dump_json()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
