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
    ArtifactShaMismatch, BASELINE_DOSE, BESIDE_BAND_FAMILIES, BRIEF_OF_RECORD,
    BRIEF_SHA256, BehavioralHarnessError, CELL_ID_TEMPLATE, CORPUS_SHA_V21, CellKind,
    CellSpec, CorpusVintageError, DOSE_LADDER, ENVELOPE_RULING_OF_RECORD,
    ExpectedNShortfall, GATE_BAND_FAMILIES, GRADE_LINE, LesionRecipeViolation,
    MAX_NEW_TOKENS, N_PER_CELL, NaiveTransplantRowMissing, NativeVectorUnavailable,
    SCORING_DOSES, SEED_MATERIAL_TEMPLATE, SEED_MATERIAL_TEMPLATE_DIGEST,
    SamplingConfig, SiteNotOfRecord, apply_dose_ladder, assert_corpus_vintage,
    assert_naive_row_banked, assert_no_lesion_recipe, baseline_cell, seed_int)
from metabasis.threads import (PRE_RULING_UNRECORDED, RULED_OMP_NUM_THREADS,
                               THREAD_COUNT_MISMATCH_LABEL, ThreadConfig,
                               stamp_thread_config, thread_config_stamp,
                               thread_count_mismatch)

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

#: **B4 (Luxia, 2026-08-04): 6 Σ-beside cells per DESIGNATED row.** The arithmetic is
#: derived from the pre-statement's own pairing convention rather than asserted:
#: §4.1/§5.1 pair every band MEMBER with every dose the signal it controls rides, which
#: is why the isotropic band is `N_BAND_MEMBERS × len(DOSE_LADDER) == 18`. The Σ band is
#: a BESIDE quoted at the SCORING doses only — the ±.3 pair that §4.2(b) and §5.5 read
#: and the same two doses ruling 5's naive null fires at — so the same pairing
#: convention over `SCORING_DOSES` gives `3 × 2 == 6`. The 18-cell alternative (the full
#: signed ladder) was the other option filed to Luxia at staging; she ruled 6.
N_SIGMA_BESIDE_CELLS = N_BAND_MEMBERS * len(SCORING_DOSES)       # 6
#: The doses a Σ-beside cell is staged at, in ladder order (a subset of DOSE_LADDER, so
#: `CellSpec`'s frozen-ladder validator accepts every one of them).
SIGMA_BESIDE_DOSES: tuple[float, ...] = tuple(
    d for d in DOSE_LADDER if d in SCORING_DOSES)

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
#:
#: RULED 2026-08-04 (session 12; `PRESTATEMENT-behavioral-harness-ceremony-2026-08-04.md`
#: §2, sha `386b500a…`) — the recipe of record is ISOTROPIC and its seed material is
#: this template VERBATIM. Before the ruling this module carried a PROPOSAL template
#: (`{basis_sha}|{owner}|L{site}|{family}|member{index:02d}`, family hardcoded to the
#: literal "band"), which is why `--construct-bands` refused: there was no recipe of
#: record to build against. The refusal now lifts for EXACTLY this recipe string and
#: still refuses everything else, so a band built years from now is re-derivable from
#: the stamp alone and a band built under any other recipe cannot exist.
BAND_RECIPE_OF_RECORD = "isotropic-unit-gaussian/prestatement-2026-08-04-§2"
RANDOM_BAND_SEED_TEMPLATE = (
    "{corpus_sha}|{node_key}|{arm}|L{site}|{band_kind}|{vector_key}|member{m:02d}")
RANDOM_BAND_SEED_TEMPLATE_DIGEST = hashlib.sha256(
    RANDOM_BAND_SEED_TEMPLATE.encode()).hexdigest()
RANDOM_BAND_CONSTRUCTION = (
    "unit(standard normal drawn from np.random.default_rng(sha256(seed_material)[:8])) "
    "in the model's residual dim; banked UNIT because the residual-write hook "
    "re-normalizes at attach, so matched-norm is realized by the cell's own α and only "
    "ORIENTATION is load-bearing (banked convention, build_injection_banks.py stamp)")

#: §2's Σ-BESIDE, ruled on the DESIGNATED CELLS ONLY (open word O-2). Members are
#: Σ^{1/2}g renormalized to unit — randoms that "look like" typical residual
#: directions, a strictly harder null than isotropic. **LABELED BESIDE, never the null
#: of record, never inside any gate.** `band_kind` is `SigmaBand` so its seed material
#: is phase-separate from `Rband`/`gRband`: the Σ band and the isotropic band at the
#: same cell are different draws, never the same draw reshaped.
SIGMA_BAND_RECIPE_OF_RECORD = "sigma-shaped-beside/prestatement-2026-08-04-§2"
SIGMA_BAND_KIND = "SigmaBand"
SIGMA_BAND_CONSTRUCTION = (
    "unit(Σ^{1/2} g), g = standard normal drawn from "
    "np.random.default_rng(sha256(seed_material)[:8]) with band_kind=SigmaBand; "
    "Σ^{1/2} = V diag(sqrt(evals + ridge)) Vᵀ from the banked sigma_L*.npz beside the "
    "vector (evals ascending, evecs in COLUMNS, the builder's own eigh convention); the "
    "banked ridge is applied and named rather than dropped. BESIDE ONLY — never the "
    "null of record, never an input to any gate (pre-statement §2, ruling O-2)")

#: The two cells Luxia designated for the Σ-beside (O-2), as DATA. A designation is
#: (node, arm, site, band family) — the tuple a cell is identified by — so the guard is
#: mechanical and a third Σ cell cannot be added by prose. `vector_key` is deliberately
#: NOT part of the designation: a cell's band travels with the cell, not with the key.
SIGMA_BESIDE_DESIGNATIONS: dict[str, tuple[str, str, int, str]] = {
    # the 3B certification node's CALIBRATION cell (native band, its own site)
    "3b-certification-calibration": ("qwen2.5-3b-instruct", "native", 26, "Rband"),
    # the bridge row's TRANSPORTED cell (8b → qwen-7b, the banked dose-monotone leg)
    "bridge-row-transported": ("qwen-7b", "native", 21, "gRband"),
}


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


# --- HALT C (Luxia's ruling 1, 2026-08-05): the site guard's robustness-site role ---
class SiteRoleRefused(SiteNotOfRecord):
    """A DECLARED site role does not license the site being staged.

    The parent class is `SiteNotOfRecord` on purpose: every caller that already
    treats an off-record site as a HALT keeps treating these as HALTs, and the
    subclasses only make the REASON machine-readable.
    """


class RobustnessSiteNotRegistered(SiteRoleRefused):
    """`site_role="robustness_site"` was declared for a site no registry carries.

    §4.2's remedy (i) re-calibrates at the node's REGISTERED robustness site — a
    registered site, never a new one. A spec that could nominate any site by
    writing a role string would be site-fishing with an extra field.
    """


class RobustnessSiteOffGrid(SiteRoleRefused):
    """The declared robustness site is not on the node's fixed fit grid (`SITES`).

    Separate from `RobustnessSiteNotRegistered` because the remedies differ: this
    one is a registry disagreement (`ROBUSTNESS_SITES` vs `SITES`) and is a desk
    HALT, not a spec typo.
    """


# --- HALT D (Luxia's ruling 2, 2026-08-05): the vector-class contract -------------
class VectorClassContractError(SpecError):
    """The vector-class declaration and the object it describes disagree.

    ROOT CAUSE THIS EXISTS FOR, in one sentence (the pilot HALT's own): a CLASS
    column has TWO BASES — the GENERATION basis (the corpus manifest, which is
    what the prompts and the generations are of) and the VECTOR basis (the RULED
    CONTRAST SET the class object was built from) — and the harness was written
    when every object was an entropy-gradient vector, i.e. when the two bases
    coincided. Naming both is the fix; conflating them is the defect.
    """


class FDGateWaiverRefused(VectorClassContractError):
    """`fd_gate_not_applicable` was claimed by an object that must still be gated.

    The FD gate is the entropy-gradient object's OWN acceptance test (§9 item 3:
    present, FD-gated and FD-PASS at the basis of record). A CAA/PCA class object
    has no FD-gate analogue — its builder's stamp says so in as many words — but
    an EGV that waived the gate would be an ungated lever wearing a new field.
    """


class VectorBasisMissing(VectorClassContractError):
    """A class vector did not name the RULED contrast-set basis it was built from."""


class VectorBasisConflated(VectorClassContractError):
    """One basis was written where the other belongs — the two-bases defect itself."""


# ---------------------------------------------------------------- spec types
#: HALT C: the two site ROLES §4.2 distinguishes. `site_of_record` is the ruled site
#: a column normally fires at; `robustness_site` is remedy (i)'s REGISTERED second
#: site (gemma3-27b L41, llama-3.1-70b-instruct L43 — the only two in the campaign).
#: A role is DECLARED in the spec or it does not exist: `None` means "no role was
#: declared", which is the default and refuses exactly as this module always did.
SiteRole = Literal["site_of_record", "robustness_site"]

#: HALT D: the vector CLASS a reference declares. `entropy_gradient` is the campaign's
#: object of record and the default, so a spec written before this ruling means
#: exactly what it meant before. The other two are the class-pilot's constructions
#: (`build_contrast_vectors`: text-contrast CAA of record + the repeng-PCA method row).
VectorClass = Literal["entropy_gradient", "caa", "repeng_pca"]
#: The one class whose acceptance test IS the FD gate. Everything keyed off this name
#: rather than off `!= "caa"`, so a third class added later inherits the class rules.
EGV_VECTOR_CLASS: str = "entropy_gradient"
CLASS_VECTOR_CLASSES: tuple[str, ...] = ("caa", "repeng_pca")

#: The two BASES a behavioral cell stands on, as a closed vocabulary. `corpus-manifest`
#: is the GENERATION basis (the corpus the prompts and generations are of);
#: `contrast-set` is the VECTOR basis of a class object (the RULED pinned set
#: `build_contrast_vectors` refuses to build without). Naming both is the whole fix.
VectorBasisKind = Literal["corpus-manifest", "contrast-set"]
class VectorBasis(BaseModel):
    """WHICH basis an object stands on, with the KIND named beside the sha.

    HALT D's fix in one type. Before this, a stamp carried a bare
    `corpus_manifest_sha256` for everything, so a CAA object could only be stamped by
    writing the corpus sha into a field that means "the basis this object was built
    from" — false provenance — or by leaving a hole. A basis is now a (kind, sha256)
    pair, and the kind is a closed vocabulary, so the GENERATION basis and the VECTOR
    basis can sit in one stamp without either being readable as the other.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: VectorBasisKind
    sha256: str = Field(min_length=64, max_length=64)
    #: the pin/ruling this basis is of record under, quoted rather than implied — a
    #: contrast set is admissible only because its sha is RULED in a pin file
    #: (`build_contrast_vectors.assert_contrast_set_pinned`), and the stamp should
    #: say which ruling, not just which bytes.
    provenance: str = ""

    @field_validator("sha256")
    @classmethod
    def _is_hex(cls, v: str) -> str:
        int(v, 16)              # raises ValueError on a non-hex digest (M40)
        return v


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
    #: The BLAS thread count `sha256` above was banked at (Luxia ruling
    #: 2026-08-01). None = the digest predates the ruling and its count is
    #: UNRECORDED — a fact to be QUOTED when the file on disk disagrees, never
    #: inferred, even though every deployed job script of that vintage exported
    #: OMP_NUM_THREADS=1. This exists so a sha mismatch can name both sides:
    #: without it the census could only say what the file on disk was built at.
    expected_thread_count: Optional[int] = Field(
        default=None, ge=1,
        description="thread count of record for `sha256`; None = pre-ruling")

    #: HALT D. WHAT KIND OF OBJECT THIS IS. Defaulted to the campaign's object of
    #: record, so every spec written before the 2026-08-05 ruling means exactly what
    #: it meant before and its census/stamp bytes do not move.
    vector_class: VectorClass = EGV_VECTOR_CLASS  # type: ignore[assignment]
    #: HALT D. The FD gate DOES NOT APPLY to this object — admissible for a CLASS
    #: vector ONLY. `build_contrast_vectors`'s own stamp is the authority for the
    #: claim ("no FD-gate analogue exists for a CAA object — build acceptance is
    #: stamp completeness + the pair-count assertion + the anchor cosines"), and an
    #: EGV that set this would be an ungated lever, which `FDGateWaiverRefused`
    #: refuses at spec load time rather than at fire time.
    fd_gate_not_applicable: bool = False
    #: HALT D. The basis THIS OBJECT was built from, named. Required for a class
    #: vector (its basis is the RULED contrast set, never the corpus manifest) and
    #: refused for an EGV, whose basis is the corpus and is already carried by the
    #: vintage chain — writing it twice under two names is how the two bases blur.
    vector_basis: Optional[VectorBasis] = None

    @property
    def is_class_vector(self) -> bool:
        """True for a CAA/PCA class object — the two-bases case, in one predicate."""
        return self.vector_class != EGV_VECTOR_CLASS

    @model_validator(mode="after")
    def _sha_is_a_sha(self) -> "VectorRef":
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError(f"{self.key}: sha256 must be 64 hex chars")
        return self

    @model_validator(mode="after")
    def _class_contract(self) -> "VectorRef":
        """HALT D's contract: the class, the gate waiver and the basis agree, or HALT.

        Four refusals, each with its own class because each has its own remedy:
        an EGV waiving its gate (`FDGateWaiverRefused`), a class vector that both
        waives the gate and names one (`FDGateWaiverRefused` — a waiver and a claim
        are not compatible), a class vector with no ruled basis
        (`VectorBasisMissing`), and either kind of basis written under the other's
        name (`VectorBasisConflated`).
        """
        if self.fd_gate_not_applicable and not self.is_class_vector:
            raise FDGateWaiverRefused(
                f"{self.key}: vector_class={self.vector_class!r} claims "
                "fd_gate_not_applicable. The FD gate is the entropy-gradient "
                "object's OWN acceptance test (§9 item 3: present, FD-gated and "
                "FD-PASS at the basis of record) — waiving it would bank an "
                "UNGATED lever. The waiver is admissible for a class object "
                f"({', '.join(CLASS_VECTOR_CLASSES)}) and for nothing else.")
        if self.fd_gate_not_applicable and self.fd_gate is not None:
            raise FDGateWaiverRefused(
                f"{self.key}: fd_gate_not_applicable is set AND an fd_gate file is "
                f"named ({self.fd_gate}). A waiver and a claim cannot both be true: "
                "either the object has a gate (drop the waiver and the census will "
                "read it) or it does not (drop the path).")
        if self.is_class_vector and self.vector_basis is None:
            raise VectorBasisMissing(
                f"{self.key}: vector_class={self.vector_class!r} names no "
                "`vector_basis`. A class object's basis is the RULED CONTRAST SET "
                "it was built from (build_contrast_vectors refuses to build from a "
                "set whose sha256 is not RULED in a pin file); writing the corpus "
                "sha instead would be false provenance, and writing nothing would "
                "be a hole. Both are refused here (§9 item 2).")
        if self.is_class_vector and self.vector_basis.kind != "contrast-set":
            raise VectorBasisConflated(
                f"{self.key}: a class object's vector_basis.kind is "
                f"{self.vector_basis.kind!r}. The VECTOR basis of a class object is "
                "the contrast set; the corpus manifest is the GENERATION basis and "
                "rides its own field. Conflating them is the defect this field "
                "exists to make impossible.")
        if not self.is_class_vector and self.vector_basis is not None:
            raise VectorBasisConflated(
                f"{self.key}: an entropy-gradient object declared a vector_basis "
                f"({self.vector_basis.kind}). An EGV's basis IS the corpus manifest "
                "and is already carried by the vintage chain and the stamp's "
                "corpus_manifest_sha256 — a second name for the same basis is how "
                "the two bases blur.")
        return self


class RandomBandRecipe(BaseModel):
    """The RULED isotropic band recipe (pre-statement §2) — the only one that builds.

    The v2.1/v3 bands do not exist anywhere as banked files (the certification census
    found zero rband files in the node tree; the v1 anamnesis `load_axes` banks were the
    precedent and have no successor). This recipe is what a band IS built from, written
    so the members are re-derivable from the recipe string alone — and `recipe_of_record`
    pins it to the ruled string, so a spec cannot smuggle in a different construction
    while still passing `--construct-bands`.

    Every field of the seed material is a field here, in the ruled order:
    `{corpus_sha}|{node_key}|{arm}|L{site}|{band_kind}|{vector_key}|member{m:02d}`.
    `band_kind` phase-separates the native band from the transported one (§4.1's
    `Rband*` vs `gRband*`), and `vector_key` phase-separates two bands that ride the
    same cell for different objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: the ruled recipe string. Any other value is refused at construction, which is
    #: what "the refusal lifts for exactly this recipe" means mechanically.
    recipe_of_record: str = BAND_RECIPE_OF_RECORD
    #: the basis the band is anchored to — the corpus manifest sha, so a band built at
    #: one basis can never be silently reused at another.
    corpus_sha: str = Field(min_length=8)
    node_key: str = Field(min_length=1, description="whose space the band lives in")
    arm: Literal["native", "raw"]
    site: int = Field(ge=0)
    band_kind: Literal["Rband", "gRband"]
    vector_key: str = Field(
        min_length=1,
        description="the object this band is the control FOR (its cell's vector key)")
    #: WHOSE SPACE THE GAUSSIAN IS DRAWN IN — the one thing the ruled seed material
    #: does not itself say. The seed material names the CELL (target node, target site,
    #: arm); the draw happens in the space the member starts in, which is the target's
    #: for `Rband` (a native control at the node's own site) and the SOURCE's for
    #: `gRband`, because §5.1's transported band is "the source's randoms through the
    #: SAME map" — the control travels the same road as the signal or it is not a
    #: control for transport. Recorded explicitly so the two readings can never blur.
    draw_space_key: str = Field(min_length=1)
    dim: int = Field(gt=0, description="hidden dimension of `draw_space_key` at its site")
    n_members: int = Field(gt=0, default=N_BAND_MEMBERS)
    construction: str = RANDOM_BAND_CONSTRUCTION

    @model_validator(mode="after")
    def _draw_space_matches_the_family(self) -> "RandomBandRecipe":
        if self.band_kind == "Rband" and self.draw_space_key != self.node_key:
            raise ValueError(
                f"Rband draws in the node's OWN space: draw_space_key="
                f"{self.draw_space_key!r} != node_key={self.node_key!r} (§4.1: the "
                "native band asks whether the SITE actuates)")
        if self.band_kind == "gRband" and self.draw_space_key == self.node_key:
            raise ValueError(
                f"gRband draws in the SOURCE's space and is carried through the same "
                f"map (§5.1); draw_space_key={self.draw_space_key!r} is the target "
                "itself, which would make the control a native band wearing a "
                "transported name")
        return self

    @model_validator(mode="after")
    def _is_the_recipe_of_record(self) -> "RandomBandRecipe":
        if self.recipe_of_record != BAND_RECIPE_OF_RECORD:
            raise ValueError(
                f"recipe_of_record={self.recipe_of_record!r} is not the ruled recipe "
                f"{BAND_RECIPE_OF_RECORD!r} (pre-statement §2). This module builds ONE "
                "band recipe and refuses every other by name.")
        if self.construction != RANDOM_BAND_CONSTRUCTION:
            raise ValueError(
                "construction text differs from the ruled construction — the recipe "
                "string and its construction travel together or the stamp lies")
        return self

    def seed_material(self, index: int) -> str:
        """The ruled seed material for member `index`, built from the template."""
        return RANDOM_BAND_SEED_TEMPLATE.format(
            corpus_sha=self.corpus_sha, node_key=self.node_key, arm=self.arm,
            site=self.site, band_kind=self.band_kind, vector_key=self.vector_key,
            m=index)


class SigmaBandRecipe(BaseModel):
    """§2's Σ-BESIDE — the stricter null, on the two DESIGNATED cells only (O-2).

    Never the null of record and never an input to a gate: `build_sigma_band` refuses
    to build for any cell outside `SIGMA_BESIDE_DESIGNATIONS`, and every member it does
    build is keyed `SigmaBand*` so a scorer that pools it into a band would have to do
    so by name.

    The Σ estimator is NAMED here rather than implied: it is the banked
    `sigma_L{site}_{node}.npz` written beside the v3 entropy-gradient vector — the
    TOKEN-LEVEL residual covariance over the completion positions of a deterministic
    60-text stride through the frozen corpus manifest, eigendecomposed (`evals`
    ascending, `evecs` in columns) with a relative ridge banked beside it. That is a
    covariance of token residuals, not of texts and not of the fit's train matrix; it is
    stated because a "Σ-shaped" null means nothing until Σ is identified.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_of_record: str = SIGMA_BAND_RECIPE_OF_RECORD
    designation: str = Field(
        min_length=1, description="key into SIGMA_BESIDE_DESIGNATIONS")
    corpus_sha: str = Field(min_length=8)
    node_key: str = Field(min_length=1)
    arm: Literal["native", "raw"]
    site: int = Field(ge=0)
    #: the isotropic band this Σ band sits BESIDE (Rband on a calibration cell,
    #: gRband on a transported cell) — recorded so the beside is readable as a beside.
    beside_band_kind: Literal["Rband", "gRband"]
    vector_key: str = Field(min_length=1)
    #: the SPACE the Σ lives in, which is the space the member is drawn in: the node's
    #: own for a calibration (`Rband`) beside, and the SOURCE's for a transported
    #: (`gRband`) beside — the Σ band travels the same road as the band it sits beside,
    #: so `sigma_npz`/`dim` name the source's covariance on a transported designation.
    dim: int = Field(gt=0)
    n_members: int = Field(gt=0, default=N_BAND_MEMBERS)
    sigma_npz: Path
    sigma_sha256: Optional[str] = None
    sigma_estimator: str = Field(
        min_length=20,
        description="the estimator, NAMED — never 'the covariance'")
    sigma_provenance: str = Field(min_length=10)
    apply_banked_ridge: bool = True
    construction: str = SIGMA_BAND_CONSTRUCTION

    @model_validator(mode="after")
    def _is_the_recipe_of_record(self) -> "SigmaBandRecipe":
        if self.recipe_of_record != SIGMA_BAND_RECIPE_OF_RECORD:
            raise ValueError(
                f"recipe_of_record={self.recipe_of_record!r} is not the ruled Σ-beside "
                f"recipe {SIGMA_BAND_RECIPE_OF_RECORD!r} (pre-statement §2 / O-2)")
        if self.construction != SIGMA_BAND_CONSTRUCTION:
            raise ValueError(
                "Σ-beside construction text differs from the ruled construction")
        return self

    def seed_material(self, index: int) -> str:
        """Same template, `band_kind=SigmaBand` — phase-separate from the isotropic."""
        return RANDOM_BAND_SEED_TEMPLATE.format(
            corpus_sha=self.corpus_sha, node_key=self.node_key, arm=self.arm,
            site=self.site, band_kind=SIGMA_BAND_KIND, vector_key=self.vector_key,
            m=index)


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
    #: the Σ-shaped BESIDE, on the two designated cells only (O-2). Never the null of
    #: record; carried on the band it sits beside so the two are read together.
    sigma_beside: Optional[SigmaBandRecipe] = None

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
        if self.recipe is not None and self.recipe.band_kind != self.family:
            raise ValueError(
                f"{self.family}: recipe's band_kind is {self.recipe.band_kind!r} — the "
                "seed material would phase-separate the band from its own cell "
                "(§4.1 keeps Rband* and gRband* distinct in name AND in seed)")
        if (self.sigma_beside is not None
                and self.sigma_beside.beside_band_kind != self.family):
            raise ValueError(
                f"{self.family}: Σ-beside declares itself beside "
                f"{self.sigma_beside.beside_band_kind!r}")
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


class ComposedMapRef(BaseModel):
    """A TWO-HOP composed map: source → hub → target, through two banked hub legs.

    The composed-object behavioral leg (brief ruling 6 / pre-statement §4) carries the
    SOURCE's object back to the hub through `hub_to_source` in REVERSE and out to the
    target through `hub_to_target` FORWARDS. That is exactly the arithmetic
    `read_composed_predictions.composed_exchange_rate` performs on the two banked hub
    legs — reused, not re-derived, so the behavioral object rides the same composition
    the â_comp prediction is made of.

    The two legs must name ONE hub at ONE hub site in ONE arm and ONE family, or the
    two-hop is not composable (the composed-path module raises on exactly this) — so
    the agreement is a validator here rather than a comment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_key: str = Field(min_length=1)
    hub_site: int = Field(ge=0)
    family: str = Field(min_length=1)
    arm: Literal["native", "raw"]
    #: the hub→SOURCE leg, ridden in REVERSE (source space → hub space).
    hub_to_source: TransportMapRef
    #: the hub→TARGET leg, ridden FORWARDS (hub space → target space).
    hub_to_target: TransportMapRef
    #: the pre-statement's ranking rule for the pair, recorded so the artifact and the
    #: spec agree on WHY this pair is here (B2: the product of the two legs' âs).
    selection_rule: str = ""
    a_hat_hub_to_source: Optional[float] = None
    a_hat_hub_to_target: Optional[float] = None
    a_hat_product: Optional[float] = None
    product_rank: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_hub_one_arm_one_family(self) -> "ComposedMapRef":
        for name, leg in (("hub_to_source", self.hub_to_source),
                          ("hub_to_target", self.hub_to_target)):
            if leg.arm != self.arm:
                raise ValueError(
                    f"{name} is banked in the {leg.arm!r} arm but the composed map "
                    f"runs in {self.arm!r} — a behavioral cell inherits ONE arm from "
                    "the map it rides (v3 prereg §3.2), so two legs in different arms "
                    "are not composable")
            if leg.family != self.family:
                raise ValueError(
                    f"{name} is family {leg.family!r}, the composed map is "
                    f"{self.family!r} — the composed object must ride ONE family end "
                    "to end or its â_comp is not the filed prediction's")
        if self.hub_to_source.fit == self.hub_to_target.fit:
            raise ValueError(
                "the two hub legs are the SAME fit — a composed 'pair' whose two legs "
                "coincide is the identity dressed as a composition")
        return self

    @property
    def legs(self) -> tuple[TransportMapRef, TransportMapRef]:
        return (self.hub_to_source, self.hub_to_target)


class BankSpec(BaseModel):
    """Everything one (node, arm, site) column is staged from — the whole basis.

    Nothing outside this object decides what gets built. That is the basis-agnostic
    contract: a v3 column is this file plus a different spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_key: str = Field(min_length=1, description="the TARGET node")
    arm: Literal["native", "raw"]
    site: int = Field(ge=0, description="the target's site of record")
    #: HALT C (Luxia's ruling, 2026-08-05). The ROLE `site` is being staged in.
    #: `None` — the default and every pre-ruling spec — declares no role and is
    #: refused exactly as this module always refused: any site ≠ SITE_OF_RECORD
    #: HALTs. `"robustness_site"` is §4.2's frozen remedy (i) and admits ONLY the
    #: node's REGISTERED robustness site (gemma3-27b L41, llama-3.1-70b L43), and
    #: only when that site is also on the fixed fit grid. The role is a WRITTEN
    #: declaration in the spec because that is what makes it not site-fishing:
    #: `actuation_calibration.assert_no_site_fishing` already implements exactly
    #: this admissibility, and the staging guard was the one layer that lacked it.
    site_role: Optional[SiteRole] = None
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
    #: the composed-object alternative to `transport_map` (brief ruling 6 / B2). Exactly
    #: one of the two is set on a transported column: a direct pair fit OR a two-hop
    #: through the crowned hub, never both, because a cell rides ONE road.
    composed_map: Optional[ComposedMapRef] = None
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
        if self.transport_map is not None and self.composed_map is not None:
            raise ValueError(
                "a transported column rides ONE road: `transport_map` (a direct pair "
                "fit) or `composed_map` (the two-hop through the hub), never both")
        for half, band in (("native_band", self.native_band),
                           ("source_band", self.source_band)):
            sig = band.sigma_beside
            if sig is None:
                continue
            # The Σ-beside designates a CELL TUPLE (O-2). A spec that carried a recipe
            # describing SOMEONE ELSE's cell would pass `assert_sigma_beside_designated`
            # (which reads the recipe, not the spec) while banking the beside into this
            # column, so the two are tied together here.
            if (sig.node_key, sig.arm, sig.site) != (self.node_key, self.arm,
                                                     self.site):
                raise ValueError(
                    f"{half}: Σ-beside recipe describes "
                    f"({sig.node_key}, {sig.arm}, L{sig.site}) but this spec is "
                    f"({self.node_key}, {self.arm}, L{self.site}) — O-2 designates "
                    "CELLS, and a beside cannot be banked into a cell it was not "
                    "designated for")
            assert_sigma_beside_designated(sig)
        return self

    @property
    def transported_prefix(self) -> str:
        """The banked `g`-prefix convention for a transported object."""
        return "g"

    @property
    def map_of_record(self) -> Optional[TransportMapRef | ComposedMapRef]:
        """Whichever road the transported half rides, or None on a calibration-only."""
        return self.transport_map or self.composed_map

    @property
    def map_family(self) -> Optional[str]:
        m = self.map_of_record
        return None if m is None else m.family

    @property
    def sigma_beside_designations(self) -> tuple[str, ...]:
        """The Σ-beside designations this spec carries, in (native, transported) order."""
        return tuple(b.sigma_beside.designation
                     for b in (self.native_band, self.source_band)
                     if b.sigma_beside is not None)


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
    #: Populated ONLY when the two sides' thread counts differ (Luxia ruling
    #: 2026-08-01, scope 3). A byte difference a count mismatch explains is a
    #: CHARACTERIZED break; the same difference reported bare reads as
    #: corruption, and the two demand opposite responses. None on agreement,
    #: which is why it is a separate field rather than prose inside
    #: `blocking_reason`: a reader scanning rows can test it.
    thread_count_mismatch: Optional[str] = None
    constructible: bool = False
    #: HALT D. Populated ONLY for a CLASS object (None on every entropy-gradient row,
    #: which is every row of every column banked to date). `corpus_vintage` above is
    #: the GENERATION basis and stays exactly what it was; `vector_basis` is the
    #: VECTOR basis, declared and re-read from the object's own build stamp. Two
    #: fields because they are two facts — that is the whole HALT-D fix.
    vector_class: Optional[str] = None
    fd_gate_not_applicable: Optional[bool] = None
    vector_basis: Optional[dict] = None


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
    #: HALT D. The class of the object THIS cell rides, carried per cell because a
    #: cell rides exactly one vector — the same place `fd_gate`/`build_stamp` already
    #: live. Defaulted to the object of record, so an EGV column's document means
    #: exactly what it meant before the ruling.
    vector_class: VectorClass = EGV_VECTOR_CLASS  # type: ignore[assignment]
    #: HALT D. The VECTOR basis for a class cell (kind `contrast-set`), None for an
    #: EGV cell whose basis is the generation basis and is carried by the vintage
    #: chain. The engine stamps BOTH bases, each under its own name, and refuses a
    #: class cell that arrives without this.
    vector_basis: Optional[VectorBasis] = None

    @model_validator(mode="after")
    def _class_cell_names_its_basis(self) -> "StagedCell":
        if self.vector_class != EGV_VECTOR_CLASS and self.vector_basis is None:
            raise VectorBasisMissing(
                f"{self.spec.cell_id}: vector_class={self.vector_class!r} with no "
                "`vector_basis` — a class cell's VECTOR basis is the ruled contrast "
                "set and the engine will refuse to stamp it (§9 item 2).")
        if self.vector_class == EGV_VECTOR_CLASS and self.vector_basis is not None:
            raise VectorBasisConflated(
                f"{self.spec.cell_id}: an entropy-gradient cell carries a "
                f"vector_basis ({self.vector_basis.kind}); the EGV's basis is the "
                "GENERATION basis and has exactly one name.")
        return self


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
    #: HALT C. The role the column's site was staged in, carried into the job so the
    #: engine's own re-derived cross-check asks the SAME question the staging module
    #: answered (§4.3 forbids trusting a snapshot; it does not forbid carrying the
    #: declaration). None = no role declared = the pre-ruling behaviour.
    site_role: Optional[SiteRole] = None
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
    stamp alone. The material is the RULED template (pre-statement §2) built through
    `RandomBandRecipe.seed_material`, so the string in the stamp and the string that
    seeded the draw are the same string by construction.
    """
    assert_construction_admissible("random_band",
                                   key=f"{recipe.node_key}/member{index}")
    if recipe.recipe_of_record != BAND_RECIPE_OF_RECORD:      # pragma: no cover
        raise ConstructionRefused(
            f"{recipe.node_key}: band recipe {recipe.recipe_of_record!r} is not the "
            f"recipe of record {BAND_RECIPE_OF_RECORD!r} — REFUSED")
    if not 1 <= index <= recipe.n_members:
        raise ValueError(
            f"band member index {index} out of range 1..{recipe.n_members}")
    rng = np.random.default_rng(seed_int(recipe.seed_material(index)))
    return unit(rng.standard_normal(recipe.dim))


def build_random_band(recipe: RandomBandRecipe, family: Literal["Rband", "gRband"]
                      ) -> dict[str, np.ndarray]:
    """The whole band, keyed `{family}{i}` in member order (§4.1's naming).

    `family` must be the recipe's own `band_kind`: the key prefix and the seed material
    are two views of the same fact, and letting them disagree would bank a `gRband`
    keyed file whose members were drawn in the `Rband` phase.
    """
    if family != recipe.band_kind:
        raise ConstructionRefused(
            f"{recipe.node_key}: asked for a {family} band from a "
            f"{recipe.band_kind} recipe — the key prefix and the seed material must "
            "name the same band (§4.1)")
    return {f"{family}{i}": random_band_member(recipe, i)
            for i in range(1, recipe.n_members + 1)}


def assert_sigma_beside_designated(recipe: SigmaBandRecipe) -> tuple[str, str, int, str]:
    """O-2's guard: the Σ-beside exists on TWO named cells and nowhere else.

    Returns the designated tuple on success. Raises `ConstructionRefused` both when the
    designation is unknown and when it is known but describes a different cell — the
    second is the one that matters, because a copied spec with an edited node key is
    exactly how a "beside" quietly becomes a third arm of the experiment.
    """
    want = SIGMA_BESIDE_DESIGNATIONS.get(recipe.designation)
    if want is None:
        raise ConstructionRefused(
            f"Σ-beside designation {recipe.designation!r} is not one of the two Luxia "
            f"designated (O-2): {sorted(SIGMA_BESIDE_DESIGNATIONS)}. The Σ band is a "
            "BESIDE on two cells, not an instrument of the column.")
    got = (recipe.node_key, recipe.arm, recipe.site, recipe.beside_band_kind)
    if got != want:
        raise ConstructionRefused(
            f"Σ-beside {recipe.designation!r} is designated for {want} but this recipe "
            f"describes {got} — REFUSED (O-2 designates CELLS, not recipes)")
    return want


def _sigma_half(sigma_npz: Path, dim: int, *, apply_ridge: bool,
                key: str = "") -> tuple[np.ndarray, dict]:
    """Σ^{1/2} from a banked `sigma_L*.npz`, plus the facts its stamp must carry.

    The banked convention is `np.linalg.eigh`'s: `evals` ASCENDING, `evecs` in COLUMNS,
    and the relative ridge stored BESIDE the spectrum rather than folded into it
    (`build_entropy_gradient.py`). Both are re-asserted here rather than assumed,
    because a transposed `evecs` would still produce a plausible-looking band.
    """
    if not sigma_npz.exists():
        raise ConstructionRefused(
            f"{key or 'Σ-beside'}: no banked covariance at {sigma_npz} — the Σ-beside "
            "names its estimator or it does not build")
    try:
        with np.load(sigma_npz) as z:
            missing = {"evals", "evecs"} - set(z.files)
            if missing:
                raise ConstructionRefused(
                    f"{key or 'Σ-beside'}: {sigma_npz} lacks {sorted(missing)} "
                    f"(keys: {sorted(z.files)})")
            evals = np.asarray(z["evals"], dtype=np.float64).reshape(-1)
            evecs = np.asarray(z["evecs"], dtype=np.float64)
            ridge = float(np.asarray(z["ridge"])) if "ridge" in z.files else 0.0
            n_positions = (int(np.asarray(z["n_positions"]))
                           if "n_positions" in z.files else None)
    except (OSError, ValueError, EOFError) as exc:
        raise ConstructionRefused(
            f"{key or 'Σ-beside'}: {sigma_npz} is not a readable npz "
            f"({type(exc).__name__}: {exc})") from exc
    if evecs.shape != (dim, dim) or evals.shape != (dim,):
        raise DimensionMismatch(
            f"{key or 'Σ-beside'}: banked Σ is {evecs.shape}/{evals.shape}, the space "
            f"is dim {dim}")
    if not np.all(np.diff(evals) >= -1e-9):
        raise ConstructionRefused(
            f"{key or 'Σ-beside'}: banked eigenvalues are not ascending — this is not "
            "the builder's eigh convention and the square root would be built on a "
            "misread spectrum")
    floor = ridge if apply_ridge else 0.0
    lam = np.clip(evals, 0.0, None) + floor
    half = (evecs * np.sqrt(lam)) @ evecs.T
    facts = {
        "sigma_npz": str(sigma_npz),
        "sigma_sha256": sha256_file(sigma_npz),
        "banked_ridge": ridge,
        "ridge_applied": bool(apply_ridge),
        "n_positions": n_positions,
        "eigenvalue_convention": "ascending, eigenvectors in COLUMNS (np.linalg.eigh)",
        "n_negative_eigenvalues_clipped": int(np.sum(evals < 0.0)),
        "top_eigenvalue": float(evals[-1]),
        "median_eigenvalue": float(np.median(evals)),
    }
    return half, facts


def sigma_band_member(recipe: SigmaBandRecipe, index: int, *,
                      sigma_half: Optional[np.ndarray] = None) -> np.ndarray:
    """One Σ-beside member: unit(Σ^{1/2} g), g the SAME kind of seeded unit Gaussian."""
    assert_construction_admissible("random_band",
                                   key=f"{recipe.node_key}/Σmember{index}")
    assert_sigma_beside_designated(recipe)
    if not 1 <= index <= recipe.n_members:
        raise ValueError(
            f"Σ band member index {index} out of range 1..{recipe.n_members}")
    half = (sigma_half if sigma_half is not None else
            _sigma_half(recipe.sigma_npz, recipe.dim,
                        apply_ridge=recipe.apply_banked_ridge,
                        key=recipe.designation)[0])
    rng = np.random.default_rng(seed_int(recipe.seed_material(index)))
    return unit(half @ rng.standard_normal(recipe.dim))


def build_sigma_band(recipe: SigmaBandRecipe) -> tuple[dict[str, np.ndarray], dict]:
    """The whole Σ-beside, keyed `SigmaBand{i}`, plus the estimator facts for the stamp.

    The square root is computed ONCE for the band (it is the expensive step and it is
    identical across members), and the per-member randomness still comes from each
    member's own digest — so a partial rebuild of member 2 is in phase with 1 and 3.
    """
    assert_sigma_beside_designated(recipe)
    half, facts = _sigma_half(recipe.sigma_npz, recipe.dim,
                              apply_ridge=recipe.apply_banked_ridge,
                              key=recipe.designation)
    if recipe.sigma_sha256 is not None and facts["sigma_sha256"] != recipe.sigma_sha256:
        raise ArtifactShaMismatch(
            f"Σ-beside {recipe.designation}: {recipe.sigma_npz} sha "
            f"{facts['sigma_sha256'][:12]}… != expected {recipe.sigma_sha256[:12]}… "
            "(M4 — a sha mismatch on a number-bearing artifact is a HALT)")
    band = {f"{SIGMA_BAND_KIND}{i}": sigma_band_member(recipe, i, sigma_half=half)
            for i in range(1, recipe.n_members + 1)}
    facts.update({
        "recipe_of_record": recipe.recipe_of_record,
        "designation": recipe.designation,
        "beside_band_kind": recipe.beside_band_kind,
        "estimator": recipe.sigma_estimator,
        "estimator_provenance": recipe.sigma_provenance,
        "seed_material_template": RANDOM_BAND_SEED_TEMPLATE,
        "seed_material_template_digest": RANDOM_BAND_SEED_TEMPLATE_DIGEST,
        "seed_materials": [recipe.seed_material(i)
                           for i in range(1, recipe.n_members + 1)],
        "GRADE": "BESIDE ONLY — never the null of record, never an input to any gate",
        "thread_config": thread_config_stamp(),
    })
    return band, facts


# ---------------------------------------------------------------- the census (§4.3)
def _thread_mismatch_note(ref: VectorRef, *, what: str) -> Optional[str]:
    """A LABELED thread-count mismatch between a banked digest and the file on disk.

    SCOPE 3 OF THE 2026-08-01 RULING, at the one place in this module where a
    rebuild meets a banked pre-ruling artifact: `verify_sha` compares the bytes
    on disk against the digest the spec carries, and before this the ONLY thing
    it could report was that they differ. It could not say whether they differ
    because something is wrong or because eigh was run at a different thread
    count — a difference that is bitwise-expected (eigh is deterministic at a
    FIXED count and its bytes move across counts) and completely benign.

    BOTH sides are quoted whenever they differ:

      * the BANKED side comes from `VectorRef.expected_thread_count`, which is
        None for every digest banked before the ruling. That reads as
        "unrecorded", NOT as 1: the deployed job scripts of that vintage did
        export OMP_NUM_THREADS=1, but the digest does not say so and an
        inference is not a record.
      * the ON-DISK side is read out of the artifact's own build stamp with the
        shared backward-compatible reader, so a rebuild that carries the field
        is quoted from its own stamp rather than from this process's threadpool
        — the file may well have been built somewhere else.

    Returns None only when both counts are known AND equal. A missing build
    stamp is a mismatch to be quoted, not an agreement (M19: a degraded read is
    described, never upgraded to a pass).
    """
    banked = (None if ref.expected_thread_count is None else
              ThreadConfig(effective_num_threads=ref.expected_thread_count,
                           consensus="agreed",
                           matches_ruled_default=(ref.expected_thread_count
                                                  == RULED_OMP_NUM_THREADS)))
    on_disk = stamp_thread_config(_read_json(ref.build_stamp,
                                             what=f"{what} build stamp"))
    note = thread_count_mismatch(banked, on_disk, banked_label="banked digest",
                                 rebuilt_label="artifact on disk")
    if note is None:
        return None
    if on_disk is None:
        note += (" The artifact on disk carries no thread_config either, so "
                 f"neither side can vouch for its count ({PRE_RULING_UNRECORDED}"
                 "); rebuild it with a stamping builder before adjudicating.")
    return note


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
        #  A SHA MISMATCH IS STILL A HALT (M4) — this does not soften it. What
        #  it adds is the one distinction the halt could not make on its own:
        #  whether the two artifacts were built at DIFFERENT thread counts, in
        #  which case the byte difference is the characterized 2026-08-01 break
        #  rather than corruption. Both counts are quoted, never one.
        note = _thread_mismatch_note(ref, what=name)
        return ArtifactRow(
            name=name, role=role, path=str(ref.npz), present=True,
            sha256=sha256_file(ref.npz), expected_sha256=ref.sha256,
            sha_verified=False, cells_at_risk=cells_at_risk, ready=False,
            thread_count_mismatch=note,
            blocking_reason=str(exc) + (f" — {note}" if note else ""))
    fd = _read_json(ref.fd_gate, what=f"{name} FD gate")
    fd_passed = None if fd is None else bool(
        fd.get("PASSES_FD_GATE", fd.get("PASSES", False)))
    stamp = _read_json(ref.build_stamp, what=f"{name} build stamp")
    reasons = []

    # HALT D, part (a): the FD gate is REQUIRED unless the object's own class says it
    # has no analogue, and only a CLASS object may say that (`VectorRef` refuses the
    # claim from an EGV at construction). So an entropy-gradient vector still hits the
    # exact same requirement it hit before the ruling — including a class-pilot column
    # that forgets to declare its class.
    waived = ref.fd_gate_not_applicable
    if require_fd_gate and not waived and fd_passed is not True:
        reasons.append("FD gate absent or not PASSED (§9 item 3)")

    # HALT D, the two bases. An EGV's basis IS the corpus manifest, and the check is
    # unchanged. A CLASS object's basis is the RULED CONTRAST SET, and its builder
    # stamps that sha under `set.sha256` — it carries no corpus vintage at all,
    # because a CAA vector is not "of" a corpus. Checking it against the corpus sha is
    # the two-bases defect; checking it against the DECLARED ruled set sha is the fix.
    vector_basis: Optional[dict] = None
    if ref.is_class_vector:
        declared = ref.vector_basis          # required by VectorRef's own validator
        built_from = None if stamp is None else (stamp.get("set") or {}).get("sha256")
        vector_basis = {"kind": declared.kind, "declared_sha256": declared.sha256,
                        "build_stamp_sha256": built_from,
                        "provenance": declared.provenance or None}
        vintage = None
        if built_from is None:
            reasons.append(
                "class vector's build stamp names no contrast set (`set.sha256`) — "
                "the VECTOR basis is a hole, and a hole is not an agreement "
                "(§9 item 2)")
        elif built_from != declared.sha256:
            reasons.append(
                f"class vector built from contrast set {built_from[:12]}…, not the "
                f"RULED set {declared.sha256[:12]}… the spec declares (§9 item 2: "
                "MIXED basis)")
    else:
        vintage = None if stamp is None else stamp.get("corpus_manifest_sha256")
        if vintage is None:
            reasons.append("build stamp carries no corpus vintage (§9 item 2: a hole "
                           "is a hole, not an agreement)")
        elif vintage != corpus_sha_of_record:
            reasons.append(f"built at basis {vintage[:12]}…, not the basis of record "
                           f"{corpus_sha_of_record[:12]}… (§9 item 2)")
    return ArtifactRow(
        name=name, role=role, path=str(ref.npz), present=True, sha256=sha,
        expected_sha256=ref.sha256, sha_verified=verified, corpus_vintage=vintage,
        fd_gate_passed=fd_passed, cells_at_risk=0 if not reasons else cells_at_risk,
        ready=not reasons, blocking_reason="; ".join(reasons) or None,
        vector_class=(ref.vector_class if ref.is_class_vector else None),
        fd_gate_not_applicable=(True if waived else None),
        vector_basis=vector_basis)


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
        basis_ok = band.recipe.corpus_sha == corpus_sha_of_record
        return ArtifactRow(
            name=name, role=role, path=None, present=False,
            cells_at_risk=0 if basis_ok else cells_at_risk,
            ready=False, constructible=basis_ok,
            blocking_reason=(
                f"no banked members; CONSTRUCTIBLE from the RULED recipe "
                f"({band.recipe.recipe_of_record}: {band.recipe.n_members} members at "
                f"dim {band.recipe.dim}, basis {band.recipe.corpus_sha[:12]}…, "
                f"band_kind {band.recipe.band_kind}) — pass --construct-bands to build "
                "it, which is a bank-content act and not the default"
                if basis_ok else
                f"recipe is anchored to basis {band.recipe.corpus_sha[:12]}…, not the "
                f"basis of record {corpus_sha_of_record[:12]}… — a cross-basis band is "
                "REFUSED, never silently reused (§9 item 2)"))
    return ArtifactRow(
        name=name, role=role, path=None, present=False, cells_at_risk=cells_at_risk,
        ready=False,
        blocking_reason=f"no banked {band.family} members and no construction recipe. "
                        "The random bands have never been built for any node as files "
                        "(the v1 load_axes banks were the precedent and have no "
                        "successor) — this is the OWED object, reported, not "
                        "improvised.")


def _map_row(name: str, tm: Optional[TransportMapRef], *, cells_at_risk: int,
             corpus_sha_of_record: str) -> ArtifactRow:
    """One transport-map fit's from-disk row — a direct fit or one composed leg.

    An ABSENT fit is a row, never an exception: that is exactly the state a
    PENDING-FIT column is in (the 4 hub→race-member columns, B1), and the census's
    job is to report it as OWED with the count of cells it kills.
    """
    if tm is None or not tm.fit.exists():
        return ArtifactRow(
            name=name, role="transport_map", path=str(tm.fit) if tm else None,
            present=False, cells_at_risk=cells_at_risk, ready=False,
            blocking_reason=("absent from disk — the fit is OWED; no placeholder map "
                             "is ever substituted") if tm else "not named in the spec")
    try:
        sha, verified = verify_sha(tm.fit, tm.sha256, what="transport map fit")
    except ArtifactShaMismatch as exc:
        return ArtifactRow(
            name=name, role="transport_map", path=str(tm.fit), present=True,
            sha256=sha256_file(tm.fit), expected_sha256=tm.sha256, sha_verified=False,
            cells_at_risk=cells_at_risk, ready=False, blocking_reason=str(exc))
    blocking = None
    if tm.corpus_vintage is None:
        blocking = ("the map's corpus vintage is unrecorded — §2.8 requires the "
                    "corpus vintage OF THE MAP on every stamp")
    elif tm.corpus_vintage != corpus_sha_of_record:
        blocking = (f"map fit at basis {tm.corpus_vintage[:12]}…, not the basis of "
                    f"record {corpus_sha_of_record[:12]}… (§9 item 2: MIXED vintage)")
    return ArtifactRow(
        name=name, role="transport_map", path=str(tm.fit), present=True, sha256=sha,
        expected_sha256=tm.sha256, sha_verified=verified,
        corpus_vintage=tm.corpus_vintage,
        cells_at_risk=0 if not blocking else cells_at_risk,
        ready=not blocking, blocking_reason=blocking)


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
        transported_at_risk = (N_TRANSPORTED_SIGNAL_CELLS
                               + N_TRANSPORTED_BAND_CELLS
                               + (N_SIGMA_BESIDE_CELLS
                                  if spec.source_band.sigma_beside is not None else 0))
        if spec.composed_map is not None:
            # Two legs, two rows. A composed object is only as staged as its WEAKER
            # leg, so each leg reports the SAME kill count rather than half of it.
            rows.append(_map_row(
                f"composed leg hub→source ({spec.composed_map.hub_key}→"
                f"{spec.source_key}, ridden REV)", spec.composed_map.hub_to_source,
                cells_at_risk=transported_at_risk,
                corpus_sha_of_record=spec.corpus_sha_of_record))
            rows.append(_map_row(
                f"composed leg hub→target ({spec.composed_map.hub_key}→"
                f"{spec.node_key}, ridden FWD)", spec.composed_map.hub_to_target,
                cells_at_risk=transported_at_risk,
                corpus_sha_of_record=spec.corpus_sha_of_record))
        else:
            rows.append(_map_row(
                "transport map fit", spec.transport_map,
                cells_at_risk=transported_at_risk,
                corpus_sha_of_record=spec.corpus_sha_of_record))

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
def planned_cell_count(spec: BankSpec, *, of_record_only: bool = False) -> int:
    """The cell arithmetic this spec plans — computed, never narrated.

    `of_record_only=True` returns the column WITHOUT its Σ-beside cells: the count the
    §4.1/§5.1 table fixes and the one §7's expected-N guard asserts on the column of
    record. Keeping the two counts separate is what stops a beside from filling a hole
    (B4: never a gate input, and expected-N is a gate).
    """
    n = N_BASELINE_CELLS
    if spec.include_calibration:
        n += N_CALIBRATION_SIGNAL_CELLS + N_CALIBRATION_BAND_CELLS
    if spec.include_transported:
        n += N_TRANSPORTED_SIGNAL_CELLS + N_TRANSPORTED_BAND_CELLS
    if spec.include_naive:
        n += N_NAIVE_CELLS
    if not of_record_only:
        n += planned_sigma_beside_count(spec)
    return n


def planned_sigma_beside_count(spec: BankSpec) -> int:
    """B4's 6 cells per DESIGNATED row, counted from the designations the spec carries."""
    n = 0
    if spec.include_calibration and spec.native_band.sigma_beside is not None:
        n += N_SIGMA_BESIDE_CELLS
    if spec.include_transported and spec.source_band.sigma_beside is not None:
        n += N_SIGMA_BESIDE_CELLS
    return n


def _ladder_cells(vector_key: str, site: int, *, kind: CellKind,
                  band_family: Optional[Literal["Rband", "gRband"]],
                  provenance: str, npz: Optional[str], n: int,
                  norm: float) -> list[tuple[CellSpec, float]]:
    return apply_dose_ladder(
        vector_key, site, per_token_median_resid_norm=norm, kind=kind,
        band_family=band_family, vector_npz=npz, vector_provenance=provenance, n=n)


def _sigma_beside_cells(recipe: SigmaBandRecipe, site: int, *, kind: CellKind,
                        provenance: str, npz: Optional[str], n: int,
                        norm: float) -> list[tuple[CellSpec, float]]:
    """B4's Σ-beside cells for ONE designated row: 3 members × the SCORING doses.

    The designation is re-asserted here — the guard already fires inside
    `build_sigma_band`, but that only runs under `--construct-bands`, and a spec that
    reached `--census`/`plan_cells` with an undesignated Σ recipe would otherwise
    ANNOUNCE cells that could never be built. Refusing at plan time is what makes the
    designation part of the engine-facing contract rather than of the builder alone.
    """
    assert_sigma_beside_designated(recipe)
    assert_no_lesion_recipe(provenance, SIGMA_BAND_KIND)
    out: list[tuple[CellSpec, float]] = []
    for i in range(1, recipe.n_members + 1):
        key = f"{SIGMA_BAND_KIND}{i}"
        for frac in SIGMA_BESIDE_DOSES:
            out.append((CellSpec(
                cell_id=CELL_ID_TEMPLATE.format(vector_key=key, site=site, frac=frac),
                kind=kind, vector_key=key, site=site, alpha_frac=frac, n=n,
                band_family=SIGMA_BAND_KIND, vector_npz=npz,
                vector_provenance=provenance), frac * norm))
    if len(out) != N_SIGMA_BESIDE_CELLS:                          # pragma: no cover
        raise ExpectedNShortfall(
            f"Σ-beside {recipe.designation}: planned {len(out)} cells, B4 says "
            f"{N_SIGMA_BESIDE_CELLS} ({recipe.n_members} members × "
            f"{len(SIGMA_BESIDE_DOSES)} scoring doses)")
    return out


def _class_of(ref: Optional[VectorRef]) -> dict:
    """HALT D: the class declaration one HALF of a column inherits, as kwargs.

    A cell rides exactly one object, and which object it is follows from the half it
    belongs to: the calibration half rides the node's OWN vector, the transported and
    naive halves ride the SOURCE's. The half's band members and its Σ-beside inherit
    the same declaration on purpose — they are the controls FOR that object, they
    carry the same vector custody in the stamp, and a class column has no FD gate
    anywhere in it, so a band cell left at the default would demand a gate that does
    not exist while sitting beside a signal cell that is allowed none.

    An absent ref (a calibration-only column has no source vector) is the object of
    record's default, which is what every pre-ruling column carries.
    """
    if ref is None or not ref.is_class_vector:
        return {}
    return {"vector_class": ref.vector_class, "vector_basis": ref.vector_basis}


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
    # HALT D: each half's class declaration, resolved once from the spec's own refs.
    cal_class = _class_of(spec.native_vector)
    src_class = _class_of(spec.source_vector)
    cells: list[StagedCell] = [StagedCell(spec=baseline_cell(spec.site, n=n),
                                          **cal_class)]

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
                                    vector_sha256=vector_sha256, **cal_class))
        for i in range(1, N_BAND_MEMBERS + 1):
            key = f"Rband{i}"
            prov = (f"{spec.node_key} OWN native matched-norm random band member {key} "
                    f"at L{spec.site} — §4.1's control asks whether the SITE actuates, "
                    "never whether the transport carries")
            for c, a in _ladder_cells(key, spec.site, kind="calibration_band",
                                      band_family="Rband", provenance=prov,
                                      npz=vector_npz, n=n, norm=norm):
                cells.append(StagedCell(spec=c, provisional_alpha=a,
                                        vector_sha256=vector_sha256, **cal_class))
        if spec.native_band.sigma_beside is not None:
            sig = spec.native_band.sigma_beside
            sprov = (f"Σ-BESIDE ({sig.designation}) — {spec.node_key}'s own site "
                     f"covariance at L{spec.site}: unit(Σ^{{1/2}}g) beside the Rband "
                     "at the SCORING doses. A stricter null quoted BESIDE the null of "
                     "record; NEVER the null of record and never an input to any gate "
                     "(pre-statement §2 / O-2; Luxia's B4 ruling 2026-08-04)")
            for c, a in _sigma_beside_cells(sig, spec.site, kind="calibration_band",
                                            provenance=sprov, npz=vector_npz, n=n,
                                            norm=norm):
                cells.append(StagedCell(spec=c, provisional_alpha=a,
                                        vector_sha256=vector_sha256, **cal_class))

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
                                    naive_row=naive_row, vector_sha256=vector_sha256,
                                    **src_class))
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
                                        vector_sha256=vector_sha256, **src_class))
        if spec.source_band.sigma_beside is not None:
            sig = spec.source_band.sigma_beside
            sprov = (f"Σ-BESIDE ({sig.designation}) — {src}'s site covariance, drawn "
                     f"in the SOURCE's space and carried through the SAME {fam} map "
                     f"→ {spec.node_key} L{spec.site}, at the SCORING doses. A "
                     "stricter null quoted BESIDE the transported null of record; "
                     "NEVER the null of record and never an input to any gate "
                     "(pre-statement §2 / O-2; Luxia's B4 ruling 2026-08-04)")
            for c, a in _sigma_beside_cells(sig, spec.site, kind="transported_band",
                                            provenance=sprov, npz=vector_npz, n=n,
                                            norm=norm):
                cells.append(StagedCell(spec=c, provisional_alpha=a,
                                        transport_map=transport_map_stamp,
                                        naive_row=naive_row,
                                        vector_sha256=vector_sha256, **src_class))

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
                                    vector_sha256=vector_sha256, **src_class))

    expected = planned_cell_count(spec)
    if len(cells) != expected:
        raise ExpectedNShortfall(
            f"{spec.node_key}: planned {len(cells)} cells, the §4.1+§5.1 table says "
            f"{expected} (§9 item 10 / M23: a count one short is a rake, not a "
            "rounding)")
    # …and again on the column MINUS its besides, so a Σ cell can never be the reason
    # the total came out right (B4: never a gate input, and expected-N is a gate).
    of_record = [c for c in cells if not c.spec.is_beside]
    expected_of_record = planned_cell_count(spec, of_record_only=True)
    if len(of_record) != expected_of_record:                      # pragma: no cover
        raise ExpectedNShortfall(
            f"{spec.node_key}: planned {len(of_record)} cells OF RECORD, the "
            f"§4.1+§5.1 table says {expected_of_record} "
            f"({len(cells) - len(of_record)} Σ-beside cells were staged beside them)")
    return cells


# ---------------------------------------------------------------- site cross-check
def _registered_robustness_site(node_key: str) -> Optional[int]:
    """The node's REGISTERED robustness site, or None — read, never inferred.

    Its own function so the refusal path can consult the registry without the
    §4.2 module's import being load-bearing for an ordinary HALT message.
    """
    try:
        from metabasis.scripts.actuation_calibration import ROBUSTNESS_SITES
    except ImportError:                                       # pragma: no cover
        return None
    return dict(ROBUSTNESS_SITES).get(node_key)


def site_cross_check(node_key: str, site: int, *,
                     site_role: Optional[SiteRole] = None) -> dict:
    """§2.8's `SITES`/`SITE_OF_RECORD` cross-check, from the LIVE registries.

    Reports rather than raises for an unknown node (the registries do not carry every
    key), but RAISES when they carry the node and disagree with the site being staged:
    a retired site (gemma L36, 70B L17) must never resolve by default (§9 item 4).

    HALT C (Luxia's ruling, 2026-08-05 morning). `site_role` is the DECLARED role the
    site is being staged in, and it is the ONLY thing that can admit a site other than
    the site of record:

      * `None` — no role declared, the default, and what every spec written before the
        ruling carries — behaves EXACTLY as this function always behaved, down to the
        returned dict's keys: any site ≠ `SITE_OF_RECORD` HALTs. The ruling widened
        what a spec may SAY, never what silence means.
      * `"site_of_record"` — the same checks, said out loud. The only difference from
        `None` is that the returned dict records the role.
      * `"robustness_site"` — §4.2's frozen remedy (i). Admissible only when the node
        has a REGISTERED robustness site, the staged site IS it, and it is on the
        fixed fit grid. `actuation_calibration.assert_no_site_fishing` is then called
        on the same (node, site) as the final word, so the two layers cannot drift:
        that module already implemented this admissibility and this guard was the one
        layer that lacked it, which is what blocked gemma3-27b's L41 column.

    A declared role never LOOSENS the grid check and never invents a site: the role
    selects which registry row is allowed to license the site, and every row comes
    from the campaign's registries, never from behavioral evidence (§4.2's closing
    prohibition).
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
    # A role appears in the record ONLY when one was declared. An undeclared role is
    # not "site_of_record" written in invisible ink — it is silence, and the stamp of
    # a column staged before the ruling must not grow a key it never carried.
    if site_role is not None:
        out["site_role"] = site_role

    if site_role == "robustness_site":
        try:
            from metabasis.scripts.actuation_calibration import (
                ROBUSTNESS_SITES, assert_no_site_fishing)
        except ImportError as exc:                            # pragma: no cover
            raise RobustnessSiteNotRegistered(
                f"{node_key}: a robustness-site column cannot be staged without the "
                f"§4.2 registry ({exc}) — the role is admitted by a REGISTERED row "
                "or not at all") from exc
        registered = dict(ROBUSTNESS_SITES).get(node_key)
        if registered is None:
            raise RobustnessSiteNotRegistered(
                f"{node_key}: site_role='robustness_site' declared for L{site}, but "
                f"this node has NO registered robustness site (registered: "
                f"{sorted(dict(ROBUSTNESS_SITES))}). §4.2's remedy (i) re-calibrates "
                "at a REGISTERED site — a role string cannot register one.")
        if site != registered:
            raise RobustnessSiteNotRegistered(
                f"{node_key}: site_role='robustness_site' declared for L{site}, but "
                f"the REGISTERED robustness site is L{registered}. Sites come from "
                "curves and Luxia's rulings, never from a spec field (§4.2).")
        if grid and site not in grid:
            raise RobustnessSiteOffGrid(
                f"{node_key}: robustness site L{site} is not on the fixed fit grid "
                f"{grid} — the two registries disagree, which is a desk HALT rather "
                "than a preference (§9 item 4).")
        # The final word is the module that already implements this admissibility,
        # called on the same (node, site) — so a future change to §4.2's rule lands
        # in ONE place and this guard cannot silently diverge from it.
        assert_no_site_fishing(
            node_key, site,
            evidence=f"spec declares site_role='robustness_site' at L{site}; "
                     f"admitted by the REGISTERED robustness-site row, never by any "
                     f"behavioral reading")
        out["robustness_site_registered"] = registered
        out["site_role_rule"] = (
            "§4.2 remedy (i): a REGISTERED robustness site, cross-checked against "
            "actuation_calibration.ROBUSTNESS_SITES ∧ fit_transport_maps.SITES and "
            "admitted by assert_no_site_fishing")
        out["agrees"] = True
        return out

    if ruled is not None and site != ruled:
        # The remedy pointer is appended for EXACTLY one case — an undeclared column
        # standing on the node's own REGISTERED robustness site, i.e. the enactor who
        # forgot the declaration — so the refusal text every other caller sees
        # (including every column ever banked) is byte-identical to the pre-ruling
        # engine's. A hint is worth a HALT round-trip; a moved message is not.
        hint = ""
        if site_role is None and _registered_robustness_site(node_key) == site:
            hint = (" If this is §4.2's remedy (i), the spec must SAY so: declare "
                    "site_role='robustness_site' and the REGISTERED robustness site "
                    "is admitted (Luxia's ruling, 2026-08-05). Silence is refused.")
        raise SiteNotOfRecord(
            f"{node_key}: staging cells at L{site} but SITE_OF_RECORD is L{ruled} "
            "(§9 item 4). A retired site must never resolve by default." + hint)
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

    cross = site_cross_check(spec.node_key, spec.site, site_role=spec.site_role)
    vectors: dict[str, np.ndarray] = {}
    magnitudes: dict[str, float] = {}
    constructions: dict[str, str] = {}
    #: the exact seed material behind every CONSTRUCTED band member, so the stamp
    #: carries the string the draw was seeded from rather than a template to re-render.
    band_seed_materials: dict[str, str] = {}
    #: the Σ-beside's estimator facts, per half, when one was designated and built.
    sigma_facts: dict[str, dict] = {}

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
            if band.recipe.corpus_sha != spec.corpus_sha_of_record:
                raise ConstructionRefused(
                    f"native band recipe is anchored to {band.recipe.corpus_sha[:12]}… "
                    f"but this bank's basis of record is "
                    f"{spec.corpus_sha_of_record[:12]}… — a cross-basis band is refused")
            for key, v in build_random_band(band.recipe, "Rband").items():
                if v.size != lever.size:
                    raise DimensionMismatch(
                        f"{key}: recipe dim {v.size} != lever dim {lever.size}")
                vectors[key] = v.astype(np.float32)
                constructions[key] = RANDOM_BAND_CONSTRUCTION
                band_seed_materials[key] = band.recipe.seed_material(
                    int(key.removeprefix("Rband")))
        if band.sigma_beside is not None and construct_bands:
            sband_vectors, sigma_facts["native"] = build_sigma_band(band.sigma_beside)
            for key, v in sband_vectors.items():
                vectors[key] = v.astype(np.float32)
                constructions[key] = SIGMA_BAND_CONSTRUCTION

    # --- §5's transported half ------------------------------------------------
    tm_stamp: Optional[dict] = None
    if spec.include_transported:
        if spec.source_vector is None or spec.map_of_record is None:  # pragma: no cover
            raise SpecError("the transported half needs a source vector and a map")
        from metabasis.scripts.fit_transport_maps import load_transport_map

        if spec.composed_map is not None:
            comp = spec.composed_map
            leg_in = load_transport_map(comp.hub_to_source.fit)
            leg_out = load_transport_map(comp.hub_to_target.fit)
            in_dir, out_dir_ = comp.hub_to_source.direction, comp.hub_to_target.direction
            # `hub_to_source` is a hub→source leg ridden BACKWARDS and `hub_to_target`
            # a hub→target leg ridden FORWARDS, each composed with whatever orientation
            # the fit was banked in (§8.3). `_flip` turns a banked orientation into the
            # one this hop needs, so a mirror-banked leg composes identically.
            def _flip(d: str) -> str:
                return "rev" if d == "fwd" else "fwd"

            def carry(v: np.ndarray) -> np.ndarray:
                at_hub = leg_in.transport(v, direction=_flip(in_dir))
                return leg_out.transport(at_hub, direction=out_dir_)

            direction = "composed"
        else:
            tmap = load_transport_map(spec.transport_map.fit)
            direction = spec.transport_map.direction

            def carry(v: np.ndarray) -> np.ndarray:
                return tmap.transport(v, direction=direction)

        src = load_npz_vector(spec.source_vector)
        gkey = f"{spec.transported_prefix}entropy_gradient"
        gv, mag = transport_vector(carry, src, key=gkey)
        vectors[gkey] = gv.astype(np.float32)
        magnitudes[gkey] = round(mag, 6)
        if spec.composed_map is not None:
            constructions[gkey] = (
                f"COMPOSED two-hop transport({spec.composed_map.family}): "
                f"{spec.source_key} →(rev {spec.composed_map.hub_key} leg)→ hub "
                f"→(fwd)→ {spec.node_key}, then unit — the same composition "
                "`read_composed_predictions.composed_exchange_rate` makes â_comp from")
        else:
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
            if sband.recipe.corpus_sha != spec.corpus_sha_of_record:
                raise ConstructionRefused(
                    f"transported band recipe is anchored to "
                    f"{sband.recipe.corpus_sha[:12]}… but this bank's basis of record "
                    f"is {spec.corpus_sha_of_record[:12]}… — refused")
            # The members are drawn in the SOURCE's space (§5.1: the control travels
            # the same road as the signal) and keyed `gRband*` on arrival; the seed
            # material names the CELL, which is why the recipe carries both.
            built = build_random_band(sband.recipe, "gRband")
            for i in range(1, sband.recipe.n_members + 1):
                key = f"{spec.transported_prefix}Rband{i}"
                v = built[f"gRband{i}"]
                if v.size != src.size:
                    raise DimensionMismatch(
                        f"{key}: recipe draws at dim {v.size} but the SOURCE vector is "
                        f"{src.size} — a transported band must start in the source's "
                        "space")
                gvi, magi = transport_vector(carry, v, key=key)
                vectors[key] = gvi.astype(np.float32)
                magnitudes[key] = round(magi, 6)
                constructions[key] = ("random_band (source-side, constructed) then "
                                      + constructions[gkey])
                band_seed_materials[key] = sband.recipe.seed_material(i)
        if sband.sigma_beside is not None and construct_bands:
            sig_vectors, sigma_facts["transported"] = build_sigma_band(
                sband.sigma_beside)
            for key, v in sig_vectors.items():
                if v.size != src.size:
                    raise DimensionMismatch(
                        f"{key}: Σ-beside drawn at dim {v.size}, source vector is "
                        f"{src.size} — the Σ-beside for a TRANSPORTED cell is drawn "
                        "in the source's space and carried through the same map")
                gvi, magi = transport_vector(carry, v, key=key)
                vectors[key] = gvi.astype(np.float32)
                magnitudes[key] = round(magi, 6)
                constructions[key] = (SIGMA_BAND_CONSTRUCTION + " then "
                                      + constructions[gkey])
        if spec.include_naive:
            nkey = f"{NAIVE_KEY_PREFIX}_entropy_gradient"
            vectors[nkey] = coordinate_identify(src, target_dim, key=nkey
                                                ).astype(np.float32)
            constructions[nkey] = NAIVE_CONSTRUCTION
        if spec.composed_map is not None:
            comp = spec.composed_map
            tm_stamp = {
                "road": "composed",
                "hub_key": comp.hub_key, "hub_site": comp.hub_site,
                "family": comp.family, "arm": comp.arm, "direction": "composed",
                "legs": [
                    {"role": "hub_to_source", "ridden": "rev",
                     "fit_path": str(comp.hub_to_source.fit),
                     "fit_sha256": sha256_file(comp.hub_to_source.fit),
                     "banked_direction": comp.hub_to_source.direction,
                     "corpus_vintage": comp.hub_to_source.corpus_vintage},
                    {"role": "hub_to_target", "ridden": "fwd",
                     "fit_path": str(comp.hub_to_target.fit),
                     "fit_sha256": sha256_file(comp.hub_to_target.fit),
                     "banked_direction": comp.hub_to_target.direction,
                     "corpus_vintage": comp.hub_to_target.corpus_vintage},
                ],
                # §2.8 wants ONE fit sha per stamp field; a composed road has two, so
                # the field carries a digest OVER the two in leg order and the legs are
                # listed above. A stamp that named only one leg would be unreproducible.
                "fit_sha256": hashlib.sha256(
                    (sha256_file(comp.hub_to_source.fit) + "|"
                     + sha256_file(comp.hub_to_target.fit)).encode()).hexdigest(),
                "fit_path": (f"{comp.hub_to_source.fit} (rev) ∘ "
                             f"{comp.hub_to_target.fit} (fwd)"),
                "corpus_vintage": (comp.hub_to_target.corpus_vintage
                                   if (comp.hub_to_source.corpus_vintage
                                       == comp.hub_to_target.corpus_vintage) else None),
                "selection_rule": comp.selection_rule or None,
                "a_hat_hub_to_source": comp.a_hat_hub_to_source,
                "a_hat_hub_to_target": comp.a_hat_hub_to_target,
                "a_hat_product": comp.a_hat_product,
                "product_rank": comp.product_rank,
            }
        else:
            tm_stamp = {
                "road": "direct",
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
        "composed_leg_hub_to_source": (spec.composed_map.hub_to_source.corpus_vintage
                                       if spec.composed_map else None),
        "composed_leg_hub_to_target": (spec.composed_map.hub_to_target.corpus_vintage
                                       if spec.composed_map else None),
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
                       battery_item_set_sha256=BATTERY_ITEM_SET_SHA256,
                       band_seed_materials=band_seed_materials,
                       sigma_beside=sigma_facts)

    doc = CellsDocument(
        node_key=spec.node_key, arm=spec.arm, site=spec.site,
        site_role=spec.site_role,
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
               battery_item_set_sha256: str,
               band_seed_materials: Optional[dict[str, str]] = None,
               sigma_beside: Optional[dict[str, dict]] = None) -> dict:
    """The staging half of §2.8's custody, as one document.

    Everything here is a fact this module established from disk or constructed itself.
    Nothing is defaulted into existence: an unknown is written as `None` and shows up
    in the desk's OWED column (§10), which is the point.
    """
    fd = _read_json(spec.native_vector.fd_gate if spec.native_vector else None,
                    what="native FD gate")
    build = _read_json(spec.native_vector.build_stamp if spec.native_vector else None,
                       what="native build stamp")
    stamp = {
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
        "n_cells_of_record": sum(1 for c in cells if not c.spec.is_beside),
        "n_sigma_beside_cells": sum(1 for c in cells if c.spec.is_beside),
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
        "transport_road": (transport_map or {}).get("road"),
        # A composed road's two legs, verbatim — the single `transport_map_fit_sha256`
        # above is a digest OVER them and cannot be resolved back to either leg.
        "composed_map": (transport_map
                         if (transport_map or {}).get("road") == "composed" else None),
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
            "SigmaBand": "the Σ-shaped BESIDE on the two designated cells (O-2) — "
                         "never the null of record, never inside a gate",
            "why_distinct": "conflating them is the fastest way to make a null "
                            "uninterpretable (§4.1); `band_kind` in the seed material "
                            "keeps all three phase-separate",
        },
        "band_recipe": {
            "recipe_of_record": BAND_RECIPE_OF_RECORD,
            "ruling": "Luxia 2026-08-04 (session 12) — ISOTROPIC + Σ BESIDE; "
                      "PRESTATEMENT-behavioral-harness-ceremony-2026-08-04.md §2",
            "seed_material_template": RANDOM_BAND_SEED_TEMPLATE,
            "seed_material_template_digest": RANDOM_BAND_SEED_TEMPLATE_DIGEST,
            "seed_derivation": "seed_int = int.from_bytes(sha256(material)[:8], 'big') "
                               "(brief §2.3's scheme, M25-safe)",
            "construction": RANDOM_BAND_CONSTRUCTION,
            "n_members": N_BAND_MEMBERS,
            "constructed_member_seed_materials": band_seed_materials or None,
            "note": "an EMPTY constructed-member map means this bank consumed BANKED "
                    "members, not constructed ones — the two are different provenances "
                    "and the stamp says which",
        },
        "sigma_beside": sigma_beside or None,
        "sigma_beside_cells": {
            "designations": list(spec.sigma_beside_designations) or None,
            "n_cells_per_designated_row": N_SIGMA_BESIDE_CELLS,
            "doses": list(SIGMA_BESIDE_DOSES),
            "ruling": "B4 (Luxia, 2026-08-04): 6 Σ-beside cells at the SCORING doses "
                      "per designated row; the CellSpec band-family widening authorized "
                      "as a beside family only",
            "derivation": f"§4.1/§5.1 pair every band MEMBER with every dose the "
                          f"signal it controls rides ({N_BAND_MEMBERS} × "
                          f"{len(DOSE_LADDER)} = {N_CALIBRATION_BAND_CELLS} for the "
                          f"isotropic band). The Σ band is quoted at the SCORING doses "
                          f"only, so the same convention over {list(SCORING_DOSES)} "
                          f"gives {N_BAND_MEMBERS} × {len(SIGMA_BESIDE_DOSES)} = "
                          f"{N_SIGMA_BESIDE_CELLS}",
            "gate_discipline": "these cells enter NO gate: not the §2.7 replay gate "
                               "(no stratum), not the §4 actuation criteria (Rband "
                               "only, 18-cell arity), not the §7 expected-N of record "
                               "(counted separately)",
        } if spec.sigma_beside_designations else None,
        "lesion_recipe_law": "§5.4: asserted at CONSTRUCTION — the admissible set is "
                             "closed (transport, coordinate_identify, random_band, "
                             "identity) and every forbidden name is refused",
        "provisional_alpha_note": PROVISIONAL_ALPHA_NOTE,
        "stamp_field_partition": {"staged_here": list(STAGED_STAMP_FIELDS),
                                  "filled_in_job": list(IN_JOB_STAMP_FIELDS)},
    }
    # HALT D. THE TWO BASES, BOTH NAMED — and written ONLY when the column actually
    # carries a class object, so an entropy-gradient column's stamp is byte-identical
    # to the pre-ruling engine's (the flag-absent condition of the re-freeze).
    class_refs = [(half, ref) for half, ref in
                  (("native_vector", spec.native_vector),
                   ("source_vector", spec.source_vector))
                  if ref is not None and ref.is_class_vector]
    if class_refs:
        stamp["vector_class"] = {half: ref.vector_class for half, ref in class_refs}
        stamp["vector_basis"] = {
            half: {**ref.vector_basis.model_dump(), "vector_key": ref.key}
            for half, ref in class_refs}
        stamp["generation_basis"] = {
            "kind": "corpus-manifest",
            "sha256": stamp["corpus_manifest_sha256"] or spec.corpus_sha_of_record,
            "provenance": "the corpus manifest the PROMPTS and GENERATIONS are of "
                          "(§2.8's corpus_manifest_sha256) — unchanged by the class "
                          "ruling and never the class object's basis"}
        stamp["two_bases_note"] = (
            "HALT D (Luxia's ruling, 2026-08-05): a CLASS column stands on TWO bases "
            "— the GENERATION basis (the corpus manifest) and the VECTOR basis (the "
            "RULED contrast set the class object was built from). Both are named "
            "here, each under its own key. Writing the corpus sha as the vector's "
            "vintage would be false provenance; writing nothing would be a hole.")
        stamp["fd_gate_not_applicable"] = {
            half: bool(ref.fd_gate_not_applicable) for half, ref in class_refs}
        stamp["fd_gate_not_applicable_note"] = (
            "no FD-gate analogue exists for a class object — its builder's own stamp "
            "says so (build_contrast_vectors: 'build acceptance is stamp "
            "completeness + the pair-count assertion + the anchor cosines'). An "
            "ENTROPY-GRADIENT object may never make this claim: `VectorRef` refuses "
            "it at spec load and the census still demands the gate.")
    return stamp


# ---------------------------------------------------------------- example spec
def example_spec() -> dict:
    """A spec template with every field named and NO basis baked in.

    Emitted rather than documented so the shape cannot drift from the model, and
    deliberately full of obvious placeholders: a spec that ran as-is would be a bank
    built from nothing.

    A CLASS column's vector ref (HALT D) is the same object with three fields moved::

        "source_vector": {"key": "caa_formality_L26", "npz": "<PATH>/caa_….npz",
                          "build_stamp": "<PATH>/caa_…_stamps.json",
                          "vector_class": "caa",
                          "fd_gate": null, "fd_gate_not_applicable": true,
                          "vector_basis": {"kind": "contrast-set",
                                           "sha256": "<THE RULED SET SHA, 64 hex>",
                                           "provenance": "<the pin row that rules it>"}}

    — no `fd_gate` (a CAA object has no analogue), the waiver said out loud, and the
    RULED contrast set named as the VECTOR basis. The corpus manifest above stays the
    GENERATION basis; the two never share a field.
    """
    return {
        "node_key": "<TARGET-NODE-KEY>",
        "arm": "native",
        "site": 0,
        # HALT C: omit (or null) for an ordinary column at its site of record;
        # "robustness_site" for §4.2's remedy (i) at the node's REGISTERED
        # robustness site, which is the ONLY thing that admits a second site.
        "site_role": None,
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
            "recipe": {"recipe_of_record": BAND_RECIPE_OF_RECORD,
                       "corpus_sha": "<THE BASIS OF RECORD>",
                       "node_key": "<TARGET-NODE-KEY>", "arm": "native", "site": 0,
                       "band_kind": "Rband", "vector_key": "entropy_gradient",
                       "draw_space_key": "<TARGET-NODE-KEY>", "dim": 0,
                       "n_members": N_BAND_MEMBERS}},
        "source_vector": {
            "key": "entropy_gradient", "npz": "<PATH>/entropy_gradient_<hub>_L<site>.npz",
            "fd_gate": "<PATH>/..._fd_gate.json",
            "build_stamp": "<PATH>/..._stamps.json",
            # HALT D: omit for the entropy-gradient object of record. A CLASS object
            # ("caa" / "repeng_pca") declares its class, waives the FD gate it has no
            # analogue for, and NAMES the ruled contrast set it was built from — the
            # VECTOR basis, which is never the corpus manifest.
            "vector_class": EGV_VECTOR_CLASS,
            "fd_gate_not_applicable": False,
            "vector_basis": None},
        "source_band": {
            "family": "gRband", "members": [],
            "recipe": {"recipe_of_record": BAND_RECIPE_OF_RECORD,
                       "corpus_sha": "<THE BASIS OF RECORD>",
                       "node_key": "<TARGET-NODE-KEY>", "arm": "native", "site": 0,
                       "band_kind": "gRband", "vector_key": "gentropy_gradient",
                       "draw_space_key": "<SOURCE-NODE-KEY>", "dim": 0,
                       "n_members": N_BAND_MEMBERS},
            "sigma_beside": "<null, or a SigmaBandRecipe on ONE of the two O-2 "
                            "designated cells: " + ", ".join(
                                sorted(SIGMA_BESIDE_DESIGNATIONS)) + ">"},
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

        # ---- 2. the RULED random band recipe (M25-safe, reproducible) ---------
        print("== selftest 2: the constructed random band — the RULED recipe (M25) ==")
        recipe = RandomBandRecipe(
            corpus_sha=basis, node_key="toy-node", arm="native", site=7,
            band_kind="Rband", vector_key="entropy_gradient",
            draw_space_key="toy-node", dim=d_tgt)
        band1 = build_random_band(recipe, "Rband")
        band2 = build_random_band(recipe, "Rband")
        check("a band is 3 members keyed Rband1..3 (§4.1's naming)",
              sorted(band1) == ["Rband1", "Rband2", "Rband3"])
        check("the band is bitwise reproducible from the recipe alone (BUILD-TWICE)",
              all(band1[k].tobytes() == band2[k].tobytes() for k in band1),
              "byte equality, not allclose — the recipe is the artifact")
        check("the seed material is the RULED template, verbatim",
              recipe.seed_material(1)
              == f"{basis}|toy-node|native|L7|Rband|entropy_gradient|member01"
              and RANDOM_BAND_SEED_TEMPLATE
              == "{corpus_sha}|{node_key}|{arm}|L{site}|{band_kind}|{vector_key}"
                 "|member{m:02d}",
              RANDOM_BAND_SEED_TEMPLATE_DIGEST[:12] + "…")
        check("members are distinct directions, not one vector three times",
              len({v.tobytes() for v in band1.values()}) == 3)
        check("every member is UNIT (the hook re-normalizes; orientation is the object)",
              all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-12 for v in band1.values()))
        other = build_random_band(
            recipe.model_copy(update={"corpus_sha": other_basis}), "Rband")
        check("a DIFFERENT basis gives a different band (no silent cross-basis reuse)",
              not any(np.array_equal(band1[k], other[k]) for k in band1))
        grecipe = recipe.model_copy(update={"band_kind": "gRband",
                                            "draw_space_key": "toy-source",
                                            "vector_key": "gentropy_gradient"})
        gband = build_random_band(grecipe, "gRband")
        check("gRband is the transported family AND a different phase of the seed",
              sorted(gband) == ["gRband1", "gRband2", "gRband3"]
              and not any(np.array_equal(gband[f"gRband{i}"], band1[f"Rband{i}"])
                          for i in (1, 2, 3)),
              "`band_kind` in the seed material keeps native and transported bands "
              "phase-separate (§4.1)")
        check("every seed-material field moves the band (arm, site, vector_key)",
              all(not np.array_equal(
                  build_random_band(recipe.model_copy(update=u), "Rband")["Rband1"],
                  band1["Rband1"])
                  for u in ({"arm": "raw"}, {"site": 8},
                            {"vector_key": "other_object"},
                            {"node_key": "toy-other", "draw_space_key": "toy-other"})))
        # DESCRIPTIVE, never a gate: the 3 members' mutual geometry. In d=10 a
        # random pair is nowhere near orthogonal, so this is a reported statistic,
        # not an assertion — the assertion is only that it is finite and in range.
        mem = [band1[f"Rband{i}"] for i in (1, 2, 3)]
        cosines = [abs(float(mem[a] @ mem[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
        check("member orthogonality statistics are finite and in [0,1] (DESCRIPTIVE)",
              all(np.isfinite(c) and 0.0 <= c <= 1.0 for c in cosines),
              f"|cos| pairwise at d={d_tgt}: "
              + ", ".join(f"{c:.4f}" for c in cosines)
              + f"; max {max(cosines):.4f} (E|cos| ~ sqrt(2/(pi*d)) = "
                f"{np.sqrt(2 / (np.pi * d_tgt)):.4f}) — reported, never a gate")
        big = RandomBandRecipe(
            corpus_sha=basis, node_key="toy-node", arm="native", site=7,
            band_kind="Rband", vector_key="entropy_gradient",
            draw_space_key="toy-node", dim=2048)
        bigband = build_random_band(big, "Rband")
        bigcos = [abs(float(bigband[f"Rband{a}"] @ bigband[f"Rband{b}"]))
                  for a, b in ((1, 2), (1, 3), (2, 3))]
        check("at a REAL residual dimension the members are near-orthogonal "
              "(DESCRIPTIVE)",
              all(np.isfinite(c) for c in bigcos),
              f"d=2048 |cos|: " + ", ".join(f"{c:.5f}" for c in bigcos)
              + f"; E|cos| ~ {np.sqrt(2 / (np.pi * 2048)):.5f}")
        check("an out-of-range member index is refused",
              _raises(lambda: random_band_member(recipe, 0), ValueError)
              and _raises(lambda: random_band_member(recipe, 4), ValueError))

        # ---- 2b. the refusal that lifts for EXACTLY this recipe ---------------
        print("== selftest 2b: the refusal lifts for ONE recipe and no other ==")
        check("a recipe string that is not the recipe of record is REFUSED",
              _raises(lambda: RandomBandRecipe(
                  recipe_of_record="anisotropic-something/2026-08-05",
                  corpus_sha=basis, node_key="toy-node", arm="native", site=7,
                  band_kind="Rband", vector_key="v", draw_space_key="toy-node",
                  dim=4), ValueError),
              f"only {BAND_RECIPE_OF_RECORD!r} builds")
        check("an edited CONSTRUCTION string is refused even under the ruled name",
              _raises(lambda: RandomBandRecipe(
                  corpus_sha=basis, node_key="toy-node", arm="native", site=7,
                  band_kind="Rband", vector_key="v", draw_space_key="toy-node",
                  dim=4, construction="unit(uniform draw)"), ValueError))
        check("a band asked for under the WRONG family prefix is refused",
              _raises(lambda: build_random_band(recipe, "gRband"),
                      ConstructionRefused))
        check("an Rband recipe drawing in someone else's space is refused",
              _raises(lambda: RandomBandRecipe(
                  corpus_sha=basis, node_key="toy-node", arm="native", site=7,
                  band_kind="Rband", vector_key="v", draw_space_key="toy-source",
                  dim=4), ValueError))
        check("a gRband recipe drawing in the TARGET's space is refused (§5.1)",
              _raises(lambda: RandomBandRecipe(
                  corpus_sha=basis, node_key="toy-node", arm="native", site=7,
                  band_kind="gRband", vector_key="v", draw_space_key="toy-node",
                  dim=4), ValueError))
        check("a BandRef whose family and recipe band_kind disagree is refused",
              _raises(lambda: BandRef(family="gRband", recipe=recipe), ValueError))
        check("--build with NO recipe and no banked members still refuses (BY DESIGN)",
              _raises(lambda: build_banks(BankSpec(
                  node_key="toy-node", arm="native", site=7, corpus_manifest=None,
                  include_transported=False, include_naive=False,
                  native_vector=None), construct_bands=True, write=False),
                  CensusNotReady),
              "the lifted refusal is recipe-shaped, not flag-shaped")

        # ---- 2c. the Σ-BESIDE: two designated cells and nowhere else ----------
        print("== selftest 2c: the Σ-beside is a BESIDE on two designated cells ==")
        d_sig = 16
        rng_sig = np.random.default_rng(2026)
        A = rng_sig.standard_normal((d_sig, d_sig))
        Sig = A @ A.T + np.eye(d_sig)
        ev, evec = np.linalg.eigh(Sig)
        sigma_npz = root / "sigma_toy.npz"
        np.savez(sigma_npz, evals=ev, evecs=evec,
                 mean=np.zeros(d_sig), ridge=np.float64(1e-3 * float(ev.mean())),
                 n_positions=np.int64(1234))
        node_d, arm_d, site_d, kind_d = SIGMA_BESIDE_DESIGNATIONS[
            "3b-certification-calibration"]

        def _sigma_recipe(**over: Any) -> SigmaBandRecipe:
            base = dict(
                designation="3b-certification-calibration", corpus_sha=basis,
                node_key=node_d, arm=arm_d, site=site_d, beside_band_kind=kind_d,
                vector_key="entropy_gradient", dim=d_sig, sigma_npz=sigma_npz,
                sigma_estimator="token-level residual covariance (selftest toy)",
                sigma_provenance="selftest synthetic SPD matrix")
            base.update(over)
            return SigmaBandRecipe(**base)

        sig_band, sig_facts = build_sigma_band(_sigma_recipe())
        sig_band2, _ = build_sigma_band(_sigma_recipe())
        check("the Σ band is 3 members keyed SigmaBand1..3, never Rband/gRband",
              sorted(sig_band) == ["SigmaBand1", "SigmaBand2", "SigmaBand3"])
        check("the Σ band is bitwise reproducible (BUILD-TWICE)",
              all(sig_band[k].tobytes() == sig_band2[k].tobytes() for k in sig_band))
        check("Σ members are UNIT after the Σ^{1/2} shaping",
              all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-12
                  for v in sig_band.values()))
        iso_same_cell = build_random_band(RandomBandRecipe(
            corpus_sha=basis, node_key=node_d, arm=arm_d, site=site_d,
            band_kind=kind_d, vector_key="entropy_gradient",
            draw_space_key=node_d, dim=d_sig), kind_d)
        check("the Σ band is a DIFFERENT draw from the isotropic band of the same cell",
              not any(np.array_equal(sig_band[f"SigmaBand{i}"],
                                     iso_same_cell[f"{kind_d}{i}"])
                      for i in (1, 2, 3)),
              "band_kind=SigmaBand phase-separates it")
        # the whole point of Σ-shaping: members should lean toward the top
        # eigendirection relative to isotropic draws. DESCRIPTIVE.
        top = evec[:, -1]
        lean_sigma = float(np.mean([abs(float(v @ top)) for v in sig_band.values()]))
        lean_iso = float(np.mean([abs(float(v @ top))
                                  for v in iso_same_cell.values()]))
        check("Σ shaping is visible against isotropic on the top eigendirection "
              "(DESCRIPTIVE)",
              np.isfinite(lean_sigma) and np.isfinite(lean_iso),
              f"mean |cos| with the top eigenvector: Σ {lean_sigma:.4f} vs isotropic "
              f"{lean_iso:.4f} (d={d_sig}) — reported, never a gate")
        check("the Σ stamp NAMES its estimator, its provenance and its grade",
              sig_facts["estimator"].startswith("token-level residual covariance")
              and sig_facts["GRADE"].startswith("BESIDE ONLY")
              and sig_facts["banked_ridge"] > 0.0
              and sig_facts["ridge_applied"] is True
              and "thread_config" in sig_facts)
        check("a NON-DESIGNATED cell is REFUSED for the Σ-beside (O-2)",
              _raises(lambda: build_sigma_band(_sigma_recipe(node_key="some-other-node")),
                      ConstructionRefused)
              and _raises(lambda: build_sigma_band(_sigma_recipe(site=site_d + 1)),
                          ConstructionRefused)
              and _raises(lambda: build_sigma_band(_sigma_recipe(arm="raw")),
                          ConstructionRefused)
              and _raises(lambda: build_sigma_band(
                  _sigma_recipe(beside_band_kind="gRband")), ConstructionRefused),
              "the designation is a CELL, not a recipe: node, arm, site and family all "
              "have to match")
        check("an unknown designation is REFUSED by name",
              _raises(lambda: build_sigma_band(
                  _sigma_recipe(designation="a-third-cell")), ConstructionRefused),
              f"designated: {sorted(SIGMA_BESIDE_DESIGNATIONS)}")
        check("a Σ recipe string that is not the ruled one is refused",
              _raises(lambda: _sigma_recipe(recipe_of_record="whitened/2026"),
                      ValueError))
        check("a sha mismatch on the banked Σ is a HALT (M4)",
              _raises(lambda: build_sigma_band(_sigma_recipe(sigma_sha256="0" * 64)),
                      ArtifactShaMismatch))
        check("a Σ npz with a DESCENDING spectrum is refused, not silently misread",
              _raises(lambda: build_sigma_band(_sigma_recipe(
                  sigma_npz=_write_npz(root / "sigma_desc.npz",
                                       evals=ev[::-1].copy(), evecs=evec,
                                       ridge=np.float64(1e-3)))),
                      ConstructionRefused))
        check("a Σ npz at the wrong dimension is a named DimensionMismatch",
              _raises(lambda: build_sigma_band(_sigma_recipe(dim=d_sig + 1)),
                      DimensionMismatch))
        check("an absent Σ npz is a named refusal",
              _raises(lambda: build_sigma_band(
                  _sigma_recipe(sigma_npz=root / "no_such_sigma.npz")),
                  ConstructionRefused))

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
        native_recipe = RandomBandRecipe(
            corpus_sha=corpus_sha, node_key="toy-node", arm="native", site=7,
            band_kind="Rband", vector_key="entropy_gradient",
            draw_space_key="toy-node", dim=d_tgt)
        source_recipe = RandomBandRecipe(
            corpus_sha=corpus_sha, node_key="toy-node", arm="native", site=7,
            band_kind="gRband", vector_key="gentropy_gradient",
            draw_space_key="toy-hub", dim=d_src)
        spec_c = spec.model_copy(update={
            "native_band": BandRef(family="Rband", recipe=native_recipe),
            "source_band": BandRef(family="gRband", recipe=source_recipe)})
        cen_c = census(spec_c)
        check("a CONSTRUCTIBLE band is a third census state, still not READY",
              not cen_c.ready and all(r.constructible for r in cen_c.owed),
              "CONSTRUCTIBLE")
        check("a recipe anchored to ANOTHER basis is NOT constructible — it blocks",
              not any(r.constructible for r in census(spec.model_copy(update={
                  "native_band": BandRef(family="Rband", recipe=native_recipe
                                         .model_copy(update={"corpus_sha":
                                                             other_basis}))})).owed
                      if r.role == "native_band"),
              "a cross-basis band is refused at the census, before any build")
        built_c, doc_c = build_banks(spec_c, construct_bands=True, write=False)
        keys_c = {c.spec.vector_key for c in doc_c.cells}
        check("--construct-bands + the ruled recipe BUILDS the column the refusal "
              "used to block (42/51 cells)",
              built_c.n_cells == N_FULL_COLUMN_CELLS
              and {"Rband1", "Rband2", "Rband3"} <= keys_c
              and {"gRband1", "gRband2", "gRband3"} <= keys_c,
              f"{built_c.n_cells} cells, {built_c.n_vectors} vectors")
        # sha and vintage failures
        bad_sha = spec.model_copy(update={
            "native_vector": native.model_copy(update={"sha256": "0" * 64})})
        check("a sha mismatch is reported as a blocking row (M4), never skipped",
              not census(bad_sha).ready
              and any("sha" in (r.blocking_reason or "").lower()
                      for r in census(bad_sha).owed))
        #  ── the 2026-08-01 ruling, scope 3: a sha mismatch between a rebuild
        #  and a banked PRE-RULING digest must read as a LABELED count
        #  mismatch, never as a silent byte diff. Three cases, all exercised.
        thread_stamp = root / "threaded_stamps.json"
        thread_stamp.write_text(json.dumps({
            "corpus_manifest_sha256": corpus_sha, "builder": "selftest",
            "thread_config": thread_config_stamp(
                ThreadConfig(effective_num_threads=RULED_OMP_NUM_THREADS,
                             consensus="agreed", matches_ruled_default=True))}))
        rebuilt_at_8 = native.model_copy(update={
            "sha256": "0" * 64, "build_stamp": thread_stamp,
            "expected_thread_count": 1})
        row8 = [r for r in census(spec.model_copy(
            update={"native_vector": rebuilt_at_8})).rows
            if r.role == "native_vector"][0]
        check("a rebuild-vs-banked sha mismatch QUOTES BOTH thread counts, "
              "labeled, rather than reporting a bare byte difference",
              row8.thread_count_mismatch is not None
              and THREAD_COUNT_MISMATCH_LABEL in row8.thread_count_mismatch
              and "banked digest 1" in row8.thread_count_mismatch
              and "artifact on disk 8" in row8.thread_count_mismatch,
              (row8.thread_count_mismatch or "")[:96])
        check("and the labeled mismatch travels in the BLOCKING REASON too, so "
              "a reader of the census table sees it without a second lookup",
              THREAD_COUNT_MISMATCH_LABEL in (row8.blocking_reason or "")
              and "sha" in (row8.blocking_reason or "").lower(),
              "the sha mismatch is still a HALT (M4) — this only says WHY")
        pre_ruling = native.model_copy(update={"sha256": "0" * 64})
        row_pre = [r for r in census(spec.model_copy(
            update={"native_vector": pre_ruling})).rows
            if r.role == "native_vector"][0]
        check("when NEITHER side records a count, both are quoted as unrecorded "
              "— 'both unknown' is never reported as 'both equal'",
              row_pre.thread_count_mismatch is not None
              and PRE_RULING_UNRECORDED in row_pre.thread_count_mismatch,
              (row_pre.thread_count_mismatch or "")[-96:])
        agreeing = native.model_copy(update={
            "build_stamp": thread_stamp,
            "expected_thread_count": RULED_OMP_NUM_THREADS})
        row_ok = [r for r in census(spec.model_copy(
            update={"native_vector": agreeing})).rows
            if r.role == "native_vector"][0]
        check("two artifacts at the SAME recorded count carry NO mismatch note, "
              "so the field means what it says when it is populated",
              row_ok.thread_count_mismatch is None and row_ok.sha_verified is True)

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

        # ---- 6b. B4: the Σ-beside CELLS, on designated rows only ---------------
        print("== selftest 6b: Σ-beside CELLS — 6 per designated row (B4) ==")
        dn, da, ds, dk = SIGMA_BESIDE_DESIGNATIONS["3b-certification-calibration"]
        sig_native = _sigma_recipe(vector_key="entropy_gradient")

        def _designated_spec(**over: Any) -> BankSpec:
            base: dict[str, Any] = dict(
                node_key=dn, arm=da, site=ds, corpus_manifest=corpus,
                corpus_sha_of_record=corpus_sha, native_vector=native,
                native_band=BandRef(family=dk,
                                    recipe=native_recipe.model_copy(update={
                                        "node_key": dn, "arm": da, "site": ds,
                                        "draw_space_key": dn}),
                                    sigma_beside=sig_native),
                include_transported=False, include_naive=False,
                out_dir=root / "out_sigma")
            base.update(over)
            return BankSpec(**base)

        sig_spec = _designated_spec()
        sig_cells = plan_cells(sig_spec)
        beside = [c for c in sig_cells if c.spec.is_beside]
        check("a designated calibration row plans 25 + 6 = 31 cells",
              len(sig_cells) == 31 and planned_cell_count(sig_spec) == 31
              and planned_cell_count(sig_spec, of_record_only=True) == 25,
              f"{len(sig_cells)} cells, {len(beside)} beside")
        check("B4's 6 = 3 members × the 2 SCORING doses, and the doses ARE ±0.3",
              len(beside) == N_SIGMA_BESIDE_CELLS == 6
              and sorted({c.spec.vector_key for c in beside})
              == ["SigmaBand1", "SigmaBand2", "SigmaBand3"]
              and sorted({c.spec.alpha_frac for c in beside}) == sorted(SCORING_DOSES),
              str(sorted(c.spec.cell_id for c in beside)))
        check("every Σ cell is a BAND cell of the family it sits beside's KIND, and "
              "carries band_family=SigmaBand",
              all(c.spec.kind == "calibration_band" and c.spec.band_family == "SigmaBand"
                  and c.spec.is_beside and not c.spec.is_null_of_record
                  for c in beside))
        check("the isotropic band of record is UNTOUCHED beside them (18 Rband cells)",
              sum(1 for c in sig_cells
                  if c.spec.band_family == "Rband") == N_CALIBRATION_BAND_CELLS)
        check("a Σ recipe describing ANOTHER cell is refused BY THE SPEC (O-2 "
              "designates cells), before any census or build",
              _raises(lambda: _designated_spec(site=ds + 1), ValueError)
              and _raises(lambda: _designated_spec(
                  native_band=BandRef(family=dk, recipe=native_recipe,
                                      sigma_beside=_sigma_recipe(
                                          node_key="toy-node"))), ValueError),
              "the guard that used to live only in build_sigma_band now also gates "
              "the spec, so an undesignated Σ never reaches the engine contract")
        check("an undesignated Σ is refused at PLAN time too (not only at --build)",
              _raises(lambda: _sigma_beside_cells(
                  _sigma_recipe(designation="a-third-cell"), ds,
                  kind="calibration_band", provenance="p", npz=None, n=80, norm=1.0),
                  ConstructionRefused))

        # ---- 6c. B2: the COMPOSED road (source → hub → target) ----------------
        print("== selftest 6c: the composed two-hop road (B2's composed-leg pairs) ==")
        d_hub = 11
        (root / "leg_in").mkdir(exist_ok=True)
        (root / "leg_out").mkdir(exist_ok=True)
        leg_in_path = _toy_map_npz(root / "leg_in", d_hub, d_src, seed=21)
        leg_out_path = _toy_map_npz(root / "leg_out", d_hub, d_tgt, seed=22)

        def _leg(path: Path, **over: Any) -> TransportMapRef:
            base: dict[str, Any] = dict(fit=path, family="proc_k256", arm="native",
                                        direction="fwd", corpus_vintage=corpus_sha)
            base.update(over)
            return TransportMapRef(**base)

        composed = ComposedMapRef(
            hub_key="toy-hub-of-record", hub_site=26, family="proc_k256", arm="native",
            hub_to_source=_leg(leg_in_path), hub_to_target=_leg(leg_out_path),
            selection_rule="B2: product of the two crowned-hub leg âs",
            a_hat_hub_to_source=0.5, a_hat_hub_to_target=0.4, a_hat_product=0.2,
            product_rank=1)
        comp_spec = BankSpec(
            node_key="toy-node", arm="native", site=7, source_key="toy-hub",
            source_site=3, pair="src->tgt", corpus_manifest=corpus,
            corpus_sha_of_record=corpus_sha, native_vector=native,
            source_vector=source, composed_map=composed, naive_gate=gate,
            native_band=BandRef(family="Rband", recipe=native_recipe),
            source_band=BandRef(family="gRband", recipe=source_recipe),
            out_dir=root / "out_composed")
        check("a spec may name a direct fit OR a composed road, never both",
              _raises(lambda: comp_spec.model_copy(update={}).__class__(
                  **{**json.loads(comp_spec.model_dump_json()),
                     "transport_map": {"fit": str(map_path), "family": "proc_k256",
                                       "arm": "native", "corpus_vintage": corpus_sha}}),
                  ValueError))
        check("two legs in different arms / families / the same fit are refused",
              _raises(lambda: ComposedMapRef(
                  hub_key="h", hub_site=26, family="proc_k256", arm="native",
                  hub_to_source=_leg(leg_in_path),
                  hub_to_target=_leg(leg_out_path, arm="raw")), ValueError)
              and _raises(lambda: ComposedMapRef(
                  hub_key="h", hub_site=26, family="proc_k256", arm="native",
                  hub_to_source=_leg(leg_in_path),
                  hub_to_target=_leg(leg_out_path, family="proc_k128")), ValueError)
              and _raises(lambda: ComposedMapRef(
                  hub_key="h", hub_site=26, family="proc_k256", arm="native",
                  hub_to_source=_leg(leg_in_path),
                  hub_to_target=_leg(leg_in_path)), ValueError))
        comp_cen = census(comp_spec)
        check("the census reports BOTH composed legs as their own rows",
              sum(1 for r in comp_cen.rows if r.role == "transport_map") == 2
              and all(r.ready for r in comp_cen.rows if r.role == "transport_map"),
              str([r.name for r in comp_cen.rows if r.role == "transport_map"]))
        comp_res, comp_doc = build_banks(comp_spec, construct_bands=True, write=True)
        with np.load(comp_res.vectors_npz) as _cz:
            comp_dims = {k: int(np.asarray(_cz[k]).reshape(-1).size) for k in _cz.files}
        check("every composed object lands in the TARGET's dimension after two hops "
              f"(source d={d_src} → hub d={d_hub} → target d={d_tgt})",
              comp_res.n_cells == 51 and set(comp_dims.values()) == {d_tgt}
              and {"gentropy_gradient", "gRband1", "gRband2", "gRband3"}
              <= set(comp_dims),
              str(sorted(set(comp_dims.values()))))
        st = comp_doc.bank_stamp
        check("the stamp says COMPOSED and names BOTH legs with their shas and the "
              "orientation each was ridden in",
              st["transport_road"] == "composed"
              and [lg["role"] for lg in st["composed_map"]["legs"]]
              == ["hub_to_source", "hub_to_target"]
              and [lg["ridden"] for lg in st["composed_map"]["legs"]] == ["rev", "fwd"]
              and all(len(lg["fit_sha256"]) == 64
                      for lg in st["composed_map"]["legs"]))
        check("the composed construction NAMES the two-hop rather than a single map",
              "COMPOSED two-hop" in st["vector_constructions"]["gentropy_gradient"])
        check("the composed vintage chain carries a link PER LEG (a hole is visible)",
              set(comp_doc.vintage_chain) >= {"composed_leg_hub_to_source",
                                              "composed_leg_hub_to_target"}
              and comp_doc.vintage_chain["transport_map"] is None
              and comp_doc.vintage_chain["composed_leg_hub_to_target"] == corpus_sha)
        # the fit sha field is a DIGEST over the two legs — a changed leg must move it
        alt = ComposedMapRef(**{**json.loads(composed.model_dump_json()),
                                "hub_to_source": json.loads(
                                    _leg(leg_out_path).model_dump_json()),
                                "hub_to_target": json.loads(
                                    _leg(leg_in_path).model_dump_json())})
        check("the composed fit-sha digest is ORDER-SENSITIVE (swapping the legs "
              "moves it — the two hops are not interchangeable)",
              hashlib.sha256((sha256_file(alt.hub_to_source.fit) + "|"
                              + sha256_file(alt.hub_to_target.fit)).encode()
                             ).hexdigest() != st["composed_map"]["fit_sha256"])

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
                                            "source_vector_build", "transport_map",
                                            "composed_leg_hub_to_source",
                                            "composed_leg_hub_to_target"},
              str(sorted(loaded.vintage_chain)))
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

        # ---- 10b. HALT C: the robustness-site role (RULED 2026-08-05) ----------
        print("== selftest 10b: HALT C — robustness_site admissibility ==")
        try:
            from metabasis.scripts.actuation_calibration import (
                ROBUSTNESS_SITES as _ROB, assert_no_site_fishing as _fishing)
            from metabasis.scripts.fit_transport_maps import SITES as _SITES2
            halt_c_ok = True
        except ImportError as exc:                                # pragma: no cover
            halt_c_ok = False
            skip("HALT C's robustness-site admissibility",
                 f"the §4.2 registry is unimportable ({exc})")
        if halt_c_ok:
            rob_node, rob_site = "gemma3-27b", _ROB["gemma3-27b"]
            grid = tuple(dict(_SITES2).get(rob_node, ()))
            # (1) TODAY'S REFUSAL, UNCHANGED. No role declared → exactly the
            # pre-ruling behaviour, which is what blocked gemma L41's column.
            check("(HALT C) with NO site_role, the registered robustness site is "
                  "STILL refused — the ruling widened what a spec may SAY, never "
                  "what silence means",
                  _raises(lambda: site_cross_check(rob_node, rob_site),
                          SiteNotOfRecord),
                  f"{rob_node} L{rob_site}, undeclared")
            check("(HALT C) …and the refusal is the SAME class the guard always "
                  "raised, so every caller that treated it as a HALT still does",
                  type(_msg_exc(lambda: site_cross_check(rob_node, rob_site)))
                  is SiteNotOfRecord)
            # (2) DECLARED → ADMITTED, for the REGISTERED site only.
            rob_out = site_cross_check(rob_node, rob_site,
                                       site_role="robustness_site")
            check("(HALT C) with site_role='robustness_site' DECLARED, the "
                  "REGISTERED robustness site is admitted (§4.2's frozen remedy (i))",
                  rob_out["agrees"] is True
                  and rob_out["site_role"] == "robustness_site"
                  and rob_out["robustness_site_registered"] == rob_site,
                  f"{rob_node} L{rob_site}; SITE_OF_RECORD stays "
                  f"L{rob_out['SITE_OF_RECORD']}")
            check("(HALT C) the admitted row still names BOTH registries, so the "
                  "stamp says which site of record it stands beside",
                  rob_out["SITES"] == list(grid)
                  and rob_out["SITE_OF_RECORD"] != rob_site
                  and rob_site in rob_out["SITES"])
            # (3) THE ADMISSIBILITY IS THE §4.2 MODULE'S, not a second reading.
            check("(HALT C) the guard admits EXACTLY what assert_no_site_fishing "
                  "admits — the same (node, site) passes there too",
                  _ok(lambda: _fishing(rob_node, rob_site,
                                       evidence="the registered robustness site")))
            # (4) A SITE OUTSIDE THE REGISTRY REFUSES EVEN WITH THE ROLE.
            off_grid = max(grid or (0,)) + 11
            check("(HALT C) a site OUTSIDE the node's SITES grid is refused even "
                  "WITH site_role — a role string cannot register a site",
                  _raises(lambda: site_cross_check(rob_node, off_grid,
                                                   site_role="robustness_site"),
                          RobustnessSiteNotRegistered),
                  f"L{off_grid} ∉ {grid}")
            check("(HALT C) a RETIRED site (gemma L36) is refused with the role too",
                  _raises(lambda: site_cross_check(rob_node, 36,
                                                   site_role="robustness_site"),
                          RobustnessSiteNotRegistered))
            check("(HALT C) the node's own SITE OF RECORD is refused under the "
                  "ROBUSTNESS role — a mislabelled site is not a licensed one",
                  _raises(lambda: site_cross_check(rob_node,
                                                   int(rob_out["SITE_OF_RECORD"]),
                                                   site_role="robustness_site"),
                          RobustnessSiteNotRegistered))
            check("(HALT C) a node with NO registered robustness site cannot "
                  "declare one (the role is admitted by a REGISTERED row or not "
                  "at all)",
                  _raises(lambda: site_cross_check("qwen2.5-3b-instruct", 26,
                                                   site_role="robustness_site"),
                          RobustnessSiteNotRegistered))
            check("(HALT C) every robustness refusal is still a SiteNotOfRecord "
                  "(§9 item 4) — the subclasses name the reason, not a new state",
                  issubclass(RobustnessSiteNotRegistered, SiteNotOfRecord)
                  and issubclass(RobustnessSiteOffGrid, SiteNotOfRecord)
                  and issubclass(SiteRoleRefused, SiteNotOfRecord))
            # (5) THE FLAG-ABSENT PATH IS BYTE-IDENTICAL.
            plain = site_cross_check(rob_node, int(rob_out["SITE_OF_RECORD"]))
            check("(HALT C) the UNDECLARED cross-check dict carries NO site_role "
                  "key at all — a column staged before the ruling cannot grow a "
                  "key it never had",
                  set(plain) == {"SITES", "SITE_OF_RECORD", "staged_site", "agrees"},
                  str(sorted(plain)))
            declared = site_cross_check(rob_node, int(rob_out["SITE_OF_RECORD"]),
                                        site_role="site_of_record")
            check("(HALT C) declaring the ORDINARY role changes nothing but the "
                  "record of the declaration",
                  {k: v for k, v in declared.items() if k != "site_role"} == plain
                  and declared["site_role"] == "site_of_record")
            # (6) THE ROLE FLOWS INTO THE STAMP AND THE ENGINE'S DOCUMENT.
            rob_spec = spec.model_copy(update={
                "node_key": rob_node, "site": rob_site,
                "site_role": "robustness_site",
                "include_transported": False, "include_naive": False,
                "native_band": BandRef(family="Rband", members=(
                    native.model_copy(update={"key": f"Rband{i}"})
                    for i in (1, 2, 3)))})
            rob_stamp = bank_stamp(
                rob_spec, cells=plan_cells(rob_spec), vectors={}, magnitudes={},
                constructions={}, transport_map=None, naive_row=None,
                site_cross_check=site_cross_check(rob_node, rob_site,
                                                  site_role="robustness_site"),
                prompt_pool_sha256=None, battery_item_set_sha256="a" * 64)
            check("(HALT C) the declared role rides the BANK STAMP's site_cross_check "
                  "block, which is the §2.8 field the cell stamp carries",
                  rob_stamp["site_cross_check"]["site_role"] == "robustness_site"
                  and rob_stamp["site"] == rob_site)

        # ---- 11. HALT D: the vector-class contract (RULED 2026-08-05) ----------
        print("== selftest 11: HALT D — the vector-class contract, two bases ==")
        set_sha = "1a" * 32
        (root / "classvec").mkdir(exist_ok=True)
        class_npz = _toy_class_vector_npz(root / "classvec", "caa_formality_L7",
                                          d_tgt, seed=11, set_sha256=set_sha)
        basis = VectorBasis(kind="contrast-set", sha256=set_sha,
                            provenance="pin row: RULED contrast set (selftest)")
        class_ref = VectorRef(key="caa_formality_L7", npz=class_npz,
                              sha256=sha256_file(class_npz),
                              build_stamp=class_npz.with_name(
                                  "caa_formality_L7_stamps.json"),
                              vector_class="caa", fd_gate_not_applicable=True,
                              vector_basis=basis,
                              provenance="text-contrast CAA of record (selftest)")
        check("(HALT D) an EGV that claims fd_gate_not_applicable is REFUSED — the "
              "FD gate is the entropy-gradient object's OWN acceptance test",
              _raises(lambda: VectorRef.model_validate(
                  {**native.model_dump(), "fd_gate_not_applicable": True}),
                  FDGateWaiverRefused),
              "the real banked lever's own fields, one flag flipped")
        check("(HALT D) …by direct construction too (the path a spec file takes)",
              _raises(lambda: VectorRef(key="entropy_gradient", npz=class_npz,
                                        fd_gate_not_applicable=True),
                      FDGateWaiverRefused))
        check("(HALT D) a class vector that BOTH waives the gate and names one is "
              "refused — a waiver and a claim cannot both be true",
              _raises(lambda: VectorRef(
                  key="caa_x", npz=class_npz, vector_class="caa",
                  fd_gate_not_applicable=True, vector_basis=basis,
                  fd_gate=class_npz), FDGateWaiverRefused))
        check("(HALT D) a class vector with NO vector_basis is refused (its basis "
              "is the RULED contrast set; a hole is a hole)",
              _raises(lambda: VectorRef(key="caa_x", npz=class_npz,
                                        vector_class="caa",
                                        fd_gate_not_applicable=True),
                      VectorBasisMissing))
        check("(HALT D) a class vector whose basis KIND is the corpus manifest is "
              "refused — that is the conflation itself",
              _raises(lambda: VectorRef(
                  key="caa_x", npz=class_npz, vector_class="caa",
                  fd_gate_not_applicable=True,
                  vector_basis=VectorBasis(kind="corpus-manifest",
                                           sha256=corpus_sha)),
                  VectorBasisConflated))
        check("(HALT D) an EGV that names a vector_basis is refused — the EGV's "
              "basis IS the corpus manifest and has exactly one name",
              _raises(lambda: VectorRef(key="entropy_gradient", npz=class_npz,
                                        vector_basis=basis),
                      VectorBasisConflated))
        # the census: (a) the waiver is HONOURED for a class object…
        class_spec = spec.model_copy(update={
            "native_vector": class_ref, "include_transported": False,
            "include_naive": False})
        class_row = [r for r in census(class_spec).rows
                     if r.role == "native_vector"][0]
        check("(HALT D)(a) the census HONOURS fd_gate_not_applicable: a class "
              "vector with no FD gate is READY and its row says why",
              class_row.ready and class_row.fd_gate_passed is None
              and class_row.fd_gate_not_applicable is True
              and class_row.vector_class == "caa",
              class_row.blocking_reason or "no blocking reason")
        check("(HALT D)(a) the class row reads its VECTOR basis from the object's "
              "own build stamp and checks it against the RULED sha — the corpus "
              "vintage field stays EMPTY rather than borrowing the corpus sha",
              class_row.vector_basis["build_stamp_sha256"] == set_sha
              and class_row.vector_basis["declared_sha256"] == set_sha
              and class_row.corpus_vintage is None)
        wrong_set = _toy_class_vector_npz(root / "classvec", "caa_wrong", d_tgt,
                                          seed=12, set_sha256="2b" * 32)
        wrong_ref = class_ref.model_copy(update={
            "key": "caa_wrong", "npz": wrong_set, "sha256": sha256_file(wrong_set),
            "build_stamp": wrong_set.with_name("caa_wrong_stamps.json")})
        wrong_row = [r for r in census(class_spec.model_copy(update={
            "native_vector": wrong_ref})).rows if r.role == "native_vector"][0]
        check("(HALT D)(a) a class vector built from ANOTHER contrast set blocks "
              "(§9 item 2 applies to the VECTOR basis exactly as to the corpus)",
              not wrong_row.ready
              and "RULED set" in (wrong_row.blocking_reason or ""),
              (wrong_row.blocking_reason or "")[:80])
        # …(b) and the EGV's refusal SURVIVES, which is the point of the ruling.
        (root / "egv_nogate").mkdir(exist_ok=True)
        egv_nofd = _toy_vector_npz(root / "egv_nogate", "entropy_gradient", d_tgt,
                                   seed=13, vintage=corpus_sha, fd_pass=False)
        egv_row = [r for r in census(class_spec.model_copy(update={
            "native_vector": egv_nofd})).rows if r.role == "native_vector"][0]
        check("(HALT D)(a) an ENTROPY-GRADIENT object still REQUIRES its FD gate — "
              "the refusal the ruling had to survive, proven on the census",
              not egv_row.ready
              and "FD gate" in (egv_row.blocking_reason or "")
              and egv_row.fd_gate_not_applicable is None
              and egv_row.vector_class is None)
        no_gate_class = class_ref.model_copy(update={"fd_gate_not_applicable": False})
        undeclared_row = [r for r in census(class_spec.model_copy(update={
            "native_vector": no_gate_class})).rows if r.role == "native_vector"][0]
        check("(HALT D)(a) a class vector that does NOT waive the gate is held to "
              "it — claiming a gate you do not have is refused, not excused",
              not undeclared_row.ready
              and "FD gate" in (undeclared_row.blocking_reason or ""))
        # the staged document + bank stamp carry both bases, and only when class
        class_cells = plan_cells(class_spec)
        check("(HALT D)(b) every planned cell of a class column carries the class "
              "declaration and its VECTOR basis (the band members too — they are "
              "the controls FOR that object and share its custody)",
              all(c.vector_class == "caa"
                  and c.vector_basis is not None
                  and c.vector_basis.kind == "contrast-set" for c in class_cells),
              f"{len(class_cells)} cells")
        class_stamp = bank_stamp(
            class_spec, cells=class_cells, vectors={}, magnitudes={},
            constructions={}, transport_map=None, naive_row=None,
            site_cross_check={"agrees": True}, prompt_pool_sha256=None,
            battery_item_set_sha256="a" * 64)
        egv_stamp = bank_stamp(
            class_spec.model_copy(update={"native_vector": native}),
            cells=plan_cells(class_spec.model_copy(update={
                "native_vector": native})), vectors={}, magnitudes={},
            constructions={}, transport_map=None, naive_row=None,
            site_cross_check={"agrees": True}, prompt_pool_sha256=None,
            battery_item_set_sha256="a" * 64)
        check("(HALT D)(b) the class bank stamp names BOTH bases, each under its "
              "own key, and never one as the other",
              class_stamp["vector_basis"]["native_vector"]["kind"] == "contrast-set"
              and class_stamp["vector_basis"]["native_vector"]["sha256"] == set_sha
              and class_stamp["generation_basis"]["kind"] == "corpus-manifest"
              and class_stamp["generation_basis"]["sha256"] == corpus_sha
              and set_sha != corpus_sha,
              "contrast set ≠ corpus manifest, and the stamp says which is which")
        check("(HALT D)(b) the class stamp records the FD-gate waiver and quotes "
              "the builder's own reason for it",
              class_stamp["fd_gate_not_applicable"]["native_vector"] is True
              and "no FD-gate analogue" in class_stamp["fd_gate_not_applicable_note"])
        _class_only = {"vector_class", "vector_basis", "generation_basis",
                       "two_bases_note", "fd_gate_not_applicable",
                       "fd_gate_not_applicable_note"}
        check("(HALT D)(b) an ENTROPY-GRADIENT column's bank stamp grows NONE of "
              "the class keys — the flag-absent path, structurally proven",
              set(egv_stamp) & _class_only == set()
              and set(class_stamp) - set(egv_stamp) == _class_only,
              f"class-only keys: {sorted(_class_only)}")
        check("(HALT D)(b) …and the two stamps agree on EVERY other key, so the "
              "declaration adds and never moves",
              all(json.dumps(egv_stamp[k], sort_keys=True, default=str)
                  == json.dumps(class_stamp[k], sort_keys=True, default=str)
                  for k in set(egv_stamp) & set(class_stamp)
                  if k not in ("vector_fd_gate", "vector_build_stamp")),
              "vector_fd_gate/vector_build_stamp differ BY CONSTRUCTION (the class "
              "object has no gate) and are the fields the ruling is about")
        check("(HALT D) a StagedCell that declares a class without a basis is "
              "refused before it can reach the engine",
              _raises(lambda: StagedCell(spec=class_cells[0].spec,
                                         vector_class="caa"), VectorBasisMissing)
              and _raises(lambda: StagedCell(spec=class_cells[0].spec,
                                             vector_basis=basis),
                          VectorBasisConflated))

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


def _toy_class_vector_npz(dirpath: Path, key: str, dim: int, *, seed: int,
                          set_sha256: str) -> Path:
    """A synthetic CLASS vector + the build stamp `build_contrast_vectors` writes.

    The stamp shape is that module's own: the contrast set's sha lives under
    `set.sha256` and there is NO `corpus_manifest_sha256` anywhere, because a CAA
    object is not "of" a corpus. That absence is precisely what made the census
    block every class column before HALT D was ruled, so the fixture reproduces it
    rather than papering over it. No FD-gate sidecar is written either — a class
    object has no analogue to write one from.
    """
    rng = np.random.default_rng(seed)
    npz = dirpath / f"{key}.npz"
    np.savez(npz, **{key: rng.standard_normal(dim).astype(np.float32)})
    (dirpath / f"{key}_stamps.json").write_text(json.dumps({
        "construction": "caa", "axis": "selftest-axis",
        "set": {"set_id": "selftest-set", "sha256": set_sha256, "n_pairs": 4},
        "what_this_is_not": ["no FD-gate analogue exists for a CAA object"],
    }))
    return npz


def _msg_exc(fn: Callable[[], Any]) -> Optional[BaseException]:
    """The exception `fn` raised, for a check that asserts its exact class."""
    try:
        fn()
    except BaseException as exc:                       # noqa: BLE001 — that is the point
        return exc
    return None


def _write_npz(path: Path, **arrays: Any) -> Path:
    """Selftest scaffolding: a toy npz written where a builder would write one."""
    np.savez(path, **arrays)
    return path


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
