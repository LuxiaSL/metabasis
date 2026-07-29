"""Composed-path predictor — â_comp from two banked hub maps, before any pair fit.

WHAT THIS IS. `read_exchange_rates` reads â for a pair that HAS a fitted map.
This module computes the pair's â *without one*, from the two banked
primary-hub maps alone:

    â_comp(A→B) = cos( M_hub→B( M_hub→A^rev(v_A) ), v_B )

Draft Addendum E §E1 (`docs/planning/DRAFT-ADDENDUM-E-2026-07-28.md`) — NOT
RATIFIED. **THIS TOOL FILES NOTHING.** It computes â_comp, proves fidelity to
the archived operationalization of record, and reports which candidate slots
have both hub maps banked in their applicable arm. Filing machinery activates
only on ratification; if the outside reviewer moves the E1 definition, this
module is reworked before anything files.

────────────────────────────────────────────────────────────────────────────────
THE ALGEBRA — why the two-hop is well-defined, and what "rev" actually is
────────────────────────────────────────────────────────────────────────────────
A banked hub map M_hub→A has the HUB on its source side and model A on its
target side, so `transport(·, "fwd")` goes hub→A and `transport(·, "rev")`
goes A→hub. Composing rev(A→hub) with fwd(hub→B) is therefore dimensionally
forced: both maps share the hub-side PCA basis's dimension, and nothing
pair-specific is touched.

`TransportMap.transport(·, "rev")` IS the adjoint identity of record — for a
proc map

    fwd:  out = ((v / src_norm) @ va.T @ Ω  @ vb) * scale * tgt_norm
    rev:  out = ((v / tgt_norm) @ vb.T @ Ω.T @ va) / scale * src_norm

i.e. Ω_MA = Ω_AMᵀ over the SAME PCA bases (`va`, `vb`), which is the bit-exact
identity established in the 8b↔3b closure. `adjoint_map()` below constructs
that swapped-basis map explicitly and the selftest proves the two agree to
machine precision — the identity is demonstrated in code, never asserted in a
comment.

Two consequences worth stating, both proved in the selftest:

  * **The self-composition identity.** With M_hub→B ≡ M_hub→A ≡ M, the two-hop
    collapses exactly to the orthogonal projector onto the map's image:
    fwd(rev(v)) = v @ vbᵀ Ω.T Ω vb = v @ vbᵀ vb = P_image(v), because `va` has
    orthonormal rows (va vaᵀ = I_k) and Ω is orthogonal. So
    â_comp(A→A) = ‖P_image v_A‖ = the â-CEILING of that map at v_A. The
    composed path is thus bounded by the same ceiling algebra
    `read_ahat_ceilings` measures — â_comp ≤ ceiling(M_hub→B, v_B) always.
  * **Normalization invariance.** Every scalar in the path (`src_norm`,
    `tgt_norm`, `scale`, and the input vector's own norm) is positive and
    factors out of a cosine, so â_comp is invariant to all of them — exactly
    as â is (`read_exchange_rates`, normalization section). Vectors are
    unit-normalized on load in fp64 and passed to `transport` UNMODIFIED; the
    map owns its own normalization round trip.

────────────────────────────────────────────────────────────────────────────────
THE OPERATIONALIZATION OF RECORD, AND THE GATE
────────────────────────────────────────────────────────────────────────────────
The archived worst-pair glue
`outputs/collection/enactment-archives/worstpair-diag/wp_composition.py` is the
operationalization of record. This module does not mirror its arithmetic — it
CALLS THE SAME FUNCTIONS the archive calls (`TransportMap.transport`,
`read_exchange_rates.cos`, `load_entropy_gradient`, `load_transport_map`), so
the composed expression is the same object, not a copy of it.

`--gate` proves that empirically and is re-runnable at filing time (draft E1's
requirement). It:

  1. verifies the archived `.py` and `.json` against the archive's own
     `MANIFEST-worstpair-diag.sha256` (sha mismatch on a number-bearing
     artifact = HALT, never a warning);
  2. executes the archived glue BY PATH with `sys.dont_write_bytecode = True`
     (RAKE M16 — an importlib load-by-path otherwise drops a `__pycache__`
     inside a tree we are read-only over), and recomputes each archived
     retrodiction at FULL fp64 precision through the archive's own path
     resolution;
  3. recomputes the same five through THIS module's registry-driven resolution
     and checks |Δ| < 1e-8 against (2) — the frozen E1 gate;
  4. cross-checks the banked 6-dp values in `wp_composition.json` and the
     4-dp display figures quoted in the draft, so a silent archive edit is
     caught even if step 2 and step 3 drift together;
  5. checks RESOLUTION PARITY: the fit npz this module resolves must be the
     same file the archive resolved, per side.

`main(["--gate"])` exits NONZERO on any failure. A tool that cannot reproduce
the archive does not file.

────────────────────────────────────────────────────────────────────────────────
ARM AVAILABILITY, AND WHY N/A-AT-FILING IS A FIRST-CLASS RESULT
────────────────────────────────────────────────────────────────────────────────
Prereg §3: instruct↔base pairs are predicted in the RAW system, instruct↔
instruct in NATIVE (base models run raw only — the arm-consistency rule, §1).
Draft E1: â_comp files only where BOTH hub maps exist banked in the pair's
applicable arm; where they do not, the slot is **N/A-AT-FILING** — never
proxied from another arm, never backfilled after the pair is fit. This module
therefore reports missing maps as a structured `HubMapRef` with every probed
path, and refuses to substitute an arm. `require_hub_map()` is the loud
variant for callers that must have the map.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.scripts.read_composed_predictions --selftest
  python -m metabasis.scripts.read_composed_predictions --gate
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --out /tmp/claude-output/composed_candidates.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from metabasis.roster import ROSTER
from metabasis.scripts.fit_transport_maps import (
    A8_SEED, ARMS, FitGridError, TransportMap, load_transport_map, require_site)
from metabasis.scripts.read_ahat_ceilings import image_basis, projection_norm
from metabasis.scripts.read_exchange_rates import (
    BANK_ROOT, COLLECTION_ROOT, FAMILIES, FAMILY_OF_RECORD, HUB_MODEL,
    NEAR_ZERO_CARVE_OUT, VectorSpec, cos, fit_path_for, load_entropy_gradient,
    unit)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("read_composed_predictions")

# ---------------------------------------------------------------- constants
#: The hub column of record for the composed predictor: the primary hub 8B at
#: its only banked site, rebuilt-L16 (`HUB_BANKED_SITES == (16,)`).
HUB_SITE_OF_RECORD = 16
#: Draft E1 gate: |Δ| against the archived retrodictions, at full precision.
GATE_TOLERANCE = 1e-8
#: `wp_composition.json` rounds `a_composed_hub_mediated` to 6 dp, so the
#: banked-JSON leg of the gate can only be checked to half an ulp of that.
BANKED_JSON_TOLERANCE = 5e-7
#: The draft quotes 4-dp display figures; they must round-trip exactly.
DISPLAY_DECIMALS = 4

ARCHIVE_ROOT = COLLECTION_ROOT / "enactment-archives" / "worstpair-diag"
ARCHIVE_GLUE = ARCHIVE_ROOT / "wp_composition.py"
ARCHIVE_JSON = ARCHIVE_ROOT / "wp_composition.json"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "MANIFEST-worstpair-diag.sha256"

#: Draft E1 / E2: the five archived retrodictions as the draft displays them.
#: Keyed by the archive's own `pair` string so a reordering of either side
#: cannot silently pair the wrong numbers.
DRAFT_DISPLAY_TARGETS: dict[str, float] = {
    "qwen2.5-32b-instructL46->qwen2.5-14b-instructL29": 0.5824,
    "qwen2.5-3b-instructL26->qwen2.5-32b-instructL46": 0.5407,
    "dsv2-liteL22->mistral-7b-instruct-v0.3L15": 0.2820,
    "phi-3.5-mini-instructL13->qwen2.5-32b-instructL46": 0.3051,
    "mistral-7b-instruct-v0.3L15->phi-4L19": 0.3170,
}

#: The banked naive-transplant gate column — the enumeration of record for the
#: current candidate pairs (66 rows + a gpt2-xl supplement that is NOT a
#: candidate: that model's site of record is deferred).
CANDIDATE_PAIRS_JSON = (COLLECTION_ROOT / "readouts"
                        / "naive_transplant_gate_2026-07-28.json")


class ComposedPathError(RuntimeError):
    """A composed-path resolution or arithmetic failure that must be LOUD.

    Never raised for a legitimately absent map — that is N/A-AT-FILING and is
    reported as data (`HubMapRef.resolved is None`). This is for the cases
    where continuing would produce a number that means something other than
    what its label says: an unknown model key, a dimension mismatch between
    the two hub legs, a corrupted archive.
    """


class ArchiveIntegrityError(ComposedPathError):
    """The archived operationalization does not match its own manifest sha."""


# ---------------------------------------------------------------- registries
#: Site of record per candidate model. Every entry is validated against the
#: model's FIXED FIT GRID via `require_site` (registry API, commit 8c00896) —
#: no site here can be a retired or mistyped one that a bank happens to hold
#: (the 70B's L17 is exactly that case and is absent by construction).
#: Sources: wave-1 + hub-rungs-2 ratifications (Luxia 2026-07-27, recorded in
#: `fit_transport_maps.SITES` with the ⋆ marks) and the carried banked nodes'
#: banked sites (CAMPAIGN-MAP banked-4).
SITE_OF_RECORD: dict[str, int] = {
    # carried banked nodes
    "3b": 14,
    "dsv2-lite": 22,
    "qwen-7b": 21,
    # gemma3-27b's banked site is L36 — a CARRIED banked site, not a fresh
    # ratification: the entire banked gemma object roster (the needle, field,
    # temperature and repetition members) lives at L36, which is why L36 is the
    # load-bearing member of its fixed fit grid (34, 36, 38), and the banked
    # hub→gemma maps are keyed to it. Absent from this registry until
    # 2026-07-28, which is what made the banked hub map unreachable from here
    # (the needle-preview gap: a bit-identical hub chart the composed tool could
    # not resolve). The vector side is a separate, still-open gap — see
    # `vector_bank_dirs`.
    "gemma3-27b": 36,
    # wave-1 graduations
    "qwen2.5-3b-instruct": 26,
    "qwen2.5-14b-instruct": 29,
    "qwen2.5-32b-instruct": 46,
    "mistral-7b-instruct-v0.3": 15,
    "olmo2-7b-instruct": 15,
    "phi-4": 19,
    "phi-3.5-mini-instruct": 13,
    # hub-rungs-2 graduations. The 70B's site of record is L37 (L43 is the
    # robustness site, L17 retired); gpt2-xl is deliberately ABSENT — its site
    # of record is DEFERRED (Luxia 2026-07-27), so it has no candidate slot.
    "llama-3.1-70b-instruct": 37,
    "pythia-6.9b": 31,
    # big-chain graduation (Luxia 2026-07-28). ⋆ L15, the same site and the same
    # fractional depth as its dense family-mate mistral-7b-instruct-v0.3 — the
    # MoE raises the ceiling, not the site (`fit_transport_maps.SITES` carries
    # the full rationale; the two registries must agree and are cross-checked in
    # selftest 7).
    "mixtral-8x7b-instruct-v0.1": 15,
}

#: Checkpoint identity for models NOT in `metabasis.roster.ROSTER` (which holds
#: only the wave-1 + hub-rungs-2 nodes). These are the carried banked nodes;
#: each is the prereg §1 roster row's checkpoint, named here so the arm rule
#: never depends on a substring match against a bank key.
CARRIED_CHECKPOINT_IDENTITY: dict[str, Literal["instruct", "base"]] = {
    "3b": "instruct",            # meta-llama/Llama-3.2-3B-Instruct (roster row 1)
    "8b": "instruct",            # meta-llama/Llama-3.1-8B-Instruct (row 2, the hub)
    "qwen-7b": "instruct",       # Qwen/Qwen2.5-7B-Instruct (row 3)
    "dsv2-lite": "instruct",     # deepseek-ai/DeepSeek-V2-Lite-Chat (row 4)
    "gemma3-27b": "instruct",    # google/gemma-3-27b-it (row 5)
    "olmo2-7b": "base",          # allenai/OLMo-2-1124-7B (row 6, RLHF-free)
}

#: Corpus provenance of a hub-map directory. The composed predictor consumes
#: whatever hub map is banked, so which CORPUS the fit was made on rides with
#: every prediction rather than living in a ledger row nobody reads at filing
#: time. `legacy` = the pre-freeze anamnesis corpus; `frozen` = the frozen
#: campaign corpus.
CorpusProvenance = Literal["frozen", "legacy"]


class HubMapDir(BaseModel):
    """One candidate directory for a banked primary-hub→model map."""
    path: Path
    corpus: CorpusProvenance
    note: str = ""

    model_config = {"arbitrary_types_allowed": True}


def hub_map_dirs(model: str) -> list[HubMapDir]:
    """Ordered candidate directories for the banked hub→`model` map.

    Order is PREFERENCE, and the preference is frozen-corpus-first: the
    collection-phase convention (`outputs/collection/<model>/fits_scan_<model>/`)
    is probed before any carried battery tree, so when a frozen-corpus refit
    lands for a node whose only banked map today is a legacy-corpus one, the
    refit supersedes automatically instead of needing a code change. Every
    probed path is echoed into the returned record either way.
    """
    dirs = [HubMapDir(path=COLLECTION_ROOT / model / f"fits_scan_{model}",
                      corpus="frozen",
                      note="collection-phase convention")]
    a8 = BANK_ROOT / "arms" / "A8_conjugation"
    if model == "3b":
        # Hub→3B maps were fit by the desk onto the frozen-corpus A8-smalls
        # banks and integrated to the pair-tree convention (LEDGER, hub-3b-maps
        # 2026-07-27) — not to a collection node, because 3B is a carried node
        # with no collection tree of its own.
        dirs.append(HubMapDir(path=Path("outputs") / "pairs" / "8b__3b"
                                   / "fits_8bL16__3bL14",
                              corpus="frozen",
                              note="carried node; desk-fit on the frozen-corpus "
                                   "A8-smalls banks, integrated to the pair tree"))
    if model == "dsv2-lite":
        # leg2 holds the shared corpus; the dsv2-lite bank is leg2/states and
        # was gate-verified frozen-corpus and clean (LEDGER, bank gate 4/4).
        # `fits_modefree` beside it is a DIFFERENT fit lineage and is never the
        # map of record — it is deliberately not probed.
        dirs.append(HubMapDir(path=a8 / "leg2" / "fits", corpus="frozen",
                              note="carried node; battery leg2 (shared frozen corpus)"))
    if model == "qwen-7b":
        # leg1 fits are LEGACY-CORPUS (LEDGER 2026-07-28 desk pre-check). They
        # are the only banked hub→qwen-7b maps today, so composed predictions
        # on qwen-7b pairs carry the flag until the frozen-corpus qwen-7b banks
        # land and the collection dir above starts resolving.
        dirs.append(HubMapDir(path=a8 / "leg1" / "fits", corpus="legacy",
                              note="carried node; battery leg1 — LEGACY-corpus fits"))
    if model == "gemma3-27b":
        # The banked hub→gemma maps live in the desk-smalls extension-pair tree.
        # FROZEN-corpus, and that is checked rather than assumed: the smalls
        # state banks both sides were fit against stamp
        # `corpus_manifest_sha256 = a6712ca0…`, the frozen campaign corpus —
        # the same bank family the hub→3b maps were desk-fit onto, and one of
        # the two trees rake M12 names as matching the collection-phase corpus.
        # `fits_gemma_modefree` beside it is a DIFFERENT fit lineage (as
        # `fits_modefree` is for dsv2-lite) and is deliberately NOT probed.
        dirs.append(HubMapDir(path=a8 / "smalls" / "fits_gemma", corpus="frozen",
                              note="carried node; desk-smalls extension pair "
                                   "(frozen-corpus smalls banks)"))
    return dirs


def vector_bank_paths(model: str, site: int) -> list[Path]:
    """Ordered candidate paths for `model`'s banked entropy-gradient vectors.

    Deliberately NARROW. `read_exchange_rates.target_vector_candidates` probes
    `a5_vectors.npz` banks too, which is right for a tool that knows which key
    it wants — but here a near-miss is worse than a miss: an `a5_vectors` bank
    holds the frozen-era object roster, NOT the entropy-gradient vector, and
    loading one would produce a plausible number for a different estimand. Only
    the entropy-gradient convention is probed (the merged bank first, then the
    per-site build name used where sites were built separately), and an absent
    bank is reported as data by `resolve_vector_bank` rather than resolved
    sideways into a different object.
    """
    vectors = COLLECTION_ROOT / model / "vectors"
    return [vectors / f"entropy_gradient_{model}.npz",
            vectors / f"entropy_gradient_{model}_L{site}.npz"]


def site_of_record(model: str) -> int:
    """The model's site of record, validated against its FIXED FIT GRID.

    Two failure modes, both loud: an unknown model key (no candidate slot
    exists for it) and a site that is not on the model's fit grid (a retired
    or mistyped site — `require_site` raises `FitGridError`).
    """
    site = SITE_OF_RECORD.get(model)
    if site is None:
        raise ComposedPathError(
            f"{model!r} has no site of record in this module's registry: it is "
            f"either not a candidate model (gpt2-xl's site of record is "
            f"DEFERRED and is deliberately absent) or the key is misspelled. "
            f"Known candidate keys: {sorted(SITE_OF_RECORD)}")
    return require_site(model, site)


def checkpoint_identity(model: str) -> Literal["instruct", "base"]:
    """`instruct` or `base` for `model` — the roster first, carried nodes next."""
    node = ROSTER.get(model)
    if node is not None:
        return node.checkpoint_identity
    identity = CARRIED_CHECKPOINT_IDENTITY.get(model)
    if identity is None:
        raise ComposedPathError(
            f"{model!r}: checkpoint identity unknown — it is in neither "
            f"metabasis.roster.ROSTER {sorted(ROSTER)} nor the carried-node "
            f"table {sorted(CARRIED_CHECKPOINT_IDENTITY)}. The applicable arm "
            f"cannot be resolved without it (prereg §3), and guessing from the "
            f"bank key's spelling is exactly the mistake this refusal prevents")
    return identity


def applicable_arm(model_a: str, model_b: str) -> tuple[str, str]:
    """The pair's applicable arm and the rule that produced it (prereg §3).

    instruct↔instruct → native. Anything involving a base checkpoint → raw:
    base models run raw ONLY (the arm-consistency rule, prereg §1 — their
    constants live in the raw system, never native), so a mixed pair has raw
    as its only system in common and a base↔base pair has raw as its only
    system at all.
    """
    ia, ib = checkpoint_identity(model_a), checkpoint_identity(model_b)
    if ia == "instruct" and ib == "instruct":
        return "native", "instruct↔instruct → native (prereg §3)"
    if ia == "base" and ib == "base":
        return "raw", "base↔base → raw (arm-consistency rule, prereg §1)"
    return "raw", "instruct↔base → raw (prereg §3)"


# ---------------------------------------------------------------- data model
class HubMapRef(BaseModel):
    """One side's banked primary-hub→model map, resolved or explicitly absent."""
    model: str
    site: int
    arm: str
    family: str
    hub: str = HUB_MODEL
    hub_site: int = HUB_SITE_OF_RECORD
    resolved: Optional[str] = Field(
        default=None, description="the fit npz actually used; None => N/A-AT-FILING")
    corpus: Optional[CorpusProvenance] = None
    dir_note: str = ""
    probed_paths: list[str] = []

    @property
    def available(self) -> bool:
        return self.resolved is not None


class VectorBankRef(BaseModel):
    """One side's banked entropy-gradient bank, resolved or explicitly absent.

    The vector is the OTHER half of a composed slot's backing. A hub map with
    no vector to move is exactly as unfilable as a vector with no map, and the
    absence is reported in the same shape so a roster sweep names the gap
    instead of dying on a `FileNotFoundError` halfway through.
    """
    model: str
    site: int
    resolved: Optional[str] = Field(
        default=None, description="the entropy-gradient npz actually used; None "
                                  "=> the target build has not landed")
    probed_paths: list[str] = []

    @property
    def available(self) -> bool:
        return self.resolved is not None


class ComposedPrediction(BaseModel):
    """One â_comp: a pair × arm × family, computed from banked hub maps ONLY."""
    prediction_id: str = Field(
        description="draft E2 ID shape: composed-prediction/<src>→<tgt>/<arm>-k128")
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    arm_rule: str
    family: str
    a_comp: float
    magnitude_only: bool = Field(
        description="draft E2 mirrors prereg §3: |â_comp| < .08 → sign unscored")
    ceiling_target_hub_map: float = Field(
        description="‖P_image(M_hub→B) v_B‖ — the hard cap on |â_comp| through "
                    "this map; reported beside, never multiplied in")
    dim_source: int
    dim_target: int
    dim_hub_side: int
    hub_map_source: HubMapRef
    hub_map_target: HubMapRef
    source_vector: VectorSpec
    target_vector: VectorSpec
    flags: list[str] = []


class NotFilable(BaseModel):
    """A candidate slot that is N/A-AT-FILING, and exactly which piece is absent.

    `missing_sides` names the pieces in the vocabulary the resolver uses —
    `"source hub map"`, `"target vector bank"`, … — so a reader never has to
    guess whether "source" meant the map or the vector. Two DIFFERENT rules put
    a slot here and `reason` names whichever applies:

      * a missing HUB MAP is draft E1's arm-availability rule verbatim;
      * a missing VECTOR BANK is the same shape of fact — nothing is banked, so
        there is nothing to compute — but it is NOT an E1 clause. It is
        reported rather than raised so a roster-wide sweep can name the gap;
        the desk rules on whether such a slot files under the same heading.
    """
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    arm_rule: str
    family: str
    verdict: Literal["N/A-AT-FILING"] = "N/A-AT-FILING"
    missing_sides: list[str]
    hub_map_source: HubMapRef
    hub_map_target: HubMapRef
    vector_bank_source: Optional[VectorBankRef] = None
    vector_bank_target: Optional[VectorBankRef] = None
    reason: str = (
        "draft E1 arm-availability rule: â_comp files only where BOTH hub maps "
        "exist banked in the pair's applicable arm. Never proxied from another "
        "arm, never backfilled after the pair is fit.")

    @staticmethod
    def reason_for(missing: Sequence[str]) -> str:
        """The rule that put this slot here, named rather than assumed."""
        maps = [m for m in missing if m.endswith("hub map")]
        vectors = [m for m in missing if m.endswith("vector bank")]
        parts: list[str] = []
        if maps:
            parts.append(
                "draft E1 arm-availability rule: â_comp files only where BOTH "
                "hub maps exist banked in the pair's applicable arm. Never "
                "proxied from another arm, never backfilled after the pair is "
                f"fit. Absent here: {', '.join(maps)}.")
        if vectors:
            parts.append(
                "The entropy-gradient target build has not landed for "
                f"{', '.join(vectors)}, so there is no vector to move — a "
                "GPU-side gap, not an arm-availability one. Reported as data so "
                "a roster sweep names it; never substituted from another bank.")
        return " ".join(parts)


class ComposedReadout(BaseModel):
    STATUS: str = (
        "UNSTAMPED — computation only. Fits nothing, refits nothing, FILES NO "
        "PREDICTION, moves no band, writes nothing under outputs/. Draft "
        "Addendum E §E1 is NOT RATIFIED. The desk rules.")
    estimand: str = (
        "â_comp(A→B) = cos( M_hub→B( M_hub→A^rev(v_A) ), v_B ), banked "
        "primary-hub maps at the rebuilt-L16 hub column, proc k128, the pair's "
        "applicable arm; reverse pass = the adjoint (Ω^T over the same PCA "
        "bases); unit-normalized fp64 input over the fp32 banked fits")
    generated: str
    hub: str
    hub_site: int
    family: str
    gate: Optional["GateResult"] = None
    predictions: list[ComposedPrediction] = []
    na_at_filing: list[NotFilable] = []


class GateRow(BaseModel):
    """One archived retrodiction, reproduced three ways and differenced."""
    pair_id: str
    arm: str
    family: str
    a_comp_tool: float = Field(
        description="this module's registry-driven computation, full fp64")
    a_comp_archive_recompute: float = Field(
        description="the archived glue's own path resolution + the same "
                    "arithmetic, executed by path from the archive, full fp64")
    a_comp_banked_json: float = Field(
        description="wp_composition.json's banked value (6 dp as archived)")
    a_comp_draft_display: float = Field(
        description="the 4-dp figure quoted in draft Addendum E §E1")
    delta_vs_archive_recompute: float
    delta_vs_banked_json: float
    display_matches: bool
    resolution_parity: bool = Field(
        description="both hub fit npz paths resolved identically by this "
                    "module and by the archive")
    tool_fit_source: str
    tool_fit_target: str
    archive_fit_source: str
    archive_fit_target: str
    passes: bool


class GateResult(BaseModel):
    """Draft E1's frozen implementation gate, as a re-runnable record."""
    STATUS: str = (
        "draft Addendum E §E1 implementation gate — reproduce all five archived "
        "retrodictions to |Δ| < 1e-8 at full precision. A tool that cannot "
        "reproduce the archive does not file.")
    generated: str
    tolerance: float = GATE_TOLERANCE
    banked_json_tolerance: float = BANKED_JSON_TOLERANCE
    archive_dir: str = str(ARCHIVE_ROOT)
    archive_sha256: dict[str, str] = {}
    archive_sha_verified: dict[str, bool] = {}
    rows: list[GateRow] = []
    n_rows: int = 0
    n_pass: int = 0
    passed: bool = False
    failures: list[str] = []


ComposedReadout.model_rebuild()


# ---------------------------------------------------------------- map algebra
def _source_dim(tm: TransportMap) -> int:
    """Dimension of the space `transport(·, 'fwd')` accepts."""
    if tm.kind == "proc":
        if tm.va is None:
            raise ComposedPathError("proc map missing `va` — cannot read source dim")
        return int(tm.va.shape[1])
    if tm.left is None:
        raise ComposedPathError("ridge map missing `left` — cannot read source dim")
    return int(tm.left.shape[0])


def _target_dim(tm: TransportMap) -> int:
    """Dimension of the space `transport(·, 'fwd')` emits into."""
    if tm.kind == "proc":
        if tm.vb is None:
            raise ComposedPathError("proc map missing `vb` — cannot read target dim")
        return int(tm.vb.shape[1])
    if tm.right is None:
        raise ComposedPathError("ridge map missing `right` — cannot read target dim")
    return int(tm.right.shape[1])


def adjoint_map(tm: TransportMap) -> TransportMap:
    """The map whose `fwd` IS `tm`'s `rev` — the adjoint identity, explicit.

    Ω_MA = Ω_AMᵀ over the SAME PCA bases: swap `va`/`vb`, transpose Ω, swap the
    median norms and invert the scale. Built for the selftest, which proves the
    construction agrees with `transport(direction='rev')` to machine precision
    rather than asserting the identity in prose. Nothing in the prediction path
    calls this — the prediction path uses `transport(..., 'rev')` directly, so
    the arithmetic of record is never re-derived here.
    """
    if tm.kind == "proc":
        if tm.va is None or tm.vb is None or tm.omega is None:
            raise ComposedPathError("proc map missing va/vb/omega — no adjoint")
        if tm.scale == 0.0:
            raise ComposedPathError("proc map has scale 0 — adjoint undefined")
        return TransportMap(kind="proc", src_norm=tm.tgt_norm, tgt_norm=tm.src_norm,
                            va=tm.vb, vb=tm.va, omega=tm.omega.T,
                            scale=1.0 / tm.scale)
    if tm.left is None or tm.right is None:
        raise ComposedPathError("ridge map missing left/right — no adjoint")
    return TransportMap(kind="ridge", src_norm=tm.tgt_norm, tgt_norm=tm.src_norm,
                        left=tm.right.T, right=tm.left.T)


def composed_exchange_rate(hub_to_source: TransportMap, hub_to_target: TransportMap,
                           v_source: np.ndarray, v_target: np.ndarray) -> float:
    """THE estimand — â_comp(A→B). Two banked hub maps, two banked vectors.

    `v_source` / `v_target` are the banked unit vectors in each model's RAW
    residual space; pass them unmodified, the maps own their normalization
    (`read_exchange_rates` normalization section). The arithmetic here is the
    archived operationalization's, called through the same functions the
    archive calls — not a re-derivation of it.
    """
    if v_source.ndim != 1 or v_target.ndim != 1:
        raise ComposedPathError(
            f"vectors must be 1-D, got {v_source.shape} / {v_target.shape}")
    d_src, d_tgt = _target_dim(hub_to_source), _target_dim(hub_to_target)
    if int(v_source.shape[0]) != d_src:
        raise ComposedPathError(
            f"source vector dim {v_source.shape[0]} != hub→source map's target "
            f"dim {d_src} — wrong map, wrong site, or wrong model")
    if int(v_target.shape[0]) != d_tgt:
        raise ComposedPathError(
            f"target vector dim {v_target.shape[0]} != hub→target map's target "
            f"dim {d_tgt} — wrong map, wrong site, or wrong model")
    hub_a, hub_b = _source_dim(hub_to_source), _source_dim(hub_to_target)
    if hub_a != hub_b:
        raise ComposedPathError(
            f"the two hub legs disagree on the HUB-side dimension "
            f"({hub_a} vs {hub_b}) — they were not fit against the same hub "
            f"site/bank, so the two-hop is not composable")

    v_at_hub = hub_to_source.transport(v_source, direction="rev")
    if int(v_at_hub.shape[0]) != hub_b:
        raise ComposedPathError(                                  # pragma: no cover
            f"reverse pass landed in dim {v_at_hub.shape[0]}, hub→target expects "
            f"{hub_b}")
    moved = hub_to_target.transport(v_at_hub, direction="fwd")
    return cos(moved, v_target)


# ---------------------------------------------------------------- resolution
def resolve_hub_map(model: str, site: int, arm: str, family: str) -> HubMapRef:
    """Find the banked primary-hub→`model` map, or report it absent as data.

    Absent is NOT an error here — draft E1 makes it the N/A-AT-FILING verdict,
    and the returned record carries every path probed so the desk can see what
    was looked for. Use `require_hub_map` when a caller must have the map.
    """
    probed: list[str] = []
    for candidate in hub_map_dirs(model):
        path = fit_path_for(candidate.path, HUB_MODEL, HUB_SITE_OF_RECORD,
                            model, site, arm, family)
        probed.append(str(path))
        if path.exists():
            return HubMapRef(model=model, site=site, arm=arm, family=family,
                             resolved=str(path), corpus=candidate.corpus,
                             dir_note=candidate.note, probed_paths=probed)
    return HubMapRef(model=model, site=site, arm=arm, family=family,
                     probed_paths=probed)


def require_hub_map(model: str, site: int, arm: str, family: str) -> HubMapRef:
    """`resolve_hub_map`, but a miss is a LOUD failure naming every path probed."""
    ref = resolve_hub_map(model, site, arm, family)
    if not ref.available:
        raise ComposedPathError(
            f"no banked primary-hub map for {model!r} L{site} in arm {arm!r} "
            f"(family {family}, hub {HUB_MODEL}L{HUB_SITE_OF_RECORD}). Probed, "
            f"in order:\n  " + "\n  ".join(ref.probed_paths)
            + "\n(draft E1 forbids proxying from another arm — if this pair's "
              "slot is genuinely unbacked it is N/A-AT-FILING, not a fallback)")
    return ref


def vector_bank(model: str) -> Path:
    """The canonical merged entropy-gradient bank for `model`."""
    return COLLECTION_ROOT / model / "vectors" / f"entropy_gradient_{model}.npz"


def resolve_vector_bank(model: str, site: int) -> VectorBankRef:
    """Find `model`'s banked entropy-gradient vectors, or report them absent.

    Absent is data, not an exception: a node whose hub map is banked but whose
    target build has not run yet (gemma3-27b today) must produce a NAMED gap
    rather than a `FileNotFoundError` from inside the predictor.
    """
    probed: list[str] = []
    for path in vector_bank_paths(model, site):
        probed.append(str(path))
        if path.exists():
            return VectorBankRef(model=model, site=site, resolved=str(path),
                                 probed_paths=probed)
    return VectorBankRef(model=model, site=site, probed_paths=probed)


# ---------------------------------------------------------------- the predictor
def compose_pair(source_model: str, target_model: str,
                 family: str = FAMILY_OF_RECORD,
                 arm: Optional[str] = None,
                 source_site: Optional[int] = None,
                 target_site: Optional[int] = None,
                 ) -> ComposedPrediction | NotFilable:
    """â_comp for one candidate slot, or the N/A-AT-FILING record for it.

    Sites default to each model's site of record (grid-validated); `arm`
    defaults to the pair's applicable arm (prereg §3). Overriding either is
    allowed for diagnostics and is echoed into the record — draft E1's filing
    path never overrides.
    """
    s_site = site_of_record(source_model) if source_site is None \
        else require_site(source_model, source_site)
    t_site = site_of_record(target_model) if target_site is None \
        else require_site(target_model, target_site)
    resolved_arm, arm_rule = applicable_arm(source_model, target_model)
    if arm is not None and arm != resolved_arm:
        arm_rule = (f"OVERRIDDEN to {arm!r} (applicable arm by prereg §3 is "
                    f"{resolved_arm!r}: {arm_rule})")
    use_arm = arm or resolved_arm
    if use_arm not in ARMS:
        raise ComposedPathError(f"unknown arm {use_arm!r}; known: {ARMS}")

    pair_id = f"{source_model}L{s_site}->{target_model}L{t_site}"
    ref_src = resolve_hub_map(source_model, s_site, use_arm, family)
    ref_tgt = resolve_hub_map(target_model, t_site, use_arm, family)
    vec_src = resolve_vector_bank(source_model, s_site)
    vec_tgt = resolve_vector_bank(target_model, t_site)
    missing = [label for label, ref in (("source hub map", ref_src),
                                        ("target hub map", ref_tgt),
                                        ("source vector bank", vec_src),
                                        ("target vector bank", vec_tgt))
               if not ref.available]
    if missing:
        return NotFilable(
            pair_id=pair_id, source_model=source_model, source_site=s_site,
            target_model=target_model, target_site=t_site, arm=use_arm,
            arm_rule=arm_rule, family=family, missing_sides=missing,
            hub_map_source=ref_src, hub_map_target=ref_tgt,
            vector_bank_source=vec_src, vector_bank_target=vec_tgt,
            reason=NotFilable.reason_for(missing))

    assert vec_src.resolved is not None and vec_tgt.resolved is not None
    v_src, spec_src = load_entropy_gradient(Path(vec_src.resolved),
                                            source_model, s_site)
    v_tgt, spec_tgt = load_entropy_gradient(Path(vec_tgt.resolved),
                                            target_model, t_site)
    assert ref_src.resolved is not None and ref_tgt.resolved is not None
    tm_src = load_transport_map(Path(ref_src.resolved))
    tm_tgt = load_transport_map(Path(ref_tgt.resolved))
    a_comp = composed_exchange_rate(tm_src, tm_tgt, v_src, v_tgt)

    flags: list[str] = []
    for side, ref in (("source", ref_src), ("target", ref_tgt)):
        if ref.corpus == "legacy":
            flags.append(
                f"LEGACY-CORPUS hub map on the {side} side ({ref.model}): "
                f"{ref.dir_note}. The composed value is computable, but the "
                f"map was fit on the pre-freeze corpus — flag rides the slot "
                f"until a frozen-corpus refit lands.")
    if arm is not None and arm != resolved_arm:
        flags.append("ARM OVERRIDDEN — diagnostic only; not a filable slot.")

    return ComposedPrediction(
        prediction_id=f"composed-prediction/{source_model}→{target_model}"
                      f"/{use_arm}-k128",
        pair_id=pair_id, source_model=source_model, source_site=s_site,
        target_model=target_model, target_site=t_site, arm=use_arm,
        arm_rule=arm_rule, family=family, a_comp=a_comp,
        magnitude_only=bool(abs(a_comp) < NEAR_ZERO_CARVE_OUT),
        ceiling_target_hub_map=projection_norm(image_basis(tm_tgt), v_tgt),
        dim_source=int(v_src.shape[0]), dim_target=int(v_tgt.shape[0]),
        dim_hub_side=_source_dim(tm_tgt),
        hub_map_source=ref_src, hub_map_target=ref_tgt,
        source_vector=spec_src, target_vector=spec_tgt, flags=flags)


# ---------------------------------------------------------------- candidates
class CandidateSlot(BaseModel):
    """One unordered candidate pair as enumerated by the banked naive gate."""
    pair_id: str
    model_a: str
    site_a: int
    model_b: str
    site_b: int


def load_candidate_slots(path: Path = CANDIDATE_PAIRS_JSON) -> list[CandidateSlot]:
    """The candidate pairs of record, read from the banked naive-gate column.

    The enumeration is NOT re-derived here: the gate column is the banked
    artifact that says which pairs exist today (66 rows; the gpt2-xl
    supplement is excluded there because that model's site of record is
    deferred). Each row's sites are cross-checked against this module's site
    registry — a disagreement is a HALT, not a silent preference, because it
    would mean the banked column and the fit-grid registry describe different
    objects.
    """
    if not path.exists():
        raise ComposedPathError(
            f"candidate-pair enumeration absent: {path}. This is the banked "
            f"naive-transplant gate column (draft E4 item 1); without it the "
            f"candidate roster would have to be re-derived, which is exactly "
            f"what the banked column exists to prevent")
    try:
        doc = json.loads(path.read_text())
        rows = doc["pairs"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ComposedPathError(f"unreadable candidate enumeration {path}: {exc}") from exc

    slots: list[CandidateSlot] = []
    problems: list[str] = []
    for row in rows:
        try:
            slot = CandidateSlot(pair_id=row["pair_id"], model_a=row["model_a"],
                                 site_a=int(row["site_a"]), model_b=row["model_b"],
                                 site_b=int(row["site_b"]))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"malformed row {row!r}: {exc}")
            continue
        for model, site in ((slot.model_a, slot.site_a), (slot.model_b, slot.site_b)):
            try:
                expected = site_of_record(model)
            except (ComposedPathError, FitGridError) as exc:
                problems.append(f"{slot.pair_id}: {exc}")
                continue
            if expected != site:
                problems.append(
                    f"{slot.pair_id}: banked column has {model} at L{site} but "
                    f"the site registry says L{expected}")
        slots.append(slot)
    if problems:
        raise ComposedPathError(
            f"candidate enumeration {path} disagrees with the site registry:\n  "
            + "\n  ".join(problems))
    return slots


def run_candidates(family: str = FAMILY_OF_RECORD,
                   pairs_json: Path = CANDIDATE_PAIRS_JSON) -> ComposedReadout:
    """â_comp (or N/A-AT-FILING) for every current candidate slot."""
    readout = ComposedReadout(generated=date.today().isoformat(), hub=HUB_MODEL,
                              hub_site=HUB_SITE_OF_RECORD, family=family)
    for slot in load_candidate_slots(pairs_json):
        result = compose_pair(slot.model_a, slot.model_b, family=family)
        if isinstance(result, NotFilable):
            readout.na_at_filing.append(result)
            logger.warning("N/A-AT-FILING %-52s %s::%s — missing %s",
                           result.pair_id, result.arm, family,
                           ", ".join(result.missing_sides))
            continue
        readout.predictions.append(result)
        logger.info("â_comp %-52s %s::%-9s = %+.6f  ceiling=%.4f%s%s",
                    result.pair_id, result.arm, family, result.a_comp,
                    result.ceiling_target_hub_map,
                    "  MAGNITUDE-ONLY" if result.magnitude_only else "",
                    "  FLAGGED" if result.flags else "")
    return readout


# ------------------------------------------------------- the resolution sweep
class ResolutionRow(BaseModel):
    """What this module's registries can reach for ONE model, and what they miss.

    `on_disk_hub_maps` is found by SEARCHING the output tree for the hub-map
    filename rather than by asking the registry where it should be — the whole
    point is to answer rake M3's question ("what has no record") in the
    resolution direction: which banked charts exist that the registry cannot
    see. A hit the registry deliberately declines (a different fit lineage, e.g.
    the `*_modefree` trees) is listed under `not_of_record` and is not a gap.
    """
    model: str
    site: int
    arm: str
    family: str
    hub_map: HubMapRef
    vector_bank: VectorBankRef
    on_disk_hub_maps: list[str] = []
    not_of_record: list[str] = []
    gap: Optional[str] = None

    @property
    def unreachable(self) -> bool:
        """A banked chart exists on disk and the registry resolves NOTHING."""
        return bool(self.on_disk_hub_maps) and not self.hub_map.available


class ResolutionSweep(BaseModel):
    """The registry-reachability record for every model with a site of record."""
    STATUS: str = (
        "registry reachability only — resolves paths, loads nothing, computes "
        "no â_comp, writes nothing under outputs/.")
    generated: str
    hub: str = HUB_MODEL
    hub_site: int = HUB_SITE_OF_RECORD
    arm_note: str = (
        "each model is probed in the arm its own checkpoint identity implies "
        "for a same-identity pair (instruct -> native, base -> raw); a pair's "
        "APPLICABLE arm is still resolved per pair by `applicable_arm`")
    family: str
    search_root: str = str(COLLECTION_ROOT.parent)
    searched: bool = Field(
        default=True,
        description="False => the output tree is not present, so the on-disk "
                    "leg was SKIPPED and this sweep proves reachability only")
    rows: list[ResolutionRow] = []
    unreachable: list[str] = []
    gaps: list[str] = []

    @property
    def passed(self) -> bool:
        """No model has a banked chart the registry cannot reach. Named gaps
        (nothing banked yet) are a RESULT, not a failure."""
        return not self.unreachable


def _sweep_arm(model: str) -> str:
    """The arm a same-identity pair of `model` with itself would resolve to."""
    return applicable_arm(model, model)[0]


def registry_resolution_sweep(family: str = FAMILY_OF_RECORD,
                              search: bool = True) -> ResolutionSweep:
    """Resolve every registered model's hub map + vector bank, and name the gaps.

    This is the guard the gemma3-27b gap slipped through: a banked, bit-identical
    hub chart sat in the desk-smalls tree while `SITE_OF_RECORD` had no key for
    the model at all, so nothing in the tool could even ASK for it — and no
    existing check compared what is banked against what is reachable.
    """
    root = COLLECTION_ROOT.parent
    do_search = search and root.exists()
    sweep = ResolutionSweep(generated=date.today().isoformat(), family=family,
                            searched=do_search)
    for model in sorted(SITE_OF_RECORD):
        site = site_of_record(model)
        arm = _sweep_arm(model)
        hub_map = resolve_hub_map(model, site, arm, family)
        vectors = resolve_vector_bank(model, site)
        name = fit_path_for(Path(), HUB_MODEL, HUB_SITE_OF_RECORD, model, site,
                            arm, family).name
        on_disk = sorted(str(p) for p in root.rglob(name)) if do_search else []
        row = ResolutionRow(model=model, site=site, arm=arm, family=family,
                            hub_map=hub_map, vector_bank=vectors,
                            on_disk_hub_maps=on_disk,
                            not_of_record=[p for p in on_disk
                                           if p != hub_map.resolved])
        missing: list[str] = []
        probed: list[str] = []
        if not hub_map.available:
            missing.append("hub map")
            probed.extend(hub_map.probed_paths)
        if not vectors.available:
            missing.append("entropy-gradient vector bank")
            probed.extend(vectors.probed_paths)
        if missing:
            row.gap = (f"{model} L{site} [{arm}::{family}]: no banked "
                       + " and no banked ".join(missing)
                       + f". Probed: {'; '.join(probed)}")
            sweep.gaps.append(row.gap)
        if row.unreachable:
            sweep.unreachable.append(
                f"{model} L{site} [{arm}::{family}]: {len(on_disk)} banked hub "
                f"map(s) ON DISK that the registry cannot reach — "
                f"{'; '.join(on_disk)}. Probed instead: "
                f"{'; '.join(hub_map.probed_paths)}")
        sweep.rows.append(row)
    return sweep


# ---------------------------------------------------------------- the gate
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(manifest: Path = ARCHIVE_MANIFEST,
                   names: Sequence[str] = ("wp_composition.py",
                                           "wp_composition.json"),
                   ) -> tuple[dict[str, str], dict[str, bool]]:
    """Check the archived operationalization against its own manifest.

    A sha mismatch on a NUMBER-BEARING artifact is a HALT: the gate compares
    against the archive, so an archive that is not the archive of record makes
    the comparison meaningless rather than merely stale.
    """
    if not manifest.exists():
        raise ArchiveIntegrityError(f"archive manifest absent: {manifest}")
    expected: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0]
    digests: dict[str, str] = {}
    verified: dict[str, bool] = {}
    bad: list[str] = []
    for name in names:
        path = manifest.parent / name
        if not path.exists():
            raise ArchiveIntegrityError(f"archived artifact absent: {path}")
        digest = sha256_of(path)
        digests[name] = digest
        want = expected.get(name)
        if want is None:
            raise ArchiveIntegrityError(
                f"{name} is not listed in {manifest} (has {sorted(expected)}) — "
                f"provenance of the operationalization of record is unverifiable")
        verified[name] = digest == want
        if digest != want:
            bad.append(f"{name}: manifest {want}, on disk {digest}")
    if bad:
        raise ArchiveIntegrityError(
            "HALT — the archived operationalization does not match its manifest:\n  "
            + "\n  ".join(bad))
    return digests, verified


def load_archived_glue(path: Path = ARCHIVE_GLUE) -> ModuleType:
    """Execute the archived glue by path, WITHOUT writing bytecode into it.

    RAKE M16: an `importlib` load-by-path drops a `__pycache__` beside the
    source, which violates read-only-over-`outputs/` and is invisible to
    `git status` because the tree is gitignored. `sys.dont_write_bytecode` is
    set across the load and restored afterwards. The module is NOT registered
    in `sys.modules` — it exists only for the duration of the gate.
    """
    if not path.exists():
        raise ArchiveIntegrityError(f"archived glue absent: {path}")
    spec = importlib.util.spec_from_file_location("wp_composition_archived", path)
    if spec is None or spec.loader is None:                      # pragma: no cover
        raise ArchiveIntegrityError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run_gate(archive_dir: Path = ARCHIVE_ROOT,
             tolerance: float = GATE_TOLERANCE) -> GateResult:
    """Draft E1's frozen implementation gate. Returns the record; never exits."""
    manifest = archive_dir / "MANIFEST-worstpair-diag.sha256"
    digests, verified = verify_archive(manifest)
    archived = load_archived_glue(archive_dir / "wp_composition.py")
    banked = json.loads((archive_dir / "wp_composition.json").read_text())
    banked_by_pair = {row["pair"]: float(row["a_composed_hub_mediated"])
                      for row in banked["rows"]}

    result = GateResult(generated=date.today().isoformat(), tolerance=tolerance,
                        archive_dir=str(archive_dir), archive_sha256=digests,
                        archive_sha_verified=verified)

    for spec in archived.PAIRS:
        pair_id = f"{spec.src}L{spec.src_site}->{spec.tgt}L{spec.tgt_site}"

        # --- leg 1: the archive's OWN resolution + arithmetic, full precision.
        a_src, _ = load_entropy_gradient(archived.vec_path(spec.src), spec.src,
                                         spec.src_site)
        a_tgt, _ = load_entropy_gradient(archived.vec_path(spec.tgt), spec.tgt,
                                         spec.tgt_site)
        arc_src = fit_path_for(archived.hub_fits_dir(spec.src), archived.HUB,
                               archived.HUB_SITE, spec.src, spec.src_site,
                               archived.ARM, archived.FAM)
        arc_tgt = fit_path_for(archived.hub_fits_dir(spec.tgt), archived.HUB,
                               archived.HUB_SITE, spec.tgt, spec.tgt_site,
                               archived.ARM, archived.FAM)
        tm_a, tm_b = load_transport_map(arc_src), load_transport_map(arc_tgt)
        a_archive = cos(tm_b.transport(tm_a.transport(a_src, direction="rev"),
                                       direction="fwd"), a_tgt)

        # --- leg 2: THIS module's registry-driven path.
        tool = compose_pair(spec.src, spec.tgt, family=archived.FAM)
        if isinstance(tool, NotFilable):
            result.failures.append(
                f"{pair_id}: this module reports N/A-AT-FILING for an archived "
                f"pair (missing {', '.join(tool.missing_sides)}) — the gate "
                f"cannot pass while the tool cannot even resolve the archive's "
                f"own maps")
            continue

        d_archive = abs(tool.a_comp - a_archive)
        json_value = banked_by_pair.get(pair_id)
        if json_value is None:
            result.failures.append(
                f"{pair_id}: absent from the banked wp_composition.json rows "
                f"{sorted(banked_by_pair)}")
            continue
        d_json = abs(tool.a_comp - json_value)
        display = DRAFT_DISPLAY_TARGETS.get(pair_id)
        if display is None:
            result.failures.append(
                f"{pair_id}: no draft display target registered — the draft's "
                f"five figures and the archive's five rows disagree")
            continue
        display_ok = round(tool.a_comp, DISPLAY_DECIMALS) == display
        parity = (tool.hub_map_source.resolved == str(arc_src)
                  and tool.hub_map_target.resolved == str(arc_tgt))
        passes = bool(d_archive < tolerance and d_json < BANKED_JSON_TOLERANCE
                      and display_ok and parity
                      and tool.arm == archived.ARM)

        row = GateRow(
            pair_id=pair_id, arm=tool.arm, family=tool.family,
            a_comp_tool=tool.a_comp, a_comp_archive_recompute=a_archive,
            a_comp_banked_json=json_value, a_comp_draft_display=display,
            delta_vs_archive_recompute=d_archive, delta_vs_banked_json=d_json,
            display_matches=display_ok, resolution_parity=parity,
            tool_fit_source=str(tool.hub_map_source.resolved),
            tool_fit_target=str(tool.hub_map_target.resolved),
            archive_fit_source=str(arc_src), archive_fit_target=str(arc_tgt),
            passes=passes)
        result.rows.append(row)
        if not passes:
            result.failures.append(
                f"{pair_id}: |Δ| vs archive {d_archive:.3e} (tol {tolerance:.0e}), "
                f"|Δ| vs banked json {d_json:.3e}, display_ok={display_ok}, "
                f"resolution_parity={parity}, arm={tool.arm}/{archived.ARM}")
        logger.info("GATE %-52s tool %+.17g  archive %+.17g  |Δ| %.3e  %s",
                    pair_id, row.a_comp_tool, row.a_comp_archive_recompute,
                    d_archive, "PASS" if passes else "FAIL")

    result.n_rows = len(result.rows)
    result.n_pass = sum(r.passes for r in result.rows)
    expected_rows = len(archived.PAIRS)
    result.passed = bool(not result.failures and result.n_rows == expected_rows
                         and result.n_pass == expected_rows)
    if result.n_rows != expected_rows:
        result.failures.append(
            f"gate produced {result.n_rows} rows for {expected_rows} archived pairs")
    return result


# ---------------------------------------------------------------- selftest
def _proc_map(rng: np.random.Generator, d_hub: int, d_model: int, k: int,
              src_norm: float, tgt_norm: float, scale: float) -> TransportMap:
    """A synthetic hub→model proc map: hub on the SOURCE side, model on target."""
    va = np.linalg.qr(rng.standard_normal((d_hub, k)))[0].T        # [k, d_hub]
    vb = np.linalg.qr(rng.standard_normal((d_model, k)))[0].T      # [k, d_model]
    omega = np.linalg.qr(rng.standard_normal((k, k)))[0]
    return TransportMap(kind="proc", src_norm=src_norm, tgt_norm=tgt_norm,
                        va=va, vb=vb, omega=omega, scale=scale)


def selftest() -> int:                                   # noqa: C901 — a checklist
    """Synthetic validation of the composed-path algebra and the registries."""
    rng = np.random.default_rng(A8_SEED)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    d_hub, d_a, d_b, k = 96, 64, 80, 24
    tm_a = _proc_map(rng, d_hub, d_a, k, src_norm=5.6915, tgt_norm=2.2506, scale=1.1541)
    tm_b = _proc_map(rng, d_hub, d_b, k, src_norm=5.6915, tgt_norm=3.9012, scale=0.8375)
    v_a = unit(rng.standard_normal(d_a))
    v_b = unit(rng.standard_normal(d_b))

    print("== selftest 1: the adjoint identity of record (Ω_MA = Ω_AMᵀ) ==")
    adj = adjoint_map(tm_a)
    lhs = tm_a.transport(v_a, direction="rev")
    rhs = adj.transport(v_a, direction="fwd")
    check(lhs.shape == (d_hub,), f"reverse pass lands in the hub space: {lhs.shape}")
    check(float(np.abs(lhs - rhs).max()) < 1e-12,
          f"transport(·,'rev') == the explicitly-constructed adjoint's fwd "
          f"(max |Δ| {float(np.abs(lhs - rhs).max()):.3e}) — the identity is "
          f"demonstrated, not asserted")
    probe_hub = unit(rng.standard_normal(d_hub))
    round_trip = adjoint_map(adj)
    check(float(np.abs(round_trip.transport(probe_hub, direction="fwd")
                       - tm_a.transport(probe_hub, direction="fwd")).max()) < 1e-12,
          "the adjoint of the adjoint reproduces the original map's fwd pass "
          "(the construction is an involution, as an adjoint must be)")

    print("== selftest 2: the self-composition identity — two-hop = projection ==")
    #  fwd(rev(v)) with the SAME map collapses to P_image(v), so â_comp(A→A) is
    #  exactly the â-ceiling. This is the strongest closed-form check available.
    a_self = composed_exchange_rate(tm_a, tm_a, v_a, v_a)
    ceil_a = projection_norm(image_basis(tm_a), v_a)
    check(abs(a_self - ceil_a) < 1e-12,
          f"â_comp(A→A) = {a_self:.15f} == ceiling {ceil_a:.15f} (identical maps "
          f"⇒ the two-hop IS the orthogonal projector onto the image)")
    projected = tm_a.transport(tm_a.transport(v_a, direction="rev"), direction="fwd")
    basis = image_basis(tm_a)
    check(float(np.abs(projected - (basis.T @ (basis @ v_a))).max()) < 1e-12,
          "the composed image equals P_image v elementwise, not merely in cosine")

    print("== selftest 3: the ceiling bound holds for genuine two-hops ==")
    a_comp = composed_exchange_rate(tm_a, tm_b, v_a, v_b)
    ceil_b = projection_norm(image_basis(tm_b), v_b)
    check(abs(a_comp) <= ceil_b + 1e-12,
          f"|â_comp| {abs(a_comp):.6f} <= ceiling(hub→B, v_B) {ceil_b:.6f}")
    #  Plant the target INSIDE hub→B's image: the ceiling is then 1, and â_comp
    #  must equal the raw in-subspace alignment with no ceiling deflation left.
    basis_b = image_basis(tm_b)
    planted = unit(basis_b.T @ (basis_b @ v_b))
    check(abs(projection_norm(basis_b, planted) - 1.0) < 1e-12,
          "the planted target sits exactly in hub→B's image (ceiling = 1)")
    a_planted = composed_exchange_rate(tm_a, tm_b, v_a, planted)
    check(abs(a_planted) <= 1.0 + 1e-12,
          f"â_comp against an in-image target is a genuine cosine "
          f"({a_planted:+.6f} ∈ [-1, 1])")
    check(abs(a_planted - a_comp / ceil_b) < 1e-12,
          "â_comp / ceiling is exactly the in-subspace alignment — the same "
          "factorization read_ahat_ceilings establishes for â")

    print("== selftest 4: normalization invariance (the silent-failure guard) ==")
    base = composed_exchange_rate(tm_a, tm_b, v_a, v_b)
    for c in (1e-6, 7.0, 1e3):
        check(abs(composed_exchange_rate(tm_a, tm_b, v_a * c, v_b) - base) < 1e-12,
              f"source vector scaled ×{c:g} leaves â_comp unchanged")
        check(abs(composed_exchange_rate(tm_a, tm_b, v_a, v_b * c) - base) < 1e-12,
              f"target vector scaled ×{c:g} leaves â_comp unchanged")
    tm_a2 = TransportMap(kind="proc", src_norm=tm_a.src_norm * 13.0,
                         tgt_norm=tm_a.tgt_norm / 29.0, va=tm_a.va, vb=tm_a.vb,
                         omega=tm_a.omega, scale=tm_a.scale * 3.0)
    tm_b2 = TransportMap(kind="proc", src_norm=tm_b.src_norm / 11.0,
                         tgt_norm=tm_b.tgt_norm * 5.0, va=tm_b.va, vb=tm_b.vb,
                         omega=tm_b.omega, scale=tm_b.scale / 2.0)
    check(abs(composed_exchange_rate(tm_a2, tm_b2, v_a, v_b) - base) < 1e-12,
          "perturbing src_norm/tgt_norm/scale on BOTH maps leaves â_comp "
          "bit-for-bit unchanged (the whole median-norm path cancels)")

    print("== selftest 5: dimension mismatches are caught, never broadcast ==")
    for label, args in (
            ("source vector of the wrong dim",
             (tm_a, tm_b, unit(rng.standard_normal(d_a + 1)), v_b)),
            ("target vector of the wrong dim",
             (tm_a, tm_b, v_a, unit(rng.standard_normal(d_b + 3)))),
            ("hub legs fit at different hub dims",
             (tm_a, _proc_map(rng, d_hub + 8, d_b, k, 1.0, 1.0, 1.0), v_a, v_b))):
        try:
            composed_exchange_rate(*args)                        # type: ignore[arg-type]
            check(False, f"{label} must raise")
        except ComposedPathError as exc:
            check(True, f"{label} raises: {exc!s:.66}")

    print("== selftest 6: the applicable-arm rule (prereg §3 / §1) ==")
    arm_native, rule_native = applicable_arm("qwen2.5-32b-instruct", "phi-4")
    check(arm_native == "native", f"instruct↔instruct → {arm_native} ({rule_native})")
    arm_mixed, rule_mixed = applicable_arm("3b", "pythia-6.9b")
    check(arm_mixed == "raw", f"instruct↔base → {arm_mixed} ({rule_mixed})")
    check(applicable_arm("pythia-6.9b", "gpt2-xl")[0] == "raw",
          "base↔base → raw (base models run raw only)")
    check(applicable_arm("dsv2-lite", "qwen-7b")[0] == "native",
          "carried instruct nodes resolve through the carried-identity table")
    try:
        applicable_arm("no-such-model", "phi-4")
        check(False, "an unknown model key must refuse to resolve an arm")
    except ComposedPathError as exc:
        check("neither" in str(exc), f"unknown key refuses loudly: {exc!s:.66}")

    print("== selftest 7: site-of-record resolution is grid-validated ==")
    #  EVERY key, not a sample: the two registries (this module's SITE_OF_RECORD
    #  and fit_transport_maps.SITES) must agree, and `site_of_record` proves it
    #  through `require_site`. A key added here whose site is off the model's
    #  fixed fit grid fails the sweep at selftest time, not at filing time.
    grid_problems: list[str] = []
    for key in sorted(SITE_OF_RECORD):
        try:
            site_of_record(key)
        except (ComposedPathError, FitGridError) as exc:
            grid_problems.append(f"{key}: {exc}")
    check(not grid_problems,
          f"all {len(SITE_OF_RECORD)} registered sites of record are on their "
          f"model's fixed fit grid" + ("" if not grid_problems
                                       else f" — {grid_problems}"))
    check(site_of_record("mixtral-8x7b-instruct-v0.1")
          == site_of_record("mistral-7b-instruct-v0.3") == 15,
          "the MoE rung and its dense family-mate share the site of record "
          "(⋆L15 both) — the sparsity raises the ceiling, not the site")
    check(site_of_record("gemma3-27b") == 36,
          "gemma3-27b resolves at its carried banked site L36 (the site the "
          "whole banked gemma object roster lives at)")
    check(site_of_record("llama-3.1-70b-instruct") == 37,
          "70B site of record is L37 (L43 robustness, L17 retired)")
    check(site_of_record("phi-3.5-mini-instruct") == 13,
          "phi-3.5-mini site of record is L13 (double-humped curve, 4-site grid)")
    try:
        site_of_record("gpt2-xl")
        check(False, "gpt2-xl has no site of record — must refuse")
    except ComposedPathError as exc:
        check("DEFERRED" in str(exc), f"gpt2-xl refuses loudly: {exc!s:.66}")
    try:
        compose_pair("llama-3.1-70b-instruct", "phi-4", source_site=17)
        check(False, "a retired site must refuse")
    except FitGridError as exc:
        check("NOT in its fixed fit grid" in str(exc),
              f"the 70B's retired L17 refuses: {exc!s:.60}")

    print("== selftest 8: an absent hub map is N/A-AT-FILING, never a proxy ==")
    absent = resolve_hub_map("phi-4", 19, "native", "proc_kNOPE")
    check(not absent.available and absent.probed_paths,
          f"a nonexistent family reports absent with {len(absent.probed_paths)} "
          f"path(s) probed, not an exception")
    try:
        require_hub_map("phi-4", 19, "native", "proc_kNOPE")
        check(False, "require_hub_map must raise on an absent map")
    except ComposedPathError as exc:
        check("Probed, in order" in str(exc) and "phi-4" in str(exc),
              "require_hub_map names the model and every path probed")
    check(len(hub_map_dirs("qwen-7b")) == 2
          and hub_map_dirs("qwen-7b")[1].corpus == "legacy",
          "qwen-7b probes the frozen-corpus collection dir FIRST, then the "
          "legacy-corpus battery leg1 fallback")
    check(hub_map_dirs("phi-4")[0].corpus == "frozen"
          and len(hub_map_dirs("phi-4")) == 1,
          "a collection-phase node has exactly one, frozen-corpus, candidate dir")
    gemma_dirs = hub_map_dirs("gemma3-27b")
    check(len(gemma_dirs) == 2 and gemma_dirs[1].path.name == "fits_gemma"
          and gemma_dirs[1].corpus == "frozen",
          f"gemma3-27b probes the collection convention first, then the "
          f"frozen-corpus desk-smalls extension-pair tree "
          f"({gemma_dirs[1].path})")
    check(all("modefree" not in str(d.path) for d in gemma_dirs
              + hub_map_dirs("dsv2-lite")),
          "no `*_modefree` tree is ever a candidate — a different fit lineage "
          "is never the map of record")
    check(all("a5_vectors" not in str(p) for p in vector_bank_paths("3b", 14)),
          "the vector-bank probe never reaches a frozen-era object-roster bank "
          "(a different estimand would load without erroring)")

    print("== selftest 9: the gate comparator fails when it should ==")
    #  Exercise the pass/fail arithmetic itself on synthetic numbers, so a
    #  green gate on real data cannot be a comparator that always says yes.
    truth = 0.58241361173118944
    for delta, want_pass in ((0.0, True), (1e-12, True), (9e-9, True),
                             (1.1e-8, False), (5e-4, False)):
        got = abs((truth + delta) - truth) < GATE_TOLERANCE
        check(got is want_pass,
              f"|Δ| {delta:.1e} vs tol {GATE_TOLERANCE:.0e} → "
              f"{'pass' if got else 'fail'} (want {'pass' if want_pass else 'fail'})")
    check(round(0.58241361173118944, DISPLAY_DECIMALS) == 0.5824,
          "the 4-dp display check is a real round-trip, not a tolerance")
    check(len(DRAFT_DISPLAY_TARGETS) == 5,
          f"the draft's five display targets are registered "
          f"({sorted(DRAFT_DISPLAY_TARGETS.values())})")

    print("== selftest 10: bytecode suppression is actually in force (M16) ==")
    #  Load a throwaway module by path and assert nothing lands beside it.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="composed_selftest_") as td:
        probe = Path(td) / "probe_glue.py"
        probe.write_text("VALUE = 41 + 1\n")
        before = sorted(p.name for p in Path(td).iterdir())
        flag_before = sys.dont_write_bytecode
        module = load_archived_glue(probe)
        after = sorted(p.name for p in Path(td).iterdir())
        check(getattr(module, "VALUE", None) == 42,
              "load_archived_glue executes the module by path")
        check(before == after,
              f"no bytecode written beside the source (before {before}, after {after})")
        check("wp_composition_archived" not in sys.modules,
              "the archived glue is never registered in sys.modules")
        check(sys.dont_write_bytecode == flag_before,
              f"sys.dont_write_bytecode restored to its prior value "
              f"({flag_before!r}) — the suppression is scoped to the load")
        #  and the control case: without the guard, CPython DOES write there.
        sys.dont_write_bytecode = False
        try:
            control = Path(td) / "control_glue.py"
            control.write_text("VALUE = 7\n")
            spec = importlib.util.spec_from_file_location("control_glue", control)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            check((Path(td) / "__pycache__").exists(),
                  "control: an unguarded load-by-path DOES drop __pycache__ — "
                  "rake M16 is a live hazard, and the guard above is what stops it")
        finally:
            sys.dont_write_bytecode = flag_before

    print("== selftest 11: archive integrity check refuses a tampered manifest ==")
    with tempfile.TemporaryDirectory(prefix="composed_manifest_") as td:
        root = Path(td)
        (root / "wp_composition.py").write_text("X = 1\n")
        good = sha256_of(root / "wp_composition.py")
        (root / "M.sha256").write_text(f"{good}  wp_composition.py\n")
        _, ver = verify_archive(root / "M.sha256", names=("wp_composition.py",))
        check(ver["wp_composition.py"] is True, "a matching sha verifies")
        (root / "wp_composition.py").write_text("X = 2\n")
        try:
            verify_archive(root / "M.sha256", names=("wp_composition.py",))
            check(False, "a mismatched sha must HALT")
        except ArchiveIntegrityError as exc:
            check("HALT" in str(exc), f"sha mismatch halts: {exc!s:.60}")
        try:
            verify_archive(root / "M.sha256", names=("not_listed.py",))
            check(False, "an absent artifact must raise")
        except ArchiveIntegrityError as exc:
            check(True, f"absent artifact raises: {exc!s:.60}")

    print("== selftest 12: every registered model's banked chart is REACHABLE ==")
    #  The gemma3-27b regression: a banked, bit-identical hub chart that the
    #  composed tool could not resolve because the model had no registry key at
    #  all. The check is deliberately asymmetric — "nothing banked yet" is a
    #  RESULT and is only required to be NAMED, while "banked on disk and
    #  unreachable from the registry" is a FAILURE.
    sweep = registry_resolution_sweep()
    if not sweep.searched:
        print(f"  [SKIP] the output tree is absent at {sweep.search_root!r} — "
              f"the on-disk leg of this check needs the data tree and is "
              f"NODE-OWED here, not silently passed")
    for row in sweep.rows:
        print(f"  {row.model:<28} L{row.site:<3} {row.arm:<7} "
              f"map={'yes' if row.hub_map.available else 'NO ':<3} "
              f"vec={'yes' if row.vector_bank.available else 'NO ':<3} "
              f"on-disk={len(row.on_disk_hub_maps)}"
              + (f"  ({len(row.not_of_record)} not of record)"
                 if row.not_of_record else ""))
    check(sweep.passed,
          "no registered model has a banked hub map on disk that the registry "
          "cannot reach" + ("" if sweep.passed else f" — {sweep.unreachable}"))
    if sweep.searched:
        gemma = next(r for r in sweep.rows if r.model == "gemma3-27b")
        check(gemma.hub_map.available,
              f"gemma3-27b's banked hub map RESOLVES (the needle-preview gap): "
              f"{gemma.hub_map.resolved}")
        check(bool(gemma.not_of_record),
              f"and the sibling fit lineage beside it stays deliberately "
              f"unprobed: {gemma.not_of_record}")
    for gap in sweep.gaps:
        print(f"  NAMED GAP: {gap[:150]}")
    check(all(row.gap is not None or (row.hub_map.available
                                      and row.vector_bank.available)
              for row in sweep.rows),
          "every model either fully resolves or carries a NAMED gap — no row "
          "is silently incomplete")

    print(f"\nselftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic validation of the composed algebra, the "
                         "registries and the gate comparator; no data needed")
    ap.add_argument("--gate", action="store_true",
                    help="draft E1's implementation gate: reproduce all five "
                         "archived retrodictions to |Δ| < 1e-8 at full "
                         "precision. EXITS NONZERO ON FAILURE.")
    ap.add_argument("--candidates", action="store_true",
                    help="â_comp / N/A-AT-FILING for every current candidate "
                         "slot, read from the banked naive-gate enumeration")
    ap.add_argument("--resolution-sweep", action="store_true",
                    help="registry reachability for every registered model: "
                         "hub map + entropy-gradient bank, against what is "
                         "actually banked on disk. EXITS NONZERO if any banked "
                         "hub map is unreachable from the registry.")
    ap.add_argument("--pairs-json", type=Path, default=CANDIDATE_PAIRS_JSON,
                    help=f"candidate-pair enumeration (default {CANDIDATE_PAIRS_JSON})")
    ap.add_argument("--archive-dir", type=Path, default=ARCHIVE_ROOT,
                    help="the archived operationalization of record")
    ap.add_argument("--family", default=FAMILY_OF_RECORD, choices=FAMILIES)
    ap.add_argument("--source-model", default=None, help="single-slot mode")
    ap.add_argument("--target-model", default=None, help="single-slot mode")
    ap.add_argument("--arm", default=None, choices=ARMS,
                    help="DIAGNOSTIC override of the applicable arm; the filing "
                         "path never overrides (draft E1 forbids arm proxying)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the readout JSON here (parents created). Never "
                         "point this inside outputs/ — this tool is read-only "
                         "over the data tree.")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not (args.gate or args.candidates or args.resolution_sweep
            or args.source_model):
        raise SystemExit("pass --selftest, --gate, --candidates, "
                         "--resolution-sweep, or --source-model/--target-model")

    readout = ComposedReadout(generated=date.today().isoformat(), hub=HUB_MODEL,
                              hub_site=HUB_SITE_OF_RECORD, family=args.family)
    status = 0

    if args.gate:
        try:
            gate = run_gate(args.archive_dir)
        except ArchiveIntegrityError as exc:
            #  An EXPECTED halt with a meaningful message — reported cleanly and
            #  nonzero, so a filing-time gate run says why it stopped instead of
            #  handing the operator a traceback. Every OTHER failure (dimension
            #  mismatch, unresolvable registry key) propagates with its stack:
            #  those are bugs, and a bug must not look like a tidy verdict.
            print(f"\nGATE HALT — {exc}")
            return 1
        readout.gate = gate
        print(f"\n{'GATE — draft Addendum E §E1':52s} {'tool':>20s} "
              f"{'archive':>20s} {'|Δ|':>11s} {'banked':>9s} {'disp':>7s} par")
        for row in gate.rows:
            print(f"{row.pair_id:52s} {row.a_comp_tool:+20.17f} "
                  f"{row.a_comp_archive_recompute:+20.17f} "
                  f"{row.delta_vs_archive_recompute:11.3e} "
                  f"{row.a_comp_banked_json:+9.6f} "
                  f"{row.a_comp_draft_display:+7.4f} "
                  f"{'ok' if row.resolution_parity else 'MISMATCH'}")
        print(f"\ngate: {gate.n_pass}/{gate.n_rows} rows within "
              f"|Δ| < {gate.tolerance:.0e} — "
              f"{'PASSED' if gate.passed else 'FAILED'}")
        for failure in gate.failures:
            print(f"  FAILURE: {failure}")
        if not gate.passed:
            status = 1

    if args.resolution_sweep:
        sweep = registry_resolution_sweep(args.family)
        print(f"\n{'registered model':<28} {'site':>5} {'arm':<7} {'hub map':<8} "
              f"{'vectors':<8} on-disk  resolved")
        for row in sweep.rows:
            print(f"{row.model:<28} {('L' + str(row.site)):>5} {row.arm:<7} "
                  f"{('banked' if row.hub_map.available else 'ABSENT'):<8} "
                  f"{('banked' if row.vector_bank.available else 'ABSENT'):<8} "
                  f"{len(row.on_disk_hub_maps):>7}  "
                  f"{row.hub_map.resolved or '—'}")
        if not sweep.searched:
            print(f"\nON-DISK LEG SKIPPED — no output tree at "
                  f"{sweep.search_root!r}; reachability only")
        for gap in sweep.gaps:
            print(f"  NAMED GAP: {gap}")
        for bad in sweep.unreachable:
            print(f"  UNREACHABLE: {bad}")
        print(f"\nresolution sweep: {len(sweep.rows)} model(s), "
              f"{len(sweep.gaps)} named gap(s), {len(sweep.unreachable)} "
              f"unreachable banked map(s) — "
              f"{'PASSED' if sweep.passed else 'FAILED'}")
        if not sweep.passed:
            status = 1

    if args.candidates:
        computed = run_candidates(args.family, args.pairs_json)
        readout.predictions.extend(computed.predictions)
        readout.na_at_filing.extend(computed.na_at_filing)
        total = len(computed.predictions) + len(computed.na_at_filing)
        print(f"\n{'candidate slot':52s} {'arm':7s} {'â_comp':>9s} "
              f"{'ceiling':>8s} {'src map':>8s} {'tgt map':>8s} verdict")
        for pred in computed.predictions:
            print(f"{pred.pair_id:52s} {pred.arm:7s} {pred.a_comp:+9.6f} "
                  f"{pred.ceiling_target_hub_map:8.4f} "
                  f"{(pred.hub_map_source.corpus or '?'):>8s} "
                  f"{(pred.hub_map_target.corpus or '?'):>8s} "
                  f"{'FLAGGED' if pred.flags else 'filable'}")
        for na in computed.na_at_filing:
            print(f"{na.pair_id:52s} {na.arm:7s} {'—':>9s} {'—':>8s} "
                  f"{'—':>8s} {'—':>8s} N/A-AT-FILING ({', '.join(na.missing_sides)})")
        print(f"\ncandidates: {len(computed.predictions)}/{total} filable, "
              f"{len(computed.na_at_filing)}/{total} N/A-AT-FILING")

    if args.source_model:
        if not args.target_model:
            raise SystemExit("--source-model needs --target-model")
        one = compose_pair(args.source_model, args.target_model,
                           family=args.family, arm=args.arm)
        if isinstance(one, NotFilable):
            readout.na_at_filing.append(one)
            print(f"{one.pair_id}: N/A-AT-FILING (missing "
                  f"{', '.join(one.missing_sides)})")
        else:
            readout.predictions.append(one)
            print(f"{one.pair_id} [{one.arm}::{one.family}] "
                  f"â_comp = {one.a_comp:+.17g}")

    payload = readout.model_dump_json(indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        logger.info("wrote %s (%d prediction(s), %d N/A-at-filing)", args.out,
                    len(readout.predictions), len(readout.na_at_filing))
    return status


if __name__ == "__main__":
    sys.exit(main())
