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

────────────────────────────────────────────────────────────────────────────────
CORPUS VINTAGE — THE v2.1 ROOT, AND WHY THE GATE REFUSES IT
────────────────────────────────────────────────────────────────────────────────
Addendum G §G1 makes corpus-v2.1 the go-forward basis for all future filings and
§G2(a) makes the corpus manifest sha ride EVERY quoted â. `--v21-root` points
the resolver at a v2.1 re-bank mirror: its `fits_v21_<model>/` hub maps and
`vectors/<model>/entropy_gradient_<model>_L<site>.npz` vectors are PREPENDED to
the existing preference-ordered probes, so a node with no v2.1 fit degrades to
the frozen tree instead of to N/A. Default is None — with no `--v21-root` the
probe lists are byte-identical to what they were before this option existed
(selftest 13 proves that path-by-path, and the E1 gate proves it on the data).

The root is not taken on trust: `set_v21_root` hashes the root's own
`corpus/corpus_manifest.json` and refuses unless it equals `CORPUS_SHA_V21`
(the constant is named WITH its vintage per rake M26). Only a prediction whose
FOUR resolved artifacts — both hub maps, both vectors — all came from that
verified root carries `corpus_manifest_sha256`; a partial resolution is flagged
MIXED-VINTAGE and carries no sha, because a sha that covers half a computation
is worse than none.

**`--gate` and `--v21-root` are MUTUALLY EXCLUSIVE, and `run_gate` refuses to
run while a v2.1 root is set.** The gate is a fixed-vintage proof ABOUT THE
ARCHIVE: it re-resolves the five archived v1 pairs through this module's own
`hub_map_dirs` and asserts RESOLUTION PARITY against the archive's own npz
files. A global v2.1 root would silently move that leg onto v2.1 objects — the
gate would either fail for the wrong reason or, worse, pass on the wrong
objects. Two independent guards, because one of them being bypassed
programmatically is exactly how this trap gets sprung.

────────────────────────────────────────────────────────────────────────────────
SCORING — A SEPARATE, DESK-INITIATED ACT
────────────────────────────────────────────────────────────────────────────────
`--score-record <filed record> --observed <desk-supplied â>` emits a
machine-readable scored-record JSON beside the filing record. **Nothing here
ever auto-scores at filing time**: no prediction path computes a verdict, the
mode requires an explicit observed-â artifact that only the desk can produce at
first-read, and it refuses to overwrite an existing scored record. This tool
makes the ARTIFACT; the desk does the scoring act and rules on what it means.
Rules are prereg §3 + Addendum E §E2 verbatim: frozen ±.05 absolute band as
FILED (never recomputed, never moved), |predicted| < .08 scored MAGNITUDE-ONLY
(sign unscored), the ±.04 hit reported descriptively beside. Campaign gates
(G-star-hit / G-comp-hit) are NOT evaluated here — they are aggregates over the
whole 190 and are the desk's read.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.scripts.read_composed_predictions --selftest
  python -m metabasis.scripts.read_composed_predictions --gate
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --out /tmp/claude-output/composed_candidates.json
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --v21-root <v2.1 re-bank root> --out /tmp/claude-output/composed_v21.json
  python -m metabasis.scripts.read_composed_predictions \
      --score-record outputs/collection/predictions/<record>.json \
      --observed /tmp/claude-output/observed-<batch>.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal, Optional, Sequence

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

#: Corpus-v2.1 manifest sha — ADDENDUM 2026-07-28-G §G1's go-forward collection
#: basis, and the sha every â computed on it must carry (§G2(a): "every quoted â
#: carries its corpus manifest sha"). Verified locally against the pulled v2.1
#: arm mirror's own `corpus/corpus_manifest.json`.
#: NAMED WITH ITS VINTAGE (rake M26): a version sweep must be able to SEE this
#: constant by name, not discover it holds the wrong vintage's digest.
CORPUS_SHA_V21 = "5ae355bc5d130f8e9c3ae426f5e71bf2b6e99c74b95369a874bec2abcd59b5d9"
#: Where a v2.1 re-bank root keeps the manifest whose sha pins its vintage.
V21_CORPUS_MANIFEST_RELPATH = Path("corpus") / "corpus_manifest.json"

#: Prereg §3 / Addendum E §E2: the FROZEN half-width of every scored band, for
#: both predictors. Never used to move a filed band — only to CHECK that a
#: filed band is the frozen one (`_check_filed_band`).
FROZEN_BAND_HALF_WIDTH = 0.05
#: Addendum E §E2: the ±.04 hit reported DESCRIPTIVELY beside the scored band
#: (Luxia ruling 2026-07-28 — recorded per pair at scoring, never a gate).
DESCRIPTIVE_BAND_HALF_WIDTH = 0.04
#: A filed band is 4-dp rounded arithmetic on a 4-dp predicted value; anything
#: further than this from `predicted ± .05` means the record is malformed, not
#: rounded, and scoring it would score a band nobody froze.
BAND_ARITHMETIC_TOLERANCE = 1e-6
#: Band edges are INCLUSIVE ("± .05 absolute"); this absorbs binary
#: representation error at an exact edge hit, nothing more.
BAND_EDGE_EPSILON = 1e-12

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


class CorpusVintageError(ComposedPathError):
    """A corpus-vintage root is not what it claims to be, or is set where it
    must not be (the E1 gate). Always LOUD: a vintage confusion produces
    numbers that are individually valid and collectively meaningless."""


class ScoringError(RuntimeError):
    """A scoring input cannot be read as what it claims to be.

    Scoring is a desk act performed ONCE per record against frozen bands; a
    malformed record, an unmatched observation, or a band that is not the
    frozen one must stop the run rather than produce a verdict nobody can
    audit.
    """


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
#: campaign corpus (v1); `v21` = the corpus-v2.1 re-bank (Addendum G §G1's
#: go-forward basis), reachable only when a verified `--v21-root` is set.
CorpusProvenance = Literal["frozen", "legacy", "v21"]

#: Which corpus vintage a whole PREDICTION rides on — a property of all four
#: resolved artifacts together, never of one side. `MIXED` is a first-class
#: result and is flagged, not silently averaged over.
PredictionVintage = Literal["v2.1", "pre-v2.1", "MIXED"]


class HubMapDir(BaseModel):
    """One candidate directory for a banked primary-hub→model map."""
    path: Path
    corpus: CorpusProvenance
    note: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ------------------------------------------------------- the corpus-v2.1 root
#: The verified corpus-v2.1 re-bank root, or None (the default, and the vintage
#: every existing caller gets). MODULE-LEVEL because the resolution helpers
#: (`hub_map_dirs`, `vector_bank_paths`) are called from deep inside the
#: predictor and from other tools with fixed signatures — but it is never set by
#: import and only ever set through `set_v21_root`, which VERIFIES the root.
V21_ROOT: Optional[Path] = None
#: The sha of the manifest that verified `V21_ROOT`, recomputed at set time.
#: Never assumed from the constant: a root that cannot prove its vintage is
#: refused, so this and `CORPUS_SHA_V21` agree by construction (rake M26 —
#: the constant is checked by value, not trusted by name).
V21_CORPUS_SHA: Optional[str] = None


def set_v21_root(root: Optional[Path],
                 expected_corpus_sha: str = CORPUS_SHA_V21,
                 manifest_relpath: Path = V21_CORPUS_MANIFEST_RELPATH,
                 ) -> Optional[str]:
    """Point the resolvers at a corpus-v2.1 re-bank root, or clear it (None).

    The root must prove its vintage: its own `corpus/corpus_manifest.json` is
    hashed and must equal `expected_corpus_sha`. Addendum G §G2(a) requires the
    corpus sha to ride every â — a root taken on the operator's word would
    stamp predictions with a digest nobody verified, which is the failure rake
    M26 describes at constant grain and rake M12 at bank grain.

    `expected_corpus_sha` is a parameter ONLY so the selftest can exercise the
    machinery against a synthetic root; every caller in the filing path takes
    the default.
    """
    global V21_ROOT, V21_CORPUS_SHA
    if root is None:
        V21_ROOT, V21_CORPUS_SHA = None, None
        return None
    root = Path(root)
    if not root.is_dir():
        raise CorpusVintageError(
            f"--v21-root {root} is not a directory. The v2.1 re-bank root is "
            f"the tree holding `fits_v21_<model>/`, `vectors/<model>/` and "
            f"`{manifest_relpath}`")
    manifest = root / manifest_relpath
    if not manifest.is_file():
        raise CorpusVintageError(
            f"v2.1 root {root} carries no {manifest_relpath} — its vintage "
            f"cannot be VERIFIED, and Addendum G §G2(a) requires every â to "
            f"carry a corpus manifest sha that was checked, not asserted")
    digest = sha256_of(manifest)
    if digest != expected_corpus_sha:
        raise CorpusVintageError(
            f"HALT — {manifest} hashes to {digest}, not the expected "
            f"{expected_corpus_sha}. This root is NOT the corpus-v2.1 basis of "
            f"record (Addendum G §G1); refusing rather than stamping "
            f"predictions with a vintage they do not have")
    V21_ROOT, V21_CORPUS_SHA = root, digest
    logger.info("corpus-v2.1 root set: %s (manifest sha %s… VERIFIED)",
                root, digest[:8])
    return digest


@contextmanager
def v21_root_scope(root: Optional[Path],
                   expected_corpus_sha: str = CORPUS_SHA_V21,
                   manifest_relpath: Path = V21_CORPUS_MANIFEST_RELPATH,
                   ) -> Iterator[Optional[str]]:
    """`set_v21_root` for the duration of a block, restored on any exit path."""
    global V21_ROOT, V21_CORPUS_SHA
    previous_root, previous_sha = V21_ROOT, V21_CORPUS_SHA
    try:
        yield set_v21_root(root, expected_corpus_sha, manifest_relpath)
    finally:
        V21_ROOT, V21_CORPUS_SHA = previous_root, previous_sha


def v21_hub_map_dir(model: str, root: Optional[Path] = None) -> Optional[HubMapDir]:
    """The v2.1 hub-map candidate dir for `model`, or None when no root is set.

    Layout of record for the v2.1 re-bank mirror (the pulled `arm-v21` tree):
    one flat `fits_v21_<model>/` per model, holding the same
    `fit_8bL<hub>__<model>L<site>_<arm>_<family>.npz` names
    `read_exchange_rates.fit_path_for` builds — so only the DIRECTORY differs
    from the frozen convention, never the filename.
    """
    base = V21_ROOT if root is None else Path(root)
    if base is None:
        return None
    return HubMapDir(path=base / f"fits_v21_{model}", corpus="v21",
                     note="corpus-v2.1 re-bank mirror (Addendum G §G1 "
                          "go-forward basis; root sha-verified at set time)")


def _under_v21_root(path: Optional[str | Path]) -> bool:
    """True iff `path` resolves inside the currently-set v2.1 root."""
    if path is None or V21_ROOT is None:
        return False
    try:
        Path(path).resolve().relative_to(V21_ROOT.resolve())
    except (ValueError, OSError):
        return False
    return True


def hub_map_dirs(model: str) -> list[HubMapDir]:
    """Ordered candidate directories for the banked hub→`model` map.

    Order is PREFERENCE, and the preference is frozen-corpus-first: the
    collection-phase convention (`outputs/collection/<model>/fits_scan_<model>/`)
    is probed before any carried battery tree, so when a frozen-corpus refit
    lands for a node whose only banked map today is a legacy-corpus one, the
    refit supersedes automatically instead of needing a code change. Every
    probed path is echoed into the returned record either way.

    When a VERIFIED corpus-v2.1 root is set (`set_v21_root`; off by default) its
    `fits_v21_<model>/` dir is PREPENDED — the go-forward vintage wins, and a
    node with no v2.1 fit falls through to exactly the list below rather than to
    N/A. With no root set the list is byte-identical to what it was before the
    option existed: selftest 13 proves that path-by-path for every registered
    model, and the E1 gate proves it on the data.
    """
    dirs: list[HubMapDir] = []
    v21 = v21_hub_map_dir(model)
    if v21 is not None:
        dirs.append(v21)
    dirs.append(HubMapDir(path=COLLECTION_ROOT / model / f"fits_scan_{model}",
                          corpus="frozen",
                          note="collection-phase convention"))
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

    A VERIFIED corpus-v2.1 root (off by default) PREPENDS its own per-site bank
    `vectors/<model>/entropy_gradient_<model>_L<site>.npz` — the same per-site
    naming convention the frozen tree already uses as its second probe, under
    the v2.1 mirror's per-model subdirectory. With no root set the list is
    byte-identical to what it was before the option existed.
    """
    paths: list[Path] = []
    if V21_ROOT is not None:
        paths.append(V21_ROOT / "vectors" / model
                     / f"entropy_gradient_{model}_L{site}.npz")
    vectors = COLLECTION_ROOT / model / "vectors"
    paths.extend([vectors / f"entropy_gradient_{model}.npz",
                  vectors / f"entropy_gradient_{model}_L{site}.npz"])
    return paths


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
    corpus: Optional[CorpusProvenance] = Field(
        default=None,
        description="`v21` iff the bank resolved inside a VERIFIED corpus-v2.1 "
                    "root. Left None otherwise ON PURPOSE: a vector build only "
                    "re-derives when its own recipe consumes the corpus "
                    "(Addendum F §3), so labelling a collection-tree bank with "
                    "a corpus vintage would assert what its stamp, not this "
                    "resolver, is entitled to say")
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
    corpus_vintage: PredictionVintage = Field(
        default="pre-v2.1",
        description="the vintage of ALL FOUR resolved artifacts together "
                    "(both hub maps, both vectors). `MIXED` means they "
                    "disagree and is flagged, never averaged over")
    corpus_manifest_sha256: Optional[str] = Field(
        default=None,
        description="Addendum G §G2(a) — the corpus manifest sha this â rides "
                    "on. Populated ONLY for a wholly-v2.1 prediction, from the "
                    "root's own manifest as verified at set time; None for the "
                    "pre-v2.1 trees (whose per-bank stamps are the record) and "
                    "None for MIXED, because a sha covering half a computation "
                    "is worse than no sha at all")
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
    v21_root: Optional[str] = Field(
        default=None, description="the corpus-v2.1 re-bank root in force, if any")
    corpus_manifest_sha256_v21: Optional[str] = Field(
        default=None, description="that root's own manifest sha, recomputed and "
                                  "verified against CORPUS_SHA_V21 at set time")
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
                                 corpus="v21" if _under_v21_root(path) else None,
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

    #  Addendum G §G2(a): the corpus sha rides the â — but only when the WHOLE
    #  computation rode one vintage. All four artifacts, or none.
    v21_sides = [ref.corpus == "v21"
                 for ref in (ref_src, ref_tgt, vec_src, vec_tgt)]
    if all(v21_sides):
        vintage: PredictionVintage = "v2.1"
        corpus_sha = V21_CORPUS_SHA
    elif any(v21_sides):
        vintage = "MIXED"
        corpus_sha = None
        flags.append(
            "MIXED-VINTAGE — some of {source hub map, target hub map, source "
            "vector, target vector} resolved inside the corpus-v2.1 root and "
            "some fell through to the pre-v2.1 trees: "
            + ", ".join(f"{label}={'v2.1' if is_v21 else 'pre-v2.1'}"
                        for label, is_v21 in zip(
                            ("source hub map", "target hub map",
                             "source vector", "target vector"), v21_sides))
            + ". The value is computable but carries NO corpus sha — Addendum "
              "G §G2(a) wants a vintage tag on every â, and a tag covering "
              "half a computation is worse than none. Cross-vintage reads are "
              "calibration, never scored (§G2(c)).")
    else:
        vintage = "pre-v2.1"
        corpus_sha = None

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
        corpus_vintage=vintage, corpus_manifest_sha256=corpus_sha,
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
    """Draft E1's frozen implementation gate. Returns the record; never exits.

    REFUSES to run while a corpus-v2.1 root is set. The gate's leg 2 re-resolves
    the five ARCHIVED (v1) pairs through this module's own `hub_map_dirs` and
    asserts resolution parity against the archive's own npz files; a v2.1 root
    prepends itself to that probe and silently moves the leg onto v2.1 objects —
    the gate would fail for a reason that is not about fidelity, or pass on the
    wrong objects. `--gate` and `--v21-root` are mutually exclusive at the CLI;
    this is the second, programmatic guard on the same trap.
    """
    if V21_ROOT is not None:
        raise CorpusVintageError(
            f"the E1 gate cannot run while a corpus-v2.1 root is set "
            f"({V21_ROOT}). The gate is a FIXED-VINTAGE proof about the "
            f"archived v1 operationalization — it resolves the five archived "
            f"pairs through hub_map_dirs() and asserts resolution parity "
            f"against the archive's own npz files. Clear the root "
            f"(`set_v21_root(None)`) and re-run; `--gate` and `--v21-root` are "
            f"mutually exclusive by design, not by accident")
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


# ---------------------------------------------------------------- scoring
#  A SEPARATE, DESK-INITIATED ACT. Nothing in the prediction path above reaches
#  anything below this line: `compose_pair`, `run_candidates` and `run_gate`
#  neither import nor call a scorer, and no filed record can be scored without
#  an observed-â artifact the desk supplies at first-read. What this section
#  builds is the ARTIFACT the scoring act has been missing — verdicts have lived
#  in canon prose only (slate-prep §review note, the hygiene gap this closes).
#
#  The rules are prereg §3 + Addendum E §E2 verbatim and are never re-derived:
#    * the band is the FILED band, read off the record; it is CHECKED to be the
#      frozen `predicted ± .05` and a record that disagrees is a HALT, not a
#      band this tool quietly repairs. Bands never move after filing.
#    * |predicted| < .08 → MAGNITUDE-ONLY: |observed| is scored against the band
#      of |predicted| and the SIGN IS UNSCORED (reported beside, descriptively).
#    * the ±.04 hit is DESCRIPTIVE and is reported only where the record filed
#      that band — this tool never invents one.
#  Campaign gates (G-star-hit, G-comp-hit, the superiority rule) are NOT
#  evaluated here: they are aggregates over the whole 190 with frozen
#  interpretive rules, and they are the desk's read, not a tool's verdict.

#: The two predictors racing under Addendum E. `star` = prereg §3's scalar
#: c_A·c_B; `composed` = E1's two-hop â_comp.
Predictor = Literal["star", "composed"]

#: Every verdict this tool can emit. `UNSCORED-NO-OBSERVATION` is a first-class
#: result: a first-read that covers part of a record must say which slots it did
#: not reach, never silently drop them from the denominator.
Verdict = Literal[
    "in-band",
    "out-of-band-high",
    "out-of-band-low",
    "magnitude-only-in-band",
    "magnitude-only-out-of-band-high",
    "magnitude-only-out-of-band-low",
    "UNSCORED-NO-OBSERVATION",
]


def utc_now() -> str:
    """The scoring timestamp, to the second, in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slot_key(source: str, target: str, arm: str, family: str) -> str:
    """The join key between a filed slot and an observed â. Direction MATTERS —
    â is orientation-dependent, and batch 3 ruled source→target as filed."""
    return f"{source}->{target}/{arm}-{family}"


class FiledPrediction(BaseModel):
    """One predictor's filed slot, as read off the record. Nothing is recomputed."""
    predictor: Predictor
    prediction_id: str
    predicted: float
    band: list[float] = Field(description="the FROZEN band as filed, [lo, hi]")
    magnitude_only: bool
    descriptive_band_04: Optional[list[float]] = Field(
        default=None, description="E2's descriptive ±.04 band, only where the "
                                  "record filed one; never invented here")
    corpus_manifest_sha256: Optional[str] = Field(
        default=None, description="the vintage this prediction was computed on "
                                  "(Addendum G §G2(a)); None on pre-G records")


class FiledSlot(BaseModel):
    """One pair × arm × family as filed, with every predictor filed for it."""
    key: str
    pair_id: str
    source: str
    target: str
    arm: str
    family: str
    direction_of_record: Optional[str] = None
    predictions: list[FiledPrediction] = []
    flags: list[str] = []


class FilingRecord(BaseModel):
    """A filed prediction record, parsed into slots. READ-ONLY over the record."""
    path: str
    sha256: str
    filed_utc: Optional[str] = None
    status: Optional[str] = None
    corpus_manifest_sha256: Optional[str] = None
    slots: list[FiledSlot] = []


class ObservedAhat(BaseModel):
    """One observed â, supplied by the DESK at first-read. Never computed here.

    `corpus_manifest_sha256` is the vintage the FIT rode. Addendum G §G2(b)
    makes within-vintage scoring the thing the frozen ceremony tests, and
    §G2(c) makes a cross-vintage comparison a calibration read that is never
    scored — so the fit's vintage is part of the observation, not an
    afterthought.
    """
    source: str
    target: str
    arm: str
    family: str = FAMILY_OF_RECORD
    a_hat: float
    corpus_manifest_sha256: Optional[str] = None
    fit_path: Optional[str] = None
    observed_utc: Optional[str] = None
    note: str = ""

    @property
    def key(self) -> str:
        return slot_key(self.source, self.target, self.arm, self.family)


class ObservationSet(BaseModel):
    """The desk's observed-â artifact for one filing record."""
    path: str
    sha256: str
    corpus_manifest_sha256: Optional[str] = Field(
        default=None, description="record-wide default vintage for rows that "
                                  "do not carry their own")
    observed_utc: Optional[str] = None
    note: str = ""
    observations: list[ObservedAhat] = []


class ScoredSlot(BaseModel):
    """One (slot × predictor) verdict, with everything needed to re-derive it."""
    key: str
    pair_id: str
    prediction_id: str
    predictor: Predictor
    source: str
    target: str
    arm: str
    family: str
    predicted: float
    band: list[float] = Field(description="the frozen band AS FILED")
    scored_band: list[float] = Field(
        description="the band actually compared against: the filed band, or "
                    "the band of |predicted| for a magnitude-only slot")
    magnitude_only: bool
    observed: Optional[float] = None
    scored_value: Optional[float] = Field(
        default=None, description="observed, or |observed| when magnitude-only")
    verdict: Verdict
    in_band: Optional[bool] = None
    error: Optional[float] = Field(
        default=None, description="observed − predicted (signed, always on the "
                                  "raw values — descriptive for magnitude-only)")
    abs_error: Optional[float] = None
    band_excess: Optional[float] = Field(
        default=None, description="distance outside the scored band; 0.0 in-band")
    sign_scored: bool = Field(
        description="False under the near-zero carve-out (prereg §3 / E2)")
    sign_agrees: Optional[bool] = Field(
        default=None, description="reported even when unscored — descriptive")
    descriptive_band_04: Optional[list[float]] = None
    descriptive_04_in_band: Optional[bool] = Field(
        default=None, description="E2's ±.04 hit — DESCRIPTIVE, never a gate")
    corpus_manifest_sha256_prediction: Optional[str] = None
    corpus_manifest_sha256_fit: Optional[str] = None
    cross_vintage: bool = Field(
        default=False,
        description="prediction and fit rode DIFFERENT corpus vintages "
                    "(both known and unequal) — Addendum G §G2(c) makes such a "
                    "comparison a calibration read, never scored")
    scored_of_record: bool = Field(
        description="False for an unobserved slot and for a cross-vintage "
                    "calibration read; only True slots enter the aggregates")
    filed_utc: Optional[str] = None
    observed_utc: Optional[str] = None
    scored_utc: str
    fit_path: Optional[str] = None
    flags: list[str] = []


class PredictorAggregate(BaseModel):
    """Counts for one predictor over one record. Counts only — no gate verdict."""
    predictor: Predictor
    n_filed: int = 0
    n_scored_of_record: int = 0
    n_unscored_no_observation: int = 0
    n_cross_vintage_excluded: int = 0
    n_in_band: int = 0
    n_out_of_band: int = 0
    n_out_of_band_high: int = 0
    n_out_of_band_low: int = 0
    n_magnitude_only: int = 0
    n_magnitude_only_in_band: int = 0
    in_band_fraction: Optional[float] = Field(
        default=None, description="n_in_band / n_scored_of_record; None when "
                                  "nothing in this record is scored of record")
    n_descriptive_04_reported: int = 0
    n_descriptive_04_in_band: int = 0
    descriptive_04_fraction: Optional[float] = None


class HeadToHead(BaseModel):
    """Addendum E §E3's per-pair 2×2, over slots where BOTH are scored of record."""
    n_slots_both_scored: int = 0
    n_both_in_band: int = 0
    n_star_only: int = 0
    n_composed_only: int = 0
    n_both_out_of_band: int = 0
    n_slots_incomplete: int = Field(
        default=0, description="slots where at least one predictor is not "
                               "scored of record — excluded from the 2×2")


class ScoredRecord(BaseModel):
    """The machine-readable scoring artifact for ONE filed prediction record."""
    STATUS: str = (
        "SCORED RECORD — the artifact of a desk scoring act. Fits nothing, "
        "refits nothing, files no prediction, moves no band. Verdicts are "
        "arithmetic against the FROZEN bands as filed (prereg §3 + Addendum E "
        "§E2); campaign gates (G-star-hit / G-comp-hit / the superiority rule) "
        "are aggregates over the whole 190 and are NOT evaluated here.")
    scoring_rules: str = (
        "band = the filed band, checked to be predicted ± .05 absolute and "
        "never recomputed; edges inclusive. |predicted| < .08 → MAGNITUDE-ONLY: "
        "|observed| scored against the band of |predicted|, sign unscored "
        "(reported beside). ±.04 reported descriptively where the record filed "
        "that band. A slot whose prediction and fit rode different corpus "
        "vintages is a calibration read (Addendum G §G2(c)): its verdict is "
        "computed and shown, but it is excluded from the aggregates.")
    generated: str
    scored_utc: str
    prediction_record: str
    prediction_record_sha256: str
    prediction_record_filed_utc: Optional[str] = None
    observations_source: str
    observations_sha256: str
    n_observations: int = 0
    n_observations_unmatched: int = 0
    tool: str = "metabasis/scripts/read_composed_predictions.py"
    slots: list[ScoredSlot] = []
    aggregates: list[PredictorAggregate] = []
    head_to_head: HeadToHead = HeadToHead()
    warnings: list[str] = []
    record_sha256: Optional[str] = Field(
        default=None, description="sha256 of this document with this field null "
                                  "— recompute by nulling it and re-hashing the "
                                  "canonical (sort_keys, compact) JSON")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """sha256 over a canonical JSON serialization — stable across runs."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check_filed_band(prediction_id: str, predicted: float,
                      band: Sequence[float]) -> list[float]:
    """The filed band must BE the frozen band. A HALT, never a repair."""
    if len(band) != 2:
        raise ScoringError(
            f"{prediction_id}: filed band {list(band)!r} is not a [lo, hi] pair")
    lo, hi = float(band[0]), float(band[1])
    if lo > hi:
        raise ScoringError(f"{prediction_id}: filed band [{lo}, {hi}] is inverted")
    want_lo = predicted - FROZEN_BAND_HALF_WIDTH
    want_hi = predicted + FROZEN_BAND_HALF_WIDTH
    if (abs(lo - want_lo) > BAND_ARITHMETIC_TOLERANCE
            or abs(hi - want_hi) > BAND_ARITHMETIC_TOLERANCE):
        raise ScoringError(
            f"{prediction_id}: filed band [{lo}, {hi}] is not the FROZEN "
            f"predicted ± {FROZEN_BAND_HALF_WIDTH} = [{want_lo}, {want_hi}] "
            f"(|Δ| lo {abs(lo - want_lo):.3e}, hi {abs(hi - want_hi):.3e}, tol "
            f"{BAND_ARITHMETIC_TOLERANCE:.0e}). Bands never move after filing, "
            f"so this record disagrees with the frozen contract and is NOT "
            f"scored — the desk rules on which side is wrong")
    return [lo, hi]


def _check_carve_out(prediction_id: str, predicted: float,
                     magnitude_only: bool) -> None:
    """The filed carve-out flag must be the frozen rule's own verdict."""
    expected = bool(abs(predicted) < NEAR_ZERO_CARVE_OUT)
    if expected != magnitude_only:
        raise ScoringError(
            f"{prediction_id}: the record files magnitude_only={magnitude_only} "
            f"for predicted {predicted:+.6f}, but the frozen near-zero carve-out "
            f"(prereg §3 / E2: |predicted| < {NEAR_ZERO_CARVE_OUT}) says "
            f"{expected}. Scoring on a carve-out flag that disagrees with the "
            f"frozen rule would score a contract nobody froze")


def _band_position(value: float, lo: float, hi: float) -> tuple[bool, str, float]:
    """(in_band, 'in'|'high'|'low', distance outside the band). Edges INCLUSIVE."""
    if value > hi + BAND_EDGE_EPSILON:
        return False, "high", value - hi
    if value < lo - BAND_EDGE_EPSILON:
        return False, "low", lo - value
    return True, "in", 0.0


def score_prediction(slot: FiledSlot, filed: FiledPrediction,
                     observed: Optional[ObservedAhat], *,
                     filed_utc: Optional[str] = None,
                     scored_utc: Optional[str] = None) -> ScoredSlot:
    """Score ONE filed prediction against ONE observed â. Pure arithmetic.

    Every branch is decided by the frozen rules and nothing else: no threshold
    is read from the environment, no band is recomputed, no verdict depends on
    the other predictor.
    """
    stamp = scored_utc or utc_now()
    band = _check_filed_band(filed.prediction_id, filed.predicted, filed.band)
    _check_carve_out(filed.prediction_id, filed.predicted, filed.magnitude_only)

    flags: list[str] = list(slot.flags)
    common = dict(
        key=slot.key, pair_id=slot.pair_id, prediction_id=filed.prediction_id,
        predictor=filed.predictor, source=slot.source, target=slot.target,
        arm=slot.arm, family=slot.family, predicted=filed.predicted, band=band,
        magnitude_only=filed.magnitude_only,
        descriptive_band_04=filed.descriptive_band_04,
        corpus_manifest_sha256_prediction=filed.corpus_manifest_sha256,
        sign_scored=not filed.magnitude_only,
        filed_utc=filed_utc, scored_utc=stamp)

    #  The scored band: magnitude-only moves the comparison onto |·| (prereg §3
    #  "|observed| within band of |predicted|; sign unscored"), and the band
    #  travels with it — the frozen half-width never changes.
    if filed.magnitude_only:
        scored_band = [abs(filed.predicted) - FROZEN_BAND_HALF_WIDTH,
                       abs(filed.predicted) + FROZEN_BAND_HALF_WIDTH]
    else:
        scored_band = band

    if observed is None:
        return ScoredSlot(scored_band=scored_band,
                          verdict="UNSCORED-NO-OBSERVATION", in_band=None,
                          scored_of_record=False,
                          flags=flags + ["NO OBSERVED â SUPPLIED for this slot — "
                                         "reported, never dropped from the "
                                         "denominator"],
                          **common)                     # type: ignore[arg-type]

    value = abs(observed.a_hat) if filed.magnitude_only else observed.a_hat
    in_band, where, excess = _band_position(value, scored_band[0], scored_band[1])
    if filed.magnitude_only:
        verdict: Verdict = ("magnitude-only-in-band" if in_band
                            else f"magnitude-only-out-of-band-{where}")  # type: ignore[assignment]
    else:
        verdict = "in-band" if in_band else f"out-of-band-{where}"  # type: ignore[assignment]

    descriptive_hit: Optional[bool] = None
    if filed.descriptive_band_04 is not None:
        d_lo, d_hi = (float(filed.descriptive_band_04[0]),
                      float(filed.descriptive_band_04[1]))
        if filed.magnitude_only:
            d_lo = abs(filed.predicted) - DESCRIPTIVE_BAND_HALF_WIDTH
            d_hi = abs(filed.predicted) + DESCRIPTIVE_BAND_HALF_WIDTH
        descriptive_hit = _band_position(value, d_lo, d_hi)[0]

    fit_sha = observed.corpus_manifest_sha256
    cross_vintage = bool(filed.corpus_manifest_sha256 and fit_sha
                         and filed.corpus_manifest_sha256 != fit_sha)
    scored_of_record = not cross_vintage
    if cross_vintage:
        flags.append(
            f"CROSS-VINTAGE — the prediction rode corpus "
            f"{filed.corpus_manifest_sha256[:8]}… and the fit rode "
            f"{(fit_sha or '')[:8]}…. Addendum G §G2(c): a cross-vintage â "
            f"comparison is a CALIBRATION READ, never scored. The verdict is "
            f"shown for inspection and EXCLUDED from the aggregates; the desk "
            f"rules on whether a within-vintage refit is owed.")
    if filed.corpus_manifest_sha256 is None or fit_sha is None:
        flags.append(
            "VINTAGE UNVERIFIED — "
            + ("the filed prediction carries no corpus manifest sha"
               if filed.corpus_manifest_sha256 is None else "")
            + ("; " if filed.corpus_manifest_sha256 is None and fit_sha is None
               else "")
            + ("the observed â carries no corpus manifest sha"
               if fit_sha is None else "")
            + ". Addendum G §G2(a) wants one on both sides; pre-G records have "
              "none, so within-vintage scoring here rests on the desk's own "
              "provenance trail, not on this artifact.")

    return ScoredSlot(
        scored_band=scored_band, observed=observed.a_hat, scored_value=value,
        verdict=verdict, in_band=in_band,
        error=observed.a_hat - filed.predicted,
        abs_error=abs(observed.a_hat - filed.predicted), band_excess=excess,
        sign_agrees=((filed.predicted >= 0.0) == (observed.a_hat >= 0.0)),
        descriptive_04_in_band=descriptive_hit,
        corpus_manifest_sha256_fit=fit_sha, cross_vintage=cross_vintage,
        scored_of_record=scored_of_record, observed_utc=observed.observed_utc,
        fit_path=observed.fit_path, flags=flags,
        **common)                                       # type: ignore[arg-type]


def _filed_prediction_from(block: dict[str, Any], predictor: Predictor,
                           fallback_sha: Optional[str]) -> FiledPrediction:
    """Read one predictor's block out of a racing-shaped filing record row."""
    try:
        band = list(block["band"])
        return FiledPrediction(
            predictor=predictor, prediction_id=str(block["id"]),
            predicted=float(block["predicted"]), band=[float(b) for b in band],
            magnitude_only=bool(block["magnitude_only"]),
            descriptive_band_04=([float(b) for b in block["descriptive_band_04"]]
                                 if block.get("descriptive_band_04") else None),
            corpus_manifest_sha256=block.get("corpus_manifest_sha256",
                                             fallback_sha))
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoringError(
            f"malformed {predictor} prediction block (keys {sorted(block)}): "
            f"{exc}") from exc


def parse_filing_record(path: Path) -> FilingRecord:
    """Read a filed prediction record into slots. TWO shapes are on record.

      * RACING (batch 3 onward): a row carries `source`/`target` plus a
        `star_prediction` and/or `composed_prediction` block.
      * SCALAR-FLAT (canary, batch 2): a row IS the star prediction —
        `source_model`/`target_model`, `predicted_a_hat`, `band`,
        `near_zero_carveout`.

    Any third shape is a HALT naming the keys it saw: guessing which number is
    the prediction is exactly how a scoring artifact becomes unauditable.
    """
    if not path.is_file():
        raise ScoringError(f"filed prediction record absent: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ScoringError(f"unreadable prediction record {path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("predictions"), list):
        raise ScoringError(
            f"{path}: not a filing record — expected a top-level object with a "
            f"`predictions` list, got keys "
            f"{sorted(doc) if isinstance(doc, dict) else type(doc).__name__}")

    record_sha = doc.get("corpus_manifest_sha256")
    record = FilingRecord(path=str(path), sha256=sha256_of(path),
                          filed_utc=doc.get("filed_utc"),
                          status=doc.get("STATUS"),
                          corpus_manifest_sha256=record_sha)
    seen: dict[str, str] = {}
    for row in doc["predictions"]:
        if not isinstance(row, dict):
            raise ScoringError(f"{path}: prediction row is not an object: {row!r}")
        row_sha = row.get("corpus_manifest_sha256", record_sha)
        filed: list[FiledPrediction] = []
        if "star_prediction" in row or "composed_prediction" in row:
            source, target = str(row["source"]), str(row["target"])
            arm, family = str(row["arm"]), str(row["family"])
            for predictor, block_key in (("star", "star_prediction"),
                                         ("composed", "composed_prediction")):
                block = row.get(block_key)
                if isinstance(block, dict):
                    filed.append(_filed_prediction_from(
                        block, predictor, row_sha))  # type: ignore[arg-type]
        elif "predicted_a_hat" in row:
            source, target = str(row["source_model"]), str(row["target_model"])
            arm, family = str(row["arm"]), str(row["family"])
            filed.append(FiledPrediction(
                predictor="star", prediction_id=str(row["id"]),
                predicted=float(row["predicted_a_hat"]),
                band=[float(b) for b in row["band"]],
                magnitude_only=bool(row.get("near_zero_carveout", False)),
                corpus_manifest_sha256=row_sha))
        else:
            raise ScoringError(
                f"{path}: unrecognized prediction-row shape (keys "
                f"{sorted(row)}). Known shapes: racing "
                f"(`star_prediction`/`composed_prediction` blocks) and "
                f"scalar-flat (`predicted_a_hat`). A third shape must be taught "
                f"to this parser explicitly — never guessed")
        if not filed:
            raise ScoringError(
                f"{path}: row {source}->{target} files no predictor block at all")
        key = slot_key(source, target, arm, family)
        if key in seen:
            raise ScoringError(
                f"{path}: duplicate slot {key} — a filing record must name each "
                f"pair×arm×family once (rake M18: comparisons key by the FULL "
                f"identity and assert no duplicates)")
        seen[key] = key
        record.slots.append(FiledSlot(
            key=key, pair_id=f"{source}->{target}", source=source, target=target,
            arm=arm, family=family,
            direction_of_record=row.get("direction_of_record"),
            predictions=filed, flags=[str(f) for f in row.get("flags", [])]))
    return record


def load_observations(path: Path) -> ObservationSet:
    """Read the desk's observed-â artifact. Shape is documented in `--help`.

    ```json
    {"corpus_manifest_sha256": "<the FIT vintage, default for all rows>",
     "observed": [{"source": "...", "target": "...", "arm": "native",
                   "family": "proc_k128", "a_hat": 0.41,
                   "corpus_manifest_sha256": "...", "fit_path": "...",
                   "observed_utc": "..."}]}
    ```
    """
    if not path.is_file():
        raise ScoringError(
            f"observed-â artifact absent: {path}. Scoring REQUIRES one — this "
            f"tool never observes, never fits, and never auto-scores at filing "
            f"time; the desk supplies the observed â at first-read")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ScoringError(f"unreadable observation set {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ScoringError(f"{path}: expected a top-level object")
    rows = doc.get("observed", doc.get("observations"))
    if not isinstance(rows, list):
        raise ScoringError(
            f"{path}: expected an `observed` (or `observations`) list, got keys "
            f"{sorted(doc)}")
    default_sha = doc.get("corpus_manifest_sha256")
    default_utc = doc.get("observed_utc")
    obs: list[ObservedAhat] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ScoringError(f"{path}: observation is not an object: {row!r}")
        try:
            obs.append(ObservedAhat(
                source=str(row["source"]), target=str(row["target"]),
                arm=str(row["arm"]),
                family=str(row.get("family", FAMILY_OF_RECORD)),
                a_hat=float(row["a_hat"]),
                corpus_manifest_sha256=row.get("corpus_manifest_sha256",
                                               default_sha),
                fit_path=row.get("fit_path"),
                observed_utc=row.get("observed_utc", default_utc),
                note=str(row.get("note", ""))))
        except (KeyError, TypeError, ValueError) as exc:
            raise ScoringError(
                f"{path}: malformed observation {row!r}: {exc}") from exc
    keys: dict[str, int] = {}
    for i, o in enumerate(obs):
        if o.key in keys:
            raise ScoringError(
                f"{path}: duplicate observation for {o.key} (rows {keys[o.key]} "
                f"and {i}) — which â is of record cannot be guessed")
        keys[o.key] = i
    return ObservationSet(path=str(path), sha256=sha256_of(path),
                          corpus_manifest_sha256=default_sha,
                          observed_utc=default_utc,
                          note=str(doc.get("note", "")), observations=obs)


def aggregate_scored(slots: Sequence[ScoredSlot]) -> list[PredictorAggregate]:
    """Counts per predictor. Only `scored_of_record` slots enter the fraction."""
    by_predictor: dict[str, PredictorAggregate] = {}
    for s in slots:
        agg = by_predictor.setdefault(
            s.predictor, PredictorAggregate(predictor=s.predictor))
        agg.n_filed += 1
        if s.verdict == "UNSCORED-NO-OBSERVATION":
            agg.n_unscored_no_observation += 1
            continue
        if s.cross_vintage:
            agg.n_cross_vintage_excluded += 1
            continue
        agg.n_scored_of_record += 1
        if s.magnitude_only:
            agg.n_magnitude_only += 1
            if s.in_band:
                agg.n_magnitude_only_in_band += 1
        if s.in_band:
            agg.n_in_band += 1
        else:
            agg.n_out_of_band += 1
            if s.verdict.endswith("high"):
                agg.n_out_of_band_high += 1
            elif s.verdict.endswith("low"):
                agg.n_out_of_band_low += 1
        if s.descriptive_04_in_band is not None:
            agg.n_descriptive_04_reported += 1
            if s.descriptive_04_in_band:
                agg.n_descriptive_04_in_band += 1
    for agg in by_predictor.values():
        if agg.n_scored_of_record:
            agg.in_band_fraction = agg.n_in_band / agg.n_scored_of_record
        if agg.n_descriptive_04_reported:
            agg.descriptive_04_fraction = (agg.n_descriptive_04_in_band
                                           / agg.n_descriptive_04_reported)
    return [by_predictor[k] for k in sorted(by_predictor)]


def head_to_head(slots: Sequence[ScoredSlot]) -> HeadToHead:
    """Addendum E §E3's 2×2, over slots where BOTH predictors scored of record."""
    by_key: dict[str, dict[str, ScoredSlot]] = {}
    for s in slots:
        by_key.setdefault(s.key, {})[s.predictor] = s
    h2h = HeadToHead()
    for pair in by_key.values():
        star, comp = pair.get("star"), pair.get("composed")
        if (star is None or comp is None or not star.scored_of_record
                or not comp.scored_of_record):
            h2h.n_slots_incomplete += 1
            continue
        h2h.n_slots_both_scored += 1
        if star.in_band and comp.in_band:
            h2h.n_both_in_band += 1
        elif star.in_band:
            h2h.n_star_only += 1
        elif comp.in_band:
            h2h.n_composed_only += 1
        else:
            h2h.n_both_out_of_band += 1
    return h2h


def score_record(record_path: Path, observations_path: Path,
                 scored_utc: Optional[str] = None) -> ScoredRecord:
    """Build the scored record for one filing record. WRITES NOTHING.

    Unmatched observations are a HALT: an observation that matches no filed slot
    is a typo, a wrong record, or a direction flip, and every one of those
    produces a scoring artifact that looks complete and is not.
    """
    stamp = scored_utc or utc_now()
    filing = parse_filing_record(record_path)
    observed = load_observations(observations_path)
    by_key = {o.key: o for o in observed.observations}

    filed_keys = {slot.key for slot in filing.slots}
    unmatched = sorted(set(by_key) - filed_keys)
    if unmatched:
        raise ScoringError(
            f"{observations_path}: {len(unmatched)} observation(s) match no "
            f"filed slot in {record_path.name}: {unmatched}. Filed slots: "
            f"{sorted(filed_keys)}. Direction of record is source→target as "
            f"filed and â is orientation-dependent — an unmatched key is a "
            f"HALT, never a skipped row")

    scored: list[ScoredSlot] = []
    for slot in filing.slots:
        obs = by_key.get(slot.key)
        for filed in slot.predictions:
            scored.append(score_prediction(slot, filed, obs,
                                           filed_utc=filing.filed_utc,
                                           scored_utc=stamp))

    warnings: list[str] = []
    n_missing = sum(1 for s in scored if s.verdict == "UNSCORED-NO-OBSERVATION")
    if n_missing:
        warnings.append(
            f"{n_missing} filed prediction(s) have no observed â in "
            f"{observations_path.name} — recorded UNSCORED-NO-OBSERVATION and "
            f"excluded from the fractions, never dropped")
    n_cross = sum(1 for s in scored if s.cross_vintage)
    if n_cross:
        warnings.append(
            f"{n_cross} slot(s) are CROSS-VINTAGE calibration reads (Addendum G "
            f"§G2(c)) — verdicts shown, excluded from the aggregates")

    record = ScoredRecord(
        generated=date.today().isoformat(), scored_utc=stamp,
        prediction_record=str(record_path),
        prediction_record_sha256=filing.sha256,
        prediction_record_filed_utc=filing.filed_utc,
        observations_source=str(observations_path),
        observations_sha256=observed.sha256,
        n_observations=len(observed.observations),
        n_observations_unmatched=0,
        slots=scored, aggregates=aggregate_scored(scored),
        head_to_head=head_to_head(scored), warnings=warnings)
    record.record_sha256 = _canonical_sha256(record.model_dump(mode="json"))
    return record


def default_scored_path(record_path: Path) -> Path:
    """Beside the filing record, named for it: `scored-<record name>`."""
    return record_path.parent / f"scored-{record_path.name}"


def write_scored_record(record: ScoredRecord, out: Path,
                        overwrite: bool = False) -> Path:
    """Write the scored record. Refuses to clobber an existing one by default."""
    if out.exists() and not overwrite:
        raise ScoringError(
            f"a scored record already exists at {out}. Scoring happens ONCE per "
            f"record; re-scoring it would silently replace a verdict that may "
            f"already be ledgered. Pass --overwrite-scored deliberately, or "
            f"write elsewhere with --scored-out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record.model_dump_json(indent=1))
    logger.info("wrote scored record %s (%d slot(s), sha %s…)", out,
                len(record.slots), (record.record_sha256 or "")[:12])
    return out


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

    print("== selftest 13: the corpus-v2.1 root — default-off, prepend-only ==")
    #  The whole point of this check is the NEGATIVE: with no root set, every
    #  probe list must be what it was before the option existed. The frozen
    #  paths are captured FIRST, compared after a root is set and cleared, and
    #  the E1 gate proves the same invariant on the real data.
    check(V21_ROOT is None and V21_CORPUS_SHA is None,
          "V21_ROOT defaults to None — no import sets a vintage")
    check(len(CORPUS_SHA_V21) == 64
          and all(c in "0123456789abcdef" for c in CORPUS_SHA_V21),
          f"CORPUS_SHA_V21 is a sha256 and is NAMED with its vintage (rake "
          f"M26): {CORPUS_SHA_V21[:12]}…")
    frozen_hub = {m: [(str(d.path), d.corpus, d.note) for d in hub_map_dirs(m)]
                  for m in sorted(SITE_OF_RECORD)}
    frozen_vec = {m: [str(p) for p in vector_bank_paths(m, site_of_record(m))]
                  for m in sorted(SITE_OF_RECORD)}
    with tempfile.TemporaryDirectory(prefix="composed_v21_") as td:
        root = Path(td)
        (root / V21_CORPUS_MANIFEST_RELPATH).parent.mkdir(parents=True)
        (root / V21_CORPUS_MANIFEST_RELPATH).write_text(
            '{"synthetic": "corpus-v2.1 stand-in for the selftest"}\n')
        synthetic_sha = sha256_of(root / V21_CORPUS_MANIFEST_RELPATH)
        try:
            set_v21_root(root)                      # the REAL expected sha
            check(False, "a root whose manifest sha is not CORPUS_SHA_V21 must "
                         "be refused")
        except CorpusVintageError as exc:
            check("HALT" in str(exc) and CORPUS_SHA_V21 in str(exc),
                  f"a root that cannot prove its vintage is REFUSED: {exc!s:.60}")
        check(V21_ROOT is None, "a refused root leaves V21_ROOT unset")
        try:
            set_v21_root(root / "nope", expected_corpus_sha=synthetic_sha)
            check(False, "a nonexistent root must be refused")
        except CorpusVintageError as exc:
            check("not a directory" in str(exc),
                  f"a nonexistent root refuses loudly: {exc!s:.60}")
        bare = root / "bare"
        bare.mkdir()
        try:
            set_v21_root(bare, expected_corpus_sha=synthetic_sha)
            check(False, "a root with no corpus manifest must be refused")
        except CorpusVintageError as exc:
            check("cannot be VERIFIED" in str(exc),
                  f"a root with no manifest refuses loudly: {exc!s:.60}")

        with v21_root_scope(root, expected_corpus_sha=synthetic_sha) as sha:
            check(sha == synthetic_sha and V21_CORPUS_SHA == synthetic_sha,
                  "a verified root records the sha it actually hashed, not the "
                  "one it was told to expect")
            prepend_ok, tail_ok = True, True
            for model in sorted(SITE_OF_RECORD):
                dirs = hub_map_dirs(model)
                prepend_ok &= (dirs[0].corpus == "v21"
                               and dirs[0].path == root / f"fits_v21_{model}")
                tail_ok &= ([(str(d.path), d.corpus, d.note) for d in dirs[1:]]
                            == frozen_hub[model])
                paths = vector_bank_paths(model, site_of_record(model))
                prepend_ok &= (paths[0] == root / "vectors" / model
                               / f"entropy_gradient_{model}_L{site_of_record(model)}.npz")
                tail_ok &= ([str(p) for p in paths[1:]] == frozen_vec[model])
            check(prepend_ok, f"every one of the {len(SITE_OF_RECORD)} registered "
                              f"models gains the v2.1 hub-map dir and per-site "
                              f"vector bank at the FRONT of its probe list")
            check(tail_ok, "and the rest of every probe list is byte-identical "
                           "to the frozen list — the prepend adds, never edits")
            try:
                run_gate()
                check(False, "run_gate must refuse while a v2.1 root is set")
            except CorpusVintageError as exc:
                check("FIXED-VINTAGE" in str(exc),
                      f"the E1 gate refuses a v2.1 root programmatically — the "
                      f"resolution-parity trap, closed: {exc!s:.60}")
        check(V21_ROOT is None and V21_CORPUS_SHA is None,
              "v21_root_scope restores the previous (unset) root on exit")
        after_hub = {m: [(str(d.path), d.corpus, d.note) for d in hub_map_dirs(m)]
                     for m in sorted(SITE_OF_RECORD)}
        after_vec = {m: [str(p) for p in vector_bank_paths(m, site_of_record(m))]
                     for m in sorted(SITE_OF_RECORD)}
        check(after_hub == frozen_hub and after_vec == frozen_vec,
              "with the root cleared, EVERY probe list is byte-identical to "
              "before it was ever set (the frozen resolution is untouched)")
        try:
            main(["--gate", "--v21-root", str(root)])
            check(False, "--gate with --v21-root must not run")
        except SystemExit as exc:
            check(exc.code == 2,
                  f"--gate and --v21-root are MUTUALLY EXCLUSIVE at the CLI "
                  f"(argparse exit {exc.code})")
        check(V21_ROOT is None, "the refused CLI combination set no root")

    print("== selftest 14: scoring — every verdict branch, on synthetic slots ==")

    def _slot(src: str, tgt: str, arm: str = "native") -> FiledSlot:
        return FiledSlot(key=slot_key(src, tgt, arm, FAMILY_OF_RECORD),
                         pair_id=f"{src}->{tgt}", source=src, target=tgt,
                         arm=arm, family=FAMILY_OF_RECORD)

    def _filed(predictor: str, predicted: float, band04: bool = False,
               sha: Optional[str] = None) -> FiledPrediction:
        return FiledPrediction(
            predictor=predictor,                     # type: ignore[arg-type]
            prediction_id=f"{predictor}-prediction/synthetic/native-k128",
            predicted=predicted,
            band=[predicted - FROZEN_BAND_HALF_WIDTH,
                  predicted + FROZEN_BAND_HALF_WIDTH],
            magnitude_only=abs(predicted) < NEAR_ZERO_CARVE_OUT,
            descriptive_band_04=([predicted - DESCRIPTIVE_BAND_HALF_WIDTH,
                                  predicted + DESCRIPTIVE_BAND_HALF_WIDTH]
                                 if band04 else None),
            corpus_manifest_sha256=sha)

    def _obs(src: str, tgt: str, a_hat: float, arm: str = "native",
             sha: Optional[str] = None) -> ObservedAhat:
        return ObservedAhat(source=src, target=tgt, arm=arm,
                            family=FAMILY_OF_RECORD, a_hat=a_hat,
                            corpus_manifest_sha256=sha)

    slot = _slot("alpha", "beta")
    branches: list[tuple[str, float, Optional[float], str, Optional[bool]]] = [
        ("comfortably inside the band",        0.30, 0.32, "in-band", True),
        ("above the band",                     0.30, 0.40, "out-of-band-high", False),
        ("below the band",                     0.30, 0.20, "out-of-band-low", False),
        ("exactly on the upper edge",          0.30, 0.30 + 0.05, "in-band", True),
        ("exactly on the lower edge",          0.30, 0.30 - 0.05, "in-band", True),
        ("near-zero, |obs| inside, SIGN FLIPPED",
         0.03, -0.05, "magnitude-only-in-band", True),
        ("near-zero, magnitude too large",     0.03, -0.20,
         "magnitude-only-out-of-band-high", False),
        ("near-zero, magnitude too small",     0.07, 0.005,
         "magnitude-only-out-of-band-low", False),
    ]
    for label, predicted, observation, want_verdict, want_in in branches:
        filed = _filed("composed", predicted)
        got = score_prediction(slot, filed,
                               _obs("alpha", "beta", observation)  # type: ignore[arg-type]
                               if observation is not None else None)
        check(got.verdict == want_verdict and got.in_band is want_in,
              f"{label}: predicted {predicted:+.3f} observed "
              f"{observation:+.4f} → {got.verdict} (want {want_verdict})")
    near_zero = score_prediction(slot, _filed("composed", 0.03),
                                 _obs("alpha", "beta", -0.05))
    check(near_zero.magnitude_only and not near_zero.sign_scored
          and near_zero.sign_agrees is False and near_zero.scored_value == 0.05,
          "the near-zero carve-out scores |observed| and marks the sign "
          "UNSCORED while still reporting that it disagrees (prereg §3 / E2)")
    check(near_zero.scored_band == [abs(0.03) - 0.05, abs(0.03) + 0.05],
          f"magnitude-only moves the comparison onto the band of |predicted| "
          f"{near_zero.scored_band}")
    unscored = score_prediction(slot, _filed("star", 0.30), None)
    check(unscored.verdict == "UNSCORED-NO-OBSERVATION"
          and unscored.in_band is None and not unscored.scored_of_record,
          "a slot with no observed â is UNSCORED-NO-OBSERVATION, not a miss")
    d04_hit = score_prediction(slot, _filed("composed", 0.30, band04=True),
                               _obs("alpha", "beta", 0.32))
    d04_miss = score_prediction(slot, _filed("composed", 0.30, band04=True),
                                _obs("alpha", "beta", 0.345))
    check(d04_hit.descriptive_04_in_band is True
          and d04_miss.descriptive_04_in_band is False
          and d04_miss.verdict == "in-band",
          "the ±.04 read is DESCRIPTIVE and independent: .345 is in the scored "
          "±.05 band and outside the descriptive ±.04 one")
    check(score_prediction(slot, _filed("composed", 0.30),
                           _obs("alpha", "beta", 0.32)).descriptive_04_in_band
          is None,
          "a record that filed no ±.04 band gets none invented for it")
    cross = score_prediction(slot, _filed("composed", 0.30, sha="a" * 64),
                             _obs("alpha", "beta", 0.32, sha="b" * 64))
    check(cross.cross_vintage and not cross.scored_of_record
          and cross.verdict == "in-band"
          and any("CALIBRATION READ" in f for f in cross.flags),
          "prediction and fit on DIFFERENT vintages → verdict computed, slot "
          "excluded from the aggregates (Addendum G §G2(c))")
    same = score_prediction(slot, _filed("composed", 0.30, sha="a" * 64),
                            _obs("alpha", "beta", 0.32, sha="a" * 64))
    check(not same.cross_vintage and same.scored_of_record
          and not any("VINTAGE UNVERIFIED" in f for f in same.flags),
          "same vintage on both sides scores of record with no vintage flag")
    try:
        score_prediction(slot, FiledPrediction(
            predictor="star", prediction_id="star-prediction/bad/native-k128",
            predicted=0.30, band=[0.25, 0.36], magnitude_only=False),
            _obs("alpha", "beta", 0.30))
        check(False, "a band that is not predicted ± .05 must HALT")
    except ScoringError as exc:
        check("FROZEN" in str(exc), f"a moved band HALTs: {exc!s:.60}")
    try:
        score_prediction(slot, FiledPrediction(
            predictor="star", prediction_id="star-prediction/bad2/native-k128",
            predicted=0.03, band=[-0.02, 0.08], magnitude_only=False),
            _obs("alpha", "beta", 0.03))
        check(False, "a carve-out flag disagreeing with the frozen rule must HALT")
    except ScoringError as exc:
        check("near-zero carve-out" in str(exc),
              f"a wrong magnitude_only flag HALTs: {exc!s:.60}")

    agg_slots = [
        score_prediction(_slot("a", "b"), _filed("star", 0.30),
                         _obs("a", "b", 0.32)),
        score_prediction(_slot("a", "b"), _filed("composed", 0.30),
                         _obs("a", "b", 0.32)),
        score_prediction(_slot("c", "d"), _filed("star", 0.30),
                         _obs("c", "d", 0.60)),
        score_prediction(_slot("c", "d"), _filed("composed", 0.30),
                         _obs("c", "d", 0.32)),
        score_prediction(_slot("e", "f"), _filed("star", 0.30),
                         _obs("e", "f", 0.60)),
        score_prediction(_slot("e", "f"), _filed("composed", 0.30),
                         _obs("e", "f", 0.60)),
        score_prediction(_slot("g", "h"), _filed("star", 0.30), None),
        score_prediction(_slot("g", "h"), _filed("composed", 0.30), None),
    ]
    aggs = {a.predictor: a for a in aggregate_scored(agg_slots)}
    check(aggs["star"].n_filed == 4 and aggs["star"].n_scored_of_record == 3
          and aggs["star"].n_in_band == 1 and aggs["star"].n_out_of_band == 2
          and aggs["star"].n_unscored_no_observation == 1
          and abs((aggs["star"].in_band_fraction or 0) - 1 / 3) < 1e-12,
          f"star aggregate counts 1/3 in band over the SCORED slots only "
          f"(fraction {aggs['star'].in_band_fraction})")
    check(aggs["composed"].n_in_band == 2 and aggs["composed"].n_out_of_band == 1,
          "composed aggregate counts 2 in band, 1 out")
    h2h = head_to_head(agg_slots)
    check((h2h.n_slots_both_scored, h2h.n_both_in_band, h2h.n_star_only,
           h2h.n_composed_only, h2h.n_both_out_of_band, h2h.n_slots_incomplete)
          == (3, 1, 0, 1, 1, 1),
          f"the E3 2×2 over slots where both scored: both {h2h.n_both_in_band}, "
          f"star-only {h2h.n_star_only}, composed-only {h2h.n_composed_only}, "
          f"both-out {h2h.n_both_out_of_band}, incomplete "
          f"{h2h.n_slots_incomplete}")

    print("== selftest 15: scoring is a SEPARATE act — never automatic ==")
    check(not any(f in ComposedPrediction.model_fields
                  for f in ("observed", "verdict", "in_band", "scored")),
          "a ComposedPrediction has no observed/verdict field at all — the "
          "filing path cannot carry a score even by accident")
    with tempfile.TemporaryDirectory(prefix="composed_scoring_") as td:
        root = Path(td)
        record_path = root / "predictions-synthetic-2026-07-28.json"
        record_path.write_text(json.dumps({
            "STATUS": "SYNTHETIC — selftest fixture, files nothing",
            "filed_utc": "2026-07-28",
            "corpus_manifest_sha256": "c" * 64,
            "predictions": [
                {"source": "alpha", "target": "beta", "arm": "native",
                 "family": FAMILY_OF_RECORD,
                 "direction_of_record": "forward as listed",
                 "star_prediction": {
                     "id": "star-prediction/alpha→beta/native-k128",
                     "predicted": 0.30, "band": [0.25, 0.35],
                     "magnitude_only": False},
                 "composed_prediction": {
                     "id": "composed-prediction/alpha→beta/native-k128",
                     "predicted": 0.34, "band": [0.29, 0.39],
                     "descriptive_band_04": [0.30, 0.38],
                     "magnitude_only": False}},
                {"id": "star-prediction/gamma→delta/native-k128",
                 "source_model": "gamma", "target_model": "delta",
                 "arm": "native", "family": FAMILY_OF_RECORD,
                 "predicted_a_hat": 0.04, "band": [-0.01, 0.09],
                 "near_zero_carveout": True}],
        }, indent=1))
        obs_path = root / "observed.json"
        obs_path.write_text(json.dumps({
            "corpus_manifest_sha256": "c" * 64,
            "observed_utc": "2026-07-28T00:00:00+00:00",
            "observed": [
                {"source": "alpha", "target": "beta", "arm": "native",
                 "family": FAMILY_OF_RECORD, "a_hat": 0.33},
                {"source": "gamma", "target": "delta", "arm": "native",
                 "family": FAMILY_OF_RECORD, "a_hat": -0.06}],
        }, indent=1))
        scored = score_record(record_path, obs_path)
        by_id = {s.prediction_id: s for s in scored.slots}
        check(len(scored.slots) == 3,
              "both record shapes parse: 2 racing predictors + 1 scalar-flat row")
        check(by_id["star-prediction/alpha→beta/native-k128"].verdict == "in-band"
              and by_id["composed-prediction/alpha→beta/native-k128"].verdict
              == "in-band",
              "the racing row scores both predictors against their own bands")
        gamma = by_id["star-prediction/gamma→delta/native-k128"]
        check(gamma.verdict == "magnitude-only-in-band" and not gamma.sign_scored
              and gamma.sign_agrees is False,
              "the scalar-flat row's near-zero carve-out scores on magnitude "
              "with the sign unscored (observed −.06 vs predicted +.04)")
        check(all(s.scored_of_record for s in scored.slots)
              and scored.record_sha256 is not None,
              "same-vintage slots score of record and the artifact is sha'd")
        payload = scored.model_dump(mode="json")
        payload["record_sha256"] = None
        check(_canonical_sha256(payload) == scored.record_sha256,
              "record_sha256 recomputes from the document with the field nulled")
        out = default_scored_path(record_path)
        check(out.name == "scored-predictions-synthetic-2026-07-28.json",
              f"the scored record is named for the filing record: {out.name}")
        write_scored_record(scored, out)
        check(out.is_file(), "the scored record writes where the desk expects it")
        try:
            write_scored_record(scored, out)
            check(False, "re-scoring must refuse to clobber")
        except ScoringError as exc:
            check("already exists" in str(exc),
                  f"an existing scored record is never silently replaced: "
                  f"{exc!s:.60}")
        bad_obs = root / "observed-typo.json"
        bad_obs.write_text(json.dumps({"observed": [
            {"source": "beta", "target": "alpha", "arm": "native",
             "a_hat": 0.33}]}))
        try:
            score_record(record_path, bad_obs)
            check(False, "an observation matching no filed slot must HALT")
        except ScoringError as exc:
            check("match no filed slot" in str(exc),
                  f"a flipped/typo'd observation HALTs rather than scoring "
                  f"nothing quietly: {exc!s:.60}")
        partial = root / "observed-partial.json"
        partial.write_text(json.dumps({"observed": [
            {"source": "alpha", "target": "beta", "arm": "native",
             "a_hat": 0.33}]}))
        part = score_record(record_path, partial)
        check(sum(1 for s in part.slots
                  if s.verdict == "UNSCORED-NO-OBSERVATION") == 1
              and any("UNSCORED-NO-OBSERVATION" in w for w in part.warnings),
              "a partial first-read names its unscored slots in the warnings")
        check(all("VINTAGE UNVERIFIED" in " ".join(s.flags) for s in part.slots
                  if s.verdict != "UNSCORED-NO-OBSERVATION"),
              "an observed â carrying no corpus sha is flagged VINTAGE "
              "UNVERIFIED, never silently treated as same-vintage (an "
              "UNOBSERVED slot has no fit and so no vintage to compare)")
        try:
            score_record(record_path, root / "no-such-observations.json")
            check(False, "scoring without an observed-â artifact must refuse")
        except ScoringError as exc:
            check("never auto-scores" in str(exc),
                  f"there is no path to a verdict without the desk's observed "
                  f"â: {exc!s:.60}")
        try:
            main(["--observed", str(obs_path)])
            check(False, "--observed without --score-record must not run")
        except SystemExit as exc:
            check(exc.code == 2,
                  f"--observed and --score-record are required together "
                  f"(argparse exit {exc.code})")

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
    ap.add_argument("--v21-root", type=Path, default=None,
                    help="root of a corpus-v2.1 re-bank mirror (Addendum G "
                         "§G1's go-forward basis). Its fits_v21_<model>/ hub "
                         "maps and vectors/<model>/ banks are PREPENDED to the "
                         "existing probes; the root must prove its vintage via "
                         "corpus/corpus_manifest.json. MUTUALLY EXCLUSIVE WITH "
                         "--gate (the gate is a fixed-vintage proof about the "
                         "archived v1 objects).")
    ap.add_argument("--v21-corpus-manifest", type=Path,
                    default=V21_CORPUS_MANIFEST_RELPATH,
                    help=f"path of the vintage manifest RELATIVE to --v21-root "
                         f"(default {V21_CORPUS_MANIFEST_RELPATH})")
    ap.add_argument("--score-record", type=Path, default=None,
                    help="a FILED prediction record to score. Requires "
                         "--observed; never runs at filing time.")
    ap.add_argument("--observed", type=Path, default=None,
                    help='the desk-supplied observed â artifact: {"corpus_'
                         'manifest_sha256": "<fit vintage>", "observed": '
                         '[{"source","target","arm","family","a_hat", '
                         '"corpus_manifest_sha256","fit_path","observed_utc"}]}')
    ap.add_argument("--scored-out", type=Path, default=None,
                    help="where the scored record is written (default: beside "
                         "the filing record as scored-<record name>)")
    ap.add_argument("--overwrite-scored", action="store_true",
                    help="permit replacing an existing scored record — a "
                         "deliberate desk act, never the default")
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

    #  THE RESOLUTION-PARITY TRAP, closed at the door. The E1 gate re-resolves
    #  the five ARCHIVED v1 pairs through this module's own hub_map_dirs and
    #  asserts parity against the archive's own npz files; a v2.1 root would
    #  move that leg onto v2.1 objects. `run_gate` guards this too — one guard
    #  can be bypassed programmatically, and this one cannot be bypassed at all.
    if args.gate and args.v21_root is not None:
        ap.error(
            "--gate and --v21-root are MUTUALLY EXCLUSIVE. The E1 implementation "
            "gate is a FIXED-VINTAGE proof about the archived v1 "
            "operationalization: it resolves the five archived pairs through "
            "the same hub_map_dirs() the predictor uses and asserts RESOLUTION "
            "PARITY against the archive's own npz files. Under a v2.1 root that "
            "leg silently moves onto v2.1 objects — the gate would fail for a "
            "reason that has nothing to do with fidelity, or pass on the wrong "
            "objects. Run the gate on its own, then run the v2.1 work.")
    if (args.score_record is None) != (args.observed is None):
        ap.error(
            "--score-record and --observed go together. Scoring is a DESK ACT "
            "against observed â values the desk produces at first-read; this "
            "tool never observes, never fits, and never auto-scores at filing "
            "time — it only builds the machine-readable artifact.")

    if not (args.gate or args.candidates or args.resolution_sweep
            or args.source_model or args.score_record):
        raise SystemExit("pass --selftest, --gate, --candidates, "
                         "--resolution-sweep, --source-model/--target-model, "
                         "or --score-record/--observed")

    if args.v21_root is not None:
        try:
            set_v21_root(args.v21_root,
                         manifest_relpath=args.v21_corpus_manifest)
        except CorpusVintageError as exc:
            #  An EXPECTED halt with a meaningful message: the operator pointed
            #  at something that is not the v2.1 basis of record.
            print(f"\nVINTAGE HALT — {exc}")
            return 1

    readout = ComposedReadout(generated=date.today().isoformat(), hub=HUB_MODEL,
                              hub_site=HUB_SITE_OF_RECORD, family=args.family,
                              v21_root=None if V21_ROOT is None else str(V21_ROOT),
                              corpus_manifest_sha256_v21=V21_CORPUS_SHA)
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

    if args.score_record is not None and args.observed is not None:
        out = args.scored_out or default_scored_path(args.score_record)
        try:
            scored = score_record(args.score_record, args.observed)
            print(f"\n{'scored slot':44s} {'predictor':10s} {'pred':>8s} "
                  f"{'obs':>8s} {'err':>8s}  verdict")
            for s in scored.slots:
                print(f"{s.key:44s} {s.predictor:10s} {s.predicted:+8.4f} "
                      f"{'—' if s.observed is None else f'{s.observed:+8.4f}'} "
                      f"{'—' if s.error is None else f'{s.error:+8.4f}'}  "
                      f"{s.verdict}"
                      + ("  [NOT SCORED OF RECORD]" if not s.scored_of_record
                         and s.verdict != "UNSCORED-NO-OBSERVATION" else ""))
            for agg in scored.aggregates:
                frac = ("—" if agg.in_band_fraction is None
                        else f"{agg.in_band_fraction:.3f}")
                print(f"\n{agg.predictor}: {agg.n_in_band}/"
                      f"{agg.n_scored_of_record} in band (fraction {frac}); "
                      f"{agg.n_out_of_band} out "
                      f"({agg.n_out_of_band_high} high / "
                      f"{agg.n_out_of_band_low} low); "
                      f"{agg.n_magnitude_only} magnitude-only; "
                      f"{agg.n_unscored_no_observation} unobserved; "
                      f"{agg.n_cross_vintage_excluded} cross-vintage excluded")
            h = scored.head_to_head
            print(f"\nhead-to-head over {h.n_slots_both_scored} slot(s) scored "
                  f"for both: both {h.n_both_in_band} · star-only "
                  f"{h.n_star_only} · composed-only {h.n_composed_only} · both "
                  f"out {h.n_both_out_of_band} ({h.n_slots_incomplete} "
                  f"incomplete)")
            for warning in scored.warnings:
                print(f"  WARNING: {warning}")
            written = write_scored_record(scored, out, args.overwrite_scored)
            print(f"\nscored record: {written} (sha "
                  f"{scored.record_sha256}). Campaign gates are NOT evaluated "
                  f"here — the desk reads these counts.")
        except ScoringError as exc:
            print(f"\nSCORING HALT — {exc}")
            return 1

    payload = readout.model_dump_json(indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        logger.info("wrote %s (%d prediction(s), %d N/A-at-filing)", args.out,
                    len(readout.predictions), len(readout.na_at_filing))
    return status


if __name__ == "__main__":
    sys.exit(main())
