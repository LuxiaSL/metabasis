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
WHICH TREE ANSWERED — THE PROVENANCE LEDGER AND THE FILED-PATHS PIN (E4 A1)
────────────────────────────────────────────────────────────────────────────────
`REPORT-e4-nulls-batch4-2026-07-29` anomaly A1: with no `--v21-root` the probe
order starts at the frozen collection tree, so **0/24 filed batch-4 legs
resolved to the corpus-v2.1 maps their record was filed from**. That enactment
loaded the filed `resolved` paths BY HAND and was proved right by exact E1
parity (|Δ| = 0.0 ×12). The probe order itself is not the defect — `--v21-root`
already prepends the go-forward vintage when the operator asks for it, and the
E1 gate's parity leg depends on the default order staying put. The defect is
that a resolve never SAID which tree answered, so a caller could not tell a
same-vintage hit from a cross-vintage one without reading paths by eye. Two
additions, neither of which moves the default probe order:

  * **The provenance ledger.** Every `resolve_hub_map` / `resolve_vector_bank`
    appends a `ResolutionProvenance`: the model, the artifact kind, WHICH TREE
    answered, that tree's corpus vintage, where in the probe order it sat, and
    everything probed and missed. Read it with `resolution_provenance()`, scope
    it with `provenance_scope()`, write it with `--provenance-out`. It is quiet
    for an ordinary resolve and LOUD where a vintage confusion is born: a
    resolve that falls THROUGH a set v2.1 root to a pre-v2.1 tree is warned and
    flagged `cross_vintage_fallback` (today such a slot surfaces only later, as
    MIXED-VINTAGE, and only if some OTHER side did resolve to v2.1 — where all
    four fall through, nothing says anything at all).
  * **Filed-paths mode**, for scoring and null tooling: `--filed-paths <filed
    record>` (`load_filed_paths` / `set_filed_paths` / `filed_paths_scope`)
    PINS resolution to that record's own `resolved` paths, so the artifacts
    re-derived are the ones the record was filed from rather than whatever the
    probe order finds. A pinned artifact that has gone HALTS; a key the record
    does not file HALTS under the strict default, or probes WARNED and recorded
    as UNPINNED with `--filed-paths-allow-unpinned`. **There is no silent
    fallback across vintages** — falling back from a pinned v2.1 artifact to a
    probed v1 one is rake M21b's lesson at map grain. `--filed-paths` is
    mutually exclusive with `--gate` (parity would test the pin, not the
    resolver; `run_gate` guards it programmatically as well) and with
    `--v21-root` (two different answers to "which artifact", per key).

Both are DEFAULT-OFF and neither changes what any existing caller resolves: with
no root and no pin, `--gate` and `--candidates` are byte-identical to what they
were before this section existed, which selftests 21–22 prove path-by-path.

────────────────────────────────────────────────────────────────────────────────
THE DIRECTIONAL STAR — A THIRD PREDICTOR COLUMN (ADDENDUM 2026-07-29-H)
────────────────────────────────────────────────────────────────────────────────
Addendum H invokes Addendum C item 2 and ADOPTS the directional star

    â(A→B) = c_A^out · c_B^in

for NOT-YET-FILED slots. It is a third column beside the symmetric star and the
composed path, on structurally identical terms (H item 3): ±.05 absolute bands
frozen at filing, near-zero carve-out at |pred| < .08, corpus sha per G2. **The
symmetric star column is UNTOUCHED** (H item 2 — it keeps filing per Addendum E
on its own frozen terms), and nothing in this module recomputes, moves or
re-derives it.

This module does not FIT c^out / c^in — that derivation is the hub protocol's
(H item 5, both fit directions on corpus-v2.1). It CONSUMES a banked directional
constants readout and emits filable slots:

    `directional-prediction/<source>→<target>/<arm>-k128`

The readout is read against a NAMED, VERSIONED schema
(`directional-constants-readout/v1`, `load_directional_constants`) and every
departure from it is a HALT whose message states exactly what was expected —
the schema is a contract with the derivation lane, not a shape to be inferred.
Availability mirrors E1's arm rule at scalar order: a slot files only where the
source has a c^out AND the target a c^in **in the pair's applicable arm**; where
it does not, the slot is `N/A-AT-FILING` and is never proxied from another arm.

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
(G-star-hit / G-comp-hit / the H-item-3 directional gate) are NOT evaluated here
— they are aggregates over the whole 190 and are the desk's read.

Since Addendum H the scored record carries THREE predictor columns. The E3 2×2
is preserved verbatim (`head_to_head`, star vs composed — continuity: the same
field, the same field names, the same denominators) and the three-predictor
reads are ADDED beside it: every predictor PAIR's 2×2 (`head_to_head_pairwise`)
and the per-slot hit-set census across all three (`head_to_head_three_way`).

Addendum D ACTIVATES with H (its scope trigger is exactly this adoption). Its
constant-α companion is NEVER auto-computed here: ᾱ is a grand mean over the
scored set that only the desk can form, and the ceiling/calibration terms are
the adopted gauge's. Supply it with `--alpha-companion` against the named schema
`constant-alpha-companion/v1`; supply nothing and the scored record NAMES the
companion as OWED, with D2/D3's frozen text attached.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.scripts.read_composed_predictions --selftest
  python -m metabasis.scripts.read_composed_predictions --gate
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --out /tmp/claude-output/composed_candidates.json
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --v21-root <v2.1 re-bank root> --out /tmp/claude-output/composed_v21.json
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --directional-constants <directional constants readout>.json \
      --directional-out /tmp/claude-output/directional_slots.json
  python -m metabasis.scripts.read_composed_predictions --candidates \
      --filed-paths outputs/collection/predictions/<record>.json \
      --filed-paths-allow-unpinned \
      --provenance-out /tmp/claude-output/resolution_provenance.json
  python -m metabasis.scripts.read_composed_predictions \
      --score-record outputs/collection/predictions/<record>.json \
      --observed /tmp/claude-output/observed-<batch>.json \
      [--alpha-companion /tmp/claude-output/constant-alpha-<batch>.json]
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

#: The primary hub's OWN entropy-gradient bank at the rebuilt-L16 column — the
#: `xi` of the hub-frame decomposition (see `hub_frame_coordinates`). ONE path,
#: no fallback: the neighbouring `entropy_gradient_8b.npz` is a different
#: column, and a near-miss here would silently rotate every alpha (rake M12).
#: Its absence degrades the descriptive companions, never the prediction.
HUB_VECTOR_PATH = (COLLECTION_ROOT / HUB_MODEL / "vectors"
                   / f"entropy_gradient_{HUB_MODEL}_L{HUB_SITE_OF_RECORD}rebuild.npz")
#: The hub-frame decomposition needs BOTH hub maps to share one hub-side PCA
#: basis (`va`). Banked maps fit against the same hub bank agree to ~2e-12
#: (chart-overlap preview: hub_basis_max_dev 1.82e-12 across 12 models); a
#: genuinely different basis differs by O(1), so this separates them with room
#: to spare. Beyond it the companions are NOT emitted — the identity
#: â_comp = alpha_AB·ceil_B only holds over a common hub frame.
HUB_BASIS_MAX_DEV_TOLERANCE = 1e-6
#: The decomposition must reproduce the predictor itself; anything above this is
#: a broken companion, not a rounding difference.
COMPANION_IDENTITY_TOLERANCE = 1e-8

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

#: Filing convention of record for a scalar predicted â and its band: 4 decimal
#: places (batches canary/2/3 all file that way — e.g. star .3614 with band
#: [.3114, .4114]). A directional slot therefore emits BOTH the full-precision
#: arithmetic and the 4-dp `filed_*` pair, so the band the desk freezes is the
#: band of the value it files rather than of an unrounded one it never saw.
FILED_DECIMALS = 4
#: A portability coefficient is a cosine-scale quantity; |c| > 1 is not
#: impossible for a chain-break/gauge-divided read, so it is FLAGGED on the
#: slot, never silently accepted and never used to refuse the readout. A
#: non-finite constant IS refused (there is no reading of NaN·c that files).
DIRECTIONAL_CONSTANT_FLAG_ABS = 1.0

#: ADDENDUM 2026-07-29-H — the directional star's constants readout, by NAME and
#: VERSION. The derivation lane (hub protocol, both fit directions, corpus-v2.1)
#: produces this file; this module only consumes it, and consumes it against
#: this contract. NAMED WITH ITS VERSION per rake M26 so a schema bump is
#: visible to a mechanical sweep instead of hiding inside a string literal.
SCHEMA_DIRECTIONAL_CONSTANTS_V1 = "directional-constants-readout/v1"
#: ADDENDUM 2026-07-27-D §D2 — the constant-α companion baseline artifact. Same
#: discipline: named, versioned, desk-supplied, never computed here.
SCHEMA_CONSTANT_ALPHA_COMPANION_V1 = "constant-alpha-companion/v1"

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


class DirectionalConstantsError(ComposedPathError):
    """A directional-constants readout is not `directional-constants-readout/v1`.

    ALWAYS LOUD, and always by naming the contract: the readout is produced by a
    different lane (the hub-protocol derivation of c^out / c^in), so the failure
    a reader must never see is a filed â that silently rode a mis-keyed, mis-
    armed or half-parsed constants table. Every message here states what this
    module expected, so the two lanes can be reconciled from the error alone.
    """


class ScoringError(RuntimeError):
    """A scoring input cannot be read as what it claims to be.

    Scoring is a desk act performed ONCE per record against frozen bands; a
    malformed record, an unmatched observation, or a band that is not the
    frozen one must stop the run rather than produce a verdict nobody can
    audit.
    """


class AlphaCompanionError(ScoringError):
    """A constant-α companion artifact is not `constant-alpha-companion/v1`.

    Addendum D's companion is a BASELINE the desk computes (ᾱ is a grand mean
    over the scored set; the ceiling terms are the adopted gauge's). A malformed
    companion must halt rather than degrade to "absent", because "absent" is
    itself a reportable state — OWED — and the two must never be confused.
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
    # gemma3-27b: L38, RULED BY LUXIA 2026-07-29 (session-5 close) from the
    # six-site â evidence table — readout of record
    # `site_evidence_gemma3-27b_20260729-055507.json`, sha `7f59af50…`, 48 rows,
    # strict norms. L38 ⋆ primary, L41 robustness; the carried-provisional L36
    # is RETIRED to scanned-history (legitimate, mid-pack, DOMINATED by L38
    # +22.0% / +9.1% on both hub columns, with the higher ceiling and coherence).
    # The r²/â INVERSION is the deciding structure — r(r², â) = −.964, and the
    # r² peak L13 is SUB-NULL on â — so this site is never re-derivable from an
    # r² curve; `fit_transport_maps.SITES` carries the full rationale and the two
    # registries must agree (cross-checked in selftest 7).
    # ⚠ THE VECTOR GAP REMAINS OPEN, AND IT IS THE BINDING ONE. There is no
    # corpus-v2.1 entropy-gradient vector at ANY gemma site: the six site-
    # evidence vectors (L13/L17/L35/L36/L38/L41) are FROZEN-v1 (`a6712ca0…`)
    # SELECTION instruments that never file, and the banked hub→gemma maps this
    # registry can reach are the same v1 vintage. Before ANY gemma slot files, a
    # fresh FD-gated corpus-v2.1 (`5ae355bc…`) L38 build + L38/L41 v2.1 state
    # banks + their hub fits are REQUIRED — the re-bank is QUEUED, and until it
    # lands `--resolution-sweep` reports this model as a NAMED GAP at L38, which
    # is the honest state and not a regression (see also `vector_bank_paths`).
    "gemma3-27b": 38,
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
    # llama-3.1-405b-instruct: L99, RULED BY THE DESK 2026-07-29 UNDER LUXIA'S
    # OVERNIGHT DELEGATION 2 (ledgered) from the five-site â evidence table —
    # readout of record
    # `site_evidence_llama-3.1-405b-instruct_20260729-113819.json`,
    # sha `04f2a2c4…`, 40 rows, strict (fit-local) norms, frozen-v1 evidence
    # basis. L99 ⋆ primary (fractional depth .786), L107 robustness (.849).
    # Rank-1 (L99), rank-2 (L107) and LAST (L19) are each UNANIMOUS across all
    # four robustness columns (native k128 / k32, raw k128, rebuilt-L16 hub);
    # only the middle two sites swap. The r² peak L43 ranks 3rd/4th on â and
    # goes SUB-NULL in the rebuilt-L16 column, and L19 — second-highest r² on
    # the grid — is last everywhere, sub-null in both hub columns, coherence
    # .109, and the only `direction_specific = False` build in any fd-gate
    # artifact the campaign holds. That is the r²/â inversion's FOURTH instance
    # and its third on the Llama family, so this site is never re-derivable from
    # an r² curve; `fit_transport_maps.SITES` carries the full rationale, the
    # correlation structure and the scale caveat, and the two registries must
    # agree (cross-checked in selftest 7).
    # SCALE CAVEAT, CARRIED FORWARD: the campaign's largest model is its weakest
    # transporter — best â +.1453 against the 3.3-70B's +.3058 and
    # qwen3-30b-a3b's +.3283, â/ceiling .254 against ~.52 — at a COMPARABLE
    # ceiling. Any 405B number read downstream states this.
    # ⚠ THE VECTOR GAP IS OPEN, AND IT IS THE BINDING ONE. NO corpus-v2.1
    # entropy-gradient vector exists for this node at ANY site: the five
    # site-evidence vectors are FROZEN-v1 SELECTION instruments that never file.
    # Before ANY 405B slot files, a fresh FD-gated corpus-v2.1 L99 build +
    # L99/L107 v2.1 state banks + their hub fits are REQUIRED. The re-bank is
    # QUEUED, and it gates more than this row: the ρ-law ceremony's AUDIT PAIRS
    # wait on the same build (pre-statement `7542b377…`). Until it lands
    # `--resolution-sweep` reports this model as a NAMED GAP at L99 — the honest
    # state, not a regression — and `--candidates` must NOT enumerate it (see
    # also `vector_bank_paths`).
    "llama-3.1-405b-instruct": 99,
    # llama-3.3-70b-instruct: L58, RULED BY THE DESK 2026-07-29 UNDER LUXIA'S
    # OVERNIGHT DELEGATION 2 (ledgered) from the five-site â evidence table —
    # readout of record
    # `site_evidence_llama-3.3-70b-instruct_20260729-075541.json`,
    # sha `e6d584aa…`, 40 rows, strict (fit-local) norms, frozen-v1
    # evidence basis. L58 ⋆ primary, L63 robustness; rank-1 and rank-2 are
    # UNANIMOUS across all four robustness columns (native k128 / k32, raw k128,
    # rebuilt-L16 hub) and the deep cluster {58, 63, 68} dominates. The r² peak
    # L17 is SUB-NULL on â in BOTH hub columns — the r²/â inversion's THIRD
    # instance (3.1-70B L17, gemma L13, 3.3-70B L17), all at ~21% fractional
    # depth — so this site is never re-derivable from an r² curve;
    # `fit_transport_maps.SITES` carries the full rationale and the two
    # registries must agree (cross-checked in selftest 7).
    # NOT INHERITED FROM ROW 11: llama-3.1-70b-instruct is the same pretrain
    # family at a different post-training vintage and rules to (37, 43) ⋆ L37.
    # The two registrations are independent, and their disagreement is a
    # recorded observation, not an error in either.
    # ⚠ THE VECTOR GAP IS OPEN, AND IT IS THE BINDING ONE. NO corpus-v2.1
    # entropy-gradient vector exists for this node at ANY site: the five
    # site-evidence vectors are FROZEN-v1 SELECTION instruments that never file.
    # Before ANY 3.3-70b slot files, a fresh FD-gated corpus-v2.1 L58 build +
    # L58/L63 v2.1 state banks + their hub fits are REQUIRED — the re-bank is
    # QUEUED, and until it lands `--resolution-sweep` reports this model as a
    # NAMED GAP at L58, which is the honest state and not a regression (see also
    # `vector_bank_paths`).
    "llama-3.3-70b-instruct": 58,
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


# ------------------------------------------- resolution provenance (E4 A1)
#  WHAT ANSWERED, AND FROM WHICH VINTAGE — recorded on EVERY resolve.
#
#  The E4 batch-4 enactment (REPORT-e4-nulls-batch4-2026-07-29, anomaly A1)
#  found the shape of the gap exactly: with no `--v21-root` the probe order
#  starts at the frozen collection tree, so 0/24 filed batch-4 legs resolved to
#  the corpus-v2.1 maps their record was FILED from. The enactment had to load
#  the filed `resolved` paths by hand — and was proven right by exact E1 parity
#  (|Δ| = 0.0 ×12). Nothing in the resolver was wrong; the resolver simply never
#  SAID which tree answered, so a caller could not tell a same-vintage hit from
#  a cross-vintage one without reading the path by eye.
#
#  Two mechanisms close that, and NEITHER moves the default probe order (the
#  E1 gate's resolution-parity leg depends on it, and `--v21-root` already
#  prepends the go-forward vintage when the operator asks for it):
#
#    1. THIS LEDGER. Every `resolve_hub_map` / `resolve_vector_bank` call
#       appends a `ResolutionProvenance` naming the model, the artifact kind,
#       WHICH TREE answered, that tree's corpus vintage, where in the probe
#       order it sat, and what was probed and missed. It is data, always
#       present, retrievable with `resolution_provenance()` and writable with
#       `--provenance-out`.
#    2. THE FILED-PATHS PIN (below): scoring and null tooling pins resolution
#       to a record's OWN `resolved` paths instead of probing at all.
#
#  WHERE IT GETS LOUD. The ledger is written at DEBUG for an ordinary resolve —
#  a routine v1 resolve in the v1 regime is not news, and stdout is a frozen
#  surface here. It ESCALATES exactly where a vintage confusion can be born:
#    * a v2.1 root is in force and a resolve fell through it to a pre-v2.1 tree
#      (`cross_vintage_fallback`) -> WARNING, and the record says so. This is
#      the silent half of A1: today such a slot only surfaces later, as
#      MIXED-VINTAGE, and only if some OTHER side did resolve to v2.1. Where
#      ALL sides fall through, nothing said anything at all.
#    * a filed-paths pin is in force and the pinned artifact is gone, or the
#      key is unpinned under the strict default -> HALT (`FiledPathsError`).
#      Never a fallback: falling back from a pinned v2.1 artifact to a probed
#      v1 one is rake M21b's lesson at map grain.
ResolutionKind = Literal["hub map", "vector bank"]
#: How a resolve was answered. `pin` = a filed-paths record named the artifact;
#: `probe` = the preference-ordered candidate list answered; `absent` = every
#: probe missed (N/A-AT-FILING, which is data, not a failure).
ResolutionSource = Literal["pin", "probe", "absent"]


class ResolutionProvenance(BaseModel):
    """One resolve, and which tree answered it. DESCRIPTIVE — never a verdict.

    Nothing in the prediction path reads this back: `compose_pair` computes
    â_comp from the `HubMapRef`/`VectorBankRef` exactly as before, and this
    ledger only records what those resolvers did. It exists so the question the
    E4 enactment had to answer by eye — "is this the vintage the record was
    filed from?" — is answerable mechanically.
    """
    kind: ResolutionKind
    model: str
    site: int
    arm: Optional[str] = Field(
        default=None, description="hub maps only; a vector bank is arm-free")
    family: Optional[str] = None
    source: ResolutionSource
    resolved: Optional[str] = Field(
        default=None, description="the artifact actually used; None => absent")
    corpus: Optional[CorpusProvenance] = Field(
        default=None,
        description="the vintage of the tree that answered. None on a vector "
                    "bank outside a verified v2.1 root, for the reason "
                    "`VectorBankRef.corpus` states: only the bank's own stamp "
                    "is entitled to claim a vintage there")
    tree: str = Field(
        description="WHICH TREE answered, in words — the candidate dir's own "
                    "note, or the pin's record path. The field the A1 read "
                    "needed and did not have")
    probe_index: Optional[int] = Field(
        default=None, description="0-based position in the probe order that "
                                  "answered; None for a pin or an absence")
    n_probed: int = 0
    probed_paths: list[str] = []
    v21_root: Optional[str] = Field(
        default=None, description="the verified v2.1 root in force at resolve "
                                  "time, if any")
    pin_record: Optional[str] = Field(
        default=None, description="the filing record the pin was built from")
    cross_vintage_fallback: bool = Field(
        default=False,
        description="a v2.1 root was in force and this resolve fell THROUGH it "
                    "to a pre-v2.1 tree. Computable before, sayable now")
    notes: list[str] = []


class ResolutionProvenanceReadout(BaseModel):
    """The ledger as an artifact (`--provenance-out`). Writes nothing itself."""
    STATUS: str = (
        "DESCRIPTIVE — a record of which tree answered each resolve. Computes "
        "no â, moves no band, writes nothing under outputs/.")
    generated: str
    v21_root: Optional[str] = None
    filed_paths_record: Optional[str] = None
    n_resolves: int = 0
    n_by_source: dict[str, int] = {}
    n_by_corpus: dict[str, int] = {}
    n_cross_vintage_fallback: int = 0
    n_absent: int = 0
    records: list[ResolutionProvenance] = []


#: The append-only resolution ledger for this process. MODULE-LEVEL for the same
#: reason `V21_ROOT` is: the resolvers are called from deep inside the predictor
#: through fixed signatures. It grows with every resolve — a long-lived process
#: should scope it (`provenance_scope()`) or clear it
#: (`clear_resolution_provenance()`); a CLI run is bounded by construction.
_RESOLUTION_PROVENANCE: list[ResolutionProvenance] = []


def resolution_provenance() -> tuple[ResolutionProvenance, ...]:
    """Every resolve recorded so far, in call order. A COPY — the ledger is
    appended to by the resolvers and must not be mutated from outside."""
    return tuple(_RESOLUTION_PROVENANCE)


def clear_resolution_provenance() -> int:
    """Empty the ledger; returns how many records were dropped."""
    n = len(_RESOLUTION_PROVENANCE)
    _RESOLUTION_PROVENANCE.clear()
    return n


@contextmanager
def provenance_scope() -> Iterator[list[ResolutionProvenance]]:
    """Record provenance for the duration of a block, restoring the ledger after.

    Yields the list the block's own records land in, so a caller can read them
    without having to diff a global.
    """
    global _RESOLUTION_PROVENANCE
    previous = _RESOLUTION_PROVENANCE
    scoped: list[ResolutionProvenance] = []
    _RESOLUTION_PROVENANCE = scoped
    try:
        yield scoped
    finally:
        _RESOLUTION_PROVENANCE = previous


def _record_provenance(record: ResolutionProvenance) -> ResolutionProvenance:
    """Append one resolve to the ledger, at the volume its content deserves."""
    _RESOLUTION_PROVENANCE.append(record)
    if record.cross_vintage_fallback:
        logger.warning(
            "CROSS-VINTAGE FALLBACK — %s %s L%s resolved to the %s tree (%s) "
            "while corpus-v2.1 root %s is in force. The value is computable and "
            "is NOT the go-forward vintage; a prediction mixing this with a "
            "v2.1 side is flagged MIXED-VINTAGE, one where every side falls "
            "through carries no flag at all, which is why this line exists.",
            record.kind, record.model, record.site, record.corpus, record.tree,
            record.v21_root)
    else:
        logger.debug("resolved %s %s L%s <- %s [%s] via %s", record.kind,
                     record.model, record.site, record.resolved, record.corpus,
                     record.tree)
    return record


def provenance_readout() -> ResolutionProvenanceReadout:
    """The ledger, summarized, as the `--provenance-out` artifact."""
    records = list(_RESOLUTION_PROVENANCE)
    by_source: dict[str, int] = {}
    by_corpus: dict[str, int] = {}
    for rec in records:
        by_source[rec.source] = by_source.get(rec.source, 0) + 1
        label = rec.corpus or "unlabelled"
        by_corpus[label] = by_corpus.get(label, 0) + 1
    return ResolutionProvenanceReadout(
        generated=date.today().isoformat(),
        v21_root=None if V21_ROOT is None else str(V21_ROOT),
        filed_paths_record=None if FILED_PATHS is None else FILED_PATHS.record,
        n_resolves=len(records), n_by_source=by_source, n_by_corpus=by_corpus,
        n_cross_vintage_fallback=sum(r.cross_vintage_fallback for r in records),
        n_absent=sum(r.source == "absent" for r in records),
        records=records)


# ------------------------------------------------- the filed-paths pin (E4 A1)
class FiledPathsError(ComposedPathError):
    """A filed-paths pin cannot answer a resolve, and MUST NOT fall back.

    Three cases, all loud, all naming the record: the pinned artifact is gone
    from disk; the record files two different paths for one key; a key is
    unpinned while the pin is STRICT. The fourth possible behaviour — quietly
    probing instead — is the one this class exists to make impossible, because
    a probe answers from the frozen tree and the record was filed from v2.1.
    """


class FiledPath(BaseModel):
    """One artifact path as a filing record filed it."""
    key: str
    path: str
    corpus: Optional[CorpusProvenance] = Field(
        default=None, description="the vintage the RECORD says that path had; "
                                  "carried through as filed, never re-derived")
    filed_in_slot: str = Field(description="the record slot it was read from")


class FiledPaths(BaseModel):
    """A record's own resolved paths, as a resolution pin. READ-ONLY over it.

    Built by `load_filed_paths` from a filed prediction record's
    `composed_prediction` blocks. Keys are the resolver's own identity —
    `<model>L<site>/<arm>-<family>` for a hub map, `<model>L<site>` for a vector
    bank — so `resolve_hub_map`/`resolve_vector_bank` keep their signatures and
    every caller (including `compose_pair` and the E1 machinery) is pinned by
    the pin's mere presence, with no argument threading.

    `strict` is the default and the point: a key the record does not file is a
    HALT, not a silent return to probing. `strict=False` is the sweep-shaped
    escape valve — unpinned keys probe as usual, but every one of them is
    WARNED and recorded with `notes` saying the pin did not cover it.
    """
    record: str
    sha256: str
    strict: bool = True
    hub_maps: dict[str, FiledPath] = {}
    vector_banks: dict[str, FiledPath] = {}

    @property
    def n_pinned(self) -> int:
        return len(self.hub_maps) + len(self.vector_banks)


def hub_map_pin_key(model: str, site: int, arm: str, family: str) -> str:
    """The pin key for a hub map — the resolver's full identity (rake M18)."""
    return f"{model}L{site}/{arm}-{family}"


def vector_bank_pin_key(model: str, site: int) -> str:
    """The pin key for an entropy-gradient bank (arm-free by construction)."""
    return f"{model}L{site}"


#: The filed-paths pin in force, or None (the default, and what every existing
#: caller gets). Set ONLY through `set_filed_paths`, exactly as `V21_ROOT` is.
FILED_PATHS: Optional[FiledPaths] = None


def _pin_conflict(record: Path, key: str, kind: str, first: FiledPath,
                  second: str, slot: str) -> FiledPathsError:
    return FiledPathsError(
        f"HALT — {record} files TWO different {kind} paths for one key {key!r}:\n"
        f"  {first.path}   (slot {first.filed_in_slot})\n"
        f"  {second}   (slot {slot})\n"
        f"A pin cannot choose between them, and choosing silently is how a "
        f"scored read ends up mixing vintages within one record. Reconcile the "
        f"record (or score the halves as separate records) — this tool will not "
        f"guess which artifact is of record.")


def _filed_corpus(record: Path, key: str, value: Any) -> Optional[CorpusProvenance]:
    """The vintage a record claims for one artifact, checked against the enum.

    A record naming a vintage this module does not know is a HALT, not a
    coerced None: `CorpusProvenance` is a closed set, and an unrecognized label
    means the record and this parser disagree about what the vintages ARE —
    which is precisely the confusion a pin exists to remove.
    """
    if value is None:
        return None
    if value not in ("frozen", "legacy", "v21"):
        raise FiledPathsError(
            f"HALT — {record} labels the artifact for {key!r} with corpus "
            f"vintage {value!r}, which is not one of the vintages this module "
            f"knows ('frozen', 'legacy', 'v21'). Either the record was written "
            f"by a different schema or the label is a typo; a pin will not "
            f"carry a vintage it cannot name.")
    return value                                    # type: ignore[return-value]


def load_filed_paths(path: Path, strict: bool = True) -> FiledPaths:
    """Build a resolution pin from a filed prediction record's own paths.

    Reads the `composed_prediction` block of every row and takes the FOUR
    artifacts it names: `hub_map_source.resolved`, `hub_map_target.resolved`,
    `source_vector.path`, `target_vector.path`. Nothing is recomputed, nothing
    is probed, and no path is checked for existence HERE — a missing artifact is
    reported at the resolve that wanted it, where the message can name the key.

    Paths are used AS FILED. The records file repo-relative paths and the whole
    module resolves repo-relative (`COLLECTION_ROOT = Path("outputs")`), so a
    pin is used from the same working directory as the record was filed from.
    """
    if not path.is_file():
        raise FiledPathsError(
            f"filed prediction record absent: {path}. Filed-paths mode pins "
            f"resolution to a RECORD's own artifacts; without the record there "
            f"is nothing to pin to")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise FiledPathsError(f"unreadable prediction record {path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("predictions"), list):
        raise FiledPathsError(
            f"{path}: not a filing record — expected a top-level object with a "
            f"`predictions` list, got keys "
            f"{sorted(doc) if isinstance(doc, dict) else type(doc).__name__}")

    pin = FiledPaths(record=str(path), sha256=sha256_of(path), strict=strict)
    for row in doc["predictions"]:
        if not isinstance(row, dict):
            raise FiledPathsError(f"{path}: prediction row is not an object: {row!r}")
        block = row.get("composed_prediction")
        if not isinstance(block, dict):
            continue                      # a row that files no composed column
        slot = str(block.get("pair_id") or block.get("prediction_id") or row)
        for side in ("hub_map_source", "hub_map_target"):
            ref = block.get(side)
            if not isinstance(ref, dict) or not ref.get("resolved"):
                continue
            try:
                key = hub_map_pin_key(str(ref["model"]), int(ref["site"]),
                                      str(ref["arm"]), str(ref["family"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise FiledPathsError(
                    f"{path}: slot {slot} {side} is missing the identity a pin "
                    f"keys on (model/site/arm/family): {exc}") from exc
            resolved = str(ref["resolved"])
            seen = pin.hub_maps.get(key)
            if seen is not None and seen.path != resolved:
                raise _pin_conflict(path, key, "hub map", seen, resolved, slot)
            pin.hub_maps[key] = FiledPath(
                key=key, path=resolved,
                corpus=_filed_corpus(path, key, ref.get("corpus")),
                filed_in_slot=slot)
        for side in ("source_vector", "target_vector"):
            spec = block.get(side)
            if not isinstance(spec, dict) or not spec.get("path"):
                continue
            try:
                key = vector_bank_pin_key(str(spec["model"]), int(spec["site"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise FiledPathsError(
                    f"{path}: slot {slot} {side} is missing the identity a pin "
                    f"keys on (model/site): {exc}") from exc
            resolved = str(spec["path"])
            seen = pin.vector_banks.get(key)
            if seen is not None and seen.path != resolved:
                raise _pin_conflict(path, key, "vector bank", seen, resolved, slot)
            pin.vector_banks[key] = FiledPath(
                key=key, path=resolved, filed_in_slot=slot)

    if not pin.n_pinned:
        raise FiledPathsError(
            f"{path}: no `composed_prediction` block in it names a resolved "
            f"artifact, so the pin would be empty. A star-only or "
            f"directional-only record has no composed resolution to pin to")
    logger.info("filed-paths pin: %s (%d hub map(s), %d vector bank(s), "
                "strict=%s)", path, len(pin.hub_maps), len(pin.vector_banks),
                strict)
    return pin


def set_filed_paths(pin: Optional[FiledPaths]) -> Optional[FiledPaths]:
    """Pin resolution to a record's own paths, or clear the pin (None)."""
    global FILED_PATHS
    FILED_PATHS = pin
    return pin


@contextmanager
def filed_paths_scope(pin: Optional[FiledPaths]) -> Iterator[Optional[FiledPaths]]:
    """`set_filed_paths` for the duration of a block, restored on any exit."""
    global FILED_PATHS
    previous = FILED_PATHS
    try:
        yield set_filed_paths(pin)
    finally:
        FILED_PATHS = previous


def _pinned_path(kind: ResolutionKind, key: str, model: str,
                 site: int) -> Optional[FiledPath]:
    """The pinned artifact for `key`, or None when no pin is in force.

    Raises rather than returning None when a pin IS in force and cannot answer:
    a strict pin that does not cover the key, always; and — in either mode — a
    pinned path that is no longer on disk. Falling back would silently swap the
    vintage the record was filed from for whatever the probe order finds.
    """
    if FILED_PATHS is None:
        return None
    table = (FILED_PATHS.hub_maps if kind == "hub map"
             else FILED_PATHS.vector_banks)
    filed = table.get(key)
    if filed is None:
        if FILED_PATHS.strict:
            raise FiledPathsError(
                f"filed-paths pin {FILED_PATHS.record} does not file a {kind} "
                f"for {key!r} ({model} L{site}), and the pin is STRICT. It "
                f"files {len(table)} {kind}(s): {sorted(table)[:8]}"
                + (" …" if len(table) > 8 else "")
                + f".\nThe strict pin refuses to probe for an unfiled key on "
                  f"purpose: probing answers from the frozen collection tree "
                  f"first, which is exactly how 0/24 filed batch-4 legs missed "
                  f"the corpus-v2.1 maps of record (E4 anomaly A1). Score the "
                  f"record's own slots, or pass allow_unpinned=True "
                  f"(--filed-paths-allow-unpinned) to let unfiled keys probe "
                  f"— WARNED and recorded, never silent.")
        return None
    if not Path(filed.path).exists():
        raise FiledPathsError(
            f"HALT — filed-paths pin {FILED_PATHS.record} files the {kind} for "
            f"{key!r} at {filed.path} (record slot {filed.filed_in_slot}, "
            f"vintage as filed: {filed.corpus or 'unlabelled'}), and that path "
            f"is not on disk.\nThis tool will NOT fall back to the probe order: "
            f"the probe order answers from the frozen collection tree and would "
            f"substitute a different corpus vintage for the artifact the record "
            f"was filed from — a number that is individually valid and "
            f"collectively meaningless (rake M21b at map grain). Restore the "
            f"artifact, or re-file the record against what is banked.")
    return filed


def _unpinned_notes(kind: ResolutionKind, key: str) -> list[str]:
    """The warning + note for a resolve a NON-STRICT pin did not cover.

    Empty when no pin is in force, which is every existing caller. A strict pin
    never reaches here — `_pinned_path` has already raised.
    """
    if FILED_PATHS is None:
        return []
    logger.warning(
        "UNPINNED RESOLVE — filed-paths pin %s files no %s for %r; probing "
        "instead (allow_unpinned). The probe order answers from the frozen "
        "collection tree first, so what it finds may be a DIFFERENT corpus "
        "vintage from the one the record was filed on.",
        FILED_PATHS.record, kind, key)
    return [f"UNPINNED — the filed-paths pin {FILED_PATHS.record} files no "
            f"{kind} for {key!r}, so this resolve PROBED. Permitted only "
            f"because the pin is non-strict (allow_unpinned): the vintage here "
            f"is whatever the probe order found, NOT what the record filed."]


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


class DescriptiveCompanions(BaseModel):
    """The hub-frame decomposition of a composed prediction. NEVER SCORED.

    Descriptive companions only: no band, no gate, no verdict and no aggregate
    depends on any field here. They are emitted beside the prediction because
    they are FREE — every quantity comes from the two hub maps and two vectors
    already loaded to compute â_comp — and because the decomposition is the
    mechanism claim's own vocabulary (SURVIVAL-AUDIT §4: alpha is a pair-chart
    overlap property, not a per-model constant).

    Construction and algebra: see the `descriptive companions` section below
    (`chart_companions`), whose convention is the chart-overlap preview's
    verbatim. `â_comp = alpha_pair · ceiling_target` is an IDENTITY, so this is
    a decomposition of the filed number and never a second prediction.
    """
    STATUS: str = (
        "DESCRIPTIVE — never scored, never a gate, never a band. A decomposition "
        "of the filed â_comp, not a second prediction: â_comp = alpha_pair · "
        "ceiling_target is an IDENTITY, verified per record below.")
    operationalization: str = (
        "hub-frame coordinates per staging/chart-overlap-preview/"
        "chart_overlap_preview.py (zeta_M = omega_M @ (vb_M @ v_M), "
        "xi = va @ v_hub); convention reused verbatim, not re-derived")
    alpha_source: Optional[float] = Field(
        default=None, description="cos(xi, zeta_A) — the source endpoint's "
                                  "in-chart alignment with the hub axis")
    alpha_target: Optional[float] = Field(
        default=None, description="cos(xi, zeta_B)")
    alpha_pair: Optional[float] = Field(
        default=None, description="cos(zeta_A, zeta_B) — the pair-chart overlap")
    rank1_implied: Optional[float] = Field(
        default=None, description="alpha_source · alpha_target — what the "
                                  "rank-one (scalar-star) law claims alpha_pair is")
    delta_excess: Optional[float] = Field(
        default=None, description="alpha_pair − alpha_source·alpha_target")
    rho_offaxis: Optional[float] = Field(
        default=None, description="delta_excess / (sin_A · sin_B) — the overlap "
                                  "of the chart components ORTHOGONAL to the hub "
                                  "axis; None when either sin is 0")
    ceiling_target: Optional[float] = Field(
        default=None, description="||zeta_B|| — the same ceiling reported beside "
                                  "the prediction, recomputed in hub coordinates")
    identity_a_comp: Optional[float] = Field(
        default=None, description="alpha_pair · ceiling_target — must equal the "
                                  "filed â_comp")
    identity_abs_delta: Optional[float] = Field(
        default=None, description="|â_comp − alpha_pair·ceiling_target|; the "
                                  "decomposition is corroborated iff this is "
                                  "below the tolerance")
    identity_holds: Optional[bool] = None
    hub_basis_max_dev: Optional[float] = Field(
        default=None, description="max |va_source − va_target|: 0 means the two "
                                  "maps share one hub frame, which the identity "
                                  "requires")
    hub_vector: Optional[str] = Field(
        default=None, description="the hub's own entropy-gradient bank used for "
                                  "xi; None => the alpha_* endpoint terms are "
                                  "not computable and are reported absent")
    notes: list[str] = []


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
    descriptive_companions: Optional[DescriptiveCompanions] = Field(
        default=None,
        description="the hub-frame decomposition — DESCRIPTIVE ONLY, clearly "
                    "separated from a_comp/magnitude_only and from anything a "
                    "band or gate reads. Never scored.")
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


# ------------------------------------------------- descriptive companions (c)
#  THE HUB-FRAME DECOMPOSITION. Operationalization of record:
#  `staging/chart-overlap-preview/chart_overlap_preview.py` (its module
#  docstring derives the construction and its --selftest verifies it
#  numerically). The convention is reused here EXACTLY, not re-derived:
#
#      m_M    := vb_M @ v_M        model M's banked vector in its own map basis
#      zeta_M := omega_M @ m_M     the SAME vector in HUB coordinates
#      xi     := va @ v_hub        the hub's own vector in hub coordinates
#
#  with ||zeta_M|| = ceil_M. Two exact consequences:
#
#      â(hub→M) = cos(xi, zeta_M) · ceil_M      => alpha_M   = cos(xi, zeta_M)
#      â_comp(A→B) = cos(zeta_A, zeta_B) · ceil_B => alpha_AB = cos(zeta_A, zeta_B)
#
#  So `alpha_AB` is NOT a new number: â_comp = alpha_AB · ceil_B is an identity,
#  and this block is a DECOMPOSITION of the filed prediction. The rank-one
#  (scalar-star) law is exactly the claim alpha_AB = alpha_A · alpha_B, so the
#  excess delta = alpha_AB − alpha_A·alpha_B and its normalized form
#  rho = delta / (sin_A · sin_B) measure the off-axis chart overlap.
#
#  Both consequences hold only over ONE common hub-side basis, which is checked
#  rather than assumed. And per rake M19 — instrumentation that only DESCRIBES a
#  run must never be able to FAIL it — every failure mode here degrades to
#  `None` + a named note, and the prediction files regardless.


def hub_frame_coordinates(tm: TransportMap, vector: np.ndarray) -> np.ndarray:
    """zeta_M = omega_M @ (vb_M @ v_M) — the banked vector in HUB coordinates.

    Loud on a map that cannot carry the construction: only a proc map has the
    (va, vb, omega) triple the hub frame is defined over, and a ridge map has no
    such coordinates at all.
    """
    if tm.kind != "proc" or tm.vb is None or tm.omega is None:
        raise ComposedPathError(
            f"hub-frame coordinates need a proc map with vb/omega; got kind "
            f"{tm.kind!r} — the chart decomposition is undefined for it")
    vb = np.asarray(tm.vb, np.float64)
    omega = np.asarray(tm.omega, np.float64)
    return omega @ (vb @ np.asarray(vector, np.float64))


def load_hub_vector(path: Path = HUB_VECTOR_PATH
                    ) -> tuple[Optional[np.ndarray], Optional[str]]:
    """The hub's own vector for `xi`, or (None, None) if it is not banked."""
    if not path.exists():
        return None, None
    vector, spec = load_entropy_gradient(path, HUB_MODEL, HUB_SITE_OF_RECORD)
    return np.asarray(vector, np.float64), spec.path


def chart_companions(tm_source: TransportMap, tm_target: TransportMap,
                     v_source: np.ndarray, v_target: np.ndarray,
                     a_comp: float,
                     hub_vector_path: Path = HUB_VECTOR_PATH,
                     ) -> DescriptiveCompanions:
    """The descriptive companion block for one composed prediction.

    Computed from the SAME two maps and two vectors the prediction was computed
    from — nothing is re-resolved, nothing is re-loaded except the hub's own
    vector, and `a_comp` is passed IN rather than recomputed so this can only
    ever describe the filed number, never replace it.
    """
    block = DescriptiveCompanions()
    notes: list[str] = []
    try:
        zeta_a = hub_frame_coordinates(tm_source, v_source)
        zeta_b = hub_frame_coordinates(tm_target, v_target)
    except ComposedPathError as exc:
        block.notes = [f"companions NOT computed: {exc}"]
        return block

    va_a, va_b = tm_source.va, tm_target.va
    if va_a is None or va_b is None or va_a.shape != va_b.shape:
        block.notes = ["companions NOT computed: the two hub maps do not carry "
                       "comparable hub-side bases, so there is no common hub "
                       "frame for the decomposition"]
        return block
    dev = float(np.abs(np.asarray(va_a, np.float64)
                       - np.asarray(va_b, np.float64)).max())
    block.hub_basis_max_dev = dev
    if dev > HUB_BASIS_MAX_DEV_TOLERANCE:
        block.notes = [
            f"companions NOT computed: the two hub maps' hub-side bases differ "
            f"by {dev:.3e} (> {HUB_BASIS_MAX_DEV_TOLERANCE:.0e}) — they were not "
            f"fit over one common hub frame, and the identity â_comp = "
            f"alpha_pair·ceil_B does not hold across different frames"]
        return block

    ceiling_target = float(np.linalg.norm(zeta_b))
    alpha_pair = float(cos(zeta_a, zeta_b))
    identity = alpha_pair * ceiling_target
    block.alpha_pair = alpha_pair
    block.ceiling_target = ceiling_target
    block.identity_a_comp = identity
    block.identity_abs_delta = abs(a_comp - identity)
    block.identity_holds = bool(block.identity_abs_delta
                                <= COMPANION_IDENTITY_TOLERANCE)
    if not block.identity_holds:
        notes.append(
            f"IDENTITY DID NOT HOLD: |â_comp − alpha_pair·ceiling_target| = "
            f"{block.identity_abs_delta:.3e} > "
            f"{COMPANION_IDENTITY_TOLERANCE:.0e}. The prediction stands (it is "
            f"the predictor's own arithmetic); the DECOMPOSITION is what is "
            f"suspect and must not be read.")

    v_hub, hub_path = load_hub_vector(hub_vector_path)
    if v_hub is None:
        notes.append(
            f"the hub's own entropy-gradient bank is not present at "
            f"{hub_vector_path}, so xi — and with it alpha_source, "
            f"alpha_target, delta_excess and rho_offaxis — is not computable. "
            f"alpha_pair and the identity above are unaffected (they need only "
            f"the two charts).")
        block.notes = notes
        return block
    basis = np.asarray(va_a, np.float64)
    if int(basis.shape[1]) != int(v_hub.shape[0]):
        notes.append(
            f"the hub vector at {hub_path} has dim {int(v_hub.shape[0])} but the "
            f"hub-side basis expects {int(basis.shape[1])} — xi is not "
            f"computable, so the alpha_* endpoint terms are reported absent "
            f"rather than computed against a mismatched hub column")
        block.notes = notes
        return block
    block.hub_vector = hub_path
    xi = basis @ v_hub
    alpha_a = float(cos(xi, zeta_a))
    alpha_b = float(cos(xi, zeta_b))
    block.alpha_source, block.alpha_target = alpha_a, alpha_b
    block.rank1_implied = alpha_a * alpha_b
    block.delta_excess = alpha_pair - block.rank1_implied
    sin_a = float(np.sqrt(max(0.0, 1.0 - alpha_a ** 2)))
    sin_b = float(np.sqrt(max(0.0, 1.0 - alpha_b ** 2)))
    if sin_a > 0.0 and sin_b > 0.0:
        block.rho_offaxis = block.delta_excess / (sin_a * sin_b)
    else:
        notes.append("rho_offaxis undefined: an endpoint chart lies exactly on "
                     "the hub axis (sin = 0), so there is no off-axis component "
                     "to correlate")
    block.notes = notes
    return block


# ---------------------------------------------------------------- resolution
def resolve_hub_map(model: str, site: int, arm: str, family: str) -> HubMapRef:
    """Find the banked primary-hub→`model` map, or report it absent as data.

    Absent is NOT an error here — draft E1 makes it the N/A-AT-FILING verdict,
    and the returned record carries every path probed so the desk can see what
    was looked for. Use `require_hub_map` when a caller must have the map.

    A FILED-PATHS PIN, when one is set, answers BEFORE any probe and never falls
    back to one (`_pinned_path`). With no pin the probe order is exactly what it
    was — and either way the resolve is recorded in the provenance ledger, which
    names the tree that answered and its vintage.
    """
    pinned = _pinned_path("hub map", hub_map_pin_key(model, site, arm, family),
                          model, site)
    if pinned is not None:
        assert FILED_PATHS is not None
        _record_provenance(ResolutionProvenance(
            kind="hub map", model=model, site=site, arm=arm, family=family,
            source="pin", resolved=pinned.path, corpus=pinned.corpus,
            tree=f"filed-paths pin: {FILED_PATHS.record} (slot "
                 f"{pinned.filed_in_slot})",
            v21_root=None if V21_ROOT is None else str(V21_ROOT),
            pin_record=FILED_PATHS.record))
        return HubMapRef(
            model=model, site=site, arm=arm, family=family,
            resolved=pinned.path, corpus=pinned.corpus,
            dir_note=f"FILED-PATHS PIN — taken from {FILED_PATHS.record} as "
                     f"filed, not probed",
            probed_paths=[pinned.path])
    probed: list[str] = []
    unpinned_note = _unpinned_notes("hub map",
                                    hub_map_pin_key(model, site, arm, family))
    for index, candidate in enumerate(hub_map_dirs(model)):
        path = fit_path_for(candidate.path, HUB_MODEL, HUB_SITE_OF_RECORD,
                            model, site, arm, family)
        probed.append(str(path))
        if path.exists():
            _record_provenance(ResolutionProvenance(
                kind="hub map", model=model, site=site, arm=arm, family=family,
                source="probe", resolved=str(path), corpus=candidate.corpus,
                tree=candidate.note or str(candidate.path),
                probe_index=index, n_probed=len(probed), probed_paths=probed,
                v21_root=None if V21_ROOT is None else str(V21_ROOT),
                pin_record=None if FILED_PATHS is None else FILED_PATHS.record,
                cross_vintage_fallback=(V21_ROOT is not None
                                        and candidate.corpus != "v21"),
                notes=unpinned_note))
            return HubMapRef(model=model, site=site, arm=arm, family=family,
                             resolved=str(path), corpus=candidate.corpus,
                             dir_note=candidate.note, probed_paths=probed)
    _record_provenance(ResolutionProvenance(
        kind="hub map", model=model, site=site, arm=arm, family=family,
        source="absent", tree="none — every probe missed (N/A-AT-FILING)",
        n_probed=len(probed), probed_paths=probed,
        v21_root=None if V21_ROOT is None else str(V21_ROOT),
        pin_record=None if FILED_PATHS is None else FILED_PATHS.record,
        notes=unpinned_note))
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


def _vector_tree_note(path: Path, site: int) -> str:
    """Which tree a vector-bank probe answered from, in words.

    `vector_bank_paths` offers at most three: the verified v2.1 mirror's
    per-site bank, the collection tree's MERGED bank, and the collection tree's
    per-site build. They are told apart by where they are and how they are
    named, never by probe position, so the note stays right if the order moves.
    """
    if _under_v21_root(path):
        return "corpus-v2.1 re-bank mirror (verified root, per-site bank)"
    if path.stem.endswith(f"_L{site}"):
        return "collection tree, per-site build"
    return "collection tree, merged bank"


def resolve_vector_bank(model: str, site: int) -> VectorBankRef:
    """Find `model`'s banked entropy-gradient vectors, or report them absent.

    Absent is data, not an exception: a node whose hub map is banked but whose
    target build has not run yet (gemma3-27b today) must produce a NAMED gap
    rather than a `FileNotFoundError` from inside the predictor.

    Pinned and recorded exactly as `resolve_hub_map` is: a filed-paths pin
    answers first and never falls back, and every resolve lands in the
    provenance ledger naming the tree that answered.
    """
    pinned = _pinned_path("vector bank", vector_bank_pin_key(model, site),
                          model, site)
    if pinned is not None:
        assert FILED_PATHS is not None
        #  Corpus stays as `VectorBankRef.corpus` documents it: `v21` only when
        #  the path is inside a VERIFIED root right now. A pin carries the
        #  record's word for the vintage, and the record's word is not a
        #  verification — it rides the provenance record, never the ref.
        _record_provenance(ResolutionProvenance(
            kind="vector bank", model=model, site=site, source="pin",
            resolved=pinned.path,
            corpus="v21" if _under_v21_root(pinned.path) else None,
            tree=f"filed-paths pin: {FILED_PATHS.record} (slot "
                 f"{pinned.filed_in_slot})",
            v21_root=None if V21_ROOT is None else str(V21_ROOT),
            pin_record=FILED_PATHS.record))
        return VectorBankRef(
            model=model, site=site, resolved=pinned.path,
            corpus="v21" if _under_v21_root(pinned.path) else None,
            probed_paths=[pinned.path])
    probed: list[str] = []
    unpinned_note = _unpinned_notes("vector bank", vector_bank_pin_key(model, site))
    for index, path in enumerate(vector_bank_paths(model, site)):
        probed.append(str(path))
        if path.exists():
            in_v21 = _under_v21_root(path)
            _record_provenance(ResolutionProvenance(
                kind="vector bank", model=model, site=site, source="probe",
                resolved=str(path), corpus="v21" if in_v21 else None,
                tree=_vector_tree_note(path, site), probe_index=index,
                n_probed=len(probed), probed_paths=probed,
                v21_root=None if V21_ROOT is None else str(V21_ROOT),
                pin_record=None if FILED_PATHS is None else FILED_PATHS.record,
                cross_vintage_fallback=V21_ROOT is not None and not in_v21,
                notes=unpinned_note))
            return VectorBankRef(model=model, site=site, resolved=str(path),
                                 corpus="v21" if in_v21 else None,
                                 probed_paths=probed)
    _record_provenance(ResolutionProvenance(
        kind="vector bank", model=model, site=site, source="absent",
        tree="none — every probe missed (the target build has not landed)",
        n_probed=len(probed), probed_paths=probed,
        v21_root=None if V21_ROOT is None else str(V21_ROOT),
        pin_record=None if FILED_PATHS is None else FILED_PATHS.record,
        notes=unpinned_note))
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

    #  DESCRIPTIVE COMPANIONS — computed AFTER â_comp, from the objects already
    #  loaded, and passed the finished â_comp so they can only describe it.
    #  Rake M19: instrumentation that only describes a run must never be able to
    #  fail it, so any failure here degrades to a note-bearing block (or None)
    #  and the prediction files unchanged.
    try:
        companions: Optional[DescriptiveCompanions] = chart_companions(
            tm_src, tm_tgt, v_src, v_tgt, a_comp)
    except Exception as exc:                                 # noqa: BLE001
        companions = DescriptiveCompanions(
            notes=[f"companions NOT computed — {type(exc).__name__}: {exc}. The "
                   f"prediction is unaffected: â_comp is the predictor's own "
                   f"arithmetic and nothing below this line feeds it."])
        logger.warning("descriptive companions failed for %s: %s", pair_id, exc)

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
    if FILED_PATHS is not None:
        #  The pin decides resolution, so say so on the slot itself rather than
        #  leave a reader to infer it from paths. It also explains the sha:
        #  `corpus_manifest_sha256` below is populated from a root VERIFIED this
        #  run, and a pin verifies no root — so a wholly-v2.1-as-resolved slot
        #  can carry the vintage label and no sha, which would otherwise read
        #  as a bug rather than as the honest state.
        flags.append(
            f"FILED-PATHS PIN in force ({FILED_PATHS.record}): every artifact "
            f"above was taken from that record's own `resolved` paths, not "
            f"probed. corpus_manifest_sha256 is populated only from a root "
            f"VERIFIED this run, which a pin does not perform.")

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
        source_vector=spec_src, target_vector=spec_tgt,
        descriptive_companions=companions, flags=flags)


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


# --------------------------------------------- the directional star (ADD. H)
#  ADDENDUM 2026-07-29-H, item 3: â(A→B) = c_A^out · c_B^in files as a THIRD
#  predictor column beside the symmetric star and the composed path, on
#  structurally identical terms. Item 2: the symmetric star column is UNTOUCHED
#  — nothing below reads, recomputes or moves it.
#
#  THE SCHEMA OF RECORD — `directional-constants-readout/v1`
#  ─────────────────────────────────────────────────────────────────────────────
#  Produced by the derivation lane (H item 5: c^out and c^in hub-derived on
#  corpus-v2.1 from BOTH fit directions, anchored per the §3 hub protocol).
#  Consumed here, never derived here.
#
#    {
#      "schema": "directional-constants-readout/v1",   REQUIRED, exact string
#      "corpus_manifest_sha256": "<64 lowercase hex>", REQUIRED  (G2(a))
#      "gauge": "<the anchor/gauge of record, free text>",  REQUIRED
#      "generated":  "YYYY-MM-DD",                     optional
#      "derivation": "<free text>",                    optional
#      "hub": "8b",                                    optional, CHECKED
#      "hub_site": 16,                                 optional, CHECKED
#      "arm":    "native",                             optional row default
#      "family": "proc_k128",                          optional row default
#      "constants": [                                  REQUIRED, non-empty
#        {"model": "qwen2.5-32b-instruct",             REQUIRED
#         "arm":    "native",                          REQUIRED (or doc default)
#         "family": "proc_k128",                       REQUIRED (or doc default)
#         "c_out": 0.6642,                             REQUIRED, finite
#         "c_in":  0.5441,                             REQUIRED, finite
#         "site": 46,                                  optional, CROSS-CHECKED
#         "corpus_manifest_sha256": "<64 hex>",        optional row override
#         "note": "<free text>"}                       optional
#      ]
#    }
#
#  `constants` may equivalently be an OBJECT mapping model → the same row
#  WITHOUT its `model` key. Both encodings are named in the contract and in
#  every error message; a third encoding is a HALT, never an inference.
#
#  WHAT IS CHECKED, AND WHY EACH CHECK IS A HALT
#    * `schema` exact-match — a v2 readout parsed as v1 would file numbers whose
#      meaning moved underneath their names;
#    * doc-level corpus sha, 64 lowercase hex — G2(a) makes the vintage tag part
#      of the â, and a tag that is not a digest is not a tag;
#    * every model resolvable through `site_of_record` — an unknown or DEFERRED
#      key (gpt2-xl) has no candidate slot, so a constant for it is a lane
#      disagreement, not a bonus row;
#    * a row's `site`, when present, must EQUAL the site registry's — the same
#      halt `load_candidate_slots` makes, for the same reason: two artifacts
#      describing different objects under one name;
#    * (model, arm, family) unique — rake M18: key by the FULL identity and
#      assert no duplicates, because "which c^out is of record" cannot be
#      guessed;
#    * c_out / c_in finite. |c| > 1 is FLAGGED on the slot (a gauge-divided
#      chain-break read can legitimately exceed 1), never silently dropped.


class DirectionalConstantRow(BaseModel):
    """One model's directional pair (c^out, c^in) in ONE arm × family."""
    model: str
    arm: str
    family: str
    c_out: float = Field(description="the SOURCE-role constant: â(A→·) ∝ c_A^out")
    c_in: float = Field(description="the TARGET-role constant: â(·→B) ∝ c_B^in")
    site: Optional[int] = Field(
        default=None, description="the readout's own site claim, cross-checked "
                                  "against SITE_OF_RECORD; None => not claimed")
    corpus_manifest_sha256: Optional[str] = None
    note: str = ""
    unrecognized_keys: list[str] = Field(
        default=[], description="keys this schema version does not know — "
                                "carried so a silent schema drift is VISIBLE "
                                "rather than dropped on the floor")

    model_config = {"protected_namespaces": ()}

    @property
    def key(self) -> str:
        return f"{self.model}/{self.arm}-{self.family}"


class DirectionalConstants(BaseModel):
    """A parsed, validated `directional-constants-readout/v1`.

    Holds the readout's own provenance beside the rows: which file, which sha,
    which gauge, which corpus vintage. Every directional prediction quotes it,
    so a filed slot can be traced to the constants table it rode without
    consulting a ledger row (the same discipline `HubMapRef` applies to maps).
    """
    schema_name: str = SCHEMA_DIRECTIONAL_CONSTANTS_V1
    path: str
    sha256: str
    generated: Optional[str] = None
    corpus_manifest_sha256: str
    gauge: str
    derivation: str = ""
    hub: str = HUB_MODEL
    hub_site: int = HUB_SITE_OF_RECORD
    rows: list[DirectionalConstantRow] = []
    notes: list[str] = []

    def row(self, model: str, arm: str, family: str
            ) -> Optional[DirectionalConstantRow]:
        """The row for this model IN THIS ARM, or None. NEVER cross-arm.

        Addendum H item 3 scores the directional column on E1's terms; E1's
        arm-availability rule forbids proxying a quantity from another arm, and
        a scalar constant is no more proxiable than a map.
        """
        for candidate in self.rows:
            if (candidate.model == model and candidate.arm == arm
                    and candidate.family == family):
                return candidate
        return None

    @property
    def arms(self) -> list[str]:
        return sorted({row.arm for row in self.rows})


def _is_sha256_hex(value: Any) -> bool:
    """A 64-character lowercase hex digest, and nothing else."""
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


#: Keys `directional-constants-readout/v1` knows at row level. Anything else is
#: recorded on the row (never dropped, never obeyed) so schema drift is seen.
_DIRECTIONAL_ROW_KEYS = frozenset(
    {"model", "arm", "family", "c_out", "c_in", "site",
     "corpus_manifest_sha256", "note"})


def _directional_schema_halt(path: Path, problem: str) -> DirectionalConstantsError:
    """One place that builds the contract-quoting halt, so every failure path
    says exactly the same thing about what was expected."""
    return DirectionalConstantsError(
        f"{path}: {problem}\n"
        f"EXPECTED `{SCHEMA_DIRECTIONAL_CONSTANTS_V1}`:\n"
        f"  top level  : object with REQUIRED `schema` == "
        f"{SCHEMA_DIRECTIONAL_CONSTANTS_V1!r}, REQUIRED "
        f"`corpus_manifest_sha256` (64 lowercase hex), REQUIRED `gauge` (free "
        f"text provenance), REQUIRED non-empty `constants`; OPTIONAL "
        f"`generated`, `derivation`, `hub` (must be {HUB_MODEL!r} if present), "
        f"`hub_site` (must be {HUB_SITE_OF_RECORD} if present), `arm`, "
        f"`family` (the last two are row defaults).\n"
        f"  `constants`: EITHER a list of row objects each carrying `model`, OR "
        f"an object mapping model -> the same row object without `model`. No "
        f"third encoding is accepted.\n"
        f"  each row   : REQUIRED `model` (a key of SITE_OF_RECORD), REQUIRED "
        f"`arm` and `family` (or the doc-level defaults), REQUIRED finite "
        f"`c_out` and `c_in`; OPTIONAL `site` (must equal the site registry's), "
        f"`corpus_manifest_sha256`, `note`. (model, arm, family) must be "
        f"UNIQUE across the readout.\n"
        f"This module CONSUMES the readout and never derives it: if the "
        f"derivation lane's actual output differs from the above, the contract "
        f"is what must be reconciled — not the parse.")


def load_directional_constants(path: Path) -> DirectionalConstants:
    """Read and VALIDATE a `directional-constants-readout/v1` readout.

    Every failure raises `DirectionalConstantsError` quoting the whole expected
    contract, so a mismatch with the derivation lane's actual output is legible
    from the error alone (that is the point of naming and versioning the schema
    before the file exists).
    """
    if not path.is_file():
        raise _directional_schema_halt(
            path, "directional-constants readout absent. Addendum H's third "
                  "column cannot file without one, and no constant is ever "
                  "defaulted, inferred or carried from another readout")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise _directional_schema_halt(
            path, f"unreadable as JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise _directional_schema_halt(
            path, f"top level is {type(doc).__name__}, not an object")

    declared = doc.get("schema")
    if declared != SCHEMA_DIRECTIONAL_CONSTANTS_V1:
        raise _directional_schema_halt(
            path, f"declares schema {declared!r}, not "
                  f"{SCHEMA_DIRECTIONAL_CONSTANTS_V1!r}. A readout of another "
                  f"name or version is NEVER parsed on the assumption that its "
                  f"fields still mean what they meant here")

    corpus_sha = doc.get("corpus_manifest_sha256")
    if not _is_sha256_hex(corpus_sha):
        raise _directional_schema_halt(
            path, f"`corpus_manifest_sha256` is {corpus_sha!r}, not 64 "
                  f"lowercase hex. Addendum G §G2(a) rides a corpus manifest "
                  f"sha on every quoted â; an unverifiable tag is worse than a "
                  f"missing one")
    gauge = doc.get("gauge")
    if not isinstance(gauge, str) or not gauge.strip():
        raise _directional_schema_halt(
            path, f"`gauge` is {gauge!r}; the gauge/anchor of record is "
                  f"REQUIRED provenance — c^out and c^in are only meaningful "
                  f"relative to the anchor that fixed them")
    hub = doc.get("hub", HUB_MODEL)
    if hub != HUB_MODEL:
        raise _directional_schema_halt(
            path, f"`hub` is {hub!r}, but this module's hub of record is "
                  f"{HUB_MODEL!r} at L{HUB_SITE_OF_RECORD}")
    hub_site = doc.get("hub_site", HUB_SITE_OF_RECORD)
    if hub_site != HUB_SITE_OF_RECORD:
        raise _directional_schema_halt(
            path, f"`hub_site` is {hub_site!r}, but the hub column of record is "
                  f"L{HUB_SITE_OF_RECORD} (rebuilt-L16)")

    raw = doc.get("constants")
    if isinstance(raw, dict):
        entries = []
        for model_key, body in raw.items():
            if not isinstance(body, dict):
                raise _directional_schema_halt(
                    path, f"`constants[{model_key!r}]` is "
                          f"{type(body).__name__}, not an object")
            if "model" in body and body["model"] != model_key:
                raise _directional_schema_halt(
                    path, f"`constants[{model_key!r}]` carries model "
                          f"{body['model']!r} — the mapping key and the row's "
                          f"own `model` disagree, and which one is of record "
                          f"cannot be guessed")
            entries.append({**body, "model": model_key})
    elif isinstance(raw, list):
        entries = list(raw)
    else:
        raise _directional_schema_halt(
            path, f"`constants` is {type(raw).__name__ if raw is not None else 'absent'}"
                  f", not a list of rows or a model->row object")
    if not entries:
        raise _directional_schema_halt(
            path, "`constants` is empty — a readout that names no constant "
                  "cannot back a single directional slot")

    default_arm = doc.get("arm")
    default_family = doc.get("family")
    rows: list[DirectionalConstantRow] = []
    seen: dict[str, int] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _directional_schema_halt(
                path, f"`constants[{i}]` is {type(entry).__name__}, not an object")
        model = entry.get("model")
        if not isinstance(model, str) or not model:
            raise _directional_schema_halt(
                path, f"`constants[{i}]` has no usable `model` (got {model!r})")
        arm = entry.get("arm", default_arm)
        family = entry.get("family", default_family)
        if not isinstance(arm, str) or arm not in ARMS:
            raise _directional_schema_halt(
                path, f"`constants[{i}]` ({model}): arm {arm!r} is not one of "
                      f"{ARMS} and no usable doc-level `arm` default is set. "
                      f"The arm is load-bearing — H item 3 scores on E1's "
                      f"terms and E1 forbids proxying across arms")
        if not isinstance(family, str) or family not in FAMILIES:
            raise _directional_schema_halt(
                path, f"`constants[{i}]` ({model}): family {family!r} is not "
                      f"one of {FAMILIES} and no usable doc-level `family` "
                      f"default is set")
        values: dict[str, float] = {}
        for field in ("c_out", "c_in"):
            value = entry.get(field)
            try:
                as_float = float(value)                 # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise _directional_schema_halt(
                    path, f"`constants[{i}]` ({model}): `{field}` is {value!r}, "
                          f"which is not a number") from exc
            if not np.isfinite(as_float):
                raise _directional_schema_halt(
                    path, f"`constants[{i}]` ({model}): `{field}` is "
                          f"{as_float!r} — a non-finite constant has no product "
                          f"that can be banded, so it is refused rather than "
                          f"filed")
            values[field] = as_float
        try:
            registry_site = site_of_record(model)
        except (ComposedPathError, FitGridError) as exc:
            raise _directional_schema_halt(
                path, f"`constants[{i}]`: {exc}") from exc
        claimed_site = entry.get("site")
        if claimed_site is not None and int(claimed_site) != registry_site:
            raise _directional_schema_halt(
                path, f"`constants[{i}]` ({model}): the readout says L"
                      f"{int(claimed_site)} but the site registry says "
                      f"L{registry_site}. Two artifacts describing different "
                      f"objects under one name is a HALT, never a preference")
        row_sha = entry.get("corpus_manifest_sha256", corpus_sha)
        if not _is_sha256_hex(row_sha):
            raise _directional_schema_halt(
                path, f"`constants[{i}]` ({model}): row "
                      f"`corpus_manifest_sha256` is {row_sha!r}, not 64 "
                      f"lowercase hex")
        row = DirectionalConstantRow(
            model=model, arm=arm, family=family, c_out=values["c_out"],
            c_in=values["c_in"], site=registry_site,
            corpus_manifest_sha256=row_sha, note=str(entry.get("note", "")),
            unrecognized_keys=sorted(set(entry) - _DIRECTIONAL_ROW_KEYS))
        if row.key in seen:
            raise _directional_schema_halt(
                path, f"duplicate constants row for {row.key} (entries "
                      f"{seen[row.key]} and {i}) — rake M18: key by the FULL "
                      f"identity and assert no duplicates; which c^out is of "
                      f"record cannot be guessed")
        seen[row.key] = i
        rows.append(row)

    notes: list[str] = []
    drifted = sorted({k for row in rows for k in row.unrecognized_keys})
    if drifted:
        notes.append(
            f"SCHEMA DRIFT (reported, not obeyed): the readout carries row keys "
            f"{drifted} that {SCHEMA_DIRECTIONAL_CONSTANTS_V1} does not define. "
            f"They are recorded per row and IGNORED — if they are load-bearing, "
            f"the schema needs a version bump, not a silent read")
    mixed = sorted({r.corpus_manifest_sha256 for r in rows
                    if r.corpus_manifest_sha256 != corpus_sha
                    and r.corpus_manifest_sha256 is not None})
    if mixed:
        notes.append(
            f"MIXED-VINTAGE READOUT: {len(mixed)} row-level corpus sha(s) "
            f"differ from the document's {corpus_sha[:8]}…. Per-slot vintage is "
            f"resolved from the two rows a slot actually uses; a slot whose two "
            f"constants disagree carries NO sha (Addendum G §G2(a))")
    logger.info("directional constants: %d row(s), arms %s, gauge %r, corpus "
                "%s… (%s)", len(rows), sorted({r.arm for r in rows}), gauge,
                corpus_sha[:8], SCHEMA_DIRECTIONAL_CONSTANTS_V1)
    return DirectionalConstants(
        path=str(path), sha256=sha256_of(path), generated=doc.get("generated"),
        corpus_manifest_sha256=corpus_sha, gauge=gauge,
        derivation=str(doc.get("derivation", "")), hub=hub, hub_site=hub_site,
        rows=rows, notes=notes)


class DirectionalPrediction(BaseModel):
    """One â(A→B) = c_A^out · c_B^in, filable under Addendum H item 3.

    Carries its OWN frozen band, because H item 3 freezes the ±.05 band at
    filing: the value that files and the band that files are computed together,
    from the same rounding, so no downstream glue can pair a rounded prediction
    with an unrounded band.
    """
    prediction_id: str = Field(
        description="Addendum H item 3 ID shape, mirroring E2's: "
                    "directional-prediction/<src>→<tgt>/<arm>-k128")
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    arm_rule: str
    family: str
    estimand: str = (
        "â(A→B) = c_A^out · c_B^in — the Addendum-C item-2 directional star, "
        "ADOPTED by Addendum 2026-07-29-H for not-yet-filed slots")
    c_out_source: float
    c_in_target: float
    predicted: float = Field(
        description="c_A^out · c_B^in at full precision — the arithmetic")
    band: list[float] = Field(
        description="the FROZEN ±.05 absolute band of `predicted`, [lo, hi]")
    filed_predicted: float = Field(
        description=f"`predicted` at the {FILED_DECIMALS}-dp filing convention "
                    f"of record (batches canary/2/3)")
    filed_band: list[float] = Field(
        description="the frozen ±.05 band OF THE FILED VALUE — the band that "
                    "goes into the record, so the two never disagree")
    magnitude_only: bool = Field(
        description="Addendum H item 3 mirrors prereg §3 / E2: |predicted| < "
                    ".08 → sign unscored, MAGNITUDE-ONLY")
    corpus_vintage: PredictionVintage = Field(
        default="pre-v2.1",
        description="the vintage of BOTH constants together; MIXED when the "
                    "two rows disagree, and MIXED carries no sha")
    corpus_manifest_sha256: Optional[str] = Field(
        default=None,
        description="Addendum G §G2(a) / H item 5 — the corpus manifest sha "
                    "this â rides on, taken from the constants rows themselves")
    constants_readout: str = Field(description="the readout file consumed")
    constants_readout_sha256: str
    constants_gauge: str = Field(
        description="the anchor/gauge of record the constants were fixed "
                    "against — c^out and c^in are only meaningful relative to it")
    descriptive_reverse_predicted: Optional[float] = Field(
        default=None,
        description="DESCRIPTIVE, NEVER FILED, NEVER SCORED: c_B^out · c_A^in, "
                    "the reverse-orientation value. Free from the same two "
                    "rows; recorded because the orientation asymmetry is the "
                    "trigger evidence Addendum H rests on")
    descriptive_asymmetry_ratio: Optional[float] = Field(
        default=None,
        description="DESCRIPTIVE: predicted / reverse, the â(A→B)/â(B→A) form "
                    "quoted in Addendum H (1.320 at the family of record). "
                    "None when the reverse value is 0")
    flags: list[str] = []


class DirectionalNotFilable(BaseModel):
    """A directional slot with no constant behind it — N/A-AT-FILING.

    Distinct from the composed column's `NotFilable` and deliberately so: the
    composed slot is unbacked when a banked MAP is missing, the directional slot
    when a banked CONSTANT is. Reporting them under one model would let a reader
    conclude that a pair is unfilable for both columns when it is unfilable for
    one — the two columns race, and their availability is independent.
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
    constants_readout: str
    constants_readout_sha256: str
    available_keys: list[str] = Field(
        default=[], description="every (model/arm-family) the readout DOES "
                                "carry, so the gap is legible without opening "
                                "the readout")
    reason: str = (
        "Addendum H item 3 files the directional column on Addendum E §E1's "
        "terms: a slot files only where the SOURCE has a c^out and the TARGET a "
        "c^in in the pair's applicable arm. Never proxied from another arm, "
        "never from the other role's constant, never backfilled after the pair "
        "is fit.")


class DirectionalReadout(BaseModel):
    """The directional column for a set of candidate slots.

    A SEPARATE artifact from `ComposedReadout` on purpose: the composed readout
    is the record of an existing, already-filed-against column, and bolting a
    new column into its shape would move a document other tools read. The
    directional column arrives beside it, in its own file, and the desk's
    filing glue lays the two side by side in one filing record.
    """
    STATUS: str = (
        "UNSTAMPED — computation only. Fits nothing, refits nothing, FILES NO "
        "PREDICTION, moves no band, writes nothing under outputs/. The "
        "SYMMETRIC star column is untouched by everything here (Addendum H "
        "item 2). The desk rules.")
    estimand: str = (
        "â(A→B) = c_A^out · c_B^in, per-model directional constants read from a "
        f"{SCHEMA_DIRECTIONAL_CONSTANTS_V1} readout in the pair's applicable "
        "arm; ±.05 absolute band frozen at filing; |predicted| < .08 "
        "MAGNITUDE-ONLY (Addendum 2026-07-29-H item 3)")
    generated: str
    hub: str = HUB_MODEL
    hub_site: int = HUB_SITE_OF_RECORD
    family: str
    direction_rule: str = (
        "source→target AS ENUMERATED (the candidate column's model_a→model_b, "
        "the same direction the composed column takes and the same "
        "'forward as listed' direction of record batch 3 ruled). â is "
        "orientation-dependent and the directional star is asymmetric BY "
        "CONSTRUCTION, so the orientation is never inferred here")
    constants_readout: str
    constants_readout_sha256: str
    constants_gauge: str
    constants_corpus_manifest_sha256: str
    constants_notes: list[str] = []
    predictions: list[DirectionalPrediction] = []
    na_at_filing: list[DirectionalNotFilable] = []


def _frozen_band(predicted: float) -> list[float]:
    """`predicted` ± the FROZEN half-width. The only band arithmetic there is."""
    return [predicted - FROZEN_BAND_HALF_WIDTH,
            predicted + FROZEN_BAND_HALF_WIDTH]


def directional_pair(source_model: str, target_model: str,
                     constants: DirectionalConstants,
                     family: str = FAMILY_OF_RECORD,
                     arm: Optional[str] = None,
                     ) -> DirectionalPrediction | DirectionalNotFilable:
    """â(A→B) = c_A^out · c_B^in for one slot, or its N/A-AT-FILING record.

    `arm` defaults to the pair's applicable arm (prereg §3); overriding it is a
    diagnostic and is echoed into the record with a flag, exactly as
    `compose_pair` does. The filing path never overrides.
    """
    s_site = site_of_record(source_model)
    t_site = site_of_record(target_model)
    resolved_arm, arm_rule = applicable_arm(source_model, target_model)
    if arm is not None and arm != resolved_arm:
        arm_rule = (f"OVERRIDDEN to {arm!r} (applicable arm by prereg §3 is "
                    f"{resolved_arm!r}: {arm_rule})")
    use_arm = arm or resolved_arm
    if use_arm not in ARMS:
        raise ComposedPathError(f"unknown arm {use_arm!r}; known: {ARMS}")

    pair_id = f"{source_model}L{s_site}->{target_model}L{t_site}"
    row_src = constants.row(source_model, use_arm, family)
    row_tgt = constants.row(target_model, use_arm, family)
    missing = [label for label, row in
               ((f"source c^out ({source_model}/{use_arm}-{family})", row_src),
                (f"target c^in ({target_model}/{use_arm}-{family})", row_tgt))
               if row is None]
    if missing:
        return DirectionalNotFilable(
            pair_id=pair_id, source_model=source_model, source_site=s_site,
            target_model=target_model, target_site=t_site, arm=use_arm,
            arm_rule=arm_rule, family=family, missing_sides=missing,
            constants_readout=constants.path,
            constants_readout_sha256=constants.sha256,
            available_keys=sorted(row.key for row in constants.rows))

    assert row_src is not None and row_tgt is not None
    predicted = float(row_src.c_out) * float(row_tgt.c_in)
    filed_predicted = round(predicted, FILED_DECIMALS)
    reverse = float(row_tgt.c_out) * float(row_src.c_in)

    flags: list[str] = []
    for role, row, field, value in (("source", row_src, "c_out", row_src.c_out),
                                    ("target", row_tgt, "c_in", row_tgt.c_in)):
        if abs(value) > DIRECTIONAL_CONSTANT_FLAG_ABS:
            flags.append(
                f"OUT-OF-RANGE CONSTANT — {row.model}'s {field} is {value:+.6f}, "
                f"|c| > {DIRECTIONAL_CONSTANT_FLAG_ABS}. A gauge-divided "
                f"chain-break read can legitimately exceed 1, so this is "
                f"reported on the slot rather than refused; the desk rules "
                f"whether the {role} constant is usable")
    if arm is not None and arm != resolved_arm:
        flags.append("ARM OVERRIDDEN — diagnostic only; not a filable slot.")

    #  Addendum G §G2(a): the sha rides the â, and only when the WHOLE
    #  computation rode one vintage. Two constants, both or neither.
    if row_src.corpus_manifest_sha256 == row_tgt.corpus_manifest_sha256:
        corpus_sha = row_src.corpus_manifest_sha256
        vintage: PredictionVintage = (
            "v2.1" if corpus_sha == CORPUS_SHA_V21 else "pre-v2.1")
    else:
        corpus_sha = None
        vintage = "MIXED"
        flags.append(
            f"MIXED-VINTAGE — {source_model}'s c^out rode corpus "
            f"{(row_src.corpus_manifest_sha256 or '')[:8]}… and "
            f"{target_model}'s c^in rode "
            f"{(row_tgt.corpus_manifest_sha256 or '')[:8]}…. The product is "
            f"computable but carries NO corpus sha: Addendum G §G2(a) wants a "
            f"vintage tag on every â and a tag covering half a computation is "
            f"worse than none. Cross-vintage reads are calibration, never "
            f"scored (§G2(c))")

    return DirectionalPrediction(
        prediction_id=f"directional-prediction/{source_model}→{target_model}"
                      f"/{use_arm}-k128",
        pair_id=pair_id, source_model=source_model, source_site=s_site,
        target_model=target_model, target_site=t_site, arm=use_arm,
        arm_rule=arm_rule, family=family,
        c_out_source=float(row_src.c_out), c_in_target=float(row_tgt.c_in),
        predicted=predicted, band=_frozen_band(predicted),
        filed_predicted=filed_predicted,
        filed_band=[round(b, FILED_DECIMALS)
                    for b in _frozen_band(filed_predicted)],
        magnitude_only=bool(abs(predicted) < NEAR_ZERO_CARVE_OUT),
        corpus_vintage=vintage, corpus_manifest_sha256=corpus_sha,
        constants_readout=constants.path,
        constants_readout_sha256=constants.sha256,
        constants_gauge=constants.gauge,
        descriptive_reverse_predicted=reverse,
        descriptive_asymmetry_ratio=(None if reverse == 0.0
                                     else predicted / reverse),
        flags=flags)


def run_directional(constants: DirectionalConstants,
                    family: str = FAMILY_OF_RECORD,
                    pairs_json: Path = CANDIDATE_PAIRS_JSON) -> DirectionalReadout:
    """The directional column for every current candidate slot."""
    readout = DirectionalReadout(
        generated=date.today().isoformat(), family=family,
        constants_readout=constants.path,
        constants_readout_sha256=constants.sha256,
        constants_gauge=constants.gauge,
        constants_corpus_manifest_sha256=constants.corpus_manifest_sha256,
        constants_notes=list(constants.notes))
    for slot in load_candidate_slots(pairs_json):
        result = directional_pair(slot.model_a, slot.model_b, constants,
                                  family=family)
        if isinstance(result, DirectionalNotFilable):
            readout.na_at_filing.append(result)
            logger.warning("N/A-AT-FILING %-52s %s::%s — missing %s",
                           result.pair_id, result.arm, family,
                           ", ".join(result.missing_sides))
            continue
        readout.predictions.append(result)
        logger.info("â_dir %-52s %s::%-9s = %+.6f  band [%+.4f, %+.4f]%s%s",
                    result.pair_id, result.arm, family, result.predicted,
                    result.filed_band[0], result.filed_band[1],
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
    if FILED_PATHS is not None:
        raise CorpusVintageError(
            f"the E1 gate cannot run while a filed-paths pin is set "
            f"({FILED_PATHS.record}). Same trap, other mechanism: the gate's "
            f"leg 2 asserts RESOLUTION PARITY between what this module resolves "
            f"and the ARCHIVE's own npz files, and a pin answers from a filing "
            f"record instead — parity would then be testing the pin, not the "
            f"resolver. Clear it (`set_filed_paths(None)`) and re-run; "
            f"`--gate` and `--filed-paths` are mutually exclusive at the CLI "
            f"too, and this is the programmatic guard on the same trap.")
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

#: The predictors racing under Addendum E + Addendum H. `star` = prereg §3's
#: symmetric scalar c_A·c_B, which Addendum H item 2 leaves UNTOUCHED and
#: filing on its own frozen terms; `composed` = E1's two-hop â_comp;
#: `directional` = H item 3's â(A→B) = c_A^out·c_B^in.
#: Order matters only for display; every aggregate keys by the value.
Predictor = Literal["star", "composed", "directional"]
#: The predictor pairs whose 2×2 the head-to-head read emits, in a FIXED order.
#: `("star", "composed")` is FIRST and its counts are ALSO emitted in the
#: original `HeadToHead` shape — Addendum E §E3's read is preserved verbatim for
#: continuity, and the directional 2×2s arrive BESIDE it, never in place of it.
PREDICTOR_PAIRS: tuple[tuple[Predictor, Predictor], ...] = (
    ("star", "composed"),
    ("star", "directional"),
    ("composed", "directional"),
)
#: Predictor -> the block key it files under in a RACING filing record row.
#: Insertion order is the order predictions are read off a row (star first,
#: unchanged from batch 3) so a scored record's slot order does not move when
#: the third column starts filing.
RACING_BLOCK_KEYS: dict[Predictor, str] = {
    "star": "star_prediction",
    "composed": "composed_prediction",
    "directional": "directional_prediction",
}
#: Addendum D §D1's scope trigger, as a SET rather than a hardcoded name: "any
#: ceiling-aware gauge (the Addendum-C directional star c_A^out·c_B^in, or any
#: explicitly ratified α-based refinement)". Today exactly one gauge is adopted
#: (Addendum 2026-07-29-H). A future α-based refinement joins this tuple and the
#: companion starts being owed for it too — which is the point of naming it.
CEILING_AWARE_GAUGES: tuple[Predictor, ...] = ("directional",)

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


# ------------------------------------- ADDENDUM D — the constant-α companion
#  D1's scope trigger IS Addendum H's adoption of the directional star, so from
#  the first directional filing this companion is owed at every scoring read.
#
#  WHAT THIS TOOL DOES AND DOES NOT DO. It does NOT compute ᾱ — D2 defines ᾱ as
#  the grand mean of the fitted per-model (or per-role) α values ACROSS THE
#  SCORED SET, which is a desk quantity formed once the scored set exists, and a
#  tool that auto-computed it would silently pick its own scored set. It does
#  NOT compute the ceiling/calibration terms — D2 says they are "the same terms
#  the adopted gauge uses", which is the derivation lane's object. It does NOT
#  run the α-permutation null (D3, n=1000).
#
#  It scores what the desk supplies, on structurally identical terms, and when
#  nothing is supplied it NAMES THE COMPANION AS OWED with D2/D3's frozen text.
#  Absent and malformed are different states and never collapse into each other.
#
#  THE SCHEMA OF RECORD — `constant-alpha-companion/v1`
#    {
#      "schema": "constant-alpha-companion/v1",       REQUIRED, exact string
#      "gauge": "directional",                        REQUIRED, a Predictor name
#      "alpha_bar": 0.7213,                           REQUIRED, finite
#      "alpha_bar_provenance": "<free text>",         REQUIRED
#      "corpus_manifest_sha256": "<64 hex>",          optional
#      "note": "<free text>",                         optional
#      "rows": [                                      REQUIRED, non-empty
#        {"source": "...", "target": "...", "arm": "native",
#         "family": "proc_k128",
#         "predicted": 0.3104,                        REQUIRED, finite
#         "ceiling_term": 0.4304,                     optional descriptive
#         "note": ""}                                 optional
#      ],
#      "alpha_permutation_null": {                    optional; absent => OWED
#         "n_permutations": 1000,
#         "true_in_band_count": 12,
#         "percentile_rank": 99.4}
#    }
#
#  `predicted` is ᾱ · (the adopted gauge's ceiling/calibration terms) for that
#  slot, computed by the desk. This module bands it at the SAME frozen ±.05 and
#  applies the SAME |predicted| < .08 carve-out — D2's "against the SAME frozen
#  bands", read as the same band RULE around the baseline's own prediction
#  (which is how Addendum C item 1 applies it to the constant grand-mean
#  predictor: "the mean of all 190 observed â, applied to every pair"). That
#  reading is stated on the artifact, not left to a reader.


class AlphaPermutationNull(BaseModel):
    """D3's α-permutation null, AS SUPPLIED. Never computed here."""
    n_permutations: int
    true_in_band_count: int
    percentile_rank: float = Field(
        description="the true star's in-band count as a percentile of the "
                    "permutation distribution; D3 wants ≥ 99")
    note: str = ""


class ConstantAlphaRow(BaseModel):
    """The desk's constant-α prediction for ONE slot."""
    source: str
    target: str
    arm: str
    family: str = FAMILY_OF_RECORD
    predicted: float = Field(
        description="ᾱ · [the adopted gauge's ceiling/calibration terms]")
    ceiling_term: Optional[float] = Field(
        default=None, description="descriptive: the ceiling/calibration term "
                                  "the desk multiplied ᾱ by")
    note: str = ""

    @property
    def key(self) -> str:
        return slot_key(self.source, self.target, self.arm, self.family)


class ConstantAlphaCompanion(BaseModel):
    """A parsed, validated `constant-alpha-companion/v1` artifact."""
    schema_name: str = SCHEMA_CONSTANT_ALPHA_COMPANION_V1
    path: str
    sha256: str
    gauge: Predictor = Field(
        description="which filed predictor column this baseline companions — "
                    "D1/D3's 'the adopted gauge'")
    alpha_bar: float
    alpha_bar_provenance: str
    corpus_manifest_sha256: Optional[str] = None
    note: str = ""
    rows: list[ConstantAlphaRow] = []
    alpha_permutation_null: Optional[AlphaPermutationNull] = None


class ConstantAlphaVerdict(BaseModel):
    """The constant-α baseline's verdict for ONE slot. A BASELINE, not a filing.

    Deliberately NOT a `ScoredSlot` and deliberately NOT a member of
    `Predictor`: nothing filed a band for it before the fit, so it can never
    enter `aggregates`, `head_to_head` or any gate denominator. It rides on the
    scored slot of the gauge it companions and is read only by D3's margin.
    """
    STATUS: str = (
        "ADDENDUM-D BASELINE — a companion, never a filed prediction. Scored "
        "against the SAME frozen ±.05 rule and the SAME |predicted| < .08 "
        "carve-out as the adopted gauge (D2: 'against the SAME frozen bands'), "
        "but no band for it was frozen before the fit, so it enters no gate and "
        "no in-band fraction that a gate reads.")
    alpha_bar: float
    predicted: float
    band: list[float]
    scored_band: list[float]
    magnitude_only: bool
    ceiling_term: Optional[float] = None
    observed: Optional[float] = None
    scored_value: Optional[float] = None
    verdict: Verdict
    in_band: Optional[bool] = None
    error: Optional[float] = None
    abs_error: Optional[float] = None
    band_excess: Optional[float] = None
    companion_source: str
    companion_sha256: str


class AddendumDCompanion(BaseModel):
    """The record-level Addendum-D read: SCORED, OWED, or NOT-TRIGGERED."""
    STATUS: Literal["SCORED", "OWED", "NOT-TRIGGERED"]
    scope_trigger: str = (
        "Addendum D §D1: this companion activates if and when any ceiling-aware "
        "gauge (the Addendum-C directional star c_A^out·c_B^in, or any "
        "explicitly ratified α-based refinement) is adopted for not-yet-filed "
        "predictions. Addendum 2026-07-29-H item 4 ACTIVATES it.")
    interpretive_rule: str = (
        "Addendum D §D3, frozen: the per-model-structure claim — 'models carry "
        "intrinsic, role-specific portability beyond measurement calibration' — "
        "is DEMONSTRATED only if the adopted star beats the constant-α "
        "predictor by ≥ 15 percentage points in in-band count AND an "
        "α-permutation null (per-model α permuted across models, n = 1000) "
        "places the true star's in-band count at ≥ the 99th percentile. "
        "Passing G-star-hit while failing this companion is scored "
        "'calibration-driven; per-model structure not demonstrated.'")
    band_reading: str = (
        "D2's 'against the SAME frozen bands' is applied as the same band RULE "
        "around the baseline's own prediction (±.05 absolute, |pred| < .08 "
        "magnitude-only) — the reading Addendum C item 1 already uses for the "
        "constant grand-mean predictor. Stated here rather than assumed.")
    gauge: Optional[Predictor] = Field(
        default=None, description="the adopted gauge this companion is read "
                                  "against; None when not triggered")
    alpha_bar: Optional[float] = None
    alpha_bar_provenance: Optional[str] = None
    companion_source: Optional[str] = None
    companion_sha256: Optional[str] = None
    n_gauge_scored_of_record: int = 0
    n_gauge_in_band: int = 0
    gauge_in_band_fraction: Optional[float] = None
    n_companion_scored: int = 0
    n_companion_in_band: int = 0
    companion_in_band_fraction: Optional[float] = None
    n_slots_companion_missing: int = Field(
        default=0, description="gauge slots scored of record for which the "
                               "companion artifact supplies no row — named, "
                               "never dropped from the gauge's denominator")
    margin_percentage_points: Optional[float] = Field(
        default=None,
        description="100 × (gauge in-band fraction − companion in-band "
                    "fraction), over slots where BOTH are scored. D3 wants "
                    "≥ 15. Computed, never adjudicated: the desk rules")
    margin_meets_d3_threshold: Optional[bool] = None
    alpha_permutation_null: Optional[AlphaPermutationNull] = None
    permutation_rank_meets_d3_threshold: Optional[bool] = None
    owed: list[str] = Field(
        default=[], description="exactly what is still owed before D3 can be "
                                "read at all — never silently absent")


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
    constant_alpha: Optional[ConstantAlphaVerdict] = Field(
        default=None,
        description="Addendum D §D2's constant-α baseline verdict for this "
                    "slot. Present ONLY on slots of the adopted gauge and ONLY "
                    "when the desk supplied a constant-α companion; NEVER "
                    "auto-computed (ᾱ is a grand mean over the scored set and "
                    "the ceiling terms are the gauge's). None here means the "
                    "companion was not supplied for this slot — the record's "
                    "`addendum_d` block names what is owed")
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
    """Addendum E §E3's per-pair 2×2, over slots where BOTH are scored of record.

    PRESERVED VERBATIM across the Addendum-H adoption: same field names, same
    denominators, same star-vs-composed semantics. Addendum H item 1 makes the
    directional star model selection over not-yet-filed slots, not a patch, so
    the E3 race's own record must keep reading the same way in every scored
    record before and after the adoption. The three-predictor reads are
    `PairwiseHeadToHead` and `ThreeWayHeadToHead`, beside this — never instead.
    """
    n_slots_both_scored: int = 0
    n_both_in_band: int = 0
    n_star_only: int = 0
    n_composed_only: int = 0
    n_both_out_of_band: int = 0
    n_slots_incomplete: int = Field(
        default=0, description="slots where at least one predictor is not "
                               "scored of record — excluded from the 2×2")


class PairwiseHeadToHead(BaseModel):
    """The E3 2×2 for ONE ordered predictor pair, over slots both scored."""
    predictor_a: Predictor
    predictor_b: Predictor
    n_slots_both_scored: int = 0
    n_both_in_band: int = 0
    n_a_only: int = Field(default=0, description="predictor_a hit, b missed")
    n_b_only: int = Field(default=0, description="predictor_b hit, a missed")
    n_both_out_of_band: int = 0
    n_slots_incomplete: int = Field(
        default=0, description="slots where at least one of the two is not "
                               "scored of record — excluded from this 2×2")


class ThreeWayHeadToHead(BaseModel):
    """The per-slot hit census across ALL THREE predictors.

    Addendum E §E3's 2×2 generalizes to a 2³ once three columns file: the
    natural object is the SET of predictors that hit on each slot. Reported as
    counts keyed by that set (`"none"`, `"star"`, `"star+composed"`, …) over
    slots where all three are scored of record, so no slot is counted under a
    denominator some predictor never entered.
    """
    n_slots_all_scored: int = 0
    n_slots_incomplete: int = Field(
        default=0, description="slots where fewer than all three predictors "
                               "are scored of record")
    predictors: list[Predictor] = []
    n_by_hit_set: dict[str, int] = Field(
        default={},
        description="hit-set label -> count; label is the '+'-joined predictor "
                    "names in PREDICTOR order, or 'none'. Keys with count 0 "
                    "are emitted too, so the census is a complete partition")


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
    head_to_head: HeadToHead = Field(
        default=HeadToHead(),
        description="Addendum E §E3's star-vs-composed 2×2, PRESERVED VERBATIM "
                    "across the Addendum-H adoption — same field, same names, "
                    "same denominators, so the E3 race reads identically in "
                    "records filed before and after the third column")
    head_to_head_pairwise: list[PairwiseHeadToHead] = Field(
        default=[],
        description="the same 2×2 for every predictor pair (star×composed, "
                    "star×directional, composed×directional) — Addendum H item "
                    "3's 'scored on structurally identical terms', beside E3's "
                    "read and never in place of it")
    head_to_head_three_way: ThreeWayHeadToHead = Field(
        default=ThreeWayHeadToHead(),
        description="the per-slot hit-set census across all three predictors, "
                    "over slots where all three are scored of record")
    addendum_d: AddendumDCompanion = Field(
        default=AddendumDCompanion(STATUS="NOT-TRIGGERED"),
        description="Addendum D's constant-α companion: SCORED when the desk "
                    "supplied one, OWED when a directional column is filed and "
                    "it was not, NOT-TRIGGERED when no ceiling-aware gauge is "
                    "filed in this record")
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


def _scored_band_for(predicted: float, band: Sequence[float],
                     magnitude_only: bool) -> list[float]:
    """The band the comparison actually runs against.

    Magnitude-only moves the comparison onto |·| (prereg §3: "|observed| within
    the band of |predicted|; sign unscored") and the band travels with it — the
    FROZEN half-width never changes. Otherwise the band is the filed one,
    verbatim. ONE implementation, shared by the filed predictors and by
    Addendum D's companion baseline, so "structurally identical terms" is a
    property of the code and not of two copies agreeing today.
    """
    if magnitude_only:
        return [abs(predicted) - FROZEN_BAND_HALF_WIDTH,
                abs(predicted) + FROZEN_BAND_HALF_WIDTH]
    return [float(band[0]), float(band[1])]


def _verdict_for(value: float, scored_band: Sequence[float],
                 magnitude_only: bool) -> tuple[Verdict, bool, float]:
    """(verdict, in_band, band_excess) for one value against one scored band."""
    in_band, where, excess = _band_position(value, float(scored_band[0]),
                                            float(scored_band[1]))
    if magnitude_only:
        verdict: Verdict = ("magnitude-only-in-band" if in_band
                            else f"magnitude-only-out-of-band-{where}")  # type: ignore[assignment]
    else:
        verdict = "in-band" if in_band else f"out-of-band-{where}"  # type: ignore[assignment]
    return verdict, in_band, excess


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
    scored_band = _scored_band_for(filed.predicted, band, filed.magnitude_only)

    if observed is None:
        return ScoredSlot(scored_band=scored_band,
                          verdict="UNSCORED-NO-OBSERVATION", in_band=None,
                          scored_of_record=False,
                          flags=flags + ["NO OBSERVED â SUPPLIED for this slot — "
                                         "reported, never dropped from the "
                                         "denominator"],
                          **common)                     # type: ignore[arg-type]

    value = abs(observed.a_hat) if filed.magnitude_only else observed.a_hat
    verdict, in_band, excess = _verdict_for(value, scored_band,
                                            filed.magnitude_only)

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

      * RACING (batch 3 onward): a row carries `source`/`target` plus any of a
        `star_prediction`, `composed_prediction` and — from Addendum H,
        batch 4 onward — `directional_prediction` block. Every block present is
        read; a row filing only one of the three is legal (the columns' filing
        availability is independent) and a row filing none is a HALT.
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
        if any(k in row for k in RACING_BLOCK_KEYS.values()):
            source, target = str(row["source"]), str(row["target"])
            arm, family = str(row["arm"]), str(row["family"])
            for predictor, block_key in RACING_BLOCK_KEYS.items():
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
                f"{sorted(row)}). Known shapes: racing (any of "
                f"{sorted(RACING_BLOCK_KEYS.values())} blocks) and scalar-flat "
                f"(`predicted_a_hat`). A further shape must be taught to this "
                f"parser explicitly — never guessed")
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


def _alpha_companion_halt(path: Path, problem: str) -> AlphaCompanionError:
    """The contract-quoting halt for the constant-α companion artifact."""
    return AlphaCompanionError(
        f"{path}: {problem}\n"
        f"EXPECTED `{SCHEMA_CONSTANT_ALPHA_COMPANION_V1}`:\n"
        f"  top level: object with REQUIRED `schema` == "
        f"{SCHEMA_CONSTANT_ALPHA_COMPANION_V1!r}, REQUIRED `gauge` (one of "
        f"{sorted(RACING_BLOCK_KEYS)} — the adopted gauge this baseline "
        f"companions), REQUIRED finite `alpha_bar`, REQUIRED "
        f"`alpha_bar_provenance` (free text), REQUIRED non-empty `rows`; "
        f"OPTIONAL `corpus_manifest_sha256`, `note`, `alpha_permutation_null` "
        f"({{n_permutations, true_in_band_count, percentile_rank}}).\n"
        f"  each row : REQUIRED `source`, `target`, `arm`, finite `predicted` "
        f"(= ᾱ · the adopted gauge's ceiling/calibration terms for that slot); "
        f"OPTIONAL `family` (default {FAMILY_OF_RECORD!r}), `ceiling_term`, "
        f"`note`. (source, target, arm, family) must be UNIQUE.\n"
        f"This tool NEVER computes ᾱ, the ceiling terms, or the α-permutation "
        f"null: Addendum D §D2 defines ᾱ over the scored set and §D3's null is "
        f"a desk computation. Supplying nothing is legal and is reported as "
        f"OWED — supplying something malformed is not.")


def load_alpha_companion(path: Path) -> ConstantAlphaCompanion:
    """Read and VALIDATE a `constant-alpha-companion/v1` artifact."""
    if not path.is_file():
        raise _alpha_companion_halt(
            path, "constant-α companion artifact absent. (Passing no "
                  "--alpha-companion at all is the legal way to leave it OWED; "
                  "naming a file that does not exist is not)")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise _alpha_companion_halt(path, f"unreadable as JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise _alpha_companion_halt(
            path, f"top level is {type(doc).__name__}, not an object")
    declared = doc.get("schema")
    if declared != SCHEMA_CONSTANT_ALPHA_COMPANION_V1:
        raise _alpha_companion_halt(
            path, f"declares schema {declared!r}, not "
                  f"{SCHEMA_CONSTANT_ALPHA_COMPANION_V1!r}")
    gauge = doc.get("gauge")
    if gauge not in RACING_BLOCK_KEYS:
        raise _alpha_companion_halt(
            path, f"`gauge` is {gauge!r}, not one of {sorted(RACING_BLOCK_KEYS)}")
    try:
        alpha_bar = float(doc["alpha_bar"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _alpha_companion_halt(
            path, f"`alpha_bar` is {doc.get('alpha_bar')!r}, which is not a "
                  f"number") from exc
    if not np.isfinite(alpha_bar):
        raise _alpha_companion_halt(
            path, f"`alpha_bar` is {alpha_bar!r} — non-finite")
    provenance = doc.get("alpha_bar_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise _alpha_companion_halt(
            path, f"`alpha_bar_provenance` is {provenance!r}; ᾱ is a grand mean "
                  f"over a SCORED SET, and which set it was formed over is the "
                  f"only thing that makes D3's margin auditable")
    raw_rows = doc.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise _alpha_companion_halt(
            path, f"`rows` is {type(raw_rows).__name__ if raw_rows is not None else 'absent'}"
                  f", not a non-empty list")

    rows: list[ConstantAlphaRow] = []
    seen: dict[str, int] = {}
    for i, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            raise _alpha_companion_halt(
                path, f"`rows[{i}]` is {type(entry).__name__}, not an object")
        try:
            predicted = float(entry["predicted"])
            row = ConstantAlphaRow(
                source=str(entry["source"]), target=str(entry["target"]),
                arm=str(entry["arm"]),
                family=str(entry.get("family", FAMILY_OF_RECORD)),
                predicted=predicted,
                ceiling_term=(None if entry.get("ceiling_term") is None
                              else float(entry["ceiling_term"])),
                note=str(entry.get("note", "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise _alpha_companion_halt(
                path, f"malformed `rows[{i}]` {entry!r}: {exc}") from exc
        if not np.isfinite(row.predicted):
            raise _alpha_companion_halt(
                path, f"`rows[{i}]` ({row.key}): `predicted` is non-finite")
        if row.key in seen:
            raise _alpha_companion_halt(
                path, f"duplicate companion row for {row.key} (rows "
                      f"{seen[row.key]} and {i}) — which baseline value is of "
                      f"record cannot be guessed")
        seen[row.key] = i
        rows.append(row)

    null_doc = doc.get("alpha_permutation_null")
    null: Optional[AlphaPermutationNull] = None
    if null_doc is not None:
        if not isinstance(null_doc, dict):
            raise _alpha_companion_halt(
                path, f"`alpha_permutation_null` is {type(null_doc).__name__}, "
                      f"not an object")
        try:
            null = AlphaPermutationNull(
                n_permutations=int(null_doc["n_permutations"]),
                true_in_band_count=int(null_doc["true_in_band_count"]),
                percentile_rank=float(null_doc["percentile_rank"]),
                note=str(null_doc.get("note", "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise _alpha_companion_halt(
                path, f"malformed `alpha_permutation_null` {null_doc!r}: "
                      f"{exc}") from exc

    logger.info("constant-α companion: gauge %s, ᾱ %.6f, %d row(s), "
                "permutation null %s (%s)", gauge, alpha_bar, len(rows),
                "supplied" if null else "OWED",
                SCHEMA_CONSTANT_ALPHA_COMPANION_V1)
    return ConstantAlphaCompanion(
        path=str(path), sha256=sha256_of(path),
        gauge=gauge,                                 # type: ignore[arg-type]
        alpha_bar=alpha_bar, alpha_bar_provenance=provenance,
        corpus_manifest_sha256=doc.get("corpus_manifest_sha256"),
        note=str(doc.get("note", "")), rows=rows, alpha_permutation_null=null)


def score_constant_alpha(row: ConstantAlphaRow, companion: ConstantAlphaCompanion,
                         observed: Optional[ObservedAhat]) -> ConstantAlphaVerdict:
    """Score the constant-α baseline for one slot. Same band rule, no gate."""
    magnitude_only = bool(abs(row.predicted) < NEAR_ZERO_CARVE_OUT)
    band = _frozen_band(row.predicted)
    scored_band = _scored_band_for(row.predicted, band, magnitude_only)
    common = dict(alpha_bar=companion.alpha_bar, predicted=row.predicted,
                  band=band, scored_band=scored_band,
                  magnitude_only=magnitude_only, ceiling_term=row.ceiling_term,
                  companion_source=companion.path,
                  companion_sha256=companion.sha256)
    if observed is None:
        return ConstantAlphaVerdict(
            verdict="UNSCORED-NO-OBSERVATION", in_band=None,
            **common)                                # type: ignore[arg-type]
    value = abs(observed.a_hat) if magnitude_only else observed.a_hat
    verdict, in_band, excess = _verdict_for(value, scored_band, magnitude_only)
    return ConstantAlphaVerdict(
        observed=observed.a_hat, scored_value=value, verdict=verdict,
        in_band=in_band, error=observed.a_hat - row.predicted,
        abs_error=abs(observed.a_hat - row.predicted), band_excess=excess,
        **common)                                    # type: ignore[arg-type]


def attach_constant_alpha(slots: Sequence[ScoredSlot],
                          companion: Optional[ConstantAlphaCompanion],
                          observations: dict[str, ObservedAhat],
                          ) -> AddendumDCompanion:
    """Attach D2's baseline to the adopted gauge's slots and read D3's margin.

    Mutates the gauge's `ScoredSlot.constant_alpha` in place and returns the
    record-level block. Three outcomes, never conflated:

      * NOT-TRIGGERED — no ceiling-aware gauge is filed in this record at all;
      * OWED          — a gauge IS filed and no companion was supplied, or one
                        was supplied without the α-permutation null;
      * SCORED        — the margin is computed. Computed, not adjudicated: D3's
                        thresholds are reported as booleans beside the numbers
                        and the desk rules.
    """
    filed_gauges = {s.predictor for s in slots}
    triggered = [g for g in CEILING_AWARE_GAUGES if g in filed_gauges]
    if companion is None and not triggered:
        return AddendumDCompanion(STATUS="NOT-TRIGGERED", gauge=None)
    gauge: Predictor = companion.gauge if companion else triggered[0]
    if companion is not None and gauge not in filed_gauges:
        raise AlphaCompanionError(
            f"{companion.path}: companions the {companion.gauge!r} gauge, but "
            f"this record files no {companion.gauge!r} prediction (it files "
            f"{sorted(filed_gauges)}). Scoring a baseline against a column "
            f"nobody filed would produce a margin with no gauge on the other "
            f"side of it")

    gauge_slots = [s for s in slots if s.predictor == gauge]
    n_gauge_scored = sum(1 for s in gauge_slots if s.scored_of_record)
    n_gauge_in_band = sum(1 for s in gauge_slots
                          if s.scored_of_record and s.in_band)

    if companion is None:
        return AddendumDCompanion(
            STATUS="OWED", gauge=gauge,
            n_gauge_scored_of_record=n_gauge_scored,
            n_gauge_in_band=n_gauge_in_band,
            gauge_in_band_fraction=(n_gauge_in_band / n_gauge_scored
                                    if n_gauge_scored else None),
            owed=[f"the constant-α companion for the {gauge!r} column "
                  f"(Addendum D §D2): ᾱ = the grand mean of the fitted "
                  f"per-model/per-role α over the scored set, and ᾱ · [the "
                  f"same ceiling/calibration terms the adopted gauge uses] per "
                  f"slot. NEVER auto-computed here — supply it as "
                  f"`{SCHEMA_CONSTANT_ALPHA_COMPANION_V1}` via "
                  f"--alpha-companion.",
                  "the α-permutation null (Addendum D §D3): per-model α "
                  "assignments permuted across models, n = 1000, reporting the "
                  "true star's in-band count as a percentile.",
                  "until both land, D3's per-model-structure claim cannot be "
                  "read at all — neither as demonstrated nor as refuted."])

    by_key = {row.key: row for row in companion.rows}
    n_missing = 0
    n_companion_scored = 0
    n_companion_in_band = 0
    n_both = 0
    n_gauge_in_band_both = 0
    for slot in gauge_slots:
        row = by_key.get(slot.key)
        if row is None:
            n_missing += 1
            continue
        verdict = score_constant_alpha(row, companion,
                                       observations.get(slot.key))
        slot.constant_alpha = verdict
        if not slot.scored_of_record or verdict.in_band is None:
            continue
        n_companion_scored += 1
        n_both += 1
        if verdict.in_band:
            n_companion_in_band += 1
        if slot.in_band:
            n_gauge_in_band_both += 1

    unmatched = sorted(set(by_key) - {s.key for s in gauge_slots})
    if unmatched:
        raise AlphaCompanionError(
            f"{companion.path}: {len(unmatched)} companion row(s) match no "
            f"filed {gauge!r} slot: {unmatched}. A baseline row for a slot the "
            f"record never filed is a typo, a wrong record or a direction flip "
            f"— every one of which yields a margin that looks complete and is "
            f"not")

    margin: Optional[float] = None
    if n_both:
        margin = 100.0 * ((n_gauge_in_band_both / n_both)
                          - (n_companion_in_band / n_both))
    null = companion.alpha_permutation_null
    owed: list[str] = []
    if n_missing:
        owed.append(
            f"{n_missing} scored {gauge!r} slot(s) have no companion row — "
            f"D3's margin is read over the {n_both} slot(s) where BOTH are "
            f"scored, and the shortfall is named rather than absorbed")
    if null is None:
        owed.append(
            "the α-permutation null (Addendum D §D3): per-model α assignments "
            "permuted across models, n = 1000, true star's in-band count at "
            "≥ the 99th percentile. NOT computed here. Until it lands the "
            "margin alone cannot demonstrate per-model structure")
    return AddendumDCompanion(
        STATUS="SCORED", gauge=gauge, alpha_bar=companion.alpha_bar,
        alpha_bar_provenance=companion.alpha_bar_provenance,
        companion_source=companion.path, companion_sha256=companion.sha256,
        n_gauge_scored_of_record=n_gauge_scored,
        n_gauge_in_band=n_gauge_in_band,
        gauge_in_band_fraction=(n_gauge_in_band / n_gauge_scored
                                if n_gauge_scored else None),
        n_companion_scored=n_companion_scored,
        n_companion_in_band=n_companion_in_band,
        companion_in_band_fraction=(n_companion_in_band / n_companion_scored
                                    if n_companion_scored else None),
        n_slots_companion_missing=n_missing,
        margin_percentage_points=margin,
        margin_meets_d3_threshold=(None if margin is None else margin >= 15.0),
        alpha_permutation_null=null,
        permutation_rank_meets_d3_threshold=(
            None if null is None else null.percentile_rank >= 99.0),
        owed=owed)


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


def _slots_by_key(slots: Sequence[ScoredSlot]) -> dict[str, dict[str, ScoredSlot]]:
    """slot key -> {predictor: ScoredSlot}."""
    by_key: dict[str, dict[str, ScoredSlot]] = {}
    for s in slots:
        by_key.setdefault(s.key, {})[s.predictor] = s
    return by_key


def head_to_head_pairwise(slots: Sequence[ScoredSlot]) -> list[PairwiseHeadToHead]:
    """The E3 2×2 for EVERY predictor pair, in `PREDICTOR_PAIRS` order.

    The `("star", "composed")` entry reproduces `head_to_head()` exactly — same
    slots, same denominators — which selftest 18 proves rather than asserts, so
    the E3 read has one arithmetic and two presentations, never two arithmetics.
    A pair neither of whose predictors is filed in the record yields a row of
    zeros: an empty 2×2 is a fact about the record, not a reason to omit it.
    """
    by_key = _slots_by_key(slots)
    out: list[PairwiseHeadToHead] = []
    for predictor_a, predictor_b in PREDICTOR_PAIRS:
        row = PairwiseHeadToHead(predictor_a=predictor_a, predictor_b=predictor_b)
        for pair in by_key.values():
            a, b = pair.get(predictor_a), pair.get(predictor_b)
            if (a is None or b is None or not a.scored_of_record
                    or not b.scored_of_record):
                row.n_slots_incomplete += 1
                continue
            row.n_slots_both_scored += 1
            if a.in_band and b.in_band:
                row.n_both_in_band += 1
            elif a.in_band:
                row.n_a_only += 1
            elif b.in_band:
                row.n_b_only += 1
            else:
                row.n_both_out_of_band += 1
        out.append(row)
    return out


def head_to_head_three_way(slots: Sequence[ScoredSlot]) -> ThreeWayHeadToHead:
    """The per-slot hit-set census over slots where ALL THREE columns scored."""
    predictors: list[Predictor] = list(RACING_BLOCK_KEYS)
    #  Enumerate every subset in PREDICTOR order, so the census is a COMPLETE
    #  partition of the all-scored slots and a zero cell is visible as a zero
    #  rather than as a missing key.
    labels: list[str] = ["none"]
    for mask in range(1, 1 << len(predictors)):
        labels.append("+".join(p for i, p in enumerate(predictors)
                               if mask & (1 << i)))
    census = ThreeWayHeadToHead(predictors=predictors,
                                n_by_hit_set={label: 0 for label in labels})
    for pair in _slots_by_key(slots).values():
        present = [pair.get(p) for p in predictors]
        if any(s is None or not s.scored_of_record for s in present):
            census.n_slots_incomplete += 1
            continue
        census.n_slots_all_scored += 1
        hits = [p for p, s in zip(predictors, present)
                if s is not None and s.in_band]
        census.n_by_hit_set["+".join(hits) if hits else "none"] += 1
    return census


def score_record(record_path: Path, observations_path: Path,
                 scored_utc: Optional[str] = None,
                 alpha_companion_path: Optional[Path] = None) -> ScoredRecord:
    """Build the scored record for one filing record. WRITES NOTHING.

    Unmatched observations are a HALT: an observation that matches no filed slot
    is a typo, a wrong record, or a direction flip, and every one of those
    produces a scoring artifact that looks complete and is not.
    """
    stamp = scored_utc or utc_now()
    filing = parse_filing_record(record_path)
    observed = load_observations(observations_path)
    companion = (None if alpha_companion_path is None
                 else load_alpha_companion(alpha_companion_path))
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

    #  Addendum D §D1's scope trigger is Addendum H's adoption, so this runs at
    #  every scoring read once a directional column is filed. It attaches the
    #  baseline to the gauge's slots and NEVER computes ᾱ or the null itself.
    addendum_d = attach_constant_alpha(scored, companion, by_key)

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
    if addendum_d.STATUS == "OWED":
        warnings.append(
            "ADDENDUM D OWED — a ceiling-aware gauge is filed in this record, "
            "so §D1's scope trigger has fired and the constant-α companion is "
            "part of the read: " + " · ".join(addendum_d.owed))
    elif addendum_d.owed:
        warnings.append("ADDENDUM D PARTIAL — " + " · ".join(addendum_d.owed))

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
        head_to_head=head_to_head(scored),
        head_to_head_pairwise=head_to_head_pairwise(scored),
        head_to_head_three_way=head_to_head_three_way(scored),
        addendum_d=addendum_d, warnings=warnings)
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
    check(site_of_record("gemma3-27b") == 38,
          "gemma3-27b resolves at its RULED site L38 (Luxia 2026-07-29, six-site "
          "â evidence table `7f59af50…`; L41 robustness, the carried-provisional "
          "L36 retired)")
    try:
        compose_pair("gemma3-27b", "phi-4", source_site=36)
        check(False, "gemma's retired L36 must refuse")
    except FitGridError as exc:
        check("NOT in its fixed fit grid" in str(exc),
              f"gemma's retired L36 refuses: {exc!s:.60}")
    check(site_of_record("llama-3.1-70b-instruct") == 37,
          "70B site of record is L37 (L43 robustness, L17 retired)")
    #  No retired-site refusal check rides with the 3.3-70B: this node was
    #  SCAN-REGISTRY ONLY before the ruling, so it has no prior registration to
    #  retire — unlike gemma's L36 and the 3.1-70B's L17 above.
    check(site_of_record("llama-3.3-70b-instruct") == 58,
          "llama-3.3-70b-instruct resolves at its RULED site L58 (desk "
          "2026-07-29 under Luxia's overnight delegation 2, five-site â "
          "evidence table `e6d584aa…`; L63 robustness). Its 3.1 vintage-mate "
          "rules to L37 from its own table — independent registrations")
    #  No retired-site refusal check rides with the 405B either, and for the
    #  same reason as the 3.3-70B directly above: it was SCAN-REGISTRY ONLY
    #  before the ruling, so there is no prior registration to retire. The
    #  refusal checks exist for gemma's L36 and the 3.1-70B's L17, which were
    #  registered and then MOVED; a first registration has nothing to refuse.
    check(site_of_record("llama-3.1-405b-instruct") == 99,
          "llama-3.1-405b-instruct resolves at its RULED site L99 (desk "
          "2026-07-29 under Luxia's overnight delegation 2, five-site â "
          "evidence table `04f2a2c4…`; L107 robustness). Unanimous rank-1 in "
          "all four robustness columns; the r² peak L43 ranks 3rd/4th on â and "
          "goes sub-null on the rebuilt-L16 hub — never re-derive this from r²")
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

    print("== selftest 16: the chart decomposition IS the predictor (identity) ==")
    #  â_comp = alpha_pair · ceil_B is an IDENTITY over a common hub frame, so
    #  the descriptive companions are a decomposition of the filed number and
    #  never a second one. Proved twice: closed-form on synthetic maps (always
    #  on), and on the FIVE ARCHIVED GATE PAIRS (free corroboration that the
    #  decomposition matches the operationalization of record bit-for-bit).
    shared_va = np.linalg.qr(rng.standard_normal((d_hub, k)))[0].T
    def _shared_map(d_model: int, scale: float, src_n: float, tgt_n: float
                    ) -> TransportMap:
        return TransportMap(
            kind="proc", src_norm=src_n, tgt_norm=tgt_n, va=shared_va,
            vb=np.linalg.qr(rng.standard_normal((d_model, k)))[0].T,
            omega=np.linalg.qr(rng.standard_normal((k, k)))[0], scale=scale)

    m_a = _shared_map(d_a, 1.1541, 5.6915, 2.2506)
    m_b = _shared_map(d_b, 0.8375, 5.6915, 3.9012)
    z_a = hub_frame_coordinates(m_a, v_a)
    z_b = hub_frame_coordinates(m_b, v_b)
    check(abs(float(np.linalg.norm(z_b))
              - projection_norm(image_basis(m_b), v_b)) < 1e-12,
          "‖zeta_B‖ == the â-ceiling of hub→B at v_B (the hub-frame norm IS "
          "the ceiling already reported beside the prediction)")
    synth_comp = composed_exchange_rate(m_a, m_b, v_a, v_b)
    synth_identity = float(cos(z_a, z_b)) * float(np.linalg.norm(z_b))
    check(abs(synth_comp - synth_identity) < 1e-12,
          f"â_comp {synth_comp:+.15f} == alpha_pair·ceil_B {synth_identity:+.15f} "
          f"on synthetic maps sharing one hub frame "
          f"(|Δ| {abs(synth_comp - synth_identity):.3e})")
    synth_block = chart_companions(m_a, m_b, v_a, v_b, synth_comp,
                                   hub_vector_path=Path("/nonexistent/hub.npz"))
    check(synth_block.identity_holds is True
          and synth_block.alpha_pair is not None
          and synth_block.alpha_source is None
          and any("not present" in n for n in synth_block.notes),
          "with the hub's own vector absent, alpha_pair and the identity still "
          "compute and the alpha_* endpoint terms are reported ABSENT with a "
          "named note — the prediction never depends on this block (rake M19)")
    off_frame = _proc_map(rng, d_hub, d_b, k, 1.0, 1.0, 1.0)   # its own va
    off_block = chart_companions(m_a, off_frame, v_a, v_b, 0.0,
                                 hub_vector_path=Path("/nonexistent/hub.npz"))
    check(off_block.alpha_pair is None
          and any("common hub frame" in n for n in off_block.notes),
          "two maps NOT fit over one hub frame produce no companions at all — "
          "the identity does not hold across frames, so no number is emitted")
    ridge = TransportMap(kind="ridge", src_norm=1.0, tgt_norm=1.0,
                         left=np.eye(d_hub, d_a), right=np.eye(d_a, d_b))
    ridge_block = chart_companions(ridge, m_b, v_a, v_b, 0.0)
    check(ridge_block.alpha_pair is None and ridge_block.notes,
          f"a ridge map has no hub-frame coordinates and degrades with a note "
          f"instead of raising: {ridge_block.notes[0][:60]}")

    if not ARCHIVE_GLUE.exists():
        print(f"  [SKIP] the archive is absent at {ARCHIVE_GLUE} — the "
              f"archived-pair leg of this check needs the data tree and is "
              f"NODE-OWED here, not silently passed")
    else:
        archived_pairs = load_archived_glue(ARCHIVE_GLUE)
        worst_identity = 0.0
        n_checked = 0
        problems: list[str] = []
        for spec in archived_pairs.PAIRS:
            got = compose_pair(spec.src, spec.tgt, family=archived_pairs.FAM)
            if isinstance(got, NotFilable):
                problems.append(f"{spec.src}->{spec.tgt}: N/A-AT-FILING")
                continue
            block = got.descriptive_companions
            if block is None or block.identity_abs_delta is None:
                problems.append(f"{spec.src}->{spec.tgt}: no companions emitted")
                continue
            n_checked += 1
            worst_identity = max(worst_identity, block.identity_abs_delta)
            if not block.identity_holds:
                problems.append(
                    f"{spec.src}->{spec.tgt}: |Δ| {block.identity_abs_delta:.3e}")
            print(f"  {spec.src}L{spec.src_site}->{spec.tgt}L{spec.tgt_site:<4} "
                  f"â_comp {got.a_comp:+.9f} = alpha_pair "
                  f"{block.alpha_pair:+.9f} × ceil_B {block.ceiling_target:.9f} "
                  f"→ |Δ| {block.identity_abs_delta:.3e}"
                  + (f"  rho {block.rho_offaxis:+.4f}"
                     if block.rho_offaxis is not None else ""))
        check(n_checked == len(archived_pairs.PAIRS) and not problems,
              f"the decomposition reproduces â_comp on all "
              f"{len(archived_pairs.PAIRS)} archived gate pairs (worst |Δ| "
              f"{worst_identity:.3e}, tol {COMPANION_IDENTITY_TOLERANCE:.0e})"
              + ("" if not problems else f" — {problems}"))
        check(worst_identity <= COMPANION_IDENTITY_TOLERANCE,
              f"worst identity |Δ| over the archived pairs "
              f"{worst_identity:.3e} <= {COMPANION_IDENTITY_TOLERANCE:.0e}")

    check(not any(f in DescriptiveCompanions.model_fields
                  for f in ("band", "verdict", "predicted", "in_band",
                            "magnitude_only")),
          "the companion block carries no band, no verdict and no predicted "
          "value — it cannot be mistaken for, or read as, a prediction")
    check("descriptive_companions" not in ScoredSlot.model_fields,
          "and nothing in the SCORED record reads it — the companions are "
          "never scored, by construction and not by convention")

    print("== selftest 17: the directional-constants schema, and its refusals ==")
    #  ADDENDUM 2026-07-29-H. The readout is produced by a DIFFERENT lane, so
    #  every departure from the named contract must halt with a message that
    #  states the contract — that is how the two lanes reconcile without a
    #  round trip through a person's memory.
    import tempfile as _tempfile

    _SHA_A = "a" * 64
    _SHA_B = "b" * 64

    def _constants_doc(**overrides: Any) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SCHEMA_DIRECTIONAL_CONSTANTS_V1,
            "corpus_manifest_sha256": CORPUS_SHA_V21,
            "gauge": "in-lineage anchor c_8B=.8375 (synthetic selftest fixture)",
            "generated": "2026-07-29",
            "arm": "native", "family": FAMILY_OF_RECORD,
            "constants": [
                {"model": "phi-4", "c_out": 0.6000, "c_in": 0.5000, "site": 19},
                {"model": "qwen2.5-32b-instruct", "c_out": 0.8000,
                 "c_in": 0.7000},
                {"model": "qwen2.5-3b-instruct", "c_out": 0.1000,
                 "c_in": 0.1000},
            ],
        }
        doc.update(overrides)
        return doc

    with _tempfile.TemporaryDirectory(prefix="directional_schema_") as td:
        root = Path(td)

        def _write(name: str, payload: Any) -> Path:
            path = root / name
            path.write_text(payload if isinstance(payload, str)
                            else json.dumps(payload, indent=1))
            return path

        good_path = _write("constants-good.json", _constants_doc())
        loaded = load_directional_constants(good_path)
        check(loaded.schema_name == SCHEMA_DIRECTIONAL_CONSTANTS_V1
              and len(loaded.rows) == 3 and loaded.sha256 == sha256_of(good_path)
              and loaded.corpus_manifest_sha256 == CORPUS_SHA_V21,
              f"a well-formed {SCHEMA_DIRECTIONAL_CONSTANTS_V1} readout loads: "
              f"{len(loaded.rows)} rows, gauge recorded, own sha carried")
        check(loaded.row("phi-4", "native", FAMILY_OF_RECORD) is not None
              and loaded.row("phi-4", "raw", FAMILY_OF_RECORD) is None,
              "a row resolves IN ITS ARM only — the arm is never proxied "
              "(Addendum E §E1's rule, applied at scalar order by H item 3)")
        check(loaded.row("phi-4", "native", FAMILY_OF_RECORD).site == 19,  # type: ignore[union-attr]
              "the row carries the SITE REGISTRY's site, cross-checked against "
              "the readout's own claim")

        mapped = load_directional_constants(_write("constants-map.json", _constants_doc(
            constants={"phi-4": {"c_out": 0.6, "c_in": 0.5, "site": 19},
                       "qwen2.5-32b-instruct": {"c_out": 0.8, "c_in": 0.7},
                       "qwen2.5-3b-instruct": {"c_out": 0.1, "c_in": 0.1}})))
        check([(r.model, r.c_out, r.c_in) for r in mapped.rows]
              == [(r.model, r.c_out, r.c_in) for r in loaded.rows],
              "the model->row OBJECT encoding yields the same rows as the list "
              "encoding — both are named in the contract, neither is inferred")

        drifted = load_directional_constants(_write(
            "constants-drift.json", _constants_doc(constants=[
                {"model": "phi-4", "c_out": 0.6, "c_in": 0.5,
                 "c_sideways": 0.4, "vintage": "v2.1"}])))
        check(drifted.rows[0].unrecognized_keys == ["c_sideways", "vintage"]
              and any("SCHEMA DRIFT" in n for n in drifted.notes),
              "row keys the schema does not define are RECORDED and ignored, "
              "never obeyed and never dropped silently")

        big = load_directional_constants(_write("constants-big.json", _constants_doc(
            constants=[{"model": "phi-4", "c_out": 1.4, "c_in": 0.5},
                       {"model": "qwen2.5-32b-instruct", "c_out": 0.8,
                        "c_in": 0.7}])))
        big_slot = directional_pair("phi-4", "qwen2.5-32b-instruct", big)
        check(isinstance(big_slot, DirectionalPrediction)
              and any("OUT-OF-RANGE CONSTANT" in f for f in big_slot.flags),
              "|c| > 1 is FLAGGED on the slot and still files — a gauge-divided "
              "chain-break read can legitimately exceed 1, and the desk rules")

        refusals: list[tuple[str, Any]] = [
            ("a file that does not exist", None),
            ("not JSON at all", "{not json"),
            ("a top-level list", [1, 2, 3]),
            ("a readout of another schema name",
             _constants_doc(schema="portability-constants/v1")),
            ("a readout of a FUTURE version",
             _constants_doc(schema="directional-constants-readout/v2")),
            ("no schema declaration at all",
             {k: v for k, v in _constants_doc().items() if k != "schema"}),
            ("a corpus sha that is not 64 hex",
             _constants_doc(corpus_manifest_sha256="deadbeef")),
            ("no corpus sha at all",
             {k: v for k, v in _constants_doc().items()
              if k != "corpus_manifest_sha256"}),
            ("no gauge/anchor provenance", _constants_doc(gauge="   ")),
            ("a hub that is not the hub of record", _constants_doc(hub="70b")),
            ("a hub site that is not the column of record",
             _constants_doc(hub_site=14)),
            ("constants of the wrong type", _constants_doc(constants="c_out")),
            ("an empty constants table", _constants_doc(constants=[])),
            ("a constants row that is not an object",
             _constants_doc(constants=[0.6])),
            ("a row with no model", _constants_doc(constants=[{"c_out": .6, "c_in": .5}])),
            ("a row whose arm is not an arm of record",
             _constants_doc(arm=None, constants=[
                 {"model": "phi-4", "c_out": .6, "c_in": .5, "arm": "hybrid"}])),
            ("a row with no arm and no doc default",
             _constants_doc(arm=None, constants=[
                 {"model": "phi-4", "c_out": .6, "c_in": .5}])),
            ("a row whose family is not a family of record",
             _constants_doc(family=None, constants=[
                 {"model": "phi-4", "c_out": .6, "c_in": .5,
                  "family": "proc_k9000"}])),
            ("a c_out that is not a number",
             _constants_doc(constants=[{"model": "phi-4", "c_out": "point six",
                                        "c_in": .5}])),
            ("a missing c_in",
             _constants_doc(constants=[{"model": "phi-4", "c_out": .6}])),
            ("a non-finite constant",
             _constants_doc(constants=[{"model": "phi-4", "c_out": "NaN",
                                        "c_in": .5}])),
            ("a model with no site of record (gpt2-xl is DEFERRED)",
             _constants_doc(constants=[{"model": "gpt2-xl", "c_out": .6,
                                        "c_in": .5}])),
            ("a misspelled model key",
             _constants_doc(constants=[{"model": "phi4", "c_out": .6, "c_in": .5}])),
            ("a row whose site disagrees with the site registry",
             _constants_doc(constants=[{"model": "phi-4", "c_out": .6,
                                        "c_in": .5, "site": 21}])),
            ("a row-level corpus sha that is not 64 hex",
             _constants_doc(constants=[{"model": "phi-4", "c_out": .6, "c_in": .5,
                                        "corpus_manifest_sha256": "v2.1"}])),
            ("two rows for the same (model, arm, family)",
             _constants_doc(constants=[
                 {"model": "phi-4", "c_out": .6, "c_in": .5},
                 {"model": "phi-4", "c_out": .7, "c_in": .4}])),
            ("a mapping key and row `model` that disagree",
             _constants_doc(constants={"phi-4": {"model": "phi-3.5-mini-instruct",
                                                 "c_out": .6, "c_in": .5}})),
            ("a mapping whose value is not an object",
             _constants_doc(constants={"phi-4": 0.6})),
        ]
        for i, (label, payload) in enumerate(refusals):
            path = (root / "constants-absent.json" if payload is None
                    else _write(f"constants-bad-{i}.json", payload))
            try:
                load_directional_constants(path)
                check(False, f"{label} must be REFUSED")
            except DirectionalConstantsError as exc:
                check(f"EXPECTED `{SCHEMA_DIRECTIONAL_CONSTANTS_V1}`" in str(exc),
                      f"{label} → halt quoting the contract: "
                      f"{str(exc).splitlines()[0]!s:.78}")

        print("== selftest 18: directional slots — band, carve-out, arm, vintage ==")
        pred = directional_pair("phi-4", "qwen2.5-32b-instruct", loaded)
        assert isinstance(pred, DirectionalPrediction)
        check(pred.prediction_id
              == "directional-prediction/phi-4→qwen2.5-32b-instruct/native-k128",
              f"the ID is Addendum H's shape: {pred.prediction_id}")
        check(abs(pred.predicted - (0.6000 * 0.7000)) < 1e-15
              and pred.c_out_source == 0.6 and pred.c_in_target == 0.7,
              f"predicted = c_src^out · c_tgt^in = {pred.predicted:.6f} "
              f"(0.6 × 0.7); the SOURCE contributes c^out and the TARGET c^in")
        check(abs(pred.band[0] - (pred.predicted - 0.05)) < 1e-15
              and abs(pred.band[1] - (pred.predicted + 0.05)) < 1e-15,
              f"the ±.05 band is frozen at filing around the predicted value: "
              f"{pred.band}")
        check(pred.filed_predicted == round(pred.predicted, FILED_DECIMALS)
              and pred.filed_band == [round(pred.filed_predicted - 0.05,
                                            FILED_DECIMALS),
                                      round(pred.filed_predicted + 0.05,
                                            FILED_DECIMALS)],
              f"the FILED value and the FILED band are the same rounding: "
              f"{pred.filed_predicted} {pred.filed_band} — the band that files "
              f"is the band OF the value that files")
        check(_check_filed_band(pred.prediction_id, pred.filed_predicted,
                                pred.filed_band) == pred.filed_band,
              "and the filed pair survives the scorer's own frozen-band check, "
              "so a slot this tool emits can be scored by this tool")
        check(not pred.magnitude_only,
              f"|{pred.predicted:.4f}| >= {NEAR_ZERO_CARVE_OUT} → sign SCORED")
        near = directional_pair("qwen2.5-3b-instruct", "qwen2.5-3b-instruct",
                                loaded)
        assert isinstance(near, DirectionalPrediction)
        check(abs(near.predicted - 0.01) < 1e-15 and near.magnitude_only,
              f"|{near.predicted:.4f}| < {NEAR_ZERO_CARVE_OUT} → MAGNITUDE-ONLY "
              f"(prereg §3 / E2 / H item 3, the same carve-out verbatim)")
        check(near.corpus_vintage == "v2.1"
              and near.corpus_manifest_sha256 == CORPUS_SHA_V21,
              "a slot whose BOTH constants rode corpus-v2.1 carries the v2.1 "
              "sha (Addendum G §G2(a) / H item 5)")
        check(abs(pred.descriptive_reverse_predicted - (0.8 * 0.5)) < 1e-15
              and abs(pred.descriptive_asymmetry_ratio
                      - (0.6 * 0.7) / (0.8 * 0.5)) < 1e-12,
              f"the reverse value and the â(A→B)/â(B→A) ratio are reported "
              f"DESCRIPTIVELY ({pred.descriptive_asymmetry_ratio:.4f}) — the "
              f"asymmetry the symmetric star cannot express by construction")
        check(not any(f in DirectionalPrediction.model_fields
                      for f in ("observed", "verdict", "in_band", "a_hat")),
              "a DirectionalPrediction has no observed/verdict field — the "
              "filing path cannot carry a score even by accident")

        raw_only = load_directional_constants(_write(
            "constants-raw.json", _constants_doc(arm="raw", constants=[
                {"model": "phi-4", "c_out": .6, "c_in": .5},
                {"model": "qwen2.5-32b-instruct", "c_out": .8, "c_in": .7}])))
        na = directional_pair("phi-4", "qwen2.5-32b-instruct", raw_only)
        check(isinstance(na, DirectionalNotFilable)
              and len(na.missing_sides) == 2 and na.arm == "native"
              and "Never proxied from another arm" in na.reason,
              f"constants banked in the RAW arm do not back a NATIVE slot: "
              f"N/A-AT-FILING, missing {getattr(na, 'missing_sides', None)}")
        one_sided = load_directional_constants(_write(
            "constants-one.json", _constants_doc(constants=[
                {"model": "phi-4", "c_out": .6, "c_in": .5}])))
        na2 = directional_pair("phi-4", "qwen2.5-32b-instruct", one_sided)
        check(isinstance(na2, DirectionalNotFilable)
              and na2.missing_sides == ["target c^in (qwen2.5-32b-instruct/"
                                        "native-proc_k128)"]
              and na2.available_keys == ["phi-4/native-proc_k128"],
              "a slot missing ONLY the target's c^in names exactly that side, "
              "and lists what the readout does carry")
        na3 = directional_pair("qwen2.5-32b-instruct", "phi-4", one_sided)
        check(isinstance(na3, DirectionalNotFilable)
              and na3.missing_sides[0].startswith("source c^out"),
              "and the SOURCE-role miss is named as the source's c^out — the "
              "two roles are never substituted for each other")

        mixed = load_directional_constants(_write(
            "constants-mixed.json", _constants_doc(constants=[
                {"model": "phi-4", "c_out": .6, "c_in": .5,
                 "corpus_manifest_sha256": _SHA_A},
                {"model": "qwen2.5-32b-instruct", "c_out": .8, "c_in": .7,
                 "corpus_manifest_sha256": _SHA_B}])))
        mix = directional_pair("phi-4", "qwen2.5-32b-instruct", mixed)
        assert isinstance(mix, DirectionalPrediction)
        check(mix.corpus_vintage == "MIXED"
              and mix.corpus_manifest_sha256 is None
              and any("MIXED-VINTAGE" in f for f in mix.flags),
              "two constants on DIFFERENT vintages → the product is computable "
              "and carries NO sha: a tag covering half a computation is worse "
              "than none (Addendum G §G2(a))")
        check(any("MIXED-VINTAGE READOUT" in n for n in mixed.notes),
              "and the readout itself notes that its row shas disagree")

        cross_arm = directional_pair("3b", "pythia-6.9b", loaded)
        check(isinstance(cross_arm, DirectionalNotFilable)
              and cross_arm.arm == "raw"
              and "instruct↔base → raw" in cross_arm.arm_rule,
              "an instruct↔base pair resolves to the RAW arm before any "
              "constant is looked up (prereg §3), and misses there")
        try:
            directional_pair("gpt2-xl", "phi-4", loaded)
            check(False, "a model with no site of record must refuse")
        except ComposedPathError as exc:
            check("DEFERRED" in str(exc),
                  f"gpt2-xl refuses loudly in the directional path too: "
                  f"{exc!s:.60}")

    print("== selftest 19: three predictor columns — aggregates and 2×2s ==")
    check(set(RACING_BLOCK_KEYS) == {"star", "composed", "directional"}
          and RACING_BLOCK_KEYS["directional"] == "directional_prediction",
          f"the racing record shape knows all three columns: "
          f"{sorted(RACING_BLOCK_KEYS.values())}")
    tri_slots: list[ScoredSlot] = []
    #  four pairs, chosen so every 2×2 cell and several hit-sets are populated
    tri_spec: list[tuple[str, float, float, float, float]] = [
        # (pair, star pred, composed pred, directional pred, observed)
        ("p1", 0.30, 0.30, 0.30, 0.32),   # all three hit
        ("p2", 0.30, 0.30, 0.60, 0.32),   # star + composed hit, directional miss
        ("p3", 0.30, 0.60, 0.60, 0.32),   # star only
        ("p4", 0.90, 0.90, 0.30, 0.32),   # directional only
    ]
    for pair_name, s_pred, c_pred, d_pred, obs_value in tri_spec:
        slot_i = _slot(pair_name, pair_name + "-t")
        for predictor, predicted in (("star", s_pred), ("composed", c_pred),
                                     ("directional", d_pred)):
            tri_slots.append(score_prediction(
                slot_i, _filed(predictor, predicted),
                _obs(pair_name, pair_name + "-t", obs_value)))
    #  a fifth pair where only the directional column is observed at all
    lone = _slot("p5", "p5-t")
    tri_slots.append(score_prediction(lone, _filed("directional", 0.30),
                                      _obs("p5", "p5-t", 0.32)))
    tri_slots.append(score_prediction(lone, _filed("star", 0.30), None))

    tri_aggs = {a.predictor: a for a in aggregate_scored(tri_slots)}
    check(sorted(tri_aggs) == ["composed", "directional", "star"],
          f"the aggregates cover all three columns: {sorted(tri_aggs)}")
    check((tri_aggs["star"].n_in_band, tri_aggs["star"].n_out_of_band) == (3, 1)
          and (tri_aggs["composed"].n_in_band,
               tri_aggs["composed"].n_out_of_band) == (2, 2)
          and (tri_aggs["directional"].n_in_band,
               tri_aggs["directional"].n_out_of_band) == (3, 2),
          f"per-column counts are independent: star "
          f"{tri_aggs['star'].n_in_band}/4, composed "
          f"{tri_aggs['composed'].n_in_band}/4, directional "
          f"{tri_aggs['directional'].n_in_band}/5")
    check(tri_aggs["star"].n_unscored_no_observation == 1
          and tri_aggs["star"].n_scored_of_record == 4,
          "an unobserved slot stays in n_filed and out of the fraction, "
          "exactly as before the third column existed")

    legacy = head_to_head(tri_slots)
    pairwise = {(r.predictor_a, r.predictor_b): r
                for r in head_to_head_pairwise(tri_slots)}
    star_comp = pairwise[("star", "composed")]
    check((star_comp.n_slots_both_scored, star_comp.n_both_in_band,
           star_comp.n_a_only, star_comp.n_b_only, star_comp.n_both_out_of_band,
           star_comp.n_slots_incomplete)
          == (legacy.n_slots_both_scored, legacy.n_both_in_band,
              legacy.n_star_only, legacy.n_composed_only,
              legacy.n_both_out_of_band, legacy.n_slots_incomplete),
          f"the star×composed 2×2 is UNMOVED by the third column: pairwise "
          f"{(star_comp.n_both_in_band, star_comp.n_a_only, star_comp.n_b_only, star_comp.n_both_out_of_band)}"
          f" == E3's {(legacy.n_both_in_band, legacy.n_star_only, legacy.n_composed_only, legacy.n_both_out_of_band)}"
          f" (Addendum H item 1: model selection, not a patch)")
    check(len(pairwise) == 3 and set(pairwise) == set(PREDICTOR_PAIRS),
          f"every predictor PAIR gets its own 2×2: {sorted(pairwise)}")
    star_dir = pairwise[("star", "directional")]
    check((star_dir.n_slots_both_scored, star_dir.n_both_in_band,
           star_dir.n_a_only, star_dir.n_b_only, star_dir.n_both_out_of_band)
          == (4, 1, 2, 1, 0),
          f"star×directional over the 4 slots both scored: both "
          f"{star_dir.n_both_in_band}, star-only {star_dir.n_a_only}, "
          f"directional-only {star_dir.n_b_only}, both out "
          f"{star_dir.n_both_out_of_band}")
    check(star_dir.n_slots_incomplete == 1,
          "the slot where the star has no observation is INCOMPLETE for that "
          "2×2, never counted as a miss")
    three = head_to_head_three_way(tri_slots)
    check(three.n_slots_all_scored == 4 and three.n_slots_incomplete == 1
          and sum(three.n_by_hit_set.values()) == 4,
          f"the three-way census partitions the {three.n_slots_all_scored} "
          f"slots scored for all three ({three.n_slots_incomplete} incomplete)")
    check(len(three.n_by_hit_set) == 8
          and three.n_by_hit_set["star+composed+directional"] == 1
          and three.n_by_hit_set["star+composed"] == 1
          and three.n_by_hit_set["star"] == 1
          and three.n_by_hit_set["directional"] == 1
          and three.n_by_hit_set["none"] == 0,
          f"all 2³ hit-sets are emitted, zeros included: {three.n_by_hit_set}")

    print("== selftest 20: Addendum D — the constant-α companion, and OWED ==")
    with _tempfile.TemporaryDirectory(prefix="directional_scoring_") as td:
        root = Path(td)
        racing_row = {
            "source": "alpha", "target": "beta", "arm": "native",
            "family": FAMILY_OF_RECORD,
            "star_prediction": {
                "id": "star-prediction/alpha→beta/native-k128",
                "predicted": 0.30, "band": [0.25, 0.35],
                "magnitude_only": False},
            "composed_prediction": {
                "id": "composed-prediction/alpha→beta/native-k128",
                "predicted": 0.34, "band": [0.29, 0.39],
                "magnitude_only": False},
            "directional_prediction": {
                "id": "directional-prediction/alpha→beta/native-k128",
                "predicted": 0.33, "band": [0.28, 0.38],
                "magnitude_only": False},
        }
        second_row = {
            "source": "gamma", "target": "delta", "arm": "native",
            "family": FAMILY_OF_RECORD,
            "star_prediction": {
                "id": "star-prediction/gamma→delta/native-k128",
                "predicted": 0.30, "band": [0.25, 0.35],
                "magnitude_only": False},
            "directional_prediction": {
                "id": "directional-prediction/gamma→delta/native-k128",
                "predicted": 0.30, "band": [0.25, 0.35],
                "magnitude_only": False},
        }
        record_path = root / "predictions-batch4-synthetic-2026-07-29.json"
        record_path.write_text(json.dumps({
            "STATUS": "SYNTHETIC — selftest fixture, files nothing",
            "filed_utc": "2026-07-29",
            "predictions": [racing_row, second_row]}, indent=1))
        obs_path = root / "observed.json"
        obs_path.write_text(json.dumps({"observed": [
            {"source": "alpha", "target": "beta", "arm": "native",
             "family": FAMILY_OF_RECORD, "a_hat": 0.32},
            {"source": "gamma", "target": "delta", "arm": "native",
             "family": FAMILY_OF_RECORD, "a_hat": 0.60}]}, indent=1))

        scored = score_record(record_path, obs_path)
        by_predictor: dict[str, list[ScoredSlot]] = {}
        for s in scored.slots:
            by_predictor.setdefault(s.predictor, []).append(s)
        check(len(scored.slots) == 5 and len(by_predictor["directional"]) == 2,
              f"a racing record's `directional_prediction` block parses beside "
              f"the other two: {len(scored.slots)} scored slots across "
              f"{sorted(by_predictor)}")
        check(by_predictor["directional"][0].verdict == "in-band"
              and by_predictor["directional"][1].verdict == "out-of-band-high",
              "the directional column scores against ITS OWN filed band, "
              "independently of the star's on the same slot")
        check(scored.addendum_d.STATUS == "OWED"
              and scored.addendum_d.gauge == "directional"
              and len(scored.addendum_d.owed) == 3
              and any("ADDENDUM D OWED" in w for w in scored.warnings),
              f"a filed directional column with no companion is Addendum D "
              f"{scored.addendum_d.STATUS} — named, with what is owed spelled "
              f"out, never silently absent")
        check(all(s.constant_alpha is None for s in scored.slots),
              "and NOTHING is auto-computed: not one slot carries a constant-α "
              "verdict the desk did not supply (D2's ᾱ is a grand mean over "
              "the scored set, which only the desk can form)")
        check(scored.addendum_d.n_gauge_scored_of_record == 2
              and scored.addendum_d.n_gauge_in_band == 1,
              "the OWED block still reports the gauge's own counts, so the "
              "margin is one artifact away rather than a re-score away")

        star_only_record = root / "predictions-star-only-2026-07-29.json"
        star_only_record.write_text(json.dumps({
            "filed_utc": "2026-07-29",
            "predictions": [{
                "source": "alpha", "target": "beta", "arm": "native",
                "family": FAMILY_OF_RECORD,
                "star_prediction": {
                    "id": "star-prediction/alpha→beta/native-k128",
                    "predicted": 0.30, "band": [0.25, 0.35],
                    "magnitude_only": False}}]}, indent=1))
        star_obs = root / "observed-star-only.json"
        star_obs.write_text(json.dumps({"observed": [
            {"source": "alpha", "target": "beta", "arm": "native",
             "family": FAMILY_OF_RECORD, "a_hat": 0.32}]}))
        star_scored = score_record(star_only_record, star_obs)
        check(star_scored.addendum_d.STATUS == "NOT-TRIGGERED"
              and star_scored.addendum_d.gauge is None
              and not any("ADDENDUM D" in w for w in star_scored.warnings),
              "a record filing no ceiling-aware gauge is NOT-TRIGGERED — "
              "Addendum D §D1's scope trigger has not fired, and 'owed' and "
              "'not applicable' are different states")
        check((star_scored.head_to_head.n_slots_both_scored,
               star_scored.head_to_head.n_both_in_band) == (0, 0)
              and len(star_scored.head_to_head_pairwise) == 3,
              "and its E3 2×2 still reads exactly as it always did")

        def _companion(**overrides: Any) -> dict[str, Any]:
            doc: dict[str, Any] = {
                "schema": SCHEMA_CONSTANT_ALPHA_COMPANION_V1,
                "gauge": "directional",
                "alpha_bar": 0.70,
                "alpha_bar_provenance": "SYNTHETIC — selftest fixture",
                "rows": [
                    {"source": "alpha", "target": "beta", "arm": "native",
                     "family": FAMILY_OF_RECORD, "predicted": 0.31,
                     "ceiling_term": 0.4429},
                    {"source": "gamma", "target": "delta", "arm": "native",
                     "family": FAMILY_OF_RECORD, "predicted": 0.20}],
            }
            doc.update(overrides)
            return doc

        comp_path = root / "constant-alpha.json"
        comp_path.write_text(json.dumps(_companion(), indent=1))
        with_comp = score_record(record_path, obs_path,
                                 alpha_companion_path=comp_path)
        d_block = with_comp.addendum_d
        dir_slots = [s for s in with_comp.slots if s.predictor == "directional"]
        check(d_block.STATUS == "SCORED" and d_block.alpha_bar == 0.70
              and d_block.n_companion_scored == 2
              and d_block.n_companion_in_band == 1,
              f"a supplied companion scores on the SAME band rule: "
              f"{d_block.n_companion_in_band}/{d_block.n_companion_scored} in "
              f"band (ᾱ-predicted ± .05)")
        check(all(s.constant_alpha is not None for s in dir_slots)
              and all(s.constant_alpha is None for s in with_comp.slots
                      if s.predictor != "directional"),
              "the baseline attaches ONLY to the adopted gauge's slots — the "
              "star and composed columns are untouched by Addendum D")
        check(dir_slots[0].constant_alpha.band                # type: ignore[union-attr]
              == [0.31 - FROZEN_BAND_HALF_WIDTH, 0.31 + FROZEN_BAND_HALF_WIDTH],
              "the companion's band is the FROZEN ±.05 around its own "
              "prediction (D2's 'the SAME frozen bands', the Addendum-C item-1 "
              "reading, stated on the artifact)")
        check(abs((d_block.margin_percentage_points or 0.0) - 0.0) < 1e-9
              and d_block.margin_meets_d3_threshold is False,
              f"D3's margin is computed over the slots where BOTH are scored "
              f"({d_block.margin_percentage_points:+.1f} pp; 1/2 gauge vs 1/2 "
              f"companion) and reported against the ≥15 pp threshold — "
              f"computed, never adjudicated")
        check(d_block.alpha_permutation_null is None
              and d_block.permutation_rank_meets_d3_threshold is None
              and any("permutation null" in o for o in d_block.owed),
              "with no α-permutation null supplied, D3's second leg is OWED — "
              "n=1000 permutations are a desk computation, not a tool's")
        check(not any(a.predictor == "constant-alpha"  # type: ignore[comparison-overlap]
                      for a in with_comp.aggregates)
              and with_comp.head_to_head_three_way.n_slots_all_scored
              == scored.head_to_head_three_way.n_slots_all_scored,
              "and the baseline enters NO aggregate and NO 2×2: it is a "
              "companion, not a fourth racing column")

        null_path = root / "constant-alpha-with-null.json"
        null_path.write_text(json.dumps(_companion(
            rows=[{"source": "alpha", "target": "beta", "arm": "native",
                   "family": FAMILY_OF_RECORD, "predicted": 0.90},
                  {"source": "gamma", "target": "delta", "arm": "native",
                   "family": FAMILY_OF_RECORD, "predicted": 0.20}],
            alpha_permutation_null={"n_permutations": 1000,
                                    "true_in_band_count": 47,
                                    "percentile_rank": 99.4}), indent=1))
        with_null = score_record(record_path, obs_path,
                                 alpha_companion_path=null_path)
        check(abs((with_null.addendum_d.margin_percentage_points or 0) - 50.0)
              < 1e-9
              and with_null.addendum_d.margin_meets_d3_threshold is True
              and with_null.addendum_d.permutation_rank_meets_d3_threshold
              is True
              and not with_null.addendum_d.owed,
              f"a companion the gauge beats by "
              f"{with_null.addendum_d.margin_percentage_points:+.1f} pp with a "
              f"99.4th-percentile null meets BOTH D3 legs, and nothing is left "
              f"owed — the desk still rules")

        partial_path = root / "constant-alpha-partial.json"
        partial_path.write_text(json.dumps(_companion(rows=[
            {"source": "alpha", "target": "beta", "arm": "native",
             "family": FAMILY_OF_RECORD, "predicted": 0.31}]), indent=1))
        partial = score_record(record_path, obs_path,
                               alpha_companion_path=partial_path)
        check(partial.addendum_d.n_slots_companion_missing == 1
              and any("no companion row" in o for o in partial.addendum_d.owed)
              and any("ADDENDUM D PARTIAL" in w for w in partial.warnings),
              "a companion covering only some gauge slots names the shortfall "
              "rather than absorbing it into a denominator")

        alpha_refusals: list[tuple[str, Any]] = [
            ("a companion of another schema name",
             _companion(schema="alpha-baseline/v1")),
            ("no schema declaration",
             {k: v for k, v in _companion().items() if k != "schema"}),
            ("a gauge that is not a predictor of record",
             _companion(gauge="composed-path")),
            ("a non-numeric alpha_bar", _companion(alpha_bar="seven tenths")),
            ("a null alpha_bar", _companion(alpha_bar=None)),
            ("a non-finite alpha_bar", _companion(alpha_bar=float("inf"))),
            ("no alpha_bar provenance", _companion(alpha_bar_provenance="")),
            ("no rows", _companion(rows=[])),
            ("a row that is not an object", _companion(rows=[0.31])),
            ("a row with no predicted",
             _companion(rows=[{"source": "alpha", "target": "beta",
                               "arm": "native"}])),
            ("a row with a non-finite predicted",
             _companion(rows=[{"source": "alpha", "target": "beta",
                               "arm": "native", "predicted": float("nan")}])),
            ("two rows for the same slot",
             _companion(rows=[{"source": "alpha", "target": "beta",
                               "arm": "native", "predicted": 0.31},
                              {"source": "alpha", "target": "beta",
                               "arm": "native", "predicted": 0.29}])),
            ("a malformed permutation null",
             _companion(alpha_permutation_null={"n_permutations": 1000})),
        ]
        for i, (label, payload) in enumerate(alpha_refusals):
            bad = root / f"constant-alpha-bad-{i}.json"
            bad.write_text(json.dumps(payload))
            try:
                load_alpha_companion(bad)
                check(False, f"{label} must be REFUSED")
            except AlphaCompanionError as exc:
                check(f"EXPECTED `{SCHEMA_CONSTANT_ALPHA_COMPANION_V1}`"
                      in str(exc),
                      f"{label} → halt quoting the contract: "
                      f"{str(exc).splitlines()[0]!s:.72}")
        try:
            load_alpha_companion(root / "no-such-companion.json")
            check(False, "a named-but-absent companion must be REFUSED")
        except AlphaCompanionError as exc:
            check("Passing no --alpha-companion at all is the legal way" in str(exc),
                  "an absent NAMED companion halts, and the message names the "
                  "legal way to leave it OWED — absent and malformed never "
                  "collapse into each other")

        wrong_gauge = root / "constant-alpha-wrong-gauge.json"
        wrong_gauge.write_text(json.dumps(_companion(gauge="composed")))
        try:
            score_record(star_only_record, star_obs,
                         alpha_companion_path=wrong_gauge)
            check(False, "a companion for an unfiled gauge must HALT")
        except AlphaCompanionError as exc:
            check("files no 'composed' prediction" in str(exc),
                  f"a baseline whose gauge this record never filed HALTs: "
                  f"{exc!s:.70}")
        stray = root / "constant-alpha-stray.json"
        stray.write_text(json.dumps(_companion(rows=[
            {"source": "alpha", "target": "beta", "arm": "native",
             "family": FAMILY_OF_RECORD, "predicted": 0.31},
            {"source": "beta", "target": "alpha", "arm": "native",
             "family": FAMILY_OF_RECORD, "predicted": 0.31}])))
        try:
            score_record(record_path, obs_path, alpha_companion_path=stray)
            check(False, "a companion row matching no filed slot must HALT")
        except AlphaCompanionError as exc:
            check("match no filed" in str(exc),
                  f"a flipped/typo'd companion row HALTs rather than yielding "
                  f"a margin that looks complete: {exc!s:.60}")
        try:
            main(["--alpha-companion", str(comp_path), "--candidates"])
            check(False, "--alpha-companion without --score-record must not run")
        except SystemExit as exc:
            check(exc.code == 2,
                  f"--alpha-companion is a SCORING input and argparse refuses "
                  f"it at filing time (exit {exc.code})")

    print("== selftest 21: the resolution-provenance ledger (E4 A1) ==")
    #  The ledger must (a) exist for EVERY resolve, hit or miss, (b) name the
    #  tree AND the vintage that answered, (c) change nothing about what the
    #  resolvers return, and (d) get LOUD exactly where a vintage confusion is
    #  born — a resolve that falls THROUGH a set v2.1 root to a pre-v2.1 tree.
    import tempfile as _tmp21

    check(FILED_PATHS is None and V21_ROOT is None,
          "the resolvers default to unpinned + no root — every existing caller "
          "gets the resolution it always got")
    probe_model = "phi-4"
    probe_site = site_of_record(probe_model)
    outer_ledger_before = len(resolution_provenance())
    with provenance_scope() as scoped:
        ref_hit = resolve_hub_map(probe_model, probe_site, "native",
                                  FAMILY_OF_RECORD)
        vec_hit = resolve_vector_bank(probe_model, probe_site)
        ref_miss = resolve_hub_map(probe_model, probe_site, "native",
                                   "proc_kNOPE")
        check(len(scoped) == 3,
              f"one provenance record per resolve, hits and misses alike "
              f"({len(scoped)} for 3 resolves)")
        check([r.kind for r in scoped]
              == ["hub map", "vector bank", "hub map"],
              "each record names the artifact KIND it resolved")
        check(scoped[2].source == "absent" and scoped[2].resolved is None
              and scoped[2].n_probed == len(ref_miss.probed_paths),
              "an absent artifact is recorded as `absent` with everything "
              "probed — N/A-AT-FILING is data here too, never a failure")
        if ref_hit.available:
            check(scoped[0].source == "probe"
                  and scoped[0].resolved == ref_hit.resolved
                  and scoped[0].corpus == ref_hit.corpus
                  and scoped[0].tree == ref_hit.dir_note
                  and scoped[0].probe_index is not None,
                  f"a probe hit records WHICH TREE answered and its vintage: "
                  f"{scoped[0].corpus} — {scoped[0].tree!s:.44}")
            check(not scoped[0].cross_vintage_fallback
                  and scoped[0].v21_root is None,
                  "with no root set nothing is a cross-vintage fallback — the "
                  "v1 regime is not news, and the ledger does not cry wolf")
        else:
            check(True, "phi-4's hub map is not banked in this checkout — the "
                        "hit branch is exercised by the --candidates re-run")
        check(vec_hit.available is (scoped[1].source == "probe"),
              "the vector-bank record agrees with the ref it was built beside")
    check(len(resolution_provenance()) == outer_ledger_before,
          f"provenance_scope restores the previous ledger on exit — the 3 "
          f"scoped records did NOT leak into it (still "
          f"{outer_ledger_before} record(s))")

    with _tmp21.TemporaryDirectory(prefix="composed_prov_") as td:
        root21 = Path(td)
        (root21 / V21_CORPUS_MANIFEST_RELPATH).parent.mkdir(parents=True)
        (root21 / V21_CORPUS_MANIFEST_RELPATH).write_text(
            '{"synthetic": "corpus-v2.1 stand-in for selftest 21"}\n')
        sha21 = sha256_of(root21 / V21_CORPUS_MANIFEST_RELPATH)
        with v21_root_scope(root21, expected_corpus_sha=sha21):
            with provenance_scope() as under_root:
                ref_fall = resolve_hub_map(probe_model, probe_site, "native",
                                           FAMILY_OF_RECORD)
                check(len(under_root) == 1, "one record, as ever")
                if ref_fall.available:
                    check(under_root[0].cross_vintage_fallback
                          and under_root[0].corpus != "v21"
                          and under_root[0].v21_root == str(root21),
                          "a resolve that falls THROUGH a set v2.1 root to a "
                          "pre-v2.1 tree is FLAGGED cross_vintage_fallback — "
                          "the silent half of E4 anomaly A1, now sayable")
                else:
                    check(not under_root[0].cross_vintage_fallback,
                          "an absent artifact is not a cross-vintage fallback "
                          "— there was nothing to fall through to")
    check(V21_ROOT is None, "selftest 21 leaves no root set")

    print("== selftest 22: filed-paths mode — pinned, loud, never a fallback ==")
    #  The mode E4 anomaly A1 asks for: scoring/null tooling pins resolution to
    #  a record's OWN resolved paths. Every refusal below is the same refusal —
    #  a pin that cannot answer must HALT, because the fallback it would take is
    #  the probe order, which answers from the frozen tree the record is not.
    import tempfile as _tmp22

    with _tmp22.TemporaryDirectory(prefix="composed_pin_") as td:
        root22 = Path(td)
        pinned_map = root22 / "fits_v21_synthetic" / "fit_map.npz"
        pinned_vec = root22 / "vectors" / "entropy_gradient_synthetic.npz"
        for artifact in (pinned_map, pinned_vec):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"synthetic artifact - selftest 22 fixture")

        def _record(map_path: str, vec_path: str,
                    second: Optional[dict[str, Any]] = None) -> Path:
            block = {
                "prediction_id": "composed-prediction/alpha→beta/native-k128",
                "pair_id": "alphaL1->betaL2",
                "hub_map_source": {"model": "alpha", "site": 1, "arm": "native",
                                   "family": FAMILY_OF_RECORD,
                                   "resolved": map_path, "corpus": "v21"},
                "hub_map_target": {"model": "beta", "site": 2, "arm": "native",
                                   "family": FAMILY_OF_RECORD,
                                   "resolved": map_path, "corpus": "v21"},
                "source_vector": {"model": "alpha", "site": 1, "path": vec_path},
                "target_vector": {"model": "beta", "site": 2, "path": vec_path}}
            rows: list[dict[str, Any]] = [
                {"source": "alpha", "target": "beta", "arm": "native",
                 "family": FAMILY_OF_RECORD, "composed_prediction": block}]
            if second is not None:
                rows.append(second)
            out = root22 / f"record-{len(rows)}-{abs(hash(map_path)) % 9973}.json"
            out.write_text(json.dumps({"predictions": rows}))
            return out

        rec22 = _record(str(pinned_map), str(pinned_vec))
        pin = load_filed_paths(rec22)
        check(pin.strict and pin.n_pinned == 4
              and sorted(pin.hub_maps) == [f"alphaL1/native-{FAMILY_OF_RECORD}",
                                           f"betaL2/native-{FAMILY_OF_RECORD}"]
              and sorted(pin.vector_banks) == ["alphaL1", "betaL2"],
              f"a pin keys on the resolver's FULL identity — "
              f"{sorted(pin.hub_maps)} + {sorted(pin.vector_banks)}")
        check(pin.sha256 == sha256_of(rec22),
              "the pin carries the record's sha, so a scored read can prove "
              "WHICH record it resolved from")

        with filed_paths_scope(pin), provenance_scope() as pinned_prov:
            ref = resolve_hub_map("alpha", 1, "native", FAMILY_OF_RECORD)
            vec = resolve_vector_bank("beta", 2)
            check(ref.resolved == str(pinned_map)
                  and ref.probed_paths == [str(pinned_map)],
                  "a pinned hub map is TAKEN, not probed — the probe order is "
                  "never consulted, so the frozen tree cannot answer first")
            check(vec.resolved == str(pinned_vec),
                  "and so is a pinned vector bank")
            check([p.source for p in pinned_prov] == ["pin", "pin"]
                  and all(p.pin_record == str(rec22) for p in pinned_prov)
                  and all("filed-paths pin" in p.tree for p in pinned_prov),
                  "every pinned resolve is recorded as such, naming the record")
            check(ref.corpus == "v21" and ref.dir_note.startswith("FILED-PATHS"),
                  "the ref carries the vintage AS FILED and says on its face "
                  "that it was pinned")
            try:
                resolve_hub_map("gamma", 3, "native", FAMILY_OF_RECORD)
                check(False, "a strict pin must refuse an unfiled key")
            except FiledPathsError as exc:
                check("STRICT" in str(exc) and "A1" in str(exc),
                      f"an unfiled key under a STRICT pin HALTs, naming the "
                      f"gap it exists to prevent: {exc!s:.60}")
            try:
                run_gate()
                check(False, "run_gate must refuse while a pin is set")
            except CorpusVintageError as exc:
                check("RESOLUTION PARITY" in str(exc),
                      f"the E1 gate refuses a pin programmatically — parity "
                      f"would test the pin, not the resolver: {exc!s:.60}")
        check(FILED_PATHS is None, "filed_paths_scope restores the pin on exit")

        #  THE A1 SHAPE ITSELF: the pin must beat a probe that WOULD HAVE
        #  SUCCEEDED. `alpha`/`beta` above are synthetic and resolve to nothing,
        #  so they cannot show this; a REGISTERED model whose frozen-tree map is
        #  banked can, and that is exactly the case anomaly A1 describes — the
        #  probe order answers, from the wrong vintage, and the pin must win.
        unpinned_ref = resolve_hub_map(probe_model, probe_site, "native",
                                       FAMILY_OF_RECORD)
        if unpinned_ref.available:
            over_block = {
                "source": probe_model, "target": probe_model, "arm": "native",
                "family": FAMILY_OF_RECORD,
                "composed_prediction": {
                    "pair_id": f"{probe_model}L{probe_site}-override",
                    "hub_map_source": {
                        "model": probe_model, "site": probe_site,
                        "arm": "native", "family": FAMILY_OF_RECORD,
                        "resolved": str(pinned_map), "corpus": "v21"},
                    "source_vector": {"model": probe_model, "site": probe_site,
                                      "path": str(pinned_vec)}}}
            rec_over = root22 / "record-override.json"
            rec_over.write_text(json.dumps({"predictions": [over_block]}))
            with filed_paths_scope(load_filed_paths(rec_over)):
                over = resolve_hub_map(probe_model, probe_site, "native",
                                       FAMILY_OF_RECORD)
            check(over.resolved == str(pinned_map)
                  and over.resolved != unpinned_ref.resolved
                  and str(pinned_map) not in unpinned_ref.probed_paths,
                  f"THE A1 CASE: with the probe order able to answer "
                  f"({unpinned_ref.resolved}), the pin wins and the record's "
                  f"own artifact is used instead — a filed v2.1 leg cannot be "
                  f"silently served from the frozen tree")
        else:
            check(True, f"{probe_model}'s frozen-tree map is not banked in this "
                        f"checkout, so the pin-beats-probe case has no probe to "
                        f"beat; the real proof is the --filed-paths re-run")

        #  THE CENTRAL REFUSAL: a pinned artifact that has gone must HALT, and
        #  must NOT quietly become whatever the probe order finds instead.
        gone = root22 / "fits_v21_synthetic" / "vanished.npz"
        gone.write_bytes(b"about to vanish")
        rec_gone = _record(str(gone), str(pinned_vec))
        pin_gone = load_filed_paths(rec_gone)
        gone.unlink()
        with filed_paths_scope(pin_gone):
            try:
                resolve_hub_map("alpha", 1, "native", FAMILY_OF_RECORD)
                check(False, "a vanished pinned artifact must HALT")
            except FiledPathsError as exc:
                check("will NOT fall back" in str(exc) and "M21b" in str(exc),
                      f"a pinned artifact that is gone HALTs and REFUSES the "
                      f"probe-order fallback: {exc!s:.60}")

        #  Non-strict: unfiled keys probe, but never silently.
        pin_loose = load_filed_paths(rec22, strict=False)
        with filed_paths_scope(pin_loose), provenance_scope() as loose_prov:
            loose = resolve_hub_map(probe_model, probe_site, "native",
                                    FAMILY_OF_RECORD)
            ordinary = [str(fit_path_for(d.path, HUB_MODEL, HUB_SITE_OF_RECORD,
                                         probe_model, probe_site, "native",
                                         FAMILY_OF_RECORD))
                        for d in hub_map_dirs(probe_model)]
            check(bool(loose.probed_paths)
                  and loose.probed_paths == ordinary[:len(loose.probed_paths)],
                  f"a non-strict pin lets an unfiled key probe the ORDINARY "
                  f"list, in order ({len(loose.probed_paths)} of "
                  f"{len(ordinary)} probed before it stopped)")
            check(loose_prov[0].source == "probe"
                  and any("UNPINNED" in n for n in loose_prov[0].notes),
                  "and the resolve is RECORDED as unpinned — permitted, "
                  "warned, never silent")

        #  A record that files two different paths for one key cannot be pinned.
        conflict_block = {
            "source": "alpha", "target": "gamma", "arm": "native",
            "family": FAMILY_OF_RECORD,
            "composed_prediction": {
                "pair_id": "alphaL1->gammaL3",
                "hub_map_source": {"model": "alpha", "site": 1, "arm": "native",
                                   "family": FAMILY_OF_RECORD,
                                   "resolved": str(pinned_vec), "corpus": "frozen"},
                "source_vector": {"model": "alpha", "site": 1,
                                  "path": str(pinned_vec)}}}
        rec_conflict = _record(str(pinned_map), str(pinned_vec), conflict_block)
        try:
            load_filed_paths(rec_conflict)
            check(False, "a record filing two paths for one key must HALT")
        except FiledPathsError as exc:
            check("TWO different" in str(exc),
                  f"a record that files two artifacts under one key HALTs — a "
                  f"pin will not choose: {exc!s:.60}")

        bad_vintage = root22 / "record-bad-vintage.json"
        bad_vintage.write_text(json.dumps({"predictions": [{
            "source": "alpha", "target": "beta", "arm": "native",
            "family": FAMILY_OF_RECORD,
            "composed_prediction": {
                "pair_id": "alphaL1->betaL2",
                "hub_map_source": {
                    "model": "alpha", "site": 1, "arm": "native",
                    "family": FAMILY_OF_RECORD, "resolved": str(pinned_map),
                    "corpus": "corpus-v3-from-the-future"}}}]}))
        try:
            load_filed_paths(bad_vintage)
            check(False, "a record claiming an unknown vintage must HALT")
        except FiledPathsError as exc:
            check("not one of the vintages" in str(exc),
                  f"a vintage label this module cannot name HALTs as a "
                  f"FiledPathsError, not as a raw ValidationError: {exc!s:.60}")

        empty = root22 / "record-empty.json"
        empty.write_text(json.dumps({"predictions": [
            {"source": "a", "target": "b", "arm": "native",
             "family": FAMILY_OF_RECORD, "star_prediction": {"predicted": 0.1}}]}))
        try:
            load_filed_paths(empty)
            check(False, "a record with no composed resolution must HALT")
        except FiledPathsError as exc:
            check("would be empty" in str(exc),
                  f"a star-only record cannot be a composed pin: {exc!s:.60}")

        for argv22, why in (
                (["--gate", "--filed-paths", str(rec22)],
                 "--gate and --filed-paths are MUTUALLY EXCLUSIVE"),
                (["--candidates", "--filed-paths", str(rec22),
                  "--v21-root", str(root22)],
                 "--filed-paths and --v21-root are MUTUALLY EXCLUSIVE"),
                (["--candidates", "--filed-paths-allow-unpinned"],
                 "--filed-paths-allow-unpinned needs a pin to relax")):
            try:
                main(argv22)
                check(False, f"{why} — argparse must refuse")
            except SystemExit as exc:
                check(exc.code == 2, f"{why} (argparse exit {exc.code})")
        check(FILED_PATHS is None and V21_ROOT is None,
              "no refused CLI combination left a pin or a root set")

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
    ap.add_argument("--filed-paths", type=Path, default=None,
                    help="a FILED prediction record whose own `resolved` paths "
                         "PIN resolution (E4 anomaly A1). Every hub map and "
                         "vector is taken from the record instead of probed, so "
                         "scoring and null tooling re-derive the artifacts the "
                         "record was filed from rather than whatever the probe "
                         "order finds. A pinned artifact that is gone HALTS; a "
                         "key the record does not file HALTS unless "
                         "--filed-paths-allow-unpinned. MUTUALLY EXCLUSIVE WITH "
                         "--gate and --v21-root.")
    ap.add_argument("--filed-paths-allow-unpinned", action="store_true",
                    help="let keys the record does not file fall through to the "
                         "normal probes. Each one is WARNED and recorded in the "
                         "provenance ledger as UNPINNED — never silent, because "
                         "a probe answers from the frozen tree first and the "
                         "record was filed from v2.1.")
    ap.add_argument("--provenance-out", type=Path, default=None,
                    help="write the resolution-provenance ledger here: which "
                         "tree answered every resolve, and which vintage. A "
                         "SEPARATE artifact from --out on purpose — the "
                         "composed readout's shape does not move for it.")
    ap.add_argument("--directional-constants", type=Path, default=None,
                    help=f"a {SCHEMA_DIRECTIONAL_CONSTANTS_V1} readout of "
                         f"per-model c_out/c_in (ADDENDUM 2026-07-29-H). With "
                         f"--candidates or --source-model/--target-model it "
                         f"emits the DIRECTIONAL column: "
                         f"directional-prediction/<src>→<tgt>/<arm>-k128, "
                         f"predicted = c_src^out · c_tgt^in, ±.05 band frozen "
                         f"at filing. The SYMMETRIC star column is untouched "
                         f"(H item 2).")
    ap.add_argument("--directional-out", type=Path, default=None,
                    help="write the directional readout JSON here. A SEPARATE "
                         "artifact from --out on purpose: the composed "
                         "readout's shape does not move for the new column.")
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
    ap.add_argument("--alpha-companion", type=Path, default=None,
                    help=f"a {SCHEMA_CONSTANT_ALPHA_COMPANION_V1} artifact "
                         f"(Addendum D §D2), supplied by the DESK. Never "
                         f"computed here: ᾱ is a grand mean over the scored "
                         f"set and §D3's permutation null is a desk "
                         f"computation. Omit it and the scored record names "
                         f"the companion OWED.")
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
    #  THE SAME TRAP, THE OTHER MECHANISM. A pin answers resolution from a
    #  FILING RECORD; the gate's parity leg exists to compare this module's
    #  resolution against the ARCHIVE's, so under a pin it would test the pin.
    #  `run_gate` guards this too — one guard can be bypassed programmatically.
    if args.gate and args.filed_paths is not None:
        ap.error(
            "--gate and --filed-paths are MUTUALLY EXCLUSIVE. The E1 gate is a "
            "FIXED-VINTAGE proof about the archived v1 operationalization and "
            "asserts RESOLUTION PARITY against the archive's own npz files; a "
            "filed-paths pin replaces resolution with a record's filed paths, "
            "so the parity leg would be testing the pin rather than the "
            "resolver. Run the gate on its own, then run the pinned work.")
    #  A pin and a v2.1 root are two different answers to 'which artifact'. The
    #  pin wins for keys it files and the root would silently decide the rest,
    #  which is a mixed regime nobody asked for — refuse instead of ranking them.
    if args.filed_paths is not None and args.v21_root is not None:
        ap.error(
            "--filed-paths and --v21-root are MUTUALLY EXCLUSIVE. Both decide "
            "which artifact answers a resolve: the pin names it outright, the "
            "root re-orders the probes. Combined, pinned keys would come from "
            "the record and unpinned ones from the root — a mixed regime whose "
            "vintage is a per-key fact, which is exactly what E4 anomaly A1 was "
            "about. Pin to the record, or point at the root; not both.")
    if args.filed_paths_allow_unpinned and args.filed_paths is None:
        ap.error(
            "--filed-paths-allow-unpinned only means anything with "
            "--filed-paths: it relaxes a pin, and with no pin every resolve "
            "probes already.")
    if (args.score_record is None) != (args.observed is None):
        ap.error(
            "--score-record and --observed go together. Scoring is a DESK ACT "
            "against observed â values the desk produces at first-read; this "
            "tool never observes, never fits, and never auto-scores at filing "
            "time — it only builds the machine-readable artifact.")
    if args.alpha_companion is not None and args.score_record is None:
        ap.error(
            "--alpha-companion is a SCORING input (Addendum D §D2's constant-α "
            "baseline) and only applies with --score-record/--observed. It is "
            "never consulted at filing time: a baseline scored before the fit "
            "exists would have nothing to be a baseline against.")
    if args.directional_out is not None and args.directional_constants is None:
        ap.error(
            "--directional-out needs --directional-constants: the directional "
            "column is COMPUTED FROM a "
            f"{SCHEMA_DIRECTIONAL_CONSTANTS_V1} readout and is never emitted "
            "empty, which would look like 'no slots file' rather than 'no "
            "constants were supplied'.")
    if (args.directional_constants is not None
            and not (args.candidates or args.source_model)):
        ap.error(
            "--directional-constants needs --candidates or "
            "--source-model/--target-model — it names the constants, not the "
            "slots to compute.")

    if not (args.gate or args.candidates or args.resolution_sweep
            or args.source_model or args.score_record):
        raise SystemExit("pass --selftest, --gate, --candidates, "
                         "--resolution-sweep, --source-model/--target-model, "
                         "or --score-record/--observed")

    directional_constants: Optional[DirectionalConstants] = None
    if args.directional_constants is not None:
        try:
            directional_constants = load_directional_constants(
                args.directional_constants)
        except DirectionalConstantsError as exc:
            #  An EXPECTED halt with a meaningful message: the constants readout
            #  is not the contract this module consumes. Reported cleanly and
            #  nonzero so the derivation lane can be reconciled from the error.
            print(f"\nDIRECTIONAL CONSTANTS HALT — {exc}")
            return 1

    if args.v21_root is not None:
        try:
            set_v21_root(args.v21_root,
                         manifest_relpath=args.v21_corpus_manifest)
        except CorpusVintageError as exc:
            #  An EXPECTED halt with a meaningful message: the operator pointed
            #  at something that is not the v2.1 basis of record.
            print(f"\nVINTAGE HALT — {exc}")
            return 1

    if args.filed_paths is not None:
        try:
            set_filed_paths(load_filed_paths(
                args.filed_paths, strict=not args.filed_paths_allow_unpinned))
        except FiledPathsError as exc:
            #  An EXPECTED halt: the record cannot be read as a pin.
            print(f"\nFILED-PATHS HALT — {exc}")
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
        try:
            sweep = registry_resolution_sweep(args.family)
        except FiledPathsError as exc:
            print(f"\nFILED-PATHS HALT — {exc}")
            return 1
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
        try:
            computed = run_candidates(args.family, args.pairs_json)
        except FiledPathsError as exc:
            print(f"\nFILED-PATHS HALT — {exc}")
            return 1
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

    #  THE DIRECTIONAL COLUMN (Addendum 2026-07-29-H). Emitted only when the
    #  constants readout is supplied, into its OWN readout: nothing above this
    #  point changes shape, and the symmetric star column is untouched.
    directional_readout: Optional[DirectionalReadout] = None
    if directional_constants is not None:
        directional_readout = DirectionalReadout(
            generated=date.today().isoformat(), family=args.family,
            constants_readout=directional_constants.path,
            constants_readout_sha256=directional_constants.sha256,
            constants_gauge=directional_constants.gauge,
            constants_corpus_manifest_sha256=(
                directional_constants.corpus_manifest_sha256),
            constants_notes=list(directional_constants.notes))
        if args.candidates:
            computed_dir = run_directional(directional_constants, args.family,
                                           args.pairs_json)
            directional_readout.predictions.extend(computed_dir.predictions)
            directional_readout.na_at_filing.extend(computed_dir.na_at_filing)
        if args.source_model:
            if not args.target_model:
                raise SystemExit("--source-model needs --target-model")
            one_dir = directional_pair(args.source_model, args.target_model,
                                       directional_constants,
                                       family=args.family, arm=args.arm)
            if isinstance(one_dir, DirectionalNotFilable):
                directional_readout.na_at_filing.append(one_dir)
            else:
                directional_readout.predictions.append(one_dir)
        total_dir = (len(directional_readout.predictions)
                     + len(directional_readout.na_at_filing))
        print(f"\n{'directional slot (Addendum H)':52s} {'arm':7s} "
              f"{'c_out':>9s} {'c_in':>9s} {'â_dir':>9s} "
              f"{'band lo':>9s} {'band hi':>9s} verdict")
        for pred in directional_readout.predictions:
            print(f"{pred.pair_id:52s} {pred.arm:7s} "
                  f"{pred.c_out_source:+9.4f} {pred.c_in_target:+9.4f} "
                  f"{pred.filed_predicted:+9.4f} "
                  f"{pred.filed_band[0]:+9.4f} {pred.filed_band[1]:+9.4f} "
                  + ("MAGNITUDE-ONLY " if pred.magnitude_only else "")
                  + ("FLAGGED" if pred.flags else "filable"))
        for na_dir in directional_readout.na_at_filing:
            print(f"{na_dir.pair_id:52s} {na_dir.arm:7s} {'—':>9s} {'—':>9s} "
                  f"{'—':>9s} {'—':>9s} {'—':>9s} N/A-AT-FILING "
                  f"({', '.join(na_dir.missing_sides)})")
        for note in directional_readout.constants_notes:
            print(f"  CONSTANTS NOTE: {note}")
        print(f"\ndirectional: {len(directional_readout.predictions)}/"
              f"{total_dir} filable, "
              f"{len(directional_readout.na_at_filing)}/{total_dir} "
              f"N/A-AT-FILING; constants {directional_constants.path} sha "
              f"{directional_constants.sha256[:12]}…, gauge "
              f"{directional_constants.gauge!r}, corpus "
              f"{directional_constants.corpus_manifest_sha256[:8]}…")
        if args.directional_out:
            args.directional_out.parent.mkdir(parents=True, exist_ok=True)
            args.directional_out.write_text(
                directional_readout.model_dump_json(indent=1))
            logger.info("wrote %s (%d directional prediction(s), %d "
                        "N/A-at-filing)", args.directional_out,
                        len(directional_readout.predictions),
                        len(directional_readout.na_at_filing))

    if args.source_model:
        if not args.target_model:
            raise SystemExit("--source-model needs --target-model")
        try:
            one = compose_pair(args.source_model, args.target_model,
                               family=args.family, arm=args.arm)
        except FiledPathsError as exc:
            print(f"\nFILED-PATHS HALT — {exc}")
            return 1
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
            scored = score_record(args.score_record, args.observed,
                                  alpha_companion_path=args.alpha_companion)
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
            print(f"\nhead-to-head (Addendum E §E3, star × composed) over "
                  f"{h.n_slots_both_scored} slot(s) scored for both: both "
                  f"{h.n_both_in_band} · star-only {h.n_star_only} · "
                  f"composed-only {h.n_composed_only} · both out "
                  f"{h.n_both_out_of_band} ({h.n_slots_incomplete} incomplete)")
            for pair_row in scored.head_to_head_pairwise:
                print(f"  2×2 {pair_row.predictor_a:>11s} × "
                      f"{pair_row.predictor_b:<11s} n={pair_row.n_slots_both_scored:<4d} "
                      f"both {pair_row.n_both_in_band} · "
                      f"{pair_row.predictor_a}-only {pair_row.n_a_only} · "
                      f"{pair_row.predictor_b}-only {pair_row.n_b_only} · "
                      f"both out {pair_row.n_both_out_of_band} "
                      f"({pair_row.n_slots_incomplete} incomplete)")
            three = scored.head_to_head_three_way
            print(f"  three-way hit census over {three.n_slots_all_scored} "
                  f"slot(s) scored for all three "
                  f"({three.n_slots_incomplete} incomplete): "
                  + " · ".join(f"{label} {count}"
                               for label, count in three.n_by_hit_set.items()))
            d = scored.addendum_d
            print(f"\nAddendum D companion: {d.STATUS}"
                  + (f" (gauge {d.gauge!r})" if d.gauge else "")
                  + (f" — ᾱ {d.alpha_bar:+.6f}; gauge "
                     f"{d.n_gauge_in_band}/{d.n_gauge_scored_of_record} in "
                     f"band, constant-α {d.n_companion_in_band}/"
                     f"{d.n_companion_scored}; margin "
                     f"{d.margin_percentage_points:+.1f} pp "
                     f"(D3 wants >= 15: "
                     f"{'MET' if d.margin_meets_d3_threshold else 'not met'})"
                     if d.STATUS == "SCORED"
                     and d.margin_percentage_points is not None else ""))
            for item in d.owed:
                print(f"  OWED: {item}")
            for warning in scored.warnings:
                print(f"  WARNING: {warning}")
            written = write_scored_record(scored, out, args.overwrite_scored)
            print(f"\nscored record: {written} (sha "
                  f"{scored.record_sha256}). Campaign gates are NOT evaluated "
                  f"here — the desk reads these counts.")
        except ScoringError as exc:
            print(f"\nSCORING HALT — {exc}")
            return 1

    #  THE RESOLUTION-PROVENANCE LEDGER (E4 anomaly A1). Printed only when the
    #  operator asked for a pin or for the artifact: the no-root stdout of every
    #  existing mode is a frozen surface (the E1 gate and the --candidates
    #  column are re-run byte-for-byte at review), so this reports on demand and
    #  is otherwise carried as data.
    if args.filed_paths is not None or args.provenance_out is not None:
        prov = provenance_readout()
        print(f"\nresolution provenance: {prov.n_resolves} resolve(s) — "
              + " · ".join(f"{k} {v}" for k, v in sorted(prov.n_by_source.items()))
              + " | vintage " + " · ".join(f"{k} {v}" for k, v
                                           in sorted(prov.n_by_corpus.items())))
        if prov.filed_paths_record:
            assert FILED_PATHS is not None
            print(f"  pinned to {prov.filed_paths_record} (sha "
                  f"{FILED_PATHS.sha256[:12]}…, {len(FILED_PATHS.hub_maps)} hub "
                  f"map(s) + {len(FILED_PATHS.vector_banks)} vector bank(s), "
                  f"strict={FILED_PATHS.strict})")
        if prov.n_cross_vintage_fallback:
            print(f"  CROSS-VINTAGE FALLBACK on {prov.n_cross_vintage_fallback} "
                  f"resolve(s): a corpus-v2.1 root was in force and these fell "
                  f"THROUGH it to a pre-v2.1 tree")
        if prov.n_absent:
            print(f"  {prov.n_absent} resolve(s) found nothing (N/A-AT-FILING "
                  f"or an un-landed target build) — data, not a failure")
        if args.provenance_out:
            args.provenance_out.parent.mkdir(parents=True, exist_ok=True)
            args.provenance_out.write_text(prov.model_dump_json(indent=1))
            logger.info("wrote %s (%d resolve record(s))", args.provenance_out,
                        prov.n_resolves)

    payload = readout.model_dump_json(indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        logger.info("wrote %s (%d prediction(s), %d N/A-at-filing)", args.out,
                    len(readout.predictions), len(readout.na_at_filing))
    return status


if __name__ == "__main__":
    sys.exit(main())
