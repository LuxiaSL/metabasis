"""A8 Leg-0 — T2: Phase B fit grid (CPU, local). Charter §2/§4, session spec Phase B.

Fits the structure-preserving map g between two models' residual state spaces from
paired forced-replay means, per (site-pair x template-arm x family):
  - semi-orthogonal Procrustes on rank-k PC subspaces, k in {32,128,512}
    (per-side PCA fit on TRAIN rows only; W^T W = I; optional isotropic scale)
  - ridge affine (liberal variant; alpha by closed-form LOO PRESS on train)

Gates per fit (frozen rule, prereg §1 FIT VALIDITY):
  held-out state-prediction R^2 separates from BOTH the shuffled-pair null AND the
  stratum-preserving shuffle null (re-pair with a different text from the same
  (stratum, voice, mode) group — the sharp null); per-stratum R^2 carries in >=2 of
  THE STRATA THE PINNED CORPUS MANIFEST NAMES. Plus: CKA before/after, and the
  two-arm g-agreement robustness read.

THE STRATUM VOCABULARY IS BASIS-DERIVED (desk ruling 2026-08-03). No stratum name
appears in this module: `derive_strata` reads them off the manifest's per-entry
`stratum` field in first-occurrence order, so the set is a pure function of the
manifest bytes (deterministic under the `corpus_manifest_sha256` in every stamp).
On webtext-v3 that is the four strata frozen prereg §3.4 names; on v2.1 manifests
it is S1/S2/S3 (+S5 on the Leg-4 augmented corpus) — identical to the tuple this
module used to hardcode, which is why no banked v2.1 readout moves.

STATE-BANK CONTRACT (T4 `collect_mean_states.py` imports save_state_bank from here —
single source of truth):
  states/states_{model}_{arm}.npz
      text_ids : unicode array [n]   (must cover the corpus manifest exactly)
      L{site}  : float32 [n, hidden] (per-text MEAN over completion-token states, raw
                 unnormalized fp32 — normalization happens HERE at fit time)
  states/norms_{model}_{arm}.json
      {"L{site}": median over texts of ||per-text mean state||_2, ...}

Artifacts (under --arm-root/fits/):
  pca_{model}_L{site}_{arm}.npz            per-side PCA bank (kmax components)
  fit_{src}L{s}__{tgt}L{t}_{arm}_proc_k{k}.npz / _ridge.npz + .json sidecars
  cp2_summary.json                         the CP-2 table, one record per fit

THE SPLIT IS BASIS-DEPENDENT AND HAS THREE LANES (brief BRIEF-v3-splits-wiring
-2026-08-03). The webtext-v3 membership is NOT derived at fit time: it is FROZEN
by `derive_webtext_splits.py` per prereg §2 / §6-I1 and CONSUMED here.
  --splits-artifact …/splits.json   the frozen v3 main split (280/1200)
  --half a|b + --halves-artifact …/halves.json
                                    one frozen §6-I1 half, fitted on that half's
                                    OWN internal train/test split
  (neither)                         the LEGACY in-code v2.1 topic-grouped
                                    derivation — byte-exact, and the ONLY lane
                                    that derives anything
A webtext-v3 manifest with no artifact REFUSES (never an empty test set, never a
re-derivation); an artifact that does not belong to the loaded corpus REFUSES.

Run (from pipeline/):
  python -m metabasis.scripts.fit_transport_maps --selftest          # synthetic validation
  python -m metabasis.scripts.fit_transport_maps                      # real grid (post CP-1)
  python -m metabasis.scripts.fit_transport_maps \\
      --splits-artifact staging/webtext-v3-draft/splits.json          # the v3 grid
  python -m metabasis.scripts.fit_transport_maps --half a \\
      --halves-artifact staging/webtext-v3-draft/halves.json \\
      --fits-dirname fits_half_a                                      # the I1 half
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from metabasis.roster import SCAN_GRIDS, fiat_grid_problems
from metabasis.threads import THREAD_STAMP_KEY, thread_config_stamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_transport_maps")

# ---------------------------------------------------------------- constants (frozen)
A8_SEED = 80
K_GRID = (32, 128, 512)
N_NULL_REPS = 20
HELD_OUT_TOPICS = 5           # of 20 (topic-grouped split: a topic is wholly one side)
HELD_OUT_S2 = 40              # of 160 S2 shards
N_PROBES = 50                 # random unit probes for the two-arm g-agreement read
SITES = {"3b": (13, 14, 18), "8b": (14, 16, 18), "qwen-7b": (19, 21, 23),
         "dsv2-lite": (18, 22),
         # --- extension pairs (desk-smalls, authorized DESK-RULINGS-LEG6 §4) ---
         # gemma3-27b: RULED BY LUXIA 2026-07-29 (session-5 close) from the
         # SIX-SITE â EVIDENCE TABLE — readout of record
         # `site_evidence_gemma3-27b_20260729-055507.json`, sha `7f59af50…`,
         # 48 rows, strict (fit-local) norms, both hub source columns.
         # L38 is the site of record ⋆ and L41 is the robustness site; the
         # carried-banked peak region (34, 36, 38) is HISTORY, and L36 is
         # RETIRED to scanned-history with it.
         # THE DECIDING STRUCTURE, so nobody re-derives this from the r² curve:
         #  · The r² curve and the â column are INVERTED — r(r², â) = −.964 at
         #    n=6. The r² PEAK L13 (.5274) is SUB-NULL on â (+.019/+.038 against
         #    a q95 floor ~.09), build coherence .2334, bucket diffuse-target;
         #    L17 (global) is the same story. NEVER CROWN A SITE FROM r² ALONE —
         #    this is the 70B's retired-L17 case replicating at the same ~21%
         #    depth, one model over.
         #  · What DOES order the sites is the coherence-depth law (campaign
         #    level r=.941 across 10 builds), and it reappears inside this one
         #    model: r(depth, coherence) = .9946, r(coherence, â) = .997.
         #    â ranking L38 > L41 > L35 ≈ L36 > L17 ≈ L13.
         #  · L38 (local, â +.338/+.407, ceiling .627, coherence .724) DOMINATES
         #    the incumbent L36 (+.2772/+.3725, ceiling .603, coherence .669) on
         #    BOTH hub columns, +22.0% / +9.1%, with the higher ceiling and the
         #    higher coherence. L36's number is legitimate and mid-pack — it is
         #    retired for being dominated, not for being wrong.
         #  · L41 is the robustness site (global, â +.324/+.389), the same
         #    two-site shape as the 70B's L37 ⋆ / L43 pair. Global attention buys
         #    nothing at matched depth in either column (L35-vs-L36: the two hub
         #    columns disagree on the sign of a ≤.007 gap) — the ordering tracks
         #    depth/coherence, not attention type.
         # WHAT STAYS BANKED AS EVIDENCE (retired ≠ deleted): L36's 16-site
         # alignment-curve cells and its FD-gated site-evidence vector remain
         # banked and re-readable, as do the (34, 36, 38) fits — the whole
         # frozen-era gemma object roster (V7, Vrep⊥, Veos⊥, Vconf, Vtemp, V3)
         # lives at L36 and `read_transported_axes` still anchors there BY
         # CONSTRUCTION — that reader resolves banked fits BY PATH from
         # cp2_summary, so the historical reads are untouched by this grid. What
         # DOES change is that a REFIT at L36 now needs `--tgt-sites 36` on the
         # command line, which is the point: a retired site never resolves by
         # default again (the 70B's L17 precedent).
         # Both 38 and 41 are on the model's own effective scan grid (the ruled
         # 15-site extension; 41 is in the computed 12), so the ratification
         # invariant is satisfied — this is a curve-visited pair re-ranked by â
         # evidence, not a fiat grid.
         # THE VECTOR SIDE IS STILL OPEN: the six site-evidence vectors are
         # FROZEN-v1 (`a6712ca0…`) SELECTION instruments and never file. A fresh
         # FD-gated corpus-v2.1 (`5ae355bc…`) L38 build is required before any
         # gemma slot files — see `read_composed_predictions.SITE_OF_RECORD`.
         "gemma3-27b": (38, 41),                       # ⋆ L38 primary, L41 robustness
         # olmo2-7b: NO banked site curve exists (A3/A5 were cut for this model), so
         # the sites are a mid-depth BAND and the site of record is picked from the
         # fit's own alignment curve afterwards — never by fiat (baton item 1).
         "olmo2-7b": (12, 16, 20, 24),
         # --- wave-1 graduations (collection phase) ----------------------------
         # Ratified by Luxia 2026-07-27 from the wave-1 12-site alignment-curve
         # scans: `outputs/collection/<key>/fits_scan_<key>/cp2_summary.json`,
         # held-out r² at proc_k128, NATIVE arm (the family of record), hub 8B.
         # Each grid is the peak flanked by its two neighbouring scanned sites;
         # ⋆ marks the site of record. These keys stay in metabasis.roster too —
         # SCAN_GRIDS remains the record of what was scanned to get here.
         "qwen2.5-3b-instruct": (24, 26, 28),          # ⋆ L26
         "qwen2.5-14b-instruct": (26, 29, 32),         # ⋆ L29
         "qwen2.5-32b-instruct": (42, 46, 50),         # ⋆ L46
         "mistral-7b-instruct-v0.3": (13, 15, 17),     # ⋆ L15
         "olmo2-7b-instruct": (13, 15, 17),            # ⋆ L15
         "phi-4": (16, 19, 21),                        # ⋆ L19
         # phi-3.5-mini-instruct: FOUR sites, Luxia-ratified — the native curve is
         # double-humped (L13 .632 / L17 .630 twin peaks with an L15 dip), so both
         # humps stay in the grid. Precedent for >3-site grids: olmo2-7b above.
         "phi-3.5-mini-instruct": (11, 13, 15, 17),    # ⋆ L13
         # --- hub-rungs-2 graduations (collection phase) -----------------------
         # Ratified by Luxia 2026-07-27 from the hub-rungs-2 scans; same read as
         # wave-1 (held-out r² at proc_k128, hub 8B, peak flanked by neighbours).
         # llama-3.1-70b-instruct: the scale ceiling. RE-RATIFIED 2026-07-27
         # (Luxia, session 3) — the grid is the two DEEP sites, L37 primary and
         # L43 robustness, and L17 is RETIRED from the fit grid.
         # What changed and why, so nobody re-derives it from the r² curve alone:
         #  · The alignment-curve scan picked L17 (peak UNANIMOUS across all six
         #    columns, r²=.577–.626, cka_after .87–.93 — a textbook interior
         #    peak), and the proportional-depth region L43–48 was that curve's
         #    TROUGH. On r² alone L17 still looks like the answer.
         #  · The 70B chain-break read (in-lineage star, session 3) settled it
         #    the other way on the evidence that actually matters here: deep
         #    â = .4557 (L37) / .4480 (L43), null-clearing ~5×, implied
         #    c = .5441/.5349 — while the L17 rebuilt-column â (.067) is
         #    SUB-NULL and was never a valid c measurement at all. L17 is a
         #    shallow-basin SITE ARTIFACT; it retires to that case study.
         #  · Two deep sites, not one, by explicit ruling: the pair is the
         #    same-model robustness check behind "c_M ≠ representable fraction"
         #    (â flat, coherence .431/.432 flat across L37→L43, while the
         #    ceilings differ 4.5% the other way).
         # Both sites are ON the model's own 12-site scan grid (…,32,37,43,48,…),
         # so the ratification invariant is satisfied — this is a curve-visited
         # pair re-ranked by â evidence, not a fiat grid. gpt2-xl below is the
         # same shape of ruling (sites of evidence, record deferred to â).
         # L17 is still fully banked (the canonical npz holds L17+L37+L43); any
         # re-read of the site-artifact case study passes `--tgt-sites 17`
         # EXPLICITLY, which is the point — the retired site never resolves by
         # default again.
         "llama-3.1-70b-instruct": (37, 43),           # ⋆ L37 primary, L43 robustness
         # pythia-6.9b: ⋆ L31, and the grid is drawn from the RULED 16-site
         # extension (L28–31), not the base 12 — the audit invariant checks the
         # extension-inclusive scan grid, which is why this is not a fiat grid.
         # THREE FLAGS travel with this node:
         #  (a) EDGE BY ARCHITECTURE EXHAUSTION — the curve rises monotonically to
         #      L31 = num_hidden_layers-1 and is still accelerating there
         #      (L30→L31 is the largest step in the curve). There is no L32; the
         #      window did not fail, the model ran out of layers.
         #  (b) GEOMETRY-POOR EVERYWHERE — cka_after ≤ .19 at every site (against
         #      the 70B's .87–.93), rising monotonically with r². Both criteria
         #      select L31, so the pick is unambiguous, but this node is
         #      predictable without being geometrically aligned.
         #  (c) STANDING DESK THREAD on that predictability-without-geometry gap;
         #      any pythia result leaning on transported geometry cites this.
         "pythia-6.9b": (29, 30, 31),                  # ⋆ L31
         # gpt2-xl: ratified by Luxia 2026-07-27 from the COMPLETED 25-site curve
         # (computed 12 + ruled both-edge extension L0–L6, L42–L47).
         # NO SITE OF RECORD YET — deliberately. The curve is a BATHTUB, not a
         # peak: it rises L0→L7, falls off a cliff (L7 .449 → L10 .110), troughs
         # at L13–16 (≈.05), then climbs monotonically to L47 and is still
         # climbing there (L47 = n_layer-1, an edge by architecture exhaustion).
         # The two ends are degenerate and the winner FLIPS with the hub source
         # (8bL14/8bL16 → L7 by .007/.002; 8bL18 → L47 by .011), all inside the
         # edge nulls (q95 .02–.066). Picking either end would be fiat wearing a
         # curve's clothes.
         # THREE EXPERIMENT SITES, ratified together: L7 (low end), L26 (middle
         # probe), L47 (high end); L6 and L46 flank the two ends as the usual
         # neighbours. The site of record is DEFERRED to their â evidence —
         # Luxia's rationale: injecting into the ends is likely weak whatever the
         # r², so a middle site is extra evidence even if its fit is poor.
         # L26 is picked MECHANICALLY, not by eye: the scanned site nearest 50%
         # depth (L24.0 of 48). L22 and L26 tie at |Δ|=2.0; the tie breaks on
         # higher r² (L26 .127/.110/.113 vs L22 .088/.076/.077 — L26 higher from
         # all three hub sources). Its geometry is poor (cka_after ≈ .10 against
         # ≈ .64 at the ends), which is the point of probing it.
         # TRUNCATION CAVEAT RIDES EVERY USE (prereg ADDENDUM 2026-07-27-B): this
         # node's states are collected with per-text truncation to the first 1024
         # tokens, so for the 148 affected texts any gpt2-xl pair compares
         # full-text against truncated-text mean states. Flagged at scoring.
         "gpt2-xl": (6, 7, 26, 46, 47),                # ⋆ DEFERRED (L7|L26|L47)
         # --- big-chain graduations (collection phase) -------------------------
         # llama-3.1-405b-instruct (roster row 20, THE DENSE SCALE CEILING and
         # the prereg §3 quad-hub SCALE-PROBE AUDIT HUB):
         # RULED BY THE DESK 2026-07-29 UNDER LUXIA'S OVERNIGHT DELEGATION 2
         # (ledgered) from the FIVE-SITE â EVIDENCE TABLE — readout of record
         # `site_evidence_llama-3.1-405b-instruct_20260729-113819.json`,
         # sha `04f2a2c4…`, 40 rows, strict (fit-local) norms, FROZEN-v1
         # evidence basis (corpus `a6712ca0…`). This row was SCAN-REGISTRY ONLY
         # until now; L99 is the site of record ⋆ and L107 is the robustness
         # site. Sites of evidence: L19, L43, L91, L99, L107.
         # THE EVIDENCE, so nobody re-derives this from the r² curve:
         #  · UNANIMITY, TOP AND BOTTOM. Across ALL FOUR robustness columns —
         #    native k128, native k32, raw k128 (all hub 8bL14) and the
         #    rebuilt-L16 hub — the ranking opens L99 > L107 and closes on L19:
         #      native/k128/L14   L99 > L107 > L43  > L91  > L19
         #      native/k32 /L14   L99 > L107 > L91  > L43  > L19
         #      raw   /k128/L14   L99 > L107 > L43  > L91  > L19
         #      native/k128/L16   L99 > L107 > L91  > L43  > L19
         #    rank-1, rank-2 and LAST are each unanimous; only the middle two
         #    swap. L99 sits at fractional depth .786, L107 at .849.
         #  · L19 IS THE SHALLOW r² TRAP IN ITS PUREST FORM. It holds the
         #    SECOND-HIGHEST r² on the grid (.5246) and finishes DEAD LAST on â
         #    in every column: â = −.0275 / +.0003 / −.0487 / −.0493, SUB-NULL
         #    in BOTH hub columns (the only site sub-null in the L14 column at
         #    all), coherence .109 against .32–.46 elsewhere. Its FD gate passes
         #    but is the worst on the grid (best_median_rel_error .11571 against
         #    .00719–.02847 at the other four sites), and it is the ONLY build
         #    with `direction_specific = False` in ANY fd-gate artifact the
         #    campaign holds — 21 builds over four models (gemma3-27b ×6,
         #    3.3-70B ×5, qwen3-30b-a3b ×5, this node ×5), one False, and it is
         #    this one. A high r² bought with a non-direction-specific gradient
         #    is exactly the artifact â exists to catch.
         #  · THE r²/â INVERSION, FOURTH INSTANCE — and the THIRD on the Llama
         #    family. The r² PEAK is L43 (.5953, fractional depth .341); it
         #    ranks only 3rd of 5 on â in the L14 column and 4th of 5 in the
         #    rebuilt-L16 column, where it goes SUB-NULL (+.0795 against a q95
         #    floor of .0890). Running tally: llama-3.1-70B L17, gemma3-27b L13,
         #    llama-3.3-70B L17, and now this node's L43. NEVER CROWN A SITE
         #    FROM r² ALONE.
         #    DEPTH CAVEAT ON THAT TALLY: the first three all inverted at ~21%
         #    fractional depth, which the 3.3-70B comment below calls a
         #    campaign-wide signature. This node does NOT reproduce the 21%
         #    number — its inverting peak is at .341 and its worst site at .151.
         #    What generalizes is the SHALLOW HALF of the grid, not the specific
         #    depth; do not read 21% as a constant.
         #  · The correlation structure says it quantitatively: r(r², â) is
         #    NEGATIVE in both hub columns (−.412 / −.776) while
         #    r(coherence, â) = +.927 / +.990. Coherence is the reliable â
         #    predictor on ALL THREE quartet nodes (405B +.93/+.99 · 3.3-70B
         #    +.96/+.97 · qwen3-30b-a3b +.91/+.92); r² is not — it inverts here
         #    and on the 3.3-70B, and only agrees on qwen3 (+.58/+.58).
         #    Supporting: r(depth, coherence) = +.949, r(depth, â) = +.765,
         #    r(ceiling, â) = +.844, r(â L14, â rebuiltL16) = +.887.
         #  · MARGIN, STATED HONESTLY: L99 over L107 is NARROW — +.0152 /
         #    +.0729 / +.0034 / +.0102 across the four columns, so the raw k128
         #    column decides it by .0034 — but it is SIGN-CONSISTENT in all
         #    four, never once flipping. L107 rides along as the robustness site
         #    precisely because the margin is thin, the same two-site shape as
         #    3.1-70B's L37 ⋆ / L43, gemma's L38 ⋆ / L41 and 3.3-70B's
         #    L58 ⋆ / L63.
         # THE SCALE CAVEAT, ON THE RECORD BECAUSE IT CUTS AGAINST THE ROW'S OWN
         # PREMISE: the campaign's LARGEST model is its WEAKEST TRANSPORTER of
         # the overnight quartet. Best â in the canonical native/k128/L14 cell
         # is +.1453 here against +.3058 (3.3-70B L58) and +.3283 (qwen3-30b-a3b
         # L35) — roughly HALF — and â/ceiling is .254 against .521 and .500.
         # The CEILINGS are comparable (.573 vs .587 / .656), so what halves is
         # the â, not the headroom. This FEEDS the scale/ρ reading (row 20 is
         # the scale-probe audit hub; the larger-hubs-transfer-better
         # conjecture is exactly what its audit set tests) and does NOT
         # undermine the within-node site ruling, which is unanimous in all
         # four columns regardless of the absolute level.
         # ARCHITECTURE NOTE: LlamaConfig with NO `layer_types` — uniform full
         # attention at every one of the 126 layers, confirmed per-site in the
         # readout. As on rows 11/12 there is no local/global contrast to read,
         # so depth and coherence are the only structure available.
         # Both 99 and 107 are ON the model's own computed 12-site scan grid
         # (19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107 — 126 layers, so the
         # rungs differ from the 80-layer 70Bs' and the comparison to them is by
         # FRACTIONAL DEPTH, never site-for-site), so the ratification invariant
         # is satisfied by construction — a curve-visited pair re-ranked by â
         # evidence, never a fiat grid.
         # THE VECTOR SIDE IS OPEN, AND IT GATES MORE THAN THIS ROW: the five
         # site-evidence vectors are FROZEN-v1 SELECTION instruments and never
         # file. A fresh FD-gated corpus-v2.1 L99 build is required before any
         # 405B slot files — and the same re-bank gates the ρ-law ceremony's
         # audit pairs (pre-statement `7542b377…`). See
         # `read_composed_predictions.SITE_OF_RECORD`.
         "llama-3.1-405b-instruct": (99, 107),         # ⋆ L99 primary, L107 robustness
         # llama-3.3-70b-instruct (roster row 12, the post-training-vintage row):
         # RULED BY THE DESK 2026-07-29 UNDER LUXIA'S OVERNIGHT DELEGATION 2
         # (ledgered) from the FIVE-SITE â EVIDENCE TABLE — readout of record
         # `site_evidence_llama-3.3-70b-instruct_20260729-075541.json`,
         # sha `e6d584aa…`, 40 rows, strict (fit-local) norms, FROZEN-v1
         # evidence basis. This row was SCAN-REGISTRY ONLY until now; L58 is the
         # site of record ⋆ and L63 is the robustness site.
         # THE EVIDENCE, so nobody re-derives this from the r² curve:
         #  · UNANIMITY: rank-1 (L58) and rank-2 (L63) hold across ALL FOUR
         #    robustness columns — native k128, native k32, raw k128, and the
         #    rebuilt-L16 hub. The DEEP CLUSTER {58, 63, 68} dominates the table
         #    outright; the pick is a within-cluster ordering, not a coin flip
         #    between regions.
         #  · THE r²/â INVERSION, THIRD INSTANCE. The r² PEAK L17 (.6237, at
         #    fractional depth .212) is SUB-NULL on â in BOTH hub columns
         #    (+.032 / +.019 against q95 floors .073 / .063). That is now three
         #    independent models — llama-3.1-70B L17, gemma3-27b L13, and this
         #    node's L17 — all inverting at ~21% FRACTIONAL DEPTH. NEVER CROWN A
         #    SITE FROM r² ALONE; the shallow-basin r² peak is a site artifact
         #    with a campaign-wide depth signature.
         #  · The correlation structure says the same thing quantitatively:
         #    r(r², â) = −.92 / −.95, r(coherence, â) = +.96 / +.97, and
         #    r(depth, coherence) = +.98 — the coherence-depth law orders these
         #    sites, the r² curve anti-orders them.
         #  · MARGIN, STATED HONESTLY: L58 over L63 is NARROW (raw k128 +.0046)
         #    but SIGN-CONSISTENT in all four columns. L63 is the robustness
         #    site precisely because the margin is thin — the pair is the read,
         #    the same two-site shape as 3.1-70B's L37 ⋆ / L43 and gemma's
         #    L38 ⋆ / L41.
         # ARCHITECTURE NOTE: LlamaConfig, UNIFORM FULL ATTENTION at every
         # layer — no local/global contrast EXISTS on this lineage, so gemma's
         # interleave question is not merely answered here, it is not askable.
         # Depth and coherence are the only structure available, and they are
         # what ordered the table.
         # VINTAGE DISTINCTION, AND IT IS NOT AN ERROR: llama-3.1-70b-instruct
         # (roster row 11) is the SAME PRETRAIN FAMILY at a DIFFERENT
         # POST-TRAINING VINTAGE — architecturally identical, 80 layers, and
         # therefore the SAME computed 12-site scan grid — and its ruled grid is
         # (37, 43) ⋆ L37, MID-DEPTH. These are two INDEPENDENT evidence-based
         # registrations, each read off its own â table; their DISAGREEMENT
         # (3.1: 37/43 mid-depth · 3.3: 58/63 deep) is a RECORDED OBSERVATION
         # about what RLHF vintage moves, not a bug in either ruling. Nothing
         # here is inherited from row 11 and nothing here should be reconciled
         # against it by hand.
         # Both 58 and 63 are ON the model's own computed 12-site scan grid
         # (12, 17, 22, 27, 32, 37, 43, 48, 53, 58, 63, 68), so the ratification
         # invariant is satisfied by construction — a curve-visited pair
         # re-ranked by â evidence, never a fiat grid.
         # THE VECTOR SIDE IS OPEN: the five site-evidence vectors are FROZEN-v1
         # SELECTION instruments and never file. A fresh FD-gated corpus-v2.1
         # L58 build is required before any 3.3-70b slot files — see
         # `read_composed_predictions.SITE_OF_RECORD`.
         "llama-3.3-70b-instruct": (58, 63),           # ⋆ L58 primary, L63 robustness
         # mixtral-8x7b-instruct-v0.1 (roster row 19, the few-wide-experts MoE):
         # ratified by Luxia 2026-07-28 from its own 12-site scan — held-out r² at
         # proc_k128, hub 8B, same read as wave-1 — peak L15 r²=.7092, clean and
         # unimodal, flanked by its two neighbouring scanned sites L13/L17.
         # THE POINT OF THE ROW, and why the site is not a surprise: its dense
         # family-mate mistral-7b-instruct-v0.3 has the SAME grid and the SAME
         # site of record (32 layers each, ⋆L15 each, identical fractional
         # depth), while the MoE's peak r² is materially higher (.7092 vs .6609)
         # — the sparsity raises the CEILING, it does not move the SITE. The raw
         # arm is healthy here and in fact beats native (.7248), unlike the Qwen
         # raw arms affected by the corpus finding.
         "mixtral-8x7b-instruct-v0.1": (13, 15, 17),   # ⋆ L15
         # qwen3-30b-a3b (roster row 18, the QWEN-FAMILY MoE — the freeze's
         # controlled family-matched sparsity comparison):
         # RULED DIRECTLY BY LUXIA 2026-07-29 (morning), AND THE PROVENANCE OF
         # THIS RULING IS NOT THE SAME AS THE THREE ROWS ABOVE. Those were desk
         # rulings under her overnight delegation 2. This one is HERS FIRST-HAND:
         # the overnight pass PARKED this node — the delegated rule requires a
         # clear dominant site and the table did not supply one (rank-1 FLIPS
         # across the robustness columns) — so the enactor hit the ambiguity
         # clause honestly, escalated with the table, and Luxia adjudicated it
         # off the parked evidence. Do not re-file this as a delegated ruling.
         # Readout of record `site_evidence_qwen3-30b-a3b_20260729-075641.json`,
         # sha `13527972…`, 40 rows / 0 problems, strict (fit-local) norms,
         # FROZEN-v1 evidence basis (corpus `a6712ca0…`). This row was
         # SCAN-REGISTRY ONLY until now: a FIRST registration, not a re-ranking,
         # so NOTHING retires and no refusal check accompanies it.
         # L38 is the site of record ⋆ and L35 is the robustness site.
         # THE CANDIDATE SET WAS AMENDED, ARITHMETICALLY AND ON THE RECORD
         # BEFORE ANY â EXISTED: the campaign rule (r²-top-2 overall ∪ r²-top-3
         # at depth ≥ .50, scanned sites only) collapsed to THREE here because
         # this node's top r² sites are ALL deep (L38 .7155 d.79 · L26 .7118
         # d.54 · L35 .6936 d.73). The desk amended it — those 3 ∪ top-2 r² at
         # depth < .50, giving L22 (.6853, d.458) and L19 (.6698, d.396) — to
         # restore the shallow/deep contrast so the r²/â inversion could be
         # TESTED rather than assumed absent. Purely arithmetic, dated, ledgered,
         # no â peeked. Sites of evidence: L19, L22, L26, L35, L38.
         # THE EVIDENCE, so nobody re-derives this from the r² curve:
         #  · THE TOP PAIR IS UNANIMOUS, THE ORDER WITHIN IT IS NOT. {L35, L38}
         #    take rank-1 and rank-2 in ALL FOUR robustness columns; rank-1
         #    FLIPS 2–2:
         #      native/k128/L14   L35 > L38 > L26 > L22 > L19   (L35 by +.0010)
         #      native/k32 /L14   L35 > L38 > L22 > L19 > L26   (L35 by +.0216)
         #      raw   /k128/L14   L38 > L35 > L22 > L26 > L19   (L38 by +.0025)
         #      native/k128/L16   L38 > L35 > L26 > L22 > L19   (L38 by +.0094)
         #    L19 is last in three of the four columns (L26 falls last in k32).
         #  · THE STRONGEST FACT AGAINST THE RULING, STATED FIRST SO IT IS NEVER
         #    DISCOVERED LATER AS A GOTCHA: the parked table's summary lines
         #    named only THREE of the four rank-1 margins (+.0010 native/L14 for
         #    L35, +.0025 raw and +.0094 rebuilt-L16 for L38). The FOURTH column,
         #    native/k32, ALSO ranks L35 first — and by +.0216, the LARGEST
         #    rank-1 margin anywhere on this table, ~21× the +.0010 the ruling
         #    reads as noise. So the flip is 2–2 on columns, not 1–3, and L35's
         #    best column beats L38's best column. Why that does not overturn
         #    L38, on the evidence rather than by deference: (a) k32 is the
         #    LOW-RANK ROBUSTNESS family, not the canonical cell — every constant
         #    of record in this campaign is native::proc_k128, and the parity
         #    gate is a k128 proof; (b) that column is the NOISIEST on the table
         #    by its own internal evidence — it is the only one whose ordering
         #    breaks depth monotonicity, dropping L26 to LAST beneath L19, which
         #    no k128 column does; (c) L38 takes â/ceiling in ALL THREE k128
         #    columns (.5186 / .4717 / .6024 against L35's .5004 / .4567 /
         #    .5651), so on the canonical family the ruling is not close.
         #  · WHAT LUXIA'S RULING RESTS ON: L38 carries the node's best coherence
         #    (.750, the maximum over the five-site FD set, against .442–.638
         #    elsewhere), its best â/ceiling (.6024, rebuilt-L16), and the
         #    depth→coherence→â law that all four ruled nodes follow —
         #    r(coherence, â) = +.909 / +.920 here. It also holds the CLEANEST FD
         #    gate on the grid (best_median_rel_error .00199 against
         #    .00579–.01606), which is independent of the â ranking and points
         #    the same way.
         #  · NO r²/â INVERSION ON THIS NODE, AND IT IS THE ONLY ONE: r(r², â) =
         #    +.579 / +.580 — mildly POSITIVE, where the inverting nodes run
         #    −.41 to −.96. All 40 rows clear their null floors; NOTHING is
         #    sub-null anywhere on this table, which is also unique among the
         #    four ruled nodes. THE READING, AND IT IS NOT "MoE IS DIFFERENT":
         #    the four inversions on record (3.1-70B L17 · gemma3-27b L13 ·
         #    3.3-70B L17 · 405B L43) are all DENSE nodes whose r² peak or
         #    near-peak sits in the SHALLOW half of the grid, so r² crowns a
         #    shallow site that â then refutes. Here the r² PEAK **IS** the deep
         #    site — L38 tops the curve from all three hub sources (.7155 8bL14 /
         #    .7015 8bL16 / .6945 8bL18) at fractional depth .792, and the whole
         #    shallow half tops out at .6853 below both deep sites. There is no
         #    shallow r² trap on this node, so there is nothing to invert. The
         #    discriminating variable is WHERE THE r² PEAK SITS, not the family
         #    or the sparsity; treat "the inversion is a dense/Llama phenomenon"
         #    as a description of the sample, never as a mechanism. (Note the
         #    tally is 3 Llama + 1 Gemma — it was never Llama-only.)
         #  · MoE FD NOISE IS PRESENT AS EXPECTED — the mixtral precedent. All
         #    5/5 gates PASS with `direction_specific = True`, sign consistency
         #    20/20 and control ratios 11.4–27.3, but the accepted eps rung is
         #    JAGGED across sites (.1 / .01 / .005 / .03 / .03) and the
         #    best-rung errors scatter .002–.016 rather than sitting on one
         #    scale. Expect that from a top-8-of-128 router and do not read a
         #    jagged rung as a failing gate.
         #  · THE CURVE IS THE FLATTEST OF THE OVERNIGHT QUARTET — r² span .1020
         #    (8bL14) against .30/.38 on the dense nodes. That is exactly the
         #    regime where r² carries the least information and the â evidence
         #    carries the most, and it is why this ruling turned on coherence and
         #    â/ceiling rather than on the curve.
         # FAMILY-MATCHED OBSERVATION, RECORDED WITH ITS CONFOUNDS NAMED: this
         # node has EXACTLY the same layer count (48) as its dense family-mate
         # qwen2.5-14b-instruct, so for once the comparison is available
         # site-for-site and not only by fractional depth — and the sites
         # DISAGREE: dense L29 (depth .604) against MoE L38 (.792). That is the
         # OPPOSITE of the mixtral row, where the MoE and its dense mate share
         # both site and depth (L15 of 32 each) and the sparsity raised only the
         # ceiling. Two MoE rows, two different answers to "does sparsity move
         # the site". CONFOUNDS, so this is an observation and not a claim: the
         # two Qwen nodes are different GENERATIONS (qwen3 vs qwen2.5), different
         # parameter scales (30B-A3B vs 14B dense) and different widths (2048 vs
         # 5120); mixtral/mistral is the better-controlled pair. What is solid is
         # that the whole Qwen family rules DEEP — .604 / .719 / .722 dense and
         # .792 here — so the MoE site is in-band for its family either way.
         # ARCHITECTURE NOTE: Qwen3MoeConfig with NO `layer_types` — uniform full
         # attention on every one of the 48 layers, confirmed per-site in the
         # readout (all five sites "full"). 128 experts, top-8 routed at EVERY
         # layer; d_model 2048. As on the Llama rows there is no local/global
         # contrast to read, so depth and coherence are the only structure
         # available — and here they are also sufficient.
         # ⚠ CROSS-NODE COMPARABILITY CAVEAT, AND IT RIDES EVERY â/ceiling NUMBER
         # THIS ROW PRODUCES: `ceiling_random_q95` is ≈ .275 here (k128, d=2048)
         # against ≈ .136 on the d=8192 nodes. That is a pure √(k/d) artifact of
         # the narrow residual stream, not a property of the model's
         # transportability, and the arithmetic checks out both ways: the width
         # ratio 8192/2048 = 4 predicts a q95 ratio of √4 = 2, and .275/2 = .1375
         # lands on the wide-node .136; within this readout the k32 family sits at
         # ≈ .148 against k128's ≈ .275, the same √(k) direction (the k check is
         # looser than the d check — .275/2 = .1375 vs .148 observed — because a
         # q95 order statistic is only asymptotically √(k/d)). NEVER compare â or
         # â/ceiling across nodes of different width without stating this.
         # Both 35 and 38 are ON the model's own computed 12-site scan grid
         # (7, 10, 13, 16, 19, 22, 26, 29, 32, 35, 38, 41 — no
         # `scan_grid_extension` needed, unlike gemma), so the ratification
         # invariant is satisfied by construction — a curve-visited pair
         # re-ranked by â evidence, never a fiat grid.
         # THE VECTOR SIDE IS OPEN: the five site-evidence vectors are FROZEN-v1
         # SELECTION instruments and never file. A fresh FD-gated corpus-v2.1
         # L38 build + L35/L38 v2.1 state banks + their hub fits are REQUIRED
         # before any qwen3 slot files — the re-bank is QUEUED, census-first. See
         # `read_composed_predictions.SITE_OF_RECORD`.
         "qwen3-30b-a3b": (35, 38),                   # ⋆ L38 primary, L35 robustness
         # --- webtext-v3 base siblings (roster rows 25/26) ----------------------
         # BOTH RULED BY LUXIA 2026-08-03 (ledger "TWO SITE RULINGS") from the
         # webtext-v3 sibling SCAN FITS — 12-site grids, raw arm, hub 8B, fitted
         # on the FROZEN `--splits-artifact` membership (920/280, 36/36 cells
         # valid each). These are the two §5 CORE adds that were SCAN-REGISTRY
         # ONLY at freeze; §5 required the ceremony inside their collection
         # window, and this is it.
         #
         # EACH GRID CARRIES ONE SITE THE CURVE NEVER VISITED, AND THAT IS THE
         # POINT OF THE RULING. Frozen §7's C4 read compares each sibling pair
         # base-vs-instruct AT THE SAME REGISTERED SITE and EXCLUDES a pair whose
         # sites differ as site-confounded (the F3 lesson). So each base row is
         # registered at its own curve pick PLUS its instruct partner's site of
         # record — a requirement of the read, ruled explicitly, never a curve
         # finding. Both extra sites are carried in `metabasis.roster` as
         # `scan_grid_extension`, which is what makes them part of the EFFECTIVE
         # scan grid and keeps the ratification invariant true rather than
         # waived: the check below is against the extension-inclusive grid, the
         # same mechanism gemma3-27b and pythia-6.9b use.
         # ⚠ NEITHER PAIR-READ SITE IS BANKED YET (the scans collected the
         # computed 12). The ruled re-collects join wave 3; until they land, a
         # default-grid fit here resolves one site the bank does not hold, which
         # fails loudly at load rather than quietly.
         #
         # llama-3.1-8b-base (row 25) — cp2 of record `488a81d1…`:
         #  · L15 ⋆ is the RIDGE + CKA peak (ridge .7224, cka_after .9781).
         #  · L13 is kept because it is the PROC-family peak (k128 AND k32) — the
         #    families disagree on this node, and the grid records both answers
         #    rather than hiding the disagreement behind one number.
         #  · L16 is the C4 pair-read site, matched to instruct `8b`'s L16 (the
         #    hub's own banked column, HUB_SITE_OF_RECORD). A 32-layer computed
         #    grid is odd-only, so no curve could have produced it.
         "llama-3.1-8b-base": (13, 15, 16),            # ⋆ L15 (L13 proc peak, L16 C4)
         # qwen2.5-7b-base (row 26) — cp2 of record `cc9d474c…`:
         #  · L20 ⋆ is the UNANIMOUS peak: every fit family agrees, .7796. This is
         #    the clean case, and it is worth naming as such beside the llama
         #    sibling's family split.
         #  · 18 and 22 are its curve bracket (the usual flanking neighbours).
         #  · L21 is the C4 pair-read site, matched to instruct `qwen-7b`'s L21.
         #  · CURVE CAVEAT: the proc family COLLAPSES over L4–L8 (shallow-half
         #    divergence); all families agree from L15 on, which is where the
         #    ruling is drawn from. Nothing shallow is registered.
         # The tuple is in the ORDER LUXIA RULED IT — bracket first, pair-read
         # site appended — and is deliberately NOT re-sorted, so a reader sees
         # the ruling's shape. Every consumer treats it as a set or iterates it.
         "qwen2.5-7b-base": (18, 20, 22, 21),         # ⋆ L20 (L21 C4 pair-read)
         # --- webtext-v3 scale add (roster row 24) ------------------------------
         # qwen2.5-72b-instruct: RULED BY LUXIA 2026-08-04 (ledger "SITE RULING")
         # from the webtext-v3 12-site SCAN FITS on this node — cp2_summary of
         # record `5f787171…`, 72/72 cells valid in BOTH arms, fitted on the
         # FROZEN `--splits-artifact` membership. This is the third and last of
         # the frozen §5 CORE adds to receive its ceremony (rows 25/26 were ruled
         # 2026-08-03); §5 requires the ceremony inside the collection window and
         # BEFORE any fit of this node is quoted, and this is it. L58 is the site
         # of record ⋆ and L63 is the robustness site.
         # THE EVIDENCE, so nobody re-derives this from one curve:
         #  · L58 IS THE PEAK ON FIVE OF THE SIX INSTRUMENTS the scan carries —
         #    all THREE native fit families, plus BOTH raw procrustes families
         #    and raw cka. A 5/6 majority ACROSS TWO ARMS is what the ruling
         #    rests on, and the count is stated rather than one family's number,
         #    because a single family is exactly what the r²/â rakes warn against.
         #  · THE SIXTH INSTRUMENT IS NATIVE CKA, AND IT PEAKS AT L63 — which is
         #    why L63 is kept rather than dropped. It is both the dissenting
         #    instrument's pick AND the bracket on a BROAD L48–L63 PLATEAU: the
         #    ⋆ sits at one end of a flat region, so the pair is the read, the
         #    same two-site shape as llama-3.1-70b's L37 ⋆ / L43,
         #    llama-3.3-70b's L58 ⋆ / L63 and gemma3-27b's L38 ⋆ / L41. The grid
         #    records the disagreement instead of hiding it behind the majority.
         #  · SHALLOW-HALF CAVEAT, NAMED: RAW RIDGE INFLATES THE SHALLOW HALF
         #    here. That is the rows-25/26 sibling divergence pattern with the
         #    ROLES SWAPPED (there it was the proc family collapsing shallow),
         #    and it is the same lesson either way — the families agree from L27
         #    on, and the ruling is drawn only from where they agree. Nothing
         #    shallow is registered.
         # THE COINCIDENCE WITH ROW 12 IS WORTH NAMING, AND IT IS A COINCIDENCE:
         # (58, 63) ⋆ L58 is EXACTLY llama-3.3-70b-instruct's registered grid,
         # arrived at independently, off this node's own scan. All three 80-layer
         # rows (11, 12, 24) share ONE computed grid, so the frozen §7 C2
         # recipe-vs-range read is site-for-site readable in any case; the
         # coincidence makes the footing against row 12 site-IDENTICAL as well as
         # grid-identical. C2's own paired sign test is scored against ROW 11,
         # whose independent ruling is (37, 43) ⋆ L37 — mid-depth, unreconciled
         # with either deep pair on purpose (the vintage disagreement recorded on
         # row 12 above). Nothing here is inherited from row 11 or row 12.
         # ⚠ AND THE DESK FLAGGED A C2 RIDER ON EXACTLY THIS: the 72B's r² TROUGH
         # sits on row 11's sites (37, 43) while its r² PEAK sits on row 12's
         # (58, 63). A C2 comparison quoted at the OTHER row's site is therefore
         # reading this node at a curve extremum of its own, in one direction or
         # the other. Site-for-site is a footing, not an immunity, and the read
         # must say which site it is quoting at.
         # Both 58 and 63 are ON the model's own computed 12-site scan grid
         # (12, 17, 22, 27, 32, 37, 43, 48, 53, 58, 63, 68), so NO
         # `scan_grid_extension` is needed and the ratification invariant is
         # satisfied by construction — unlike rows 25/26, this is a
         # curve-visited pair with nothing ruled off-grid. See
         # `read_composed_predictions.SITE_OF_RECORD`; the two registries must
         # agree.
         "qwen2.5-72b-instruct": (58, 63)}             # ⋆ L58 primary, L63 robustness

# Every model key the CLIs will accept. SITES = models whose FIT grid is fixed;
# metabasis.roster.SCAN_GRIDS = the 12-site alignment-curve scan grids (prereg §4,
# "sites from curves, never fiat"). The two registries OVERLAP by design: a scan
# node that the desk has ratified is ADDED to SITES and KEPT in SCAN_GRIDS, so the
# scan that produced its site of record stays re-derivable. Membership therefore
# reads: in SITES => the fit grid is fixed, no --src-sites/--tgt-sites needed; in
# SCAN_GRIDS only => still looking, pass the grid explicitly (`--sites` on the
# collector, `--src-sites` / `--tgt-sites` here), which also puts the grid in the
# command line where the stamp and the shell history can both see it.
MODEL_KEYS: tuple[str, ...] = tuple(sorted(set(SITES) | set(SCAN_GRIDS)))

# Import-time ratification invariant: every key in BOTH registries must have its
# fixed grid drawn from its own 12-site scan grid. A fiat grid (or a foreign
# model reusing a bank key) would otherwise fit silently — fail before any fit.
_RATIFICATION_PROBLEMS = fiat_grid_problems(SITES)
if _RATIFICATION_PROBLEMS:
    raise ValueError("SITES violates the ratification invariant:\n  "
                     + "\n  ".join(_RATIFICATION_PROBLEMS))


class FitGridError(LookupError):
    """A fit-grid lookup that must fail LOUDLY rather than resolve silently.

    Covers both failure modes the campaign has actually hit:

      * a model key with NO fixed grid (an un-ratified scan-registry node) —
        historically a bare ``KeyError('gpt2-xl')`` raised from inside
        ``run_grid``, which says nothing about what to do next;
      * a site the model's fixed grid does not contain — historically no error
        at all, because nothing checked.

    Every message names the MODEL and the requested SITE/GRID. Silent
    wrong-grid resolution is exactly the bug this class exists to prevent: the
    70B carried the retired scan-era L17 grid after its site of record moved to
    L37/L43, so any tool that did not pass ``--tgt-sites`` explicitly fit the
    wrong sites without a word (session-3 ruling, 2026-07-27).

    ``LookupError``, deliberately NOT ``KeyError``: callers that wrap a dict
    lookup in ``except KeyError`` must not swallow this.
    """


def sites_for(model: str,
              registry: Mapping[str, tuple[int, ...]] | None = None,
              ) -> tuple[int, ...]:
    """The fixed fit grid for `model` — or a loud refusal naming the model.

    `registry` defaults to `SITES`; pass a merged map (`{**SITES, **override}`)
    to resolve against command-line overrides without losing the diagnostics.
    """
    grids: Mapping[str, tuple[int, ...]] = SITES if registry is None else registry
    grid = grids.get(model)
    if grid is None:
        scan = SCAN_GRIDS.get(model)
        hint = (f"it IS in the scan registry with 12-site scan grid {scan} — "
                f"pass those sites explicitly (--src-sites/--tgt-sites on the "
                f"fit CLI, --sites on the collector); sites from curves, never "
                f"fiat" if scan is not None else
                "it is in NEITHER registry — check the bank key spelling")
        raise FitGridError(
            f"{model!r} has no fixed fit grid: {hint}. "
            f"Known fixed-grid keys: {sorted(grids)}")
    if not grid:
        raise FitGridError(
            f"{model!r}: fixed fit grid is EMPTY — a registry key with no sites "
            f"cannot resolve; fix the registry or pass the grid explicitly")
    return tuple(int(s) for s in grid)


def require_site(model: str, site: int,
                 registry: Mapping[str, tuple[int, ...]] | None = None) -> int:
    """Assert `site` is in `model`'s fixed fit grid; return it, or refuse loudly.

    For callers that carry a site of record around (readouts, ceiling rosters,
    glue scripts) and would otherwise resolve a retired or mistyped site
    against a bank that happily holds it.
    """
    grid = sites_for(model, registry)
    if int(site) not in grid:
        raise FitGridError(
            f"{model!r}: requested site L{site} is NOT in its fixed fit grid "
            f"{grid}. Either the site is retired (the 70B's L17 is — it stays "
            f"banked but must be asked for explicitly) or the registry is stale; "
            f"pass the grid explicitly if you mean to leave the registry")
    return int(site)


ARMS = ("native", "raw")
DEFAULT_ARM_ROOT = Path("outputs/battery/arms/A8_conjugation")


# ------------------------------------------------------- the basis's own strata
# THE STRATUM VOCABULARY IS DERIVED FROM THE PINNED CORPUS MANIFEST, NEVER TYPED
# (desk ruling 2026-08-03, ledger block of record; brief
# BRIEF-strata-basis-fix-2026-08-03).
#
# WHAT WENT WRONG. Both this module and the collector carried a module-level
# `STRATA` tuple in v2.1 vocabulary — ("S1","S2","S3") here, ("S1","S2","S3","S5")
# in the collector. The collector's tuple made the CP-1 spot-replay gate die with
# `KeyError: 'wikitext'` on the webtext-v3 basis (canary, 2026-08-03); this
# module's tuple was the same landmine one step further downstream and WOULD NOT
# HAVE CRASHED — it would have silently produced an EMPTY per-stratum readout on
# a v3 corpus, so `strata_carried` could never reach the frozen prereg §3.4 bar
# ("per-stratum R² carries in ≥2 of the four strata {wikitext, c4, pg19,
# stackexchange}") and EVERY v3 fit would have been marked invalid for a reason
# that is not about the data. A wrong answer that looks like a verdict.
#
# THE RULE. The basis is a PARAMETER of the campaign; its stratum names are a
# property OF THE MANIFEST and are read from it. Order is FIRST OCCURRENCE in the
# manifest's entry list, which makes the derived tuple a pure function of the
# manifest BYTES — i.e. deterministic under the `corpus_manifest_sha256` every
# stamp already carries. No name appears in this module, so a manifest with one
# stratum, four, or a vocabulary nobody has seen works by construction.
#
# BACKWARD COMPATIBILITY IS PROVEN, NOT ASSUMED. On the v2.1 fitting manifest the
# derived order is exactly ("S1","S2","S3"); on the Leg-4 S5-augmented manifest
# (v2.1 rows byte-identical, S5 appended) it is exactly ("S1","S2","S3","S5").
# See the collector's `pick_spot_ids` selftest for the byte-exact gate-identity
# proof — that gate is a certified instrument and its selection on a v2.1-shaped
# corpus provably does not move.

#: The frozen §3.4 carry bar, as a number rather than as prose in four places.
#: "per-stratum R² carries in ≥2 of the four strata" — the ≥2 is the rule; "the
#: four" is a property of the webtext-v3 basis, which is why the denominator is
#: derived and only the threshold lives here.
MIN_STRATA_CARRIED = 2

#: What this module hardcoded before 2026-08-03. DOCUMENTATION ONLY — nothing
#: reads it to make a decision, and nothing may start to. It is kept so a reader
#: of a pre-fix artifact can see which vocabulary produced it, and so the
#: backward-compatibility claim names the thing it is compatible WITH.
LEGACY_V21_STRATA: tuple[str, ...] = ("S1", "S2", "S3")


class StrataDerivationError(RuntimeError):
    """A corpus manifest cannot name its own strata — never a silent fallback.

    Raised when the manifest is empty, or an entry carries no usable `stratum`.
    Both are staging defects, and both would otherwise degrade into a per-stratum
    readout that is quietly missing rows — the failure mode the derivation
    exists to end.
    """


def derive_strata(entries: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The basis's stratum vocabulary, in FIRST-OCCURRENCE order.

    Deterministic under the manifest sha: the manifest bytes fix the entry order,
    the entry order fixes this tuple, and every stamp already records the sha.
    """
    if not entries:
        raise StrataDerivationError(
            "cannot derive strata from an EMPTY corpus manifest — the stratum "
            "vocabulary is a property of the basis, so a basis with no entries "
            "has none to read")
    seen: dict[str, None] = {}
    for i, entry in enumerate(entries):
        value = entry.get("stratum")
        if not isinstance(value, str) or not value.strip():
            text_id = entry.get("text_id", f"<entry {i}>")
            raise StrataDerivationError(
                f"{text_id}: manifest entry carries stratum={value!r}. Every "
                f"entry must name its stratum as a non-empty string — the "
                f"per-stratum readout, the stratum-preserving null and the "
                f"spot-replay pick are all keyed off it, and an entry that "
                f"names none would silently leave all three")
        seen.setdefault(value, None)
    return tuple(seen)


def stratum_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Census per stratum, keyed in the SAME first-occurrence order as
    `derive_strata` (dicts preserve insertion order), so a stamp's counts and its
    derived vocabulary can never disagree about either membership or order."""
    counts = {s: 0 for s in derive_strata(entries)}
    for entry in entries:
        counts[str(entry["stratum"])] += 1
    return counts


def manifest_entries(manifest_path: Path) -> list[dict]:
    """The manifest's entry list, IN ORDER — the one reader of the file's shape.

    Order is load-bearing here (it is what makes `derive_strata` deterministic),
    so this returns the list and never a dict keyed by text_id.
    """
    try:
        with open(manifest_path) as f:
            payload = json.load(f)
    except OSError as exc:
        raise StrataDerivationError(
            f"corpus manifest {manifest_path} cannot be read ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise StrataDerivationError(
            f"corpus manifest {manifest_path} is not valid JSON ({exc})") from exc
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise StrataDerivationError(
            f"corpus manifest {manifest_path} has no `entries` list — this is "
            f"not a corpus manifest, or its shape has changed")
    return entries


# ---------------------------------------------------------------- data model
@dataclass
class StateBank:
    """Per-text mean states for one (model, arm), all sites, aligned to text_ids."""
    model: str
    arm: str
    text_ids: list[str]
    states: dict[int, np.ndarray]        # site -> [n, hidden] float32 (raw)
    median_norms: dict[int, float]       # site -> median ||state||

    def matrix(self, site: int) -> np.ndarray:
        """Normalized (÷ site median norm) float64 states."""
        return self.states[site].astype(np.float64) / self.median_norms[site]


def save_state_bank(states_dir: Path, bank: StateBank) -> tuple[Path, Path]:
    """The one writer of the state-bank contract (T4 calls this)."""
    states_dir.mkdir(parents=True, exist_ok=True)
    npz_path = states_dir / f"states_{bank.model}_{bank.arm}.npz"
    payload: dict[str, np.ndarray] = {
        "text_ids": np.array(bank.text_ids, dtype=np.str_)}
    for site, arr in bank.states.items():
        if arr.shape[0] != len(bank.text_ids):
            raise ValueError(f"L{site}: {arr.shape[0]} rows != {len(bank.text_ids)} ids")
        payload[f"L{site}"] = arr.astype(np.float32)
    np.savez_compressed(npz_path, **payload)
    norms_path = states_dir / f"norms_{bank.model}_{bank.arm}.json"
    with open(norms_path, "w") as f:
        json.dump({f"L{s}": float(v) for s, v in bank.median_norms.items()}, f, indent=1)
    return npz_path, norms_path


def load_state_bank(states_dir: Path, model: str, arm: str) -> StateBank:
    npz_path = states_dir / f"states_{model}_{arm}.npz"
    norms_path = states_dir / f"norms_{model}_{arm}.json"
    if not npz_path.exists() or not norms_path.exists():
        raise FileNotFoundError(f"state bank missing: {npz_path} / {norms_path}")
    z = np.load(npz_path)
    with open(norms_path) as f:
        norms = {int(k[1:]): float(v) for k, v in json.load(f).items()}
    text_ids = [str(t) for t in z["text_ids"]]
    states = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    if set(states) != set(norms):
        raise ValueError(f"{npz_path}: sites {sorted(states)} != norms {sorted(norms)}")
    return StateBank(model=model, arm=arm, text_ids=text_ids,
                     states=states, median_norms=norms)


@dataclass
class Labels:
    """Corpus-manifest labels aligned to a bank's text_id order.

    `strata` is THE BASIS'S OWN VOCABULARY (`derive_strata` over the whole
    manifest, first-occurrence order) and travels with the labels so that every
    per-stratum readout downstream — `per_stratum_r2`, the per-stratum null
    envelope, `strata_carried` — indexes the same derived set, and none of them
    has to reach for a module constant that could disagree with the corpus.
    """
    stratum: np.ndarray                  # the per-row stratum, as the manifest names it
    group: np.ndarray                    # "(stratum|voice|mode)" null-permutation group
    topic: np.ndarray                    # int topic_idx, -1 where the basis has none
    s2_rank: np.ndarray                  # int rank for S2 ids, -1 otherwise
    strata: tuple[str, ...]              # the DERIVED basis vocabulary, in manifest order


def load_labels(manifest_path: Path, text_ids: list[str]) -> Labels:
    rows = manifest_entries(manifest_path)
    #  Derived over the WHOLE manifest, not over `text_ids`: the vocabulary is a
    #  property of the BASIS, so it must not shift when a bank happens to hold a
    #  subset of it (and it is what makes the derivation reproducible from the
    #  manifest sha alone).
    strata = derive_strata(rows)
    entries = {e["text_id"]: e for e in rows}
    missing = [t for t in text_ids if t not in entries]
    if missing:
        raise ValueError(f"{len(missing)} text_ids absent from manifest: {missing[:3]}…")
    strat, grp, top, s2r = [], [], [], []
    for t in text_ids:
        e = entries[t]
        strat.append(e["stratum"])
        grp.append(f"{e['stratum']}|{e['voice']}|{e['mode']}")
        top.append(e["topic_idx"] if e["topic_idx"] is not None else -1)
        #  `s2_rank` and `make_split` below carry v2.1 SPLIT vocabulary — the
        #  literal "S2" and the topic-grouped holdout rule. THAT IS DELIBERATE
        #  AND IT IS NOW BOUNDED (2026-08-03, brief BRIEF-v3-splits-wiring): the
        #  legacy derivation is the v2.1 lane and ONLY the v2.1 lane. A basis
        #  that cannot supply the legacy rule's quota (no topics, no S2 shard
        #  ranks — i.e. every webtext-v3 manifest) no longer falls through to an
        #  empty holdout: `make_split` REFUSES and names `--splits-artifact`.
        #  The v3 membership is FROZEN by `derive_webtext_splits.py` into
        #  splits.json / halves.json (prereg §2 / §6-I1) and CONSUMED below —
        #  never re-derived here.
        s2r.append(int(t.rsplit("-", 1)[1]) if e["stratum"] == "S2" else -1)
    return Labels(stratum=np.array(strat), group=np.array(grp),
                  topic=np.array(top), s2_rank=np.array(s2r), strata=strata)


#: THE CERTIFIED v2.1 SPLIT (gate-identity discipline, brief item 3). Recorded
#: 2026-08-03 by running the PRE-WIRING `make_split` on the v2.1 fitting manifest
#: of record — `corpus/fitting-v21/corpus_manifest.meta.json`, 775 entries, the
#: fit code's own A8_SEED=80 path — and re-proved by selftest 6a on every run.
#: The banked v2.1 fits were computed against THIS membership; it is an input to
#: certified results, so it must provably not move. `test_ids_sha256` is the
#: sha256 of the newline-joined held-out text_ids IN MANIFEST ORDER, which pins
#: the membership itself and not merely the mask's byte pattern.
V21_SPLIT_OF_RECORD: dict[str, Any] = {
    "manifest": "corpus/fitting-v21/corpus_manifest.meta.json",
    "manifest_sha256":
        "804996ccd7ad081cf3328cec4dc76fdadb1fc066cbdd906535119906865a4369",
    "n_entries": 775, "n_train": 598, "n_test": 177,
    "held_topics": [0, 12, 14, 18, 19],
    "split_sha256":
        "4bee46911c08d1bd21648ce11f9026350efe35584303641219dd1428640730f1",
    "test_ids_sha256":
        "d2f0554aec64e8c91bdbddcdf01505fa1c3c0f3f922a6f7a1bcdaaf7107f9887",
}


class SplitSelectionError(RuntimeError):
    """A split lane that must REFUSE rather than resolve to something plausible.

    Three failure modes, all of which the campaign can actually hit:

      * a basis the LEGACY v2.1 derivation cannot express (no topic_idx, no S2
        shard ranks — every webtext-v3 manifest). Before 2026-08-03 this held
        NOTHING out and surfaced as `n_test=0` inside a summary full of
        `valid: false`; now it names `--splits-artifact` and stops;
      * a FROZEN artifact that does not belong to the loaded corpus (its
        recorded manifest sha differs, an id does not resolve, a count
        disagrees with its own id lists);
      * an artifact whose ineligible set leaks into a realized test side —
        re-checked here rather than trusted, per the brief.

    An exception, never `sys.exit` (rake M45 rule (c)): an all-module selftest
    sweep survives it and still prints its terminal TOTAL line.
    """


def legacy_split_capacity(labels: Labels) -> tuple[int, int]:
    """(distinct topics, distinct S2 shard ranks) the LEGACY rule could draw from."""
    return (len({int(t) for t in labels.topic if t >= 0}),
            len({int(r) for r in labels.s2_rank if r >= 0}))


def make_split(labels: Labels, seed: int = A8_SEED) -> tuple[np.ndarray, np.ndarray, dict]:
    """Topic-grouped stratified split: 5/20 topics + 40/160 S2 shards held out.

    THE LEGACY (v2.1-shaped) LANE, and it stays byte-exact: the draw below is
    untouched — same seed, same two `rng.choice` calls in the same order — so the
    realized membership on the v2.1 fitting manifest of record is the certified
    input the banked fits were computed against (598/177, `split_sha256`
    `4bee4691…`; proven in selftest 6a).

    What is NEW is only the PRECONDITION: a basis that cannot supply the rule's
    quota is refused instead of silently held-out-empty. Nothing about a basis
    that CAN supply it changes.
    """
    n_topics, n_s2 = legacy_split_capacity(labels)
    if n_topics < HELD_OUT_TOPICS or n_s2 < HELD_OUT_S2:
        raise SplitSelectionError(
            f"the legacy in-code split cannot be derived on this basis: it "
            f"offers {n_topics} topic(s) (the rule holds out {HELD_OUT_TOPICS}) "
            f"and {n_s2} S2 shard rank(s) (the rule holds out {HELD_OUT_S2}). "
            f"This is the shape of EVERY webtext-v3 manifest (topic_idx is null "
            f"and no id is an S2 shard), and its split is NOT derived at fit "
            f"time — it is FROZEN by derive_webtext_splits.py per prereg §2 / "
            f"§6-I1. Pass --splits-artifact <…/splits.json> (or --half a|b with "
            f"--halves-artifact <…/halves.json>) so the fit CONSUMES the frozen "
            f"membership. Never an empty test set, never a re-derivation.")
    rng = np.random.default_rng(seed)
    topics = np.array(sorted({int(t) for t in labels.topic if t >= 0}))
    held_topics = set(rng.choice(topics, size=HELD_OUT_TOPICS, replace=False).tolist())
    s2_ranks = np.array(sorted({int(r) for r in labels.s2_rank if r >= 0}))
    held_s2 = set(rng.choice(s2_ranks, size=HELD_OUT_S2, replace=False).tolist())
    test = np.array([(int(t) in held_topics) or (int(r) in held_s2)
                     for t, r in zip(labels.topic, labels.s2_rank)])
    train = ~test
    info = {"rule": f"topic-grouped: {HELD_OUT_TOPICS}/20 topics + "
                    f"{HELD_OUT_S2}/160 S2 shards held out; seed {seed}",
            "held_topics": sorted(held_topics), "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "split_sha256": hashlib.sha256(test.tobytes()).hexdigest()}
    return train, test, info


# ------------------------------------------- THE FROZEN SPLIT ARTIFACTS (v3)
# THE v3 SPLIT IS CONSUMED, NEVER DERIVED HERE (desk brief BRIEF-v3-splits-wiring
# -2026-08-03; frozen prereg webtext-v3 §2 / §6-I1, tag `freeze/webtext-v3`).
#
# `derive_webtext_splits.py` computes the membership ONCE at freeze, from the
# two read-only full manifests, and records it in two artifacts whose shas are in
# the freeze act: splits.json (the realized 280/1200 train/test membership) and
# halves.json (the §6-I1 halving, each half carrying its OWN internal §2 split).
# This module MIRRORS those shapes below — it never reimplements the draw, and
# there is deliberately no code path here that could produce a v3 membership.
#
# WHAT IS VALIDATED BEFORE A SINGLE FIT RUNS (all refusals, no fallbacks):
#   1. the artifact's recorded manifest sha == the sha of the corpus manifest
#      this run loaded (the artifact must belong to THIS basis);
#   2. every text_id in the artifact resolves in the loaded corpus, and every
#      loaded row lands on exactly one side (never "absent => train");
#   3. the artifact's self-reported counts match its own id lists, and the
#      realized counts match the artifact (or, under an explicit row subset,
#      the coverage is recorded and named);
#   4. ZERO ineligible ids (the §2 v2.1-overlap set the artifact carries) in any
#      realized test membership — the artifact already guarantees it; the fitter
#      RE-CHECKS rather than trusts.
# The artifact's self-described rule is echoed into cp2_summary["split"], so a
# reader of a v3 fit sees which rule produced its holdout without opening staging.

#: The key `derive_webtext_splits` records the v3 corpus manifest under in
#: `basis.inputs_sha256` (it uses the input file's NAME). Only a fallback: the
#: sha is preferentially identified by matching `expected_v3_manifest_sha256`.
CORPUS_MANIFEST_NAME = "corpus_manifest.json"

#: The two §6-I1 halves, as the CLI names them -> as halves.json names them.
HalfName = Literal["a", "b"]
HALF_KEYS: dict[str, str] = {"a": "half_a", "b": "half_b"}


class ArtifactBasis(BaseModel):
    """`basis` — what the frozen draw was computed FROM. Mirrors `_basis_block`."""

    model_config = {"extra": "ignore"}

    corpus: str = ""
    prereg: str = ""
    freeze_tag: str = ""
    derivation_date: str = ""
    derived_by: str = ""
    inputs_sha256: dict[str, str] = {}
    expected_v3_manifest_sha256: str = ""

    def derived_from_manifest_sha256(self) -> tuple[str, bool]:
        """(the sha the draw ACTUALLY read, does it match the prereg identity).

        `derive_webtext_splits` WARNS rather than halts when the manifest it was
        handed is not the prereg's basis identity, and records the sha it really
        used — so the binding comparison for a fit is against THAT sha (these
        ids were drawn from THAT corpus), with the prereg-identity agreement
        reported beside it rather than conflated with it.
        """
        expected = self.expected_v3_manifest_sha256
        if expected and expected in set(self.inputs_sha256.values()):
            return expected, True
        named = self.inputs_sha256.get(CORPUS_MANIFEST_NAME)
        if named:
            return named, (not expected) or named == expected
        if len(self.inputs_sha256) == 1:
            only = next(iter(self.inputs_sha256.values()))
            return only, (not expected) or only == expected
        raise SplitSelectionError(
            f"the artifact's basis names inputs {sorted(self.inputs_sha256)} and "
            f"expected_v3_manifest_sha256={expected or '<absent>'!r}: none of them "
            f"identifies WHICH input was the corpus manifest, so the "
            f"belongs-to-this-basis check cannot be made. Refusing rather than "
            f"guessing which sha to compare against")


class OverlapBlock(BaseModel):
    """`overlap` — the §2 v2.1-overlap set: ids that are TRAIN-side BY RULE."""

    model_config = {"extra": "ignore"}

    n_shared_with_v21_S2: int = Field(ge=0)
    ledgered: Optional[int] = None
    ineligible_text_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _count_matches_list(self) -> "OverlapBlock":
        if len(self.ineligible_text_ids) != self.n_shared_with_v21_S2:
            raise ValueError(
                f"overlap block disagrees with itself: n_shared_with_v21_S2="
                f"{self.n_shared_with_v21_S2} but {len(self.ineligible_text_ids)} "
                f"ineligible ids are listed")
        if self.ledgered is not None and self.ledgered != self.n_shared_with_v21_S2:
            raise ValueError(
                f"overlap block disagrees with the LEDGERED figure "
                f"({self.n_shared_with_v21_S2} vs {self.ledgered}) — the derivation "
                f"HALTs on this, so an artifact carrying it is not of record")
        return self


class SplitRule(BaseModel):
    """`rule` of splits.json — echoed verbatim into the run's stamp."""

    model_config = {"extra": "ignore"}

    statement: str = Field(min_length=1)
    seed: Optional[int] = None
    stream: str = ""
    grouping_keys: dict[str, str] = {}
    quota_rule: str = ""
    overlap_rule: str = ""
    stream_construction: str = ""


class HalvesRule(BaseModel):
    """`rule` of halves.json — the halving stream plus each half's own stream."""

    model_config = {"extra": "ignore"}

    statement: str = Field(min_length=1)
    halving_stream: str = ""
    internal_split_streams: dict[str, str] = {}
    internal_quota: str = ""
    construction: str = ""
    independence: str = ""
    quota_rule: str = ""
    overlap_rule: str = ""
    stream_construction: str = ""


class SplitCounts(BaseModel):
    """`counts` of splits.json — the artifact's own totals, re-checked here."""

    model_config = {"extra": "ignore"}

    n_texts: int = Field(ge=0)
    n_test: int = Field(ge=0)
    n_train: int = Field(ge=0)
    per_stratum: dict[str, dict[str, Any]] = {}


class PerStratumSplit(BaseModel):
    """`per_stratum[st]` of splits.json — the realized per-stratum holdout."""

    model_config = {"extra": "ignore"}

    holdout_group_keys: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()


class InternalSplit(BaseModel):
    """A half's OWN §2 train/test split, as frozen. Never recomputed here."""

    model_config = {"extra": "ignore"}

    stream: str = ""
    test_ids: tuple[str, ...] = Field(min_length=1)
    train_ids: tuple[str, ...] = Field(min_length=1)


class HalfBlock(BaseModel):
    """`half_a` / `half_b` of halves.json: the membership + its internal split."""

    model_config = {"extra": "ignore"}

    text_ids: tuple[str, ...] = Field(min_length=1)
    internal_split: InternalSplit

    @model_validator(mode="after")
    def _internal_partitions_the_half(self) -> "HalfBlock":
        half = set(self.text_ids)
        if len(half) != len(self.text_ids):
            raise ValueError("a half repeats a text_id")
        test, train = set(self.internal_split.test_ids), set(self.internal_split.train_ids)
        if test & train:
            raise ValueError(f"{len(test & train)} id(s) on BOTH sides of a half's "
                             f"internal split")
        if test | train != half:
            raise ValueError(
                f"a half's internal split does not partition the half: "
                f"{len(half - (test | train))} half id(s) on neither side, "
                f"{len((test | train) - half)} split id(s) outside the half")
        return self


class SplitsArtifact(BaseModel):
    """splits.json — the FROZEN realized webtext-v3 train/test membership.

    A mirror of `derive_webtext_splits.build_splits_artifact`, not a
    reimplementation: every field here is READ, none is computed.
    """

    model_config = {"extra": "ignore"}

    artifact: str = ""
    basis: ArtifactBasis
    rule: SplitRule
    overlap: OverlapBlock
    counts: SplitCounts
    per_stratum: dict[str, PerStratumSplit] = {}
    test_ids: tuple[str, ...] = Field(min_length=1)
    train_ids: tuple[str, ...] = Field(min_length=1)
    #: set by the loader, never by the file
    source_path: str = ""
    sha256: str = ""

    @model_validator(mode="after")
    def _self_consistent(self) -> "SplitsArtifact":
        test, train = set(self.test_ids), set(self.train_ids)
        if len(test) != len(self.test_ids) or len(train) != len(self.train_ids):
            raise ValueError("the artifact repeats a text_id on one of its sides")
        if test & train:
            raise ValueError(f"{len(test & train)} text_id(s) on BOTH sides")
        if self.counts.n_test != len(test) or self.counts.n_train != len(train):
            raise ValueError(
                f"counts disagree with the id lists: n_test={self.counts.n_test} vs "
                f"{len(test)}, n_train={self.counts.n_train} vs {len(train)}")
        if self.counts.n_texts != len(test) + len(train):
            raise ValueError(
                f"counts.n_texts={self.counts.n_texts} != {len(test) + len(train)} ids")
        if self.per_stratum:
            union: set[str] = set()
            for block in self.per_stratum.values():
                union |= set(block.test_ids)
            if union != test:
                raise ValueError(
                    "the per-stratum holdouts do not reassemble the test side "
                    f"({len(union ^ test)} id(s) differ)")
        leaked = test & set(self.overlap.ineligible_text_ids)
        if leaked:
            raise ValueError(
                f"the artifact's OWN test side carries {len(leaked)} ineligible "
                f"id(s) (e.g. {sorted(leaked)[:3]}) — the §2 overlap rule is "
                f"violated inside the artifact itself")
        return self


class HalvesArtifact(BaseModel):
    """halves.json — the FROZEN §6-I1 halving + each half's internal §2 split."""

    model_config = {"extra": "ignore"}

    artifact: str = ""
    basis: ArtifactBasis
    rule: HalvesRule
    overlap: OverlapBlock
    counts: dict[str, Any] = {}
    half_a: HalfBlock
    half_b: HalfBlock
    #: set by the loader, never by the file
    source_path: str = ""
    sha256: str = ""

    @model_validator(mode="after")
    def _halves_are_disjoint_and_clean(self) -> "HalvesArtifact":
        a, b = set(self.half_a.text_ids), set(self.half_b.text_ids)
        if a & b:
            raise ValueError(f"the halves share {len(a & b)} text_id(s) — they are "
                             f"frozen as DISJOINT")
        ineligible = set(self.overlap.ineligible_text_ids)
        for name, block in (("half_a", self.half_a), ("half_b", self.half_b)):
            leaked = set(block.internal_split.test_ids) & ineligible
            if leaked:
                raise ValueError(
                    f"{name}: its internal test side carries {len(leaked)} "
                    f"ineligible id(s) (e.g. {sorted(leaked)[:3]}) — the §2 overlap "
                    f"rule applies INSIDE a half too")
        for key, n in (("half_a", len(a)), ("half_b", len(b))):
            recorded = self.counts.get(key)
            if isinstance(recorded, int) and recorded != n:
                raise ValueError(f"counts.{key}={recorded} != {n} listed ids")
        return self

    def half(self, half: str) -> HalfBlock:
        key = HALF_KEYS.get(half)
        if key is None:
            raise SplitSelectionError(
                f"unknown half {half!r}: the §6-I1 halving names exactly "
                f"{sorted(HALF_KEYS)}")
        return getattr(self, key)


def _load_artifact(path: Path, model: type[BaseModel], kind: str) -> Any:
    """Read a FROZEN artifact READ-ONLY, sha it, and validate its shape."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise SplitSelectionError(
            f"the frozen {kind} artifact cannot be read at {path} ({exc}). It is a "
            f"desk-side staging artifact (never git) — check the path, and never "
            f"substitute a re-derivation for it") from exc
    sha = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SplitSelectionError(
            f"the frozen {kind} artifact {path} (sha {sha[:12]}…) is not valid "
            f"JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise SplitSelectionError(
            f"the frozen {kind} artifact {path} (sha {sha[:12]}…) is not an object")
    try:
        art = model.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — pydantic's detail IS the message
        raise SplitSelectionError(
            f"the frozen {kind} artifact {path} (sha {sha[:12]}…) is not a valid "
            f"{kind} artifact: {exc}") from exc
    return art.model_copy(update={"source_path": str(path), "sha256": sha})


def load_splits_artifact(path: Path) -> SplitsArtifact:
    """splits.json, read-only, shape-validated, sha recorded."""
    return _load_artifact(path, SplitsArtifact, "splits")


def load_halves_artifact(path: Path) -> HalvesArtifact:
    """halves.json, read-only, shape-validated, sha recorded."""
    return _load_artifact(path, HalvesArtifact, "halves")


def require_basis_match(basis: ArtifactBasis, *, loaded_sha: str, manifest: Path,
                        artifact_path: str, kind: str) -> dict[str, Any]:
    """THE BELONGS-TO-THIS-BASIS GATE: the artifact's manifest sha == the loaded one."""
    derived, matches_prereg = basis.derived_from_manifest_sha256()
    if derived != loaded_sha:
        raise SplitSelectionError(
            f"the frozen {kind} artifact {artifact_path} was derived from corpus "
            f"manifest sha {derived}, but this run loaded {manifest} with sha "
            f"{loaded_sha}. A membership drawn from a different corpus is not a "
            f"split of THIS one — refusing. (Two common causes: the arm-root "
            f"manifest is the BODIES-STRIPPED meta manifest while the artifact "
            f"was derived from the full one, or the basis moved and the artifacts "
            f"were not re-derived. Neither is repairable at fit time.)")
    if not matches_prereg:
        logger.warning(
            "%s: the artifact was derived from manifest sha %s, which is NOT the "
            "prereg's basis identity %s recorded in the same artifact. The "
            "membership belongs to the corpus this run loaded (that check "
            "PASSED), but the basis identity disagreement is stamped and must be "
            "resolved by the desk before any v3 claim files",
            artifact_path, derived, basis.expected_v3_manifest_sha256)
    return {"artifact_derived_from_manifest_sha256": derived,
            "manifest_sha256_loaded": loaded_sha,
            "expected_v3_manifest_sha256": basis.expected_v3_manifest_sha256,
            "matches_prereg_basis_identity": bool(matches_prereg)}


def _require_ids_resolve(ids: Sequence[str], corpus_ids: set[str], *,
                         where: str, artifact_path: str) -> None:
    """Every id the artifact names must exist in the loaded corpus manifest."""
    missing = [i for i in ids if i not in corpus_ids]
    if missing:
        raise SplitSelectionError(
            f"{artifact_path}: {len(missing)} text_id(s) in {where} do not resolve "
            f"in the loaded corpus manifest (e.g. {missing[:3]}). The artifact "
            f"and the corpus disagree about what exists — refusing rather than "
            f"dropping ids")


def _membership_masks(row_ids: Sequence[str], test_ids: set[str], train_ids: set[str],
                      *, where: str, artifact_path: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Masks over the LOADED rows. A row on neither side is a refusal, never train."""
    unplaced = [t for t in row_ids if t not in test_ids and t not in train_ids]
    if unplaced:
        raise SplitSelectionError(
            f"{artifact_path}: {len(unplaced)} loaded row(s) appear on NEITHER side "
            f"of {where} (e.g. {unplaced[:3]}). An unplaced row would silently "
            f"become train — refusing")
    test = np.array([t in test_ids for t in row_ids], dtype=bool)
    return ~test, test


def _integrity_recheck(realized_test_ids: Sequence[str], ineligible: set[str], *,
                       where: str, artifact_path: str) -> dict[str, Any]:
    """Item 4: ZERO ineligible ids in a realized test membership. Re-checked."""
    leaked = sorted(set(realized_test_ids) & ineligible)
    if leaked:
        raise SplitSelectionError(
            f"{artifact_path}: {len(leaked)} ineligible id(s) reached the realized "
            f"test side of {where} (e.g. {leaked[:3]}). The §2 v2.1-overlap rule "
            f"makes these TRAIN-side by rule — no held-out quantity is ever "
            f"computed on a text a prior result touched")
    return {"rule": "PREREG §2 v2.1-overlap: ineligible ids are TRAIN-side by rule",
            "n_ineligible_declared": len(ineligible),
            "n_ineligible_in_test": 0,
            "checked": where}


def membership_sha256(test_ids: Sequence[str]) -> str:
    """The sha256 of the realized held-out MEMBERSHIP (ordered text_ids).

    `split_sha256` (the mask's bytes) is kept unchanged because banked v2.1
    summaries carry it — but a mask is a pattern over WHICHEVER rows were
    loaded, so two different memberships can share one. The two §6-I1 halves are
    exactly that case: same shape, disjoint rows. This sha names the ids
    themselves, so a reader can never confuse one half's split with the other's.
    """
    return hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitPlan:
    """The resolved lane: which rows to fit on, and their train/test masks.

    `keep` indexes the rows AS LOADED (the half lane restricts them); `train`
    and `test` index the KEPT rows, which is the order `run_pair_arm` wants.
    """
    keep: np.ndarray                     # bool [n_loaded]
    train: np.ndarray                    # bool [n_kept]
    test: np.ndarray                     # bool [n_kept]
    info: dict[str, Any]


def _coverage_block(row_ids: Sequence[str], artifact_test: set[str],
                    artifact_train: set[str], subset_reason: Optional[str],
                    realized_test: int, realized_train: int) -> dict[str, Any]:
    """The counts-match read: realized vs the artifact's own, subsetting named."""
    full = set(row_ids) == (artifact_test | artifact_train)
    block = {"artifact_n_test": len(artifact_test),
             "artifact_n_train": len(artifact_train),
             "realized_n_test": realized_test, "realized_n_train": realized_train,
             "covers_the_whole_artifact": bool(full),
             "row_subset_reason": subset_reason}
    if full and (realized_test != len(artifact_test)
                 or realized_train != len(artifact_train)):
        raise SplitSelectionError(
            f"realized counts {realized_train}/{realized_test} (train/test) "
            f"disagree with the artifact's {len(artifact_train)}/"
            f"{len(artifact_test)} while covering it exactly — refusing")
    if not full and subset_reason is None:
        raise SplitSelectionError(
            f"the loaded rows do not cover the artifact's membership "
            f"({len(row_ids)} rows vs {len(artifact_test | artifact_train)} ids) "
            f"and NO subset was requested. A partial corpus silently reweights "
            f"every held-out number — refusing")
    return block


def split_from_artifact(artifact: SplitsArtifact, row_ids: Sequence[str],
                        corpus_ids: set[str], *, manifest: Path,
                        manifest_sha256: str,
                        subset_reason: Optional[str] = None) -> SplitPlan:
    """THE `--splits-artifact` LANE: membership READ from splits.json."""
    basis = require_basis_match(artifact.basis, loaded_sha=manifest_sha256,
                                manifest=manifest,
                                artifact_path=artifact.source_path, kind="splits")
    _require_ids_resolve(artifact.test_ids + artifact.train_ids, corpus_ids,
                         where="splits.json", artifact_path=artifact.source_path)
    test_ids, train_ids = set(artifact.test_ids), set(artifact.train_ids)
    train, test = _membership_masks(row_ids, test_ids, train_ids,
                                    where="splits.json",
                                    artifact_path=artifact.source_path)
    realized_test = [t for t, m in zip(row_ids, test) if m]
    integrity = _integrity_recheck(realized_test,
                                   set(artifact.overlap.ineligible_text_ids),
                                   where="the main split's test side",
                                   artifact_path=artifact.source_path)
    coverage = _coverage_block(row_ids, test_ids, train_ids, subset_reason,
                               int(test.sum()), int(train.sum()))
    info = {
        "source": "splits-artifact",
        "lane": "the FROZEN webtext-v3 main split, CONSUMED (never re-derived)",
        "artifact": artifact.source_path,
        "artifact_sha256": artifact.sha256,
        "artifact_kind": artifact.artifact,
        "half": None,
        "rule": artifact.rule.statement,
        "artifact_rule": artifact.rule.model_dump(),
        "basis": {**artifact.basis.model_dump(exclude={"inputs_sha256"}), **basis},
        "artifact_counts": artifact.counts.model_dump(),
        "coverage": coverage,
        "overlap_integrity": integrity,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "split_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "test_ids_sha256": membership_sha256(realized_test),
    }
    return SplitPlan(keep=np.ones(len(row_ids), dtype=bool), train=train, test=test,
                     info=info)


def split_from_half(artifact: HalvesArtifact, half: HalfName,
                    row_ids: Sequence[str],
                    corpus_ids: set[str], *, manifest: Path, manifest_sha256: str,
                    subset_reason: Optional[str] = None) -> SplitPlan:
    """THE `--half a|b` LANE: the half's membership AND its own internal split.

    Both come from halves.json. Nothing is recomputed — not the halving, not the
    internal train/test draw — and the half identity travels in the stamp.
    """
    basis = require_basis_match(artifact.basis, loaded_sha=manifest_sha256,
                                manifest=manifest,
                                artifact_path=artifact.source_path, kind="halves")
    block = artifact.half(half)
    key = HALF_KEYS[half]
    _require_ids_resolve(block.text_ids, corpus_ids, where=f"{key}.text_ids",
                         artifact_path=artifact.source_path)
    half_ids = set(block.text_ids)
    keep = np.array([t in half_ids for t in row_ids], dtype=bool)
    kept = [t for t, k in zip(row_ids, keep) if k]
    if not kept:
        raise SplitSelectionError(
            f"{artifact.source_path}: NONE of the {len(row_ids)} loaded rows are in "
            f"{key} — the state bank and the halving do not describe the same "
            f"corpus")
    missing_rows = [t for t in block.text_ids if t not in set(row_ids)]
    if missing_rows and subset_reason is None:
        raise SplitSelectionError(
            f"{artifact.source_path}: {len(missing_rows)} id(s) of {key} are absent "
            f"from the loaded state bank (e.g. {missing_rows[:3]}) and no subset "
            f"was requested — a half fitted on part of itself is not the frozen "
            f"half; refusing")
    test_ids = set(block.internal_split.test_ids)
    train_ids = set(block.internal_split.train_ids)
    train, test = _membership_masks(kept, test_ids, train_ids,
                                    where=f"{key}.internal_split",
                                    artifact_path=artifact.source_path)
    realized_test = [t for t, m in zip(kept, test) if m]
    integrity = _integrity_recheck(realized_test,
                                   set(artifact.overlap.ineligible_text_ids),
                                   where=f"{key}'s internal test side",
                                   artifact_path=artifact.source_path)
    reason = subset_reason or (f"restricted to {key} (PREREG §6-I1)"
                               if len(kept) < len(row_ids) else None)
    coverage = _coverage_block(kept, test_ids, train_ids,
                               subset_reason if missing_rows else None,
                               int(test.sum()), int(train.sum()))
    other = artifact.half("b" if half == "a" else "a")
    info = {
        "source": "halves-artifact",
        "lane": (f"the FROZEN §6-I1 {key}: membership AND its internal §2 split, "
                 f"both CONSUMED (never recomputed)"),
        "artifact": artifact.source_path,
        "artifact_sha256": artifact.sha256,
        "artifact_kind": artifact.artifact,
        "half": half,
        "half_key": key,
        "half_stream": artifact.rule.internal_split_streams.get(key, ""),
        "halving_stream": artifact.rule.halving_stream,
        "n_half_texts": len(block.text_ids),
        "n_half_rows_loaded": len(kept),
        "disjoint_from_other_half": not (half_ids & set(other.text_ids)),
        "row_restriction": reason,
        "rule": artifact.rule.statement,
        "artifact_rule": artifact.rule.model_dump(),
        "basis": {**artifact.basis.model_dump(exclude={"inputs_sha256"}), **basis},
        "artifact_counts": artifact.counts.get("internal", {}).get(key, {}),
        "coverage": coverage,
        "overlap_integrity": integrity,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "split_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "test_ids_sha256": membership_sha256(realized_test),
    }
    return SplitPlan(keep=keep, train=train, test=test, info=info)


def resolve_split_plan(labels: Labels, row_ids: Sequence[str], corpus_ids: set[str],
                       *, manifest: Path, manifest_sha256: str,
                       splits: Optional[SplitsArtifact] = None,
                       halves: Optional[HalvesArtifact] = None,
                       half: Optional[HalfName] = None,
                       subset_reason: Optional[str] = None,
                       seed: int = A8_SEED) -> SplitPlan:
    """THE ONE PLACE A SPLIT IS CHOSEN. Three lanes, no fallback between them."""
    if half is not None:
        if halves is None:
            raise SplitSelectionError(
                f"--half {half} needs the frozen halves artifact: pass "
                f"--halves-artifact <…/halves.json>. The §6-I1 halving is FROZEN "
                f"and is never recomputed at fit time")
        if splits is not None:
            raise SplitSelectionError(
                "--splits-artifact and --half are two different memberships (the "
                "whole corpus vs one half with its own internal split) — pass one, "
                "never both")
        return split_from_half(halves, half, row_ids, corpus_ids, manifest=manifest,
                               manifest_sha256=manifest_sha256,
                               subset_reason=subset_reason)
    if splits is not None:
        return split_from_artifact(splits, row_ids, corpus_ids, manifest=manifest,
                                   manifest_sha256=manifest_sha256,
                                   subset_reason=subset_reason)
    if halves is not None:
        raise SplitSelectionError(
            "--halves-artifact was given without --half a|b: which half is the "
            "fit? Never guessed")
    train, test, info = make_split(labels, seed=seed)      # the LEGACY v2.1 lane
    info = {"source": "legacy-derivation",
            "lane": ("the in-code v2.1 topic-grouped derivation — the legacy lane, "
                     "kept byte-exact for the banked v2.1 fits' certified split"),
            "half": None, "artifact": None, "artifact_sha256": None,
            "manifest_sha256_loaded": manifest_sha256, **info,
            "test_ids_sha256": membership_sha256(
                [t for t, m in zip(row_ids, test) if m])}
    return SplitPlan(keep=np.ones(len(row_ids), dtype=bool), train=train, test=test,
                     info=info)


class TrainExclusion(BaseModel):
    """A ruled set of text_ids removed from the TRAIN side, and nothing else.

    THE ONE THING THIS IS FOR (frozen webtext-v3 prereg §6-I5's *beside*): "the
    same computation excluding the 59 shared wikitext texts' influence (legs
    refit on the shared-text-free train subset) — the overlap-clean form". The
    held-out membership is UNTOUCHED — the beside asks what the maps look like
    when the shared texts never entered the fit, not what a different held-out
    set would score. An id that lands on the test side is therefore a REFUSAL,
    not a silent removal: it would mean the loaded split is not the one the
    exclusion was derived against.

    NO FIT MATH CHANGES. `run_pair_arm` already takes its train rows as a mask;
    this only decides which mask it gets, and puts the exclusion's sha in the
    stamp so an overlap-clean object can never be read as a primary one.
    """

    model_config = {"frozen": True}

    source_path: str
    sha256: str = Field(description="sha256 of the exclusion artifact's BYTES")
    artifact: str = Field(description="the artifact's self-declared kind")
    clause: str = Field(description="the frozen clause this beside serves")
    excluded_text_ids: tuple[str, ...]
    n_excluded: int

    @model_validator(mode="after")
    def _counts_agree(self) -> "TrainExclusion":
        if len(set(self.excluded_text_ids)) != len(self.excluded_text_ids):
            raise ValueError(f"{self.source_path}: duplicate ids in the exclusion set")
        if self.n_excluded != len(self.excluded_text_ids):
            raise ValueError(
                f"{self.source_path}: n_excluded={self.n_excluded} but the list "
                f"carries {len(self.excluded_text_ids)} ids")
        if not self.excluded_text_ids:
            raise ValueError(f"{self.source_path}: an EMPTY exclusion set is not a "
                             f"beside — refusing rather than banking a duplicate "
                             f"of the primary under a beside name")
        return self


def load_train_exclusion(path: Path) -> TrainExclusion:
    """Read + digest an exclusion artifact. The sha is of the BYTES on disk."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SplitSelectionError(f"{path}: cannot read the exclusion set ({exc})") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SplitSelectionError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(doc, Mapping):
        raise SplitSelectionError(f"{path}: the exclusion artifact is not an object")
    try:
        return TrainExclusion(
            source_path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            artifact=str(doc.get("artifact", "")),
            clause=str(doc.get("clause", "")),
            excluded_text_ids=tuple(str(t) for t in doc.get("excluded_text_ids", ())),
            n_excluded=int(doc.get("n_excluded", -1)))
    except (TypeError, ValueError) as exc:
        raise SplitSelectionError(f"{path}: not a usable exclusion set ({exc})") from exc


def apply_train_exclusion(plan: SplitPlan, kept_row_ids: Sequence[str],
                          exclusion: TrainExclusion) -> SplitPlan:
    """Drop the exclusion's ids from the TRAIN mask. Test side untouched.

    Refuses on either integrity failure, because both mean the exclusion and the
    loaded split describe different objects: an excluded id that resolves to no
    loaded row, or one that resolves to a HELD-OUT row.
    """
    excluded = set(exclusion.excluded_text_ids)
    index = {t: i for i, t in enumerate(kept_row_ids)}
    missing = sorted(excluded - set(index))
    if missing:
        raise SplitSelectionError(
            f"{exclusion.source_path}: {len(missing)} excluded id(s) resolve to no "
            f"loaded row (e.g. {missing[:3]}) — the exclusion set and this corpus "
            f"are not the same object; refusing")
    mask = np.zeros(len(kept_row_ids), dtype=bool)
    for text_id in excluded:
        mask[index[text_id]] = True
    on_test = sorted(t for t in excluded if bool(plan.test[index[t]]))
    if on_test:
        raise SplitSelectionError(
            f"{exclusion.source_path}: {len(on_test)} excluded id(s) are HELD OUT in "
            f"this split (e.g. {on_test[:3]}). This beside removes TRAIN influence "
            f"only; removing a test row would change the held-out population and "
            f"make the two readings incomparable — refusing")
    n_before = int(plan.train.sum())
    train = plan.train & ~mask
    n_after = int(train.sum())
    if n_after != n_before - len(excluded):
        raise SplitSelectionError(
            f"{exclusion.source_path}: train went {n_before} -> {n_after} while "
            f"excluding {len(excluded)} ids — some excluded id was neither train "
            f"nor test; refusing")
    if n_after <= 1:
        raise SplitSelectionError(
            f"{exclusion.source_path}: the exclusion leaves {n_after} train row(s)")
    info = dict(plan.info)
    info["n_train"] = n_after
    info["train_exclusion"] = {
        "rule": ("ids removed from the TRAIN mask only; the held-out membership, "
                 "the corpus and every fit path are otherwise unchanged"),
        "artifact": exclusion.source_path,
        "artifact_sha256": exclusion.sha256,
        "artifact_kind": exclusion.artifact,
        "clause": exclusion.clause,
        "n_excluded": exclusion.n_excluded,
        "n_train_before": n_before,
        "n_train_after": n_after,
        "all_excluded_were_train_side": True,
    }
    return SplitPlan(keep=plan.keep, train=train, test=plan.test, info=info)


def split_identity(info: Mapping[str, Any]) -> dict[str, Any]:
    """The identity a fits/ directory is stamped with — what must not silently mix.

    `train_exclusion_sha256` is part of the identity: an overlap-clean object and
    a primary object have the same pair, sites, arm and held-out membership, and
    differ ONLY in which train rows the map saw. Without this key the guard would
    let one overwrite the other in place. Absent on every pre-2026-08-04 summary
    and on every run with no exclusion, where it reads None on both sides.
    """
    return {"source": info.get("source"), "half": info.get("half"),
            "artifact_sha256": info.get("artifact_sha256"),
            "split_sha256": info.get("split_sha256"),
            "test_ids_sha256": info.get("test_ids_sha256"),
            "train_exclusion_sha256":
                (info.get("train_exclusion") or {}).get("artifact_sha256")}


def guard_fits_dir(fits_dir: Path, info: Mapping[str, Any]) -> None:
    """Refuse to overwrite a fits/ directory banked under a DIFFERENT split.

    Half A's maps and half B's summary in one directory is a silent mixture, and
    a v3 run landing on the default `fits/` would overwrite banked v2.1 objects.
    Same identity (a rerun) passes; anything else names --fits-dirname.
    """
    prior_path = fits_dir / "cp2_summary.json"
    if not prior_path.exists():
        return
    try:
        prior = json.loads(prior_path.read_text()).get("split", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("%s exists but cannot be read (%s) — the split-identity "
                       "guard cannot compare against it", prior_path, exc)
        return
    now, before = split_identity(info), split_identity(prior)
    if before["source"] is None and now["source"] == "legacy-derivation":
        return                        # a pre-2026-08-03 summary; the same lane
    if before == now:
        return
    raise SplitSelectionError(
        f"{fits_dir} already holds fits banked under a DIFFERENT split "
        f"({before}) than this run's ({now}). Writing here would mix two "
        f"memberships in one directory (and overwrite the banked maps). Pass "
        f"--fits-dirname <name> for this column")


# ---------------------------------------------------------------- fit families
@dataclass
class PCABank:
    mean: np.ndarray                     # [d]
    components: np.ndarray               # [kmax, d] rows = PCs
    explained: np.ndarray                # [kmax]

    @classmethod
    def fit(cls, x_train: np.ndarray, kmax: int) -> "PCABank":
        mu = x_train.mean(axis=0)
        xc = x_train - mu
        kmax = min(kmax, min(xc.shape) - 1)
        # economy SVD: n_train x d with n < d
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        var = s ** 2 / (xc.shape[0] - 1)
        return cls(mean=mu, components=vt[:kmax], explained=var[:kmax] / var.sum())


@dataclass
class TransportMap:
    """g: source hidden space -> target hidden space (linear part, for vectors).

    Vector transport is scale-calibrated (norm ratios folded in) but every read in
    Phase C is cosine-based, so calibration never affects a pass/fail.
    """
    kind: Literal["proc", "ridge"]
    src_norm: float
    tgt_norm: float
    # procrustes
    va: Optional[np.ndarray] = None      # [k, d_a]
    vb: Optional[np.ndarray] = None      # [k, d_b]
    omega: Optional[np.ndarray] = None   # [k, k], orthogonal
    scale: float = 1.0
    # ridge (factored: B = left @ right, d_a x d_b)
    left: Optional[np.ndarray] = None    # [d_a, r]
    right: Optional[np.ndarray] = None   # [r, d_b]

    def transport(self, v: np.ndarray, direction: Literal["fwd", "rev"] = "fwd"
                  ) -> np.ndarray:
        """Map a direction vector across. 'rev' is the adjoint (exact inverse of the
        orthogonal part for Procrustes; transpose map for ridge — 'where defined')."""
        if self.kind == "proc":
            assert self.va is not None and self.vb is not None and self.omega is not None
            if direction == "fwd":
                out = (v / self.src_norm) @ self.va.T @ self.omega @ self.vb
                return out * self.scale * self.tgt_norm
            out = (v / self.tgt_norm) @ self.vb.T @ self.omega.T @ self.va
            return out / self.scale * self.src_norm
        assert self.left is not None and self.right is not None
        if direction == "fwd":
            return ((v / self.src_norm) @ self.left) @ self.right * self.tgt_norm
        return ((v / self.tgt_norm) @ self.right.T) @ self.left.T * self.src_norm


def fit_procrustes(za: np.ndarray, zb: np.ndarray) -> tuple[np.ndarray, float]:
    """min ||za @ omega - zb||_F over orthogonal omega, + isotropic scale."""
    m = za.T @ zb
    u, s, vt = np.linalg.svd(m)
    omega = u @ vt
    denom = float((za ** 2).sum())
    scale = float(s.sum() / denom) if denom > 0 else 1.0
    return omega, scale


@dataclass
class RidgeSVD:
    """Ridge from centered X to centered Y via one SVD of X_train, alpha-sweepable."""
    mu_x: np.ndarray
    mu_y: np.ndarray
    u: np.ndarray                        # [n, r]
    s: np.ndarray                        # [r]
    vt: np.ndarray                       # [r, d_a]
    uty: np.ndarray                      # [r, d_b] = U^T Yc

    @classmethod
    def prep(cls, x: np.ndarray, y: np.ndarray) -> "RidgeSVD":
        mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
        u, s, vt = np.linalg.svd(x - mu_x, full_matrices=False)
        return cls(mu_x=mu_x, mu_y=mu_y, u=u, s=s, vt=vt, uty=u.T @ (y - mu_y))

    def press(self, alpha: float, y: np.ndarray) -> float:
        """Closed-form multi-output LOO PRESS for this alpha."""
        f = self.s ** 2 / (self.s ** 2 + alpha)               # [r]
        resid = (y - self.mu_y) - self.u @ (f[:, None] * self.uty)
        h = np.einsum("ij,j,ij->i", self.u, f, self.u)        # hat diagonal
        h = np.clip(h, 0.0, 1.0 - 1e-8)
        return float(((resid / (1.0 - h)[:, None]) ** 2).sum())

    def choose_alpha(self, y: np.ndarray) -> float:
        base = float((self.s ** 2).mean())
        grid = base * np.logspace(-5, 1, 13)
        scores = [self.press(a, y) for a in grid]
        return float(grid[int(np.argmin(scores))])

    def predict(self, x_new: np.ndarray, alpha: float,
                uty: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict Y for new X; pass a permuted uty to realize nulls without refits."""
        f = self.s / (self.s ** 2 + alpha)
        proj = (x_new - self.mu_x) @ self.vt.T                # [m, r]
        return self.mu_y + (proj * f) @ (uty if uty is not None else self.uty)

    def factors(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """B = left @ right with B = V f(S) U^T Yc  (for TransportMap)."""
        f = self.s / (self.s ** 2 + alpha)
        return self.vt.T * f, self.uty


# ---------------------------------------------------------------- metrics
def r2_score(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: np.ndarray) -> float:
    sse = float(((y_true - y_pred) ** 2).sum())
    sst = float(((y_true - y_train_mean) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def per_stratum_r2(y_true, y_pred, y_train_mean, strata: np.ndarray,
                   strata_order: Sequence[str]) -> dict[str, float]:
    """Held-out R² per stratum, keyed over the BASIS-DERIVED vocabulary.

    `strata_order` is `Labels.strata` — the manifest's own first-occurrence
    order — so the keys of this dict, of the null envelope beside it and of
    `strata_carried` are the same set in the same order on every basis. A
    stratum with no row in this test slice is OMITTED (it has no R² to report),
    exactly as before; what changed is only where the vocabulary comes from.
    """
    return {s: r2_score(y_true[strata == s], y_pred[strata == s], y_train_mean)
            for s in strata_order if (strata == s).any()}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    num = float(np.linalg.norm(yc.T @ xc, "fro") ** 2)
    den = (np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro"))
    return num / float(den) if den > 0 else float("nan")


def null_permutations(labels: Labels, train: np.ndarray, rng: np.random.Generator,
                      kind: Literal["shuffled", "stratum"]) -> np.ndarray:
    """A permutation of TRAIN target rows (indices into the train subset)."""
    n = int(train.sum())
    if kind == "shuffled":
        return rng.permutation(n)
    perm = np.arange(n)
    groups = labels.group[train]
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


# ---------------------------------------------------------------- per-fit records
class FitRecord(BaseModel):
    site_pair: str                       # e.g. "3bL14->8bL16"
    arm: str
    family: str                          # "proc_k32" | ... | "ridge"
    n_train: int
    n_test: int
    r2: float
    r2_null_shuffled_q95: float
    r2_null_stratum_q95: float
    per_stratum_r2: dict[str, float]
    per_stratum_null_q95: dict[str, float]   # max of the two nulls' q95, per stratum
    strata_carried: list[str]
    cka_before: float
    cka_after: float
    valid: bool
    detail: dict = {}


def evaluate_fit(name: str, predict_fn, refit_predict_fn, x, y, train, test,
                 labels: Labels, perms: dict[str, list[np.ndarray]]) -> FitRecord:
    """predict_fn(x_test) -> y_pred using the true fit; refit_predict_fn(perm, x_test)
    -> y_pred under a permuted-train refit. perms = {'shuffled': [...], 'stratum': [...]}."""
    y_train_mean = y[train].mean(axis=0)
    y_pred = predict_fn(x[test])
    r2 = r2_score(y[test], y_pred, y_train_mean)
    strata_test = labels.stratum[test]
    #  THE DERIVED SET, everywhere: per-stratum R², the per-stratum null
    #  envelope and the carry list all range over `labels.strata` (the pinned
    #  manifest's own vocabulary), so on webtext-v3 the frozen §3.4 denominator
    #  is the four named strata and on a v2.1 manifest it is S1/S2/S3(+S5).
    basis = tuple(labels.strata)
    ps_r2 = per_stratum_r2(y[test], y_pred, y_train_mean, strata_test, basis)

    null_overall: dict[str, list[float]] = {"shuffled": [], "stratum": []}
    null_ps: dict[str, dict[str, list[float]]] = {
        "shuffled": {s: [] for s in basis}, "stratum": {s: [] for s in basis}}
    for kind in ("shuffled", "stratum"):
        for perm in perms[kind]:
            yp = refit_predict_fn(perm, x[test])
            null_overall[kind].append(r2_score(y[test], yp, y_train_mean))
            for s, v in per_stratum_r2(y[test], yp, y_train_mean, strata_test,
                                       basis).items():
                null_ps[kind][s].append(v)
    q = lambda vals: float(np.quantile(vals, 0.95)) if vals else float("nan")
    ps_null_q95 = {s: max(q(null_ps["shuffled"][s]), q(null_ps["stratum"][s]))
                   for s in basis}
    carried = [s for s in basis
               if s in ps_r2 and np.isfinite(ps_null_q95[s]) and ps_r2[s] > ps_null_q95[s]]
    shuf_q95, strat_q95 = q(null_overall["shuffled"]), q(null_overall["stratum"])
    #  Frozen prereg §3.4: "per-stratum R² carries in >= 2 of the four strata".
    #  The BAR is 2 and is frozen; "the four" is the derived basis's size, which
    #  is 4 on webtext-v3 exactly as frozen and 3-or-4 on v2.1 manifests.
    valid = ((r2 > shuf_q95) and (r2 > strat_q95)
             and (len(carried) >= MIN_STRATA_CARRIED))
    pair, arm, family = name.split("::")
    return FitRecord(
        site_pair=pair, arm=arm, family=family,
        n_train=int(train.sum()), n_test=int(test.sum()),
        r2=round(r2, 4), r2_null_shuffled_q95=round(shuf_q95, 4),
        r2_null_stratum_q95=round(strat_q95, 4),
        per_stratum_r2={k: round(v, 4) for k, v in ps_r2.items()},
        per_stratum_null_q95={k: round(v, 4) for k, v in ps_null_q95.items()},
        strata_carried=carried,
        cka_before=round(linear_cka(x[test], y[test]), 4),
        cka_after=round(linear_cka(y_pred, y[test]), 4),
        valid=valid)


# ---------------------------------------------------------------- the grid
def run_pair_arm(src_bank: StateBank, tgt_bank: StateBank, s_site: int, t_site: int,
                 labels: Labels, train: np.ndarray, test: np.ndarray,
                 rng: np.random.Generator, k_grid=K_GRID, n_null=N_NULL_REPS
                 ) -> tuple[list[FitRecord], dict[str, TransportMap]]:
    x = src_bank.matrix(s_site)
    y = tgt_bank.matrix(t_site)
    pair = f"{src_bank.model}L{s_site}->{tgt_bank.model}L{t_site}"
    arm = src_bank.arm
    perms = {kind: [null_permutations(labels, train, rng, kind) for _ in range(n_null)]
             for kind in ("shuffled", "stratum")}
    records: list[FitRecord] = []
    maps: dict[str, TransportMap] = {}

    kmax = int(min(max(k_grid), train.sum() - 1))
    pca_a = PCABank.fit(x[train], kmax)
    pca_b = PCABank.fit(y[train], kmax)
    za_full = (x - pca_a.mean) @ pca_a.components.T        # [n, kmax]
    zb_full = (y - pca_b.mean) @ pca_b.components.T

    for k in k_grid:
        k_eff = min(k, kmax)
        za, zb = za_full[:, :k_eff], zb_full[:, :k_eff]
        omega, scale = fit_procrustes(za[train], zb[train])

        def predict(x_test, *, _o=omega, _s=scale, _k=k_eff):
            zt = (x_test - pca_a.mean) @ pca_a.components[:_k].T
            return (zt @ _o) * _s @ pca_b.components[:_k] + pca_b.mean

        def refit_predict(perm, x_test, *, _k=k_eff):
            o, s = fit_procrustes(za[train], zb[train][perm])
            zt = (x_test - pca_a.mean) @ pca_a.components[:_k].T
            return (zt @ o) * s @ pca_b.components[:_k] + pca_b.mean

        name = f"{pair}::{arm}::proc_k{k}"
        rec = evaluate_fit(name, predict, refit_predict, x, y, train, test, labels, perms)
        rec.detail = {"k_effective": k_eff, "scale": round(scale, 5),
                      "pca_explained_src": round(float(pca_a.explained[:k_eff].sum()), 4),
                      "pca_explained_tgt": round(float(pca_b.explained[:k_eff].sum()), 4)}
        records.append(rec)
        maps[f"proc_k{k}"] = TransportMap(
            kind="proc", src_norm=src_bank.median_norms[s_site],
            tgt_norm=tgt_bank.median_norms[t_site],
            va=pca_a.components[:k_eff], vb=pca_b.components[:k_eff],
            omega=omega, scale=scale)
        logger.info("%s  R2=%.3f nulls(q95)=%.3f/%.3f carried=%s valid=%s",
                    name, rec.r2, rec.r2_null_shuffled_q95, rec.r2_null_stratum_q95,
                    rec.strata_carried, rec.valid)

    ridge = RidgeSVD.prep(x[train], y[train])
    alpha = ridge.choose_alpha(y[train])

    def r_predict(x_test):
        return ridge.predict(x_test, alpha)

    def r_refit_predict(perm, x_test):
        yc_perm = (y[train][perm] - y[train][perm].mean(axis=0))
        return ridge.predict(x_test, alpha, uty=ridge.u.T @ yc_perm)

    name = f"{pair}::{arm}::ridge"
    rec = evaluate_fit(name, r_predict, r_refit_predict, x, y, train, test, labels, perms)
    rec.detail = {"alpha": alpha, "rank": int(len(ridge.s))}
    records.append(rec)
    left, right = ridge.factors(alpha)
    maps["ridge"] = TransportMap(
        kind="ridge", src_norm=src_bank.median_norms[s_site],
        tgt_norm=tgt_bank.median_norms[t_site], left=left, right=right)
    logger.info("%s  R2=%.3f nulls(q95)=%.3f/%.3f carried=%s valid=%s alpha=%.3g",
                name, rec.r2, rec.r2_null_shuffled_q95, rec.r2_null_stratum_q95,
                rec.strata_carried, rec.valid, alpha)
    return records, maps


def arm_agreement(maps_by_arm: dict[str, dict[str, TransportMap]], d_src: int,
                  rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """cos between native-arm and raw-arm transported images of shared unit probes."""
    probes = rng.standard_normal((N_PROBES, d_src))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    out: dict[str, dict[str, float]] = {}
    arms = list(maps_by_arm)
    if len(arms) < 2:
        return out
    for fam in maps_by_arm[arms[0]]:
        t1 = np.stack([maps_by_arm[arms[0]][fam].transport(p) for p in probes])
        t2 = np.stack([maps_by_arm[arms[1]][fam].transport(p) for p in probes])
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
        cos = (t1 * t2).sum(axis=1)
        out[fam] = {"mean_cos": round(float(cos.mean()), 4),
                    "min_cos": round(float(cos.min()), 4),
                    "n_probes": N_PROBES}
    return out


def save_transport_map(fits_dir: Path, pair: str, arm: str, fam: str,
                       tm: TransportMap) -> Path:
    path = fits_dir / f"fit_{pair.replace('->', '__')}_{arm}_{fam}.npz"
    payload: dict[str, np.ndarray] = {
        "kind": np.array(tm.kind), "src_norm": np.array(tm.src_norm),
        "tgt_norm": np.array(tm.tgt_norm), "scale": np.array(tm.scale)}
    for key in ("va", "vb", "omega", "left", "right"):
        val = getattr(tm, key)
        if val is not None:
            payload[key] = val.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_transport_map(path: Path) -> TransportMap:
    z = np.load(path)
    get = lambda k: z[k].astype(np.float64) if k in z.files else None
    return TransportMap(kind=str(z["kind"]), src_norm=float(z["src_norm"]),
                        tgt_norm=float(z["tgt_norm"]), scale=float(z["scale"]),
                        va=get("va"), vb=get("vb"), omega=get("omega"),
                        left=get("left"), right=get("right"))


def subset_bank(bank: StateBank, keep: np.ndarray) -> StateBank:
    """Row-subset a bank. median_norms stay the COLLECTION values (stamped)."""
    return StateBank(model=bank.model, arm=bank.arm,
                     text_ids=[t for t, k in zip(bank.text_ids, keep) if k],
                     states={s: a[keep] for s, a in bank.states.items()},
                     median_norms=bank.median_norms)


def subset_labels(labels: Labels, keep: np.ndarray) -> Labels:
    """Row-subset the labels. `strata` is CARRIED THROUGH UNCHANGED.

    The derived vocabulary is the BASIS's, not the slice's: with `--fit-strata
    S1,S2` the readout still ranges over the whole basis, so a stratum the fit
    excluded reports as absent rather than silently redefining the denominator
    the §3.4 carry rule is read against. (This is also what keeps the v2.1
    `--fit-strata` behaviour byte-identical to the pre-derivation code, which
    ranged over the module constant.)
    """
    return Labels(stratum=labels.stratum[keep], group=labels.group[keep],
                  topic=labels.topic[keep], s2_rank=labels.s2_rank[keep],
                  strata=labels.strata)


def run_grid(arm_root: Path, src_model: str, tgt_model: str,
             n_null: int = N_NULL_REPS, fit_strata: Optional[list[str]] = None,
             fits_dirname: str = "fits", k_grid=K_GRID,
             sites_override: dict[str, tuple[int, ...]] | None = None,
             arms: tuple[str, ...] = ARMS,
             splits_artifact: Optional[Path] = None,
             halves_artifact: Optional[Path] = None,
             half: Optional[HalfName] = None,
             exclude_train_ids: Optional[Path] = None) -> dict:
    states_dir = arm_root / "states"
    fits_dir = arm_root / fits_dirname
    fits_dir.mkdir(parents=True, exist_ok=True)
    manifest = arm_root / "corpus" / "corpus_manifest.json"

    #  THE BASIS, READ ONCE AND UP FRONT (desk ruling 2026-08-03). Derived here
    #  as well as inside `load_labels` so the run's stamp can name the
    #  vocabulary even if the grid loop never executes, and so a basis that
    #  cannot satisfy the frozen carry rule is announced BEFORE the fits rather
    #  than inferred from a summary full of `valid: false`.
    basis_entries = manifest_entries(manifest)
    basis_strata = derive_strata(basis_entries)
    basis_counts = stratum_counts(basis_entries)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    logger.info("basis: %d strata %s from %s (sha %s…), counts %s",
                len(basis_strata), list(basis_strata), manifest.name,
                manifest_sha[:12], basis_counts)
    if len(basis_strata) < MIN_STRATA_CARRIED:
        logger.warning(
            "basis carries %d stratum/strata %s — the frozen prereg §3.4 carry "
            "rule needs >= %d, so NO fit from this basis can be marked valid. "
            "The fits are still computed and banked; read them beside this "
            "line, and do not read `valid: false` as a statement about the maps",
            len(basis_strata), list(basis_strata), MIN_STRATA_CARRIED)

    #  THE SPLIT LANE, RESOLVED ONCE AND UP FRONT (brief BRIEF-v3-splits-wiring
    #  -2026-08-03). The artifacts are read and validated against THIS corpus
    #  before any bank is loaded, so a membership that does not belong to this
    #  basis refuses in the first second of the run rather than after the grid.
    splits: Optional[SplitsArtifact] = (
        load_splits_artifact(splits_artifact) if splits_artifact else None)
    halves: Optional[HalvesArtifact] = (
        load_halves_artifact(halves_artifact) if halves_artifact else None)
    corpus_ids = {str(e["text_id"]) for e in basis_entries}
    for art, label in ((splits, "splits"), (halves, "halves")):
        if art is not None:
            require_basis_match(art.basis, loaded_sha=manifest_sha,
                                manifest=manifest, artifact_path=art.source_path,
                                kind=label)
            logger.info("frozen %s artifact: %s (sha %s…) — derived from manifest "
                        "%s…, MATCHES this run's corpus", label, art.source_path,
                        art.sha256[:12], manifest_sha[:12])
    if half is not None:
        logger.info("half lane: %s (PREREG §6-I1) — membership and internal split "
                    "both read from the artifact", HALF_KEYS.get(half, half))

    #  THE TRAIN EXCLUSION, READ AND DIGESTED BEFORE THE FIRST FIT (sealed-stamp
    #  discipline, as the pair wave). Read here rather than inside the loop so a
    #  malformed or absent exclusion refuses in the first second, and so the sha
    #  that will stamp every object is known before any state bank is opened.
    exclusion: Optional[TrainExclusion] = (
        load_train_exclusion(exclude_train_ids) if exclude_train_ids else None)
    if exclusion is not None:
        logger.info("train exclusion: %s (sha %s) — %d id(s) removed from the TRAIN "
                    "side only; this is a LABELED BESIDE, never the primary",
                    exclusion.source_path, exclusion.sha256, exclusion.n_excluded)

    all_records: list[FitRecord] = []
    agreement: dict[str, dict] = {}
    split_info: dict = {}
    # Resolve BOTH grids up front and loudly: a missing key here used to surface
    # as a bare KeyError from the middle of the loop, and a stale registry entry
    # surfaced as nothing at all (it just fit the wrong sites).
    sites_map = {**SITES, **(sites_override or {})}
    src_sites = sites_for(src_model, sites_map)
    tgt_sites = sites_for(tgt_model, sites_map)
    logger.info("fit grid: %s %s -> %s %s", src_model, src_sites, tgt_model,
                tgt_sites)
    for s_site in src_sites:
        for t_site in tgt_sites:
            maps_by_arm: dict[str, dict[str, TransportMap]] = {}
            for arm in arms:
                src = load_state_bank(states_dir, src_model, arm)
                tgt = load_state_bank(states_dir, tgt_model, arm)
                if src.text_ids != tgt.text_ids:
                    raise RuntimeError(f"text_id order mismatch {src_model}/{tgt_model} ({arm})")
                labels = load_labels(manifest, src.text_ids)
                subset_reason: Optional[str] = None
                if fit_strata:
                    keep = np.isin(labels.stratum, fit_strata)
                    src, tgt = subset_bank(src, keep), subset_bank(tgt, keep)
                    labels = subset_labels(labels, keep)
                    subset_reason = f"--fit-strata {','.join(fit_strata)}"
                plan = resolve_split_plan(labels, list(src.text_ids), corpus_ids,
                                          manifest=manifest,
                                          manifest_sha256=manifest_sha,
                                          splits=splits, halves=halves, half=half,
                                          subset_reason=subset_reason)
                if not bool(plan.keep.all()):
                    #  The half lane restricts the rows BEFORE fitting; the masks
                    #  it returns already index the kept rows.
                    src, tgt = subset_bank(src, plan.keep), subset_bank(tgt, plan.keep)
                    labels = subset_labels(labels, plan.keep)
                if exclusion is not None:
                    #  AFTER the keep-subset: `plan.train`/`plan.test` index the
                    #  KEPT rows, and `src.text_ids` is now exactly that list.
                    plan = apply_train_exclusion(plan, list(src.text_ids), exclusion)
                train, test, split_info = plan.train, plan.test, plan.info
                guard_fits_dir(fits_dir, split_info)
                rng = np.random.default_rng(A8_SEED + s_site * 100 + t_site)
                recs, maps = run_pair_arm(src, tgt, s_site, t_site, labels,
                                          train, test, rng, k_grid=k_grid,
                                          n_null=n_null)
                all_records.extend(recs)
                maps_by_arm[arm] = maps
                pair = f"{src_model}L{s_site}->{tgt_model}L{t_site}"
                for fam, tm in maps.items():
                    save_transport_map(fits_dir, pair, arm, fam, tm)
            d_src = next(iter(maps_by_arm.values()))["ridge"].left.shape[0]
            pair = f"{src_model}L{s_site}->{tgt_model}L{t_site}"
            agreement[pair] = arm_agreement(
                maps_by_arm, d_src, np.random.default_rng(A8_SEED))

    summary = {
        "arm": "A8_conjugation", "leg": 0, "builder": "fit_transport_maps.py",
        "prereg_tag": "prereg-arm8-v1",
        "pair": f"{src_model}->{tgt_model}",
        # ── THE SPLIT, AND WHERE IT CAME FROM ────────────────────────────────
        # `split` carries the whole lane record: source, the artifact's path and
        # sha, the half identity, the artifact's own rule echoed verbatim, the
        # coverage read and the re-checked overlap integrity. The three keys
        # beside it are the same identity hoisted to the top level, because every
        # downstream reader that resolves fits BY PATH from this file needs to
        # see WHICH membership banked them without walking into `split`.
        "split": split_info, "n_null_reps": n_null, "k_grid": list(k_grid),
        "split_source": split_info.get("source"),
        "half": split_info.get("half"),
        "splits_artifact_sha256": split_info.get("artifact_sha256"),
        # ── THE TRAIN EXCLUSION, HOISTED (frozen webtext-v3 §6-I5's beside) ───
        # None on every primary object. Non-None means these maps never saw the
        # excluded rows, so they are a LABELED BESIDE and are not interchangeable
        # with the primary tree's objects — hoisted beside the split identity for
        # the same reason those keys are: a reader resolving fits BY PATH must be
        # able to see it without walking into `split`.
        "train_exclusion_sha256":
            (split_info.get("train_exclusion") or {}).get("artifact_sha256"),
        "train_exclusion_n": (split_info.get("train_exclusion") or {}).get("n_excluded"),
        "fit_strata": fit_strata, "arms": list(arms),
        "null_group_key": "(stratum|voice|mode)",
        # ── THE STRATUM BASIS (desk ruling 2026-08-03) ───────────────────────
        # Derived from the pinned manifest, never typed. Recorded here because
        # `per_stratum_r2` / `per_stratum_null_q95` / `strata_carried` in every
        # record below are keyed by it, and a reader of those keys must be able
        # to see WHICH basis named them without opening the corpus.
        "strata_basis": {
            "rule": "corpus_manifest.json per-entry `stratum`, first-occurrence "
                    "order (deterministic under the manifest sha)",
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha,
            "strata": list(basis_strata),
            "n_strata": len(basis_strata),
            "counts": basis_counts,
            "carry_rule": f"per-stratum R2 carries in >= {MIN_STRATA_CARRIED} "
                          f"of the {len(basis_strata)} derived strata "
                          f"(frozen prereg webtext-v3 §3.4)",
            "carry_rule_satisfiable": len(basis_strata) >= MIN_STRATA_CARRIED,
        },
        "normalization": "per-site median norm at fit time (norms in state banks)",
        "records": [r.model_dump() for r in all_records],
        "two_arm_g_agreement": agreement,
        "n_valid": sum(r.valid for r in all_records),
        "n_fits": len(all_records),
        # ── THE EFFECTIVE THREAD CONFIGURATION (Luxia ruling 2026-08-01) ──────
        # Every map this run banked passed through `np.linalg.svd` — PCABank.fit,
        # fit_procrustes and RidgeSVD.prep — and LAPACK's SVD is a blocked GEMM
        # whose summation order depends on the thread count, exactly as eigh's
        # does. So the count is part of these fits' identity, and this sidecar is
        # where a reader of `fit_*.npz` finds it: the map npz's are bare arrays by
        # design, and cp2_summary.json is the run's stamp.
        #
        # UNCONDITIONAL: an absent key means the summary predates 2026-08-01, and
        # `metabasis.threads.stamp_thread_config` is the reader that says so.
        THREAD_STAMP_KEY: thread_config_stamp(),
    }
    out = fits_dir / "cp2_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    logger.info("cp2 summary: %s  (%d/%d fits valid)", out,
                summary["n_valid"], summary["n_fits"])
    return summary


# ---------------------------------------------------------------- selftest
def _synthetic_world(rng, n_latent=12, d_a=64, d_b=96, noise=0.15, paired=True):
    """Paired clouds from a shared latent with stratum/topic cluster structure."""
    entries = []
    for voice in ("3b", "8b"):
        for mode in ("expository", "explanatory", "argumentative", "conversational"):
            for t in range(20):
                entries.append(("S1", voice, mode, t))
    for r in range(80):
        entries.append(("S2", "neutral", "neutral", -1))
    for voice in ("3b", "8b"):
        for mode in ("linear", "socratic", "contrastive", "dialectical", "analogical"):
            for t in range(12):
                entries.append(("S3", voice, mode, t))
    n = len(entries)
    group_keys = sorted({f"{s}|{v}|{m}" for s, v, m, _ in entries})
    gmeans = {g: rng.standard_normal(n_latent) * 1.2 for g in group_keys}
    tmeans = rng.standard_normal((20, n_latent)) * 0.8
    z = np.zeros((n, n_latent))
    strat, grp, top, s2r = [], [], [], []
    s2_rank = 0
    for i, (s, v, m, t) in enumerate(entries):
        g = f"{s}|{v}|{m}"
        z[i] = gmeans[g] + (tmeans[t] if t >= 0 else 0) + rng.standard_normal(n_latent)
        strat.append(s); grp.append(g); top.append(t)
        s2r.append(s2_rank if s == "S2" else -1)
        s2_rank += (s == "S2")
    a_map = np.linalg.qr(rng.standard_normal((d_a, n_latent)))[0].T  # [r, d_a]
    b_map = np.linalg.qr(rng.standard_normal((d_b, n_latent)))[0].T
    x = z @ a_map + noise * rng.standard_normal((n, d_a))
    # unpaired control: keep GROUP cluster geometry but break every finer
    # correspondence (fresh within-cluster noise AND permuted topic effects) —
    # the world where only the stratum-preserving null's structure survives.
    topic_perm = rng.permutation(20)
    z2 = z if paired else np.array(
        [gmeans[g] + (tmeans[topic_perm[t]] if t >= 0 else 0)
         + rng.standard_normal(n_latent)
         for g, t in zip(grp, top)])
    y = z2 @ b_map + noise * rng.standard_normal((n, d_b))
    v_lat = np.zeros(n_latent); v_lat[3] = 1.0
    #  The synthetic world's labels are DERIVED from its own rows, exactly as a
    #  real run derives them from a manifest — so the selftest exercises the
    #  derivation rather than restating a vocabulary beside it.
    labels = Labels(stratum=np.array(strat), group=np.array(grp),
                    topic=np.array(top), s2_rank=np.array(s2r),
                    strata=derive_strata([{"stratum": s, "text_id": f"t{i}"}
                                          for i, s in enumerate(strat)]))
    banks = (
        StateBank("srcM", "native", [f"t{i}" for i in range(n)], {1: x.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(x, axis=1)))}),
        StateBank("tgtM", "native", [f"t{i}" for i in range(n)], {1: y.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(y, axis=1)))}))
    return banks, labels, (v_lat @ a_map, v_lat @ b_map)


def _selftest_split(labels, rng):
    topics = np.array(sorted({int(t) for t in labels.topic if t >= 0}))
    held_t = set(rng.choice(topics, size=max(2, len(topics) // 4), replace=False).tolist())
    s2 = np.array(sorted({int(r) for r in labels.s2_rank if r >= 0}))
    held_s = set(rng.choice(s2, size=max(2, len(s2) // 4), replace=False).tolist())
    test = np.array([(int(t) in held_t) or (int(r) in held_s)
                     for t, r in zip(labels.topic, labels.s2_rank)])
    return ~test, test


def _basis_world(rng, strata_names: Sequence[str], n_per: int = 40,
                 d_a: int = 48, d_b: int = 56, noise: float = 0.15):
    """A paired world whose strata are named by the CALLER — any vocabulary.

    Used to prove that the per-stratum machinery is vocabulary-free: the same
    generator produces a v2.1-shaped basis, the webtext-v3 four-stratum basis and
    a single-stratum basis, and nothing in this module knows any of their names.
    Rows are emitted stratum-block by stratum-block, which is the shape both real
    manifests have, so the derived first-occurrence order is `strata_names`.
    """
    n_latent = 10
    rows = [(s, t) for s in strata_names for t in range(n_per)]
    n = len(rows)
    gmeans = {s: rng.standard_normal(n_latent) * 1.2 for s in strata_names}
    tmeans = rng.standard_normal((20, n_latent)) * 0.8
    z = np.zeros((n, n_latent))
    strat, grp, top = [], [], []
    for i, (s, t) in enumerate(rows):
        topic = t % 20
        z[i] = gmeans[s] + tmeans[topic] + rng.standard_normal(n_latent)
        strat.append(s)
        grp.append(f"{s}|neutral|neutral")      # the real v3 manifest's group key
        top.append(topic)
    a_map = np.linalg.qr(rng.standard_normal((d_a, n_latent)))[0].T
    b_map = np.linalg.qr(rng.standard_normal((d_b, n_latent)))[0].T
    x = z @ a_map + noise * rng.standard_normal((n, d_a))
    y = z @ b_map + noise * rng.standard_normal((n, d_b))
    entries = [{"text_id": f"{s}-{i:04d}", "stratum": s}
               for i, (s, _) in enumerate(rows)]
    labels = Labels(stratum=np.array(strat), group=np.array(grp),
                    topic=np.array(top), s2_rank=np.full(n, -1),
                    strata=derive_strata(entries))
    ids = [e["text_id"] for e in entries]
    banks = (
        StateBank("srcM", "native", ids, {1: x.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(x, axis=1)))}),
        StateBank("tgtM", "native", ids, {1: y.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(y, axis=1)))}))
    return banks, labels, entries


def _topic_split(labels: Labels, rng) -> tuple[np.ndarray, np.ndarray]:
    """A topic-grouped train/test split for a basis with NO S2 shard ranks.

    `_selftest_split` reaches for `s2_rank`, which only a v2.1-shaped basis has —
    a live demonstration of the NAMED GAP recorded in `load_labels`: the SPLIT
    machinery still carries v2.1 vocabulary and is a separate question from the
    stratum vocabulary this change derives. The blocks below need a split, not a
    ruling about splits, so they take this one.
    """
    topics = np.array(sorted({int(t) for t in labels.topic if t >= 0}))
    held = set(rng.choice(topics, size=max(2, len(topics) // 4),
                          replace=False).tolist())
    test = np.array([int(t) in held for t in labels.topic])
    return ~test, test


def _raises(fn, exc: type[BaseException] = Exception) -> bool:
    """True iff `fn()` raises `exc` — a refusal asserted, never merely expected."""
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001 — the WRONG exception is not a pass
        return False
    return False


def _ok(fn) -> bool:
    try:
        fn()
        return True
    except Exception:  # noqa: BLE001
        return False


#: The four webtext-v3 stratum names appear ONLY in selftest fixtures — never in
#: the module's logic (the vocabulary is basis-derived). They are here so the
#: fixtures have the real corpus's shape.
_V3_FIXTURE_STRATA: tuple[str, ...] = ("wikitext", "c4", "pg19", "stackexchange")


def _v3_fixture_rows(n_per: int = 8,
                     strata: Sequence[str] = _V3_FIXTURE_STRATA) -> list[dict]:
    """A webtext-v3-SHAPED manifest: no topic_idx, no S2 shard ranks.

    Exactly the shape on which the legacy in-code derivation would have held
    NOTHING out, which is the defect this wiring closes.
    """
    return [{"text_id": f"{s}-{i:03d}", "stratum": s, "voice": "neutral",
             "mode": "neutral", "topic_idx": None}
            for s in strata for i in range(n_per)]


def _write_fixture_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.write_text(json.dumps({"entries": list(rows)}, indent=1))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _splits_fixture(manifest_sha: str, ids: Sequence[str], test_ids: Sequence[str],
                    ineligible: Sequence[str] = ()) -> dict:
    """A splits.json fixture in the REAL artifact's shape (mirrors the builder)."""
    test = list(test_ids)
    train = [i for i in ids if i not in set(test)]
    per_stratum: dict[str, dict[str, Any]] = {}
    for tid in test:
        st = tid.rsplit("-", 1)[0]
        per_stratum.setdefault(st, {"holdout_group_keys": [], "test_ids": []})
        per_stratum[st]["test_ids"].append(tid)
        per_stratum[st]["holdout_group_keys"].append(tid)
    return {
        "artifact": "splits.json — the realized webtext-v3 train/test membership",
        "basis": {"corpus": "webtext-v3", "prereg": "PREREG §2 (fixture)",
                  "freeze_tag": "freeze/webtext-v3",
                  "derivation_date": "2026-08-03",
                  "derived_by": "metabasis/scripts/derive_webtext_splits.py",
                  "inputs_sha256": {CORPUS_MANIFEST_NAME: manifest_sha,
                                    "v21.json": "1" * 64},
                  "expected_v3_manifest_sha256": manifest_sha},
        "rule": {"statement": "fixture holdout, per-stratum, grouped, seed 80",
                 "seed": 80, "stream": "split-v3",
                 "grouping_keys": {s: "fixture" for s in _V3_FIXTURE_STRATA},
                 "quota_rule": "exact-quota grouped draw (fixture)",
                 "overlap_rule": "PREREG §2 v2.1-overlap rule (fixture)",
                 "stream_construction": "SeedSequence(entropy=80, spawn_key=…)"},
        "overlap": {"n_shared_with_v21_S2": len(ineligible),
                    "ledgered": len(ineligible),
                    "ineligible_text_ids": sorted(ineligible)},
        "counts": {"n_texts": len(test) + len(train), "n_test": len(test),
                   "n_train": len(train),
                   "per_stratum": {s: {"n_test": len(d["test_ids"])}
                                   for s, d in per_stratum.items()}},
        "per_stratum": per_stratum,
        "test_ids": sorted(test), "train_ids": sorted(train)}


def _halves_fixture(manifest_sha: str, ids: Sequence[str], half_a: Sequence[str],
                    a_test: Sequence[str], b_test: Sequence[str],
                    ineligible: Sequence[str] = ()) -> dict:
    """A halves.json fixture: disjoint halves, each with its OWN internal split."""
    a = list(half_a)
    b = [i for i in ids if i not in set(a)]
    def block(members: Sequence[str], test: Sequence[str], stream: str) -> dict:
        return {"text_ids": sorted(members),
                "internal_split": {
                    "stream": stream, "test_ids": sorted(test),
                    "train_ids": sorted(i for i in members if i not in set(test))}}
    return {
        "artifact": "halves.json — the PREREG §6-I1 corpus half-split",
        "basis": {"corpus": "webtext-v3", "prereg": "PREREG §6-I1 (fixture)",
                  "freeze_tag": "freeze/webtext-v3",
                  "derivation_date": "2026-08-03",
                  "derived_by": "metabasis/scripts/derive_webtext_splits.py",
                  "inputs_sha256": {CORPUS_MANIFEST_NAME: manifest_sha},
                  "expected_v3_manifest_sha256": manifest_sha},
        "rule": {"statement": "two disjoint exhaustive halves; each takes the §2 "
                              "split rule internally (fixture)",
                 "halving_stream": "halfsplit-v3",
                 "internal_split_streams": {"half_a": "split-v3-half-a",
                                            "half_b": "split-v3-half-b"},
                 "internal_quota": "the §2 rate applied to the half",
                 "construction": "half A drawn, half B the complement",
                 "independence": "named substreams (fixture)",
                 "quota_rule": "exact-quota grouped draw (fixture)",
                 "overlap_rule": "PREREG §2 v2.1-overlap rule (fixture)",
                 "stream_construction": "SeedSequence(entropy=80, spawn_key=…)"},
        "overlap": {"n_shared_with_v21_S2": len(ineligible),
                    "ineligible_text_ids": sorted(ineligible)},
        "counts": {"half_a": len(a), "half_b": len(b),
                   "internal": {"half_a": {"n_test": len(a_test),
                                           "n_train": len(a) - len(a_test)},
                                "half_b": {"n_test": len(b_test),
                                           "n_train": len(b) - len(b_test)}}},
        "half_a": block(a, a_test, "split-v3-half-a"),
        "half_b": block(b, b_test, "split-v3-half-b")}


def selftest(splits_artifact: Optional[Path] = None,
             halves_artifact: Optional[Path] = None) -> int:
    rng = np.random.default_rng(0)
    failures: list[str] = []
    checks: list[str] = []
    skips: list[str] = []

    def check(cond: bool, msg: str):
        status = "PASS" if cond else "FAIL"
        checks.append(msg)
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    def skip(name: str, why: str) -> None:
        """RAKE M44: a block this CONFIGURATION cannot run, named and counted.

        A third state beside pass and fail — nothing was asserted and found
        wanting — so it never contributes to the exit code, and the reason names
        the triggering condition rather than merely noting an absence.
        """
        skips.append(name)
        checks.append(f"SKIPPED: {name}")
        print(f"  [SKIP] {name} — {why}")

    #  RAKE M44: the count below is only readable beside the configuration that
    #  produced it. The axis that matters for THIS suite is the data tree: the
    #  v2.1 fitting manifest is a repo artifact and is absent from a deployed
    #  code tree, so the block that reads it is availability-branched.
    real_v21 = Path(str(V21_SPLIT_OF_RECORD["manifest"]))
    have_splits = splits_artifact is not None and Path(splits_artifact).exists()
    have_halves = halves_artifact is not None and Path(halves_artifact).exists()
    print(f"[config] cwd={Path.cwd()}  numpy={np.__version__}  "
          f"v2.1 fitting manifest {'PRESENT' if real_v21.exists() else 'ABSENT'} "
          f"at {real_v21}  frozen splits.json "
          f"{'PRESENT' if have_splits else 'ABSENT'}  frozen halves.json "
          f"{'PRESENT' if have_halves else 'ABSENT'}")

    print("== selftest 1: planted paired world (recovery expected) ==")
    (src, tgt), labels, (v_a, v_b) = _synthetic_world(rng, paired=True)
    train, test = _selftest_split(labels, rng)
    recs, maps = run_pair_arm(src, tgt, 1, 1, labels, train, test,
                              np.random.default_rng(1), k_grid=(8, 16), n_null=10)
    by_fam = {r.family.split("::")[-1]: r for r in recs}
    for fam, r in by_fam.items():
        check(r.valid, f"{fam}: VALID (R2={r.r2}, nulls q95 "
                       f"{r.r2_null_shuffled_q95}/{r.r2_null_stratum_q95}, "
                       f"carried={r.strata_carried})")
    check(all(r.r2 > 0.5 for r in recs), "all true-fit R2 > 0.5")
    check(all(r.r2_null_stratum_q95 >= r.r2_null_shuffled_q95 - 0.05 for r in recs),
          "stratum-preserving null >= shuffled null (sharper, as designed)")
    for fam in ("proc_k16", "ridge"):
        tv = maps[fam].transport(v_a)
        cos = float(tv @ v_b / (np.linalg.norm(tv) * np.linalg.norm(v_b)))
        check(cos > 0.8, f"{fam}: planted-axis transport cos={cos:.3f} > 0.8")
        rv = maps[fam].transport(v_b, direction="rev")
        rcos = float(rv @ v_a / (np.linalg.norm(rv) * np.linalg.norm(v_a)))
        check(rcos > 0.8, f"{fam}: reverse transport cos={rcos:.3f} > 0.8")

    print("== selftest 2: unpaired noise world (validity must NOT fire) ==")
    (src_n, tgt_n), labels_n, _ = _synthetic_world(
        np.random.default_rng(7), paired=False)
    train_n, test_n = _selftest_split(labels_n, np.random.default_rng(7))
    recs_n, _ = run_pair_arm(src_n, tgt_n, 1, 1, labels_n, train_n, test_n,
                             np.random.default_rng(2), k_grid=(8, 16), n_null=10)
    for r in recs_n:
        fam = r.family
        check(not r.valid, f"{fam}: correctly INVALID on unpaired clouds "
                           f"(R2={r.r2} vs stratum-null q95={r.r2_null_stratum_q95})")

    print("== selftest 3: two-arm agreement on a shared world ==")
    (src2, tgt2), labels2, _ = _synthetic_world(np.random.default_rng(0), paired=True)
    src2.arm = tgt2.arm = "raw"
    pert = np.random.default_rng(11)
    src2.states[1] = src2.states[1] + 0.05 * pert.standard_normal(src2.states[1].shape).astype(np.float32)
    tgt2.states[1] = tgt2.states[1] + 0.05 * pert.standard_normal(tgt2.states[1].shape).astype(np.float32)
    _, maps2 = run_pair_arm(src2, tgt2, 1, 1, labels2, train, test,
                            np.random.default_rng(3), k_grid=(8, 16), n_null=2)
    agree = arm_agreement({"native": maps, "raw": maps2}, d_src=64,
                          rng=np.random.default_rng(A8_SEED))
    # NOTE: when k exceeds the effective latent rank (here 12), the extra PCs are
    # noise directions whose Procrustes rotation is arbitrary per-arm — agreement
    # degrades in that subspace BY DESIGN (the read doubles as a rank diagnostic).
    for fam, a in agree.items():
        check(a["mean_cos"] > 0.75, f"{fam}: two-arm agreement mean_cos={a['mean_cos']}")

    print("== selftest 4: state-bank round-trip ==")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="a8_selftest_") as td:
        p1, p2 = save_state_bank(Path(td), src)
        back = load_state_bank(Path(td), "srcM", "native")
        check(back.text_ids == src.text_ids, "text_ids round-trip")
        check(np.allclose(back.states[1], src.states[1]), "states round-trip (fp32)")
        check(abs(back.median_norms[1] - src.median_norms[1]) < 1e-9, "norms round-trip")

    print("== selftest 5: THE STRATUM VOCABULARY IS DERIVED, NEVER TYPED ==")
    #  The defect this closes (canary, 2026-08-03): a hardcoded v2.1 tuple here
    #  would have produced an EMPTY per-stratum readout on the webtext-v3 basis,
    #  so the frozen §3.4 carry rule could never be met and every v3 fit would
    #  have been marked invalid for a reason that is not about the data.
    v21_rows = ([{"text_id": f"S1-{i}", "stratum": "S1"} for i in range(4)]
                + [{"text_id": f"S2-{i}", "stratum": "S2"} for i in range(3)]
                + [{"text_id": f"S3-{i}", "stratum": "S3"} for i in range(2)])
    v21_s5_rows = v21_rows + [{"text_id": f"S5-{i}", "stratum": "S5"}
                              for i in range(2)]
    v3_rows = [{"text_id": f"{s}-{i}", "stratum": s}
               for s in ("wikitext", "c4", "pg19", "stackexchange")
               for i in range(3)]
    check(derive_strata(v21_rows) == LEGACY_V21_STRATA,
          f"a v2.1-shaped manifest derives {LEGACY_V21_STRATA} — exactly the "
          f"tuple this module used to hardcode (backward compatibility, proven)")
    check(derive_strata(v21_s5_rows) == ("S1", "S2", "S3", "S5"),
          "the Leg-4 S5-augmented manifest derives S1,S2,S3,S5 — the appended "
          "stratum enters the readout instead of being silently dropped, which "
          "is what the old ('S1','S2','S3') constant did to it")
    check(derive_strata(v3_rows) == ("wikitext", "c4", "pg19", "stackexchange"),
          "the webtext-v3 basis derives its four strata IN MANIFEST ORDER — the "
          "frozen prereg §3.4 denominator, read from the corpus rather than "
          "restated here")
    check(derive_strata([{"text_id": "only-0", "stratum": "only"}]) == ("only",),
          "a SINGLE-stratum basis derives one stratum and works by construction")
    check(derive_strata([{"text_id": "b", "stratum": "beta"},
                         {"text_id": "a", "stratum": "alpha"},
                         {"text_id": "b2", "stratum": "beta"}])
          == ("beta", "alpha"),
          "an UNKNOWN vocabulary derives in FIRST-OCCURRENCE order (never "
          "sorted): the order is a property of the manifest bytes, which is what "
          "makes it deterministic under the manifest sha")
    check(list(stratum_counts(v21_s5_rows).items())
          == [("S1", 4), ("S2", 3), ("S3", 2), ("S5", 2)],
          "the census is keyed in the SAME derived order, so a stamp's counts "
          "and its vocabulary cannot disagree about membership or order")
    for rows, why in (
            ([], "an EMPTY manifest cannot name a vocabulary"),
            ([{"text_id": "x"}], "an entry with NO stratum key is refused"),
            ([{"text_id": "x", "stratum": ""}], "a BLANK stratum is refused"),
            ([{"text_id": "x", "stratum": None}], "a null stratum is refused")):
        try:
            derive_strata(rows)                       # type: ignore[arg-type]
            check(False, f"{why} — must raise StrataDerivationError")
        except StrataDerivationError:
            check(True, f"{why} (StrataDerivationError, never a silent default)")

    print("== selftest 5b: the webtext-v3 basis can SATISFY the §3.4 carry rule ==")
    (src3, tgt3), labels3, entries3 = _basis_world(
        np.random.default_rng(5), ("wikitext", "c4", "pg19", "stackexchange"))
    train3, test3 = _topic_split(labels3, np.random.default_rng(5))
    recs3, _ = run_pair_arm(src3, tgt3, 1, 1, labels3, train3, test3,
                            np.random.default_rng(5), k_grid=(8,), n_null=8)
    r3 = recs3[0]
    check(tuple(r3.per_stratum_null_q95) == labels3.strata,
          f"per_stratum_null_q95 is keyed by the DERIVED set in derived order: "
          f"{list(r3.per_stratum_null_q95)}")
    check(set(r3.per_stratum_r2) == set(labels3.strata),
          f"per_stratum_r2 covers all four v3 strata: {sorted(r3.per_stratum_r2)}")
    check(len(r3.strata_carried) >= MIN_STRATA_CARRIED and r3.valid,
          f"strata_carried={r3.strata_carried} clears the frozen §3.4 bar "
          f"(>= {MIN_STRATA_CARRIED} of {len(labels3.strata)}) and the fit is "
          f"VALID — under the old hardcoded tuple this list was EMPTY and no v3 "
          f"fit could ever have been used")
    check(all(g.endswith("|neutral|neutral") for g in labels3.group)
          and all(g.split("|")[0] in labels3.strata for g in labels3.group),
          "the stratum-preserving null re-pairs within the DERIVED strata: its "
          "permutation group key leads with the manifest's own stratum name, so "
          "the sharp null needs no vocabulary of its own")
    perm = null_permutations(labels3, train3, np.random.default_rng(1), "stratum")
    strat_train = labels3.stratum[train3]
    check(bool((strat_train[perm] == strat_train).all()),
          "and that is PROVEN on the permutation itself — every re-paired row "
          "keeps its stratum, on a vocabulary this module has never seen")

    print("== selftest 5c: a single-stratum basis is ANNOUNCED, not silently invalid ==")
    (src1, tgt1), labels1, _ = _basis_world(np.random.default_rng(6), ("only",))
    train1, test1 = _topic_split(labels1, np.random.default_rng(6))
    recs1, _ = run_pair_arm(src1, tgt1, 1, 1, labels1, train1, test1,
                            np.random.default_rng(6), k_grid=(8,), n_null=4)
    check(labels1.strata == ("only",) and len(recs1[0].per_stratum_r2) == 1,
          "the readout has exactly one stratum row, named by the basis")
    check(not recs1[0].valid,
          f"and the fit cannot be VALID: §3.4 needs >= {MIN_STRATA_CARRIED} "
          f"carries and one stratum cannot supply them. `run_grid` logs a "
          f"WARNING naming this before it fits, so the verdict is never read as "
          f"a statement about the maps")

    print("== selftest 5d: the derived set survives subsetting (the basis, not the slice) ==")
    keep = np.isin(labels3.stratum, ("wikitext", "c4"))
    sub = subset_labels(labels3, keep)
    check(sub.strata == labels3.strata and len(sub.stratum) == int(keep.sum()),
          f"--fit-strata subsets the ROWS and keeps the basis vocabulary "
          f"{list(sub.strata)}: a stratum the fit excluded reports as absent "
          f"rather than redefining the §3.4 denominator (and this is what keeps "
          f"the v2.1 --fit-strata behaviour byte-identical to the old constant)")

    print("== selftest 5e: load_labels derives from a real manifest file ==")
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory(prefix="a8_basis_") as td:
        man = Path(td) / "corpus_manifest.json"
        rows = [{**e, "voice": "neutral", "mode": "neutral", "topic_idx": None}
                for e in entries3]
        man.write_text(json.dumps({"entries": rows}))
        got = load_labels(man, [rows[0]["text_id"], rows[-1]["text_id"]])
        check(got.strata == ("wikitext", "c4", "pg19", "stackexchange"),
              "the vocabulary comes from the WHOLE manifest even when the bank "
              "holds two rows — the basis is not redefined by what was banked")
        check(list(got.group) == [f"{rows[0]['stratum']}|neutral|neutral",
                                  f"{rows[-1]['stratum']}|neutral|neutral"],
              "and the null group key is built from the manifest's own fields")
        bad = Path(td) / "not_a_manifest.json"
        bad.write_text(json.dumps({"rows": []}))
        for path, why in ((bad, "a file with no `entries` list"),
                          (Path(td) / "absent.json", "an ABSENT manifest")):
            try:
                manifest_entries(path)
                check(False, f"{why} must raise StrataDerivationError")
            except StrataDerivationError:
                check(True, f"{why} raises StrataDerivationError, naming the path")

    print("== selftest 5f: the v2.1 fitting manifest of record derives S1,S2,S3 ==")
    if real_v21.exists():
        rows21 = manifest_entries(real_v21)
        derived21 = derive_strata(rows21)
        check(derived21 == LEGACY_V21_STRATA,
              f"{real_v21} ({len(rows21)} entries) derives {list(derived21)} — "
              f"identical to the constant this module carried, so no banked v2.1 "
              f"fit's per-stratum readout moves")
        check(list(stratum_counts(rows21).values()) == [320, 160, 295],
              f"census {stratum_counts(rows21)}")
    else:
        skip("the REAL v2.1 fitting manifest derives the legacy vocabulary",
             f"{real_v21} is absent from this tree (a repo data artifact; a "
             f"deployed code tree has none). The synthetic v2.1-shaped fixture "
             f"in selftest 5 covers the same claim shape")

    import tempfile as _tf

    print("== selftest 6a: THE v2.1 LEGACY SPLIT IS BYTE-EXACT (certified input) ==")
    #  The banked v2.1 fits were computed against ONE realized membership. This
    #  block is the gate-identity proof that the splits wiring did not move it:
    #  the numbers below were recorded from the PRE-WIRING code and are asserted,
    #  not printed. If this fails, no v2.1 readout downstream is comparable.
    if real_v21.exists():
        rows21 = manifest_entries(real_v21)
        ids21 = [str(e["text_id"]) for e in rows21]
        sha21 = hashlib.sha256(real_v21.read_bytes()).hexdigest()
        check(sha21 == V21_SPLIT_OF_RECORD["manifest_sha256"]
              and len(rows21) == V21_SPLIT_OF_RECORD["n_entries"],
              f"the v2.1 manifest of record is the recorded one "
              f"(sha {sha21[:12]}…, {len(rows21)} entries)")
        labels21 = load_labels(real_v21, ids21)
        tr21, te21, info21 = make_split(labels21, seed=A8_SEED)
        test_ids21 = [t for t, m in zip(ids21, te21) if m]
        ids_sha = hashlib.sha256("\n".join(test_ids21).encode()).hexdigest()
        check(int(tr21.sum()) == V21_SPLIT_OF_RECORD["n_train"]
              and int(te21.sum()) == V21_SPLIT_OF_RECORD["n_test"],
              f"counts {int(tr21.sum())}/{int(te21.sum())} == the recorded "
              f"{V21_SPLIT_OF_RECORD['n_train']}/{V21_SPLIT_OF_RECORD['n_test']}")
        check(info21["split_sha256"] == V21_SPLIT_OF_RECORD["split_sha256"],
              f"split_sha256 {info21['split_sha256'][:16]}… == the recorded mask")
        check(ids_sha == V21_SPLIT_OF_RECORD["test_ids_sha256"],
              f"the HELD-OUT MEMBERSHIP itself is byte-exact "
              f"(sha of the ordered test text_ids = {ids_sha[:16]}…)")
        check(info21["held_topics"] == V21_SPLIT_OF_RECORD["held_topics"],
              f"the held topics are the recorded {info21['held_topics']}")
        plan21 = resolve_split_plan(labels21, ids21, set(ids21),
                                    manifest=real_v21, manifest_sha256=sha21)
        check(bool((plan21.test == te21).all()) and bool(plan21.keep.all())
              and plan21.info["source"] == "legacy-derivation",
              "the lane resolver with NO artifact takes the legacy lane and "
              "reproduces the same mask — the v2.1 command line is unchanged")
        check(plan21.info["split_sha256"] == V21_SPLIT_OF_RECORD["split_sha256"]
              and plan21.info["test_ids_sha256"]
              == V21_SPLIT_OF_RECORD["test_ids_sha256"]
              and plan21.info["rule"].startswith("topic-grouped:"),
              "and its stamp carries the recorded split_sha256, the recorded "
              "membership sha and the same rule string the banked summaries carry")
    else:
        skip("the v2.1 BYTE-EXACT split proof",
             f"{real_v21} is absent from this tree (a repo data artifact). The "
             f"proof asserts recorded shas and cannot be synthesized")

    print("== selftest 6b: a webtext-v3 basis with NO artifact REFUSES ==")
    with _tf.TemporaryDirectory(prefix="a8_v3_") as td:
        root = Path(td)
        v3_rows = _v3_fixture_rows()
        v3_man = root / "corpus_manifest.json"
        v3_sha = _write_fixture_manifest(v3_man, v3_rows)
        v3_ids = [str(r["text_id"]) for r in v3_rows]
        v3_labels = load_labels(v3_man, v3_ids)
        n_top, n_s2 = legacy_split_capacity(v3_labels)
        check((n_top, n_s2) == (0, 0),
              "the v3-shaped basis offers 0 topics and 0 S2 shard ranks — the "
              "legacy rule has nothing to draw from")
        try:
            make_split(v3_labels)
            check(False, "make_split on a v3 basis must REFUSE")
        except SplitSelectionError as exc:
            check("--splits-artifact" in str(exc) and "empty test set" in str(exc),
                  "make_split REFUSES and names --splits-artifact (never an empty "
                  "test set, never a re-derivation)")
        try:
            resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                               manifest_sha256=v3_sha)
            check(False, "the lane resolver must refuse too")
        except SplitSelectionError:
            check(True, "and the lane resolver refuses on the same basis — there "
                        "is no path from a v3 manifest to a derived split")

        print("== selftest 6c: the --splits-artifact happy path ==")
        held = [f"{s}-00{i}" for s in _V3_FIXTURE_STRATA for i in (1, 5)]
        inel = [f"{_V3_FIXTURE_STRATA[0]}-000", f"{_V3_FIXTURE_STRATA[0]}-007"]
        sp_path = root / "splits.json"
        sp_path.write_text(json.dumps(_splits_fixture(v3_sha, v3_ids, held, inel),
                                      indent=1))
        art = load_splits_artifact(sp_path)
        check(art.sha256 == hashlib.sha256(sp_path.read_bytes()).hexdigest()
              and art.source_path == str(sp_path),
              "the loader records the artifact's own sha256 and path")
        plan = resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                                  manifest_sha256=v3_sha, splits=art)
        realized = [t for t, m in zip(v3_ids, plan.test) if m]
        check(sorted(realized) == sorted(held) and int(plan.test.sum()) == len(held),
              f"the realized test membership IS the artifact's ({len(held)} ids), "
              f"row for row")
        check(int(plan.train.sum()) == len(v3_ids) - len(held)
              and plan.info["coverage"]["covers_the_whole_artifact"],
              "train is the complement and the coverage read says so")
        check(plan.info["source"] == "splits-artifact"
              and plan.info["artifact_sha256"] == art.sha256
              and plan.info["rule"] == art.rule.statement,
              "the stamp names the source, the artifact sha and echoes the "
              "artifact's OWN rule statement")
        check(plan.info["overlap_integrity"]["n_ineligible_in_test"] == 0
              and plan.info["overlap_integrity"]["n_ineligible_declared"] == len(inel),
              "the overlap integrity re-check ran on the REALIZED membership and "
              "found zero ineligible ids in the test side")
        check(plan.info["basis"]["matches_prereg_basis_identity"]
              and plan.info["basis"]["manifest_sha256_loaded"] == v3_sha,
              "and the stamp records both manifest shas it compared")

        print("== selftest 6c2: the OVERLAP-CLEAN train exclusion (§6-I5 beside) ==")
        primary_identity = split_identity(plan.info)
        train_ids_fixture = [t for t, m in zip(v3_ids, plan.train) if m]
        drop = train_ids_fixture[:2]

        def _excl_doc(ids: list[str]) -> dict:
            return {"artifact": "fixture-exclusion/v1",
                    "clause": "selftest fixture, not a frozen clause",
                    "excluded_text_ids": list(ids), "n_excluded": len(ids)}

        ex_path = root / "exclusion_ok.json"
        ex_path.write_text(json.dumps(_excl_doc(drop), indent=1))
        ex = load_train_exclusion(ex_path)
        check(ex.sha256 == hashlib.sha256(ex_path.read_bytes()).hexdigest()
              and ex.n_excluded == 2,
              "the loader digests the exclusion artifact's own BYTES")
        clean = apply_train_exclusion(plan, v3_ids, ex)
        check(int(clean.train.sum()) == int(plan.train.sum()) - 2,
              f"train {int(plan.train.sum())} -> {int(clean.train.sum())}: exactly "
              f"the excluded rows left the TRAIN mask")
        check(bool((clean.test == plan.test).all()),
              "and the HELD-OUT membership is untouched, row for row")
        check(not any(bool(clean.train[v3_ids.index(t)]) for t in drop),
              "no excluded id survives anywhere in the train mask")
        blk = clean.info["train_exclusion"]
        check(blk["artifact_sha256"] == ex.sha256 and blk["n_excluded"] == 2
              and blk["n_train_before"] == int(plan.train.sum())
              and blk["n_train_after"] == int(clean.train.sum())
              and clean.info["n_train"] == int(clean.train.sum()),
              "the stamp carries the exclusion sha, the count and BOTH train sizes")
        beside_identity = split_identity(clean.info)
        check(beside_identity != primary_identity
              and beside_identity["train_exclusion_sha256"] == ex.sha256
              and primary_identity["train_exclusion_sha256"] is None,
              "the split IDENTITY separates a beside object from a primary one — "
              "so guard_fits_dir refuses to bank one over the other")

        held_path = root / "exclusion_held_out.json"
        held_path.write_text(json.dumps(_excl_doc([held[0]]), indent=1))
        check(_raises(lambda: apply_train_exclusion(
            plan, v3_ids, load_train_exclusion(held_path)), SplitSelectionError),
              "an excluded id that is HELD OUT in this split REFUSES (it would "
              "change the held-out population, not just the train influence)")
        ghost_path = root / "exclusion_ghost.json"
        ghost_path.write_text(json.dumps(_excl_doc(["ghost-999"]), indent=1))
        check(_raises(lambda: apply_train_exclusion(
            plan, v3_ids, load_train_exclusion(ghost_path)), SplitSelectionError),
              "an excluded id that resolves to no loaded row REFUSES")
        empty_path = root / "exclusion_empty.json"
        empty_path.write_text(json.dumps(_excl_doc([]), indent=1))
        check(_raises(lambda: load_train_exclusion(empty_path), SplitSelectionError),
              "an EMPTY exclusion set REFUSES — a beside that excludes nothing is "
              "a duplicate of the primary wearing a beside name")

        print("== selftest 6d: an artifact from ANOTHER corpus REFUSES ==")
        bad = _splits_fixture("f" * 64, v3_ids, held, inel)
        bad_path = root / "splits_other_basis.json"
        bad_path.write_text(json.dumps(bad, indent=1))
        try:
            resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                               manifest_sha256=v3_sha,
                               splits=load_splits_artifact(bad_path))
            check(False, "a manifest-sha mismatch must REFUSE")
        except SplitSelectionError as exc:
            check(v3_sha[:16] in str(exc) and "f" * 16 in str(exc),
                  "a membership drawn from a different corpus REFUSES, naming "
                  "both shas")

        print("== selftest 6e: ids that do not resolve REFUSE ==")
        ghost = _splits_fixture(v3_sha, v3_ids, held, inel)
        ghost["test_ids"] = sorted(set(ghost["test_ids"]) | {"ghost-999"})
        ghost["counts"]["n_test"] += 1
        ghost["counts"]["n_texts"] += 1
        ghost["per_stratum"]["ghost"] = {"holdout_group_keys": ["ghost-999"],
                                         "test_ids": ["ghost-999"]}
        gp = root / "splits_ghost.json"
        gp.write_text(json.dumps(ghost, indent=1))
        try:
            resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                               manifest_sha256=v3_sha,
                               splits=load_splits_artifact(gp))
            check(False, "an unresolvable text_id must REFUSE")
        except SplitSelectionError as exc:
            check("ghost-999" in str(exc) and "do not resolve" in str(exc),
                  "an artifact id absent from the corpus REFUSES, naming it")
        short = _splits_fixture(v3_sha, v3_ids[:-1], held, inel)
        shp = root / "splits_short.json"
        shp.write_text(json.dumps(short, indent=1))
        try:
            resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                               manifest_sha256=v3_sha,
                               splits=load_splits_artifact(shp))
            check(False, "a corpus row on NEITHER side must REFUSE")
        except SplitSelectionError as exc:
            check("NEITHER side" in str(exc),
                  "a loaded row the artifact does not place REFUSES rather than "
                  "silently becoming train")
        leak = _splits_fixture(v3_sha, v3_ids, held + [inel[0]], inel)
        lp = root / "splits_leak.json"
        lp.write_text(json.dumps(leak, indent=1))
        try:
            load_splits_artifact(lp)
            check(False, "an ineligible id on the test side must REFUSE")
        except SplitSelectionError as exc:
            check("ineligible" in str(exc),
                  "an artifact whose OWN test side carries an ineligible id is "
                  "refused at load — the fitter re-checks rather than trusts")

        print("== selftest 6f: the --half a|b lane ==")
        half_a_ids = [i for i in v3_ids if int(i.rsplit("-", 1)[1]) % 2 == 0]
        a_test = [i for i in half_a_ids if i.endswith("-002")]
        half_b_ids = [i for i in v3_ids if i not in set(half_a_ids)]
        b_test = [i for i in half_b_ids if i.endswith("-003")]
        hv_path = root / "halves.json"
        hv_path.write_text(json.dumps(
            _halves_fixture(v3_sha, v3_ids, half_a_ids, a_test, b_test, inel),
            indent=1))
        hv = load_halves_artifact(hv_path)
        plans = {}
        for hf, members, htest in (("a", half_a_ids, a_test),
                                   ("b", half_b_ids, b_test)):
            p = resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                                   manifest_sha256=v3_sha, halves=hv, half=hf)
            plans[hf] = p
            kept = [t for t, k in zip(v3_ids, p.keep) if k]
            realized = [t for t, m in zip(kept, p.test) if m]
            check(sorted(kept) == sorted(members),
                  f"half {hf}: the fit is restricted to the artifact's "
                  f"{len(members)} members")
            check(sorted(realized) == sorted(htest),
                  f"half {hf}: the internal split is the artifact's own "
                  f"({len(htest)} held out), never recomputed")
            check(p.info["half"] == hf and p.info["artifact_sha256"] == hv.sha256
                  and p.info["half_stream"] == f"split-v3-half-{hf}",
                  f"half {hf}: the stamp carries the half identity, the artifact "
                  f"sha and the half's own stream")
            check(p.info["overlap_integrity"]["n_ineligible_in_test"] == 0,
                  f"half {hf}: zero ineligible ids in its internal test side "
                  f"(re-checked on the realized membership)")
        ka = {t for t, k in zip(v3_ids, plans["a"].keep) if k}
        kb = {t for t, k in zip(v3_ids, plans["b"].keep) if k}
        check(not (ka & kb) and ka | kb == set(v3_ids),
              "the two halves are DISJOINT and exhaustive as realized by the fitter")
        check(plans["a"].info["test_ids_sha256"] != plans["b"].info["test_ids_sha256"],
              "and the two halves stamp different MEMBERSHIPS "
              "(test_ids_sha256), so no reader can confuse an I1 half with the "
              "other — note the mask-only split_sha256 CAN coincide across "
              "halves, which is why the membership sha exists")
        for kwargs, why in (
                ({"half": "a"}, "--half without the halves artifact"),
                ({"halves": hv}, "--halves-artifact without --half"),
                ({"halves": hv, "half": "a", "splits": art},
                 "--splits-artifact AND --half together")):
            try:
                resolve_split_plan(v3_labels, v3_ids, set(v3_ids), manifest=v3_man,
                                   manifest_sha256=v3_sha, **kwargs)  # type: ignore[arg-type]
                check(False, f"{why} must REFUSE")
            except SplitSelectionError:
                check(True, f"{why} REFUSES rather than guessing")
        try:
            hv.half("c")
            check(False, "an unknown half must REFUSE")
        except SplitSelectionError:
            check(True, "an unknown half name REFUSES")

        print("== selftest 6g: a fits/ dir is never silently re-banked ==")
        fdir = root / "fits"
        fdir.mkdir()
        (fdir / "cp2_summary.json").write_text(
            json.dumps({"split": plans["a"].info}))
        check(_ok(lambda: guard_fits_dir(fdir, plans["a"].info)),
              "the same split re-runs into the same directory (a rerun is fine)")
        for other, why in ((plans["b"].info, "the OTHER half"),
                           (plan.info, "the whole-corpus artifact split")):
            try:
                guard_fits_dir(fdir, other)
                check(False, f"banking {why} over half a must REFUSE")
            except SplitSelectionError:
                check(True, f"banking {why} into half a's directory REFUSES "
                            f"(--fits-dirname is the answer)")

    print("== selftest 6h: the FROZEN artifacts themselves ==")
    if have_splits:
        assert splits_artifact is not None
        real_sp = load_splits_artifact(Path(splits_artifact))
        derived_sha, matches = real_sp.basis.derived_from_manifest_sha256()
        check(real_sp.counts.n_test == len(real_sp.test_ids)
              and real_sp.counts.n_train == len(real_sp.train_ids)
              and real_sp.counts.n_texts == len(set(real_sp.test_ids)
                                                | set(real_sp.train_ids)),
              f"the frozen splits.json (sha {real_sp.sha256[:12]}…) is internally "
              f"consistent: {real_sp.counts.n_test}/{real_sp.counts.n_texts} held "
              f"out, basis {real_sp.basis.corpus}")
        check(matches and derived_sha == real_sp.basis.expected_v3_manifest_sha256,
              f"and it was derived from the prereg's basis identity "
              f"{derived_sha[:12]}…")
        check(_raises(lambda: require_basis_match(
            real_sp.basis, loaded_sha="0" * 64, manifest=Path("x"),
            artifact_path=real_sp.source_path, kind="splits"), SplitSelectionError),
              "and the belongs-to-this-basis gate REFUSES a corpus it was not "
              "derived from")
    else:
        skip("the FROZEN splits.json block",
             "no --splits-artifact path was given (the artifacts are desk-side "
             "staging, never git); the fixtures above cover the same claims")
    if have_halves:
        assert halves_artifact is not None
        real_hv = load_halves_artifact(Path(halves_artifact))
        a_ids, b_ids = set(real_hv.half_a.text_ids), set(real_hv.half_b.text_ids)
        check(not (a_ids & b_ids) and len(a_ids) == len(b_ids),
              f"the frozen halves.json (sha {real_hv.sha256[:12]}…) holds two "
              f"disjoint halves of {len(a_ids)}")
        check(all(set(getattr(real_hv, k).internal_split.test_ids)
                  <= set(getattr(real_hv, k).text_ids)
                  for k in ("half_a", "half_b")),
              "each half's internal test side lies INSIDE that half")
    else:
        skip("the FROZEN halves.json block",
             "no --halves-artifact path was given (desk-side staging, never git)")

    print(f"\nselftest: {len(failures)} failure(s)")
    #  RAKE M44: coverage is part of the verdict, and per configuration — a bare
    #  pass count cannot be read without knowing which cell produced it.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    for name in failures:
        print(f"  FAILING CHECK: {name}")
    print(f"TOTAL fit_transport_maps: {len(checks)} check(s), "
          f"{len(failures)} failure(s), {len(skips)} skip(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--source-model", default="3b", choices=MODEL_KEYS,
                    help="un-ratified scan-registry models (metabasis.roster) are "
                         "accepted but have no fixed grid — pass --src-sites for them")
    ap.add_argument("--target-model", default="8b", choices=MODEL_KEYS,
                    help="un-ratified scan-registry models (metabasis.roster) are "
                         "accepted but have no fixed grid — pass --tgt-sites for them")
    ap.add_argument("--n-null", type=int, default=N_NULL_REPS)
    ap.add_argument("--fit-strata", default=None,
                    help="comma list, e.g. S1,S2 — fit/gate g on these strata only "
                         "(the mode-free beside column; desk order 2026-07-22)")
    ap.add_argument("--src-sites", default=None, help="comma-separated source-site override")
    ap.add_argument("--tgt-sites", default=None, help="comma-separated target-site override")
    ap.add_argument("--k-grid", default=None,
                    help="comma-separated PC ranks (default 32,128,512). The add-3 rank "
                         "guard binds small-n fits: k <= n_train/1.2, k512 forbidden at n<620.")
    ap.add_argument("--fits-dirname", default="fits",
                    help="output subdir under arm-root (use fits_modefree for the "
                         "beside column — never overwrite the primary)")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma list of template arms to fit. Defaults to both. Use "
                         "'raw' alone for template-less BASE models (olmo2-7b has no "
                         "chat template, so its native arm does not exist) — and see "
                         "A8-add-7.1: a raw-arm-only model's constants live in the "
                         "PARALLEL RAW-ARM star system, never the native one.")
    ap.add_argument("--splits-artifact", type=Path, default=None,
                    help="the FROZEN webtext-v3 splits.json (prereg §2). The "
                         "membership is CONSUMED, never re-derived; without it a "
                         "webtext-v3 manifest REFUSES rather than holding nothing "
                         "out. Desk-side staging path (never git).")
    ap.add_argument("--halves-artifact", type=Path, default=None,
                    help="the FROZEN halves.json (prereg §6-I1). Required by "
                         "--half; carries each half's membership AND its own "
                         "internal train/test split.")
    ap.add_argument("--half", default=None, choices=sorted(HALF_KEYS),
                    help="fit ONE §6-I1 half: restrict to its membership and use "
                         "that half's internal split from the artifact. The half "
                         "identity and the artifact sha go into the stamp.")
    ap.add_argument("--exclude-train-ids", type=Path, default=None,
                    help="a ruled exclusion artifact ({excluded_text_ids, "
                         "n_excluded}) whose ids are dropped from the TRAIN side "
                         "only — frozen webtext-v3 §6-I5's overlap-clean BESIDE. "
                         "The held-out membership is untouched and an excluded id "
                         "on the test side REFUSES. The artifact's sha stamps every "
                         "object and joins the split identity, so a beside object "
                         "can never overwrite or be read as a primary one.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(splits_artifact=args.splits_artifact,
                        halves_artifact=args.halves_artifact)
    if args.half and not args.halves_artifact:
        raise SystemExit(
            f"--half {args.half} needs --halves-artifact <…/halves.json>: the "
            f"§6-I1 halving is FROZEN and is never recomputed at fit time")
    if args.half and args.splits_artifact:
        raise SystemExit(
            "--splits-artifact and --half are two different memberships (the whole "
            "corpus vs one half with its own internal split) — pass one, never both")
    if args.halves_artifact and not args.half:
        raise SystemExit(
            "--halves-artifact was given without --half a|b: which half is the fit? "
            "Never guessed")
    if args.exclude_train_ids and not (args.splits_artifact or args.half):
        raise SystemExit(
            "--exclude-train-ids needs a FROZEN membership (--splits-artifact, or "
            "--half with --halves-artifact). On the legacy in-code lane the split "
            "is re-derived at fit time, so 'the same computation minus these rows' "
            "has no fixed thing to be the same as — refusing")
    strata = ([s.strip() for s in args.fit_strata.split(",") if s.strip()]
              if args.fit_strata else None)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; valid: {ARMS}")
    # A model with no fixed grid (still on its scan, not yet ratified) needs its
    # grid on the command line — say so here rather than dying on a bare KeyError
    # inside run_grid. Ratified models are in SITES and pass straight through.
    for role, model, override in (("--source-model", args.source_model, args.src_sites),
                                  ("--target-model", args.target_model, args.tgt_sites)):
        if model not in SITES and not override:
            raise SystemExit(
                f"{role} {model!r} has no fixed fit grid (un-ratified scan-registry "
                f"model; its 12-site scan grid is {SCAN_GRIDS.get(model)}). Pass "
                f"{'--src-sites' if role.startswith('--source') else '--tgt-sites'} "
                f"explicitly — sites from curves, never fiat.")
    run_grid(args.arm_root, args.source_model, args.target_model, n_null=args.n_null,
             fit_strata=strata, fits_dirname=args.fits_dirname, arms=arms,
             splits_artifact=args.splits_artifact,
             halves_artifact=args.halves_artifact, half=args.half,
             exclude_train_ids=args.exclude_train_ids,
             k_grid=(tuple(int(k) for k in args.k_grid.split(","))
                     if args.k_grid else K_GRID),
             sites_override={
                 **({args.source_model: tuple(int(x) for x in args.src_sites.split(","))}
                    if args.src_sites else {}),
                 **({args.target_model: tuple(int(x) for x in args.tgt_sites.split(","))}
                    if args.tgt_sites else {})} or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
