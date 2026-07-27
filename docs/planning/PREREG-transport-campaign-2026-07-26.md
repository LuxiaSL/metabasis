# PREREG — the metabasis transport campaign (FROZEN 2026-07-26)

> **FROZEN — ratified by Luxia 2026-07-26** after four decision rounds (all
> recorded in the desk ledger). Ceremony record in §8; verify this text
> against the TAGGED commit (`git show freeze/transport-campaign:docs/planning/PREREG-transport-campaign-2026-07-26.md | sha256sum`),
> never the working file. Vocabulary per
> `docs/methodology/naming-conventions.md`. Corrections append as dated
> addenda; park, don't amend. Collection may now fire.

## TL;DR (plain language)

Steering vectors port between language models through a transport map fit
only on mean residual states over shared generic text; on 7 models the
transport fidelity factorized into one portability coefficient per model
(the star factorization), with 3/3 held-out pair predictions landing inside
pre-frozen bands. This campaign scales that to 21 collected star models (+ koto imported) —
spanning 1.5B→405B dense and 16B→671B MoE — and 190 pre-registered held-out
pair predictions, runs the control-object class
split under one protocol, and maps the boundary of the effect (base models,
far diets, MoEs, and non-transformer architectures). If the held-out
predictions land, the star factorization graduates to the star law — the
stamp is held until that first-read, by ruling. A pre-registered mechanism
panel (hub-invariance across a quad-hub design, hub drop-out in composition,
family residuals, the per-class star) tests not just whether the
factorization holds but WHY.

## 1. Roster (FROZEN)

**Tier rule (Luxia, 2026-07-26):** two-tier, maximum star membership — every
transformer is a full node (native entropy-gradient target built, star row);
behavioral tier is reserved for the SSM swings. Arms: instruct models run
native + raw; base models run raw only (the arm-consistency rule — their
constants live in the raw system, never native).

### Full nodes (star membership)

| # | model (checkpoint) | why on the roster | arms | status |
|---|---|---|---|---|
| 1 | Llama-3.2-3B-Instruct | anchor (carried; banked) | native+raw | banked |
| 2 | Llama-3.1-8B-Instruct | anchor + **primary hub** | native+raw | banked |
| 3 | Qwen2.5-7B-Instruct | anchor + **audit hub** | native+raw | banked |
| 4 | DeepSeek-V2-Lite-Chat | anchor MoE (carried; banked) | native+raw | banked |
| 5 | Gemma-3-27B-it | anchor, 3rd family (carried; banked) | native+raw | banked |
| 6 | OLMo-2-1124-7B (base) | anchor, RLHF-free (carried; banked) | raw | banked |
| 7 | **OLMo-2-1124-7B-Instruct** | the instruct-OLMo discriminator — resolves the base-alignability-vs-raw-arm-confound fork (anamnesis registry O-14): same pretrain as #6, chat template added | native+raw | new |
| 8 | Qwen2.5-3B-Instruct | scale ladder, low rung | native+raw | new |
| 9 | Qwen2.5-14B-Instruct | scale ladder | native+raw | new |
| 10 | Qwen2.5-32B-Instruct | scale ladder, high rung | native+raw | new |
| 11 | Llama-3.1-70B-Instruct | scale ceiling, lineage-matched to the 8B anchor (Luxia 2026-07-26) | native+raw | new |
| 12 | Llama-3.3-70B-Instruct | post-training-vintage row: same pretrain family as #11, different RLHF era — scale claim rides #11, vintage comparison rides this (Luxia 2026-07-26) | native+raw | new |
| 13 | Mistral-7B-Instruct-v0.3 | 4th dense family at the 7B weight class | native+raw | new |
| 14 | Phi-4 (14B) | synthetic/"textbook" diet at modern quality — maximal diet distance (Luxia 2026-07-26) | native+raw | new |
| 15 | Phi-3.5-mini-instruct (3.8B) | diet × scale mini-ladder inside the Phi family, size-matched to the 3B anchors (Luxia 2026-07-26) | native+raw | new |
| 16 | Pythia-6.9B (base) | old base, far diet (The Pile), fully open lineage | raw | new |
| 17 | GPT-2-XL (1.5B, base) | the floor-stress rung: oldest diet, smallest dims (d=1600) — where the atlas should degrade if anywhere on transformers | raw | new |
| 18 | Qwen3-30B-A3B | second MoE, family-adjacent to the dense Qwen ladder — the controlled MoE-vs-dense comparison (Luxia 2026-07-26; Qwen1.5-MoE rejected: legibility) | native+raw | new |
| 19 | Mixtral-8x7B-Instruct-v0.1 | third MoE, external legibility | native+raw | new |
| 20 | Llama-3.1-405B-Instruct | dense scale CEILING — completes the lineage-matched ladder 3B→8B→70B→405B, two orders of magnitude in one pretrain family (Luxia 2026-07-26). Sharded collection (device_map path, gated change; spot-replay gate certifies) | native+raw | new |
| 21 | DeepSeek-V3 | frontier MoE (671B, cross-lab; continuity with the anamnesis M7 pin). ⚠ NAMED DTYPE DEVIATION: runs in native FP8 (bf16 does not fit the cluster) — the bitwise spot-replay gate must certify the FP8 forward per-run; states banked fp32 as everywhere (Luxia 2026-07-26) | native+raw | new |
| — | koto | imported star row — numbers via the countersigned record only; public scope: "a custom-pretrained 3B base model with block attention residuals, Muon optimizer, and NCA init" | (imported) | imported |

### Behavioral nodes (transported-write column only; no native target, no star row)

| # | model | why |
|---|---|---|
| 22 | Falcon-Mamba-7B-Instruct | pure SSM — cleanest non-transformer regime test; hit or floor is unambiguous |
| 23 | Zamba2-7B-Instruct | Mamba+attention hybrid — brackets the regime boundary with #22 |

n_star = 21 collected + koto imported. All checkpoint picks ratified by
Luxia 2026-07-26 (decision interrogation, recorded in the ledger).

## 2. Estimands

- **Exchange rate â(source→target)** per control-object class — needle
  (mode contrast vector) and field (gradient-vector family) lines SEPARATE,
  never cross-divided; temperature contrast vector on its own line (ratified
  2026-07-26). Fit-family-labeled; arm-labeled. **Family of record:
  proc_k128** (the only family that self-validated in both arms — inherited
  finding), k32/k512/ridge filed beside; rank guard k ≤ n_train/1.2 applied
  per pair, families it forbids named in every readout.
- **Portability coefficient c_M** per model, derived in a named (arm ×
  family) star system via the hub protocol; a system must pass its own
  held-out check before hosting a constant.
- **Path-independence residuals**: cos(direct A→C, composed A→B→C) + â along
  both paths, over all triangles in the full-node set.
- **Behavioral transfer columns**: entropy-write rise vs the transported
  random band (all nodes incl. behavioral tier); judged 2AFC where budgeted
  (§6); capability battery + coherence panel on every behavioral dose cell.
- **Class-split table**: â and behavioral transfer per control-object class.

## 3. The prediction structure (the star's held-out test)

**Hub protocol — QUAD-HUB (Luxia, 2026-07-26, round 4).** Primary hub =
Llama-3.1-8B (most-banked, mid-scale, mid-family; exact continuity with the
inherited system — every constant of record derives here). Audit hubs, each
re-deriving c_M for the audit set: **Qwen2.5-7B** (family control, mid-scale)
· **Llama-3.1-405B** (the scale probe — direct test of the
larger-hubs-transfer-better conjecture) · **Gemma-3-27B** (third family).
Audit set, spanning family/scale/regime/diet: {Llama-3.2-3B,
OLMo-2-1124-7B-Instruct, Qwen2.5-32B, Mixtral-8x7B, Phi-4}. Frozen gate:
**hub-invariance** |c_M(primary) − c_M(audit hub h)| ≤ .05 per (h, M) —
the inherited system self-check width. This is a MECHANISM measurement, not
just QC: under the endpoint-projection theory the constants are
hub-invariant; a systematic trend of c_M with hub scale or hub family is
scored as a named mechanism finding either way (§5.11). Audit-hub pairs are
STILL filed as held-out predictions first (they are non-primary-hub pairs):
file → fit → score → only then reuse for audit derivation — the audit costs
no prediction and breaks no ordering.

Ceremony order per pair, enforced by phase-ordered CLI:

1. Fit primary-hub↔M for every roster model M (20 hub pairs; 4 already banked).
2. Derive c_M in the applicable (arm × family) system; system must pass its
   own out-of-sample check to host constants.
3. For every non-hub pair (A,B): FILE the prediction
   â(A→B) = c_A·c_B ± .05 absolute at k128, in the pair's applicable arm,
   BEFORE that pair is fit. Prediction IDs are content:
   `star-prediction/<source>→<target>/<arm>-k128`
   (e.g. `star-prediction/qwen2.5-32b→gemma-3-27b/native-k128`).
4. Fit the pair; score in-band / out-of-band. No band moves after filing.

**Counts:** non-hub pairs among 20 non-primary-hub collected models =
**190 held-out predictions**. Instruct↔base pairs are predicted in the raw system (both
models' raw rows exist); instruct↔instruct in native. Direction of record =
forward as listed; the asymmetry analysis (§5) reads both directions beside.

**Aggregate gates (frozen now, scored at law-at-scale first-read):**
- G-star-hit: ≥ 80% of the 190 in band → the star factorization survives at
  scale (necessary condition for the graduation stamp — Luxia's stamp
  remains a judgment, not an automatic consequence).
- G-star-structure: out-of-band failures, if any, cluster by a nameable
  model property (regime/diet/scale), not uniformly — scored descriptively.
- Near-zero carve-out: any prediction with |c_A·c_B| < .08 is scored
  MAGNITUDE-ONLY (|observed| within band of |predicted|; sign unscored).

## 4. Sites, actuation, and collection discipline

- **Sites from curves, never fiat**: every NEW model gets an alignment-curve
  site scan (12-site grid, pick the peak — the OLMo precedent) before its
  fit grid is fixed; any banked-vector site must be inside the fit grid,
  else fit the map actually needed. Banked models keep their banked sites.
- **Actuation acceptance gate**: before any transported object is
  behaviorally tested at a site, a known-good lever (the model's own native
  entropy-gradient vector at matched dose) must demonstrate actuation there,
  vs its own random band. Every behavioral cell names its site's actuation
  calibration. Cosine visibility is not functional efficacy.
- **Collection**: slim collector; shared corpus (banked, sha `a6712ca0…`);
  one residual pre-hook; per-text fp32 means; bf16 forward; collect +
  fresh-process bitwise spot-replay (K≥3) in ONE job per node; stamps carry
  both norm conventions + corpus sha + trunk facts. New banks use
  descriptive keys (`entropy_gradient_L{site}`) and canonical builder names.
- **Big-rung engineering (named, gated):** ≥70B-class nodes collect via the
  collector's sharded path (`device_map` across the 8-GPU node; determinism
  certified by the same collect+spot-replay-in-one-job gate — sharding layout
  recorded in the trunk stamp). DeepSeek-V3 forwards in native FP8 (named
  deviation, roster row 21); 405B/DSV3 entropy-gradient target builds use
  input-gradient extraction with checkpointing (never loss.backward — the
  inherited gradient-build rule).
- **New pair-trees get descriptive names** (`outputs/pairs/<modelA>__<modelB>/`);
  the legacy leg-directories stay frozen as-is.

## 5. Experiments beyond the star (each a named saga)

1. **The hub-fit sweep** — §3; the campaign's spine.
2. **The composition panel** — path independence over all full-node
   triangles; prediction: composed ≈ direct (cos ≥ .85 panel-mean at matched
   k — band set from the banked .91 with margin); plus the known
   non-multiplicativity of attenuation quoted as an expectation, not tested.
3. **The mapping-flexibility benchmark** — Procrustes vs ridge-affine vs
   small-MLP head-to-head UNDER THIS PROTOCOL on 6 chosen pairs spanning
   easy/hard; answers the affine-underperforms prior-art question.
4. **The tokenizer-divergence regression** — â and entropy-write rise vs
   tokenizer-overlap across all hub pairs; mean-pooling's independence claim
   becomes a measured curve.
5. **The asymmetry analysis** — â(A→B) vs â(B→A) across the roster;
   connects the banked reverse-ladder rider.
6. **The naive-transplant null** — zero-pad coordinate identification at
   every behavioral read (operationalization named in every bank stamp).
7. **The entropy-write column** — all 23 nodes: transported entropy-gradient
   vector, signed dose ladder ±{.03,.1,.3}, n=80/cell, vs transported random
   band; the SSM boundary probe is this column on nodes 22–23.
8. **The judged-needle column** — §6 allocation.
9. **The capability battery** — small fixed task set + coherence panel +
   random bands riding EVERY behavioral dose cell (the Gate-1 Correction-2
   requirement, designed in).
10. **The class-split table** — the full control-object roster (formality
    contrast, SAE-derived, gradient family, temperature contrast,
    model-difference/trait, whitened contrast as Σ-family) through the same
    protocol on 6 chosen pairs.
11. **The mechanism panel** (pre-registered signatures of WHY the star
    factorizes; working theory = endpoint projection: each model's lever =
    c_M-weighted chart of a shared direction + idiosyncratic remainder that
    the map scatters; â pays projection at the endpoints only):
    - **(i) Hub drop-out.** Across the composition panel, composed A→hub→B
      reads discriminate c_A·c_B (endpoint theory) against c_A·c_hub²·c_B
      (hub-as-lossy-channel). Banked instance already discriminates:
      composed 3B→8B→Qwen ≈ .32 vs c_3B·c_Qwen = .302 vs channel .171 —
      pre-registered to generalize (or fail) across all panel triangles.
    - **(ii) Family residuals.** Regression of prediction residuals
      (observed − c_A·c_B) on same-family indicator + tokenizer overlap +
      log-scale gap, over all 190 scored predictions. Family-correlated
      idiosyncratic remainders predict POSITIVE within-family residuals;
      spec frozen here, scored descriptively (no gate — a structure read).
    - **(iii) The per-class star.** At class-split, class-indexed
      coefficients (c_M for the needle line vs the field line) derived on
      the class-split pairs: does the factorization hold separately per
      class? (The needle's ~3× field rate on the same map is the purity-of-
      representation effect this formalizes.)
    - Plus the hub-invariance gate of §3, which is this panel's fourth
      signature measured for free inside the prediction machine.

## 6. Judge budget allocation (soft cap $250; Luxia raises on request)

Estimate ~$15/judged leg (blind 2AFC, ~160 pairs/leg, coherence gate
included). Proposed allocation — 14 legs ≈ $210, reserve ≈ $40 (the cap is
soft; Luxia raises on request):

| leg | pair | why |
|---|---|---|
| 1 | 8B→Qwen2.5-7B | re-anchor: the banked dose-monotone .200→1.000 leg, re-run under campaign battery |
| 2 | 8B→Gemma-3-27B | 3rd family judged crossing |
| 3 | 8B→DSV2-Lite (whitened target) | the MoE needle, judged for the first time (banked evidence is cosine-only) |
| 4 | 3B→Qwen2.5-32B | scale-ladder diagonal |
| 5 | 8B→OLMo-2-Instruct | the O-14 discriminator, behavioral column |
| 6 | 8B→Qwen3-30B-A3B | second MoE judged |
| 7 | 8B→Mixtral-8x7B | third MoE judged |
| 8 | 8B→Llama-3.1-70B | scale ceiling |
| 9 | 8B→Phi-4 | diet distance |
| 10 | 8B→Mistral-7B | 4th family |
| 11 | Qwen2.5-7B→8B | reverse direction judged |
| 12 | 8B→GPT-2-XL | the floor rung, judged only if its entropy-write column shows any signal (else leg reallocated) |
| 13 | 8B→Llama-3.1-405B | the dense ceiling, judged |
| 14 | 8B→DeepSeek-V3 | the frontier MoE, judged |

Reserve: second-judge-family cross-check on legs 1+3 if spend allows; the
capstone bait-turn readout (~$10–15) draws from this cap at capstone time.

## 7. Clauses (mandatory, frozen with this document)

- **Aggregation**: the campaign headline aggregates ONLY over the 190
  star predictions via G-star-hit; behavioral columns aggregate per-class
  per-pair, never pooled across classes; no cross-class averaging anywhere.
- **Scope**: every claim carries its quantifier domain explicitly; "across
  the roster" means the named roster of this document, nothing larger.
- **Identity ratios**: every transported identity read files within-model
  cosines + the pair's exchange rate; the normalized ratio is the
  interpretable number.
- **Whitened objects**: Σ-estimator-qualified always; quoted as families.
- **Mode pairs named on both sides** of every cross-model contrast read.
- **Envelope hygiene**: ≥100 fresh randoms + banked null-family members
  through the SAME map at every read; per-family bands where families are
  contrasted; merge → strip → re-derive for any sharded readout; expected-N
  completeness guards on every scorer.
- **The lesion-recipe law**: never build a transported object by removing
  the direction known to produce the behavior.
- **Family-consistent series**: any comparison series is re-derived at one
  family before its trend is ruled (never append cross-family).

## 8. Freeze chain

- git tag: `freeze/transport-campaign` (annotated; its message carries the
  sha256 and the attestation gist URL — a file cannot contain its own hash)
- sha256 of this file at the tag: recorded in the tag message, the
  attestation gist, and the desk ledger
- gist attestation: secret gist under the repo owner's account; URL in the
  tag message and the desk ledger
- date: 2026-07-26

*Everything is UNSTAMPED until the desk's first-read against this frozen
text (C§8). The desk never writes experiment code. The star graduation stamp
is HELD for the law-at-scale first-read (Luxia, 2026-07-26).*

---

## Addenda (dated; append-only — the frozen text above is never edited; verify the frozen body via the tag)

### ADDENDUM 2026-07-26-A — DeepSeek-V3 dtype regime (roster row 21, §4) — ratified by Luxia 2026-07-26

Design-phase source verification (transformers 5.3.0, the pinned collection
version; full evidence in the desk report of record for the FP8 lane
design) established two facts the frozen text did not have:

1. **The row-21 parenthetical "bf16 does not fit the cluster" is corrected.**
   bf16-resident DeepSeek-V3 is 1249.9 GiB = 87.3% of the collection node
   (156.2 GiB/card of 179.1) — it fits, without usable headroom. FP8 is
   43.8%. (n = exact parameter accounting from the checkpoint index,
   reproducing the published 671.03B to 0.005%; desk spot-recomputed.)
2. **§4's "input-gradient extraction with checkpointing" is not executable
   in the FP8 regime as written.** The 5.3.0 FP8 forward carries no autograd
   graph (raw Triton kernels, `is_trainable = False`); `autograd.grad`
   does not error but returns a residual-highway-only gradient with every
   attention/MLP block treated as constant — a silent wrong answer.

**Ruling (Luxia, 2026-07-26): DeepSeek-V3 collects AND builds its native
target in bf16** via dequantize-on-load (`FineGrainedFP8Config(dequantize=
True)`), i.e. the standard roster regime: standard bitwise spot-replay
gate, standard differentiable input-gradient path, whole-node scheduling.
The named FP8 deviation of row 21 is **retained as fallback only**: if
bf16 proves operationally unworkable, DSV3 runs the native-FP8 forward
with a named differentiable-wrapper construction (forward = the unmodified
FP8 kernels; backward = grad_output · dequant(W), straight-through on
activation quantization), certified per-run by the bitwise spot-replay
gate AND the finite-difference gate below.

**Adopted roster-wide (desk, same date): the directional finite-difference
gate on every entropy-gradient target build** — the extracted gradient g
must satisfy |⟨g,v⟩ − (S(+εv) − S(−εv))/2ε| / |dS| < .05 on a random unit
direction at the build site (two extra forwards). A severed graph fails by
orders of magnitude; this is the only check that catches it.

*This addendum changes no prediction, band, gate, or roster membership; it
corrects a rationale and names the construction for an already-frozen
deviation. The frozen body's sha (`33ba8290…`) remains valid at the tag;
the working file's post-addendum sha is recorded in the desk ledger.*
