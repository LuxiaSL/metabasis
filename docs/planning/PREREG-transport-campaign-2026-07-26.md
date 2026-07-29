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

### ADDENDUM 2026-07-27-B — GPT-2-XL position-ceiling deviation (roster, §4) — ratified by Luxia 2026-07-27

Collection-phase discovery (registry-extension enactment, desk report of
record REPORT-hub-rungs-2): GPT-2 carries **learned absolute position
embeddings with a hard n_positions=1024 ceiling** — architecture-level,
every GPT-2 variant. Under GPT-2's own BPE, 148 of the 780 frozen corpus
texts exceed 1024 tokens on the raw arm (max 1565, mean 643.9). The frozen
corpus/seed/split cannot be resampled (§4), so the node cannot see the
banked object as written.

**Ruling (Luxia, 2026-07-27): GPT-2-XL collects with per-text truncation
to its first 1024 tokens** — identical truncation applied to every use of
the node (collection, spot-replay, target build if any, behavioral reads),
recorded in every stamp (`truncation: first-1024`). The 632 unaffected
texts are byte-identical to the frozen objects. **Named caveat, frozen with
the deviation:** for the 148 truncated texts, any pair involving gpt2-xl
compares full-text mean states (other model) against truncated-text mean
states (gpt2-xl) — a per-pair measurement-basis mismatch confined to this
node; any gpt2-xl result that hinges on those texts is flagged at scoring.
This addendum changes no prediction, band, gate, or roster membership.

### ADDENDUM 2026-07-27-C — scoring hardening (G-star-hit context rows, directional fallback, 6-pair rule) — ratified by Luxia 2026-07-27

Filed in response to an outside methodological review (desk record
OUTSIDE-REVIEW-2026-07-27) BEFORE any of the 190 predictions is filed and
before any non-hub pair is fit. All four items are append-only: no
prediction, band, gate threshold, or roster membership moves.

1. **Null baselines beside G-star-hit.** The 190-prediction scoring read
   reports, beside the star's in-band fraction and against the SAME frozen
   ±.05 bands: (a) a **constant grand-mean predictor** (the mean of all
   190 observed â, applied to every pair); (b) a **shuffled-star null** —
   the c_M assignment permuted across models, all 190 products recomputed
   and scored; n=1000 permutations, reporting the permutation-rank of the
   true star's in-band count. Interpretive rule, frozen now: the star
   factorization is claimed as DEMONSTRATED only if G-star-hit passes AND
   the true star exceeds both baselines decisively (constant predictor by
   ≥15 percentage points; permutation rank ≥ 99th percentile). Passing the
   gate without beating the baselines is scored "consistent with, not
   demonstrated."
2. **Pre-named directional fallback.** If the asymmetry analysis (§5.5)
   finds systematic direction effects, the DESIGNATED refinement is the
   directional star **â(A→B) = c_A^out · c_B^in** (2n parameters,
   hub-derivable from both fit directions) — named now as the single
   pre-registered alternative model, so any post-hoc adoption is model
   selection between two pre-named candidates, not a patch. Its adoption
   would be scored on not-yet-filed predictions only; filed bands never
   move.
3. **The 6-pair selection rule, frozen.** Wherever the frozen text says
   "6 chosen pairs spanning easy/hard" (§5.3 mapping-flexibility, §5.10
   class-split): after all 20 hub fits are banked, rank the 20 hub pairs
   by â (native k128; raw where the pair has no native row); the six pairs
   are those at **ranks 1, 4, 8, 12, 16, and 20**. If a selected pair is
   operationally unavailable (blocked node), its neighbor rank (±1,
   preferring the harder side) substitutes, with the substitution recorded.
   The choice is arithmetic; no judgment enters after data exists.
4. **Per-model failure accounting.** The scoring read reports out-of-band
   failures as a per-model count table (failures in which model's rows,
   both as source and target) beside the aggregate — the 190 predictions
   share 20 fitted constants and their failures correlate; the table makes
   the effective sample size inspectable. G-star-structure (§3) is scored
   from this table.

### ADDENDUM 2026-07-27-D — the constant-α companion baseline — ratified by Luxia 2026-07-27

Filed in response to outside review 2 (desk record
`OUTSIDE-REVIEW-2-2026-07-27`) BEFORE any α-gauge or directional-star
prediction is filed. Append-only; no filed prediction, band, or gate
moves. Background finding on record: the canary miss diagnostic
(`REPORT-miss-diag-mistral-phi4-2026-07-27`) showed â pays a projection
cost at the target endpoint only, so target-role-identified constants
carry a per-node ceiling term — i.e. c_M(target-role) = α_M · ceil_M,
where α := â/ceiling is the in-subspace alignment.

1. **Scope trigger.** This addendum activates if and when any
   ceiling-aware gauge (the Addendum-C directional star c_A^out·c_B^in,
   or any explicitly ratified α-based refinement) is adopted for
   not-yet-filed predictions.
2. **The third baseline.** The aggregate scoring read then reports,
   beside Addendum C's two baselines and against the SAME frozen bands:
   the **constant-α predictor** — â(A→B) = ᾱ · [the same
   ceiling/calibration terms the adopted gauge uses], where ᾱ is the
   grand mean of the fitted per-model (or per-role) α values across the
   scored set, with NO per-model α structure.
3. **Frozen interpretive rule.** The per-model-structure claim —
   "models carry intrinsic, role-specific portability beyond measurement
   calibration" — is claimed as DEMONSTRATED only if the adopted star
   beats the constant-α predictor decisively under the Addendum-C
   margins: in-band count ≥ 15 percentage points higher, AND an
   α-permutation null (per-model α assignments permuted across models,
   n = 1000 permutations) placing the true star's in-band count at
   ≥ the 99th percentile. Passing G-star-hit while failing this
   companion is scored "calibration-driven; per-model structure not
   demonstrated."
4. **Rationale on record.** The ceiling correction helps every
   predictor, including structureless ones; ceilings are measurement
   calibration, not model property. The graduation-relevant content is
   per-model α structure beating constant-α under matched ceilings.

### ADDENDUM 2026-07-28-E — the composed-path predictor (racing design) — ratified by Luxia 2026-07-28

Ratified ahead of outside review by explicit ruling (Luxia,
2026-07-28): a BOUNDED first racing batch files under this addendum
tonight; the remaining slots stay protected until the outside
reviewer has seen this text, and any review feedback lands as further
dated addenda affecting unfiled slots only. Contingency ruled with
it: if the E1 implementation gate fails at filing time, the bounded
batch files scalar-only under the frozen contract.

Filed BEFORE any of the remaining 183 predictions is filed and before
any further non-hub pair is fit. Append-only: no filed prediction, band,
gate threshold, budget, or roster membership moves. The scalar star's
frozen protocol (§3) continues UNCHANGED — this addendum adds a second,
independently scored predictor beside it, plus the standing pre-filing
gate adopted at session-3 close.

**Background on record.** The worst-pair composition diagnostic
(`REPORT-worstpair-composition-2026-07-27`) found: (a) the two-hop
composed path through the hub frame recovers 94.4–103.2% of the directly
fitted â on all five scored pairs, while the scalar product c_A·c_B
under-predicts uniformly (composed-vs-product gap 1.33–3.00×) — a scalar
constant is a lossy summary of a hub map; (b) the desk check showed the
composed value retrodicts all five scored â within ±.05 (|e| = .0080 /
.0083 / .0094 / .0346 / .0353) and is computable from banked hub maps
alone, before the pair is ever fit; (c) one scored pair
(qwen2.5-32b↔qwen2.5-14b) failed the prereg's own naive-transplant
control (§5 item 6) — bare cosine +.7503 exceeds the fitted map's â —
and was set aside as artifact-contaminated (Luxia, 2026-07-27).

#### E1. Predictor definition

For a non-hub pair (A, B) in its applicable arm:

**â_comp(A→B) = cos( M_{hub→B}( M_{hub→A}^rev(v_A) ), v_B )**

where M_{hub→M} is the BANKED primary-hub→M transport map (rebuilt-L16
hub column; proc k128; the pair's applicable arm), M^rev is the reverse
pass through the same map via the adjoint identity of record (Ω_MA =
Ω_AMᵀ over the same PCA bases — the bit-exact identity established in
the 8b↔3b closure), and v_M are the banked steering vectors at each
model's site of record. The operationalization of record is the archived
two-hop glue
(`outputs/collection/enactment-archives/worstpair-diag/wp_composition.py`
and its manifest shas): unit-normalized input, fp64 over fp32 fits,
scale/norm handling exactly as archived. No pair-specific quantity of
any kind enters — both maps and both vectors exist before the pair is
touched.

**Implementation gate (frozen):** before the FIRST composed prediction
files, the repo filing tool implementing E1 must reproduce all five
archived retrodictions to |Δ| < 1e-8 against the banked values above,
with the comparison logged in the filing record. A tool that cannot
reproduce the archive does not file.

**Arm availability rule:** â_comp files only where BOTH hub maps exist
banked in the pair's applicable arm at filing time. Where they do not
(e.g. a raw-system pair whose raw hub map is not yet banked), the
composed slot is recorded N/A-AT-FILING — never proxied from another
arm, never backfilled after the pair is fit.

#### E2. Filing structure and bands

- Composed predictions file at the SAME ceremony step as the scalar
  star's (§3 step 3), in the same dated prediction record, BEFORE the
  pair is fit. IDs: `composed-prediction/<source>→<target>/<arm>-k128`.
- **Scored band: â_comp ± .05 absolute** — identical to the scalar
  star's, so the race is head-to-head on identical terms. Beside it,
  the ±.04 hit-rate is reported DESCRIPTIVELY (recorded per pair at
  scoring, never a scored gate) [Luxia ruling 2026-07-28].
- Near-zero carve-out mirrors §3: |â_comp| < .08 is scored
  MAGNITUDE-ONLY.
- Bands never move after filing. The seven already-filed slots remain
  scalar-only of record; their composed retrodictions stay descriptive.

#### E3. The racing design

- **Scope: all remaining 183 slots from ratification** [Luxia ruling
  2026-07-28]. Every future filing record carries both predictions (or
  the composed N/A-at-filing marker).
- The scalar star's gates are UNTOUCHED: G-star-hit ≥80% of the 190,
  failure budget 38, Addendum-C nulls and Addendum-D companion as
  already frozen. The scalar record stands or falls on its own frozen
  terms regardless of the race.
- **G-comp-hit (frozen): ≥80% of composed-filed-and-scored slots
  in-band; failure budget 20% of composed-filed slots** — structurally
  identical to G-star-hit [Luxia ruling 2026-07-28].
- **Head-to-head read (law-at-scale first-read):** the per-pair 2×2
  (both hit / star-only / composed-only / both miss) with per-model
  failure accounting (Addendum-C item 4 applied to both predictors).
- **Superiority rule (frozen now):** "the composed predictor is
  DEMONSTRATED SUPERIOR to the scalar star" is claimed only if, over
  the slots where both were filed and scored, the composed in-band
  fraction exceeds the scalar's by ≥15 percentage points AND a paired
  sign-flip permutation over per-slot outcomes (n=1000) places the
  observed hit-difference at ≥ the 99th percentile. Anything less is
  scored descriptively ("composed ≥ scalar" or "no separation"), not
  claimed.
- **Pre-named outcome interpretations (so neither is post-hoc):**
  G-star-hit fails + G-comp-hit passes → "scalar factorization refuted
  at scale; hub-mediated transport demonstrated at map order." Both
  pass → the superiority rule adjudicates. Both fail → hub-frame
  sufficiency is refuted at scale; §5's per-mechanism sagas proceed on
  the causal story regardless.

#### E4. Null companions (the nulls move to the new gauge too)

1. **Naive-transplant standing gate (per pair, BEFORE filing).** For
   every candidate pair: bare cos(v_A, v_B) after zero-pad/truncate
   coordinate identification (§5 item 6; operationalization of record =
   archived `wp_naive_transplant.py`; the null reseeds PER CALL — sweep
   finding F1 pins this against the phase-artifact alternative), beside
   its seeded random-unit null q95 (A8_SEED=80, n=2000). **The column
   is BANKED**: `naive_transplant_gate_2026-07-28.json` (66 candidate
   pairs + 36 gpt2-xl supplement rows; BOTH directed reads per pair —
   the read is orientation-dependent across unequal dims, the gate
   takes the max ratio; sha `368e13bc…`; desk-verified by raw-npz
   recompute). Pairs entering the roster later get their row banked
   before filing under the same operationalization.
   **Flag rule (frozen): a pair is ARTIFACT-EXPOSED iff |bare cos| >
   q95 AND |bare cos| ≥ .10 absolute** [Luxia ruling 2026-07-28]. A
   flagged pair still files and scores both predictors on the frozen
   bands, with this pre-written clause attached at filing: *"shared
   residual coordinate frame; a hit on this pair is not evidence of
   transport beyond frame-sharing; excluded from mechanism aggregates,
   retained in scored aggregates"* — the qwen2.5-32b↔qwen2.5-14b
   set-aside precedent, now automatic. One flag covers both predictors.
   *Calibration on record (why the two-condition rule):* a per-pair
   95th-percentile floor lets ~5% of pairs clear by chance (expected
   3.30 of 66; observed 6, five of them at 1.01–1.24× the floor —
   chance-band, not signal). The ruled rule flags exactly ONE of the 66
   banked pairs today (qwen2.5-14b↔qwen2.5-32b, |cos| .7503, 26.65×
   floor — the pair already set aside) and zero supplement rows; the
   magnitude condition is what separates shared-frame artifacts from
   multiplicity noise.
2. **Random-rotation composed null (per pair).** E1 recomputed with
   each hub map's Ω replaced by a seeded Haar-random orthogonal matrix
   of identical dims over the same PCA bases and scale handling
   (n=1000, seeded). Reported per pair (q95 of |â_comp^rot|) and in
   aggregate: the rotation-null's in-band count over the composed
   slots, beside G-comp-hit.
3. **Map-permutation null (aggregate; the C-item-1 analog at map
   order).** The hub-map ASSIGNMENT permuted across models (pair (A,B)
   scored with M_{hub→A'}, M_{hub→B'} for permuted A',B'), all composed
   predictions recomputed and scored on the frozen bands; n=1000
   permutations; report the permutation rank of the true assignment's
   in-band count.
4. **Frozen interpretive rule.** "Two hub maps determine every pair" is
   claimed as DEMONSTRATED only if G-comp-hit passes AND the true
   composed predictor beats the rotation-null in-band count by ≥15
   percentage points AND the map-permutation rank is ≥ the 99th
   percentile. Passing the gate without beating the nulls is scored
   "consistent with, not demonstrated" — Addendum-C discipline at map
   order. (Addendum D is untouched: its constant-α companion still
   triggers if any scalar α-gauge is separately adopted; the composed
   predictor's structureless companions are items 2–3 above, matched to
   its own object class.)

#### E5. The claim ladder (why this is the stronger claim)

The scalar star claims fidelity FACTORIZES: 20 numbers predict 190
pairs. The composed predictor claims the hub frame is SUFFICIENT: the
20 banked hub MAPS contain all pairwise transport structure — no
pair-specific fitting, at any order, is needed for any of the 190. The
scalar is the rank-one shadow of the composed object (a product of
projections); session 3 showed the shadow is what failed (uniform
1.33–3.00× under-prediction) while the frame held (94–103% recovery,
5/5). The composed claim is stronger in content — it predicts the full
â, not a bound — while remaining hub-mediated universality: falsifiable
per-pair at ±.05, refutable in aggregate by G-comp-hit, and cheaper
than the scalar to run at scale (two banked maps, CPU-seconds,
pre-fit). What graduates, if this survives 183 held-out filings, is the
claim that *transport between any two models in the class is determined
by each model's single relationship to a common hub* — the atlas claim,
at map order.

### ADDENDUM 2026-07-28-F — frozen-corpus repair (corpus-v2) — ratified by Luxia 2026-07-28

**Background on record.** The frozen collection corpus (manifest sha
`a6712ca0…`, "v1") contains 3 corrupted entries — n_tokens 2, 7, 26,
n_words = 1, truncated mid-word (desk record
`REPORT-pull-qwen3-2026-07-28` §finding; reproduction recipe in its
diary). On Qwen-lineage raw arms they explode to ~100× median state
norm; one sits in held-out topic 0 carrying 97.9% of the raw test
set's squared norm. Llama/Mistral lineages absorb them; the native
arm is statistically unharmed everywhere.

1. **The repair (deterministic, no judgment):** the 3 entries are
   REMOVED. Corpus-v2 = v1 minus exactly those 3 text_ids (780 → 777
   texts; every other entry byte-identical). The v2 manifest sha is
   computed once, recorded here at ratification, and becomes the
   collection basis of record. No entry is edited or substituted —
   exclusion only, so no new content enters the frozen apparatus.
2. **What STANDS, permanently:** every filed prediction, every frozen
   band, every scored verdict, and every stamped result to date —
   all scored on v1 banks, all immutable. v1 banks and fits are
   RETAINED on disk, labeled by their corpus sha; nothing is deleted.
3. **What RE-DERIVES on v2 (the re-bank):** state banks for every
   banked model (collect + bitwise spot-replay gate, per job — the
   standing discipline); hub maps; the â column; the c_M constants
   (native and raw, same solve chain, same anchor procedure); the
   composed-path maps. Steering vectors re-derive only if their build
   recipe consumes the corpus (per each build's own stamp); otherwise
   they carry over unchanged, stated per model. The split of record
   re-computes deterministically on 777 (same seed, same topic-group
   rule; new split_sha recorded).
4. **The v1↔v2 comparison table (mandatory, filed before any v2
   filing):** every re-derived quantity of record side-by-side, with
   pre-stated expectations: (a) native-arm quantities move ≲ .01 in
   â-units (the 3 rows are benign in native); (b) Qwen-lineage raw
   clouds recover (effective rank ≳ 20 at mid-depth, from ~1.1);
   (c) the composed predictor retrodicts the v2 â of the already-fit
   pairs within ±.05 using v2 maps (the mechanism claim is
   corpus-robust or we want to know). Any expectation violated is a
   named finding, not silently absorbed.
5. **Go-forward rule:** from the first v2 filing onward, predictions
   file on v2-derived constants and maps, and every filing record
   carries the v2 corpus sha. The raw-arm quotability holding lifts
   for models whose v2 raw clouds verify healthy. G-star-hit,
   G-comp-hit, budgets, and all frozen gates continue uninterrupted —
   they are properties of filed-vs-observed, not of the corpus
   vintage, and both predictors race on across the repair.
6. **Data baseline:** `manifests/outputs.sha256` regenerates at the
   v2 freeze point (desk-only, standing rule), covering v1 and v2
   trees side by side.

### ADDENDUM 2026-07-28-G — corpus-v2.1 · the â vintage rule · the DSV3 loading mechanism — ratified by Luxia 2026-07-28

Execution note of record (Luxia, at ratification): the v2.1 build +
re-bank does NOT fire until the currently in-flight wave (405B scan +
fits, DSV3 materialization + scan + fits, SSM downloads) has landed
and been verified — one clean starting point, no collisions. The
whitened-bank builds (3b/8b/qwen-7b) ride the same next wave.

Three dated items, each append-only; no filed band or scored verdict
moves.

**G1. Corpus-v2.1 (completing F's repair).** F§1's exclusion criterion
was a symptom threshold (M21): it removed the 3 shortest corrupt
entries and missed two 512-token exact repetition loops of the same
disease (`S3-dsv2-lite-contrastive-t00-r0`, `S3-dsv2-lite-socratic-
t12-r0`), both in held-out topics; on v2 one alone carries 98.99% of
qwen2.5-32b's raw test squared norm. Corpus-v2.1 = v2 minus every
entry that is an exact repetition of a cycle ≤ 64 characters
(deterministic detector; census: exactly 2 such entries remain) —
775 texts, exclusion-only, byte-preserving, same two-sided
construction gate as F§1 (builder refuses unless its round-trip
reproduces the v2 manifest sha `6c4d65ba…`). The v2.1 manifest sha and
split sha are recorded in the ledger at construction. Re-derivation
per F§3–4 (same comparison-table obligations; expectation revised per
G2). **The 282 byte-BPE-encoded entries REMAIN by ruling**: they are
consistent shared inputs (every model saw identical bytes); the
corruption that harms is degeneracy, not encoding; decoding would
change content of 36% of the corpus for no demonstrated benefit.
Go-forward basis for all future filings = v2.1 (ruled); every filing
carries its corpus sha.

**G2. The â vintage rule (the F§4(a) finding, on record).** Removing 3
of 780 corpus rows moved native â by median .035, max .092 (map term
≤.078, vector term ≤.088; held-out r² static): **â at proc_k128 /
n_train≈600 carries corpus-sampling variability of order ±.05–.09.**
Rules, ratified: (a) every quoted â carries its corpus manifest sha
(vintage-tagging); (b) within-vintage scoring is unaffected —
predictions and fits ride the same banks, which is exactly what the
frozen ceremony tests — so no band, gate, or budget changes; (c)
cross-vintage â comparisons are calibration reads, never scored; (d)
F§4(a)'s ±.01 expectation is retired as mis-calibrated (the violation
was the expectation's, not the apparatus'); the v2→v2.1 comparison
table expects native-â movement within the measured ±.09 envelope
and flags beyond it; (e) growing the corpus (shrinking the wobble at
source) is a named rolling item for a future collection phase.
Context on record: the composed predictor retrodicted within .041 on
BOTH vintages — the mechanism is corpus-robust; â is
vintage-relative.

**G3. DSV3 loading mechanism (correcting Addendum A's named
mechanism).** Addendum A's regime — bf16 forward, standard bitwise
spot-replay gate, differentiable target builds, fp32 banking — is
UNCHANGED and is achieved. Its named mechanism
(`FineGrainedFP8Config(dequantize=True)`) is **not executable on this
checkpoint**: `kv_a_proj_with_mqa` ([576, 7168], present in all 61
layers + dense layers) has a ragged final 128-block whose stored
scales transformers' dequantizer hard-rejects. Mechanism of record
becomes: a **streaming dequantizer** producing a bf16 mirror
(`DeepSeek-V3-bf16` on the shared volume), verified bitwise against
transformers' own `Fp8Dequantize.convert` on all 122 comparable
tensors of the probe shard; ragged tensors use the unique consistent
extension of the block rule (rows 512–575 → scale row 4, DeepSeek's
own reference behavior), recorded per-tensor in the materialization
manifest with the source checkpoint's shard shas. The scan collects
the mirror as an ordinary bf16 sharded node; its preflight asserts
the provenance chain (source config sha `cbf0b95d…`, first-shard
`b933b099…`, manifest verification.failures == []) and measures the
expert-fusion transient live, blocking on overrun. The mirror is
reusable for the row-21 vector builds.

**G3 correction (2026-07-28, later same day — ratified by Luxia):**
the streaming dequantizer's probe-shard bitwise verification was
real but structurally incomplete: it sampled only tensors whose
block scales reside in the same source shard, while the tool's scale
lookup resolved per-shard — so 155 tensors whose scale companions
landed in an adjacent shard were written through as raw FP8 in the
first materialized mirror (one per affected shard boundary, 59 of 62
layers; a by-value bf16 load casts them ~7,250× too large). Found by
the fused-build lane's added pre-gate BEFORE any DSV3 state was
collected — no bank is contaminated, no filed or scored quantity is
touched. The mechanism of record is unchanged; its implementation
and acceptance gate are corrected: (a) scale companions resolve
against the whole-checkpoint index, never one shard's key set;
(b) a materialized mirror is accepted only on a full dtype census of
the OUTPUT (zero FP8-typed tensors) plus an asserted identity
between the manifest's own scales-consumed and tensors-dequantized
counts, in addition to the existing per-shard sha chain. The
defective first mirror is deleted by ruling; the corrected mirror
re-materializes from the fp8 checkpoint under the corrected tool and
becomes the dequantization of record (its manifest sha recorded in
the desk ledger at acceptance).
