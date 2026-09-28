# Results

What metabasis has found, what it is still testing, and what its data rules out. Every item
carries its status:

- **Established:** pre-registered and scored against frozen criteria, or re-derived
  independently from raw data.
- **Exploratory:** measured and reproducible, but not pre-registered. It is a lead, not a
  claim.
- **Rejected:** a reading this project's data rules out.
- **In progress:** designed, often stated in advance, not yet run.

Numbers carry their scope. An exchange rate is a property of (model pair, corpus, site). A
behavioral effect is a property of the instrument that read it. Entropy effects name their
channel (see "Two entropy channels").

The two frozen contracts are in this repository.
[`docs/planning/PREREG-webtext-v3-2026-08-03.md`](docs/planning/PREREG-webtext-v3-2026-08-03.md)
is byte-identical to its copy at tag `freeze/webtext-v3`.
[`docs/planning/PREREG-transport-campaign-2026-07-26.md`](docs/planning/PREREG-transport-campaign-2026-07-26.md)
begins with the exact bytes tagged `freeze/transport-campaign`, followed by appended
addenda. To check: `git show <tag>:<path>`.

**What is and is not verifiable from this repository.** The code, the corpus reconstruction,
the frozen contracts and the manifests are here. The webtext-v3 prediction artifact, its
seal, the scored record and the star comparison are in
[`evidence/webtext-v3/`](evidence/webtext-v3/), and
`python -m metabasis.scripts.recompute_webtext_v3_evidence` rebuilds the headline counts
from them. The state banks, fitted maps and other readouts behind the remaining numbers are
not in the repository; they are available on request. Hashes let an obtained copy be checked; they do not show when an artifact was
made. Pair predictions share models, hub legs and corpus rows, so hit counts are
descriptive summaries, not independent trials.

## Established

### Composed transport predicts exchange rates ahead of observation

Every model gets one linear map into a hub model. Any pair A→B is predicted by composing A's
hub leg with B's, before the pair's own direct map is fit. The prediction already uses both
models' native vectors and their fitted hub legs; what it predicts is the direct fit. On the
clean web-text corpus (webtext-v3), 240 predictions (ordered pairs; 120 unordered) were
filed and hashed before the direct fits, and the seal was published as a public gist at the
same moment (link in the evidence README). Against the directly fitted exchange rates:

| hub | within ±0.05 | within ±0.025 |
|---|---|---|
| Qwen2.5-3B (hub of record) | 240/240 | 92.9 % |
| Llama-3.1-8B (the pre-registered primary column) | 240/240 | 225/240 (93.8 %) |
| Qwen2.5-32B | 240/240 | 94.2 % |
| Llama-3.2-3B | 240/240 | 96.3 % |
| Gemma-3-27B (designated poor-hub control) | 238/240 | 190/240 (79.2 %, fails) |

The one failure is the control failing the tight band, which the pre-registration named in
advance. Of the 240, the 84 predictions involving at least one of the three models added
last land 84/84 on every hub. How the 240 are counted, scored by arm, and treated near zero
is set out under "Pre-registered obligations" below.

**A descriptive comparison: the frame against a single number per model.** The
pre-registration scores this comparison descriptively, not as a gate. The star factorization
predicts each pair as a product of one coefficient per model. It lands 163/240 in-sample and
143/240 held out (leave one unordered pair out), against 240/240 for composition filed in
advance. On identical
predictions, 77 land for composition only and 0 for the star only. The star breaks
one-sidedly on same-family pairs, which it under-predicts in 16/16 cases.

On the earlier corpus, the campaign contract made the star a gate: at least 80 % of 190
pair predictions in band. **That gate failed:** 36 of 134 scored predictions landed, and at
most 47 % is reachable whatever the remaining 54 do. The composed predictor, pre-registered
as an addendum to the same contract, passed its gate (126 of 127 scored) and beat the star
by more than the pre-registered margin. The scalar summary is not the law; the composed
frame is what predicts.

**Exchange rates decompose as overlap × ceiling:**

â(A→B) = cos(ζ_A, ζ_B) · ceil_B

This is the two models' directions aligned inside the hub frame, times the fraction of B's
direction the map's image can hold. It is an algebraic identity, verified numerically to
3.5e-9 on 11 model charts and 19 pairs (fitting-v21 corpus). The star's fitted per-model
coefficient follows the ceiling less tightly (r² = 0.84).

### The entropy-gradient vector steers at every scale tested

A label-free steering vector, built from the gradient of output entropy, was tested in 21
models from 3B to Llama-3.1-405B. It raises per-token entropy in 20 of them. At every one of
the 20 the effect is ordered by dose and lies outside a matched-norm random band at 6 of 6
doses. For 19 models that holds at the standard doses; for Mixtral-8x7B it holds on a
narrower ladder. The per-model table,
the construction, and the one model that did not calibrate (Gemma-3-27B, whose random
controls swing too widely at the sites tested) are in the
[entropy-gradient repository](https://github.com/LuxiaSL/entropy-gradient).

### At 405B, the perpendicular component decided transfer between two legs

At Llama-3.1-405B, entropy vectors transported from two source models align almost equally
with the target's native vector (cos .234 vs .264). Yet their effects differ by about 50×
(local channel; 1.65× in the total). The component orthogonal to the native direction
decides it:

- **In-family leg (from Llama-3.1-70B):** its perpendicular part alone is exactly orthogonal
  to the native vector. It still steers, dose-ordered (ρ +1.00) and outside its random band
  at 6/6 doses, recovering 42 % of the full vector's effect (local channel; 26 % in the
  total).
- **Hub leg (from Qwen2.5-3B):** its perpendicular part does not carry (ρ −0.66,
  sign-inconsistent).

The carry and no-carry verdicts hold in both channels. This is two legs at one target: it
shows the perpendicular component's contribution there, not a general law of transport. The prediction was fixed before the
run in a pre-statement that is not yet published in this repository (sha256
`a92644b343c7e1a3719e27cdb283306203d8e5b54990dce294e42391ae83df03`).

### Class vectors transfer visibly at small and mid scale

Contrast vectors carried from Qwen2.5-3B into five 7–16B targets move visible behavior for
language and sentiment. Refusal stays flat, and the readout for it is still being checked.
The rate at which a transported language vector makes a model write French tracks map
alignment in rank order:

| map alignment (cos) | transported / native effect |
|---|---|
| .188 | 0.00 |
| .532 | 0.28 |
| .564 | 0.98 |
| .568 | 1.00 |
| .650 | 1.02 |

That is five points, so read it as texture, not a law.

At Llama-3.1-70B the two axes behave differently:

- **Language is a cliff.** The native vector switches the model to French in one dose step.
  The transported one acts below that step and writes no French.
- **Formality is a dial.** The transported vector moves it (ρ 0.94, outside its band at 4
  of 6 doses, including both extremes).

### Tensor-parallel inference can change the entropy readout, not the behavioral one

Splitting a model across GPUs changed the steered entropy effect at a ±0.3 dose by 29–39 %
on two dense columns: Qwen2.5-7B language at 2 GPUs, and Llama-3.1-70B at 8 (effect
measured against the baseline cell). Four other columns at 2 GPUs agreed within 0.5–4.8 %,
and a DeepSeek-V2-Lite column within 0.4 %. The size of the effect varies by model and
dose, and its mechanism is open. It is the parallelism itself: two machines reproduce each
other to six decimals.

The behavioral readout does not inherit it. A classifier scoring the generated text agrees
across 1 and 2 GPUs as follows:

- **Dense model:** within 0.23 % per cell for cells whose rate clears twice its standard
  error, and within 0.1 standard errors for the rest.
- **MoE model:** within 4.7 % on every cell.

Behavioral results at 405B are therefore compared across GPU splits. Entropy magnitudes
across split sizes are not.

## Pre-registered obligations

Every obligation in the two frozen contracts, with its status. Passed and failed are against
the frozen criterion. Descriptive items have no pass state by design.

**How the 240 web-text predictions are counted.** The core roster has 21 models:

- the 17 non-hub models of the earlier roster that carry a registered layer (DeepSeek-V3,
  GPT-2-XL and OLMo-2 base do not);
- the hub Llama-3.1-8B;
- three additions: Qwen2.5-72B-Instruct, Llama-3.1-8B base and Qwen2.5-7B base.

Five core models serve as hubs and are never endpoints. That leaves 16 endpoints and
16 × 15 = 240 ordered predictions, over 120 fitted pairs read in both directions.

Each prediction is scored in exactly one template arm and never averaged across arms. There
are 156 native predictions (both models instruction-tuned) and 84 raw (at least one base
model). The gates are computed over the whole list, and each arm passes separately on every
working hub:

| arm | within ±0.05 | within ±0.025 |
|---|---|---|
| native | 156/156 | 90.4–96.2 % |
| raw | 84/84 | 96.4–98.8 % |

The control's tight-band failure is in the native arm (75.6 %).

Two near-zero rules apply, and neither changes a count. The tight band is scored only where
the observed exchange rate is at least 0.08; all 240 clear it (smallest 0.094). Predictions
below 0.08 would be scored on magnitude alone; the smallest in any hub column is 0.083, so
none is. The 84-prediction extension line is the subset touching one of the three additions
(60 raw, 24 native). It is a different set from the 84 raw-arm predictions, which touch the
base models.

**Web-text contract**
([`PREREG-webtext-v3-2026-08-03.md`](docs/planning/PREREG-webtext-v3-2026-08-03.md))

| obligation | status | result |
|---|---|---|
| rank health at k = 256 | passed | all three clauses; k = 256 stays the rank of record |
| fit validity against both nulls | passed | every fit the scoring uses clears both nulls |
| rank guard k ≤ n_train / 1.2 | passed | 0 violations (applied by the readers) |
| composed predictions, ±0.05 | passed | 240/240 on all four working hubs |
| composed predictions, ±0.025 | passed | 92.9–96.3 %; the poor-hub control fails at 79.2 %, as pre-named |
| extension line, ≥ 70 % | passed | 84/84 |
| corpus half-split and rank-sweep invariance | passed at the tier level | the full hub ordering does not replicate (ρ .17, .61): order within a tier is noise |
| quartile separation of hubs | failed | 3 of 25 top-versus-bottom pairs miss p < .01; the hub field is compressed. The contract names no consequence, so this stands as a scope finding |
| corpus-indexing measurement | measured | structure ρ ≈ .67; level shifts small |
| hub-quality predictors | not supported | no listed quantity predicts hub quality: hubness is real, but not predicted |
| base-versus-instruct sibling read | not quotable | the collected sibling pairs sit at different layers, which the contract excludes |
| scalar star beside composition | descriptive | 163/240 in-sample, 143/240 held out |
| authored-stratum ablation | not run | |
| behavioral tier contrast | pending | hub tiers and shared targets are fixed; no cell has run |

**Campaign contract**
([`PREREG-transport-campaign-2026-07-26.md`](docs/planning/PREREG-transport-campaign-2026-07-26.md),
with its addenda)

| obligation | status | result |
|---|---|---|
| scalar star, ≥ 80 % of 190 in band | failed | 36/134 scored; at most 47 % reachable |
| composed predictor, ≥ 80 % in band | passed | 126/127 |
| composed beats star by ≥ 15 points, sign-flip p ≤ .01 | passed | +77 points at the pre-registered read (66 predictions) |
| rotation and map-permutation nulls for composition | passed at 66 predictions | not yet extended to all 127 |
| directional star and its constant-α companion | failed | 52/115; its gain over a constant-α model is calibration, not per-model structure |
| hub-invariance of per-model coefficients | failed | differences up to 0.40 against a 0.05 band |
| forward/reverse asymmetry analysis | done | the two directions differ by a ratio of ceilings, an algebraic identity |
| naive-transplant null | done | |
| the 190-prediction enumeration | partial | 134 scored, 2 withdrawn, 54 open |
| composition panel, mapping-flexibility benchmark, tokenizer-divergence regression | not run | |
| family-residual regression, per-class star | not run | |
| hub drop-out | partial | single instances only |
| constant-mean and shuffled-star baselines | not run | moot once the star gate failed |
| judged legs, class-split table, entropy-write column | partial | the behavioral design moved to the web-text basis; see "In progress" |

## Exploratory

### Why hub transport weakens at the largest model: the anchor frame loses the direction

A map fit on generic text can only carry a direction its anchor frame spans. The table
gives, for each target, the fraction of the native class direction that frame holds (k =
256). It falls with scale:

| target | language | formality |
|---|---|---|
| DeepSeek-V2-Lite | .76 | .89 |
| Qwen2.5-7B | .74 | .91 |
| Llama-3.1-70B | .47 | .76 |
| Llama-3.1-405B | .25 | .53 |

This bounds alignment from above (alignment ≤ containment) and matches every class outcome
measured so far. It is a necessary condition, not the whole account.

The fall has two regimes:

- **7B to 70B: dimensional dilution.** The share of generic-text variance along the
  direction stays flat within about 3 %.
- **405B: a genuine drop,** 2.3× on language and 1.75× on formality.

More generic text helps slowly: on language, about +0.01 containment per 235 texts, and
flattening (fitting-v21 corpus).

### A few axis-relevant texts restore it

The numbers in this section use the fitting-v21 corpus.

Adding 8 contrast pairs for an axis to the anchor set (8 texts with the property, 8
without) lifts hub→405B alignment on language from .05 to .82. Held-out generic-state
prediction barely moves: r² changes by ≤ 0.0042 at up to 16 pairs.

The lift is mostly local to the axis, with one exception: formality rises with any axis's
seeds, since register varies in every text set. Seeds drawn from pairs the evaluation never
sees still lift language from .05 to .79.

- **Parity with native construction.** Give both methods the same few target pairs:
  building the vector natively in the target, or transporting it through a seeded map. They
  produce comparable geometry, and which one wins depends on the reference vector. So
  transport's case is not per-vector superiority. It is carrying source-side objects,
  serving a library of vectors, and moving vectors that have no target-side construction.
- **One map serves many vectors, but each vector needs its own seeds.** A single map seeded
  with pairs from all four axes keeps each axis's own-seeded performance. It is higher in
  131 of 144 comparisons (by +0.02 on average), and shows no interference at up to 8 pairs
  per axis. A single-axis map given the same total budget still wins its own axis (142 of
  144).
- **Composition survives seeding.** Composing two seeded legs through an intermediate model
  (Qwen2.5-3B → Llama-3.1-70B → 405B) matches the direct seeded map within 0.011 cos.

Seeded maps are a separate instrument from the generic-only maps. Their numbers never enter
the composed-prediction results above.

### Two entropy channels

Take a steered model's entropy on its own generations, minus the unsteered baseline's
entropy on its own generations. That total splits exactly into two parts:

- **Local:** the steered model against the unsteered model on the *same* text.
- **Trajectory:** the change in *what gets written*, measured as the unsteered model's
  entropy on the steered text minus its entropy on the baseline text.

At Llama-3.1-405B, dose +0.3:

| vector | local | trajectory |
|---|---|---|
| native entropy vector | +0.10 | +1.51 |
| hub-transported | +0.0005 | +0.14 |
| in-family-transported | +0.025 | +0.21 |

The hub leg fails in the local channel: it is outside its random band at only 2 of 6 doses.
It carries in the total, at 5 of 6 including both extremes. Verdicts are stated in the
local channel, which the frozen criteria use, and name their channel.

### Layer correspondence is shallow, not depth-proportional

Held-out prediction (r²) from 11 of 12 Qwen2.5-3B layers (14–86 % depth) picks one shallow
target layer: 21 % depth in Llama-3.1-70B, 15 % in 405B. Under four other similarity
measures the best target moves, but it stays shallow (≤ 40 % depth in 70B, ≤ 34 % in 405B),
and none restores depth-proportional matching. Matching by fractional depth gives about
half the best prediction quality at 70B and 405B.

Within a fixed target layer, maps vary smoothly with source depth. A transport scheme that
spans many layers needs a correspondence rule, not depth matching.

## Rejected

- **Cosine to the native vector as a sufficient predictor of transferred effect** (local
  channel). Two legs 13 % apart in cos differ about 50× in local effect, 1.65× in the total.
  This rules out cosine as sufficient, not every relationship between cosine and effect.
- **"Large models are intolerant of imperfect directions."** An angular-efficiency reading
  predicted 405B would reject any transported vector below an alignment bar. At 405B, on the
  same model, instrument and GPU split, the in-family leg carried at efficiency 0.92 against
  the hub leg's 0.021 (local channel).
- **The machine as the source of the tensor-parallel effect.** A single-GPU run on one
  machine reproduced another machine's column to six decimals. That rules out the machine,
  not a shared software implementation.
- **Reweighting the anchor spectrum as a 405B rescue.** The direction is near chance in the
  generic span even at full rank: .272 against a chance level of .218 (fitting-v21). The
  information is missing, not down-weighted.
- **"Contrast-only maps have broken nulls."** Random vectors through a map restricted to a
  64-dimensional image sit where the subspace-conditioned expectation puts them, E|cos| ≈
  c·√(2/(πk)) ≈ 0.10. What does hold is that contrast-only maps fail held-out generic
  prediction.

Not pursued, as a design choice rather than a measured result: anchor corpora made of a
model's own generations. They were tried before this project and did not help, and they
would tie a map to the models that wrote its anchors.

## In progress

- **The 405B class demonstration.** One of eight vector × leg runs is complete. Native
  language fires: the French rate goes from .0007 to .42 at +0.3, with every generation still
  scorable. Hub-transported language stays at the floor. Predictions for the remaining seven
  are stated in advance, from geometry alone: in-family language writes **no** French, and
  in-family formality carries the most.
- **An untargeted-diversity test.** 100 texts are drawn blind from a Common Crawl segment,
  in three independent draws. The prediction, frozen before any draw: language lifts
  substantially, formality moderately, sentiment weakly-to-moderately, and refusal not at
  all. If refusal moves, the proposed split between textural axes (properties text carries)
  and interactional axes (response modes) is refuted.
- **DeepSeek-V3.** The last model into the collection: collection, site selection, and
  language and formality class runs.
- **Multi-layer vectors.** Whether a multi-layer control vector's effect (as in
  [repeng](https://github.com/vgel/repeng)) survives reduction to one layer. This is measured
  natively before any transport of the family.
- **The last 54 campaign predictions.** All of them involve DeepSeek-V3, GPT-2-XL or
  OLMo-2 base, which enter the atlas with the last collection.
