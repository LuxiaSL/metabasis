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
the frozen contracts and the manifests are here. The state banks, fitted maps, prediction
artifacts and scored outputs behind the numbers below are not; they are available on
request. Hashes let an obtained copy be checked; they do not show when an artifact was
made. Pair predictions share models, hub legs and corpus rows, so hit counts are
descriptive summaries, not independent trials.

## Established

### Composed transport predicts exchange rates ahead of observation

Every model gets one linear map into a hub model. Any pair A→B is predicted by composing A's
hub leg with B's, before the pair's own direct map is fit. The prediction already uses both
models' native vectors and their fitted hub legs; what it predicts is the direct fit. On the
clean web-text corpus (webtext-v3), 240 predictions (ordered pairs; 120 unordered) were
filed and hashed before the direct fits. All 240 cleared the pre-registered |â| ≥ 0.08
floor, so none fell back to magnitude-only scoring. Against the directly fitted exchange
rates:

| hub | within ±0.05 | within ±0.025 |
|---|---|---|
| Qwen2.5-3B (hub of record) | 240/240 | 92.9 % |
| Llama-3.1-8B (the pre-registered primary column) | 240/240 | 225/240 (93.8 %) |
| Qwen2.5-32B | 240/240 | 94.2 % |
| Llama-3.2-3B | 240/240 | 96.3 % |
| Gemma-3-27B (designated poor-hub control) | 238/240 | 190/240 (79.2 %, fails) |

The one failure is the control failing the tight band, which the pre-registration named in
advance. Of the 240, the 84 predictions involving at least one of the three models added
last land 84/84 on every hub.

**A descriptive comparison: the frame against a single number per model.** The
pre-registration scores this comparison descriptively, not as a gate. The star factorization
predicts each pair as a product of one coefficient per model. It lands 163/240 in-sample and
143/240 held out (leave one unordered pair out), against 240/240 for composition filed in
advance. On identical
predictions, 77 land for composition only and 0 for the star only. The star breaks
one-sidedly on same-family pairs, which it under-predicts in 16/16 cases. Whether any
scalar summary is adopted as the law is decided once the last model (DeepSeek-V3) is in.

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
- **The law.** Whether the star factorization, or a structure richer than one number per
  model, is stated as the law once the last model is in.
