# PRE-REGISTRATION — the webtext-v3 basis: full rebank, invariance program, hub law (FROZEN 2026-08-03; tag `freeze/webtext-v3`)

> **STATUS: FROZEN 2026-08-03.** Luxia stamped r2 (the two-reviewer
> adversarial revision; synthesis in the ledger) and delegated the
> freeze values. r3 completions placed at the freeze act, each
> disclosed in the ledger: the draw-construction specification
> (exact-quota grouped draw + deterministic residual repair; named
> streams via sha256(name)[:8]) · the half-internal holdout rate
> (35/stratum/half) · the grouping-key qualification · the v2.1
> manifest sha pin in §9 · the §5 staging-path pointer swap
> (sanitization). The freeze-time deterministic draws are derived,
> verified, and sha-recorded below (§13 executed); the
> zero-quantities sweep is recorded in the attestation. Collection
> may now fire.
>
> **Discipline (inherited from the parent, verbatim intent):**
> corrections append as dated addenda; park, don't amend. Verify
> this text against the TAGGED commit, never the working file.
>
> **Lineage:** standalone successor to
> `PREREG-transport-campaign-2026-07-26.md` (tag
> `freeze/transport-campaign`, sha `33ba8290…`), which remains the
> binding contract for every v2.1-era claim, unmodified. Adversarial
> inputs answered here: `CHALLENGE-SET-hub-law-2026-07-31.md` and
> the 2026-08-03 two-reviewer prereg pass.

## §1 The claim under test (one breath)

Steering-vector transport through a hub-and-spoke atlas — per-model
semi-orthogonal maps into a shared frame, every pair composed from
two hub legs — holds on a fitting basis that is structurally empty
of model-authored text; its observable (â) is corpus-robust in
STRUCTURE (in-band composition, tier ordering) and corpus-indexed in
LEVEL; and hub quality is a measurable, tier-stable property.
webtext-v3 is the PRIMARY basis of record for all forward claims;
the v2.1↔v3 contrast is retained as the measured corpus-indexing
result.

## §2 The basis: webtext-v3

- **Composition (built and censused; the freeze adopts the built
  draft):** four strata of public text, 300 texts each, total
  **1,200** — WikiText-103 validation (carrier; the v2.1 S2 chunker
  verbatim; revision `b08601e0…`) · a pinned C4-en slice (revision
  `1588ec45…`, April-2019 crawl) · PG-19 (revision `4d28bd77…`,
  per-book sha256 pins) · **stratum 4 RATIFIED (Luxia, 2026-08-03):
  `flax-sentence-embeddings/stackexchange_title_body_jsonl` @
  `a3d99bf2…`** (2021-06 dump, six prose-dominant sites,
  QUESTIONS-ONLY — the Q+A variant was rejected on an NC license +
  undocumented vintage; the CC BY-SA claim rests on the card body +
  upstream SE policy, verified at the pin, builder HALTs if it
  vanishes). Chunking identical across strata (150–500 tokens,
  close at 250; tokenizer pin = Llama-3.1-8B-Instruct, v2.1's).
  **Seed 80** throughout (corpus, split, halves, ablation) — the
  built draft used it, so the frozen corpus IS the verified draft,
  byte-identically.
- **The pre-LLM-era pin (design principle, quotable):** every source
  is a pre-2022 snapshot — the corpus cannot contain model-authored
  text by construction, not by filtering.
- **Census at freeze (M47):** counts by source/stratum, revisions,
  licenses, vintage — beside the sha, always. **The census at freeze
  additionally enumerates, by text_sha256, the 59 wikitext chunks
  byte-identical to v2.1 S2 entries.**
- **Publication:** the reconstruction script (`build_webtext_corpus.py`,
  merged `933f2df` + `RECONSTRUCT.md`) + the metadata manifest with
  per-text `text_sha256`. Raw bodies are not republished. **Manifest
  sha `b85f4d169ed0cb882a5690e3509308a5807e056b6a491f908014aee8e7b6085f`**
  is the basis identity every artifact cites (determinism proved
  twice, incl. a separate-process rebuild).
- **Split (rule, not target):** holdout 280/1200 by count,
  per-stratum, grouped, **seed 80**. Grouping keys per stratum:
  c4/stackexchange = source row (qualified by source file — a bare
  row id would collide across the six SE site files) · pg19 = book
  · **wikitext = individual chunk (NAMED LIMITATION: the v2.1 S2
  pool carries no article identity, so wikitext holdout chunks are
  not article-independent — disclosed, not papered over).** Draw
  construction (r3 completion, disclosed): exact-quota grouped draw
  per stratum — seeded permutation, greedy fill — with the
  deterministic residual-repair swap when group sizes cannot
  express the quota exactly (the rule and every firing recorded in
  the artifact; an unreachable quota HALTs, nothing is rounded).
  Named seeded streams map to spawn keys by sha256(name)[:8] (the
  construction stated in full inside every artifact).
- **The v2.1-overlap rule:** the 59 shared wikitext chunks are
  assigned to the TRAIN side by rule — no held-out, race-scored, or
  gate quantity is ever computed on a text any prior result touched.
- **Computed at freeze (deterministic functions of frozen inputs;
  shas recorded in the freeze act, §13):** the realized train/test
  membership · the I1 half-split memberships · the §9 ablation
  text_id list.

## §3 The rebank (full — nothing v2.1-fitted survives into v3 claims)

1. **States:** per-text mean residual states at each model's site of
   record, both template arms, full roster (§5). Existing sites
   carry over (site identity is a property of the model, not the
   corpus); new models get the standard site-curve ceremony.
   Collection + bitwise spot-replay in one job, via the templated
   jobs-v21 lanes (basis = webtext-v3 parameter).
2. **Fits:** per-side PCA (train rows only) + semi-orthogonal
   Procrustes, families native/raw × proc_k, **k ∈ {32, 128, 256}**
   (all guard-legitimate full-corpus AND per-half at the §2 sizing;
   k512 excluded — forbidden per-half). **Family of record =
   native::proc_k256** for instruct↔instruct slots; any slot with a
   base-model endpoint is scored in **raw::proc_k256** (the parent's
   arm rule, inherited); arms are never pooled in any aggregate.
   k128 and k32 always fit beside.
3. **The k256 health check (scored on the incumbent's hub legs at
   first fit, BEFORE any prediction files; binding):** (a) per-half
   k_eff/n_train < 0.9 in every family; (b) the k256 composed-error
   distribution does not undercut k128's median by more than 5× (the
   S2 saturation signature was 9×); (c) held-out r² at k256 ≥
   held-out r² at k128 − .05 on ≥90% of legs. **Any clause failing ⇒
   rank of record demotes to proc_k128 by this clause alone**, k256
   becomes a labeled beside, the demotion is quoted in every
   readout, and no other clause of this document moves. The check
   consumes no predictions (hub legs are not scoreable pairs, §8).
4. **Fit validity (the both-nulls rule, quoted in full — the prior
   draft cited a nonexistent parent section):** a fit is used only
   if held-out state-prediction R² separates from BOTH the
   shuffled-pair null AND the stratum-preserving shuffle null
   (re-pair with a different text from the same stratum group — the
   sharp null), and per-stratum R² carries in ≥2 of the four strata
   {wikitext, c4, pg19, stackexchange}. CKA before/after and the
   two-arm g-agreement read are quoted beside.
5. **Vectors:** entropy-gradient vectors rebuilt on webtext-v3
   states, FD-gated, thread-config stamped (OMP=8 ruling).
6. **The vintage rule (absolute):** no v2.1 constant is ever quoted
   against a webtext-v3 observation, and vice versa.

## §4 Rank guard (inherited, restated)

k quotable only where k ≤ n_train/1.2 for the split actually used
(full-corpus, and per-half for §6). Arithmetic at the frozen sizing:
full n_train = 920 → k ≤ 766 (32/128/256 quotable); per-half
n_train ≈ 460 → k ≤ 383 (32/128/256 quotable — the sizing was
chosen for exactly this). Realized counts re-checked at first fit;
any violation HALTs the family rather than quoting it.

## §5 Roster: core and extension

- **CORE roster (gate denominators are defined over core pairs
  only):** the 17 collected star nodes + Qwen2.5-72B-Instruct +
  Llama-3.1-8B base + Qwen2.5-7B base. Sites of record per
  `metabasis/roster.py` SITE_OF_RECORD **at the freeze commit** (the
  registry is the single authority). Core models without a
  registered site at freeze (the three adds) receive the site-curve
  ceremony inside their collection window, registered BEFORE any fit
  of theirs is quoted. **A core model failing to complete
  collection/registration HALTs the ceremony for a Luxia ruling —
  a denominator never silently shrinks.**
- **EXTENSION set:** dsv3 (FP8 split-semantics evidence pass owed) ·
  gpt2-xl (3-site calibration ruling) · olmo2-base. Their slots
  enter ONLY the doubly-held-out sub-line (§8); any that fails
  registration is reported absent WITH its named blocker — never
  silently dropped from a quoted rate.
- Behavioral tier (falcon-mamba, zamba2) unchanged. Weights for the
  adds staged and sha-certified 2026-08-03 at pinned revisions
  `495f3936…` / `d04e592b…` / `d1497293…` (staging record and
  placement in the desk ledger).

## §6 The invariance program (HL-1 — the centerpiece)

**Binding clauses are TIER-LEVEL; full-ordering statistics are
quoted beside with pre-named partial outcomes** (the field's top is
expected statistically tied — C6; a full-order miss with tier
agreement is a scope limit, not a refutation):

- **I1 corpus half-split.** Halving: per-stratum, grouped by the §2
  keys, seed 80, membership list computed and sha-recorded AT
  FREEZE. Each half takes the §2 split rule internally at §2's
  holdout RATE (35 per stratum per half — 140 of 600).
  **Binding:** every §7 eligibility decision agrees across halves.
  **Beside:** full-order Spearman ρ over all guard-quotable hubs at
  the family of record (population enumerated at first fit), ρ ≥ .8
  the pre-stated pass; ρ < .8 with the binding clause passing is
  quoted as "within-tier order is noise; tiers are invariant" — a
  pre-named partial outcome.
- **I2 rank sweep** ({128, 256} — a NAMED deviation from the
  challenge set's suggested {32, 96/128}; rationale:
  capacity-rank centering; **k32 is always quoted as a labeled
  continuity beside — the only rank connecting to the prior
  record**). Binding: eligibility agreement across ranks. Beside:
  full-order ρ ≥ .8, same partial-outcome rule.
- **I3 aggregation (descriptive, stated once):** median vs Condorcet
  hub rankings; disagreement is quoted as the named
  ordering-instability result, refuting full-order (not tier)
  invariance. I3 has no refutation weight beyond that sentence.
- **I4 separation floor (binding):** EVERY top-quartile ×
  bottom-quartile hub pair differs by paired sign test on shared
  slots, Holm-corrected, all pairs p < .01; population = all
  guard-quotable hubs at the family of record, enumerated at first
  fit.
- **I5 the corpus-indexing measurement (v2.1↔v3; a measurement with
  NO pass state).** Pinned: matched **native::proc_k128** (the only
  rank quotable in both bases' guards; v2.1's family of record),
  matched arm per slot, over exactly the v2.1 slots scoreable in
  both systems (realized slot list recorded in the artifact). One
  computation; Spearman ρ and median signed level shift always
  quoted together. **Beside: the same computation excluding the 59
  shared wikitext texts' influence (legs refit on the
  shared-text-free train subset) — the overlap-clean form.**

## §7 The hub law (HL-2) and the selection criterion

**Race slot basis, fixed:** the identical shared slot set for all
five race hubs — all scoreable pairs (§8 definition) whose endpoints
are OUTSIDE the race set {qwen2.5-3b, qwen2.5-32b, llama-3.2-3b,
8b, gemma3-27b}, both directions, arm per the §3 rule, at the family
of record. This set is enumerated in the prediction artifact BEFORE
any hub leg is fit. No alternative basis is quotable except as a
labeled beside.

**Selection criterion (ratified 2026-08-03; hardened per review):**

1. Candidates: {qwen2.5-3b, qwen2.5-32b, llama-3.2-3b,
   8b-incumbent} + gemma3-27b as predicted-poor control. All roster
   hubs raced descriptively beside (after scoring; §8 step 5).
2. Metric: median composed |â_comp − â_obs| on the fixed shared
   basis.
3. **Top-tier membership:** H is expelled only if BOTH (a) H loses
   the paired sign test against the leader on the shared basis
   (two-sided, ties dropped, n stated with every p), p < .05
   Holm-corrected across the candidate set, AND (b) H's median
   exceeds the leader's by ≥ δ = .005 (one-tenth of the primary
   band — the practical-equivalence margin, frozen here). A
   significant-but-smaller-than-δ difference is quoted as
   "statistically distinguishable, practically equivalent" and does
   not expel.
4. **Invariance eligibility:** top-tier in both halves and at both
   ranks (§6 I1/I2), with the noise-ejection guard: a candidate is
   ineligible only if it is expelled in the SAME direction in ≥2 of
   the four contexts.
5. **Selection:** hub of record = the SMALLEST eligible candidate by
   total parameter count, config-derived at the pinned revision and
   recorded in the race artifact (expected values: qwen2.5-3b ≈
   3.09B < llama-3.2-3b ≈ 3.21B < 8B < 27B < 32.8B; the
   config-derived number is binding, not this expectation).
   Rationale pre-stated: the top tier is expected tied; the tiebreak
   is the authoring-platform economics — smallest adequate model
   wins, falsifiably.
6. Named fallbacks: no eligible candidate → no crowning (hubness
   per-roster-measured; incumbent continues as
   convenience-with-caveat). Gemma top-tier → a named hit on the
   predictor layer, quoted wherever the battery is.

**Disclosed expectation (review F10, owned):** under the S2 prior
all four candidates land top-tier and the crowning is the smallest
candidate's to lose. The informative outcomes are therefore (a) any
candidate FAILING eligibility, (b) gemma entering the tier, (c) the
invariance legs — not the winner's identity.

**Predictor slate (named here; computed only after the race lands;
circularity guards binding):** bank-only chance-corrected P1b · P2 ·
P6 (**P2/P6 count as ONE cluster in every multiplicity null — their
ρ was −.93**) · family-excluded P1a computed on a disjoint fit
split (legs fit on the split half not used for the hub-quality
measurement). P1b must retain |ρ| ≥ .4 after partialing out own-leg
r². Full-ordering scoring only — no sub-setting, ever. **HL-2
SUPPORTED iff** ≥1 named predictor clears |ρ| ≥ .6 on the full
ordering AND its max-statistic permutation p < .05 AND it holds
|ρ| ≥ .4 in each I1 half independently. **The positive claim
wording of record, verbatim:** "hub quality on webtext-v3 is
predicted by [predictor] at |ρ| = […]; all other slate members
quoted beside." No other positive phrasing is quotable.

**Pre-registered roster-add reads (review: no checkbox
compliance):**
- **C2 read (the recipe-vs-range test):** qwen2.5-72b's descriptive
  hub rank and median |e|, against two pre-named outcomes:
  top-tier-equivalent at 72B supports the recipe reading;
  degradation to llama-3.1-70b's tier supports the range-artifact
  reading. Statistic: paired sign test qwen72b-vs-llama70b on their
  shared slots.
- **C4 read (post-training, siblings only):** per sibling pair —
  llama-3.1-8b, qwen2.5-7b, **olmo2** (three controlled pairs) —
  base-vs-instruct hub quality and â deltas in the raw system at
  the SAME registered site per pair; a pair whose sites differ is
  quoted site-confounded and EXCLUDED from the C4 read (the F3
  lesson). Descriptive effect sizes with permutation nulls; no gate.
- **Declined from the challenge set, on the record:** small-gemma
  (cost) · the 2-checkpoints×2-sites post-training grid (OUT OF
  SCOPE for v3; no v3 claim touches the post-training question
  beyond the sibling read above).

## §8 The prediction ceremony

**Scoreable pair:** any ordered pair (A→B), A≠B, both endpoints in
the CORE roster (§5), NEITHER endpoint in the race set (§7). The
frozen count N and the full prediction-ID list
(`v3-prediction/<src>→<tgt>/<arm>-k<rank>`) are written into the
prediction artifact; **the denominator of every gate is that
artifact's list and never changes after it is sha'd.** Extension-set
slots form the doubly-held-out sub-line, filed in the same act,
separately labeled.

Order is the ceremony, mechanically enforced:

1. Fit hub legs ONLY for the five race hubs (race hubs' legs are
   not scoreable pairs).
2. **File ALL composed predictions** — the of-record column through
   the 8b incumbent (pre-stated: predictions do not get to pick the
   winning hub); all four other race hubs' composed columns filed
   simultaneously, identically banded. **The prediction artifact is
   sha'd and timestamp-attested BEFORE the first direct-pair fit
   job is submitted; the fit lane refuses to run without the stamp
   (the census no-peeking mechanism, made binding).**
3. Fit scoreable pairs → â_obs. Score.
4. Only then: the §6 legs, the §7 race/criterion/battery, and
5. the descriptive all-roster hub race (its remaining legs are the
   now-fit pair maps — the same fitted objects, acknowledged).

**Bands and carve-out:** ±.05 primary; ±.025 beside-with-teeth (see
gates). **The near-zero carve-out is inherited verbatim from the
parent: any prediction with |â_comp| < .08 is scored MAGNITUDE-ONLY
(|observed| within band of |predicted|; sign unscored).** The
floor-clearing count (|â_obs| ≥ .08) and the near-zero count are
quoted with every gate readout.

**Gates:**
- **G-comp-v3 (continuity floor):** composed in-band ≥ 80% at ±.05
  over the frozen scoreable list.
- **G-comp-v3-tight (CO-PRIMARY):** composed in-band ≥ 80% at ±.025
  over floor-clearing slots. Calibration on record: the S2 preview
  read 83–91% at this band in its worst configuration (n_train=120,
  k32, v2.1-vintage vectors). Both gates must pass for the
  structure claim at full strength; tight-fail/floor-pass = the
  named partial "structure holds at the coarse band only."
- **G-extension (the doubly-held-out sub-line):** ≥ 70% at ±.05
  over extension + never-before-observed slots (72B, base siblings,
  dsv3, gpt2-xl, olmo2-base). Its failure with the core gates
  passing = "transport holds on the incumbent class, degrades at
  the named boundary" — pre-named, never pooled.
- **Branch structure (review A-F6):** (a) of-record column AND all
  race-hub columns fail → the structure claim takes the hit at full
  weight. (b) Of-record fails while ≥1 simultaneously-filed race
  column passes both core gates → "composed transport generalizes;
  the incumbent hub does not" — the structure claim survives on the
  passing column (identity fixed at filing time; no post-hoc
  selection), the incumbent's demotion is the headline. (c)
  Of-record passes, candidates fail → incumbent-specific
  robustness, a mechanism datum.

**The star beside (S1 continuation):** the scalar star is
re-derived on webtext-v3 under the parent's derivation rules and
quoted ONLY as the composed-vs-star descriptive contrast on the
primary basis — labeled diagnostic, never a headline, never gated.

## §9 The authored-stratum ablation (the flattery measurement, dose-honest)

Fixed **n=100, seed 80** sample from v2.1's S1/S3 (source: the
v2.1 full manifest, sha `5ae355bc…`, pinned here; drawn by text_id
order; the list computed and sha-recorded at freeze). Refit hub legs
for BOTH author models (8b, dsv2-lite) AND the two race leaders
under both conditions (web-only vs web+authored; each arm internally
consistent — legs and â_obs on the same basis, never mixed).
**Readout of record: the difference-in-differences** — (author
models' Δmedian-|e| and Δrank) minus (the non-author roster's median
Δ) — with a permutation null over which-2-models-are-labeled-authors
(1,000 relabelings, p < .05 two-sided) and bootstrap CIs on the
effect sizes. **Scope clause, binding on §11: a null retires the
flattery mechanism AT THIS DOSE (~7.7% admixture, ~4% own-text) —
quoted as "no detectable author advantage at minority
contamination" — and does NOT retire the mechanism at v2.1's 80%
dose, which remains explained by the S2-refit contrast.**

## §10 The behavioral decoupling leg (pre-registered here, executed in the behavioral phase)

**Binding readout: the TIER contrast** — pooled behavioral steering
deltas of the top-3 hubs vs the bottom-3 (by §7's metric), paired
across shared targets and control objects, one-sided sign test
p < .05, effect size quoted. Top-beats-bottom = the first
non-circular validation of â as a hub-quality construct; failure =
the named demotion "â is machinery-indexed." **The 6-hub ordering ρ
is quoted beside, descriptive only — the top tier is expected
statistically tied on â, so its behavioral order carries no
confirmatory weight in either direction (pre-stated).** Cell sizes
and dosing inherit the certified harness ceremony, pinned in that
ceremony's pre-statement before any behavioral cell fires.

## §11 What refutes what (the falsifier table, aligned to the binding clauses)

- I1/I2 BINDING clauses fail (eligibility disagreement across
  halves/ranks) → hub quality is corpus/rank-idiosyncratic; no
  crowning; §7.6 fallback binds. (Full-order ρ misses with tier
  agreement are scope limits, quoted as such — not this branch.)
- G-comp-v3 AND all race columns fail → composed transport does not
  generalize to the clean basis at scale — full weight. (The §8
  branch structure governs partial failures.)
- G-comp-v3-tight fails alone → "coarse-band-only" partial, named.
- G-extension fails alone → boundary-degradation partial, named.
- Gemma top-tier → predictor-layer hit (§7.6).
- §9: a POSITIVE DiD = the flattery mechanism measured; a null =
  no detectable author advantage at this dose (v2.1-dose mechanism
  NOT thereby retired).
- §10 tier contrast fails → â demoted to machinery observable.
- HL-2 unsupported under its nulls → "hubness is real but
  unpredicted; measure per-roster."
- k256 health check fails → rank of record = k128 by §3.3, quoted
  everywhere, nothing else moves.

## §12 Learnings binding here

Census-at-freeze beside every sha (M47) · no scalar-law headline —
the atlas is the claim, the star is a labeled diagnostic beside ·
tier language wherever separation is unproven (C6); binding clauses
are tier-level · predictors named before any quantity, circularity
guards explicit (C1/U-A); P2/P6 are one cluster · full-ordering
scoring only (F4-the-rake) · enumerated denominators, frozen before
results exist (the parent's best habit, restored) · publication
sweeps are blocking zero-asserts; filenames are content (M46) ·
thread config is instrument identity (OMP ruling) · every constant
names its basis sha, forever.

## §13 Freeze mechanics

Tag `freeze/webtext-v3` · sha of this file in the ledger · gist
attestation carrying: the corpus census (with the 59-overlap table)
· the manifest sha · the shas of the freeze-time deterministic
draws (train/test membership, I1 halves, §9 ablation text_ids) ·
**the zero-quantities attestation: a recorded sweep confirming no
states, fits, vectors, or â computed on webtext-v3 exist anywhere
at freeze — every seeded draw is deterministic, and nothing
model-derived has touched its outputs.** Luxia's stamps at freeze:
this document entire · the ceremony go. The §8 prediction-artifact
stamp mechanism is the ceremony's enforcement, not a promise.
