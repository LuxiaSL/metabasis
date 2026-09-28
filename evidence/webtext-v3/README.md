# webtext-v3 evidence package

These files are what the webtext-v3 headline results in [`RESULTS.md`](../../RESULTS.md)
are computed from: the composed predictions landing 240/240 within ±0.05 on every real hub,
the ±0.025 rates, the 84-slot subset, and the star factorization beside them at 163/240
in-sample and 143/240 held out, with a discordance of 77 to 0. One command recomputes all of
them from these files alone:

```sh
python -m metabasis.scripts.recompute_webtext_v3_evidence
```

Run it from the repository root; it needs only the CPU dependencies in `pyproject.toml`. It
exits 0 when every file hash, every recomputed count and every cross-check agrees, 1 when any
disagrees, and 2 when an input is missing. `--json <path>` also writes every gate field, the
arm split and the star summary to a file. `--selftest` runs the module's own checks,
including this recompute.

## Files

| file | what it is | sha256 | bytes |
|---|---|---|---|
| `PREDICTIONS-webtext-v3-2026-08-04.json` | the sealed prediction artifact: the frozen list of 240 slots (ordered model pairs) and, for each slot, the composed prediction filed through each of five hub models, with its ±0.05 and ±0.025 bands | `de9eb699db92824443002300f98c5a8fceb0aff981d15c938887e8ca9e033502` | 1207164 |
| `ARTIFACT-STAMP-PREDICTIONS-webtext-v3.json` | the stamp that seals it: the artifact's sha256, its sidecar's sha256, the frozen count, and the sha256 of the pre-registration it was filed under | `35ccd86ab1162fdec039204112a64f3704e3d0387d2e3ee288811cf6b960e64a` | 905 |
| `PREDICTIONS-webtext-v3-2026-08-04.sidecar.json` | the build environment of the prediction artifact (the artifact itself carries no timestamp or environment value) | `8a2746d25b0b90381b916876b71982f410820ecbe3872e23a515b95ff927e243` | 750 |
| `SCORED-webtext-v3-2026-08-04-FULL.json` | the scored record: for every slot, the observed exchange rate `observed` read from the direct pair fit, plus the verdicts and gate counts computed from it | `6b7780035d332d5ae9a9f07b1727685e51de335f1a0f6bb8164f6aa8729cdc7f` | 1398384 |
| `STAR-BESIDE-webtext-v3-2026-08-04.json` | the star-beside record: the scalar star fit on the same 240 slots, per-slot star predictions, the head-to-head against the composed column, and the held-out count | `f036d3970347d3e2baa5e98ffdb359e5555ca1181075e575287f7e2c39d24e38` | 382436 |
| `STAR-BESIDE-webtext-v3-2026-08-04.sidecar.json` | the build environment of the star-beside record | `e9e626f96bd295778ca2849a2dba05e6165ec7088dd95a7f57711453f210f3a5` | 2268 |

`SHA256SUMS` lists the same hashes in the format `sha256sum -c SHA256SUMS` checks.

The files are byte-identical to the ones the results were computed from; nothing in them is
edited. Paths inside them (`staging/…`, `outputs/…`) are repository-relative locations of
data trees that are not published here. They are part of the sealed bytes, so rewriting them
would break every hash above.

## How the files hang together

- The stamp names the prediction artifact's sha256 and its sidecar's sha256. The recompute
  checks both, through `read_composed_predictions.load_v3_prediction_artifact`, before it
  reads a single prediction.
- The artifact and the stamp both name the pre-registration's sha256,
  `1136ca80c1c6be13add85d50f6d89bb85278abaac470536d36d2ed26c78c57fa`, which is the sha256 of
  [`docs/planning/PREREG-webtext-v3-2026-08-03.md`](../../docs/planning/PREREG-webtext-v3-2026-08-03.md).
- The scored record names the artifact's sha256 it was scored against, and the star-beside
  record names the scored record's sha256 it was built from. The recompute checks both links.

## What is recomputed and what is read

The recompute reads the filed predictions and bands from the artifact, and one number per
slot, `observed`, from the scored record. Everything the scored record concluded from those
numbers is discarded and rebuilt: each slot is rescored through
`read_composed_predictions.score_v3_column`, the gates through `_column_gate`, and the star
through `star_beside_webtext_v3`'s `solve_v3_systems`, `score_slots`, `head_to_head` and
`held_out_beside`. The rebuilt verdicts and counts are then compared with the ones both
records hold; any difference fails the run.

`observed` is the cosine between the source vector carried through the direct pair-fit map
and the target's own vector (`read_exchange_rates.exchange_rate`). Recomputing it needs the
120 fitted pair maps (one per unordered pair, read forward for one slot and in reverse for
the other; about 1.3 GB at the rank of record) and the 16 models' vectors, none of which is
in this package. Here it is an input.

## What the counts mean

- **±0.05 of 240.** A slot is in band when `observed` lies within ±0.05 of the filed
  prediction, edges inclusive. The denominator is the frozen list of 240 and never shrinks.
- **±0.025 of floor-clearing.** The pre-registration scores the tight band over slots whose
  observation clears |â_obs| ≥ 0.08. All 240 clear it, so the denominator is 240 for every hub.
- **Near-zero predictions.** A prediction with |â_comp| < 0.08 would be scored on magnitude
  only, sign unscored. No filed prediction in any column is that small, so the rule never
  applies.
- **The 84-slot subset.** These are the slots with at least one endpoint among the three
  models the pre-registration names as never observed before this corpus:
  `qwen2.5-72b-instruct`, `llama-3.1-8b-base` and `qwen2.5-7b-base`. With 16 endpoints,
  16·15 − 13·12 = 84. The subset lies inside the 240; it is not 84 additional predictions.
- **Arms.** Slots between two instruct models are scored in the native arm (156 slots). Slots
  with a base-model endpoint are scored in the raw arm (84 slots; the three base endpoints are
  `llama-3.1-8b-base`, `qwen2.5-7b-base` and `pythia-6.9b`, a different set of three from the
  subset above). The recompute prints each gate split by arm.
- **The star beside.** The star predicts â(A→B) = c_A · c_B, with one coefficient per model
  in each arm's system. It is fit on the same 240 slots it is then scored on, which favours
  the star. The held-out count drops both directions of one unordered pair, refits, and
  scores the two dropped slots, once per pair. The discordance counts, slot by slot, where the
  composed column of record (hub `8b`) lands and the star does not, and the reverse.

## Expected output

The counts the command prints, and that `RESULTS.md` quotes:

| hub | ±0.05 of 240 | ±0.025 of 240 | 84-slot subset |
|---|---|---|---|
| `qwen2.5-3b-instruct` | 240 | 223 (92.9 %) | 84 |
| `qwen2.5-32b-instruct` | 240 | 226 (94.2 %) | 84 |
| `3b` (Llama-3.2-3B) | 240 | 231 (96.3 %) | 84 |
| `8b` (Llama-3.1-8B, column of record) | 240 | 225 (93.8 %) | 84 |
| `gemma3-27b` (designated poor-hub control) | 238 | 190 (79.2 %) | 84 |

Star beside: 163/240 in-sample, 143/240 held out (no pair skipped); composed-only 77,
star-only 0, both-miss 0.

## Attestation

The prediction artifact's sha256 is `de9eb699db92824443002300f98c5a8fceb0aff981d15c938887e8ca9e033502`,
and the stamp that names it has sha256
`35ccd86ab1162fdec039204112a64f3704e3d0387d2e3ee288811cf6b960e64a`. The stamp was posted,
byte for byte, as the only file of an unlisted GitHub gist:

<https://gist.github.com/LuxiaSL/c4fccd83611354fc2a16362eabb195ec>

The gist has a single revision. GitHub records its creation time; compare it with the
stamp's `sealed_utc` field. The scored record's `generated_utc` field is written by the
scoring code and is not independently attested.

What this establishes: a stamp naming this exact prediction artifact existed, unchanged, at
the gist's creation time. What it does not establish:

- **When the direct pair fits ran.** The artifact states that no scoreable pair had been fit
  when it was built (`self_description.zero_scoreable_fits`), and the fitting runs are
  recorded as launched through a lane that verified this stamp first. Neither statement can
  be checked from this package, and
  `metabasis.scripts.fit_transport_maps` does not itself refuse to fit without the stamp.
- **Independence from the endpoints.** A composed prediction is built from both endpoint
  models' vectors and the two fitted hub legs. What is predicted before observation is the
  exchange rate of the direct fit between the two endpoints.
- **The observed values themselves**, which rest on the pair-fit maps described above.

The star-beside record also carries a regression check (`regression_proof_v21`) that
reproduces an earlier corpus's banked star solution before any webtext-v3 number is computed.
Its inputs are not published, so that check's result is recorded here but cannot be rerun.
