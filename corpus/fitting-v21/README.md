# The v2.1 fitting corpus (corpus of record for all transport-map fits)

`corpus_manifest.json` is published **byte-identical** to the corpus of
record: its sha256,

```
5ae355bc5d130f8e9c3ae426f5e71bf2b6e99c74b95369a874bec2abcd59b5d9
```

is the exact value every fit artifact and prediction record in this
campaign cites as `corpus_manifest_sha256`. Verify with
`sha256sum corpus_manifest.json`.

## Composition — 777 texts, three strata

| stratum | n | what |
|---|---|---|
| S1 | 320 | Model-generated prose: two voices (Llama-3.1-8B-Instruct, DeepSeek-V2-Lite) writing on 20 fixed topics × 4 task modes ("Write about: …", bare system prompt); per (topic × mode) cell, the 2 lowest-repetition non-empty generations |
| S2 | 160 | **WikiText-103** (`wikitext-103-raw-v1`, validation split): greedy paragraph chunks of 150–500 tokens (close at 250), fixed-seed sample; headings/empty lines dropped |
| S3 | 297 | Model-generated prose, same two voices: per mode, repetition 0 of topics 0–19 plus repetition 1 of topics 0–9 |

Each entry carries its full provenance: `text_id`, stratum, voice,
mode, topic, repetition, source run/generation/seed, and the exact
prompts used.

Fit splits consume **775 of the 777** entries: a topic-grouped split
(5/20 topics + 40/160 S2 chunks held out; seed 80) yielding
n_train = 598, n_test = 177, recorded per fit in `cp2_summary.json`.

## Licenses

- **S2** derives from WikiText-103 (Wikipedia text), **CC BY-SA 3.0**
  — attribution to Wikipedia/WikiText-103 (Merity et al., 2016,
  "Pointer Sentinel Mixture Models"); this stratum's text remains
  under CC BY-SA.
- **S1/S3** are language-model outputs (Llama 3.1 8B Instruct under
  the Llama 3.1 Community License; DeepSeek-V2-Lite under the
  DeepSeek Model License) — redistributed here as generated text.
  Fictional names occurring in generated prose are fictional.
