# webtext-v3 — the fitting corpus of record

The primary basis for all forward claims: **1,200 texts, four strata of
public web text (300 each), every source a pre-2022 snapshot** — the
corpus cannot contain model-authored text by construction, not by
filtering. Pre-registered and frozen 2026-08-03 (tag `freeze/webtext-v3`,
prereg in `docs/planning/PREREG-webtext-v3-2026-08-03.md`).

Raw bodies are **not** redistributed. The corpus is published as a
deterministic reconstruction plus a per-text verification manifest:

- `RECONSTRUCT.md` — the golden path: one command, pinned inputs, the
  exact sha256 every artifact must reproduce.
- `corpus_manifest.meta.json` — all 1,200 entries with source pins
  (repo @ revision : file : shard/chunk) and each text's `text_sha256`,
  so a rebuilt corpus verifies text by text with no body ever shipped.

| identity | sha256 |
|---|---|
| `corpus_manifest.json` (full, rebuilt locally) | `b85f4d169ed0cb882a5690e3509308a5807e056b6a491f908014aee8e7b6085f` |
| `corpus_manifest.meta.json` (this directory) | `78d3a880504faaac9a6e864e1db224ada78994db437e36260c417a2ebfdd5d42` |

After reconstruction, the train/holdout split and the invariance
half-splits are derived deterministically (seed 80, grouped by the
pre-registered keys):

```
python -m metabasis.scripts.derive_webtext_splits --help
```

The predecessor basis (`corpus/fitting-v21/`, v2.1) remains for the
historical record with its composition census; no v2.1 constant is ever
quoted against a webtext-v3 observation (the vintage rule).
