# RECONSTRUCT — rebuilding `webtext-v3` (staging-draft) byte for byte

The corpus is published as a reconstruction: pinned inputs -> the frozen chunker -> a seeded selection -> exactly these bytes. No text bodies are redistributed.

## The command

```
python -m metabasis.scripts.build_webtext_corpus --seed 80 --n-per-stratum 300 --tokenizer meta-llama/Llama-3.1-8B-Instruct --out-dir <OUT>
```

## What it must produce

| artifact | sha256 |
|---|---|
| `corpus_manifest.json` (full, with bodies) | `b85f4d169ed0cb882a5690e3509308a5807e056b6a491f908014aee8e7b6085f` |
| `corpus_manifest.meta.json` (public metadata) | `78d3a880504faaac9a6e864e1db224ada78994db437e36260c417a2ebfdd5d42` |

Each entry's body verifies against its `text_sha256` in the metadata manifest, so a rebuilt corpus can be checked text by text without ever publishing a body.

## The pins

| stratum | repo | revision | files |
|---|---|---|---|
| `wikitext` | `Salesforce/wikitext` | `b08601e04326c79dfdd32d625aee71d232d685c3` | `wikitext-103-raw-v1/validation-00000-of-00001.parquet` |
| `c4` | `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | `en/c4-validation.00000-of-00008.json.gz` |
| `pg19` | `deepmind/pg19` | `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b` | `data/validation_files.txt` |
| `stackexchange` | `flax-sentence-embeddings/stackexchange_title_body_jsonl` | `a3d99bf21570ed043e19e41af46f3f19bf4e4bb6` | `astronomy.stackexchange.com.jsonl.gz`, `bicycles.stackexchange.com.jsonl.gz`, `biology.stackexchange.com.jsonl.gz`, `cooking.stackexchange.com.jsonl.gz`, `engineering.stackexchange.com.jsonl.gz`, `philosophy.stackexchange.com.jsonl.gz` |

- **PG-19 books** come from the immutable asset root `https://storage.googleapis.com/deepmind-gutenberg/` using the file list pinned above; the stamp records a sha256 for every book and for `metadata.csv`.
- **Tokenizer**: `meta-llama/Llama-3.1-8B-Instruct` — a PIN, not a convenience. The chunk boundaries are its token counts. The stamp records a sha256 for every tokenizer file used. (It is a gated repo: accept the license once, then it resolves from the local cache. A different tokenizer produces a different corpus and the builder will say so by producing a different sha.)
- **Seed**: 80. **Per-document chunk cap**: 10, at evenly spaced positions.
- **Per-stratum reading pins** (they select the pool, so they are part of the reconstruction):
  - `wikitext`: 300 texts; documents read = every document in the pinned files; edge trim = 0
  - `c4`: 300 texts; documents read = first 6000 rows per file; edge trim = 0
  - `pg19`: 300 texts; documents read = every document in the pinned files; edge trim = 0.05
  - `stackexchange`: 300 texts; documents read = first 1200 rows per file; edge trim = 0

## What is and is not environment-independent

- `corpus_manifest.json`, `corpus_manifest.meta.json`, `COMPOSITION-CENSUS.md` and this file are byte-identical anywhere the pins resolve: nothing in them is derived from a path, a clock, or a library version.
- `corpus_stamp.json` additionally records the build environment (interpreter and library versions), so it is byte-identical within an environment and may differ across environments. The manifest sha is the object of record.
- `--verify-rebuild` builds the whole corpus twice in one run and HALTS unless every artifact matches byte for byte.

