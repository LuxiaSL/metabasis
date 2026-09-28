# Data manifests

Git tracks code; the data tree (`outputs/`: state banks, transport maps, vector banks and
readouts) travels out-of-band. These manifests make a copy verifiable:

| manifest | covers |
|---|---|
| `outputs.sha256` | every file under `outputs/battery/` |
| `collection.sha256` | every file under `outputs/collection/` but one withheld provenance file |

Fitted pair maps under `outputs/pairs/` are not covered by either.

Verify a copy from the repository root:

```bash
sha256sum --check --quiet manifests/outputs.sha256
sha256sum --check --quiet manifests/collection.sha256
```

`collection.sha256` opens with `#` comment lines, which `sha256sum` skips.

For a failure `sha256sum` cannot explain, the generator also diagnoses defects in the
manifest itself (a manifest that lists its own file is a generator defect, not a digest
failure, and exits 3):

```bash
python -m metabasis.scripts.generate_tree_manifest \
    --verify manifests/outputs.sha256 --anchor .
```

## Regenerating

Regenerate with the generator rather than a hand-written pipeline. It is byte-compatible
with `sha256sum`, and it enforces four constraints a `find | sha256sum` pipeline violates:

- it follows symlinked subtrees, where `find` without `-L` silently drops them;
- it never lists the manifest file itself;
- it keeps scratch files outside the tree being hashed and checks the row count;
- it refuses to write when a path or byte pattern from a sanitization pattern file matches.

```bash
python -m metabasis.scripts.generate_tree_manifest \
    --tree outputs --output manifests/outputs.sha256 --anchor . \
    --compare-previous manifests/outputs.sha256 \
    --patterns-file <your sanitization pattern file>
```

`outputs.sha256` is ordered by locale collation, not byte order. A first regeneration with
the generator reorders its rows without changing a digest; `--sort locale` keeps the
existing order if a minimal diff is wanted.

One file under `outputs/battery/` intentionally differs from the hash an older transfer
manifest inside the data tree records for it, because a note was appended after that
manifest was written. `outputs.sha256` carries the current hash.
