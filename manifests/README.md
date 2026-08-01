# Data manifests

Git tracks code; the data tree (`outputs/`, ~8.6G of banked states, transport
maps, vector banks, and readouts) travels out-of-band. This directory makes it
verifiable: `outputs.sha256` is the sha256 of every file in `outputs/`,
regenerated at each freeze point (git history is the vintage record).

`collection.sha256` covers `outputs/collection/` on the separate session-close
cadence (desk-only, same rules).

Verify a data tree against the current baseline — `sha256sum --check` remains the
checker of record, and the in-process form adds the M42 diagnosis it lacks (a
self-listing manifest is a GENERATOR defect over innocent artifacts, exit 3, not
a digest failure):

    sha256sum --check --quiet manifests/outputs.sha256
    python -m metabasis.scripts.generate_tree_manifest \
        --verify manifests/outputs.sha256 --anchor .

Regenerate after a sanctioned data change (desk-only; the diff against the
previous git version is the change record). **Use the generator, not a hand-typed
pipeline** — it is byte-compatible with `sha256sum` and enforces the four rakes
that hand-typed pipelines have paid for (M38 symlink-following · M42 no
self-listing · M43 scratch outside the tree and the row-count check · M46 the
blocking sanitization sweep over PATHS as well as bytes):

    python -m metabasis.scripts.generate_tree_manifest \
        --tree outputs --output manifests/outputs.sha256 --anchor . \
        --compare-previous manifests/outputs.sha256 \
        --patterns-file <desk-side pattern file, never in this repo>

The pipeline it replaces was

    find outputs -type f -print0 | sort -z | xargs -0 sha256sum > manifests/outputs.sha256

and it carries two defects worth naming, since older manifests were produced by
it: `find` **without `-L`** silently drops every relocated (symlinked) subtree —
rake M38, and the diff reads as deletion — and `sort` without `LC_ALL=C` orders
by locale collation, which is why the current `outputs.sha256` is not byte-sorted
and why a first regeneration with the generator will reorder rows without
changing a single digest. (`--sort locale` reproduces the historical order if a
minimal diff is wanted at that freeze point; byte order is the portable default.)

One known historical note: `arms/A8_conjugation/leg6/ANNEX2-LEG6-FIRST-READ-2026-07-23.md`
intentionally differs from the hash inside `leg6/rsync_manifest.json` (a desk
stamp was appended after that manifest was written — adjudicated benign,
2026-07-26). This baseline carries the current, correct hash.
