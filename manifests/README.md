# Data manifests

Git tracks code; the data tree (`outputs/`, ~8.6G of banked states, transport
maps, vector banks, and readouts) travels out-of-band. This directory makes it
verifiable: `outputs.sha256` is the sha256 of every file in `outputs/`,
regenerated at each freeze point (git history is the vintage record).

Verify a data tree against the current baseline:

    sha256sum --check --quiet manifests/outputs.sha256

Regenerate after a sanctioned data change (desk-only; the diff against the
previous git version is the change record):

    find outputs -type f -print0 | sort -z | xargs -0 sha256sum > manifests/outputs.sha256

One known historical note: `arms/A8_conjugation/leg6/ANNEX2-LEG6-FIRST-READ-2026-07-23.md`
intentionally differs from the hash inside `leg6/rsync_manifest.json` (a desk
stamp was appended after that manifest was written — adjudicated benign,
2026-07-26). This baseline carries the current, correct hash.
