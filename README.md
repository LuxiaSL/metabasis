# metabasis

*μετάβασις — change of basis; metabasis eis allo genos, "crossing into another kind."*

Steering vectors port between language models through a closed-form linear
change of basis fit on nothing but mean residual states over shared generic
text — and the fidelity of that transport follows a law. This repo scales the
law across a ~20-model roster with pre-registered held-out predictions, spans
five vector classes under one procedure, and certifies every behavioral claim
against matched controls.

## Layout

```
metabasis/
  scripts/       the transport stack: corpus → collect → fit → rosetta →
                 star systems → extensions → behavioral legs
  extraction/    minimal hook machinery (decoder-layer resolution + one
                 residual-write injection hook — nothing else)
  config.py      model presets (architecture facts + dtypes)
  analysis/      small shared helpers
outputs/         data banks: states, fits, vector banks, readouts (LOCAL ONLY,
                 never tracked — see .gitignore)
docs/            research docs and planning (LOCAL ONLY for now)
```

Git tracks code only. Data (~8.6G of banked states/fits/readouts) and research
docs live in the working tree, integrity-anchored by sha256 where load-bearing,
and travel by rsync, not by git.

## Environment

```
uv venv && source .venv/bin/activate
uv pip install -e .              # CPU algebra spine: numpy/scipy/pydantic only
uv pip install -e '.[gpu]'       # + torch/transformers for collection & probes
```

The CPU spine (fits, rosetta, star systems, all banked readouts) runs anywhere
with no GPU. Cluster-side values (work roots, weights paths, API keys) come
from the environment — see the local, off-repo runbook; nothing of that class
is ever committed.

## Provenance

The program lifts out of a prior research campaign (the A8 conjugation arm of
the anamnesis project); the arm record, preregistration chain, and claims
registry of the imported results remain canonical there and are cited, not
migrated. The port certified itself at bootstrap by reproducing the banked
â(3B↔8B) native::proc_k128 exchange rate from the lifted stack (the P0 parity
gate) before anything else fired.
