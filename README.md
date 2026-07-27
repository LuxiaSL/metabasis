# metabasis

*μετάβασις — change of basis; metabasis eis allo genos, "crossing into another kind."*

Steering vectors port between language models through a closed-form linear
change of basis — a **transport map** fit on nothing but mean residual states
over shared generic text — and the fidelity of that transport (the **exchange
rate**) follows a striking regularity: it factorizes into one **portability
coefficient** per model (the **star factorization**). This repo scales that
result across a ~20-model roster with pre-registered held-out predictions,
spans the full control-object roster under one transport protocol, and
certifies every behavioral claim against matched-norm random controls. The
working interpretation under test: a partially shared **causal control
atlas** across language models, of which each model's residual space is one
chart.

## Layout

```
metabasis/
  scripts/       the transport stack, named by what each does:
                 build_paired_corpus → collect_mean_states →
                 fit_transport_maps → read_transported_axes →
                 solve_star_systems / extend_star_system /
                 difficulty_curve_rows / composition_panel /
                 path_independence → build_injection_banks →
                 entropy_write_probe / judge_blind_2afc → readouts
  extraction/    minimal hook machinery (decoder-layer resolution + one
                 residual-write injection hook — nothing else)
  config.py      model presets (architecture facts + dtypes)
  text_decode.py small shared helper
manifests/       sha256 baseline over the data tree — the replication anchor
outputs/         data: states, transport maps, control-object banks, readouts
                 (LOCAL ONLY, never tracked; verify against manifests/)
docs/            research docs, desk protocol, methodology (LOCAL ONLY)
```

Git tracks code plus the data manifests. The data tree (~8.6G) travels
out-of-band; `sha256sum --check manifests/outputs.sha256` certifies a copy.

## Vocabulary

One namespace, no decoder ring: every object is named by its construction and
the same name is used in code, stamps, and prose. The full stack and the
reading key for legacy anamnesis-era bank keys live in
`docs/methodology/naming-conventions.md`. Phases: **bootstrap → freeze →
collection → law-at-scale → class-split → capstone**.

## Environment

```
uv venv && source .venv/bin/activate
uv pip install -e .              # CPU algebra spine: numpy/scipy/pydantic only
uv pip install -e '.[gpu]'       # + torch/transformers for collection & probes
```

The CPU spine (transport-map fits, star systems, all banked readouts) runs
anywhere with no GPU. Cluster-side values (work roots, weights paths, API
keys) come from the environment — see the local, off-repo runbook; nothing of
that class is ever committed.

## Provenance

The program lifts out of a prior research campaign (the anamnesis project's
transport arm); the arm record, preregistration chain, and claims registry of
the imported results remain canonical there and are cited, not migrated. The
port certified itself at bootstrap by regenerating the banked star-system
readout byte-identically from the lifted stack — exchange rate â(3B↔8B)
native::proc_k128 = .4815 exact — before anything else fired.
