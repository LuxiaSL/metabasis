# metabasis

*μετάβασις — change of basis; metabasis eis allo genos, "crossing into another kind."*

Steering vectors port between language models. The machinery is a
**hub-and-spoke atlas**: each model gets one linear **transport map** into a
shared hub frame, fit on nothing but per-text mean residual states over a
shared, sha-pinned text corpus. Any pair of models then composes through two
hub legs; the fidelity of every transfer (the **exchange rate â**) is
predicted before it is observed, under frozen error bands; and a steering
vector crosses from one model to another by an explicit change of basis. No
map is used until it beats two shuffled nulls on held-out data, and
behavioral claims certify against matched-norm random controls.

## The pipeline (the math in one pass)

1. **Shared corpus.** A fixed, sha-pinned corpus of generic web text. The
   fitting text is deliberately empty of anything being transferred — the
   maps learn shared geometry, not task content. Every frozen input carries
   a composition census beside its hash.
2. **Forced replay → per-text means.** Each model reads every text with its
   own tokenizer. At that model's **site of record** — a residual layer
   chosen from its alignment curve, never by fiat — the **per-text mean**
   residual state is recorded (mean over completion-token states, raw fp32;
   normalization happens at fit time). Matching is at the text level: the
   mean is what makes rows comparable across tokenizers.
3. **Per-side PCA.** Each model's mean-state matrix gets its own PC basis,
   fit on training rows only. States become rank-k coordinates, with k
   bounded by the rank guard (k ≤ n_train / 1.2).
4. **Procrustes to the hub.** A semi-orthogonal map (WᵀW = I, optional
   isotropic scale) is fit from each model's coordinates to the hub's — one
   map per model. The hub itself is a measured choice under a pre-registered
   selection criterion, not an assumption.
5. **Gates.** A fit is used only if its held-out state-prediction R²
   separates from **both** a shuffled-pair null and a stratum-preserving
   shuffle null (the sharp one), with agreement across template arms read
   beside.
6. **Compose, predict, transfer.** Any pair A→B is the composition of A's
   hub leg with B's — **two hub maps determine every pair**. â for a pair
   files as a prediction with frozen bands *before* it is observed, and a
   steering vector transfers by the same change of basis.

One property to carry with every number: **constants are properties of
(pair, corpus)**. â's ordering is robust across fitting corpora; its level
is indexed to the basis it was fit on — every quoted constant names its
corpus sha.

## Using it

The intended loop, end to end (module surface is stable; **webtext-v3 is
the corpus of record** — `--help` on any module is the authority):

```
uv venv && source .venv/bin/activate
uv pip install -e '.[gpu]'        # CPU spine alone suffices for fits/transfer

# 0. corpus: reconstruct webtext-v3 byte-for-byte from pinned public
#    sources — the golden path is corpus/webtext-v3/RECONSTRUCT.md
#    (one command; every artifact sha it must reproduce is listed there;
#    per-text verification via corpus/webtext-v3/corpus_manifest.meta.json)
python -m metabasis.scripts.build_webtext_corpus --seed 80 \
    --n-per-stratum 300 --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --out-dir <OUT>
python -m metabasis.scripts.derive_webtext_splits --help

# 1. site: scan each model's alignment curve; the site of record comes
#    from the curve
python -m metabasis.scripts.collect_mean_states --help

# 2. states: forced replay over the shared corpus → per-text mean residual
#    states at the site (collection + bitwise spot-replay in one pass)
python -m metabasis.scripts.collect_mean_states ...

# 3. maps: per-side PCA + semi-orthogonal Procrustes to the hub, gated
#    against both nulls
python -m metabasis.scripts.fit_transport_maps ...

# 4. transfer + verify: compose hub legs for any pair, predict â, carry a
#    vector across, read it out in the target model
python -m metabasis.scripts.read_transported_axes ...
```

A model enters the atlas once: one site, one state bank, one hub leg. Every
pair it participates in afterwards is composition — no per-pair fitting.

## Layout

```
metabasis/
  scripts/       the transport stack, named by what each does — corpus build,
                 state collection, map fitting, transported readouts,
                 composition/difficulty/path-independence panels, injection
                 banks, behavioral probes and judging
  extraction/    minimal hook machinery (decoder-layer resolution + one
                 residual-write injection hook — nothing else)
  config.py      model presets (architecture facts + dtypes)
manifests/       sha256 baselines over the data tree — the replication anchor
outputs/         data: states, transport maps, banks, readouts
                 (LOCAL ONLY, never tracked; verify against manifests/)
corpus/          the published fitting corpora — webtext-v3/ (OF RECORD:
                 reconstruction golden path + per-text sha manifest) and
                 fitting-v21/ (historical, with its composition census)
docs/            research docs and methodology (largely local-only)
```

Git tracks code plus the data manifests. The data tree travels out-of-band;
`sha256sum --check manifests/outputs.sha256` certifies a copy (or
`python -m metabasis.scripts.generate_tree_manifest --verify …`, which adds
defect diagnosis).

## Vocabulary

One namespace, no decoder ring: every object is named by its construction,
and the same name is used in code, stamps, and prose — transport map,
exchange rate, hub leg, composed map, atlas. The full stack lives in
`docs/methodology/naming-conventions.md`.

## Environment

```
uv venv && source .venv/bin/activate
uv pip install -e .              # CPU algebra spine: numpy/scipy/pydantic only
uv pip install -e '.[gpu]'       # + torch/transformers for collection & probes
```

The CPU spine (map fits, composition, all banked readouts) runs anywhere
with no GPU. Cluster-side values (work roots, weights paths, API keys) come
from the environment; nothing of that class is ever committed.

## Provenance

The stack lifts from a prior research program whose records remain canonical
there and are cited, not migrated; the port certified itself by regenerating
a banked readout byte-identically before anything else fired. Claims made
from this repo are pre-registered, predictions file before observation, and
the manifests make every artifact independently verifiable.
