# metabasis

*μετάβασις — change of basis.*

**A steering vector found in one language model can be carried into another through a
linear map fit on shared generic text.**

A steering vector is a direction added to a model's residual stream to shift its behavior:
toward formal register, toward French, toward refusing, toward higher output entropy.
Metabasis builds a **hub-and-spoke atlas** of language models. Each model gets one
**transport map** into a shared hub model's residual space, fit on nothing but mean residual
states over a fixed web-text corpus. Any pair of models composes through two hub legs. Each
pair's **exchange rate** (â: the cosine between a source model's vector carried through the
map and the target model's own vector of the same kind) is predicted before the pair is
fit directly. A steering vector crosses from one model to another through the composed map.

## Why

A steering vector belongs to the model it was built in. Finding a lever usually means
building it again in every model you care about. Metabasis measures how much of that work
transfers:

- when a lever found once moves to another model;
- how well it moves;
- why it fails when it fails.

Every map must beat shuffled nulls on held-out data before it is used, and every behavioral
claim is certified against matched-norm random controls.

## What it has found

[`RESULTS.md`](RESULTS.md) holds the full account, with each item marked established,
exploratory, rejected, or in progress. In brief:

- **Composed maps predict exchange rates ahead of the direct fit.** On each of four hubs,
  240/240 pre-filed predictions land within ±0.05 of the directly fitted exchange rate, and
  92.9–96.3 % within ±0.025. A designated poor-hub control fails the tight band, as
  pre-registered. One coefficient per model lands 163/240 in-sample and 143/240 held out,
  and on the earlier corpus that scalar model failed its pre-registered gate while the
  composed predictor passed. [`evidence/webtext-v3/`](evidence/webtext-v3/) rebuilds these
  counts from the filed predictions.
- **The entropy-gradient vector steers from 3B to 405B.** It is dose-ordered and outside
  random controls in 20 of 21 models tested (Mixtral on a narrower dose ladder).
- **Contrast vectors transfer visibly at small and mid scale.** At 70B, language behaves
  like a cliff and formality like a dial.
- **At the largest model, the generic-text anchor frame stops holding the class
  directions.** *(Exploratory.)* Eight axis-relevant contrast pairs restore it, one seeded
  map serves many vectors, and composing two seeded legs costs almost nothing.

## Quickstart

The algebra runs anywhere: numpy, scipy and pydantic, with no GPU and no model weights. Run
from the repository root.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# Selftests: each checks a module's algebra and refusals on synthetic data. They need a
# writable temporary directory, and they do not run automatically before real inputs.
python -m metabasis.scripts.fit_transport_maps --selftest       # map fitting and gates
python -m metabasis.scripts.build_behavioral_banks --selftest   # carrying vectors across
python -m metabasis.scripts.build_contrast_vectors --selftest   # contrast-vector construction
```

Each prints its checks and exits 0. A check that needs an artifact not in this repository
is reported as a named skip. Some module output and `--help` text
still carries internal record codes; [`CONTRIBUTING.md`](CONTRIBUTING.md) says where that
stands.

The core step is an orthogonal rotation with an isotropic scale between two PCA bases. On synthetic data, with a hidden
rotation and scale between two "models" reading the same texts:

```python
import numpy as np
from metabasis.scripts.fit_transport_maps import fit_procrustes

rng = np.random.default_rng(0)
za = rng.standard_normal((600, 32))                           # model A: per-text states
q, _ = np.linalg.qr(rng.standard_normal((32, 32)))            # the hidden change of basis
zb = 0.7 * za @ q + 0.05 * rng.standard_normal((600, 32))     # model B: the same texts

omega, scale = fit_procrustes(za, zb)
v_a = rng.standard_normal(32); v_a /= np.linalg.norm(v_a)     # a direction found in A
v_b = v_a @ omega                                             # carried into B
print(v_b @ (v_a @ q) / np.linalg.norm(v_b), scale)           # ≈ 1.000, ≈ 0.70
```

Real models add everything the pipeline below exists for: tokenizers that disagree, a
basis per model, a rank limit, and nulls a map must beat before anyone trusts it.

## The pipeline

1. **Shared corpus.** A fixed corpus of generic web text, reconstructed byte for byte from
   pinned public sources. Map fitting uses no task-specific data: no contrast pairs and no
   labels, only the same texts read by each model.
2. **Per-text mean states.** Each model reads every text with its own tokenizer. At that
   model's site, a residual layer chosen from its alignment curve, the mean residual state
   over each text's completion is recorded. Matching rows by text rather than token is what
   makes them comparable across tokenizers.
3. **Per-side PCA.** Each model's state matrix gets its own principal basis, fit on training
   rows only. The fitter caps rank at n_train − 1; results are read at ranks within
   k ≤ n_train / 1.2, a guard the readers apply.
4. **Procrustes to the hub.** An orthogonal rotation (with an optional isotropic scale)
   between the two models' PCA coordinates: one map per model. In the full residual space
   the map is rank-limited: directions outside the PCA spans are projected away, so it is
   not an invertible change of basis.
5. **Gates.** A map passes when its held-out state prediction beats both a shuffled-pair
   null and a stratum-preserving shuffle (95th percentile of 20 permutations each). The
   result is recorded with the saved map; loaders do not refuse a failing map, so a
   consumer checks the recorded gate.
6. **Compose, predict, transfer.** Pair A→B is A's hub leg composed with B's. Its exchange
   rate is filed as a prediction with frozen bands before the pair is fit directly, and a
   vector transfers by the same composition.

A model enters the atlas once: one site, one state bank, one hub leg. Every pair it joins
afterwards is composition, with no per-pair fitting.

An exchange rate depends on the model pair, the vector family, the sites, the corpus, the
template arm and the map rank. Its ordering is robust across fitting corpora; its level
depends on the corpus, so every quoted exchange rate names its corpus.

## Running it on models

The GPU side needs the `gpu` extra and model weights:

```bash
uv pip install -e '.[gpu,corpus]'

# 0. The corpus: rebuild webtext-v3 from its pinned sources.
python -m metabasis.scripts.build_webtext_corpus --seed 80 --n-per-stratum 300 \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct --out-dir <OUT>

# 0b. The frozen train/holdout split the fits consume.
python -m metabasis.scripts.derive_webtext_splits --help

# 1–2. Per-text mean states at each model's site; a bitwise replay check runs as a second
#      invocation in the same job (--collect and --spot-replay are separate runs).
python -m metabasis.scripts.collect_mean_states --help

# 3–5. Maps into the hub, gated against both nulls.
python -m metabasis.scripts.fit_transport_maps --help

# Steering vectors, built natively: the entropy gradient and contrast vectors.
python -m metabasis.scripts.build_entropy_gradient --help
python -m metabasis.scripts.build_contrast_vectors --help

# 6. Carry vectors across maps into the banks a steering run reads.
python -m metabasis.scripts.build_behavioral_banks --help
```

[`corpus/webtext-v3/RECONSTRUCT.md`](corpus/webtext-v3/RECONSTRUCT.md) lists every pin and
every sha256 the rebuild must reproduce. No text bodies are redistributed; each rebuilt text
verifies against its hash in
[`corpus/webtext-v3/corpus_manifest.meta.json`](corpus/webtext-v3/corpus_manifest.meta.json).
Deriving the frozen splits also needs the full manifest of the earlier fitting corpus, which
is not in the repository; it is available on request (see Data).

The models the atlas covers, their layer grids, sites and architecture facts live in
[`metabasis/roster.py`](metabasis/roster.py) and [`metabasis/config.py`](metabasis/config.py).
`--help` on any module is the authority on its arguments.

## Data

The state banks, fitted maps and behavioral readouts are not in the repository. They are
available on request: open an issue saying what you need. Two manifests make any copy
verifiable with `sha256sum --check`:

- [`manifests/outputs.sha256`](manifests/outputs.sha256) covers `outputs/battery/`;
- [`manifests/collection.sha256`](manifests/collection.sha256) covers `outputs/collection/`
  (one provenance file is withheld).

The fitted pair maps under `outputs/pairs/` are not yet covered by a manifest.

## Layout

```
metabasis/
  scripts/       the transport stack, one module per job (table below)
  extraction/    decoder-layer resolution and the residual-write steering hook
  l4_blind/      a tool for blind human judgments of generated text, the gold labels
                 the behavioral classifiers are checked against
  templates/     job templates for running collection and fits on a GPU scheduler
  roster.py      the models, their layer grids and sites
  config.py      per-model architecture facts and dtypes
  text_decode.py decoding of banked byte-level-BPE generations back to text
  capacity.py    per-GPU memory estimation for preflight checks
  threads.py     the effective thread configuration, recorded as instrument identity
  jobs_v21.py    renders the job templates and checks them
  lineage_discriminators.py
                 metadata checks that tell base and instruct checkpoints apart
corpus/          the fitting corpora: webtext-v3 (of record) and fitting-v21
manifests/       sha256 baselines over the data tree
docs/planning/   the two frozen pre-registrations the results are scored against
pyproject.toml   the package; extras gpu, corpus, judge, whiten, plots
```

The modules a first reader needs:

| module | what it does |
|---|---|
| `build_webtext_corpus` | rebuilds the fitting corpus from pinned public sources |
| `derive_webtext_splits` | derives the frozen train/holdout split |
| `collect_mean_states` | records per-text mean residual states at a site |
| `fit_transport_maps` | per-side PCA, Procrustes, and the two null gates |
| `build_entropy_gradient` | the label-free entropy steering vector, with a finite-difference check |
| `build_contrast_vectors` | contrast (difference-of-means) vectors from paired texts |
| `build_behavioral_banks` | carries vectors through maps into steering banks, with random controls |
| `vllm_lane_column` | runs a steered dose ladder on vLLM, at any tensor-parallel size |

## Vocabulary

One name per object, used the same way in code, outputs and prose:

| term | meaning |
|---|---|
| transport map | the linear map from one model's residual space into another's |
| hub leg | a model's transport map into the hub |
| composed map | two legs chained, A → hub → B |
| exchange rate (â) | cosine between a transported source vector and the target's own vector |
| star factorization | predicting each pair's exchange rate as a product of one coefficient per model |
| site | the residual layer a model's states are read at and its vectors written at |
| dose (α) | the steering strength, as a fraction of the site's median residual norm |
| random band | the range of effects from three random vectors of the same norm, carried the same way: a descriptive null, not a statistical interval |
| alignment | cosine between a transported vector and the target model's own native vector |
| containment | the fraction of a direction the map's anchor frame can hold; an upper bound on alignment |
| in-family leg | a map from a smaller model of the same family, as opposed to the hub leg |
| atlas | the set of models and their hub legs |

## Related

- [entropy-gradient](https://github.com/LuxiaSL/entropy-gradient): the label-free entropy
  steering vector, with its construction and its results up to 405B.
- [anamnesis](https://github.com/LuxiaSL/anamnesis): the steering and replay instrument
  metabasis's transport stack builds on.
- [repeng](https://github.com/vgel/repeng): multi-layer control vectors, one of the
  vector families RESULTS.md discusses.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) explains how code and claims arrive, and the
documentation rule both are held to.

## Citation

```bibtex
@software{metabasis,
  author = {Luxia},
  title  = {metabasis: steering-vector transport between language models},
  url    = {https://github.com/LuxiaSL/metabasis},
  year   = {2026}
}
```

## License

[MIT](LICENSE)
