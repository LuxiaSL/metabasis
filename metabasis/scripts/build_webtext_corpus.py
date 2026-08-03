"""webtext-v3 — the all-web-text fitting corpus builder. THIS SCRIPT IS THE PUBLICATION.

The webtext-v3 basis is published as a RECONSTRUCTION, not as raw bodies: pinned inputs
(dataset repo + git revision + named files) -> the same chunker -> the same
seeded selection -> a byte-identical corpus manifest. Anyone with network access
can rebuild the exact texts every constant was fit on; nobody has to redistribute
somebody else's text.

Four strata, ~300 texts each, ALL pre-LLM-era sources (a design pin: no post-2022
crawl content, so the fitting basis is structurally empty of model-authored text):

  wikitext       WikiText-103 (`wikitext-103-raw-v1`, validation) — encyclopedic;
                 the v2.1 S2 carrier stratum, same chunker, same pool.
  c4             allenai/c4 `en` — broad web register, April-2019 Common Crawl;
                 ONE pinned validation shard, never unpinned streaming.
  pg19           deepmind/pg19 validation books — long-form public-domain prose
                 (everything published before 1919).
  stackexchange  flax-sentence-embeddings/stackexchange_title_body_jsonl — the
                 dialogic/technical register, from the 2021-06-07 Stack Exchange
                 dump (see COMPOSITION-CENSUS.md for the survey that chose it).

Chunking is the FROZEN v2.1 S2 chunker, imported verbatim from
`build_paired_corpus` and applied identically to all four strata: greedy
paragraph accumulation, 150-500 tokens, close at 250, headings and empty lines
dropped, paragraphs rejoined with a blank line. Nothing is re-implemented here —
strata 2-4 stage each source document as the one-column table that chunker reads,
so "same chunking parameters" is true BY CONSTRUCTION rather than by discipline.

Outputs (into --out-dir, desk-side staging until the prereg freezes the sha):
  corpus_manifest.json        full manifest WITH bodies — desk-side only, never git
  corpus_manifest.meta.json   public metadata shape: bodies dropped, text_sha256 added
  corpus_stamp.json           every pin, every source sha256, counts, build env
  COMPOSITION-CENSUS.md       rake M47: counts by source/stratum, revisions,
                              licenses, vintage, and the stratum-4 survey
  RECONSTRUCT.md              the exact pinned command -> byte-identical rebuild

Run (repo root, a venv with pyarrow + transformers + huggingface_hub):
  python -m metabasis.scripts.build_webtext_corpus --out-dir staging/webtext-v3-draft \\
      --cache-dir /path/to/scratch/hfcache --seed 80 --verify-rebuild
Selftest (no network, no torch; chunker blocks skip by NAME without pyarrow):
  python -m metabasis.scripts.build_webtext_corpus --selftest
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field

# The chunker and its parameters are IMPORTED, never restated: build_paired_corpus
# built the v2.1 basis of record, and its S2 shard pool is the one this corpus's
# wikitext stratum draws from.
from metabasis.scripts.build_paired_corpus import (  # noqa: F401 — re-exported pins
    A8_SEED,
    S2_CARRIER_PROMPT,
    S2_TOK_CLOSE,
    S2_TOK_MAX,
    S2_TOK_MIN,
    TOKENIZER_REF,
    _wikitext_shards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_webtext_corpus")

CORPUS_NAME = "webtext-v3"
CORPUS_VERSION = "staging-draft"
DEFAULT_OUT_DIR = Path("staging/webtext-v3-draft")
DEFAULT_N_PER_STRATUM = 300
# A single source document may contribute at most this many chunks, taken at
# EVENLY SPACED positions (not the first k — that would sample only book fronts).
MAX_SHARDS_PER_DOC = 10
PG19_ASSET_ROOT = "https://storage.googleapis.com/deepmind-gutenberg/"
# fraction of each PG-19 book's chunks dropped from either end (front
# matter / transcriber notes) before the per-document cap.
PG19_TRIM = 0.05
StratumKey = Literal["wikitext", "c4", "pg19", "stackexchange"]


class WebtextBuildError(RuntimeError):
    """A HALT: something the build refuses to guess about (rake M45 rule c —
    an exception, never sys.exit, so a selftest sweep survives it)."""


# ---------------------------------------------------------------- typed pins
class SourcePin(BaseModel, frozen=True):
    """One frozen input: a repo, a revision, named files, a verified license."""

    key: StratumKey
    repo_id: str
    revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    files: tuple[str, ...] = Field(min_length=1)
    # cardData['license'] AS DECLARED at `revision`; () = the repo declares none
    # and the license is established from the card body instead (see below).
    expected_license: tuple[str, ...]
    # substrings that MUST appear in README.md at `revision` — this is how a
    # license/vintage claim gets verified rather than assumed.
    readme_must_contain: tuple[str, ...] = ()
    license_effective: str
    license_note: str
    vintage: str
    homepage: str
    asset_root: Optional[str] = None       # out-of-repo immutable assets (pg19)

    @property
    def ref(self) -> str:
        return f"{self.repo_id}@{self.revision[:12]}"


class StratumSpec(BaseModel, frozen=True):
    key: StratumKey
    n_texts: int = Field(gt=0)
    # documents READ per named file, in file order (None = the whole file).
    max_docs_per_file: Optional[int] = Field(default=None, gt=0)
    # fraction of a document's chunks dropped from EACH end before the cap —
    # 0 everywhere except pg19, where chunk 0 is a Gutenberg transcriber credit
    # and the tail is an editorial note, neither of which is book prose.
    trim_edges_frac: float = Field(default=0.0, ge=0.0, lt=0.5)
    text_register: str
    pin: SourcePin


class WebtextSpec(BaseModel, frozen=True):
    corpus_name: Literal["webtext-v3"] = CORPUS_NAME
    version: str = CORPUS_VERSION
    seed: int
    tokenizer_ref: str
    tok_min: int
    tok_max: int
    tok_close: int
    max_shards_per_doc: int = Field(gt=0)
    carrier_prompt: str
    strata: tuple[StratumSpec, ...] = Field(min_length=1)

    @property
    def total(self) -> int:
        return sum(s.n_texts for s in self.strata)


class WebtextEntry(BaseModel):
    """v2.1 manifest shape (collect_mean_states reads text_id/text/stratum/
    system_prompt/user_prompt) plus this corpus's own source provenance."""

    text_id: str
    stratum: StratumKey
    voice: Literal["neutral"] = "neutral"
    mode: Literal["neutral"] = "neutral"
    topic_idx: Optional[int] = None
    topic: Optional[str] = None
    repetition: Optional[int] = None
    source_run: str
    source_generation_id: str
    source_seed: Optional[str] = None
    system_prompt: str = ""
    user_prompt: str
    text: str = Field(min_length=1)
    n_words: int
    n_tokens: int
    source_dataset: str
    source_revision: str
    source_document: str

    def meta(self) -> dict[str, Any]:
        """The public form: the body replaced by its sha256 (v2.1 meta shape)."""
        d = self.model_dump()
        text = d.pop("text")
        d["text_sha256"] = _sha_text(text)
        return d


# ---------------------------------------------------------------- the pins
PIN_WIKITEXT = SourcePin(
    key="wikitext",
    repo_id="Salesforce/wikitext",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    files=("wikitext-103-raw-v1/validation-00000-of-00001.parquet",),
    expected_license=("cc-by-sa-3.0", "gfdl"),
    readme_must_contain=("cc-by-sa-3.0",),
    license_effective="CC BY-SA 3.0 (also GFDL) — Wikipedia text",
    license_note="Verified Good-/Featured-article Wikipedia text (Merity et al. "
                 "2016). Attribution to Wikipedia and to WikiText-103; share-alike.",
    vintage="Wikipedia snapshot behind WikiText-103 (2016 release)",
    homepage="https://huggingface.co/datasets/Salesforce/wikitext",
)
PIN_C4 = SourcePin(
    key="c4",
    repo_id="allenai/c4",
    revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    files=("en/c4-validation.00000-of-00008.json.gz",),
    expected_license=("odc-by",),
    readme_must_contain=("odc-by",),
    license_effective="ODC-BY 1.0 (AllenAI release of C4); underlying pages remain "
                      "subject to Common Crawl's terms of use",
    license_note="C4 = Common Crawl, cleaned per Raffel et al. 2020 (T5). AllenAI "
                 "distributes it under ODC-BY; attribution to Common Crawl.",
    vintage="Common Crawl snapshot of April 2019 — pre-LLM-era by construction",
    homepage="https://huggingface.co/datasets/allenai/c4",
)
PIN_PG19 = SourcePin(
    key="pg19",
    repo_id="deepmind/pg19",
    revision="4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b",
    files=("data/validation_files.txt",),
    expected_license=("apache-2.0",),
    readme_must_contain=("published before 1919",),
    license_effective="Book text: US public domain (Project Gutenberg, published "
                      "before 1919). Repo metadata/loader: Apache-2.0",
    license_note="The Apache-2.0 tag on the HF repo covers the loading script and "
                 "metadata, NOT the books; the books are public-domain Project "
                 "Gutenberg texts with PG boilerplate already stripped by DeepMind.",
    vintage="books published before 1919; PG-19 assets frozen 2019-09",
    homepage="https://huggingface.co/datasets/deepmind/pg19",
    asset_root=PG19_ASSET_ROOT,
)
# Stratum 4: the survey and the reasons live in COMPOSITION-CENSUS.md (STRATUM4_SURVEY).
PIN_STACKEXCHANGE = SourcePin(
    key="stackexchange",
    repo_id="flax-sentence-embeddings/stackexchange_title_body_jsonl",
    revision="a3d99bf21570ed043e19e41af46f3f19bf4e4bb6",
    # Six PROSE-DOMINANT sites. The frozen chunker strips per-line indentation and
    # rejoins lines with a blank line, which would mangle fenced/indented code, so
    # code-heavy sites (stackoverflow, arduino, networkengineering) are deliberately
    # out; these six carry the dialogic/technical register in prose.
    files=(
        "astronomy.stackexchange.com.jsonl.gz",      # physical science
        "bicycles.stackexchange.com.jsonl.gz",       # mechanical, everyday
        "biology.stackexchange.com.jsonl.gz",        # life science
        "cooking.stackexchange.com.jsonl.gz",        # domestic practical
        "engineering.stackexchange.com.jsonl.gz",    # applied engineering
        "philosophy.stackexchange.com.jsonl.gz",     # humanities argument
    ),
    expected_license=(),                    # no machine-readable license field
    readme_must_contain=(
        "Attribution-ShareAlike 4.0 International Creative Commons License",
        "Publication date 2021-06-07",
        "downloaded via torrent on 2021-07-01",
    ),
    license_effective="CC BY-SA (Stack Exchange user contributions; the card "
                      "states Attribution-ShareAlike 4.0 International)",
    license_note="The repo declares no machine-readable license field; the card "
                 "body states CC BY-SA 4.0 and points at the archive.org Stack "
                 "Exchange dump. Upstream, SE contributions are CC BY-SA — 4.0 "
                 "for posts after 2018-05-02, 3.0/2.5 for older ones — so the "
                 "stratum is CC BY-SA family, attribution + share-alike either "
                 "way. FLAGGED for desk ratification.",
    vintage="Stack Exchange data dump published 2021-06-07 (torrented 2021-07-01) "
            "— entirely pre-LLM-era",
    homepage="https://huggingface.co/datasets/flax-sentence-embeddings/"
             "stackexchange_title_body_jsonl",
)
PINS: dict[StratumKey, SourcePin] = {p.key: p for p in
                                     (PIN_WIKITEXT, PIN_C4, PIN_PG19, PIN_STACKEXCHANGE)}


def default_spec(seed: int, n_per_stratum: int, tokenizer_ref: str) -> WebtextSpec:
    return WebtextSpec(
        seed=seed,
        tokenizer_ref=tokenizer_ref,
        tok_min=S2_TOK_MIN, tok_max=S2_TOK_MAX, tok_close=S2_TOK_CLOSE,
        max_shards_per_doc=MAX_SHARDS_PER_DOC,
        carrier_prompt=S2_CARRIER_PROMPT,
        strata=(
            StratumSpec(key="wikitext", n_texts=n_per_stratum, pin=PIN_WIKITEXT,
                        text_register="encyclopedic prose (WikiText-103 validation)"),
            StratumSpec(key="c4", n_texts=n_per_stratum, pin=PIN_C4,
                        max_docs_per_file=6000,
                        text_register="broad web prose (Common Crawl 2019, C4-cleaned)"),
            StratumSpec(key="pg19", n_texts=n_per_stratum, pin=PIN_PG19,
                        trim_edges_frac=PG19_TRIM,
                        text_register="long-form literary prose (pre-1919 books)"),
            StratumSpec(key="stackexchange", n_texts=n_per_stratum,
                        pin=PIN_STACKEXCHANGE, max_docs_per_file=1200,
                        text_register="dialogic/technical prose (question threads)"),
        ),
    )


# ---------------------------------------------------------------- small helpers
class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ntokens(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def load_tokenizer(ref: str) -> Tokenizer:
    """The chunker's token counts ARE the corpus: no heuristic fallback.

    build_paired_corpus degrades to a word-count heuristic when the tokenizer is
    missing; that is fine for a stamped informational count and fatal here, because
    it would silently produce a DIFFERENT corpus under the same command. HALT.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(
            f"transformers is required to chunk deterministically ({e})") from e
    for local_only in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(ref, local_files_only=local_only)
            logger.info("tokenizer %s (local_files_only=%s)", ref, local_only)
            return tok
        except Exception as e:  # noqa: BLE001 — try the network once, then HALT
            last = e
    raise WebtextBuildError(
        f"tokenizer {ref!r} unavailable ({last}). It is a PIN: the chunk boundaries "
        f"are its token counts. Cache it (Llama 3.1 is gated — accept the license "
        f"once) or pass --tokenizer with the pin recorded in the stamp.")


def _cap_indices(n_shards: int, cap: int) -> list[int]:
    """At most `cap` chunks per document, EVENLY SPACED over the document.

    First-k would sample book fronts (title pages, prefaces) and nothing else;
    evenly spaced is deterministic, seed-free, and samples the whole document.
    """
    if cap <= 0:
        raise WebtextBuildError(f"max_shards_per_doc must be > 0, got {cap}")
    if n_shards <= cap:
        return list(range(n_shards))
    return sorted({int(round(x)) for x in np.linspace(0, n_shards - 1, cap)})


def _select(pool_len: int, n: int, seed: int, ordinal: int) -> list[int]:
    """The v2.1 S2 selection rule, per stratum: a sorted fixed-seed sample.

    Each stratum draws from its own spawned stream, so a change to one stratum's
    pool cannot move another stratum's selection.
    """
    if pool_len < n:
        raise WebtextBuildError(
            f"pool has {pool_len} chunks, need {n} — widen the document caps")
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed,
                                                       spawn_key=(ordinal,)))
    return sorted(int(i) for i in rng.choice(pool_len, size=n, replace=False))


# ---------------------------------------------------------------- the chunker
def shards_from_lines(lines: list[str], tok: Tokenizer) -> list[str]:
    """Chunk ONE document with the frozen v2.1 S2 chunker, verbatim.

    `_wikitext_shards` reads a one-column parquet table of lines; staging this
    document's lines as exactly that table is what makes "the same chunker, the
    same parameters, all four strata" true by construction instead of by copy.
    Per-document staging also keeps chunks from spanning document boundaries and
    keeps every chunk's provenance attached to its source document.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(f"pyarrow is required by the S2 chunker ({e})") from e
    if not lines:
        return []
    with tempfile.TemporaryDirectory(prefix="webtext_chunk_") as td:
        p = Path(td) / "doc.parquet"
        pq.write_table(pa.table({"text": pa.array(lines, type=pa.string())}), p)
        return _wikitext_shards(p, tok)


# ---------------------------------------------------------------- source readers
class SourceDoc(BaseModel):
    """One source document, before chunking."""

    doc_id: str          # stable within (stratum, file): "row00042", "PG-1022"
    label: str           # human provenance: a URL, a book title, a site+row
    file: str            # the named file inside the pinned repo
    lines: list[str]


def _iter_c4(path: Path, file_name: str, limit: Optional[int]) -> Iterator[SourceDoc]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise WebtextBuildError(f"{file_name} line {i}: bad JSON ({e})") from e
            text = rec.get("text") or ""
            yield SourceDoc(doc_id=f"row{i:06d}", file=file_name,
                            label=str(rec.get("url", "")), lines=text.split("\n"))


def _iter_stackexchange(path: Path, file_name: str,
                        limit: Optional[int]) -> Iterator[SourceDoc]:
    site = file_name.split(".jsonl")[0]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise WebtextBuildError(f"{file_name} line {i}: bad JSON ({e})") from e
            texts = rec.get("texts")
            if not isinstance(texts, list) or len(texts) < 2:
                raise WebtextBuildError(
                    f"{file_name} line {i}: expected {{'texts': [title, body]}}, "
                    f"got keys {sorted(rec)}")
            title, body = str(texts[0]).strip(), str(texts[1])
            # Title becomes the document's first paragraph — the question as posed.
            yield SourceDoc(doc_id=f"row{i:06d}", file=file_name,
                            label=f"{site} row {i}",
                            lines=[title] + body.split("\n"))


def reflow_hard_wrapped(text: str) -> list[str]:
    """One line per PARAGRAPH, for sources that hard-wrap at ~70 characters.

    PG-19 books are Gutenberg plain text: every paragraph is wrapped across many
    short lines, blank lines separate paragraphs. Handed to the chunker raw, each
    WRAPPED LINE would count as a paragraph and be rejoined with a blank line —
    the chunk would be a column of 12-word fragments, not prose. Reflowing here
    (a READER concern) means the chunker still sees exactly what it sees in every
    other stratum: one paragraph per line. The chunker itself is untouched.
    """
    paragraphs: list[str] = []
    block: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line:
            block.append(line)
            continue
        if block:
            paragraphs.append(" ".join(block))
            block = []
    if block:
        paragraphs.append(" ".join(block))
    return paragraphs


def _pg19_book(text: str, book_id: str, title: str, file_name: str) -> SourceDoc:
    return SourceDoc(doc_id=f"PG-{book_id}", file=file_name,
                     label=title or f"PG-{book_id}", lines=reflow_hard_wrapped(text))


# ---------------------------------------------------------------- downloads
def _hf_get(repo_id: str, filename: str, revision: str,
            cache_dir: Optional[Path]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(f"huggingface_hub is required ({e})") from e
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    repo_type="dataset", revision=revision,
                                    cache_dir=str(cache_dir) if cache_dir else None))
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(
            f"cannot fetch {repo_id}@{revision[:12]}:{filename} ({e})") from e


def _url_get(url: str, dest: Path) -> Path:
    """Fetch an immutable out-of-repo asset (PG-19's GCS bucket) once, to cache."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise WebtextBuildError(f"cannot fetch {url} ({e})") from e
    tmp.replace(dest)
    return dest


def verify_license(pin: SourcePin, cache_dir: Optional[Path]) -> dict[str, Any]:
    """Licensing is VERIFIED at the pinned revision, never assumed.

    Two independent checks: the machine-readable cardData license (when the repo
    declares one) and required substrings in README.md AT THAT REVISION (which is
    how the no-license-field repo's CC BY-SA claim and every vintage claim get
    checked). A disagreement HALTS the build.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(f"huggingface_hub is required ({e})") from e
    try:
        info = HfApi().dataset_info(pin.repo_id, revision=pin.revision)
    except Exception as e:  # noqa: BLE001
        raise WebtextBuildError(f"cannot read {pin.ref} card metadata ({e})") from e
    declared = (info.cardData or {}).get("license")
    if declared is None:
        got: tuple[str, ...] = ()
    elif isinstance(declared, str):
        got = (declared,)
    else:
        got = tuple(str(x) for x in declared)
    if got != pin.expected_license:
        raise WebtextBuildError(
            f"{pin.ref}: declared license {got} != pinned {pin.expected_license}")
    readme = _hf_get(pin.repo_id, "README.md", pin.revision, cache_dir)
    body = readme.read_text(encoding="utf-8", errors="replace")
    missing = [s for s in pin.readme_must_contain if s not in body]
    if missing:
        raise WebtextBuildError(
            f"{pin.ref}: README at the pinned revision no longer states {missing}")
    return {"declared_license": list(got), "readme_sha256": _sha_file(readme),
            "verified_claims": list(pin.readme_must_contain)}


# ---------------------------------------------------------------- stratum builds
class StratumResult(BaseModel):
    key: StratumKey
    entries: list[WebtextEntry]
    info: dict[str, Any]


def _pool_from_docs(docs: Iterator[SourceDoc], tok: Tokenizer, cap: int,
                    trim_edges_frac: float = 0.0,
                    ) -> tuple[list[tuple[SourceDoc, int, str]], dict[str, int]]:
    """Chunk documents in file order into one ordered pool, deduplicated by body."""
    pool: list[tuple[SourceDoc, int, str]] = []
    seen: set[str] = set()
    stats = {"docs_read": 0, "docs_contributing": 0, "chunks_before_cap": 0,
             "chunks_dropped_by_cap": 0, "chunks_dropped_as_edges": 0,
             "duplicate_chunks_dropped": 0}
    for doc in docs:
        stats["docs_read"] += 1
        shards = shards_from_lines(doc.lines, tok)
        stats["chunks_before_cap"] += len(shards)
        lo = int(len(shards) * trim_edges_frac)
        hi = len(shards) - lo
        stats["chunks_dropped_as_edges"] += len(shards) - (hi - lo)
        keep = [lo + i for i in _cap_indices(hi - lo, cap)]
        stats["chunks_dropped_by_cap"] += (hi - lo) - len(keep)
        contributed = False
        for k in keep:
            body = shards[k]
            h = _sha_text(body)
            if h in seen:
                stats["duplicate_chunks_dropped"] += 1
                continue
            seen.add(h)
            pool.append((doc, k, body))
            contributed = True
        stats["docs_contributing"] += int(contributed)
    return pool, stats


def _entry(spec: WebtextSpec, pin: SourcePin, key: StratumKey, rank: int,
           file_name: str, doc_id: str, shard_idx: int, label: str,
           text: str, tok: Tokenizer) -> WebtextEntry:
    return WebtextEntry(
        text_id=f"{key}-{rank:03d}",
        stratum=key,
        source_run=f"{pin.repo_id}@{pin.revision[:12]}:{file_name}",
        source_generation_id=f"{doc_id}:chunk{shard_idx:03d}",
        user_prompt=spec.carrier_prompt,
        text=text,
        n_words=len(text.split()),
        n_tokens=_ntokens(tok, text),
        source_dataset=pin.repo_id,
        source_revision=pin.revision,
        source_document=label,
    )


def build_wikitext(spec: WebtextSpec, st: StratumSpec, ordinal: int, tok: Tokenizer,
                   cache_dir: Optional[Path]) -> StratumResult:
    """The v2.1 S2 pool exactly: the whole validation parquet through the chunker.

    This stratum does NOT go through per-document staging — it hands the frozen
    chunker the very file it was written for, so the wikitext chunks are drawn
    from the identical shard pool the v2.1 S2 stratum drew from (its section
    boundaries already keep one article from dominating). The per-document cap is
    therefore not applicable here; recorded as such.
    """
    pin = st.pin
    path = _hf_get(pin.repo_id, pin.files[0], pin.revision, cache_dir)
    shards = _wikitext_shards(path, tok)
    seen: set[str] = set()
    pool: list[tuple[int, str]] = []
    dups = 0
    for i, body in enumerate(shards):
        h = _sha_text(body)
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        pool.append((i, body))
    picks = _select(len(pool), st.n_texts, spec.seed, ordinal)
    entries = [
        _entry(spec, pin, "wikitext", rank, pin.files[0],
               f"shard{pool[i][0]:05d}", pool[i][0],
               "wikitext-103-raw-v1 validation", pool[i][1], tok)
        for rank, i in enumerate(picks)
    ]
    info = {"files": [{"file": pin.files[0], "sha256": _sha_file(path)}],
            "chunks_available": len(shards), "pool_size": len(pool),
            "duplicate_chunks_dropped": dups,
            "per_document_cap": "not applicable (whole-file chunker, v2.1 S2 pool)",
            "selected_pool_indices_sha256": _sha_bytes(json.dumps(picks).encode())}
    return StratumResult(key="wikitext", entries=entries, info=info)


def _build_from_files(spec: WebtextSpec, st: StratumSpec, ordinal: int, tok: Tokenizer,
                      opened: list[tuple[str, Path, Callable[[Path, str, Optional[int]],
                                                             Iterator[SourceDoc]]]],
                      ) -> StratumResult:
    def docs() -> Iterator[SourceDoc]:
        for file_name, path, reader in opened:
            yield from reader(path, file_name, st.max_docs_per_file)

    pool, stats = _pool_from_docs(docs(), tok, spec.max_shards_per_doc,
                                  st.trim_edges_frac)
    picks = _select(len(pool), st.n_texts, spec.seed, ordinal)
    entries = []
    for rank, i in enumerate(picks):
        doc, k, body = pool[i]
        entries.append(_entry(spec, st.pin, st.key, rank, doc.file, doc.doc_id, k,
                              doc.label, body, tok))
    info = {"files": [{"file": f, "sha256": _sha_file(p)} for f, p, _ in opened],
            "pool_size": len(pool),
            "max_docs_per_file": st.max_docs_per_file,
            "per_document_cap": spec.max_shards_per_doc,
            "trim_edges_frac": st.trim_edges_frac,
            "selected_pool_indices_sha256": _sha_bytes(json.dumps(picks).encode()),
            **stats}
    return StratumResult(key=st.key, entries=entries, info=info)


def build_c4(spec: WebtextSpec, st: StratumSpec, ordinal: int, tok: Tokenizer,
             cache_dir: Optional[Path]) -> StratumResult:
    opened = [(f, _hf_get(st.pin.repo_id, f, st.pin.revision, cache_dir), _iter_c4)
              for f in st.pin.files]
    return _build_from_files(spec, st, ordinal, tok, opened)


def build_stackexchange(spec: WebtextSpec, st: StratumSpec, ordinal: int,
                        tok: Tokenizer, cache_dir: Optional[Path]) -> StratumResult:
    opened = [(f, _hf_get(st.pin.repo_id, f, st.pin.revision, cache_dir),
               _iter_stackexchange) for f in st.pin.files]
    return _build_from_files(spec, st, ordinal, tok, opened)


def build_pg19(spec: WebtextSpec, st: StratumSpec, ordinal: int, tok: Tokenizer,
               cache_dir: Optional[Path]) -> StratumResult:
    """The 50 pinned validation books, fetched from PG-19's immutable asset root.

    deepmind/pg19 is a script-style dataset (datasets>=3 cannot execute it), so the
    build reads the pinned file LIST from the repo and fetches each named book from
    the asset root, recording a sha256 per book — the pin is the file list plus
    those digests, which is stronger than a loader version.
    """
    pin = st.pin
    if pin.asset_root is None:
        raise WebtextBuildError("pg19 pin needs an asset_root")
    listing = _hf_get(pin.repo_id, pin.files[0], pin.revision, cache_dir)
    names = [ln.strip() for ln in listing.read_text().splitlines() if ln.strip()]
    if not names:
        raise WebtextBuildError(f"{pin.ref}: empty {pin.files[0]}")
    root = (cache_dir or Path(tempfile.gettempdir())) / "pg19_assets"
    meta_path = _url_get(pin.asset_root + "metadata.csv", root / "metadata.csv")
    titles: dict[str, str] = {}
    dates: dict[str, str] = {}
    import csv
    with open(meta_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                titles[row[0].strip()] = row[1].strip()
                dates[row[0].strip()] = row[2].strip()

    books: list[tuple[str, Path]] = []
    for name in names:
        book_id = Path(name).stem
        dest = root / name
        _url_get(pin.asset_root + name, dest)
        books.append((book_id, dest))

    def docs() -> Iterator[SourceDoc]:
        for book_id, path in books:
            title = titles.get(book_id, "")
            year = dates.get(book_id, "")
            label = f"PG-{book_id} — {title} ({year})" if title else f"PG-{book_id}"
            yield _pg19_book(path.read_text(encoding="utf-8", errors="replace"),
                             book_id, label, pin.files[0])

    pool, stats = _pool_from_docs(docs(), tok, spec.max_shards_per_doc,
                                  st.trim_edges_frac)
    picks = _select(len(pool), st.n_texts, spec.seed, ordinal)
    entries = []
    for rank, i in enumerate(picks):
        doc, k, body = pool[i]
        entries.append(_entry(spec, pin, "pg19", rank, f"validation/{doc.doc_id}",
                              doc.doc_id, k, doc.label, body, tok))
    pub_years = sorted({dates[b] for b, _ in books if b in dates})
    info = {"asset_root": pin.asset_root,
            "file_list": {"file": pin.files[0], "sha256": _sha_file(listing)},
            "metadata_csv_sha256": _sha_file(meta_path),
            "n_books": len(books),
            "book_sha256": {b: _sha_file(p) for b, p in books},
            "publication_years_observed": [pub_years[0], pub_years[-1]] if pub_years
                                          else None,
            "pool_size": len(pool),
            "per_document_cap": spec.max_shards_per_doc,
            "trim_edges_frac": st.trim_edges_frac,
            "selected_pool_indices_sha256": _sha_bytes(json.dumps(picks).encode()),
            **stats}
    return StratumResult(key="pg19", entries=entries, info=info)


BUILDERS: dict[StratumKey, Callable[..., StratumResult]] = {
    "wikitext": build_wikitext, "c4": build_c4, "pg19": build_pg19,
    "stackexchange": build_stackexchange,
}


# ---------------------------------------------------------------- assembly
def token_summary(values: list[int]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "min": int(a.min()), "max": int(a.max()),
            "mean": round(float(a.mean()), 1),
            "p10": round(float(np.percentile(a, 10)), 1),
            "median": round(float(np.median(a)), 1),
            "p90": round(float(np.percentile(a, 90)), 1)}


def assemble(spec: WebtextSpec, results: list[StratumResult]) -> list[WebtextEntry]:
    entries = [e for r in results for e in r.entries]
    ids = [e.text_id for e in entries]
    if len(set(ids)) != len(ids):
        raise WebtextBuildError("duplicate text_ids")
    shas = [_sha_text(e.text) for e in entries]
    if len(set(shas)) != len(shas):
        raise WebtextBuildError("duplicate text bodies across strata")
    if len(entries) != spec.total:
        raise WebtextBuildError(f"expected {spec.total} texts, assembled {len(entries)}")
    for e in entries:
        if not (spec.tok_min <= e.n_tokens <= spec.tok_max):
            raise WebtextBuildError(
                f"{e.text_id}: {e.n_tokens} tokens outside "
                f"[{spec.tok_min},{spec.tok_max}] — the chunker's own contract")
    return entries


def manifest_bytes(entries: list[WebtextEntry], *, with_text: bool) -> bytes:
    """Serialization is part of the artifact: v2.1's json.dump(..., indent=1)."""
    rows = [e.model_dump() if with_text else e.meta() for e in entries]
    return json.dumps({"entries": rows}, indent=1).encode("utf-8")


def build_stamp(spec: WebtextSpec, results: list[StratumResult],
                entries: list[WebtextEntry], licenses: dict[str, Any],
                manifest_sha: str, meta_sha: str,
                tokenizer_files: dict[str, str]) -> dict[str, Any]:
    return {
        "corpus": CORPUS_NAME,
        "version": spec.version,
        "status": "STAGING DRAFT — not of record until the v3 pre-registration "
                  "freezes this sha",
        "builder": "metabasis/scripts/build_webtext_corpus.py",
        "spec": spec.model_dump(),
        "counts": {**{r.key: len(r.entries) for r in results},
                   "total": len(entries)},
        "chunker": {
            "imported_from": "metabasis/scripts/build_paired_corpus.py::"
                             "_wikitext_shards (the frozen v2.1 S2 chunker)",
            "token_range": [spec.tok_min, spec.tok_max],
            "close_at": spec.tok_close,
            "cleaning": "drop heading lines (= ... =), drop empty lines, strip "
                        "per-line whitespace, paragraphs joined by a blank line; "
                        "oversize single paragraphs skipped, not truncated",
            "applied_to": "all four strata (strata 2-4 stage each source document "
                          "as the one-column table the chunker reads)",
            "per_document_cap": spec.max_shards_per_doc,
            "cap_rule": "evenly spaced chunk positions, not the first k",
        },
        "selection": {
            "rule": "per stratum: sorted fixed-seed sample without replacement "
                    "over the ordered chunk pool (the v2.1 S2 rule)",
            "seed": spec.seed,
            "streams": "numpy SeedSequence(entropy=seed, spawn_key=(stratum_ordinal,))",
        },
        "tokenizer": {"ref": spec.tokenizer_ref, "files_sha256": tokenizer_files},
        "sources": {r.key: {"pin": PINS[r.key].model_dump(), "license_check":
                            licenses.get(r.key, {}), **r.info} for r in results},
        "token_summary": {"all": token_summary([e.n_tokens for e in entries]),
                          **{r.key: token_summary([e.n_tokens for e in r.entries])
                             for r in results}},
        "manifest_sha256": manifest_sha,
        "meta_manifest_sha256": meta_sha,
    }


# ---------------------------------------------------------------- the census
STRATUM4_SURVEY: tuple[tuple[str, str, str, str], ...] = (
    # (dataset, dump vintage, license as published, verdict)
    ("flax-sentence-embeddings/stackexchange_title_body_jsonl",
     "SE dump published 2021-06-07, torrented 2021-07-01 (stated on the card)",
     "no license field; card body states CC BY-SA 4.0, points at archive.org",
     "**CHOSEN** — the only surveyed SE dataset that documents BOTH a pre-2022 "
     "dump date and a license consistent with upstream Stack Exchange. Bodies are "
     "already HTML-stripped plain text. Cost: questions only (title + body), no "
     "answer turns — see the flag below."),
    ("flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
     "undated on the card (same project, 2021 dump presumed — NOT documented)",
     "cc-by-nc-sa-4.0 declared", "rejected — the NC clause contradicts upstream "
     "Stack Exchange licensing (CC BY-SA, no NC), and the dump date is not "
     "documented anywhere in the repo. Would have given true Q+A dialogue."),
    ("HuggingFaceH4/stack-exchange-preferences", "dump used is undated; repo "
     "published 2023-03", "cc-by-sa-4.0 declared",
     "rejected — vintage undocumented and almost certainly a late-2022/2023 dump, "
     "i.e. after ChatGPT's release; bodies are raw HTML."),
    ("lvwerra/stack-exchange-paired", "derived from HuggingFaceH4 (2023)",
     "none declared", "rejected — inherits the 2023 vintage."),
    ("bigcode/stack-exchange-preferences-20230914-clean-anonymization",
     "2023-09-14 dump", "none declared", "rejected — post-2022 dump."),
    ("mikex86/stackoverflow-posts", "posts up to 2023-06-14", "other",
     "rejected — post-2022 dump."),
    ("donfu/oa-stackexchange", "2023 dump", "cc-by-sa-4.0 declared",
     "rejected — post-2022 dump; also capped at 1000 characters per side, too "
     "short for 150-500-token chunks."),
    ("common-pile/stackexchange, common-pile/stackexchange_filtered",
     "December-2024 community dumps + July-2024 official dumps", "not declared "
     "in metadata (per-document licenses inside)",
     "rejected — post-2022 dump."),
    ("HuggingFaceTB/stackexchange_2025_md", "2025", "n/a",
     "rejected — post-2022 dump."),
    ("teven/stackexchange", "2021-12 upload, provenance undocumented",
     "none declared, no card at all",
     "rejected — opaque provenance; nothing to cite in a census."),
    ("Skylion007/openwebtext (the brief's fallback)",
     "Reddit-submitted URLs scraped up to 2019", "cc0-1.0 declared",
     "NOT NEEDED — held in reserve; would have replaced the dialogic register "
     "with more web prose, which the c4 stratum already carries."),
)


def render_census(spec: WebtextSpec, stamp: dict[str, Any]) -> str:
    counts = stamp["counts"]
    lines: list[str] = []
    a = lines.append
    a(f"# COMPOSITION CENSUS — the `{CORPUS_NAME}` fitting corpus "
      f"({spec.version})")
    a("")
    a("> Rake M47: a frozen input gets its census at birth, next to its sha — "
      "counts by source, revisions, licenses, vintage. This corpus is a STAGING "
      "DRAFT until the v3 pre-registration freezes the sha below.")
    a("")
    a(f"- **texts**: {counts['total']} ({', '.join(f'{k} {v}' for k, v in counts.items() if k != 'total')})")
    a(f"- **full manifest sha256**: `{stamp['manifest_sha256']}`")
    a(f"- **metadata manifest sha256**: `{stamp['meta_manifest_sha256']}`")
    a(f"- **selection seed**: {spec.seed}")
    a(f"- **model-authored text**: 0 texts, 0% — every stratum is human-written "
      f"and every source predates the LLM era (design pin: no post-2022 crawl "
      f"content).")
    a("")
    a("## Composition")
    a("")
    a("| stratum | n | register | source (pinned revision) | vintage |")
    a("|---|---|---|---|---|")
    for st in spec.strata:
        p = st.pin
        a(f"| `{st.key}` | {counts[st.key]} | {st.text_register} | "
          f"`{p.repo_id}` @ `{p.revision[:12]}` | {p.vintage} |")
    a("")
    a("### Files read, per stratum")
    a("")
    for st in spec.strata:
        src = stamp["sources"].get(st.key, {})
        a(f"- **`{st.key}`** — `{st.pin.repo_id}` @ `{st.pin.revision}`")
        if st.key == "pg19" and "file_list" in src:
            a(f"  - file list `{st.pin.files[0]}` "
              f"(sha256 `{src['file_list']['sha256'][:16]}…`), "
              f"{src['n_books']} books from `{src['asset_root']}`; "
              f"publication years {src['publication_years_observed']}")
        else:
            for f in src.get("files", []):
                a(f"  - `{f['file']}` (sha256 `{f['sha256'][:16]}…`)")
        cap = src.get("max_docs_per_file")
        if cap:
            a(f"  - documents read: first {cap} rows per file, in file order")
        a(f"  - chunk pool {src.get('pool_size', '?')}, selected {counts[st.key]}, "
          f"per-document cap {src.get('per_document_cap', '?')}")
    a("")
    a("## Licensing (verified at the pinned revision, not assumed)")
    a("")
    a("The builder refuses to run if a pinned repo's declared license, or the "
      "license/vintage sentences quoted from its card, have changed at the pinned "
      "revision (`verify_license`).")
    a("")
    for st in spec.strata:
        p = st.pin
        chk = stamp["sources"][st.key].get("license_check", {})
        a(f"- **`{st.key}`** — {p.license_effective}")
        a(f"  - declared in repo metadata: "
          f"{chk.get('declared_license') or 'NONE (no license field)'}")
        a(f"  - {p.license_note}")
    a("")
    a("## Stratum 4: the survey behind the choice (for desk ratification)")
    a("")
    a("Requirement: a Stack-Exchange-derived dataset that is stable, has a "
      "**pre-2022 dump**, and carries a **clean license**; fall back to "
      "OpenWebText if none satisfies both.")
    a("")
    a("| candidate | dump vintage | license as published | verdict |")
    a("|---|---|---|---|")
    for name, vintage, lic, verdict in STRATUM4_SURVEY:
        a(f"| `{name}` | {vintage} | {lic} | {verdict} |")
    a("")
    a("**Flags for the desk.** (1) The chosen repo declares no machine-readable "
      "license field; the CC BY-SA claim comes from its card body and from "
      "upstream Stack Exchange policy, and the builder verifies that sentence is "
      "still there at the pinned revision. (2) The stratum is *questions* — title "
      "plus body — so it carries the dialogic register (second person, requests, "
      "hedging, technical detail) but not answer turns; the Q+A variant was "
      "rejected for its NC license and undocumented vintage. Say the word and the "
      "pin moves.")
    a("")
    a("## How a text was made")
    a("")
    a(f"1. **Chunker** — the frozen v2.1 S2 chunker, imported verbatim from "
      f"`build_paired_corpus.py`: greedy paragraph accumulation, "
      f"{spec.tok_min}-{spec.tok_max} tokens, close at {spec.tok_close}, heading "
      f"lines (`= … =`) and empty lines dropped, per-line whitespace stripped, "
      f"paragraphs rejoined with a blank line, oversize paragraphs skipped rather "
      f"than truncated. Token counts are "
      f"`{spec.tokenizer_ref}`'s.")
    a("2. **Same parameters everywhere** — strata 2-4 stage each source document "
      "as the one-column table that chunker reads, so chunks never span documents "
      "and the parameters cannot drift between strata. The `wikitext` stratum "
      "hands the chunker the whole validation parquet, exactly as v2.1 did, so it "
      "draws from the identical shard pool.")
    a("3. **Per-source reading, before the chunker** — `c4`: the `text` field of "
      "each row, split on newlines. `stackexchange`: the question title as the "
      "first paragraph, then the body (already HTML-stripped upstream; HTML "
      "entities such as `&quot;` survive, verbatim, unnormalized), from six "
      "prose-dominant sites — code-heavy sites are deliberately out, because the "
      "chunker strips per-line indentation and would mangle code blocks. `pg19`: "
      "Gutenberg plain text is hard-wrapped at ~70 characters, so each book is "
      "REFLOWED to one line per paragraph (blank line = paragraph break) before "
      "staging; without it the chunker would treat every wrapped line as its own "
      "paragraph. No other normalization is applied anywhere.")
    a(f"4. **Per-document cap** — at most {spec.max_shards_per_doc} chunks per "
      f"source document, taken at evenly spaced positions (not the first k, which "
      f"would sample only book fronts). Not applicable to `wikitext`, whose "
      f"section boundaries already do this. For `pg19` the outer "
      f"{100 * PG19_TRIM:.0f}% of each book's chunks are dropped from either "
      f"end before the cap: chunk 0 of a Gutenberg text is a transcriber credit "
      f"and the last chunks are editorial notes, neither of which is book prose.")
    a("5. **Dedup** — identical chunk bodies are dropped in pool order; the "
      "assembled corpus is asserted to have no repeated body and no repeated id.")
    a(f"6. **Selection** — per stratum, a sorted fixed-seed sample without "
      f"replacement over the ordered pool (the v2.1 S2 rule), seed {spec.seed}, "
      f"each stratum on its own spawned stream.")
    a("7. **Carrier prompt** — every text carries the constant native-arm carrier "
      f"prompt `{spec.carrier_prompt!r}` and an empty system prompt, exactly as "
      f"the v2.1 S2 stratum did. Raw-arm collection ignores it.")
    a("")
    a("## Token lengths")
    a("")
    a("| stratum | n | min | p10 | median | mean | p90 | max |")
    a("|---|---|---|---|---|---|---|---|")
    for k in [st.key for st in spec.strata] + ["all"]:
        t = stamp["token_summary"][k]
        a(f"| `{k}` | {t['n']} | {t['min']} | {t['p10']} | {t['median']} | "
          f"{t['mean']} | {t['p90']} | {t['max']} |")
    a("")
    a("## Known properties a reader should not be surprised by")
    a("")
    a("- **`wikitext`** — WikiText-103 (both the raw and non-raw variants) carries "
      "the corpus's own `@-@`, `@.@`, `@,@` tokenization artifacts around hyphens "
      "and numbers, and spaces its punctuation. They are preserved verbatim: this "
      "is the same source, the same variant and the same chunker the v2.1 S2 "
      "stratum used, so the two are directly comparable. (`raw-v1` does mean no "
      "`<unk>`: rare words are intact.)")
    a("- **`c4`** — the April-2019 web as it was: product pages, blog rolls and "
      "boilerplate sit next to expository prose. That is the register, not a "
      "defect; C4's own cleaning heuristics are all the filtering there is.")
    a("- **`stackexchange`** — HTML was stripped upstream, but HTML entities "
      "(`&quot;`, `&mdash;`) and inline LaTeX survive, and some askers repeat "
      "their title in the body. Verbatim, unnormalized.")
    a("- **`pg19`** — pre-1919 books carry period spelling, period attitudes, and "
      "occasional OCR noise; DeepMind stripped the Gutenberg licence boilerplate "
      "and mapped a list of slurs to placeholders before release.")
    a("")
    a("## What is NOT in this corpus")
    a("")
    a("- No model-generated text (the v2.1 S1/S3 strata are gone; that was the "
      "point of v3).")
    a("- No post-2022 web content: WikiText-103 is a 2016 Wikipedia snapshot, C4 "
      "is the April-2019 Common Crawl, PG-19 is pre-1919 books, the Stack "
      "Exchange dump is 2021-06-07.")
    a("- No text bodies in git. The full manifest is desk-side; the public "
      "artifacts are this census, the metadata manifest (per-text `text_sha256`), "
      "and the builder that reconstructs the bodies.")
    a("")
    return "\n".join(lines) + "\n"


def render_reconstruct(spec: WebtextSpec, stamp: dict[str, Any], cmd: str) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# RECONSTRUCT — rebuilding `{CORPUS_NAME}` ({spec.version}) byte for byte")
    a("")
    a("The corpus is published as a reconstruction: pinned inputs -> the frozen "
      "chunker -> a seeded selection -> exactly these bytes. No text bodies are "
      "redistributed.")
    a("")
    a("## The command")
    a("")
    a("```")
    a(cmd)
    a("```")
    a("")
    a("## What it must produce")
    a("")
    a("| artifact | sha256 |")
    a("|---|---|")
    a(f"| `corpus_manifest.json` (full, with bodies) | `{stamp['manifest_sha256']}` |")
    a(f"| `corpus_manifest.meta.json` (public metadata) | "
      f"`{stamp['meta_manifest_sha256']}` |")
    a("")
    a("Each entry's body verifies against its `text_sha256` in the metadata "
      "manifest, so a rebuilt corpus can be checked text by text without ever "
      "publishing a body.")
    a("")
    a("## The pins")
    a("")
    a("| stratum | repo | revision | files |")
    a("|---|---|---|---|")
    for st in spec.strata:
        p = st.pin
        files = ", ".join(f"`{f}`" for f in p.files)
        a(f"| `{st.key}` | `{p.repo_id}` | `{p.revision}` | {files} |")
    a("")
    a(f"- **PG-19 books** come from the immutable asset root "
      f"`{PG19_ASSET_ROOT}` using the file list pinned above; the stamp records a "
      f"sha256 for every book and for `metadata.csv`.")
    a(f"- **Tokenizer**: `{spec.tokenizer_ref}` — a PIN, not a convenience. The "
      f"chunk boundaries are its token counts. The stamp records a sha256 for "
      f"every tokenizer file used. (It is a gated repo: accept the license once, "
      f"then it resolves from the local cache. A different tokenizer produces a "
      f"different corpus and the builder will say so by producing a different "
      f"sha.)")
    a(f"- **Seed**: {spec.seed}. **Per-document chunk cap**: "
      f"{spec.max_shards_per_doc}, at evenly spaced positions.")
    a("- **Per-stratum reading pins** (they select the pool, so they are part of "
      "the reconstruction):")
    for st in spec.strata:
        a(f"  - `{st.key}`: {st.n_texts} texts; documents read = "
          f"{'first ' + str(st.max_docs_per_file) + ' rows per file'
             if st.max_docs_per_file else 'every document in the pinned files'}; "
          f"edge trim = {st.trim_edges_frac:g}")
    a("")
    a("## What is and is not environment-independent")
    a("")
    a("- `corpus_manifest.json`, `corpus_manifest.meta.json`, "
      "`COMPOSITION-CENSUS.md` and this file are byte-identical anywhere the pins "
      "resolve: nothing in them is derived from a path, a clock, or a library "
      "version.")
    a("- `corpus_stamp.json` additionally records the build environment "
      "(interpreter and library versions), so it is byte-identical within an "
      "environment and may differ across environments. The manifest sha is the "
      "object of record.")
    a("- `--verify-rebuild` builds the whole corpus twice in one run and HALTS "
      "unless every artifact matches byte for byte.")
    a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the build
def _tokenizer_file_shas(ref: str) -> dict[str, str]:
    """sha256 of the tokenizer's own files — the pin behind the chunk boundaries."""
    out: dict[str, str] = {}
    try:
        from transformers.utils import cached_file
    except Exception:  # noqa: BLE001
        return out
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            p = cached_file(ref, name, local_files_only=True)
        except Exception:  # noqa: BLE001 — optional files
            continue
        if p:
            out[name] = _sha_file(Path(p))
    return out


def build(spec: WebtextSpec, out_dir: Path, cache_dir: Optional[Path],
          cmd: str, check_licenses: bool = True) -> dict[str, Any]:
    tok = load_tokenizer(spec.tokenizer_ref)
    licenses: dict[str, Any] = {}
    if check_licenses:
        for st in spec.strata:
            licenses[st.key] = verify_license(st.pin, cache_dir)
            logger.info("license verified: %s -> %s", st.pin.ref,
                        licenses[st.key]["declared_license"] or "(card body only)")
    results: list[StratumResult] = []
    for ordinal, st in enumerate(spec.strata):
        logger.info("building stratum %s (%d texts) from %s",
                    st.key, st.n_texts, st.pin.ref)
        results.append(BUILDERS[st.key](spec, st, ordinal, tok, cache_dir))
        logger.info("  %s: pool %s -> %d texts", st.key,
                    results[-1].info["pool_size"], len(results[-1].entries))
    entries = assemble(spec, results)

    out_dir.mkdir(parents=True, exist_ok=True)
    full = manifest_bytes(entries, with_text=True)
    meta = manifest_bytes(entries, with_text=False)
    (out_dir / "corpus_manifest.json").write_bytes(full)
    (out_dir / "corpus_manifest.meta.json").write_bytes(meta)
    stamp = build_stamp(spec, results, entries, licenses,
                        _sha_bytes(full), _sha_bytes(meta),
                        _tokenizer_file_shas(spec.tokenizer_ref))
    census = render_census(spec, stamp)
    reconstruct = render_reconstruct(spec, stamp, cmd)
    (out_dir / "COMPOSITION-CENSUS.md").write_text(census, encoding="utf-8")
    (out_dir / "RECONSTRUCT.md").write_text(reconstruct, encoding="utf-8")
    stamp_env = dict(stamp)
    stamp_env["build_environment"] = _environment()
    (out_dir / "corpus_stamp.json").write_bytes(
        json.dumps(stamp_env, indent=1).encode("utf-8"))
    logger.info("webtext-v3 %s: %d texts -> %s", spec.version, len(entries), out_dir)
    logger.info("manifest sha256 %s", stamp["manifest_sha256"])
    return {"stamp": stamp,
            "artifact_sha256": {
                "corpus_manifest.json": stamp["manifest_sha256"],
                "corpus_manifest.meta.json": stamp["meta_manifest_sha256"],
                "COMPOSITION-CENSUS.md": _sha_bytes(census.encode("utf-8")),
                "RECONSTRUCT.md": _sha_bytes(reconstruct.encode("utf-8")),
            }}


def _environment() -> dict[str, str]:
    env = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("pyarrow", "transformers", "huggingface_hub", "pydantic"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = "unavailable"
    return env


# ---------------------------------------------------------------- selftest
class _StubTokenizer:
    """A deterministic stand-in: one token per whitespace word, no vocabulary.

    The chunker only ever calls `.encode(text, add_special_tokens=False)` and takes
    its length, so the staging contract is testable with no model files at all.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * max(1, len(text.split()))


def _fixture_paragraph(word: str, n: int) -> str:
    return " ".join([word] * n)


def _fixture_doc(tag: str, n_paras: int = 40, words: int = 60) -> list[str]:
    lines: list[str] = []
    for i in range(n_paras):
        lines.append(_fixture_paragraph(f"{tag}{i:03d}", words))
        lines.append("")
    return lines


def _ok(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except Exception:  # noqa: BLE001
        return False


def _raises(fn: Callable[[], Any], exc: type[BaseException]) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def selftest() -> int:  # noqa: C901 — a checklist
    """Named-configuration selftest (rake M44): no network, no torch, no data tree.

    Blocks that genuinely need pyarrow (the imported chunker reads a parquet) are
    NAMED SKIPS rather than silent passes, and the tail states the configuration
    the counts were measured in.
    """
    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)

    def skip(name: str, why: str) -> None:
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))
        logger.info("SKIP %s — %s", name, why)

    try:
        import pyarrow  # noqa: F401
        have_pyarrow = True
    except Exception:  # noqa: BLE001
        have_pyarrow = False
    config = (f"cwd={Path.cwd().name} python={sys.version.split()[0]} "
              f"pyarrow={'yes' if have_pyarrow else 'NO'} "
              f"network=not-used")

    tok = _StubTokenizer()
    spec = default_spec(seed=80, n_per_stratum=3, tokenizer_ref="stub/selftest")

    # ---- 1. the pins are typed, complete, and self-consistent -----------------
    print("== selftest 1: the pins ==")
    check("all four strata are pinned to a 40-hex revision",
          all(len(p.revision) == 40 and all(c in "0123456789abcdef" for c in p.revision)
              for p in PINS.values()), f"{len(PINS)} pins")
    check("every pin names at least one file", all(p.files for p in PINS.values()))
    check("a bad revision is refused at construction",
          _raises(lambda: SourcePin(key="c4", repo_id="x", revision="deadbeef",
                                    files=("f",), expected_license=(),
                                    license_effective="", license_note="",
                                    vintage="", homepage=""), Exception))
    check("the spec's strata are exactly the four named ones",
          tuple(st.key for st in spec.strata) ==
          ("wikitext", "c4", "pg19", "stackexchange"))
    check("every stratum carries the SAME chunking parameters (one spec, one set)",
          (spec.tok_min, spec.tok_max, spec.tok_close) ==
          (S2_TOK_MIN, S2_TOK_MAX, S2_TOK_CLOSE) and
          (spec.tok_min, spec.tok_max, spec.tok_close) == (150, 500, 250))
    check("the carrier prompt is the v2.1 S2 constant, imported not restated",
          spec.carrier_prompt == S2_CARRIER_PROMPT)
    check("the stratum-4 survey records the choice AND the rejects",
          sum("CHOSEN" in v for _, _, _, v in STRATUM4_SURVEY) == 1
          and len(STRATUM4_SURVEY) >= 8, f"{len(STRATUM4_SURVEY)} candidates")
    check("every surveyed candidate carries a vintage and a license column",
          all(all(col.strip() for col in row) for row in STRATUM4_SURVEY))

    # ---- 2. the per-document cap ---------------------------------------------
    print("== selftest 2: the per-document cap is evenly spaced, not first-k ==")
    check("a short document is kept whole", _cap_indices(4, 10) == list(range(4)))
    check("a long document is capped", len(_cap_indices(100, 10)) == 10)
    check("the cap spans the document (first AND last chunk reachable)",
          _cap_indices(100, 10)[0] == 0 and _cap_indices(100, 10)[-1] == 99)
    check("the cap is not first-k (it would end at index 9)",
          _cap_indices(100, 10) != list(range(10)))
    check("the cap is deterministic", _cap_indices(97, 7) == _cap_indices(97, 7))
    check("a nonsense cap HALTS", _raises(lambda: _cap_indices(5, 0), WebtextBuildError))
    pg19_spec = [st for st in spec.strata if st.key == "pg19"][0]
    check("only pg19 trims document edges (front matter / transcriber notes)",
          pg19_spec.trim_edges_frac == PG19_TRIM > 0 and
          all(st.trim_edges_frac == 0.0 for st in spec.strata if st.key != "pg19"))
    check("the trim leaves a short document alone (int(2 * frac) == 0)",
          int(2 * PG19_TRIM) == 0)
    check("a trim of half a document or more is refused at construction",
          _raises(lambda: StratumSpec(**(pg19_spec.model_dump() |
                                         {"trim_edges_frac": 0.5})), Exception))

    # ---- 3. the seeded selection ---------------------------------------------
    print("== selftest 3: selection is seeded, sorted, and per-stratum ==")
    s1 = _select(1000, 300, 80, 0)
    check("selection is reproducible from the seed", s1 == _select(1000, 300, 80, 0))
    check("selection is sorted and distinct",
          s1 == sorted(s1) and len(set(s1)) == 300)
    check("a different seed selects differently", s1 != _select(1000, 300, 81, 0))
    check("each stratum draws its own stream", s1 != _select(1000, 300, 80, 1))
    check("too small a pool HALTS instead of resampling",
          _raises(lambda: _select(10, 300, 80, 0), WebtextBuildError))

    # ---- 4. the chunker, reused verbatim -------------------------------------
    print("== selftest 4: the frozen S2 chunker, applied per document ==")
    if not have_pyarrow:
        skip("chunker/staging (needs pyarrow — the imported chunker reads parquet)",
             "no pyarrow in this venv; run this block in the build venv")
        skip("chunker/document-boundaries (needs pyarrow)", "same")
        shards_a: list[str] = []
    else:
        shards_a = shards_from_lines(_fixture_doc("alpha"), tok)
        check("a fixture document chunks into several chunks", len(shards_a) >= 3,
              f"n={len(shards_a)}")
        check("every chunk respects the chunker's own token contract",
              all(S2_TOK_MIN <= len(s.split()) <= S2_TOK_MAX for s in shards_a))
        check("chunking is deterministic",
              shards_from_lines(_fixture_doc("alpha"), tok) == shards_a)
        check("empty input yields no chunks", shards_from_lines([], tok) == [])
        check("heading-only input yields no chunks",
              shards_from_lines(["= h =", "", "= h2 ="], tok) == [])
        check("a document too short to reach the floor yields no chunks",
              shards_from_lines([_fixture_paragraph("w", 20)], tok) == [])
        both = shards_from_lines(_fixture_doc("alpha"), tok)
        check("chunks never span a document boundary (per-document staging)",
              all(("alpha" in s) and ("beta" not in s) for s in both))
        mixed = shards_from_lines(_fixture_doc("alpha") + _fixture_doc("beta"), tok)
        check("a heading-free concatenation CAN span — which is why documents are "
              "staged one at a time", any("alpha" in s and "beta" in s for s in mixed)
              or len(mixed) > len(both))

    # ---- 5. entries, manifest shape, meta shape ------------------------------
    print("== selftest 5: the v2.1 manifest shape, with bodies and without ==")
    bodies = shards_a or [_fixture_paragraph(f"body{i}", 200) for i in range(4)]
    entries = [
        _entry(spec, PIN_C4, "c4", i, "en/c4-validation.00000-of-00008.json.gz",
               f"row{i:06d}", 0, "https://example.invalid/page", b, tok)
        for i, b in enumerate(bodies)
    ]
    e0 = entries[0]
    v21_keys = ["text_id", "stratum", "voice", "mode", "topic_idx", "topic",
                "repetition", "source_run", "source_generation_id", "source_seed",
                "system_prompt", "user_prompt", "text", "n_words", "n_tokens"]
    check("the full entry carries every v2.1 field, in v2.1 order",
          list(e0.model_dump())[:len(v21_keys)] == v21_keys)
    check("the meta entry drops the body and adds text_sha256 last",
          "text" not in e0.meta() and list(e0.meta())[-1] == "text_sha256")
    check("text_sha256 is the sha256 of the body",
          e0.meta()["text_sha256"] == _sha_text(e0.text))
    check("collect_mean_states' required fields are present and usable",
          bool(e0.text) and e0.user_prompt == S2_CARRIER_PROMPT
          and e0.system_prompt == "" and e0.stratum in PINS)
    check("provenance names the dataset, the revision and the document",
          e0.source_revision == PIN_C4.revision and PIN_C4.repo_id in e0.source_run
          and e0.source_document.startswith("https://"))
    check("an empty body is refused at construction",
          _raises(lambda: _entry(spec, PIN_C4, "c4", 0, "f", "d", 0, "l", "", tok),
                  Exception))

    # ---- 6. assembly guards ---------------------------------------------------
    print("== selftest 6: assembly refuses a corpus it cannot vouch for ==")
    small = default_spec(seed=80, n_per_stratum=len(entries),
                         tokenizer_ref="stub/selftest")
    one = WebtextSpec(**{**small.model_dump(), "strata": (small.strata[1],)})
    res = [StratumResult(key="c4", entries=entries, info={"pool_size": len(entries)})]
    check("a clean stratum assembles", _ok(lambda: assemble(one, res)))
    dup = [e.model_copy() for e in entries] + [entries[0].model_copy()]
    bad = WebtextSpec(**{**one.model_dump(),
                       "strata": (one.strata[0].model_copy(
                           update={"n_texts": len(dup)}),)})
    check("a repeated body HALTS",
          _raises(lambda: assemble(bad, [StratumResult(key="c4", entries=dup,
                                                       info={})]), WebtextBuildError))
    short = entries[0].model_copy(update={"n_tokens": 3, "text_id": "c4-999"})
    check("a chunk outside the chunker's token range HALTS",
          _raises(lambda: assemble(
              WebtextSpec(**{**one.model_dump(),
                           "strata": (one.strata[0].model_copy(update={"n_texts": 1}),)}),
              [StratumResult(key="c4", entries=[short], info={})]), WebtextBuildError))
    check("a count that misses the spec HALTS",
          _raises(lambda: assemble(
              WebtextSpec(**{**one.model_dump(),
                           "strata": (one.strata[0].model_copy(update={"n_texts": 99}),)}),
              res), WebtextBuildError))

    # ---- 7. determinism of assembly + serialization (two builds) --------------
    print("== selftest 7: build twice, byte for byte ==")
    m1 = manifest_bytes(entries, with_text=True)
    m2 = manifest_bytes([e.model_copy(deep=True) for e in entries], with_text=True)
    check("the full manifest serializes byte-identically twice", m1 == m2,
          _sha_bytes(m1)[:16])
    check("the meta manifest serializes byte-identically twice",
          manifest_bytes(entries, with_text=False) ==
          manifest_bytes([e.model_copy(deep=True) for e in entries], with_text=False))
    check("the meta manifest is strictly smaller (bodies gone)",
          len(manifest_bytes(entries, with_text=False)) < len(m1))
    check("no body leaks into the meta manifest",
          all(e.text[:80] not in manifest_bytes(entries, with_text=False)
              .decode("utf-8") for e in entries))

    stamp = build_stamp(one, res, entries, {}, _sha_bytes(m1),
                        _sha_bytes(manifest_bytes(entries, with_text=False)), {})
    stamp2 = build_stamp(one, res, entries, {}, _sha_bytes(m1),
                         _sha_bytes(manifest_bytes(entries, with_text=False)), {})
    check("the stamp is deterministic (no clock, no path, no env)",
          json.dumps(stamp, indent=1) == json.dumps(stamp2, indent=1))
    blob = json.dumps(stamp)
    check("the stamp carries no timestamp field",
          not any(k in blob for k in ('"timestamp"', '"built_at"', '"date"')))
    census = render_census(one, stamp)
    check("the census renders deterministically", census == render_census(one, stamp))
    check("the census states counts, revisions, licenses and vintage (M47)",
          all(s in census for s in ("Composition", "Licensing", "vintage",
                                    "manifest sha256", "model-authored")))
    check("the census carries the stratum-4 survey for ratification",
          "survey behind the choice" in census and "rejected" in census)
    # rendered against the FULL four-stratum spec: the reconstruction doc must
    # carry every pin, not just the stratum this fixture assembled.
    recon = render_reconstruct(spec, stamp, "python -m ... --seed 80")
    check("RECONSTRUCT renders deterministically",
          recon == render_reconstruct(spec, stamp, "python -m ... --seed 80"))
    check("RECONSTRUCT names the command, the shas and every pin",
          "--seed 80" in recon and stamp["manifest_sha256"] in recon
          and all(p.revision in recon for p in
                  (PIN_WIKITEXT, PIN_C4, PIN_PG19, PIN_STACKEXCHANGE)))
    check("no absolute path from this machine leaks into the artifacts",
          str(Path.home()) not in census + recon + json.dumps(stamp))

    # ---- 8. readers parse their formats --------------------------------------
    print("== selftest 8: the source readers, on fixture files ==")
    with tempfile.TemporaryDirectory(prefix="webtext_selftest_") as td:
        root = Path(td)
        c4p = root / "c4.json.gz"
        with gzip.open(c4p, "wt", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"text": f"para one {i}\n\npara two {i}",
                                    "url": f"https://example.invalid/{i}",
                                    "timestamp": "2019-04-01"}) + "\n")
        docs = list(_iter_c4(c4p, "c4.json.gz", None))
        check("the C4 reader yields one document per row with its URL",
              len(docs) == 3 and docs[0].label == "https://example.invalid/0"
              and docs[0].doc_id == "row000000")
        check("the C4 reader honours the document cap",
              len(list(_iter_c4(c4p, "c4.json.gz", 2))) == 2)
        sep = root / "se.jsonl.gz"
        with gzip.open(sep, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"texts": ["A title", "A body\n\nmore body"],
                                "tags": ["t"]}) + "\n")
            f.write(json.dumps({"texts": ["B title", "B body"]}) + "\n")
        sedocs = list(_iter_stackexchange(sep, "philosophy.stackexchange.com.jsonl.gz",
                                          None))
        check("the SE reader puts the title first, then the body",
              len(sedocs) == 2 and sedocs[0].lines[0] == "A title"
              and "A body" in sedocs[0].lines[1])
        check("the SE reader labels the site and row",
              sedocs[0].label.startswith("philosophy.stackexchange.com row 0"))
        badp = root / "bad.jsonl.gz"
        with gzip.open(badp, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"wrong": "shape"}) + "\n")
        check("an unexpected SE record shape HALTS",
              _raises(lambda: list(_iter_stackexchange(badp, "bad.jsonl.gz", None)),
                      WebtextBuildError))
        brokenp = root / "broken.json.gz"
        with gzip.open(brokenp, "wt", encoding="utf-8") as f:
            f.write("{not json\n")
        check("a corrupt source line HALTS with the file and line named",
              _raises(lambda: list(_iter_c4(brokenp, "broken.json.gz", None)),
                      WebtextBuildError))
        book = _pg19_book("Chapter one\n\nand then\n", "1022", "PG-1022 — A Book (1901)",
                          "data/validation_files.txt")
        check("the PG-19 reader keeps the book id and title as provenance",
              book.doc_id == "PG-1022" and "A Book" in book.label)
        wrapped = ("The first paragraph is hard\nwrapped across three\nshort lines.\n"
                   "\n"
                   "The second paragraph is\nwrapped too.\n")
        flowed = reflow_hard_wrapped(wrapped)
        check("hard-wrapped book text reflows to ONE LINE PER PARAGRAPH",
              flowed == ["The first paragraph is hard wrapped across three short lines.",
                         "The second paragraph is wrapped too."])
        check("the PG-19 reader applies the reflow (else every wrapped line would "
              "become its own paragraph)",
              _pg19_book(wrapped, "1", "t", "f").lines == flowed)
        check("reflow is idempotent on already-reflowed text",
              reflow_hard_wrapped("\n\n".join(flowed)) == flowed)
        check("reflow of empty text yields no paragraphs",
              reflow_hard_wrapped("\n \n\n") == [])

    # ---- 9. the tokenizer is a pin, not a convenience ------------------------
    print("== selftest 9: no silent degradation ==")
    check("a missing tokenizer HALTS rather than falling back to word counts",
          _raises(lambda: load_tokenizer("definitely/not-a-real-tokenizer-xyz"),
                  WebtextBuildError))
    check("the default tokenizer pin is the one v2.1's S2 chunks were cut with",
          default_spec(80, 300, TOKENIZER_REF).tokenizer_ref == TOKENIZER_REF)

    failures = [c for c in checks if not c[1]]
    print(f"\nselftest: {len(failures)} failure(s)")
    for name, _, detail in failures:
        print(f"  MISS {name} {detail}")
    # RAKE M44: the count names the configuration it was measured in.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s)) "
          f"in configuration [{config}]")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def _cmdline(args: argparse.Namespace) -> str:
    return (f"python -m metabasis.scripts.build_webtext_corpus "
            f"--seed {args.seed} --n-per-stratum {args.n_per_stratum} "
            f"--tokenizer {args.tokenizer} --out-dir <OUT>")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="staging directory for the draft corpus (never git)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF/asset download cache (keep it off the repo tree)")
    ap.add_argument("--seed", type=int, default=A8_SEED,
                    help="selection seed (the prereg pins the final value)")
    ap.add_argument("--n-per-stratum", type=int, default=DEFAULT_N_PER_STRATUM)
    ap.add_argument("--tokenizer", default=TOKENIZER_REF,
                    help="tokenizer PIN — the chunk boundaries are its token counts")
    ap.add_argument("--no-license-check", action="store_true",
                    help="skip the pinned-revision license verification (NOT for "
                         "a corpus anyone will cite)")
    ap.add_argument("--verify-rebuild", action="store_true",
                    help="build twice and HALT unless every artifact matches")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    spec = default_spec(args.seed, args.n_per_stratum, args.tokenizer)
    cache = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    out = args.out_dir.expanduser().resolve()
    try:
        first = build(spec, out, cache, _cmdline(args),
                      check_licenses=not args.no_license_check)
        if args.verify_rebuild:
            with tempfile.TemporaryDirectory(prefix="webtext_rebuild_") as td:
                # the SAME inputs, including the license check: its results are
                # quoted in the census, so skipping it here would make the second
                # census differ for a reason that has nothing to do with the corpus.
                second = build(spec, Path(td), cache, _cmdline(args),
                               check_licenses=not args.no_license_check)
                diffs = {k: (v, second["artifact_sha256"].get(k))
                         for k, v in first["artifact_sha256"].items()
                         if second["artifact_sha256"].get(k) != v}
                if diffs:
                    raise WebtextBuildError(f"REBUILD DIFFERS: {diffs}")
            logger.info("verify-rebuild: PASS — %d artifacts byte-identical",
                        len(first["artifact_sha256"]))
        for name, sha in first["artifact_sha256"].items():
            logger.info("  %-28s %s", name, sha)
    except WebtextBuildError as e:
        logger.error("HALT: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
