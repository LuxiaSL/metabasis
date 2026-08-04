"""webtext-v3 — the freeze-time deterministic draws (PREREG §2 / §6-I1 / §9 / §13).

Three artifacts, every one a deterministic function of frozen inputs, computed
once at freeze and sha-recorded in the freeze act (§13):

  splits.json        the realized train/test membership — holdout 280/1200 by
                     count (70 per stratum), per-stratum, GROUPED, seed 80.
  halves.json        the §6-I1 half-split — two disjoint, exhaustive 600-text
                     halves (150 per stratum), same grouping keys, seed 80 on a
                     NAMED substream `halfsplit-v3`; each half then carries its
                     own internal §2-rule train/test split.
  ablation_ids.json  the §9 authored-stratum ablation sample — n=100, seed 80,
                     uniform over v2.1's S1/S3 entries in text_id order.

plus the §2 census-at-freeze addendum: the enumeration, by `text_sha256`, of the
wikitext chunks byte-identical to v2.1 S2 entries (the ledgered figure is 59; a
re-derivation that disagrees HALTS rather than adopting either number).

THE BINDING RULES, restated from the prereg because this file implements them:

  Grouping keys (§2).  c4 / stackexchange = the source row · pg19 = the book ·
  wikitext = the individual chunk (the NAMED LIMITATION: the v2.1 S2 pool
  carries no article identity, so wikitext holdout chunks are not
  article-independent — disclosed, not papered over). Grouping is binding: a
  group never straddles train and test.

  The v2.1-overlap rule (§2).  The wikitext chunks byte-identical to v2.1 S2
  entries are assigned to the TRAIN side BY RULE — no held-out, race-scored or
  gate quantity is ever computed on a text a prior result touched. Implemented
  generically: any group containing an overlapping text is INELIGIBLE for test
  membership anywhere — the main split and each half's internal split alike.

  Seed 80 throughout (§2), on named substreams so no draw can move another.

Inputs are READ-ONLY: the desk-side full manifests (bodies present — never git).

Run (repo root, the no-torch venv; --out-dir is desk-side staging, never git):
  python -m metabasis.scripts.derive_webtext_splits \\
      --v3-manifest staging/webtext-v3-draft/corpus_manifest.json \\
      --v21-manifest staging/corpus-v21-manifest-full-DESKONLY.json \\
      --out-dir staging/webtext-v3-draft \\
      --census staging/webtext-v3-draft/COMPOSITION-CENSUS.md
Selftest (rake M44; no manifests, no network, no torch needed):
  python -m metabasis.scripts.derive_webtext_splits --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("derive_webtext_splits")

# ---------------------------------------------------------------- the pins
CORPUS_NAME = "webtext-v3"
PREREG_REF = ("docs/planning/PREREG-webtext-v3-DRAFT-2026-08-03.md (r2) "
              "§2 split · §6-I1 halving · §9 ablation · §13 freeze mechanics")
PREREG_TAG = "freeze/webtext-v3"
# The date the draws were derived. A CONSTANT, never a clock read: every
# artifact here must be byte-identical across derivations and across machines.
DERIVATION_DATE = "2026-08-03"
SEED = 80
HOLDOUT_TOTAL = 280
CORPUS_TOTAL = 1200
N_PER_STRATUM = 300
HOLDOUT_PER_STRATUM = 70
ABLATION_N = 100
LEDGERED_OVERLAP = 59          # the figure of record; a disagreement HALTS
V3_MANIFEST_SHA = "b85f4d169ed0cb882a5690e3509308a5807e056b6a491f908014aee8e7b6085f"

StratumKey = Literal["wikitext", "c4", "pg19", "stackexchange"]
STRATA: tuple[StratumKey, ...] = ("wikitext", "c4", "pg19", "stackexchange")
ABLATION_STRATA: tuple[str, ...] = ("S1", "S3")

# Stream names. Each is a NAMED substream of seed 80 (see `named_stream`): the
# name alone fixes the spawn key's leading word, so the half-draw cannot be
# perturbed by a change to the split draw, or vice versa.
STREAM_SPLIT = "split-v3"
STREAM_HALVES = "halfsplit-v3"
STREAM_HALF_INTERNAL = {"half_a": "split-v3-half-a", "half_b": "split-v3-half-b"}
STREAM_ABLATION = "ablation-v3"

STREAM_CONSTRUCTION = (
    "numpy.random.default_rng(SeedSequence(entropy=80, "
    "spawn_key=(stream_word, ordinal))) where stream_word = "
    "int.from_bytes(sha256(stream_name.encode('utf-8')).digest()[:8], 'big') "
    "and ordinal = the stratum's index in "
    "('wikitext','c4','pg19','stackexchange') (0 where a draw has no stratum). "
    "The stream NAME alone determines the leading spawn-key word, so "
    "'halfsplit-v3' is independent of 'split-v3' by construction — neither "
    "draw can move the other, and no two named draws collide by accident."
)

QUOTA_RULE = (
    "Exact-quota grouped draw. (1) The eligible groups of the stratum are put "
    "in canonical order (sorted by grouping key). (2) The named stream draws a "
    "permutation of that order. (3) Greedy pass in permutation order: a group "
    "is taken iff its size fits within the remaining quota; the pass stops the "
    "moment the quota is met exactly. (4) If a full pass leaves a residual r>0 "
    "(every unselected group is larger than r — a size-granularity residual, "
    "not a shortage), one deterministic REPAIR SWAP runs: over unselected "
    "groups u in permutation order, then selected groups s in permutation "
    "order, take the first pair with size(u) - size(s) == r and swap them, "
    "which closes the residual exactly. (5) If no such pair exists the "
    "derivation HALTS — the quota is never silently missed."
)

OVERLAP_RULE = (
    "PREREG §2, the v2.1-overlap rule: every webtext-v3 text whose body "
    "sha256 equals that of a v2.1 S2 entry is TRAIN-side by rule. Implemented "
    "as ineligibility: a grouping key containing any such text is removed from "
    "the holdout candidate pool before the draw — in the main split and in "
    "each half's internal split alike. No held-out, race-scored or gate "
    "quantity is ever computed on a text a prior result touched."
)


class SplitDerivationError(RuntimeError):
    """A HALT: something the derivation refuses to guess about.

    Rake M45 rule (c): an exception, never sys.exit, so an all-module selftest
    sweep survives it and still prints its terminal TOTAL line.
    """


# ---------------------------------------------------------------- typed inputs
class ManifestEntry(BaseModel):
    """One row of a full (bodies-present) corpus manifest, v2.1 shape."""

    model_config = {"extra": "ignore"}

    text_id: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    source_run: str = ""
    source_generation_id: str = ""
    text: str = Field(min_length=1)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class CorpusText(BaseModel, frozen=True):
    """A corpus text reduced to exactly what a draw needs (no bodies travel)."""

    text_id: str
    stratum: str
    group_key: str
    text_sha256: str = Field(min_length=64, max_length=64)


class Group(BaseModel, frozen=True):
    """A grouping unit: the atom of every draw. Groups never straddle sides."""

    key: str
    stratum: str
    text_ids: tuple[str, ...] = Field(min_length=1)
    eligible_for_test: bool = True

    @property
    def size(self) -> int:
        return len(self.text_ids)


class QuotaFill(BaseModel, frozen=True):
    """The realized result of one exact-quota grouped draw."""

    quota: int
    selected_keys: tuple[str, ...]
    text_ids: tuple[str, ...]
    method: Literal["greedy", "greedy+swap"]
    residual_repaired: int = 0
    n_candidate_groups: int
    n_candidate_texts: int

    @model_validator(mode="after")
    def _quota_met(self) -> "QuotaFill":
        if len(self.text_ids) != self.quota:
            raise ValueError(f"quota {self.quota} != {len(self.text_ids)} texts drawn")
        return self


# ---------------------------------------------------------------- helpers
def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stream_word(stream_name: str) -> int:
    """The leading spawn-key word of a NAMED substream (see STREAM_CONSTRUCTION)."""
    if not stream_name:
        raise SplitDerivationError("a stream must be named")
    return int.from_bytes(hashlib.sha256(stream_name.encode("utf-8")).digest()[:8],
                          "big")


def named_stream(stream_name: str, ordinal: int, seed: int = SEED) -> np.random.Generator:
    """seed 80, on a substream the NAME picks out. Independent by construction."""
    if ordinal < 0:
        raise SplitDerivationError(f"stream ordinal must be >= 0, got {ordinal}")
    return np.random.default_rng(
        np.random.SeedSequence(entropy=seed,
                               spawn_key=(stream_word(stream_name), ordinal)))


def stratum_ordinal(stratum: str) -> int:
    if stratum not in STRATA:
        raise SplitDerivationError(f"unknown stratum {stratum!r}; expected {STRATA}")
    return STRATA.index(stratum)


def grouping_key(entry: ManifestEntry) -> str:
    """The PREREG §2 grouping key of one webtext-v3 text.

    wikitext            the individual chunk — its text_id (the named limitation)
    c4, stackexchange   the source row  — "<source_run>::<row id>"
    pg19                the book        — "<source_run>::<PG id>"

    The source-document id is the part of `source_generation_id` before
    ":chunk"; `source_run` (repo@revision:file) is carried too, so rows from
    different files can never collide on a bare row number.
    """
    if entry.stratum == "wikitext":
        return entry.text_id
    if entry.stratum not in STRATA:
        raise SplitDerivationError(
            f"{entry.text_id}: unknown stratum {entry.stratum!r}")
    gen = entry.source_generation_id
    if ":chunk" not in gen:
        raise SplitDerivationError(
            f"{entry.text_id}: source_generation_id {gen!r} carries no ':chunk' "
            "marker — the source-document id cannot be recovered, so the "
            "grouping key would be a guess")
    doc_id = gen.rsplit(":chunk", 1)[0]
    if not doc_id or not entry.source_run:
        raise SplitDerivationError(
            f"{entry.text_id}: empty source document id or source_run")
    return f"{entry.source_run}::{doc_id}"


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Read a full manifest READ-ONLY. Shape problems HALT."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SplitDerivationError(f"manifest not found: {path}") from e
    except json.JSONDecodeError as e:
        raise SplitDerivationError(f"manifest is not JSON: {path} ({e})") from e
    if not isinstance(blob, dict) or "entries" not in blob:
        raise SplitDerivationError(f"manifest {path} has no 'entries' key")
    rows = blob["entries"]
    if not isinstance(rows, list) or not rows:
        raise SplitDerivationError(f"manifest {path} carries no entries")
    try:
        entries = [ManifestEntry(**r) for r in rows]
    except Exception as e:  # noqa: BLE001 — pydantic detail is the message
        raise SplitDerivationError(f"manifest {path} row rejected: {e}") from e
    ids = [e.text_id for e in entries]
    if len(set(ids)) != len(ids):
        raise SplitDerivationError(f"manifest {path} repeats a text_id")
    return entries


def corpus_texts(entries: Sequence[ManifestEntry],
                 overlap_ids: frozenset[str] = frozenset()) -> list[CorpusText]:
    """Reduce manifest rows to draw inputs, in manifest order."""
    out: list[CorpusText] = []
    for e in entries:
        out.append(CorpusText(text_id=e.text_id, stratum=e.stratum,
                              group_key=grouping_key(e),
                              text_sha256=e.text_sha256))
    _ = overlap_ids  # eligibility is applied at group level, in build_groups
    return out


def build_groups(texts: Sequence[CorpusText], stratum: str,
                 ineligible_ids: frozenset[str]) -> tuple[Group, ...]:
    """The stratum's groups in CANONICAL order (sorted by grouping key).

    A group carrying any ineligible text is ineligible for test membership as a
    whole — groups are the atom, so partial eligibility is not a thing.
    """
    by_key: dict[str, list[str]] = {}
    for t in texts:
        if t.stratum != stratum:
            continue
        by_key.setdefault(t.group_key, []).append(t.text_id)
    if not by_key:
        raise SplitDerivationError(f"stratum {stratum!r} has no texts")
    groups = tuple(
        Group(key=k, stratum=stratum, text_ids=tuple(v),
              eligible_for_test=not any(tid in ineligible_ids for tid in v))
        for k, v in sorted(by_key.items()))
    if stratum == "wikitext" and any(g.size != 1 for g in groups):
        raise SplitDerivationError(
            "wikitext groups must be single chunks (PREREG §2's named "
            "limitation) — a multi-text wikitext group means the key changed")
    return groups


# ---------------------------------------------------------------- the draw
def exact_quota_draw(groups: Sequence[Group], quota: int,
                     rng: np.random.Generator) -> QuotaFill:
    """QUOTA_RULE, implemented. Eligible groups only; the quota is met exactly."""
    if quota < 0:
        raise SplitDerivationError(f"quota must be >= 0, got {quota}")
    cand = [g for g in groups if g.eligible_for_test]
    n_cand_texts = sum(g.size for g in cand)
    if n_cand_texts < quota:
        raise SplitDerivationError(
            f"quota {quota} exceeds the {n_cand_texts} eligible texts in "
            f"{len(cand)} eligible groups — the overlap rule left too little")
    perm = [int(i) for i in rng.permutation(len(cand))]
    pos = {g_i: p for p, g_i in enumerate(perm)}
    sel: list[int] = []
    total = 0
    for i in perm:
        if total + cand[i].size <= quota:
            sel.append(i)
            total += cand[i].size
            if total == quota:
                break
    method: Literal["greedy", "greedy+swap"] = "greedy"
    residual = quota - total
    if residual > 0:
        chosen = set(sel)
        pair: Optional[tuple[int, int]] = None
        for u in perm:                                   # unselected, perm order
            if u in chosen:
                continue
            for s in sorted(sel, key=lambda x: pos[x]):  # selected, perm order
                if cand[u].size - cand[s].size == residual:
                    pair = (u, s)
                    break
            if pair is not None:
                break
        if pair is None:
            raise SplitDerivationError(
                f"quota {quota} unreachable: greedy reached {total}, residual "
                f"{residual}, and no single group swap closes it — the group "
                "size spectrum cannot express this quota (never rounded away)")
        u, s = pair
        sel = [i for i in sel if i != s] + [u]
        total = quota
        method = "greedy+swap"
    keys = tuple(sorted(cand[i].key for i in sel))
    ids = tuple(sorted(tid for i in sel for tid in cand[i].text_ids))
    return QuotaFill(quota=quota, selected_keys=keys, text_ids=ids, method=method,
                     residual_repaired=residual, n_candidate_groups=len(cand),
                     n_candidate_texts=n_cand_texts)


def split_stratum(groups: Sequence[Group], stratum: str, quota: int,
                  stream: str) -> QuotaFill:
    return exact_quota_draw(groups, quota,
                            named_stream(stream, stratum_ordinal(stratum)))


def derive_split(texts: Sequence[CorpusText], ineligible_ids: frozenset[str],
                 stream: str, quota_per_stratum: dict[str, int],
                 strata: Sequence[str] = STRATA) -> dict[str, Any]:
    """The §2 split rule over a text set: per-stratum, grouped, exact quotas."""
    per_stratum: dict[str, Any] = {}
    test_ids: list[str] = []
    for st in strata:
        groups = build_groups(texts, st, ineligible_ids)
        n_texts = sum(g.size for g in groups)
        fill = split_stratum(groups, st, quota_per_stratum[st], stream)
        test_ids.extend(fill.text_ids)
        per_stratum[st] = {
            "n_texts": n_texts,
            "n_groups": len(groups),
            "n_groups_ineligible_for_test": sum(1 for g in groups
                                                if not g.eligible_for_test),
            "holdout_quota": fill.quota,
            "n_test": len(fill.text_ids),
            "n_train": n_texts - len(fill.text_ids),
            "fill_method": fill.method,
            "residual_repaired": fill.residual_repaired,
            "n_candidate_groups": fill.n_candidate_groups,
            "n_candidate_texts": fill.n_candidate_texts,
            "n_holdout_groups": len(fill.selected_keys),
            "holdout_group_keys": list(fill.selected_keys),
            "test_ids": list(fill.text_ids),
        }
    all_ids = sorted(t.text_id for t in texts)
    test_set = set(test_ids)
    train_ids = [i for i in all_ids if i not in test_set]
    return {
        "stream": stream,
        "seed": SEED,
        "n_texts": len(all_ids),
        "n_test": len(test_ids),
        "n_train": len(train_ids),
        "per_stratum": per_stratum,
        "test_ids": sorted(test_ids),
        "train_ids": train_ids,
    }


def derive_halving(texts: Sequence[CorpusText],
                   strata: Sequence[str] = STRATA) -> dict[str, list[str]]:
    """The §6-I1 halving: per-stratum, grouped, on the named `halfsplit-v3` stream.

    Half A is the exact-quota draw of n/2 texts per stratum; half B is the
    complement — disjoint and exhaustive by construction, groups intact.
    """
    half_a: list[str] = []
    half_b: list[str] = []
    per_stratum: dict[str, Any] = {}
    for st in strata:
        # The halving knows nothing about the overlap rule: it partitions the
        # corpus. Test-side eligibility bites inside each half's own split.
        groups = build_groups(texts, st, frozenset())
        n_texts = sum(g.size for g in groups)
        if n_texts % 2:
            raise SplitDerivationError(
                f"stratum {st} has {n_texts} texts — an odd stratum cannot be "
                "halved into equal parts by count")
        fill = exact_quota_draw(groups, n_texts // 2,
                                named_stream(STREAM_HALVES, stratum_ordinal(st)))
        a_keys = set(fill.selected_keys)
        b_ids = sorted(tid for g in groups if g.key not in a_keys
                       for tid in g.text_ids)
        half_a.extend(fill.text_ids)
        half_b.extend(b_ids)
        per_stratum[st] = {
            "n_texts": n_texts,
            "n_groups": len(groups),
            "half_a": {"n_texts": len(fill.text_ids),
                       "n_groups": len(fill.selected_keys)},
            "half_b": {"n_texts": len(b_ids),
                       "n_groups": len(groups) - len(fill.selected_keys)},
            "fill_method": fill.method,
            "residual_repaired": fill.residual_repaired,
        }
    return {"half_a": sorted(half_a), "half_b": sorted(half_b),
            "per_stratum": per_stratum}  # type: ignore[return-value]


# ---------------------------------------------------------------- the overlap
class OverlapRow(BaseModel, frozen=True):
    text_sha256: str = Field(min_length=64, max_length=64)
    v3_text_id: str
    v21_text_id: str


def enumerate_overlap(v3: Sequence[ManifestEntry], v21: Sequence[ManifestEntry],
                      v3_stratum: str = "wikitext",
                      v21_stratum: str = "S2") -> tuple[OverlapRow, ...]:
    """The §2 census cross-join: byte-identical bodies, by sha256, both ways.

    A repeated body on either side HALTS: the pairing would be ambiguous and a
    census that quotes an ambiguous pairing is worse than no census.
    """
    def index(entries: Sequence[ManifestEntry], stratum: str,
              label: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for e in entries:
            if e.stratum != stratum:
                continue
            sha = e.text_sha256
            if sha in out:
                raise SplitDerivationError(
                    f"{label} stratum {stratum} repeats a body sha256 "
                    f"({out[sha]} and {e.text_id}) — the cross-join would be "
                    "ambiguous")
            out[sha] = e.text_id
        if not out:
            raise SplitDerivationError(f"{label} has no {stratum} entries")
        return out

    a = index(v3, v3_stratum, "webtext-v3")
    b = index(v21, v21_stratum, "v2.1")
    return tuple(OverlapRow(text_sha256=s, v3_text_id=a[s], v21_text_id=b[s])
                 for s in sorted(set(a) & set(b)))


# ---------------------------------------------------------------- the ablation
def derive_ablation(v21: Sequence[ManifestEntry], n: int = ABLATION_N,
                    strata: Sequence[str] = ABLATION_STRATA) -> dict[str, Any]:
    """§9: n=100, seed 80, uniform over v2.1's S1/S3 entries IN TEXT_ID ORDER."""
    pool = sorted((e for e in v21 if e.stratum in strata),
                  key=lambda e: e.text_id)
    if len(pool) < n:
        raise SplitDerivationError(
            f"the S1/S3 pool holds {len(pool)} texts, need {n}")
    if any(e.stratum not in strata for e in pool):
        raise SplitDerivationError("pool contamination: a non-S1/S3 stratum leaked")
    rng = named_stream(STREAM_ABLATION, 0)
    idx = sorted(int(i) for i in rng.choice(len(pool), size=n, replace=False))
    picked = [pool[i] for i in idx]
    counts: dict[str, int] = {s: 0 for s in strata}
    for e in picked:
        counts[e.stratum] += 1
    return {
        "stream": STREAM_ABLATION,
        "seed": SEED,
        "n": n,
        "pool": {"n": len(pool), "strata": list(strata),
                 "order": "ascending text_id",
                 "counts": {s: sum(1 for e in pool if e.stratum == s)
                            for s in strata}},
        "counts_by_stratum": counts,
        "text_ids": [e.text_id for e in picked],
    }


# ---------------------------------------------------------------- artifacts
def artifact_bytes(payload: dict[str, Any]) -> bytes:
    """Serialization is part of the artifact: the corpus manifests' indent=1."""
    return json.dumps(payload, indent=1, ensure_ascii=False).encode("utf-8")


def _basis_block(input_shas: dict[str, str]) -> dict[str, Any]:
    return {
        "corpus": CORPUS_NAME,
        "prereg": PREREG_REF,
        "freeze_tag": PREREG_TAG,
        "derivation_date": DERIVATION_DATE,
        "derived_by": "metabasis/scripts/derive_webtext_splits.py",
        "inputs_sha256": dict(sorted(input_shas.items())),
        "expected_v3_manifest_sha256": V3_MANIFEST_SHA,
    }


def build_splits_artifact(split: dict[str, Any], overlap: Sequence[OverlapRow],
                          input_shas: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact": "splits.json — the realized webtext-v3 train/test membership",
        "basis": _basis_block(input_shas),
        "rule": {
            "statement": (f"holdout {HOLDOUT_TOTAL}/{CORPUS_TOTAL} by count "
                          f"({HOLDOUT_PER_STRATUM} per stratum), per-stratum, "
                          f"grouped, seed {SEED} (PREREG §2)"),
            "grouping_keys": {
                "wikitext": "the individual chunk (text_id) — PREREG §2's NAMED "
                            "LIMITATION: the v2.1 S2 pool carries no article "
                            "identity, so wikitext holdout chunks are not "
                            "article-independent",
                "c4": "the source row (source_run::rowNNNNNN)",
                "stackexchange": "the source row (source_run::rowNNNNNN)",
                "pg19": "the book (source_run::PG-NNNN)",
            },
            "quota_rule": QUOTA_RULE,
            "overlap_rule": OVERLAP_RULE,
            "seed": SEED,
            "stream": split["stream"],
            "stream_construction": STREAM_CONSTRUCTION,
        },
        "overlap": {
            "n_shared_with_v21_S2": len(overlap),
            "ledgered": LEDGERED_OVERLAP,
            "ineligible_text_ids": sorted(r.v3_text_id for r in overlap),
        },
        "counts": {
            "n_texts": split["n_texts"],
            "n_test": split["n_test"],
            "n_train": split["n_train"],
            "per_stratum": {st: {k: v for k, v in d.items()
                                 if k not in ("test_ids", "holdout_group_keys")}
                            for st, d in split["per_stratum"].items()},
        },
        "per_stratum": {st: {"holdout_group_keys": d["holdout_group_keys"],
                             "test_ids": d["test_ids"]}
                        for st, d in split["per_stratum"].items()},
        "test_ids": split["test_ids"],
        "train_ids": split["train_ids"],
    }


def build_halves_artifact(halving: dict[str, Any],
                          internal: dict[str, dict[str, Any]],
                          overlap: Sequence[OverlapRow],
                          input_shas: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact": "halves.json — the PREREG §6-I1 corpus half-split, with each "
                    "half's internal §2 train/test split",
        "basis": _basis_block(input_shas),
        "rule": {
            "statement": ("two disjoint, exhaustive halves of 600 texts (150 per "
                          "stratum), per-stratum, grouped by the §2 keys, "
                          f"seed {SEED}; each half then takes the §2 split rule "
                          "internally (PREREG §6-I1)"),
            "halving_stream": STREAM_HALVES,
            "internal_split_streams": dict(STREAM_HALF_INTERNAL),
            "stream_construction": STREAM_CONSTRUCTION,
            "independence": ("the halving draws on the NAMED substream "
                             f"'{STREAM_HALVES}' and the main split on "
                             f"'{STREAM_SPLIT}': different leading spawn-key "
                             "words, so the two draws are independent by "
                             "construction, not by luck"),
            "quota_rule": QUOTA_RULE,
            "overlap_rule": OVERLAP_RULE,
            "construction": ("half A = the exact-quota grouped draw of n/2 texts "
                             "per stratum on the halving stream; half B = the "
                             "complement — disjoint and exhaustive by "
                             "construction, with no grouping key split across "
                             "halves"),
            "internal_quota": ("each half's per-stratum holdout quota is the §2 "
                               "rate applied to the half: 70/300 of 150 = 35 per "
                               "stratum, 140 of 600 per half"),
        },
        "overlap": {
            "n_shared_with_v21_S2": len(overlap),
            "ineligible_text_ids": sorted(r.v3_text_id for r in overlap),
        },
        "counts": {
            "half_a": len(halving["half_a"]),
            "half_b": len(halving["half_b"]),
            "per_stratum": halving["per_stratum"],
            "internal": {h: {"n_test": internal[h]["n_test"],
                             "n_train": internal[h]["n_train"],
                             "per_stratum": {st: {k: v for k, v in d.items()
                                                  if k not in ("test_ids",
                                                               "holdout_group_keys")}
                                             for st, d in
                                             internal[h]["per_stratum"].items()}}
                         for h in ("half_a", "half_b")},
        },
        "half_a": {
            "text_ids": halving["half_a"],
            "internal_split": {
                "stream": internal["half_a"]["stream"],
                "test_ids": internal["half_a"]["test_ids"],
                "train_ids": internal["half_a"]["train_ids"],
            },
        },
        "half_b": {
            "text_ids": halving["half_b"],
            "internal_split": {
                "stream": internal["half_b"]["stream"],
                "test_ids": internal["half_b"]["test_ids"],
                "train_ids": internal["half_b"]["train_ids"],
            },
        },
    }


def build_ablation_artifact(ab: dict[str, Any],
                            input_shas: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact": "ablation_ids.json — the PREREG §9 authored-stratum ablation "
                    "sample (v2.1 S1/S3)",
        "basis": _basis_block(input_shas),
        "rule": {
            "statement": (f"n={ABLATION_N}, seed {SEED}, drawn uniformly without "
                          "replacement from v2.1's S1/S3 entries in text_id "
                          "order (PREREG §9)"),
            "source_manifest": "the desk-side full v2.1 manifest "
                               "(staging/corpus-v21-manifest-full-DESKONLY.json)",
            "excluded": "S2 — the human/web stratum is not authored text; §9 "
                        "measures the AUTHORED-stratum dose",
            "seed": SEED,
            "stream": ab["stream"],
            "stream_construction": STREAM_CONSTRUCTION,
        },
        "counts": {"n": ab["n"], "pool": ab["pool"],
                   "by_stratum": ab["counts_by_stratum"]},
        "text_ids": ab["text_ids"],
    }


def render_overlap_addendum(overlap: Sequence[OverlapRow],
                            n_v3_wikitext: int, n_v21_s2: int,
                            v3_manifest_sha: str, v21_manifest_sha: str) -> str:
    """The §2 census-at-freeze addendum, dated. Deterministic: no clock read."""
    lines = [
        ADDENDUM_MARKER,
        "",
        f"## Addendum {DERIVATION_DATE} — the v2.1 S2 overlap enumeration "
        "(PREREG §2, census at freeze)",
        "",
        "> PREREG §2: *\"The census at freeze additionally enumerates, by "
        "`text_sha256`, the 59 wikitext chunks byte-identical to v2.1 S2 "
        "entries.\"* The enumeration is below; it is a cross-join of the two "
        "desk-side full manifests on the sha256 of the text body.",
        "",
        f"- **webtext-v3 `wikitext` texts:** {n_v3_wikitext} (distinct bodies — "
        "the builder asserts no repeated body)",
        f"- **v2.1 `S2` texts:** {n_v21_s2} (distinct bodies)",
        f"- **byte-identical bodies shared:** **{len(overlap)}** "
        f"(ledgered figure: {LEDGERED_OVERLAP} — "
        f"{'MATCHES' if len(overlap) == LEDGERED_OVERLAP else 'DISAGREES'})",
        "- **why:** the `wikitext` stratum draws from the identical WikiText-103 "
        "validation shard pool the v2.1 S2 stratum drew from, with the same "
        "chunker — an expected, disclosed intersection, not a leak.",
        "",
        "**The rule these texts carry (binding):** " + OVERLAP_RULE,
        "",
        "Input shas the table was computed from:",
        "",
        f"- webtext-v3 full manifest `{v3_manifest_sha}`",
        f"- v2.1 full manifest `{v21_manifest_sha}`",
        "",
        "| # | `text_sha256` | webtext-v3 `text_id` | v2.1 `text_id` |",
        "|---:|---|---|---|",
    ]
    for i, row in enumerate(overlap, start=1):
        lines.append(f"| {i} | `{row.text_sha256}` | `{row.v3_text_id}` | "
                     f"`{row.v21_text_id}` |")
    lines.append("")
    lines.append(f"Derived by `metabasis/scripts/derive_webtext_splits.py` "
                 f"({DERIVATION_DATE}); the same enumeration is carried in "
                 "`splits.json` and `halves.json` as the ineligible-for-test id "
                 "list.")
    lines.append("")
    return "\n".join(lines)


ADDENDUM_MARKER = "<!-- overlap-addendum: derive_webtext_splits -->"


def append_addendum(census_path: Path, addendum: str) -> str:
    """Append the addendum IDEMPOTENTLY: re-running replaces, never duplicates."""
    if not census_path.exists():
        raise SplitDerivationError(f"census not found: {census_path}")
    body = census_path.read_text(encoding="utf-8")
    head = body.split(ADDENDUM_MARKER)[0].rstrip("\n")
    new = head + "\n\n" + addendum
    census_path.write_text(new, encoding="utf-8")
    return _sha_bytes(new.encode("utf-8"))


# ---------------------------------------------------------------- integrity
def integrity_checks(texts: Sequence[CorpusText], split: dict[str, Any],
                     halving: dict[str, Any], internal: dict[str, Any],
                     ablation: dict[str, Any], overlap: Sequence[OverlapRow],
                     v21: Sequence[ManifestEntry]) -> list[tuple[str, bool, str]]:
    """Every binding count and rule, re-checked on the realized artifacts."""
    out: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    by_id = {t.text_id: t for t in texts}
    ineligible = frozenset(r.v3_text_id for r in overlap)
    keys_by_id = {t.text_id: (t.stratum, t.group_key) for t in texts}

    ck("the corpus is the pinned size", len(texts) == CORPUS_TOTAL,
       f"{len(texts)} texts")
    ck("the overlap re-derivation matches the ledgered figure",
       len(overlap) == LEDGERED_OVERLAP,
       f"{len(overlap)} vs {LEDGERED_OVERLAP}")

    # ---- main split
    test = split["test_ids"]
    train = split["train_ids"]
    ck("the main holdout is the pinned total", len(test) == HOLDOUT_TOTAL,
       f"{len(test)}/{CORPUS_TOTAL}")
    ck("train + test partition the corpus exactly",
       len(set(test) & set(train)) == 0 and
       set(test) | set(train) == set(by_id) and
       len(test) + len(train) == CORPUS_TOTAL)
    for st in STRATA:
        n = len([i for i in test if by_id[i].stratum == st])
        ck(f"the {st} holdout is exactly {HOLDOUT_PER_STRATUM}",
           n == HOLDOUT_PER_STRATUM, f"{n}")
    ck("no grouping key straddles train and test (main split)",
       _no_straddle(test, train, keys_by_id))
    ck("zero overlap chunks in the main test side",
       not (set(test) & ineligible),
       f"{len(set(test) & ineligible)} leaked")

    # ---- halves
    a, b = halving["half_a"], halving["half_b"]
    ck("the halves are 600/600", len(a) == 600 and len(b) == 600,
       f"{len(a)}/{len(b)}")
    ck("the halves are disjoint", not (set(a) & set(b)))
    ck("the halves are exhaustive", set(a) | set(b) == set(by_id))
    for st in STRATA:
        na = len([i for i in a if by_id[i].stratum == st])
        nb = len([i for i in b if by_id[i].stratum == st])
        ck(f"the halves split {st} evenly", na == nb == N_PER_STRATUM // 2,
           f"{na}/{nb}")
    ck("no grouping key straddles the two halves", _no_straddle(a, b, keys_by_id))

    # ---- each half's internal split
    for h, ids in (("half_a", a), ("half_b", b)):
        d = internal[h]
        ht, htr = d["test_ids"], d["train_ids"]
        ck(f"{h}: the internal holdout is 140/600", len(ht) == 140, f"{len(ht)}")
        ck(f"{h}: the internal split partitions the half exactly",
           set(ht) | set(htr) == set(ids) and not (set(ht) & set(htr)))
        for st in STRATA:
            n = len([i for i in ht if by_id[i].stratum == st])
            ck(f"{h}: the {st} internal holdout is exactly 35", n == 35, f"{n}")
        ck(f"{h}: no grouping key straddles its internal train/test",
           _no_straddle(ht, htr, keys_by_id))
        ck(f"{h}: zero overlap chunks in its test side",
           not (set(ht) & ineligible), f"{len(set(ht) & ineligible)} leaked")

    # ---- ablation
    ab_ids = ablation["text_ids"]
    v21_stratum = {e.text_id: e.stratum for e in v21}
    ck("the ablation sample is n=100", len(ab_ids) == ABLATION_N, f"{len(ab_ids)}")
    ck("the ablation ids are distinct", len(set(ab_ids)) == len(ab_ids))
    ck("every ablation id exists in the v2.1 manifest",
       all(i in v21_stratum for i in ab_ids))
    ck("every ablation id is S1 or S3 — never S2",
       all(v21_stratum.get(i) in ABLATION_STRATA for i in ab_ids),
       f"strata={sorted({v21_stratum.get(i, '?') for i in ab_ids})}")
    ck("the ablation ids are in text_id order", ab_ids == sorted(ab_ids))
    return out


def _no_straddle(side_a: Sequence[str], side_b: Sequence[str],
                 keys_by_id: dict[str, tuple[str, str]]) -> bool:
    ka = {keys_by_id[i] for i in side_a}
    kb = {keys_by_id[i] for i in side_b}
    return not (ka & kb)


# ---------------------------------------------------------------- derivation
def derive_all(v3_path: Path, v21_path: Path) -> dict[str, Any]:
    """Everything, from the two read-only manifests. Pure: no writes, no clock."""
    v3 = load_manifest(v3_path)
    v21 = load_manifest(v21_path)
    input_shas = {v3_path.name: _sha_file(v3_path), v21_path.name: _sha_file(v21_path)}
    if input_shas[v3_path.name] != V3_MANIFEST_SHA:
        logger.warning("the v3 manifest sha is %s, the prereg's basis identity is "
                       "%s — derivation continues, but the artifacts will name "
                       "the sha they were actually computed from",
                       input_shas[v3_path.name], V3_MANIFEST_SHA)

    overlap = enumerate_overlap(v3, v21)
    if len(overlap) != LEDGERED_OVERLAP:
        raise SplitDerivationError(
            f"OVERLAP DISAGREEMENT: re-derivation counts {len(overlap)} shared "
            f"wikitext/S2 bodies, the ledger records {LEDGERED_OVERLAP}. HALT — "
            "neither number is adopted; the desk resolves this before freeze.")
    ineligible = frozenset(r.v3_text_id for r in overlap)

    texts = corpus_texts(v3)
    split = derive_split(texts, ineligible, STREAM_SPLIT,
                         {st: HOLDOUT_PER_STRATUM for st in STRATA})

    halving = derive_halving(texts)
    by_id = {t.text_id: t for t in texts}
    internal: dict[str, Any] = {}
    for h in ("half_a", "half_b"):
        sub = [by_id[i] for i in halving[h]]
        quotas = {}
        for st in STRATA:
            n_half = len([t for t in sub if t.stratum == st])
            q, rem = divmod(HOLDOUT_PER_STRATUM * n_half, N_PER_STRATUM)
            if rem:
                raise SplitDerivationError(
                    f"{h}/{st}: the §2 holdout rate {HOLDOUT_PER_STRATUM}/"
                    f"{N_PER_STRATUM} on {n_half} texts is not a whole number "
                    "— the internal quota would have to be rounded, which the "
                    "rule does not authorize")
            quotas[st] = q
        internal[h] = derive_split(sub, ineligible, STREAM_HALF_INTERNAL[h], quotas)

    ablation = derive_ablation(v21)
    checks = integrity_checks(texts, split, halving, internal, ablation, overlap, v21)

    artifacts = {
        "splits.json": build_splits_artifact(split, overlap, input_shas),
        "halves.json": build_halves_artifact(halving, internal, overlap, input_shas),
        "ablation_ids.json": build_ablation_artifact(ablation, input_shas),
    }
    n_wt = len([e for e in v3 if e.stratum == "wikitext"])
    n_s2 = len([e for e in v21 if e.stratum == "S2"])
    addendum = render_overlap_addendum(overlap, n_wt, n_s2,
                                       input_shas[v3_path.name],
                                       input_shas[v21_path.name])
    return {"artifacts": artifacts, "addendum": addendum, "checks": checks,
            "overlap": overlap, "input_shas": input_shas,
            "artifact_sha256": {k: _sha_bytes(artifact_bytes(v))
                                for k, v in artifacts.items()}}


# ---------------------------------------------------------------- selftest
def _fixture_manifest(n_wikitext: int = 12, n_c4: int = 12, n_pg19: int = 12,
                      n_se: int = 12) -> list[ManifestEntry]:
    """A synthetic corpus with the real grouping shapes, no data tree needed.

    Rake M44 rule (c): the fixture resolves nothing relative to cwd, so the
    selftest count means the same thing from a worktree as from main.
    """
    rows: list[ManifestEntry] = []
    for i in range(n_wikitext):
        rows.append(ManifestEntry(text_id=f"wikitext-{i:03d}", stratum="wikitext",
                                  source_run="repo@rev:wt.parquet",
                                  source_generation_id=f"shard{i:05d}:chunk000",
                                  text=f"wikitext body {i}"))
    for i in range(n_c4):
        rows.append(ManifestEntry(text_id=f"c4-{i:03d}", stratum="c4",
                                  source_run="repo@rev:c4.json.gz",
                                  source_generation_id=f"row{i // 2:06d}:chunk{i % 2:03d}",
                                  text=f"c4 body {i}"))
    for i in range(n_pg19):
        rows.append(ManifestEntry(text_id=f"pg19-{i:03d}", stratum="pg19",
                                  source_run=f"repo@rev:validation/PG-{i // 3:04d}",
                                  source_generation_id=f"PG-{i // 3:04d}:chunk{i % 3:03d}",
                                  text=f"pg19 body {i}"))
    for i in range(n_se):
        rows.append(ManifestEntry(text_id=f"stackexchange-{i:03d}",
                                  stratum="stackexchange",
                                  source_run="repo@rev:astronomy.jsonl.gz",
                                  source_generation_id=f"row{i:06d}:chunk000",
                                  text=f"se body {i}"))
    return rows


def _fixture_groups(sizes: Sequence[int], stratum: str = "c4",
                    eligible: Optional[Sequence[bool]] = None) -> tuple[Group, ...]:
    el = list(eligible) if eligible is not None else [True] * len(sizes)
    return tuple(Group(key=f"g{i:03d}", stratum=stratum,
                       text_ids=tuple(f"g{i:03d}-t{j}" for j in range(s)),
                       eligible_for_test=el[i])
                 for i, s in enumerate(sizes))


def _raises(fn: Callable[[], Any], exc: type[BaseException] = Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _ok(fn: Callable[[], Any]) -> bool:
    try:
        fn()
        return True
    except Exception:  # noqa: BLE001
        return False


def selftest(v3_path: Optional[Path] = None,
             v21_path: Optional[Path] = None) -> int:  # noqa: C901 — a checklist
    """Named-configuration selftest (rake M44): no network, no torch, no data tree.

    Blocks that need the desk-side manifests are NAMED SKIPS unless the paths
    are handed in, and the tail states the configuration the counts were
    measured in.
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

    have_v3 = v3_path is not None and v3_path.exists()
    have_v21 = v21_path is not None and v21_path.exists()
    config = (f"cwd={Path.cwd().name} python={sys.version.split()[0]} "
              f"numpy={np.__version__} "
              f"desk-manifests={'yes' if have_v3 and have_v21 else 'NO'} "
              f"network=not-used torch=not-imported")

    # ---- 1. the named streams -------------------------------------------------
    print("== selftest 1: named substreams, seed 80 ==")
    check("a stream name maps to a stable 64-bit word",
          stream_word("halfsplit-v3") == stream_word("halfsplit-v3")
          and 0 <= stream_word("halfsplit-v3") < 2 ** 64)
    check("different names give different words",
          len({stream_word(n) for n in (STREAM_SPLIT, STREAM_HALVES,
                                        STREAM_ABLATION,
                                        *STREAM_HALF_INTERNAL.values())}) == 5)
    check("an unnamed stream HALTS", _raises(lambda: stream_word(""),
                                             SplitDerivationError))
    r1 = named_stream(STREAM_SPLIT, 0).random(8).tolist()
    check("a named stream is reproducible",
          named_stream(STREAM_SPLIT, 0).random(8).tolist() == r1)
    check("the halving stream is INDEPENDENT of the split stream",
          named_stream(STREAM_HALVES, 0).random(8).tolist() != r1)
    check("each stratum ordinal is its own substream",
          named_stream(STREAM_SPLIT, 1).random(8).tolist() != r1)
    check("each half's internal split has its own stream",
          named_stream(STREAM_HALF_INTERNAL["half_a"], 0).random(4).tolist() !=
          named_stream(STREAM_HALF_INTERNAL["half_b"], 0).random(4).tolist())
    check("the seed is the prereg's 80",
          SEED == 80 and named_stream(STREAM_SPLIT, 0, seed=80).random(4).tolist()
          != named_stream(STREAM_SPLIT, 0, seed=81).random(4).tolist())
    check("a negative ordinal HALTS",
          _raises(lambda: named_stream(STREAM_SPLIT, -1), SplitDerivationError))

    # ---- 2. the grouping keys -------------------------------------------------
    print("== selftest 2: the PREREG §2 grouping keys ==")
    rows = _fixture_manifest()
    keys = {e.text_id: grouping_key(e) for e in rows}
    check("wikitext groups on the individual chunk (the named limitation)",
          keys["wikitext-000"] == "wikitext-000"
          and len({keys[e.text_id] for e in rows if e.stratum == "wikitext"}) == 12)
    check("c4 groups on the source row (two chunks of a row share a key)",
          keys["c4-000"] == keys["c4-001"] and keys["c4-000"] != keys["c4-002"])
    check("stackexchange groups on the source row",
          len({keys[e.text_id] for e in rows
               if e.stratum == "stackexchange"}) == 12)
    check("pg19 groups on the book (three chunks of a book share a key)",
          keys["pg19-000"] == keys["pg19-002"] and keys["pg19-000"] != keys["pg19-003"])
    check("the key carries source_run, so rows of different files cannot collide",
          "repo@rev:c4.json.gz" in keys["c4-000"])
    check("a source_generation_id with no ':chunk' marker HALTS rather than guessing",
          _raises(lambda: grouping_key(ManifestEntry(
              text_id="c4-999", stratum="c4", source_run="r",
              source_generation_id="row000001", text="x")), SplitDerivationError))
    check("an unknown stratum HALTS",
          _raises(lambda: grouping_key(ManifestEntry(
              text_id="x-1", stratum="mystery", source_run="r",
              source_generation_id="row0:chunk0", text="x")), SplitDerivationError))
    check("a multi-chunk wikitext group HALTS (the key would have changed)",
          _raises(lambda: build_groups(
              [CorpusText(text_id="wikitext-000", stratum="wikitext",
                          group_key="shared", text_sha256="0" * 64),
               CorpusText(text_id="wikitext-001", stratum="wikitext",
                          group_key="shared", text_sha256="1" * 64)],
              "wikitext", frozenset()), SplitDerivationError))

    # ---- 3. the exact-quota draw ---------------------------------------------
    print("== selftest 3: the exact-quota grouped draw ==")
    g_singletons = _fixture_groups([1] * 30)
    f1 = exact_quota_draw(g_singletons, 10, named_stream("t", 0))
    check("the quota is met exactly", len(f1.text_ids) == 10)
    check("the draw is reproducible from the stream",
          exact_quota_draw(g_singletons, 10, named_stream("t", 0)).text_ids
          == f1.text_ids)
    check("a different stream draws differently",
          exact_quota_draw(g_singletons, 10, named_stream("u", 0)).text_ids
          != f1.text_ids)
    check("the plain greedy needs no repair here", f1.method == "greedy")
    # sizes 2..8 with a quota greedy cannot reach without the repair swap
    g_books = _fixture_groups([2] + [5] * 6 + [7] * 6 + [8] * 6)
    f2 = exact_quota_draw(g_books, 33, named_stream("books", 0))
    check("a size-granularity residual is repaired, not rounded away",
          len(f2.text_ids) == 33, f"method={f2.method} r={f2.residual_repaired}")
    check("the repair is deterministic",
          exact_quota_draw(g_books, 33, named_stream("books", 0)).selected_keys
          == f2.selected_keys)
    check("groups are the atom: every selected group is whole",
          all(all(t in f2.text_ids for t in g.text_ids)
              for g in g_books if g.key in f2.selected_keys))
    check("an unreachable quota HALTS rather than missing quietly",
          _raises(lambda: exact_quota_draw(_fixture_groups([4] * 5), 3,
                                           named_stream("z", 0)),
                  SplitDerivationError))
    check("a quota bigger than the eligible pool HALTS",
          _raises(lambda: exact_quota_draw(g_singletons, 99, named_stream("z", 0)),
                  SplitDerivationError))
    check("ineligible groups are never drawn",
          set(exact_quota_draw(
              _fixture_groups([1] * 20, eligible=[i >= 10 for i in range(20)]),
              10, named_stream("e", 0)).selected_keys)
          == {f"g{i:03d}" for i in range(10, 20)})
    check("eligibility is counted, not assumed",
          exact_quota_draw(_fixture_groups([1] * 20,
                                           eligible=[i >= 10 for i in range(20)]),
                           10, named_stream("e", 0)).n_candidate_groups == 10)

    # ---- 4. the overlap enumeration ------------------------------------------
    print("== selftest 4: the v2.1 S2 overlap cross-join ==")
    v3f = [ManifestEntry(text_id="wikitext-000", stratum="wikitext",
                         source_run="r", source_generation_id="s:chunk0",
                         text="shared body"),
           ManifestEntry(text_id="wikitext-001", stratum="wikitext",
                         source_run="r", source_generation_id="s:chunk1",
                         text="v3 only")]
    v21f = [ManifestEntry(text_id="S2-wt-000", stratum="S2", text="shared body"),
            ManifestEntry(text_id="S2-wt-001", stratum="S2", text="v21 only"),
            ManifestEntry(text_id="S1-x", stratum="S1", text="v3 only")]
    ov = enumerate_overlap(v3f, v21f)
    check("the cross-join matches on the body sha256, both ways",
          len(ov) == 1 and ov[0].v3_text_id == "wikitext-000"
          and ov[0].v21_text_id == "S2-wt-000")
    check("only the S2 stratum is joined against (S1 bodies do not count)",
          all(r.v21_text_id.startswith("S2-") for r in ov))
    check("the rows carry the full 64-hex sha256", len(ov[0].text_sha256) == 64)
    check("a repeated body on either side HALTS (ambiguous pairing)",
          _raises(lambda: enumerate_overlap(
              v3f + [ManifestEntry(text_id="wikitext-002", stratum="wikitext",
                                   source_run="r", source_generation_id="s:chunk2",
                                   text="shared body")], v21f),
                  SplitDerivationError))
    check("an empty stratum HALTS", _raises(lambda: enumerate_overlap(v3f, v3f),
                                            SplitDerivationError))
    check("the ledgered figure is a constant this module HALTS against",
          LEDGERED_OVERLAP == 59)

    # ---- 5. the split rule on a synthetic corpus ------------------------------
    print("== selftest 5: the §2 split rule, end to end on a fixture ==")
    texts = corpus_texts(rows)
    inel = frozenset({"wikitext-000", "wikitext-001", "wikitext-002"})
    # quotas the fixture's group-size spectrum can express exactly (c4 rows hold
    # 2 chunks, pg19 books 3) — the repair swap has its own block above.
    fx_quota = {"wikitext": 3, "c4": 4, "pg19": 3, "stackexchange": 3}
    sp = derive_split(texts, inel, STREAM_SPLIT, fx_quota)
    check("the holdout hits the quota in every stratum",
          all(sp["per_stratum"][st]["n_test"] == fx_quota[st] for st in STRATA))
    check("train and test partition the fixture",
          set(sp["test_ids"]) | set(sp["train_ids"]) == {t.text_id for t in texts}
          and not (set(sp["test_ids"]) & set(sp["train_ids"])))
    check("ineligible texts are TRAIN side, by rule",
          inel.issubset(set(sp["train_ids"])) and not (inel & set(sp["test_ids"])))
    kb = {t.text_id: (t.stratum, t.group_key) for t in texts}
    check("no grouping key straddles train and test",
          _no_straddle(sp["test_ids"], sp["train_ids"], kb))
    check("the split is reproducible byte for byte",
          artifact_bytes(derive_split(texts, inel, STREAM_SPLIT, fx_quota))
          == artifact_bytes(sp))
    check("a different stream name gives a different split",
          derive_split(texts, inel, "some-other-stream",
                       fx_quota)["test_ids"] != sp["test_ids"])
    check("the eligible-group count is reported per stratum",
          sp["per_stratum"]["wikitext"]["n_groups_ineligible_for_test"] == 3)

    # ---- 6. the halving ------------------------------------------------------
    print("== selftest 6: the §6-I1 halving ==")
    hv = derive_halving(texts)
    check("the halves are equal by count",
          len(hv["half_a"]) == len(hv["half_b"]) == len(texts) // 2)
    check("the halves are disjoint", not (set(hv["half_a"]) & set(hv["half_b"])))
    check("the halves are exhaustive",
          set(hv["half_a"]) | set(hv["half_b"]) == {t.text_id for t in texts})
    check("no grouping key straddles the halves",
          _no_straddle(hv["half_a"], hv["half_b"], kb))
    check("the halving is reproducible", derive_halving(texts) == hv)
    check("the halving is NOT the split draw (independent named streams)",
          set(hv["half_a"]) != set(sp["test_ids"]) and
          named_stream(STREAM_HALVES, 0).random(4).tolist() !=
          named_stream(STREAM_SPLIT, 0).random(4).tolist())
    check("an odd stratum HALTS rather than rounding a half",
          _raises(lambda: derive_halving(
              [t for t in texts if not t.text_id.endswith("-011")]),
              SplitDerivationError))
    by_id = {t.text_id: t for t in texts}
    sub_a = [by_id[i] for i in hv["half_a"]]
    fx_iq = {"wikitext": 1, "c4": 2, "pg19": 3, "stackexchange": 1}
    isp = derive_split(sub_a, inel, STREAM_HALF_INTERNAL["half_a"], fx_iq)
    check("a half carries its own internal §2 split",
          len(isp["test_ids"]) == sum(fx_iq.values()) and
          set(isp["test_ids"]).issubset(set(hv["half_a"])))
    check("the overlap rule applies INSIDE a half too",
          not (inel & set(isp["test_ids"])))
    check("the internal split uses the half's own stream, not the main one",
          isp["stream"] == STREAM_HALF_INTERNAL["half_a"] != STREAM_SPLIT)

    # ---- 7. the §9 ablation --------------------------------------------------
    print("== selftest 7: the §9 ablation draw ==")
    v21_big = ([ManifestEntry(text_id=f"S1-{i:03d}", stratum="S1", text=f"a{i}")
                for i in range(40)] +
               [ManifestEntry(text_id=f"S2-{i:03d}", stratum="S2", text=f"b{i}")
                for i in range(40)] +
               [ManifestEntry(text_id=f"S3-{i:03d}", stratum="S3", text=f"c{i}")
                for i in range(40)])
    ab = derive_ablation(v21_big, n=20)
    check("the sample is the requested size", len(ab["text_ids"]) == 20)
    check("every id is S1 or S3, never S2",
          all(not i.startswith("S2-") for i in ab["text_ids"]))
    check("the pool is the S1/S3 union, in text_id order",
          ab["pool"]["n"] == 80 and ab["pool"]["order"] == "ascending text_id")
    check("the ids come back sorted", ab["text_ids"] == sorted(ab["text_ids"]))
    check("the draw is reproducible", derive_ablation(v21_big, n=20) == ab)
    check("the draw is uniform over the pool, not stratum-balanced by fiat",
          set(ab["counts_by_stratum"]) == {"S1", "S3"}
          and sum(ab["counts_by_stratum"].values()) == 20)
    check("a pool too small HALTS",
          _raises(lambda: derive_ablation(v21_big, n=999), SplitDerivationError))
    check("the ablation stream is its own", ab["stream"] == STREAM_ABLATION)

    # ---- 8. artifacts self-describe and serialize deterministically ----------
    print("== selftest 8: artifacts self-describe; derive twice, byte for byte ==")
    shas = {"corpus_manifest.json": "0" * 64, "v21.json": "1" * 64}
    art = build_splits_artifact(sp, ov, shas)
    check("the split artifact names rule, seed, stream and prereg",
          art["rule"]["seed"] == SEED and art["rule"]["stream"] == STREAM_SPLIT
          and PREREG_REF in art["basis"]["prereg"]
          and "grouping_keys" in art["rule"])
    check("the split artifact names the stream construction",
          "SeedSequence" in art["rule"]["stream_construction"])
    check("the split artifact carries the overlap rule verbatim",
          art["rule"]["overlap_rule"] == OVERLAP_RULE)
    hart = build_halves_artifact(hv, {"half_a": isp, "half_b": isp}, ov, shas)
    check("the halves artifact names the halving stream AND its independence",
          hart["rule"]["halving_stream"] == STREAM_HALVES
          and STREAM_SPLIT in hart["rule"]["independence"])
    aart = build_ablation_artifact(ab, shas)
    check("the ablation artifact names its source manifest and exclusion",
          "S2" in aart["rule"]["excluded"] and aart["counts"]["n"] == 20)
    check("every artifact serializes byte-identically twice",
          all(artifact_bytes(x) == artifact_bytes(x) for x in (art, hart, aart)))
    blob = json.dumps([art, hart, aart])
    check("no clock, no host, no absolute path leaks into an artifact",
          not any(k in blob for k in ('"timestamp"', '"built_at"', '"hostname"'))
          and str(Path.home()) not in blob)
    check("the derivation date is a constant, not a clock read",
          DERIVATION_DATE == art["basis"]["derivation_date"] == "2026-08-03")

    # ---- 9. the census addendum ----------------------------------------------
    print("== selftest 9: the census addendum is dated, tabular and idempotent ==")
    add = render_overlap_addendum(ov, 300, 160, "a" * 64, "b" * 64)
    check("the addendum renders deterministically",
          add == render_overlap_addendum(ov, 300, 160, "a" * 64, "b" * 64))
    check("the addendum is dated and cites the prereg clause",
          DERIVATION_DATE in add and "PREREG §2" in add)
    check("the addendum tables text_sha256 with BOTH text_ids",
          "| `wikitext-000` | `S2-wt-000` |" in add and ov[0].text_sha256 in add)
    check("the addendum states the count against the ledgered figure",
          str(LEDGERED_OVERLAP) in add and "ledgered figure" in add)
    check("the addendum carries a replaceable marker (idempotent append)",
          add.startswith(ADDENDUM_MARKER))
    check("a missing census HALTS",
          _raises(lambda: append_addendum(Path("/nonexistent/CENSUS.md"), add),
                  SplitDerivationError))

    # ---- 10. integrity checks catch what they claim to ------------------------
    print("== selftest 10: the integrity checks are not decorative ==")
    good = integrity_checks(texts, sp, hv, {"half_a": isp, "half_b": isp}, ab, ov,
                            v21_big)
    check("the integrity sweep runs over the fixture", len(good) > 10,
          f"{len(good)} checks")
    bad = dict(sp)
    bad["test_ids"] = sorted(set(sp["test_ids"]) | {"wikitext-000"})
    leaked = integrity_checks(texts, bad, hv, {"half_a": isp, "half_b": isp}, ab,
                              ov, v21_big)
    check("an overlap chunk on the test side is CAUGHT",
          any(not ok for n, ok, _ in leaked if "overlap chunks in the main" in n))
    bad2 = dict(ab)
    bad2["text_ids"] = sorted(ab["text_ids"][:-1] + ["S2-000"])
    caught = integrity_checks(texts, sp, hv, {"half_a": isp, "half_b": isp}, bad2,
                              ov, v21_big)
    check("an S2 id in the ablation sample is CAUGHT",
          any(not ok for n, ok, _ in caught if "never S2" in n))

    # ---- 11. the real manifests (named skip when they are not here) ----------
    print("== selftest 11: the desk-side manifests ==")
    if not (have_v3 and have_v21):
        skip("real-manifest derivation (needs the desk-side full manifests)",
             "gitignored staging/ is absent from this checkout — pass "
             "--v3-manifest/--v21-manifest to exercise it")
        skip("real-manifest overlap count", "same")
    else:
        assert v3_path is not None and v21_path is not None
        res = derive_all(v3_path, v21_path)
        check("the real derivation passes every integrity check",
              all(ok for _, ok, _ in res["checks"]),
              f"{sum(1 for _, ok, _ in res['checks'] if not ok)} miss(es)")
        check("the real overlap count is the ledgered figure",
              len(res["overlap"]) == LEDGERED_OVERLAP, f"{len(res['overlap'])}")
        res2 = derive_all(v3_path, v21_path)
        check("deriving twice is byte-identical, every artifact",
              res["artifact_sha256"] == res2["artifact_sha256"],
              " ".join(f"{k}={v[:12]}" for k, v in
                       sorted(res["artifact_sha256"].items())))
        check("the addendum is byte-identical too",
              res["addendum"] == res2["addendum"])

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
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v3-manifest", type=Path,
                    default=Path("staging/webtext-v3-draft/corpus_manifest.json"),
                    help="the desk-side webtext-v3 FULL manifest (read-only)")
    ap.add_argument("--v21-manifest", type=Path,
                    default=Path("staging/corpus-v21-manifest-full-DESKONLY.json"),
                    help="the desk-side v2.1 FULL manifest (read-only)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("staging/webtext-v3-draft"),
                    help="where the three draw artifacts land (never git)")
    ap.add_argument("--census", type=Path, default=None,
                    help="COMPOSITION-CENSUS.md to append the dated overlap "
                         "addendum to (idempotent: re-running replaces)")
    ap.add_argument("--dry-run", action="store_true",
                    help="derive and check, write nothing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.v3_manifest if args.v3_manifest.exists() else None,
                        args.v21_manifest if args.v21_manifest.exists() else None)

    try:
        v3p = args.v3_manifest.expanduser().resolve()
        v21p = args.v21_manifest.expanduser().resolve()
        first = derive_all(v3p, v21p)
        # SELFTEST, always on: derive twice in-process and refuse to write
        # anything unless every artifact is byte-identical.
        second = derive_all(v3p, v21p)
        diffs = {k: (v, second["artifact_sha256"].get(k))
                 for k, v in first["artifact_sha256"].items()
                 if second["artifact_sha256"].get(k) != v}
        if diffs or first["addendum"] != second["addendum"]:
            raise SplitDerivationError(f"DERIVATION IS NOT DETERMINISTIC: {diffs}")
        logger.info("derive-twice: PASS — %d artifacts + addendum byte-identical",
                    len(first["artifact_sha256"]))

        misses = [(n, d) for n, ok, d in first["checks"] if not ok]
        for name, ok, detail in first["checks"]:
            logger.info("%s %s %s", "PASS" if ok else "MISS", name, detail)
        if misses:
            raise SplitDerivationError(
                f"{len(misses)} integrity check(s) MISSED: {misses}")
        logger.info("integrity: PASS — %d checks", len(first["checks"]))

        if args.dry_run:
            logger.info("--dry-run: nothing written")
        else:
            out = args.out_dir.expanduser().resolve()
            out.mkdir(parents=True, exist_ok=True)
            for name, payload in first["artifacts"].items():
                (out / name).write_bytes(artifact_bytes(payload))
            if args.census is not None:
                census_sha = append_addendum(args.census.expanduser().resolve(),
                                             first["addendum"])
                logger.info("  %-28s %s", "COMPOSITION-CENSUS.md", census_sha)
        for name, sha in sorted(first["artifact_sha256"].items()):
            logger.info("  %-28s %s", name, sha)
        for name, sha in sorted(first["input_shas"].items()):
            logger.info("  input %-22s %s", name, sha)
    except SplitDerivationError as e:
        logger.error("HALT: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
