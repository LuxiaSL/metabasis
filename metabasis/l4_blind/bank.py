"""The text bank and the blind-deck build.

The bank is a JSONL the DECODE job produces, one row per banked
generation, keyed by coordinates:

    {"column": ..., "side": ..., "cell_id": ..., "generation_id": ...,
     "text": "..."}

Nothing else is read from it. It is the join between the sealed draw and
the page, and it is consumed exactly once — at deck build, a desk step —
after which the serving path only ever sees `BlindDeck`.

DECODE OF RECORD (must be what produced the bank, verbatim, per
L2-PROBE-INGREDIENTS-FULL6 `decode_provenance`):

    metabasis.text_decode.maybe_decode(
        tokenizer.decode(list(generated_ids), skip_special_tokens=True))

with the tokenizer from `AutoTokenizer.from_pretrained(model_path)` and no
extra kwargs. For the dsv2-lite columns this is the CORRECTED decode — the
adopted logs (`staging/reading-bleed/corrected-l1-dsv2/`), NOT the banked
byte-BPE text, which is why the corrected pass exists at all.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, Iterator

from pydantic import BaseModel, ConfigDict, Field

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.draw import DrawnPairs
from metabasis.l4_blind.models import BlindDeck, BlindPair
from metabasis.l4_blind.taxonomy import AXIS_TRAIT, sha256_file

#: A generation whose decoded text is shorter than this is not judgeable as
#: a 2AFC panel; the deck build HALTs rather than showing Luxia a stub.
MIN_WORDS: Final[int] = 5


class BankRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    column: str
    side: str
    cell_id: str
    generation_id: int = Field(ge=0)
    text: str


def bank_key(column: str, side: str, cell_id: str, generation_id: int) -> str:
    return f"{column}|{side}|{cell_id}|{generation_id:03d}"


def load_bank(path: Path) -> dict[str, str]:
    """Read the text bank into a coordinate->text map. Duplicates HALT."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = BankRow.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 — line number is the useful part
                raise ValueError(f"{path}:{n} is not a valid bank row: {exc}") from exc
            k = bank_key(row.column, row.side, row.cell_id, row.generation_id)
            if k in out and out[k] != row.text:
                raise ValueError(f"{path}:{n} conflicting text for {k}")
            out[k] = row.text
    if not out:
        raise ValueError(f"{path} is empty — no text to judge")
    return out


# ── self-identification scan (a REPORT, never a rewrite) ─────────────────

#: Vendor / family tokens a generation might emit about itself. Hits are
#: reported to the desk, never edited out: the L3 judge reads the same
#: bytes, and silently diverging the two would break the very agreement
#: number L4 exists to produce.
SELF_ID_TOKENS: Final[tuple[str, ...]] = (
    "deepseek", "mistral", "olmo", "allenai", "phi-3", "phi-4", "qwen",
    "alibaba", "llama", "meta ai", "pythia", "eleutherai", "gemma",
    "microsoft", "openai", "chatgpt", "gpt-4", "anthropic", "claude",
)


def scan_self_identification(pairs: list[BlindPair]) -> list[dict[str, object]]:
    """Which blind pairs contain a token that could name their own model."""
    hits: list[dict[str, object]] = []
    for p in pairs:
        for slot, text in (("text_1", p.text_1), ("text_2", p.text_2)):
            low = text.lower()
            found = sorted({t for t in SELF_ID_TOKENS if t in low})
            if found:
                hits.append({"pair_id": p.pair_id, "slot": slot, "tokens": found})
    return hits


# ── deck build ───────────────────────────────────────────────────────────


def build_deck(drawn: DrawnPairs, bank_path: Path) -> tuple[BlindDeck, list[dict[str, object]]]:
    """Join the sealed draw to the bank and emit the blind deck.

    This is the one function that touches both sides. Its output carries
    the sealed map's sha (so the desk can prove which map unblinds it) but
    none of its content.
    """
    bank = load_bank(bank_path)
    bank_sha = sha256_file(bank_path)
    draw_sha = sha256_json(drawn.model_dump(mode="json"))
    sealed_sha = draw_sha  # the sealed map file IS the serialised draw

    pairs: list[BlindPair] = []
    missing: list[str] = []
    for p in sorted(drawn.pairs, key=lambda x: x.ordinal_global):
        kd = bank_key(p.column, p.side, p.dose_cell_id, p.generation_id)
        kb = bank_key(p.column, p.side, p.baseline_cell_id, p.generation_id)
        td, tb = bank.get(kd), bank.get(kb)
        if td is None:
            missing.append(kd)
        if tb is None:
            missing.append(kb)
        if td is None or tb is None:
            continue
        for label, t in ((kd, td), (kb, tb)):
            if len(t.split()) < MIN_WORDS:
                raise ValueError(
                    f"{label} decoded to {len(t.split())} words (< {MIN_WORDS}); "
                    f"a stub panel is unjudgeable — the desk must rule on this pair"
                )
        t1, t2 = (td, tb) if p.dose_first else (tb, td)
        pairs.append(
            BlindPair(
                pair_id=p.pair_id,
                set_label=p.set_label,
                trait=AXIS_TRAIT[p.axis],
                text_1=t1,
                text_2=t2,
            )
        )

    if missing:
        raise ValueError(
            f"the text bank is missing {len(missing)} of the drawn generations; "
            f"refusing to build a partial deck. First few:\n  - "
            + "\n  - ".join(missing[:10])
        )

    totals: dict[str, int] = {}
    for p in pairs:
        totals[p.set_label] = totals.get(p.set_label, 0) + 1

    deck = BlindDeck(
        tool_version=TOOL_VERSION,
        grade=GRADE,
        status=STATUS,
        draw_recipe=drawn.draw_recipe,
        draw_sha256=draw_sha,
        sealed_map_sha256=sealed_sha,
        bank_sha256=bank_sha,
        set_labels=sorted(totals),
        set_totals=totals,
        pairs=pairs,
    )
    return deck, scan_self_identification(pairs)


def sha256_json(obj: object) -> str:
    """Canonical sha of a JSON-able object: sorted keys, no spurious space."""
    import hashlib

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── synthetic bank (selftests only) ──────────────────────────────────────

_LOREM = (
    "The question turns on what we mean by the term, and the answer is less "
    "obvious than it first appears. Consider the ordinary case: a reader "
    "encounters a claim, weighs it against what they already believe, and "
    "either revises or does not. Nothing in that process is mysterious, yet "
    "the outcome is rarely predictable in advance."
)


def synthetic_bank(drawn: DrawnPairs, out_path: Path) -> Path:
    """A deterministic placeholder bank for the selftests ONLY.

    It exists so the ergonomics, the resume path, the counterbalance and
    the leak grep can all be proven end-to-end before the real decoded text
    arrives. Every row is stamped SYNTHETIC in its own text so a synthetic
    bank can never be mistaken for the real one in a session log.
    """
    import hashlib

    rows: list[str] = []
    seen: set[str] = set()
    for p in drawn.pairs:
        # NB: the role ("dose" / "baseline") is deliberately NOT written into
        # the placeholder text. An earlier draft did, and the leak selftest
        # caught it — a synthetic panel that announces its own arm would make
        # the whole no-leak check vacuous.
        for cell_id in (p.dose_cell_id, p.baseline_cell_id):
            k = bank_key(p.column, p.side, cell_id, p.generation_id)
            if k in seen:
                continue
            seen.add(k)
            salt = hashlib.sha256(k.encode()).hexdigest()[:8]
            text = (
                f"[SYNTHETIC SELFTEST PANEL {salt}] {_LOREM} "
                f"A second paragraph follows so the panel has real height, and it "
                f"varies by {salt} so that no two panels in the deck are equal."
            )
            rows.append(
                json.dumps(
                    {
                        "column": p.column,
                        "side": p.side,
                        "cell_id": cell_id,
                        "generation_id": p.generation_id,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_path


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


_WS = re.compile(r"[ \t]+")


def normalise_for_display(text: str) -> str:
    """Collapse runs of spaces/tabs but keep paragraph structure intact.

    Display-only. The bank's bytes are what the desk joins and what L3
    reads; this never touches them.
    """
    return "\n".join(_WS.sub(" ", ln).rstrip() for ln in text.splitlines()).strip()
