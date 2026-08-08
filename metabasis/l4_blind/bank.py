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

#: A panel shorter than this is a stub, not an utterance; the deck build
#: HALTs rather than showing Luxia a fragment.
#:
#: The floor is in CHARACTERS, not words, and that is deliberate. An
#: earlier draft used `len(text.split()) >= 5` and HALTed the whole build
#: on a dsv2-lite panel of "2 words" that was in fact 431 characters of
#: fluent English with the spaces missing (see SPACE_DEGENERATE_RATIO).
#: Word count is not a measure of length on a column that emits
#: space-less token runs; characters are.
MIN_CHARS: Final[int] = 80

#: Below this ratio of WHITESPACE to characters, a panel's words are
#: running together. Whitespace, not the space character: `str.split()`
#: breaks on newlines too, so counting only " " marked a newline-separated
#: list as degenerate when its words were perfectly separated.
#:
#: This is REPORTED, never repaired. The L3 judge reads exactly these
#: bytes under the same frozen decode, so "fixing" the text here would
#: make the L4/L3 agreement number compare two different corpora — the one
#: thing the gold exists to prevent. The desk rules on whether these pairs
#: stay in the deck.
SPACE_DEGENERATE_RATIO: Final[float] = 0.08

#: THE DIRECT MEASURE, used wherever the real text is in hand: the
#: fraction of characters sitting inside whitespace-separated tokens
#: longer than this. Glued text produces enormous "words"
#: ("youmeanthowcantusllyteachsomeone" is one 32-character token); normal
#: English does not. Unlike a whitespace ratio it is not fooled by glued
#: lines that are newline-separated, and unlike chars/word it separates
#: the 424 banked panels cleanly.
GLUED_TOKEN_CHARS: Final[int] = 25
GLUED_FRACTION_MAX: Final[float] = 0.10

#: Below this, a panel is unusually short beside its ~2000-character
#: neighbours. Reported beside the degeneracy flag as a length-bleed cue
#: (the pre-statement already pre-stated a positive length bleed on the
#: language and formality axes).
SHORT_PANEL_CHARS: Final[int] = 400


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


#: Self-identification hits the desk has SEEN and ruled on. A hit outside
#: this set is an unacknowledged blind compromise and fails the selftest.
#:
#: L4-6082bb0de9b1 — one panel opens "As an AI developed by DeepSeek",
#: which names its own node to any reader. The pair stays in the frozen
#: draw (re-drawing on a text property would make the draw depend on the
#: text, which it must not), and the recommendation to the desk is to
#: EXCLUDE IT AT SCORING TIME: agreement is computed per stratum over the
#: valid pairs, so dropping one needs no re-freeze and no new session.
#: Empty in deck v2.1: the pair that opened "As an AI developed by
#: DeepSeek" (L4-6082bb0de9b1, deck v2.1 draft) is no longer drawn — the
#: glued-generation exclusions changed the pool under it. The mechanism
#: stays because the next such panel must fail loudly rather than ship.
ACKNOWLEDGED_SELF_ID: Final[dict[str, str]] = {}


def glued_fraction(text: str) -> float:
    """Fraction of characters inside tokens longer than GLUED_TOKEN_CHARS."""
    if not text:
        return 0.0
    return sum(len(x) for x in text.split() if len(x) > GLUED_TOKEN_CHARS) / len(text)


def scan_self_identification(pairs: list[BlindPair]) -> list[dict[str, object]]:
    """Which blind pairs contain a token that could name their own model.

    Word-boundary matched. A bare substring search flagged a cosmology
    passage listing "anthropics, metamathematics, astrochemistry" as
    naming a vendor, which is the kind of false positive that trains a
    reader to ignore the flag — and this flag has to stay worth reading,
    because a genuine hit ("As an AI developed by DeepSeek") really does
    partially unblind its pair.
    """
    hits: list[dict[str, object]] = []
    pats = [(t, re.compile(r"(?<![0-9a-z])" + re.escape(t) + r"(?![0-9a-z])"))
            for t in SELF_ID_TOKENS]
    for p in pairs:
        for slot, text in (("text_1", p.text_1), ("text_2", p.text_2)):
            low = text.lower()
            found = sorted({t for t, rx in pats if rx.search(low)})
            if found:
                excerpt = ""
                for t, rx in pats:
                    m = rx.search(low)
                    if m:
                        excerpt = text[max(0, m.start() - 80):m.start() + 80]
                        break
                hits.append({"pair_id": p.pair_id, "slot": slot,
                             "tokens": found, "excerpt": excerpt})
    return hits


# ── deck build ───────────────────────────────────────────────────────────


def build_deck(
    drawn: DrawnPairs, bank_path: Path
) -> tuple[BlindDeck, list[dict[str, object]], list[dict[str, object]]]:
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
    flagged: list[dict[str, object]] = []
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
            if len(t) < MIN_CHARS:
                raise ValueError(
                    f"{label} decoded to {len(t)} characters (< {MIN_CHARS}); "
                    f"a stub panel is unjudgeable — the desk must rule on this pair"
                )
            ratio = sum(1 for ch in t if ch.isspace()) / len(t)
            if ratio < SPACE_DEGENERATE_RATIO or len(t) < SHORT_PANEL_CHARS:
                flagged.append(
                    {
                        "pair_id": p.pair_id,
                        "type_key": p.type_key,
                        "set_label": p.set_label,
                        "slot": "text_1" if (label == kd) == p.dose_first else "text_2",
                        "chars": len(t),
                        "space_ratio": round(ratio, 5),
                        "space_degenerate": ratio < SPACE_DEGENERATE_RATIO,
                        "short": len(t) < SHORT_PANEL_CHARS,
                        "coordinates_SEALED": label,
                    }
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
    return deck, scan_self_identification(pairs), flagged


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


# ── deck v2: a revealed calibration block, then the blind block ──────────


def build_deck_v2(drawn, bank_path: Path) -> tuple[BlindDeck, list, list]:
    """Join the v2 sealed map to the bank.

    The calibration pairs come FIRST and carry `revealed_steered`, so the
    page can name which panel was steered and by how much. They are
    anchors, never gold: `stratum == "calibration"` in the sealed map, and
    the unblinding step must drop them before computing any agreement.
    """
    bank = load_bank(bank_path)
    bank_sha = sha256_file(bank_path)
    draw_sha = sha256_json(drawn.model_dump(mode="json"))

    pairs: list[BlindPair] = []
    missing: list[str] = []
    flagged: list[dict[str, object]] = []

    ordered = list(sorted(drawn.calibration, key=lambda x: x.ordinal_global)) + list(
        sorted(drawn.pairs, key=lambda x: x.ordinal_global)
    )
    for p in ordered:
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
            if len(t) < MIN_CHARS:
                raise ValueError(f"{label} decoded to {len(t)} chars (< {MIN_CHARS})")
            ratio = sum(1 for ch in t if ch.isspace()) / len(t)
            glued = glued_fraction(t)
            if p.revealed and glued > GLUED_FRACTION_MAX:
                raise ValueError(
                    f"{label} is a CALIBRATION anchor whose words run together "
                    f"({glued:.1%} of its characters sit in tokens longer than "
                    f"{GLUED_TOKEN_CHARS}). An anchor teaches Luxia what a real "
                    f"effect looks like; a glued one teaches her the decode "
                    f"artefact instead. Exclude this generation and re-freeze."
                )
            if (glued > GLUED_FRACTION_MAX or ratio < SPACE_DEGENERATE_RATIO
                    or len(t) < SHORT_PANEL_CHARS):
                flagged.append({
                    "pair_id": p.pair_id, "type_key": p.type_key,
                    "stratum": p.stratum, "set_label": p.set_label,
                    "chars": len(t), "space_ratio": round(ratio, 5),
                    "space_degenerate": ratio < SPACE_DEGENERATE_RATIO,
                    "glued_fraction": round(glued, 4),
                    "glued": glued > GLUED_FRACTION_MAX,
                    "short": len(t) < SHORT_PANEL_CHARS,
                    "coordinates_SEALED": label,
                })
        t1, t2 = (td, tb) if p.dose_first else (tb, td)
        rev = None
        if p.revealed:
            rev = "text_1" if p.dose_first else "text_2"
        pairs.append(BlindPair(
            pair_id=p.pair_id,
            set_label="Calibration" if p.revealed else p.set_label,
            trait=AXIS_TRAIT[p.axis], text_1=t1, text_2=t2,
            revealed_steered=rev,
            revealed_delta=(round(p.measured_delta, 3)
                            if (p.revealed and p.measured_delta is not None) else None),
            revealed_direction=("more" if (p.revealed and p.dose.startswith("+"))
                                else "less" if p.revealed else None),
        ))

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
        tool_version=TOOL_VERSION, grade=GRADE, status=STATUS,
        draw_recipe=drawn.draw_recipe, draw_sha256=draw_sha,
        sealed_map_sha256=draw_sha, bank_sha256=bank_sha,
        set_labels=(["Calibration"] if "Calibration" in totals else [])
        + sorted(k for k in totals if k != "Calibration"),
        set_totals=totals, pairs=pairs,
    )
    return deck, scan_self_identification(pairs), flagged
