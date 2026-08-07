"""The end-of-session sealed artifact.

It carries the draw recipe, the tool version, every input sha and the
verdict tally — and NOT the sealed map. Sealing a session therefore does
not unblind it: the desk joins `sealed_map_sha256` to the map in a
separate, deliberate step.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.models import BlindDeck, SessionSeal, VerdictRow
from metabasis.l4_blind.server import VerdictLog, utcnow
from metabasis.l4_blind.taxonomy import sha256_file


def seal_session(
    deck: BlindDeck,
    deck_path: Path,
    log_path: Path,
    sealed_map_path: Path,
    session_id: str,
) -> SessionSeal:
    """Fold the append-only log into one sha'd summary. Read-only on both."""
    log = VerdictLog(log_path)
    state = log.replay()

    rows: list[VerdictRow] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(VerdictRow.model_validate_json(line))
            except Exception:  # noqa: BLE001 — counted as bad_lines by replay()
                continue

    verdicts: dict[str, str] = state["verdicts"]
    tally: dict[str, int] = {"text_1": 0, "text_2": 0, "unsure": 0}
    for choice in verdicts.values():
        tally[choice] = tally.get(choice, 0) + 1

    per_set_judged: dict[str, int] = {s: 0 for s in deck.set_labels}
    by_id = {p.pair_id: p for p in deck.pairs}
    for pid in verdicts:
        p = by_id.get(pid)
        if p is not None:
            per_set_judged[p.set_label] += 1

    elapsed = [r.elapsed_ms for r in rows if r.kind == "verdict" and r.elapsed_ms]
    median_s = round(statistics.median(elapsed) / 1000.0, 2) if elapsed else None

    return SessionSeal(
        tool_version=TOOL_VERSION,
        grade=GRADE,
        status=STATUS,
        session_id=session_id,
        sealed_at=utcnow(),
        draw_recipe=deck.draw_recipe,
        draw_sha256=deck.draw_sha256,
        sealed_map_sha256=deck.sealed_map_sha256,
        sealed_map_path=str(sealed_map_path),
        deck_sha256=sha256_file(deck_path),
        deck_path=str(deck_path),
        bank_sha256=deck.bank_sha256,
        verdict_log_path=str(log_path),
        verdict_log_sha256=sha256_file(log_path),
        n_pairs_in_deck=len(deck.pairs),
        n_pairs_judged=len(verdicts),
        n_unjudged=len(deck.pairs) - len(verdicts),
        n_amendments=sum(1 for r in rows if r.kind == "verdict" and r.amends),
        n_session_notes=len(state["session_notes"]),
        n_pair_notes=len(state["notes"]),
        choice_tally=tally,
        per_set_judged=per_set_judged,
        per_set_total=dict(deck.set_totals),
        median_seconds_per_pair=median_s,
    )


def write_json(path: Path, obj: object) -> str:
    """Write pretty, stable JSON and return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def write_manifest(paths: list[Path], out: Path, label: str) -> Path:
    """A sha256 manifest in the desk's usual shape (sha  basename)."""
    lines = [f"# {label}", f"# {GRADE} — {STATUS}", f"# {TOOL_VERSION}"]
    for p in sorted(paths):
        if p.exists():
            lines.append(f"{sha256_file(p)}  {p.name}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
