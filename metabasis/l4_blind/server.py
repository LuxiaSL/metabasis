"""The localhost server. Python stdlib only; binds 127.0.0.1 and nothing else.

Three routes and no more:

    GET  /          the blind page (rendered once at boot, served from memory)
    GET  /state     the resume state, derived from the append-only log
    POST /verdict   append one verdict row, fsync, return {"ok":true}
    POST /session-note  append one session-note row, fsync

It reads the BLIND DECK. It does not know the sealed map exists — the
constructor takes a `BlindDeck`, and `serve()` is given a deck path; there
is no code path in this module that can open a sealed map. The selftest
asserts that by grepping this module for the sealed-map filename pattern.

Durability: every append is `write` + `flush` + `os.fsync` on a handle
opened in append mode, and the browser does not advance until the POST has
returned. A kill at any point costs at most the keystroke in flight.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

from metabasis.l4_blind import GRADE, TOOL_VERSION
from metabasis.l4_blind.models import BlindDeck, VerdictRow
from metabasis.l4_blind.render import render_page

#: Hard bind. Never 0.0.0.0 — the deck is unstamped material.
BIND_HOST: Final[str] = "127.0.0.1"
MAX_BODY: Final[int] = 1 << 20


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class VerdictLog:
    """Append-only JSONL with an exclusive in-process lock and fsync."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, row: VerdictRow) -> None:
        line = row.model_dump_json() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

    def replay(self) -> dict[str, Any]:
        """Last-write-wins resume state. Malformed tail lines are tolerated
        (a kill mid-write can leave one) and counted, never guessed at."""
        verdicts: dict[str, str] = {}
        notes: dict[str, str] = {}
        session_notes: list[str] = []
        bad_lines = 0
        n_rows = 0
        if not self.path.exists():
            return {"verdicts": {}, "notes": {}, "session_notes": [], "bad_lines": 0, "n_rows": 0}
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = VerdictRow.model_validate_json(line)
                except Exception:  # noqa: BLE001 — a torn tail line is expected
                    bad_lines += 1
                    continue
                n_rows += 1
                if row.kind == "session_note":
                    session_notes.append(row.note)
                    continue
                assert row.pair_id is not None and row.choice is not None
                verdicts[row.pair_id] = row.choice
                if row.note:
                    notes[row.pair_id] = row.note
        return {
            "verdicts": verdicts,
            "notes": notes,
            "session_notes": session_notes,
            "bad_lines": bad_lines,
            "n_rows": n_rows,
        }


class _Handler(BaseHTTPRequestHandler):
    server_version = "l4-blind"
    sys_version = ""

    # injected by serve()
    page: str = ""
    deck: BlindDeck | None = None
    log: VerdictLog | None = None
    session_id: str = ""
    shown_at: dict[str, str] = {}

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console clean
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # No off-origin anything: the page must work with the network down.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; form-action 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        if self.path in ("/", "/index.html"):
            self._send(200, self.page.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/state":
            assert self.log is not None
            self._json(200, self.log.replay())
        else:
            self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        assert self.log is not None and self.deck is not None
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if n <= 0 or n > MAX_BODY:
            self._json(400, {"error": "empty or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._json(400, {"error": "body is not JSON"})
            return

        now = utcnow()
        try:
            if self.path == "/verdict":
                pid = body.get("pair_id")
                match = [p for p in self.deck.pairs if p.pair_id == pid]
                if not match:
                    self._json(400, {"error": f"pair_id {pid!r} is not in this deck"})
                    return
                pair = match[0]
                ordinal = self.deck.pairs.index(pair)
                row = VerdictRow(
                    tool_version=TOOL_VERSION,
                    grade=GRADE,
                    session_id=self.session_id,
                    draw_sha256=self.deck.draw_sha256,
                    kind="verdict",
                    pair_id=pair.pair_id,
                    set_label=pair.set_label,
                    ordinal_global=ordinal,
                    choice=body.get("choice"),
                    note=str(body.get("note") or ""),
                    shown_at=body.get("shown_at"),
                    answered_at=now,
                    elapsed_ms=body.get("elapsed_ms"),
                    amends=bool(body.get("amends")),
                )
            elif self.path == "/session-note":
                row = VerdictRow(
                    tool_version=TOOL_VERSION,
                    grade=GRADE,
                    session_id=self.session_id,
                    draw_sha256=self.deck.draw_sha256,
                    kind="session_note",
                    note=str(body.get("note") or ""),
                    answered_at=now,
                )
            else:
                self._json(404, {"error": "no such route"})
                return
        except Exception as exc:  # noqa: BLE001 — validation failure is the client's fault
            self._json(422, {"error": f"rejected: {exc}"})
            return

        self.log.append(row)
        self._json(200, {"ok": True, "at": now})


def serve(
    deck: BlindDeck,
    log_path: Path,
    session_id: str,
    port: int = 8788,
    host: str = BIND_HOST,
) -> ThreadingHTTPServer:
    """Build the page once and start the server. Caller owns shutdown."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"refusing to bind {host!r}: this deck is UNSTAMPED material and the "
            f"tool is localhost-only by construction"
        )
    handler = type(
        "_BoundHandler",
        (_Handler,),
        {
            "page": render_page(deck, session_id),
            "deck": deck,
            "log": VerdictLog(log_path),
            "session_id": session_id,
            "shown_at": {},
        },
    )
    return ThreadingHTTPServer((host, port), handler)
