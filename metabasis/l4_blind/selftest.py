"""The selftest battery. Every check the brief names, plus the ones that
would have caught the failure modes quietly.

    1  draw determinism, twice in-process
    2  draw determinism in a FRESH interpreter (catches PRNG / hash-salt
       dependence, which an in-process repeat cannot see)
    3  the draw module contains no `random` and no `hash(`  (M25)
    4  dose-sign balance: 8/8 per cell type
    5  order counterbalance: 4/4 dose-first per (type, sign)
    6  no generation drawn twice; every pair id unique
    7  prompt alignment: every drawn pair is the SAME prompt on both sides
    8  resume mid-stream: replay, torn-tail tolerance, last-write-wins
    9  NO LABEL LEAK: the rendered page's own chrome contains zero hits for
       any column, model, node, axis, side, dose, arm, wave, cell id or
       site token in the population
    10 the serving path cannot open the sealed map
    11 the blind deck's serialised rows carry no coordinate field
    12 no trait string contains its own axis key (the sentiment trap)
    13 HTTP smoke: serve, POST a verdict, read it back off disk

Run: `python -m metabasis.l4_blind.cli selftest`
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Callable

from metabasis.l4_blind import GRADE, TOOL_VERSION
from metabasis.l4_blind.bank import build_deck, normalise_for_display, synthetic_bank
from metabasis.l4_blind.draw import DrawnPairs, freeze_draw
from metabasis.l4_blind.models import VerdictRow
from metabasis.l4_blind.render import render_page
from metabasis.l4_blind.server import VerdictLog, serve
from metabasis.l4_blind.taxonomy import (
    AXIS_TRAIT,
    PAIRS_PER_TYPE,
    TaxonomyInputs,
    load_pass_population,
    sha256_file,
)

_RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _RESULTS.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def _inputs(repo: Path) -> TaxonomyInputs:
    ds = repo / "staging/reading-l2-arm8/DESK-SCORE-L2-2026-08-06.json"
    ing = repo / "staging/reading-l2-arm8/L2-PROBE-INGREDIENTS-2026-08-06.json"
    return TaxonomyInputs(
        desk_score_path=str(ds),
        desk_score_sha256=sha256_file(ds),
        ingredients_path=str(ing),
        ingredients_sha256=sha256_file(ing),
        repo_root=str(repo),
    )


# ── leak vocabulary ──────────────────────────────────────────────────────


def leak_tokens(repo: Path, inputs: TaxonomyInputs) -> list[str]:
    """Every string that would betray a source, built from the population
    itself so it can never drift from what the draw actually contains."""
    cells = load_pass_population(inputs)
    toks: set[str] = set()
    for c in cells:
        toks.update(
            {
                c.column, c.node_key, c.axis, c.side, c.dose, c.arm, c.wave,
                c.dose_cell_id, c.baseline_cell_id, f"L{c.site}", c.results_root,
            }
        )
    toks.update(
        {
            "transported", "native", "naive", "baseline", "gentropy_gradient",
            "entropy_gradient", "gcaa", "caa_", "gRband", "Rband", "dose",
            "+0.30", "-0.30", "a+0.30", "a-0.30", "class-pilot", "EGV",
            "deepseek", "mistral", "olmo", "phi-4", "qwen", "llama", "pythia",
            "gemma", "dsv2", "dsv3", "bridge-row",
        }
    )
    # Never grep for something that is legitimately part of the tool's own
    # blind vocabulary, or the test would be unfalsifiable.
    toks -= {"", None}  # type: ignore[arg-type]
    return sorted(t for t in toks if isinstance(t, str) and len(t) >= 2)


def _grep(haystack: str, tokens: list[str]) -> dict[str, int]:
    hits: dict[str, int] = {}
    low = haystack.lower()
    for t in tokens:
        pat = re.compile(r"(?<![0-9a-z])" + re.escape(t.lower()) + r"(?![0-9a-z])")
        n = len(pat.findall(low))
        if n:
            hits[t] = n
    return hits


def _banned_symbols(path: Path) -> list[str]:
    """AST scan for the two banned non-determinism sources (M25).

    Catches `import random`, `from random import …`, any `random.*`
    attribute, and any call to the builtin `hash` — and cannot be fooled
    (or falsely tripped) by prose in a docstring or comment.
    """
    import ast

    found: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "random":
                    found.add("import random")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "random":
                found.add("from random import …")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "random":
                found.add(f"random.{node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "hash":
                found.add("hash()")
    return sorted(found)


def _redact_pair_texts(html: str, deck) -> str:
    """Remove the judged texts so the grep sees the TOOL's chrome only.

    A model that writes its own name into a generation is a content
    question the desk rules on (`FLAG-self-identification`); it is not the
    tool leaking a label, and conflating the two would make this test
    unfalsifiable the moment a real generation says "as a large language
    model".
    """
    out = html
    for p in deck.pairs:
        for t in (p.text_1, p.text_2):
            for variant in (t, normalise_for_display(t)):
                out = out.replace(variant, " [REDACTED-JUDGED-TEXT] ")
                # the deck is embedded as JSON, so the escaped form too
                out = out.replace(json.dumps(variant)[1:-1], " [REDACTED-JUDGED-TEXT] ")
    return out


# ── the battery ──────────────────────────────────────────────────────────


def run_all(repo: Path, out_dir: Path) -> bool:
    print(f"L4 blind tool selftest — {TOOL_VERSION} — {GRADE}")
    print(f"  repo: {repo}")
    inputs = _inputs(repo)

    # 1 — determinism, twice in-process
    print("\n[draw]")
    d1 = freeze_draw(inputs)
    d2 = freeze_draw(inputs)
    j1 = json.dumps(d1.model_dump(mode="json"), sort_keys=True)
    j2 = json.dumps(d2.model_dump(mode="json"), sort_keys=True)
    check(j1 == j2, "draw determinism (twice in-process)",
          f"{len(d1.pairs)} pairs, {len(d1.types)} types")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        map_path = tmp / "map.json"
        map_path.write_text(json.dumps(d1.model_dump(mode="json"), sort_keys=True))

        # 2 — determinism in a fresh interpreter
        prog = (
            "import json,sys;"
            "sys.path.insert(0,%r);"
            "from pathlib import Path;"
            "from metabasis.l4_blind.draw import freeze_draw;"
            "from metabasis.l4_blind.taxonomy import TaxonomyInputs;"
            "i=TaxonomyInputs.model_validate_json(sys.stdin.read());"
            "print(json.dumps(freeze_draw(i).model_dump(mode='json'),sort_keys=True))"
        ) % str(Path(inspect.getfile(freeze_draw)).parents[2])
        r = subprocess.run(
            [sys.executable, "-c", prog],
            input=inputs.model_dump_json(), capture_output=True, text=True,
            env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
        )
        check(r.returncode == 0 and r.stdout.strip() == j1,
              "draw determinism (fresh interpreter, PYTHONHASHSEED=random)",
              r.stderr.strip()[-200:] if r.returncode else "byte-identical")

        # 3 — no PRNG, no hash(). Parsed, not grepped: the module's own
        # docstring explains WHY it shuns random/hash, and a substring
        # search would flag that prose forever.
        bad = _banned_symbols(Path(inspect.getfile(freeze_draw)))
        check(not bad, "draw module uses no PRNG and no hash() (M25, AST-checked)",
              f"found {bad}" if bad else "sha256 key-sort only")

        # 4/5/6/7 — balance and integrity
        print("\n[balance]")
        by_type: dict[str, list] = {}
        for p in d1.pairs:
            by_type.setdefault(p.type_key, []).append(p)
        ok_n = all(len(v) == PAIRS_PER_TYPE for v in by_type.values())
        check(ok_n, f"{PAIRS_PER_TYPE} pairs per cell type",
              ", ".join(f"{k}={len(v)}" for k, v in sorted(by_type.items())))

        sign_ok, flip_ok = True, True
        detail_sign, detail_flip = [], []
        for tk, ps in sorted(by_type.items()):
            plus = [p for p in ps if p.dose == "+0.30"]
            minus = [p for p in ps if p.dose == "-0.30"]
            if len(plus) != PAIRS_PER_TYPE // 2 or len(minus) != PAIRS_PER_TYPE // 2:
                sign_ok = False
                detail_sign.append(f"{tk}:{len(plus)}+/{len(minus)}-")
            for sign, sub in (("+", plus), ("-", minus)):
                df = sum(1 for p in sub if p.dose_first)
                if df != len(sub) // 2:
                    flip_ok = False
                    detail_flip.append(f"{tk}{sign}:{df}/{len(sub)}")
        check(sign_ok, "dose-sign balance 8/8 within every type",
              "; ".join(detail_sign) or "exact")
        check(flip_ok, "order counterbalance 4/4 dose-first within every (type, sign)",
              "; ".join(detail_flip) or "exact")

        nat = [p.natural_id for p in d1.pairs]
        ids = [p.pair_id for p in d1.pairs]
        check(len(set(nat)) == len(nat), "no generation drawn twice", f"{len(nat)} pairs")
        check(len(set(ids)) == len(ids), "pair ids unique")
        check(
            d1.prompt_alignment_verified.startswith(f"{len(d1.pairs)}/{len(d1.pairs)}"),
            "every pair is the SAME prompt on both sides",
            d1.prompt_alignment_verified,
        )
        globals_ok = [p.ordinal_global for p in sorted(d1.pairs, key=lambda x: x.ordinal_global)]
        check(globals_ok == list(range(len(d1.pairs))), "global ordering is a permutation")
        first10 = [p.set_label for p in sorted(d1.pairs, key=lambda x: x.ordinal_global)][:10]
        check(len(set(first10)) >= 4, "types are interleaved, not blocked",
              f"first 10: {first10}")

        # 12 — trait strings
        print("\n[traits]")
        trap = [a for a, t in AXIS_TRAIT.items() if a.lower() in t.lower()]
        check(not trap, "no trait string contains its own axis key",
              f"leaky: {trap}" if trap else "; ".join(f"{a}->{t}" for a, t in AXIS_TRAIT.items()))

        # deck build against a synthetic bank
        print("\n[deck + leak]")
        bank = synthetic_bank(d1, tmp / "SYNTHETIC-BANK.jsonl")
        deck, selfid = build_deck(d1, bank)
        check(len(deck.pairs) == len(d1.pairs), "deck covers every drawn pair",
              f"{len(deck.pairs)}")
        check(not selfid, "synthetic bank trips no self-identification flag")

        # 11 — no coordinate fields survive into the deck
        row_keys = set()
        for p in deck.pairs:
            row_keys |= set(p.model_dump(mode="json").keys())
        forbidden = {"column", "side", "dose", "axis", "node_key", "arm", "site",
                     "dose_cell_id", "baseline_cell_id", "generation_id",
                     "prompt_id", "type_key", "dose_first", "natural_id"}
        check(not (row_keys & forbidden), "blind deck rows carry no coordinate field",
              f"keys={sorted(row_keys)}")

        # 9 — the leak grep
        html = render_page(deck, "selftest")
        chrome = _redact_pair_texts(html, deck)
        toks = leak_tokens(repo, inputs)
        hits = _grep(chrome, toks)
        check(not hits, f"NO LABEL LEAK — {len(toks)} source tokens, zero hits in the page chrome",
              f"HITS: {hits}" if hits else f"tokens checked: {len(toks)}")
        raw_hits = _grep(html, toks)
        check(not raw_hits, "no leak in the raw page either (synthetic texts are clean)",
              f"{raw_hits}" if raw_hits else "clean")

        # 10 — the serving path cannot reach the sealed map
        srv_src = Path(inspect.getfile(serve)).read_text()
        bad_refs = [m for m in ("SEALED-MAP", "sealed_map", "DrawnPairs", "PairCoord")
                    if m in srv_src]
        check(not bad_refs, "the serving module cannot open the sealed map",
              f"references {bad_refs}" if bad_refs else "no symbol, no path, no import")

        # 8 — resume
        print("\n[resume]")
        log_path = tmp / "verdicts.jsonl"
        log = VerdictLog(log_path)
        mk = lambda pid, ch, note="", amends=False: VerdictRow(  # noqa: E731
            tool_version=TOOL_VERSION, grade=GRADE, session_id="selftest",
            draw_sha256=deck.draw_sha256, kind="verdict", pair_id=pid,
            set_label="Set A", ordinal_global=0, choice=ch, note=note,
            answered_at="2026-08-07T00:00:00.000Z", elapsed_ms=1234, amends=amends,
        )
        sample = [p.pair_id for p in deck.pairs[:7]]
        for i, pid in enumerate(sample):
            log.append(mk(pid, "text_1" if i % 2 == 0 else "text_2"))
        log.append(mk(sample[0], "unsure", note="changed my mind", amends=True))
        log.append(VerdictRow(
            tool_version=TOOL_VERSION, grade=GRADE, session_id="selftest",
            draw_sha256=deck.draw_sha256, kind="session_note",
            note="a session-level observation", answered_at="2026-08-07T00:00:01.000Z"))
        with log_path.open("a") as fh:
            fh.write('{"schema_name":"l4-verdict/v1","tool_ver')  # torn tail

        st = VerdictLog(log_path).replay()
        check(len(st["verdicts"]) == len(sample), "resume recovers every judged pair",
              f"{len(st['verdicts'])}/{len(sample)}")
        check(st["verdicts"][sample[0]] == "unsure", "last write wins on an amendment")
        check(st["notes"].get(sample[0]) == "changed my mind", "the pair note survives")
        check(st["session_notes"] == ["a session-level observation"], "session notes stream")
        check(st["bad_lines"] == 1, "a torn tail line is tolerated and counted",
              f"bad_lines={st['bad_lines']}")
        unjudged = [p.pair_id for p in deck.pairs if p.pair_id not in st["verdicts"]]
        check(unjudged and unjudged[0] == deck.pairs[len(sample)].pair_id,
              "resume lands on the first unjudged pair")
        check("unsure" in {v for v in st["verdicts"].values()},
              "UNSURE is its own recorded state, never coerced to a pick")

        # 13 — HTTP smoke
        print("\n[http]")
        live_log = tmp / "live.jsonl"
        httpd = serve(deck, live_log, "selftest-http", port=0)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            base = f"http://127.0.0.1:{port}"
            page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
            check("L4 micro-gold" in page and "deck-data" in page, "GET / serves the page",
                  f"{len(page)} bytes")
            body = json.dumps({"pair_id": deck.pairs[0].pair_id, "choice": "text_2",
                               "note": "smoke", "elapsed_ms": 900}).encode()
            req = urllib.request.Request(base + "/verdict", data=body,
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
            check(resp.get("ok") is True, "POST /verdict accepted")
            check(live_log.read_text().count("\n") == 1, "the verdict hit disk immediately",
                  "1 line, fsync'd")
            state = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
            check(state["verdicts"].get(deck.pairs[0].pair_id) == "text_2",
                  "GET /state replays it back")
            bad = urllib.request.Request(
                base + "/verdict",
                data=json.dumps({"pair_id": "L4-000000000000", "choice": "text_1"}).encode(),
                headers={"Content-Type": "application/json"})
            code = 0
            try:
                urllib.request.urlopen(bad, timeout=5)
            except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                code = e.code
            check(code == 400, "a pair id outside the deck is refused", f"HTTP {code}")
        finally:
            httpd.shutdown()
            httpd.server_close()

    n_fail = sum(1 for ok, _, _ in _RESULTS if not ok)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed"
          + (f" — {n_fail} FAILED" if n_fail else " — all green"))
    return n_fail == 0
