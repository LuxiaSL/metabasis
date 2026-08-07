"""`python -m metabasis.l4_blind.cli <command>` — the whole tool.

    freeze       derive + freeze the draw; write the SEALED MAP, the deck
                 request (what the decode job owes), and the taxonomy
                 proposal the desk adjudicates.
    build-deck   join the sealed map to a text bank -> the BLIND DECK.
    serve        run the localhost page against a blind deck.
    seal         fold the session log into the sealed session artifact.
    selftest     the full battery (draw determinism, resume, counterbalance,
                 no-label leak).

Order of operations, and who does what:

    desk    freeze                    (deterministic, no text needed)
    node    decode -> TEXT-BANK.jsonl (tokenizers only, 0 GPUs)
    desk    build-deck                (the only step that sees both sides)
    Luxia   serve                     (blind; she never sees the map)
    desk    seal, then unblind        (a separate step, not in this tool)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.bank import build_deck, synthetic_bank
from metabasis.l4_blind.draw import DRAW_RECIPE, deck_request, freeze_draw
from metabasis.l4_blind.models import BlindDeck
from metabasis.l4_blind.seal import seal_session, write_json, write_manifest
from metabasis.l4_blind.taxonomy import (
    AXIS_TRAIT,
    PAIRS_PER_TYPE,
    TAXONOMY_ALTERNATIVES,
    TAXONOMY_RULE,
    TaxonomyInputs,
    sha256_file,
)

#: Derived from where this file sits, never hardcoded: the repo must be
#: nameable without a username or an absolute path in the source (repo
#: sanitization rule). Pass `--repo` when running from a worktree, whose
#: `staging/` is gitignored and therefore empty.
DEFAULT_REPO = str(Path(__file__).resolve().parents[2])
DESK_SCORE_REL = "staging/reading-l2-arm8/DESK-SCORE-L2-2026-08-06.json"
INGREDIENTS_REL = "staging/reading-l2-arm8/L2-PROBE-INGREDIENTS-2026-08-06.json"
DEFAULT_CUSTODY = "staging/l4-tool"

SEALED_MAP_NAME = "SEALED-MAP-l4-microgold-2026-08-07.json"
DECK_REQUEST_NAME = "DECK-REQUEST-l4-microgold-2026-08-07.json"
PROPOSAL_NAME = "TAXONOMY-PROPOSAL-l4-microgold-2026-08-07.json"
DECK_NAME = "BLIND-DECK-l4-microgold-2026-08-07.json"


def _inputs(repo: Path) -> TaxonomyInputs:
    ds, ing = repo / DESK_SCORE_REL, repo / INGREDIENTS_REL
    for p in (ds, ing):
        if not p.is_file():
            raise SystemExit(
                f"HALT: frozen input missing: {p}\n"
                f"       `staging/` is gitignored, so a worktree does not have it. "
                f"Pass --repo <the main checkout>."
            )
    return TaxonomyInputs(
        desk_score_path=str(ds),
        desk_score_sha256=sha256_file(ds),
        ingredients_path=str(ing),
        ingredients_sha256=sha256_file(ing),
        repo_root=str(repo),
    )


def cmd_freeze(args: argparse.Namespace) -> int:
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = _inputs(repo)
    drawn = freeze_draw(inputs)

    map_path = out / SEALED_MAP_NAME
    map_sha = write_json(map_path, drawn.model_dump(mode="json"))

    req = deck_request(drawn)
    req_path = out / DECK_REQUEST_NAME
    write_json(
        req_path,
        {
            "STATUS": STATUS,
            "grade": GRADE,
            "tool_version": TOOL_VERSION,
            "what_this_is": (
                "The exact banked generations the L4 deck needs decoded TEXT for. "
                "Two rows per drawn pair (the dose generation and its baseline "
                "twin on the same prompt). Deduplicated on coordinates."
            ),
            "decode_of_record": (
                "metabasis.text_decode.maybe_decode(tokenizer.decode("
                "list(generated_ids), skip_special_tokens=True)) with "
                "AutoTokenizer.from_pretrained(model_path) and no extra kwargs — "
                "the engine's own coherence path. For the dsv2-lite columns this "
                "is the CORRECTED decode (staging/reading-bleed/corrected-l1-dsv2/), "
                "not the banked byte-BPE text."
            ),
            "bank_row_schema": {
                "column": "str", "side": "native|transported", "cell_id": "str",
                "generation_id": "int", "text": "str (decoded, verbatim)",
            },
            "sealed_map_sha256": map_sha,
            "n_rows": len(req),
            "n_pairs": len(drawn.pairs),
            "rows": req,
        },
    )

    prop_path = out / PROPOSAL_NAME
    write_json(
        prop_path,
        {
            "STATUS": STATUS,
            "grade": GRADE,
            "tool_version": TOOL_VERSION,
            "what_this_is": (
                "The PROPOSED cell-type taxonomy and draw rule, for desk "
                "adjudication BEFORE Luxia's session (brief 2026-08-07 §1). "
                "Nothing here is ruled."
            ),
            "draw_recipe": DRAW_RECIPE,
            "root_material": drawn.root_material,
            "pairs_per_type": PAIRS_PER_TYPE,
            "taxonomy_rule": TAXONOMY_RULE,
            "taxonomy_alternatives": TAXONOMY_ALTERNATIVES,
            "trait_wording_proposed": dict(AXIS_TRAIT),
            "trait_note": (
                "The judged trait is VISIBLE on the page by necessity: the L3 "
                "judge sees {trait} in the frozen 2AFC template, and an "
                "agreement number computed over two different questions is not "
                "an agreement number. What the opaque 'Set X' labels hide is "
                "the native/transported split on the same axis — the split the "
                "science turns on. The desk must rule this explicitly."
            ),
            "inputs": inputs.model_dump(mode="json"),
            "types": [t.model_dump(mode="json") for t in drawn.types],
            "prompt_alignment_verified": drawn.prompt_alignment_verified,
            "sealed_map_sha256": map_sha,
            "sealed_map_path": str(map_path),
        },
    )

    man = write_manifest(
        [map_path, req_path, prop_path],
        out / "MANIFEST-l4-tool-freeze-20260807.sha256",
        "L4 micro-gold — frozen draw (sealed map + deck request + proposal)",
    )

    print(f"draw frozen: {len(drawn.pairs)} pairs across {len(drawn.types)} cell types")
    for t in drawn.types:
        print(f"  {t.set_label}  {t.type_key:<24} {t.pairs_drawn:>3} pairs  "
              f"from {t.n_pass_cells} PASS cells ({t.n_pass_cells_plus}+/{t.n_pass_cells_minus}-)")
    print(f"sealed map : {map_path}  sha256={map_sha}")
    print(f"deck request: {req_path}  ({len(req)} generations owed text)")
    print(f"proposal   : {prop_path}")
    print(f"manifest   : {man}")
    return 0


def cmd_build_deck(args: argparse.Namespace) -> int:
    from metabasis.l4_blind.draw import DrawnPairs

    out = Path(args.out)
    map_path = Path(args.sealed_map or (out / SEALED_MAP_NAME))
    drawn = DrawnPairs.model_validate_json(map_path.read_text())
    bank = Path(args.bank)
    if not bank.is_file():
        raise SystemExit(
            f"HALT: no text bank at {bank}. The decode job owes "
            f"{DECK_REQUEST_NAME}'s rows; nothing desk-side carries decoded text."
        )
    deck, selfid = build_deck(drawn, bank)
    deck_path = out / DECK_NAME
    sha = write_json(deck_path, deck.model_dump(mode="json"))
    if selfid:
        write_json(
            out / "FLAG-self-identification-2026-08-07.json",
            {
                "STATUS": STATUS,
                "grade": GRADE,
                "what_this_is": (
                    "Blind pairs whose TEXT contains a token that could name its "
                    "own model. REPORTED, NEVER EDITED — the L3 judge reads the "
                    "same bytes, and diverging the two would break the agreement "
                    "number. The desk rules whether to keep, drop, or re-draw."
                ),
                "n_hits": len(selfid),
                "hits": selfid,
            },
        )
    print(f"blind deck : {deck_path}  sha256={sha}")
    print(f"pairs      : {len(deck.pairs)}  sets: {deck.set_totals}")
    print(f"self-id flags: {len(selfid)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import webbrowser

    deck = BlindDeck.model_validate_json(Path(args.deck).read_text())
    log_path = Path(args.log)
    session_id = args.session or datetime.now(timezone.utc).strftime("s%Y%m%dT%H%M%SZ")

    from metabasis.l4_blind.server import serve

    httpd = serve(deck, log_path, session_id, port=args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"L4 blind judging — {len(deck.pairs)} pairs, sets {sorted(deck.set_totals)}")
    print(f"  session : {session_id}")
    print(f"  log     : {log_path}   (append-only; kill-safe, resumes on reload)")
    print(f"  open    : {url}")
    print("  Ctrl-C when done, then run the `seal` command.")
    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless is fine, the URL is printed
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. Run `seal` to write the sealed session artifact.")
    finally:
        httpd.server_close()
    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    out = Path(args.out)
    deck_path = Path(args.deck)
    deck = BlindDeck.model_validate_json(deck_path.read_text())
    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"HALT: no verdict log at {log_path}")
    session_id = args.session or "unnamed"
    sealed = seal_session(deck, deck_path, log_path, out / SEALED_MAP_NAME, session_id)
    seal_path = out / f"SESSION-SEAL-l4-microgold-{session_id}.json"
    sha = write_json(seal_path, sealed.model_dump(mode="json"))
    write_manifest(
        [seal_path, log_path, deck_path],
        out / f"MANIFEST-l4-session-{session_id}.sha256",
        f"L4 micro-gold session {session_id}",
    )
    print(f"sealed: {seal_path}  sha256={sha}")
    print(f"  judged {sealed.n_pairs_judged}/{sealed.n_pairs_in_deck}, "
          f"unsure {sealed.choice_tally.get('unsure', 0)}, "
          f"median {sealed.median_seconds_per_pair}s/pair")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from metabasis.l4_blind.selftest import run_all

    return 0 if run_all(Path(args.repo), Path(args.out)) else 1


def cmd_synthetic_bank(args: argparse.Namespace) -> int:
    from metabasis.l4_blind.draw import DrawnPairs

    drawn = DrawnPairs.model_validate_json(Path(args.sealed_map).read_text())
    p = synthetic_bank(drawn, Path(args.bank))
    print(f"SYNTHETIC bank written (selftests only): {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="l4-blind", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=str(Path(DEFAULT_REPO) / DEFAULT_CUSTODY))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("freeze", help="freeze the draw").set_defaults(fn=cmd_freeze)

    bd = sub.add_parser("build-deck", help="join sealed map + text bank -> blind deck")
    bd.add_argument("--bank", required=True)
    bd.add_argument("--sealed-map", default=None)
    bd.set_defaults(fn=cmd_build_deck)

    sv = sub.add_parser("serve", help="run the blind page")
    sv.add_argument("--deck", required=True)
    sv.add_argument("--log", required=True)
    sv.add_argument("--port", type=int, default=8788)
    sv.add_argument("--session", default=None)
    sv.add_argument("--open-browser", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    sl = sub.add_parser("seal", help="write the sealed session artifact")
    sl.add_argument("--deck", required=True)
    sl.add_argument("--log", required=True)
    sl.add_argument("--session", default=None)
    sl.set_defaults(fn=cmd_seal)

    sb = sub.add_parser("synthetic-bank", help="deterministic placeholder bank (selftests only)")
    sb.add_argument("--sealed-map", required=True)
    sb.add_argument("--bank", required=True)
    sb.set_defaults(fn=cmd_synthetic_bank)

    sub.add_parser("selftest", help="run the full battery").set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
