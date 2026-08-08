"""Desk-side extraction of the token ids the deck's generations need.

The node job needs three things: the token ids, the tokenizer, and the
engine's decode convention. Only the tokenizer and the convention live on
the cluster — the ids are already desk-side in the pull-parity-verified
banked trees. So this exports a small ids file and the node decodes it.

Why this way, and not "have the node re-read its own trees":

  * FOOTPRINT. The decode node is a shared machine, often under load.
    Shipping ~1 MB of ids and decoding a couple of hundred short sequences
    is seconds of one core; re-walking five columns' banked trees is not.
  * PROVENANCE. Every exported row records the sha256 of the
    `generations.jsonl` it came from, and the node job echoes those shas
    back untouched. The join is checkable end to end without trusting
    either side's directory layout.
  * BLAST RADIUS. The job never opens a banked tree at all, so it cannot
    perturb one.

C§8 UNSTAMPED. This exports ids and coordinates. No text, no score, no
verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metabasis.l4_blind import GRADE, STATUS, TOOL_VERSION
from metabasis.l4_blind.draw import DrawnPairs, deck_request
from metabasis.l4_blind.taxonomy import sha256_file


def export_ids(drawn: DrawnPairs, out_path: Path) -> dict[str, Any]:
    """Write the ids file the node job consumes, and return its summary.

    One line per requested generation:

        {"column","side","node_key","cell_id","generation_id","prompt_id",
         "generated_ids":[...], "finished_with_eos":bool,
         "source_generations_jsonl_sha256": "..."}
    """
    repo = Path(drawn.inputs.repo_root)
    rows = deck_request(drawn)

    # results_root per (column, side) — read off the sealed map, which got
    # it from the L2 ingredients' `source_tree` (the of-record path, FLAG-F
    # included: phi-4.refusal lives in the r2b attempt dir).
    roots: dict[tuple[str, str], str] = {}
    sites: dict[str, int] = {}
    for p in drawn.pairs:
        sites[p.column] = p.site
    from metabasis.l4_blind.taxonomy import TaxonomyInputs, load_pass_population

    for c in load_pass_population(drawn.inputs):
        roots[(c.column, c.side)] = c.results_root

    cache: dict[Path, tuple[dict[int, dict], str]] = {}

    def cell_rows(column: str, side: str, cell_id: str) -> tuple[dict[int, dict], str]:
        root = roots.get((column, side))
        if root is None:
            raise ValueError(f"no results root for {column}/{side}")
        p = repo / root / cell_id / "generations.jsonl"
        if p not in cache:
            by_gen: dict[int, dict] = {}
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    by_gen[int(r["generation_id"])] = r
            cache[p] = (by_gen, sha256_file(p))
        return cache[p]

    out: list[str] = []
    n_tokens = 0
    for req in rows:
        column, side = str(req["column"]), str(req["side"])
        cell_id, gid = str(req["cell_id"]), int(req["generation_id"])
        by_gen, src_sha = cell_rows(column, side, cell_id)
        row = by_gen.get(gid)
        if row is None:
            raise ValueError(f"{column}/{cell_id} has no generation {gid}")
        if str(row["prompt_id"]) != str(req["prompt_id"]):
            raise ValueError(
                f"{column}/{cell_id} g={gid}: prompt {row['prompt_id']} != "
                f"the sealed map's {req['prompt_id']} — refusing to export a "
                f"row the draw does not describe"
            )
        ids = list(row["generated_ids"])
        if not ids:
            raise ValueError(f"{column}/{cell_id} g={gid} has zero generated ids")
        n_tokens += len(ids)
        out.append(
            json.dumps(
                {
                    "column": column,
                    "side": side,
                    "node_key": req["node_key"],
                    "cell_id": cell_id,
                    "generation_id": gid,
                    "prompt_id": req["prompt_id"],
                    "generated_ids": ids,
                    "finished_with_eos": bool(row.get("finished_with_eos", False)),
                    "source_generations_jsonl_sha256": src_sha,
                },
                ensure_ascii=False,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    return {
        "STATUS": STATUS,
        "grade": GRADE,
        "tool_version": TOOL_VERSION,
        "what_this_is": (
            "Banked token ids for exactly the generations the L4 blind deck "
            "needs decoded. Exported desk-side from the pull-parity-verified "
            "banked trees; the node job decodes these and returns text."
        ),
        "ids_file": str(out_path),
        "ids_file_sha256": sha256_file(out_path),
        "n_rows": len(out),
        "n_tokens": n_tokens,
        "n_source_files": len(cache),
        "node_keys": sorted({str(r["node_key"]) for r in rows}),
        "columns": sorted({str(r["column"]) for r in rows}),
        "source_shas": {str(p): sha for p, (_, sha) in sorted(cache.items(), key=lambda kv: str(kv[0]))},
        "decode_of_record": (
            "metabasis.text_decode.maybe_decode(tokenizer.decode("
            "list(generated_ids), skip_special_tokens=True)); tokenizer from "
            "AutoTokenizer.from_pretrained(model_path), no extra kwargs — the "
            "engine's own coherence path (run_behavioral_cells:5239/5596). The "
            "maybe_decode unwrap is what makes this the CORRECTED decode, and "
            "it is MANDATORY for the dsv2-lite columns (Ġ-marker path)."
        ),
    }


def export_ids_for(repo: Path, drawn, rows: list[dict], out_path: Path) -> dict[str, Any]:
    """Export token ids for an explicit list of coordinate rows (deck v2)."""
    roots: dict[tuple[str, str], str] = {}
    from metabasis.l4_blind.taxonomy import load_pass_population

    for c in load_pass_population(drawn.inputs):
        roots[(c.column, c.side)] = c.results_root

    cache: dict[Path, tuple[dict[int, dict], str]] = {}

    def cell_rows(column: str, side: str, cell_id: str):
        root = roots.get((column, side))
        if root is None:
            raise ValueError(f"no results root for {column}/{side}")
        p = repo / root / cell_id / "generations.jsonl"
        if p not in cache:
            by_gen: dict[int, dict] = {}
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        by_gen[int(r["generation_id"])] = r
            cache[p] = (by_gen, sha256_file(p))
        return cache[p]

    out: list[str] = []
    n_tokens = 0
    for req in rows:
        by_gen, src_sha = cell_rows(req["column"], req["side"], req["cell_id"])
        row = by_gen.get(int(req["generation_id"]))
        if row is None:
            raise ValueError(f"{req['column']}/{req['cell_id']} missing g={req['generation_id']}")
        if str(row["prompt_id"]) != str(req["prompt_id"]):
            raise ValueError(
                f"{req['column']}/{req['cell_id']} g={req['generation_id']}: prompt "
                f"{row['prompt_id']} != sealed map's {req['prompt_id']}"
            )
        ids = list(row["generated_ids"])
        if not ids:
            raise ValueError(f"{req['column']}/{req['cell_id']} g={req['generation_id']}: no ids")
        n_tokens += len(ids)
        out.append(json.dumps({
            "column": req["column"], "side": req["side"], "node_key": req["node_key"],
            "cell_id": req["cell_id"], "generation_id": int(req["generation_id"]),
            "prompt_id": req["prompt_id"], "generated_ids": ids,
            "finished_with_eos": bool(row.get("finished_with_eos", False)),
            "source_generations_jsonl_sha256": src_sha,
        }, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "STATUS": STATUS, "grade": GRADE, "tool_version": TOOL_VERSION,
        "what_this_is": "Token ids for the generations deck v2 still needs text for.",
        "ids_file": str(out_path), "ids_file_sha256": sha256_file(out_path),
        "n_rows": len(out), "n_tokens": n_tokens, "n_source_files": len(cache),
        "node_keys": sorted({str(r["node_key"]) for r in rows}),
        "columns": sorted({str(r["column"]) for r in rows}),
        "source_shas": {str(p): sha for p, (_, sha) in
                        sorted(cache.items(), key=lambda kv: str(kv[0]))},
    }
