"""The model-residency driver: one job, one card, one loaded model, many columns.

LUXIA'S RULING (2026-08-05 night), which this file enacts and nothing more: **one job
= one card = one loaded model = an ORDERED LIST of that model's cells documents
through a shared runtime**, with the boundary drawn explicitly — **residency sharing
YES, cell sharing NO**. A cell is 80 generations under ONE injection spec; batching
across cells would change the physics, so nothing here ever puts two columns' work
into one forward. What is shared is exactly the loaded weights.

WHY THIS IS A DRIVER AND NOT AN ENGINE MODULE. §2.1's job already takes its runtime
from its caller (`NodeRuntime`), so the loop is the only new orchestration — and the
one thing a loop cannot safely do from outside is re-point a loaded runtime at the
next column's (arm, site, vector bank). That lives in the engine, typed, as
`HFNodeRuntime.repoint` with its two refusals (`ResidencyModelMismatch` when the
document names another model, `ResidencyHookStillAttached` when a hook survived a
column), beside `run_resident_columns`, which is where the between-columns assertions
are made and where the CPU-toy proof drives it. This file is the CLI and the model
load: it parses a typed plan, loads ONE model, and hands the ordered list over.

PER-COLUMN INTEGRITY IS UNTOUCHED. Every column gets its own preflight, its own
canonical-layout freeze, its own §2.5 norm measurement, its own §4/§5 gates, its own
§2.8 stamps, its own §2.7 replay gate, its own M55 attempt lock and its own manifest.
Column ORDER cannot move a bit: §2.3's uniforms are per-(cell, gen_id) sha256 material
(ruling 8 / M5), so a cell's tape is a function of the cell and never of what ran
before it — the same property that makes resumption sound, used the other way round.

RESUMPTION RIDES ALONG. A plan may mark any column `"resume": true`; a completed
column is skipped whole (every cell verified, nothing to run), a partial one resumes
cell-by-cell, and both refuse by name on any identity difference. That is what makes a
residency restartable at the granularity of the wave rather than of the job.

    python -m metabasis.scripts.run_behavioral_residency --plan <PLAN>.json
    python -m metabasis.scripts.run_behavioral_residency --plan <PLAN>.json --dry-run

Heimdall CLI only for submission; the coordinator HTTP API is read-only verification.
All new node-side data lives under <NODE_DATA_ROOT> (standing rule 2026-07-29).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from metabasis.scripts.run_behavioral_cells import (BehavioralHarnessError,
                                                    HFNodeRuntime, ResidentColumn,
                                                    ResidencyPlanInvalid,
                                                    assert_no_attempt_dir_collision,
                                                    load_model_and_tokenizer,
                                                    load_vectors,
                                                    run_resident_columns)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_behavioral_residency")

RESIDENCY_SCHEMA_VERSION = "behavioral-residency/1"


class ResidencyColumnSpec(BaseModel):
    """One column of the plan, as it is written in the plan document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cells_json: Path
    prompt_pool: Path
    work_root: Path
    actuation_calibration: Optional[Path] = None
    n_per_cell: Optional[int] = None
    corpus_sha_of_record: Optional[str] = None
    characterize: bool = True
    resume: bool = False
    scheduler_card_index: Optional[str] = None


class ResidencyPlan(BaseModel):
    """One job's plan: one model, an ORDERED list of that model's columns.

    `node_key` is stated in the plan AND re-checked against every column's document
    inside the engine, from opposite directions on purpose: the plan says which model
    this job loads, and each document says which model its cells belong to. A job
    whose plan and documents disagree is refused by `ResidencyModelMismatch` before
    the second column fires, not discovered in a stamp afterwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["behavioral-residency/1"] = RESIDENCY_SCHEMA_VERSION
    node_key: str
    model_path: str
    dtype: str = "bfloat16"
    columns: tuple[ResidencyColumnSpec, ...] = Field(min_length=1)
    note: str = ""

    @model_validator(mode="after")
    def _columns_are_distinct_attempts(self) -> "ResidencyPlan":
        roots = [str(Path(c.work_root)) for c in self.columns]
        if len(set(roots)) != len(roots):
            dupes = sorted({r for r in roots if roots.count(r) > 1})
            raise ValueError(
                f"columns share attempt director{'ies' if len(dupes) > 1 else 'y'} "
                f"{dupes}. Residency shares the loaded MODEL; each column keeps its "
                "own attempt directory, its own gates and its own manifest (Luxia's "
                "boundary, 2026-08-05: residency sharing YES, cell sharing NO).")
        return self


def load_residency_plan(path: Path) -> ResidencyPlan:
    """Read and TYPE the plan, with every failure named (never a bare traceback)."""
    try:
        body = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidencyPlanInvalid(
            f"{path}: not a readable residency plan ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        return ResidencyPlan(**body)
    except ValidationError as exc:
        raise ResidencyPlanInvalid(f"{path}: {exc}") from exc


def plan_to_columns(plan: ResidencyPlan) -> list[ResidentColumn]:
    """The plan's columns as the engine's `ResidentColumn` list, path-shaped.

    Nothing is loaded here: the point of a dry run is to refuse a malformed plan
    before a model is loaded, and a plan whose paths do not exist is malformed.
    """
    missing: list[str] = []
    for i, col in enumerate(plan.columns):
        for name in ("cells_json", "prompt_pool"):
            p = Path(getattr(col, name))
            if not p.exists():
                missing.append(f"column {i}: --{name.replace('_', '-')} {p}")
        if col.actuation_calibration is not None and not Path(
                col.actuation_calibration).exists():
            missing.append(f"column {i}: actuation_calibration "
                           f"{col.actuation_calibration}")
    if missing:
        raise ResidencyPlanInvalid(
            "the residency plan names inputs that do not exist:\n  "
            + "\n  ".join(missing)
            + "\nA residency is the expensive unit; its inputs are checked before the "
              "model is loaded, never between columns.")
    columns = [ResidentColumn(
        work_root=Path(col.work_root), cells_json=Path(col.cells_json),
        prompt_pool=Path(col.prompt_pool),
        actuation_calibration=(None if col.actuation_calibration is None
                               else Path(col.actuation_calibration)),
        n_per_cell=col.n_per_cell, corpus_sha_of_record=col.corpus_sha_of_record,
        characterize=col.characterize, resume=col.resume,
        scheduler_card_index=col.scheduler_card_index) for col in plan.columns]
    assert_no_attempt_dir_collision(columns)
    return columns


def _summary(records: list[dict]) -> list[dict]:
    """The per-column line a job log is read by (never the whole ColumnResult)."""
    out = []
    for rec in records:
        result = rec.get("result")
        out.append({
            "index": rec["index"], "node_key": rec["node_key"], "arm": rec["arm"],
            "site": rec["site"], "label": rec.get("label"),
            "work_root": rec["work_root"],
            "n_cells": rec.get("n_cells"),
            "n_generations": None if result is None else result.n_generations,
            "canonical_batch_size": None if result is None else result.layout.batch_size,
            "replay_gate_passed": rec.get("replay_gate_passed"),
            "hooks_detached_after": rec.get("hooks_detached_after"),
            "refusal": rec.get("refusal")})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The model-residency driver (Luxia's ruling, 2026-08-05): one "
                    "job, one card, one loaded model, an ordered list of that "
                    "model's columns. Heimdall CLI only for submission.")
    ap.add_argument("--plan", type=Path, required=True,
                    help=f"a {RESIDENCY_SCHEMA_VERSION} plan document")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and validate the plan, resolve every input path, and "
                         "exit — no model is loaded and no cell fires")
    ap.add_argument("--continue-past-refusal", action="store_true",
                    help="do NOT stop the residency on a column's HALT. Off by "
                         "default: an enactor never adjudicates a live HALT (§3/§9), "
                         "and columns already banked are safe either way.")
    args = ap.parse_args(argv)

    runtime: Any = None
    try:
        plan = load_residency_plan(args.plan)
        columns = plan_to_columns(plan)
        logger.info("residency plan: %s, %d column(s), dtype %s", plan.node_key,
                    len(columns), plan.dtype)
        for i, col in enumerate(columns):
            logger.info("  column %d: %s → %s%s", i, col.cells_json, col.work_root,
                        " (RESUME)" if col.resume else "")
        if args.dry_run:
            print(json.dumps({"plan": args.plan.name, "node_key": plan.node_key,
                              "columns": [str(c.work_root) for c in columns],
                              "dry_run": True}, indent=1))
            return 0
        model, tok, dtype_name = load_model_and_tokenizer(plan.model_path,
                                                          dtype_name=plan.dtype)
        # ONE load. The first column's vectors are handed over at construction and
        # every subsequent column re-points through the engine's typed `repoint`.
        first = columns[0]
        from metabasis.scripts.build_behavioral_banks import load_cells_document
        first_doc = load_cells_document(Path(first.cells_json))
        runtime = HFNodeRuntime(
            model, tok, node_key=plan.node_key, arm=first_doc.arm, site=first_doc.site,
            vectors=load_vectors(Path(first_doc.vectors_npz)
                                 if first_doc.vectors_npz else None),
            dtype_name=dtype_name)
        records = run_resident_columns(
            runtime, columns, stop_on_refusal=not args.continue_past_refusal)
        print(json.dumps({"node_key": plan.node_key, "n_columns": len(records),
                          "columns": _summary(records)}, indent=1, default=str))
        return 0
    except BehavioralHarnessError as exc:
        logger.error("HALT (%s): %s", type(exc).__name__, exc)
        return 2
    except (OSError, ImportError) as exc:
        logger.error("environment failure (%s): %s", type(exc).__name__, exc)
        return 3
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:                     # noqa: BLE001 — M19(a)
                logger.warning("runtime close failed (%s: %s)",
                               type(exc).__name__, exc)


if __name__ == "__main__":
    sys.exit(main())
