"""The prod-lane (vLLM) BRIDGE COLUMN runner — bar INGREDIENTS, never verdicts.

UNSTAMPED (C§8). This fires the §3 bridge population of the stamped prod-lane
pre-statement (2026-08-07 + its dated addendum): matched cell shapes on the vLLM
lane against the ALREADY-BANKED byte-exact columns. It computes the raw
quantities the four bars consume and files them beside their banked
counterparts. **It never scores a bar and never writes a verdict** — no PASS,
no FAIL, no DEGENERATE appears in anything this module emits. The desk scores.

WHAT IS MATCHED AND WHAT IS NOT (pre-statement §3 / §2, verbatim in shape):
  * MATCHED: node · site · arm · the native/transported VECTOR (read from the
    banked bank npz, same bytes) · the frozen 6-dose signed ladder · n=80 ·
    the frozen prompt pool · max_new_tokens 512.
  * NOT MATCHED, BY DESIGN: the per-(cell,gen_id) uniform tape does not carry
    (per-request seeding), so all comparison is UNPAIRED population-level; and
    the random BANDS are drawn under the ruled recipe with LANE-TAGGED seed
    material (§3), i.e. distinct draws in the same space under the same
    construction. The ruled `RandomBandRecipe` is imported and used unedited —
    the tag rides `vector_key`, which is a field of the ruled seed template.

THE ENTROPY INSTRUMENT IS THE CERTIFIED ONE. Both halves of the read are
TEACHER-FORCED SCORING passes (`prompt_logprobs` set, one sequence per call),
which is the path the 2026-08-08 adjudication certified at MEAN level (unsteered
cell-mean agreement 0.14% relative). The generation-time path is used for ONE
thing only — pinning the index convention BY VALUE, per column, against a
reference this module produces itself — and never for a banked number.

THE FIVE FOOTGUNS, EACH GUARDED HERE RATHER THAN REMEMBERED:
  1. the prompt object is not a string  -> `render_prompt` (the engine's own),
     with `arm` passed EXPLICITLY by the caller; a hardcoded arm is a refusal.
  2. vLLM decoder layers are (positions, hidden_states, residual) with a fused
     add-and-norm -> handled inside `vllm_residual_write`; asserted attached.
  3. prefix caching / chunked prefill must be OFF -> asserted BY VALUE off the
     live engine config, not trusted to the constructor kwargs.
  4. the entropy index convention (position vs produced) -> inferred by value
     per column, with an unknown-convention refusal.
  5. the span -> every cell carries an EXACT token accounting gate: the tokens
     the wrapper skipped must equal the prompts' tokens and the tokens it wrote
     must equal the generated tokens, both counted independently.

THE TENSOR-PARALLEL PATH (re-freeze #4, 2026-08-20). `--tensor-parallel-size > 1`
selects a second attachment and NOTHING ELSE. The certified TP=1 attachment
reaches the model IN-PROCESS through `driver_worker`, which exists only on
vLLM's `UniProcExecutor`; at TP>1 the model lives in worker PROCESSES and the
parent holds no model object, so `MultiprocExecutor` has no `driver_worker` and
the in-process walk refuses loudly. The TP branch installs
`metabasis.extraction.vllm_worker_ext.MetabasisWorkerExtension` as vLLM's
`worker_extension_cls` (mixed into the Worker class in EVERY rank,
worker_base.py:262-287) and drives it by name through `collective_rpc`;
`metabasis.extraction.vllm_tp_proxy` presents the identical attribute surface in
the parent, so the ~700 lines of column orchestration below — the refusals, the
span-accounting gate, the index pinning — are untouched. The TP=1 branch is the
CERTIFIED block, verbatim: its identity against the certified tool is pinned by
`ast.unparse` digest in this module's selftest, not by a promise.

  * the two certified extraction modules are BYTE-UNCHANGED, and their shas are
    asserted in the selftest and stamped on every column. Their arithmetic was
    already TP-correct: the inter-layer residual is post-all-reduce and
    replicated (`o_proj`/`down_proj` are RowParallelLinear with
    reduce_results=True — qwen2.py:162, :94), and `compute_logits` all-gathers
    full vocab with padding stripped (platforms/interface.py:563;
    logits_processor.py:102), so every rank sees the identical tensor.
  * cross-rank agreement is a STRUCTURAL GATE, enforced in-band on every cell:
    >1e-4 entropy divergence, or any disagreement in the wrapper stats, the §2.5
    median, or the mask counters, REFUSES the column rather than averaging it.
  * `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required at TP>1 and inert at TP=1.
    Fork from a parent already dirtied by an OpenMP thread pool
    (`OMP_NUM_THREADS=8` plus ~98 CPU torch selftest checks before `LLM()`) hangs
    before a single worker line appears; vLLM's `_maybe_force_spawn()` escalates
    only when CUDA is initialised, which the CPU-only preflight never does. The
    runner exports it; this module records what it observed.

DEPLOYMENT NEUTRALITY. This module names no host, no user, and no absolute path.
The host allow-list and the required-environment list are ARGUMENTS
(`--allowed-host`, `--require-env`, both repeatable, both empty by default) and
every path the run needs is a required argument. Passing neither guard means the
tool imposes none and the RUNNER's guard is the guard — which is what the lane's
wrapper scripts already enforce, and what the column stamp records under
`deployment_guard` so a reader can see which guard actually ran. No package
management happens here and nothing is written outside `--out-dir`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

C8 = "UNSTAMPED (C§8) — BRIDGE INGREDIENTS. The desk scores the bars; nothing here verdicts."

#: vLLM's `worker_extension_cls` takes an IMPORTABLE QUALIFIED NAME resolved
#: inside each worker process. It is the repo path, never a flat module on
#: `PYTHONPATH`: a flat name would resolve against whatever beside copy the
#: deployment happens to carry, and one stale rank is a silently part-steered
#: column.
WORKER_EXTENSION_CLS = (
    "metabasis.extraction.vllm_worker_ext.MetabasisWorkerExtension")

#: The instrument's own identity, stamped on every column. The two certified
#: modules are byte-unchanged at re-freeze #4; the two attachment modules and
#: this tool are new. Read from the files at run time — a hardcoded digest of
#: your own source is a digest of what you wish were there.
INSTRUMENT_MODULES = ("metabasis/extraction/vllm_residual_write.py",
                      "metabasis/extraction/vllm_entropy_probe.py",
                      "metabasis/extraction/vllm_worker_ext.py",
                      "metabasis/extraction/vllm_tp_proxy.py",
                      "metabasis/scripts/vllm_lane_column.py")

#: The two CERTIFIED modules' shas of record (re-freeze #4). These do NOT move.
CERTIFIED_MODULE_SHA256 = {
    "metabasis/extraction/vllm_residual_write.py":
        "fafc07d78201b5e8ca909244dff419a892daf0fd316eb3520f648887a05d514f",
    "metabasis/extraction/vllm_entropy_probe.py":
        "e9902a21e4ef65b961ebf49b71687ecfe38e2632155af90d8f2737ff2bdcddc0",
}


def package_root() -> Path:
    """The directory that CONTAINS the `metabasis` package, proven by value.

    RAKE M59(2): path resolution in proof code is layout-sensitive by default.
    This resolves off THIS file rather than off a working directory, and the
    caller checks that the files it names exist before trusting a digest — a
    missing file is a refusal, never an absent entry that reads as clean.
    """
    return Path(__file__).resolve().parents[2]


def instrument_module_sha256() -> dict[str, str]:
    """sha256 of every module of the instrument, keyed by repo-relative path."""
    root = package_root()
    out: dict[str, str] = {}
    for rel in INSTRUMENT_MODULES:
        p = root / rel
        if not p.exists():
            raise LaneColumnRefused(
                f"the instrument module {rel} is not at {p} — the column stamp "
                "cannot pin an identity for a file it cannot read, and an "
                "absent entry would read as a clean stamp")
        out[rel] = sha256_file(p)
    return out


# ── re-freeze #4 identity pins (Luxia's ruling 2026-08-20) ───────────────────
#
# WHAT THESE ARE FOR. The lane column tool enters the repo as ONE tool whose
# TP>1 path is a new branch and whose TP=1 path must be the CERTIFIED tool,
# unchanged. "Unchanged" is not a claim to be repeated in prose; it is a
# property to be asserted, and the assertion has to survive re-indentation
# (the certified block now sits one level deeper inside `else:`), comment
# rewrites, and docstring edits. So the pin is over the DOCSTRING-STRIPPED
# `ast.unparse` of the block — code, normalised — and never over its text
# (rake M58: a string needle cannot tell code from prose).
#
# The digests below were computed at lift time against the certified node-side
# tool `vllm_lane_column.py`, whose sha is recorded here beside them. Each TP=1
# arm was additionally proven to be a CONTIGUOUS RUN of that tool's own
# statements, and a CPU dry-run drove both tools to the engine boundary on one
# synthetic spec and diffed the whole cell plan, the per-cell vectors by sha,
# the lane-tagged band with its seed materials, the parsed arguments and the
# column header — byte for byte identical.

#: sha256 of the CERTIFIED TP=1 tool these digests were taken from.
CERTIFIED_TOOL_SHA256 = (
    "840c3e806fe05d1c23647d76ee72e3f0b6747e0d432fdc04d00a1728119f64d0")

#: The three places the tool forks on `TP > 1`, in source order, each paired
#: with the digest its TP=1 (`else:`) arm must have.
TP1_BRANCH_DIGESTS: tuple[tuple[str, str], ...] = (
    ("attachment",
     "dea14240bd6c8987c4010f67819e123006cb2fd829fa3b395955e7d09ea9f558"),
    ("sec_2_5_reduction",
     "caabcb8e980bb41fae16680cdb5fbe6c69e31b5da0939d2470615c6642a4d01d"),
    ("detach_teardown",
     "224ff0975415e79f9785bbdc284c9dc35db202ef3b7316714250e4f00b652ff0"),
)

#: Every top-level helper the certified tool and this one share. The lane's
#: whole arithmetic lives here — the span rule, the token accounting, the
#: indexing pin, the rank correlation — and none of it moved.
CERTIFIED_HELPER_DIGESTS: dict[str, str] = {
    "EngineConfigRefused":
        "d67a01180a65c6c04820af9b5ef4b45c7479d0118d769e7904f90d0a5aa6a95f",
    "IndexingUnpinned":
        "865a8776c508abab1bab641354bbc16d673032662cc327cc5faf6cf1ccd376b8",
    "LaneColumnRefused":
        "4e5d68f76e1bcc2c19a9a04a5e43c2aca84819ccf5b15d738da3b7367b9c0747",
    "SpanAccountingRefused":
        "bba4f7c1a7a126d329f88062c7e858a171f32b93f4b061d2367b9e654dcbfa78",
    "_average_ranks":
        "2196c3dbe2345ce84a5465a33f63247f8f77a2acf3fc3ed57165b26aaf53fe69",
    "assert_span_accounting":
        "66f7740ca661f9a59c14a3e1db211bd6acdc5fbdf3a21bdbe55bba4de64512f6",
    "band_min_max":
        "f79959aa08db2ea7eab9b59974f4537f495a257e00998eeb9141cb2fce2c45c2",
    "expected_generation_tokens":
        "23b83bf5bb8a0ca3a1672b62f7f487cbc83f75cd55ebdb830cac1f53b7651338",
    "pin_indexing":
        "30c34a4c41f591208f7ffdfb80c658df3d770502d5d6f562c2a7d36a3d6d81be",
    "sha256_file":
        "36c9b9cd0767506de4b59b5ca3820a9017d5e8ca94f18ae606e1034772c2e6e5",
    "span_mask_from_positions":
        "c11ce0072ff09c9eaeee4bf696eea73cc293748c6c2ec8e857820d12cb062bf2",
    "spearman_rho":
        "8866de435255646a19aa67241c026872e187132ef037698e9768e23bf4b75226",
}
# `build_arg_parser` is deliberately ABSENT from that map: §6(b) retires
# `--allow-any-host`, so its AST must differ. It is checked by option name
# instead (`ARGPARSE_CONTRACT`), which is the contract a runner actually
# depends on.

#: The named kwargs of the single `LLM(...)` construction. At TP=1 the
#: conditional `**{…}` contributes an EMPTY dict, so this IS the certified call.
CERTIFIED_ENGINE_KWARGS: frozenset[str] = frozenset({
    "disable_log_stats", "dtype", "enable_chunked_prefill",
    "enable_prefix_caching", "enforce_eager", "gpu_memory_utilization",
    "max_model_len", "model", "tensor_parallel_size",
})

#: The tool's full option surface. 22 of the certified tool's 23 options are
#: unchanged; `--allow-any-host` is retired and `--allowed-host`,
#: `--require-env` and `--code-root` replace it with caller-supplied data.
ARGPARSE_CONTRACT: frozenset[str] = frozenset({
    "--allowed-host", "--arm", "--code-root", "--corpus-sha",
    "--engine-sec25-norm", "--expect-pool-sha256", "--gpu-memory-utilization",
    "--include", "--lane-tag", "--limit-cells", "--matched-absolute-alpha",
    "--max-model-len", "--max-new-tokens", "--model-path", "--n-per-cell",
    "--node-key", "--out-dir", "--prompt-pool", "--require-env", "--resume",
    "--site", "--source-key", "--tensor-parallel-size", "--transported-npz",
    "--vectors-npz",
})

#: Every path and identity the run needs, required rather than defaulted — a
#: repo module points at nothing on its own.
REQUIRED_ARGS: frozenset[str] = frozenset({
    "arm", "corpus_sha", "model_path", "node_key", "out_dir", "prompt_pool",
    "site", "vectors_npz",
})

#: The cell-id templates, as `ast.unparse` renders them. Cell ids are the join
#: key against the BANKED byte-exact columns, so a changed template produces a
#: column that silently matches nothing.
CELL_ID_TEMPLATES: frozenset[str] = frozenset({
    "f'baseline_L{site}_a+0.00'",
    "f'{native_key}_L{site}_a{frac:+.2f}'",
    "f'Rband{i}_L{site}_a{frac:+.2f}'",
    "f'gentropy_gradient_L{site}_a{frac:+.2f}'",
    "f'gRband{i}_L{site}_a{frac:+.2f}'",
    "f'naive_entropy_gradient_L{site}_a{frac:+.2f}'",
    "f'naive_entropy_gradient_L{site}_a{d:+.2f}'",
    "f'{native_key}_L{site}_a{frac:+.2f}@absalpha'",
})

#: The same grammar as a regex, so the ids those templates PRODUCE are checked
#: and not only the templates themselves.
CELL_ID_GRAMMAR = (
    r"(?:baseline|entropy_gradient_L\d+|Rband[123]|gentropy_gradient"
    r"|gRband[123]|naive_entropy_gradient)_L\d+_a[+-]\d\.\d{2}(?:@absalpha)?")

#: The frozen signed ladder, mirrored here ONLY so the grammar can be exercised
#: without importing the engine; the selftest asserts it equals `DOSE_LADDER`.
DOSE_LADDER_FOR_GRAMMAR: tuple[float, ...] = (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3)

#: The per-generation seed material, as `ast.unparse` renders it. It decides
#: which sampling seed every (cell, generation_id) draws; a reordered field is
#: a different column wearing the same id.
LANE_SEED_MATERIAL_TEMPLATE = (
    "f'{args.corpus_sha}|{node_key}|{arm}|L{site}|{cid}|{g:03d}'")
LANE_SEED_MATERIAL_TEMPLATE_DIGEST = (
    "aadb0342d934d7126ff81e54d542fad5c500877398c9f053fe811a4b1385cdb4")

#: The RULED random-band seed template's digest (`build_behavioral_banks`).
#: The lane tags `vector_key`, which is a FIELD of this template — the template
#: itself is never edited, and this pin is how that stays true.
RULED_BAND_TEMPLATE_DIGEST = (
    "994c4fbc2a4576372b8339788b270c015f3fe3f095252fb53338f4b003c82fcc")

# ── refusals ──────────────────────────────────────────────────────────────────

class LaneColumnRefused(RuntimeError):
    """A named refusal. This runner never banks a number it could not prove."""


class SpanAccountingRefused(LaneColumnRefused):
    """The wrapper's own token counts disagree with the batch's known shape."""


class IndexingUnpinned(LaneColumnRefused):
    """The entropy index convention could not be decided by value."""


class EngineConfigRefused(LaneColumnRefused):
    """A correctness-bearing engine flag is not what the pre-statement pins."""


class LaneCodeRootUnresolved(LaneColumnRefused):
    """`metabasis` is not importable, so the column would run against nothing."""


# ── typed documents (pydantic; nothing here is a bare dict on the wire) ───────

class DeploymentGuard(BaseModel):
    """WHICH guard actually ran, recorded on the column rather than assumed.

    The certified tool hardcoded one hostname and one environment variable. In
    the repo both are arguments, and both default to EMPTY — the lane's runner
    scripts already enforce the same two constraints one layer up, and a repo
    module that names a node is a sanitization problem, not a safety feature.
    An empty guard is therefore legitimate AND is stamped as empty, so a reader
    of a column never has to guess whether a guard was applied or merely
    assumed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(description="socket.gethostname() as observed at start")
    allowed_hosts: tuple[str, ...] = Field(
        default=(), description="--allowed-host; empty means the tool imposes none")
    required_env: tuple[str, ...] = Field(
        default=(), description="--require-env; empty means the tool imposes none")
    enforced_by_this_tool: bool = Field(
        description="False when BOTH lists are empty — the runner is the guard")
    blocked_reason: Optional[str] = Field(
        default=None, description="the refusal text, or None when the guard passed")


class RankAttachReport(BaseModel):
    """One rank's answer to `mb_attach`, validated before it is believed.

    `collective_rpc` returns whatever the far side built. A malformed or partial
    report from one rank would otherwise be read positionally and turn into a
    silently part-steered column, which is the exact failure the TP path exists
    to prevent — so the reports are parsed, not indexed.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int
    hidden_dim: int
    site: int
    wrapped_type: str
    n_layers_seen: Optional[int] = None


class ColumnIdentityPins(BaseModel):
    """The §1 identity additions re-freeze #4 puts on EVERY column stamp.

    Two lanes that differ only in `tensor_parallel_size` are the same experiment
    only if the pins say so by value. `VLLM_ENABLE_V1_MULTIPROCESSING` is pinned
    at its recorded value rather than dropped: the four certification columns ran
    with it at 0 (the investigator's own correction), and a clean V1MP=1 column
    has never been produced, so 0 is the environment of record and not an
    inference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_parallel_size: int
    attachment: str
    vllm_worker_multiproc_method: Optional[str] = None
    vllm_enable_v1_multiprocessing: Optional[str] = None
    vllm_version: str
    torch_version: str
    module_sha256: dict[str, str]
    reading: str = (
        "the §1 identity pins re-freeze #4 adds. `spawn` is REQUIRED at TP>1 "
        "(fork from an OMP-dirtied parent hangs before any worker line) and "
        "INERT at TP=1 (UniProcExecutor forks nothing); the module shas are the "
        "instrument's identity, and the two certified extraction modules are "
        "byte-unchanged from their certification.")


def apply_deployment_guard(allowed_hosts: Sequence[str],
                           required_env: Sequence[str]) -> DeploymentGuard:
    """The repo-neutral replacement for the certified tool's `--allow-any-host`.

    NO DEFAULT ALLOW-LIST AND NO DEFAULT ENVIRONMENT REQUIREMENT, because either
    would be a node name or a scheduler name compiled into the repo. What the
    certified tool expressed as a literal hostname comparison — refuse unless
    `socket.gethostname()` equals one particular machine — is expressed here as
    a list the caller supplies, and the RESULT is stamped either way. The
    selftest asserts by AST that no comparison of `gethostname()` against a
    string literal survives anywhere in this module.
    """
    host = socket.gethostname()
    allowed = tuple(str(h) for h in allowed_hosts)
    needed = tuple(str(v) for v in required_env)
    reason: Optional[str] = None
    if allowed and host not in allowed:
        reason = (f"host {host!r} is not in the caller's allow-list "
                  f"{list(allowed)} — this lane runs where its runner says it "
                  "runs, and nowhere else")
    if reason is None:
        for var in needed:
            if not os.environ.get(var):
                reason = (f"the required environment variable {var} is unset or "
                          "empty; the caller declared it a precondition of this "
                          "run (typically: this is not a scheduler job)")
                break
    return DeploymentGuard(host=host, allowed_hosts=allowed, required_env=needed,
                           enforced_by_this_tool=bool(allowed or needed),
                           blocked_reason=reason)


# ── small helpers (pure, selftested) ──────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def span_mask_from_positions(positions: Any, torch_mod: Any, *,
                             prompt_lengths: Optional[frozenset[int]] = None
                             ) -> Any:
    """generated-positions-only for a GENERATION forward, from the batch's layout.

    With chunked prefill OFF a request's prefill arrives whole. Runs are split
    where `positions` does not increase (a new request begins). A run that starts
    at absolute 0 and is longer than one token is a PREFILL run and is injected
    nowhere; anything else is decode and is injected. A single token at absolute
    0 is ambiguous (a 1-token prompt, or a mis-segmented batch) and REFUSES.

    THE ABSORPTION HAZARD, CLOSED. Segmentation splits only where `positions`
    fails to increase, so a prefill run `[0..P-1]` immediately followed by a
    DECODE token at some position q > P-1 does not split — the decode token gets
    absorbed into the prefill run and is silently NOT injected. `prompt_lengths`
    closes it: a prefill run's length must be a length some request in this batch
    actually has, and an absorbed decode token makes it P+1, which is not.
    Refusing a mixed layout we cannot segment is correct; silently under-injecting
    is not. (The end-of-cell accounting gate would also catch it, but a forward is
    the right place to find out.)
    """
    pos = positions.reshape(-1)
    n = int(pos.numel())
    mask = torch_mod.zeros(n, dtype=torch_mod.bool, device=pos.device)
    host = pos.detach().to("cpu").tolist()
    starts: list[int] = [0]
    for i in range(1, n):
        if host[i] <= host[i - 1]:
            starts.append(i)
    starts.append(n)
    for a, b in zip(starts[:-1], starts[1:]):
        first = host[a]
        if first == 0 and (b - a) > 1:
            if prompt_lengths is not None and (b - a) not in prompt_lengths:
                raise SpanAccountingRefused(
                    f"a prefill run of {b - a} tokens is not the length of any "
                    f"request in this batch (lengths {sorted(prompt_lengths)}) — "
                    "the batch mixes a prefill with a following decode token in a "
                    "way this segmentation cannot split, and injecting under a "
                    "layout we cannot read would silently skip generated positions")
            continue
        if first == 0 and (b - a) == 1:
            raise SpanAccountingRefused(
                "ambiguous run: a single token at absolute position 0 is either a "
                "1-token prompt or a mis-segmented batch; the injection span is "
                "not guessable and this lane refuses to guess it")
        mask[a:b] = True
    return mask


def expected_generation_tokens(prompt_lens: Sequence[int],
                               n_generated: Sequence[int],
                               finish_reasons: Sequence[str]) -> dict[str, int]:
    """What a GENERATION pass must have put through the model, per request.

    TWO BOOKKEEPING FACTS, BOTH MEASURED ON THIS STACK RATHER THAN ASSUMED.

    (1) THE LAST SAMPLED TOKEN IS NEVER AN INPUT. A request with prompt length P
        whose engine sampled s tokens is forwarded as one prefill of P tokens
        (producing sampled token 1) and s−1 single-token decode steps (producing
        tokens 2..s). Token s is sampled from the last step's logits and never
        fed back, so the model SEES P + (s−1) tokens and the write can only apply
        to s−1 generated positions. HF's `generate` has the same structure, so
        the lanes agree; the teacher-forced scoring pass replays the whole
        sequence and is where every generated position is seen, on both lanes.

    (2) vLLM's `CompletionOutput.token_ids` EXCLUDES THE TERMINATING EOS. So the
        REPORTED length n is not the SAMPLED length s: a request that stopped on
        EOS sampled s = n+1 (fed back n), and a request that stopped on
        max_tokens sampled s = n (fed back n−1).

        MEASURED, NOT READ OFF A CHANGELOG. Four independent runs, gated by the
        strict `seen == ΣP + Σ(n−1)` form and refused by it, pinned the residual
        exactly at the number of NON-EOS-terminated requests:
          qwen2.5-3b L26   ΣP=3216 seen=43109 -> decode 39893, Σn=39962, gap 69/80
          qwen-7b   L21    ΣP=3216 seen=43203 -> decode 39987, Σn=40049, gap 62/80
          qwen2.5-7b-base  ΣP= 896 seen=31516 -> decode 30620, Σn=30656, gap 36/80
          smoke (8 reqs, max_new=64, no EOS)  -> decode   504, Σn=  512, gap  8/8
        A gap that is exactly "one per request that did NOT stop on EOS" is fact
        (2); nothing else in the range is consistent with all four.
    """
    if not (len(prompt_lens) == len(n_generated) == len(finish_reasons)):
        raise SpanAccountingRefused(
            f"{len(prompt_lens)} prompts, {len(n_generated)} generations, "
            f"{len(finish_reasons)} finish reasons — ragged")
    fed_back, sampled = [], []
    for g, fr in zip(n_generated, finish_reasons):
        n = int(g)
        stopped_on_token = str(fr) == "stop"
        s = n + 1 if stopped_on_token else n
        sampled.append(s)
        fed_back.append(max(s - 1, 0))
    return {"prompt_tokens": int(sum(int(p) for p in prompt_lens)),
            "generated_tokens_reported": int(sum(int(g) for g in n_generated)),
            "generated_tokens_sampled": int(sum(sampled)),
            "n_stopped_on_token": int(sum(1 for f in finish_reasons
                                          if str(f) == "stop")),
            "n_stopped_on_length": int(sum(1 for f in finish_reasons
                                           if str(f) != "stop")),
            "generated_tokens_fed_back": int(sum(fed_back)),
            "expected_seen": int(sum(int(p) for p in prompt_lens) + sum(fed_back))}


def assert_span_accounting(*, seen: int, injected: int, skipped: int,
                           expect: dict[str, int], n_requests: int, where: str
                           ) -> dict[str, int]:
    """The span rule PROVEN for one cell — two STRUCTURAL gates and one bounded.

    THE TWO STRUCTURAL GATES ARE THE SPAN RULE, and they depend on no reporting
    convention whatsoever:
      * every prompt token was seen exactly once and NONE was injected — the
        failure that would write into a PROMPT;
      * every non-prompt token that was seen WAS injected — the failure that
        would silently skip a generated position.
    Both are exact equalities and both refuse.

    THE THIRD IS BOOKKEEPING, AND IT IS BOUNDED RATHER THAN EXACT — deliberately.
    It reconciles what the model saw against what vLLM REPORTS in `token_ids` +
    `finish_reason`, and that reconciliation carries a per-request ±1 that is
    vLLM's to define, not ours: the terminating EOS is excluded from `token_ids`,
    and at least one termination path (measured: qwen2.5-7b-base L20 raw, cell
    `entropy_gradient_L20_L20_a-0.30`, 51 stop / 29 length, residual exactly −1)
    does not carry the extra sampled token the EOS rule predicts. Making this an
    exact gate would block the science on a question about vLLM's reporting;
    making it absent would lose a real check. So it refuses only when the
    residual exceeds one token per request — the range no reporting convention
    can explain — and the exact residual is RECORDED on every cell either way.
    """
    problems = []
    if skipped != expect["prompt_tokens"]:
        problems.append(
            f"STRUCTURAL: prefill_tokens_skipped {skipped} != prompt tokens "
            f"{expect['prompt_tokens']} — a prompt token was injected, or a "
            "prompt token was not seen")
    if injected != seen - expect["prompt_tokens"]:
        problems.append(
            f"STRUCTURAL: tokens_injected {injected} != seen {seen} − prompts "
            f"{expect['prompt_tokens']} = {seen - expect['prompt_tokens']} — a "
            "generated position the model saw was not written")
    residual = int(seen - expect["expected_seen"])
    if abs(residual) > int(n_requests):
        problems.append(
            f"BOOKKEEPING: tokens_seen {seen} differs from the reported-token "
            f"expectation {expect['expected_seen']} by {residual}, which exceeds "
            f"one token per request ({n_requests}) — that is larger than any "
            "termination-reporting convention can account for, so it is read as "
            "a span question, not a bookkeeping one")
    if problems:
        raise SpanAccountingRefused(
            f"{where}: span accounting did not close — " + "; ".join(problems)
            + ". The span rule is 'generated positions only'; a disagreement here "
              "means the mask and the batch disagree about which tokens are which.")
    return {"tokens_seen": seen, "tokens_injected": injected,
            "prefill_tokens_skipped": skipped, **expect,
            "n_requests": int(n_requests),
            "bookkeeping_residual_tokens": residual,
            "bookkeeping_residual_per_request": round(residual / max(n_requests, 1), 6),
            "reading": ("STRUCTURAL: no prompt token was injected and no generated "
                        "position the model saw was skipped — both exact. The final "
                        "sampled token of each request is never fed back and so is "
                        "never a write site, the same structure the byte-exact lane "
                        "has. A terminating EOS is excluded from the reported "
                        "token_ids, so it is neither scored nor written; the mean "
                        "entropy this cell reports is over the REPORTED generated "
                        "tokens. `bookkeeping_residual_tokens` is the leftover "
                        "between what the model saw and what vLLM reported, in "
                        "tokens; it is bounded by one per request and is a fact "
                        "about vLLM's termination reporting, not about the span.")}


def pin_indexing(inference: dict[str, Any], *, margin: float = 3.0,
                 max_abs_nats: float = 0.25) -> str:
    """Decide the entropy index convention from `infer_indexing`'s VALUES.

    Two guards, because "the smaller number wins" is not a pin: the winner must
    beat the loser by `margin` and must itself be small in absolute terms. The
    measured separation on this stack is ~12x (0.53 vs 6.25 nats), so a run that
    does not clear 3x is telling us something changed.
    """
    got = {k: v.get("max_abs_diff") for k, v in inference.items()
           if isinstance(v, dict) and v.get("max_abs_diff") is not None}
    if len(got) < 2:
        raise IndexingUnpinned(
            f"only {sorted(got)} could be evaluated; both 'position' and "
            "'produced' must be measurable before either can be pinned")
    win = min(got, key=lambda k: got[k])
    lose = max(got, key=lambda k: got[k])
    if got[lose] < margin * max(got[win], 1e-9):
        raise IndexingUnpinned(
            f"the two conventions are not separated by value: {got} — the winner "
            f"must beat the loser by {margin}x before this lane will call it a "
            "convention rather than a coincidence")
    if got[win] > max_abs_nats:
        raise IndexingUnpinned(
            f"the best convention {win!r} still disagrees by {got[win]:.4f} nats "
            f"(> {max_abs_nats}); that is not an index question and pinning it "
            "would bank a misaligned array")
    return win


def _average_ranks(values: Sequence[float]) -> "list[float]":
    """Ranks with TIES AVERAGED — scipy's `rankdata` default, transcribed.

    The isolated lane venv has no scipy (verified by value on the lane node,
    2026-08-08),
    and `actuation_calibration.dose_ordering` — the frozen §4.2(a) reader — uses
    `scipy.stats.spearmanr`. Rather than let the two lanes compute a slightly
    different ρ, the arithmetic is written out here and CROSS-CHECKED against
    scipy wherever scipy exists (the desk), so the number the lane files is the
    number the desk's frozen reader would produce.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, for the LADDER ingredient only (never a bar decision)."""
    import numpy as np
    a = [float(x) for x in xs]
    b = [float(y) for y in ys]
    if len(a) != len(b) or len(a) < 3:
        return None
    ra = np.asarray(_average_ranks(a), dtype=np.float64)
    rb = np.asarray(_average_ranks(b), dtype=np.float64)
    da, db = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if den == 0.0:
        return None
    rho = float((da * db).sum() / den)
    return rho if np.isfinite(rho) else None


def band_min_max(members: Sequence[Optional[float]]) -> Optional[tuple[float, float]]:
    """`score_entropy_writes.band`'s arithmetic: [min, max] over present members."""
    import numpy as np
    vals = [float(m) for m in members if m is not None and np.isfinite(float(m))]
    if not vals:
        return None
    return (min(vals), max(vals))


# ── the runner ────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--node-key", required=True,
                    help="the campaign's node key (drives band seed material)")
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=("native", "raw"),
                    help="EXPLICIT. Hardcoded native rendering is a refusal case "
                         "(the base rung renders raw); the addendum names this.")
    ap.add_argument("--corpus-sha", required=True,
                    help="the basis the banked band was anchored to")
    ap.add_argument("--vectors-npz", type=Path, required=True,
                    help="the BANKED calibration bank (entropy_gradient_L<site>)")
    ap.add_argument("--transported-npz", type=Path, default=None,
                    help="the BANKED transported bank (gentropy_gradient, "
                         "naive_entropy_gradient); enables the B3/B4 cells")
    ap.add_argument("--source-key", default=None,
                    help="the hub/source node key — gRband draws in ITS space")
    ap.add_argument("--prompt-pool", type=Path, required=True)
    ap.add_argument("--expect-pool-sha256", default=None)
    ap.add_argument("--lane-tag", default="@vllm-lane",
                    help="pre-statement §3's lane-tagged seed material; rides "
                         "vector_key, which is a field of the RULED template")
    ap.add_argument("--n-per-cell", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--include", default="calibration",
                    help="comma list of blocks: calibration,transported,absbeside")
    ap.add_argument("--matched-absolute-alpha", type=float, default=None,
                    help="the labeled BESIDE (ruling ②): the native ladder re-run "
                         "at the ENGINE's absolute alpha at +0.3. Never gates.")
    ap.add_argument("--engine-sec25-norm", type=float, default=None,
                    help="the banked column's own §2.5 norm, for the delta beside")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit-cells", type=int, default=None,
                    help="SMOKE ONLY: run the first N planned cells")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already banked under --out-dir")
    # REPO-NEUTRAL GUARD (re-freeze #4, Luxia's ruling 2026-08-20 §6(b)). The
    # certified tool carried `--allow-any-host`, whose whole meaning was "skip
    # the one hostname compiled into me" — and a repo module cannot carry the
    # hostname that flag existed to escape. Both halves of that guard are now
    # caller-supplied lists that default to empty, and the lane's runner
    # scripts — which already enforce the same two constraints one layer up —
    # pass them.
    ap.add_argument("--allowed-host", action="append", default=[],
                    metavar="HOST", dest="allowed_host",
                    help="repeatable host allow-list; EMPTY (the default) means "
                         "this tool imposes no host guard and the runner's guard "
                         "is the guard. Whatever is passed is stamped on the "
                         "column under `deployment_guard`.")
    ap.add_argument("--require-env", action="append", default=[],
                    metavar="VAR", dest="require_env",
                    help="repeatable list of environment variables that must be "
                         "set and non-empty; EMPTY (the default) imposes none. "
                         "The node runners pass their scheduler's job-id "
                         "variable here.")
    ap.add_argument("--code-root", type=Path, default=None,
                    help="directory prepended to sys.path so `metabasis` "
                         "imports; falls back to $MB_CODE. Neither is required "
                         "when the package is already importable, and an "
                         "unimportable `metabasis` is a named refusal rather "
                         "than a silent partial run.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:                # noqa: C901
    args = build_arg_parser().parse_args(argv)

    guard = apply_deployment_guard(args.allowed_host, args.require_env)
    if guard.blocked_reason is not None:
        print(f"BLOCKED: {guard.blocked_reason}", file=sys.stderr)
        return 2

    code = args.code_root or os.environ.get("MB_CODE")
    if code:
        sys.path.insert(0, str(code))
    try:
        import metabasis                                          # noqa: F401
    except ImportError as exc:                                    # no fallback
        refusal = LaneCodeRootUnresolved(
            f"the metabasis package is not importable from code root {code!r} "
            f"({exc}). A column run against a half-resolved package would bank "
            "numbers produced by whatever else happened to be on sys.path, so "
            "this refuses rather than proceeding")
        print(f"BLOCKED: {refusal}", file=sys.stderr)
        return 2
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import numpy as np
    import torch

    # ---- lane modules + their selftests, BEFORE any GPU ----------------------
    from metabasis.extraction import vllm_residual_write as vrw
    from metabasis.extraction import vllm_entropy_probe as vep
    for mod in (vrw, vep):
        rc = mod.selftest()
        if rc != 0:
            print(f"BLOCKED: {mod.__name__}.selftest() -> {rc}", file=sys.stderr)
            return 2

    from metabasis.extraction.vllm_residual_write import (
        MetabasisSteeringSpec, attach_steering_layer,
        measure_per_token_median_resid_norm_from_captures, resolve_alpha)
    from metabasis.extraction.vllm_entropy_probe import (
        EntropyCapture, attach_entropy_capture, entropy_rise, infer_indexing,
        slice_generated, cross_check_generation_vs_scoring)
    from metabasis.scripts.run_behavioral_cells import (
        DOSE_LADDER, SCORING_DOSES, load_prompt_pool, render_prompt, seed_int)
    from metabasis.scripts.build_behavioral_banks import (
        BAND_RECIPE_OF_RECORD, RANDOM_BAND_SEED_TEMPLATE,
        RANDOM_BAND_SEED_TEMPLATE_DIGEST, RandomBandRecipe, build_random_band)
    from metabasis.scripts.capability_battery import coherence_panel

    site, arm, node_key = int(args.site), args.arm, args.node_key
    blocks = {b.strip() for b in args.include.split(",") if b.strip()}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)

    doc: dict[str, Any] = {
        "kind": "vllm-lane-bridge-column",
        "grade": C8,
        "role_boundary": ("INGREDIENTS ONLY. No bar is scored and no criterion "
                          "threshold is applied here; thresholds appear only as "
                          "`desk_context_*` fields, beside the raw quantities they "
                          "would be applied to. The desk scores (C§8)."),
        "node_key": node_key, "site": site, "arm": arm,
        "model_path": args.model_path,
        "heimdall_job_id": os.environ.get("HEIMDALL_JOB_ID", "(unset)"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": socket.gethostname(),
        "deployment_guard": guard.model_dump(mode="json"),
        "torch": torch.__version__,
        "n_per_cell": args.n_per_cell, "max_new_tokens": args.max_new_tokens,
        "dose_ladder": list(DOSE_LADDER), "scoring_doses": list(SCORING_DOSES),
        "blocks": sorted(blocks),
        "corpus_sha": args.corpus_sha,
        "lane_tag": args.lane_tag,
        "band_recipe_of_record": BAND_RECIPE_OF_RECORD,
        "band_seed_template": RANDOM_BAND_SEED_TEMPLATE,
        "band_seed_template_digest": RANDOM_BAND_SEED_TEMPLATE_DIGEST,
        "entropy_instrument": (
            "BOTH halves teacher-forced scoring (prompt_logprobs), one sequence "
            "per call — the path certified at MEAN level on 2026-08-08 (unsteered "
            "cell-mean agreement 0.14% relative). Per-position quantiles are "
            "recorded BESIDE and never gate."),
    }

    # ---- prompt pool ---------------------------------------------------------
    pool = load_prompt_pool(args.prompt_pool)
    doc["prompt_pool_path"] = str(args.prompt_pool)
    doc["prompt_pool_sha256"] = pool.sha256
    if args.expect_pool_sha256 and pool.sha256 != args.expect_pool_sha256:
        print(f"BLOCKED: prompt pool sha {pool.sha256} != expected "
              f"{args.expect_pool_sha256} — the banked column's pool is the pool "
              "of record and a different pool is a different experiment",
              file=sys.stderr)
        return 2
    n = int(args.n_per_cell)
    if n > len(pool.prompts):
        print(f"BLOCKED: n_per_cell {n} exceeds pool size {len(pool.prompts)}",
              file=sys.stderr)
        return 2

    # ---- vectors -------------------------------------------------------------
    doc["vectors_npz"] = str(args.vectors_npz)
    doc["vectors_npz_sha256"] = sha256_file(args.vectors_npz)
    with np.load(args.vectors_npz, allow_pickle=True) as z:
        vecs = {k: np.asarray(z[k]).reshape(-1).astype(np.float32) for k in z.files}
    native_key = f"entropy_gradient_L{site}"
    if native_key not in vecs:
        print(f"BLOCKED: {args.vectors_npz} has no {native_key!r} (keys "
              f"{sorted(vecs)}) — the lane column writes the SAME banked vector "
              "the byte-exact column wrote, never a re-derived one", file=sys.stderr)
        return 2
    dim = int(vecs[native_key].size)
    doc["hidden_dim_from_vector"] = dim

    tvecs: dict[str, "np.ndarray"] = {}
    if "transported" in blocks:
        if args.transported_npz is None or args.source_key is None:
            print("BLOCKED: --include transported needs --transported-npz and "
                  "--source-key (gRband draws in the SOURCE's space)",
                  file=sys.stderr)
            return 2
        doc["transported_npz"] = str(args.transported_npz)
        doc["transported_npz_sha256"] = sha256_file(args.transported_npz)
        doc["source_key"] = args.source_key
        with np.load(args.transported_npz, allow_pickle=True) as z:
            tvecs = {k: np.asarray(z[k]).reshape(-1).astype(np.float32)
                     for k in z.files}
        for need in ("gentropy_gradient", "naive_entropy_gradient"):
            if need not in tvecs:
                print(f"BLOCKED: transported bank has no {need!r} (keys "
                      f"{sorted(tvecs)})", file=sys.stderr)
                return 2

    # ---- the LANE-TAGGED bands (ruled recipe, unedited) ----------------------
    lane_native_band_vector_key = f"{native_key}{args.lane_tag}"
    r_recipe = RandomBandRecipe(
        corpus_sha=args.corpus_sha, node_key=node_key, arm=arm, site=site,
        band_kind="Rband", vector_key=lane_native_band_vector_key,
        draw_space_key=node_key, dim=dim)
    rband = build_random_band(r_recipe, "Rband")
    doc["lane_band_recipe_Rband"] = {
        "vector_key": lane_native_band_vector_key,
        "draw_space_key": node_key,
        "seed_material": [r_recipe.seed_material(i) for i in (1, 2, 3)],
        "reading": ("pre-statement §3: bands drawn under the RULED recipe with "
                    "LANE-TAGGED seed material. The recipe object is the frozen "
                    "one, imported unedited; the tag rides vector_key, a field of "
                    "the ruled template. Distinct draws from the banked band — "
                    "§4.2(b) is a WITHIN-column control either way."),
    }
    band_l2 = {k: float(np.linalg.norm(v)) for k, v in rband.items()}
    doc["lane_band_member_l2"] = band_l2
    banked_band_cos = {}
    for k, v in rband.items():
        if k in vecs:
            a, b = np.asarray(v, np.float64), np.asarray(vecs[k], np.float64)
            banked_band_cos[k] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    doc["lane_vs_banked_band_cosine"] = banked_band_cos

    gband: dict[str, "np.ndarray"] = {}
    if "transported" in blocks:
        # THE TRANSPORTED BAND IS NOT RE-DRAWN, AND THAT IS THE RULED RECIPE
        # SPEAKING, NOT A SHORTCUT. `RandomBandRecipe` draws a gRband in the
        # SOURCE's space and §5.1 then carries it through the SAME fitted map as
        # the signal — "the control travels the same road as the signal or it is
        # not a control for transport". The fit is not on this lane's path, so a
        # lane-tagged re-draw could only produce an isotropic in the TARGET's
        # space, which is precisely the object the recipe's own validator refuses
        # by name ("a native band wearing a transported name"). The banked
        # gRband1..3 — already carried through the map — are therefore used as
        # the matched objects they are, and this deviation from §3's
        # "lane-tagged seed material" is named here rather than quietly taken.
        for i in (1, 2, 3):
            if f"gRband{i}" not in tvecs:
                print(f"BLOCKED: transported bank has no gRband{i}", file=sys.stderr)
                return 2
            gband[f"gRband{i}"] = tvecs[f"gRband{i}"]
        doc["lane_band_recipe_gRband"] = {
            "construction": "BANKED gRband1..3, re-used verbatim (not re-drawn)",
            "named_deviation_from_prestatement_3": (
                "§3 says bands are drawn under the ruled recipe with LANE-TAGGED "
                "seed material. That is constructible for the native Rband and is "
                "done. It is NOT constructible for the gRband: the ruled recipe "
                "draws it in the SOURCE's space and §5.1 carries it through the "
                "fitted transport map, which this lane does not run. Re-drawing "
                "isotropically in the target's space would be the object the "
                "recipe's own `_draw_space_matches_the_family` validator refuses "
                "by name. The banked members are used instead, which makes the "
                "transported control a MATCHED-OBJECT comparison."),
            "source_key": args.source_key,
            "member_l2": {k: float(np.linalg.norm(v)) for k, v in gband.items()},
        }

    # ---- the cell plan (ids EXACTLY the banked ones) -------------------------
    plan: list[dict[str, Any]] = []
    plan.append({"cell_id": f"baseline_L{site}_a+0.00", "kind": "baseline",
                 "vector_key": None, "alpha_frac": 0.0, "band_family": None,
                 "vector": None, "alpha_mode": "baseline"})
    if "calibration" in blocks:
        for frac in DOSE_LADDER:
            plan.append({"cell_id": f"{native_key}_L{site}_a{frac:+.2f}",
                         "kind": "calibration", "vector_key": native_key,
                         "alpha_frac": float(frac), "band_family": None,
                         "vector": vecs[native_key], "alpha_mode": "frac"})
        for i in (1, 2, 3):
            for frac in DOSE_LADDER:
                plan.append({"cell_id": f"Rband{i}_L{site}_a{frac:+.2f}",
                             "kind": "calibration_band", "vector_key": f"Rband{i}",
                             "alpha_frac": float(frac), "band_family": "Rband",
                             "vector": rband[f"Rband{i}"], "alpha_mode": "frac"})
    if "transported" in blocks:
        for frac in DOSE_LADDER:
            plan.append({"cell_id": f"gentropy_gradient_L{site}_a{frac:+.2f}",
                         "kind": "transported", "vector_key": "gentropy_gradient",
                         "alpha_frac": float(frac), "band_family": None,
                         "vector": tvecs["gentropy_gradient"], "alpha_mode": "frac"})
        for i in (1, 2, 3):
            for frac in DOSE_LADDER:
                plan.append({"cell_id": f"gRband{i}_L{site}_a{frac:+.2f}",
                             "kind": "transported_band", "vector_key": f"gRband{i}",
                             "alpha_frac": float(frac), "band_family": "gRband",
                             "vector": gband[f"gRband{i}"], "alpha_mode": "frac"})
        for frac in SCORING_DOSES:
            plan.append({"cell_id": f"naive_entropy_gradient_L{site}_a{frac:+.2f}",
                         "kind": "naive", "vector_key": "naive_entropy_gradient",
                         "alpha_frac": float(frac), "band_family": None,
                         "vector": tvecs["naive_entropy_gradient"],
                         "alpha_mode": "frac"})
    if "absbeside" in blocks:
        if args.matched_absolute_alpha is None:
            print("BLOCKED: --include absbeside needs --matched-absolute-alpha "
                  "(the ENGINE's absolute alpha at +0.3)", file=sys.stderr)
            return 2
        for frac in DOSE_LADDER:
            plan.append({
                "cell_id": f"{native_key}_L{site}_a{frac:+.2f}@absalpha",
                "kind": "calibration_beside_absolute_alpha",
                "vector_key": native_key, "alpha_frac": float(frac),
                "band_family": None, "vector": vecs[native_key],
                "alpha_mode": "absolute",
                "beside_note": ("ruling ②'s labeled BESIDE: the same ladder at the "
                                "ENGINE's ABSOLUTE alpha instead of the lane's own "
                                "norm-relative one. Decomposes dose-vs-"
                                "implementation. NEVER gates, never pooled.")})
    if args.limit_cells:
        plan = plan[:int(args.limit_cells)]
    doc["planned_cells"] = [c["cell_id"] for c in plan]
    doc["n_planned_cells"] = len(plan)

    # ---- the engine ----------------------------------------------------------
    import vllm
    from vllm import LLM, SamplingParams
    doc["vllm"] = vllm.__version__

    # ---- the §1 identity pins re-freeze #4 adds (on EVERY column) ------------
    doc["identity_pins"] = ColumnIdentityPins(
        tensor_parallel_size=int(args.tensor_parallel_size),
        attachment=("worker_extension_cls (per-rank), driven by collective_rpc"
                    if int(args.tensor_parallel_size) > 1
                    else "in-process (UniProcExecutor driver_worker) — the "
                         "certified TP=1 attachment"),
        vllm_worker_multiproc_method=os.environ.get(
            "VLLM_WORKER_MULTIPROC_METHOD"),
        vllm_enable_v1_multiprocessing=os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"),
        vllm_version=vllm.__version__,
        torch_version=torch.__version__,
        module_sha256=instrument_module_sha256(),
    ).model_dump(mode="json")

    t_load = time.perf_counter()
    llm = LLM(model=args.model_path, dtype="bfloat16",
              tensor_parallel_size=int(args.tensor_parallel_size),
              enforce_eager=True,
              gpu_memory_utilization=float(args.gpu_memory_utilization),
              max_model_len=int(args.max_model_len), disable_log_stats=True,
              enable_chunked_prefill=False, enable_prefix_caching=False,
              # TP-ONLY BESIDE TOOL (desk, 2026-08-20). At TP>1 the model lives in
              # worker PROCESSES, so the certified in-process attachment cannot
              # reach it (`_find_model` refuses: MultiprocExecutor has no
              # driver_worker). `worker_extension_cls` mixes the metabasis
              # instrument into the Worker class INSIDE every rank
              # (vllm/v1/worker/worker_base.py:262-287) and the parent drives it
              # by name through collective_rpc. Inert at TP=1.
              **({"worker_extension_cls": WORKER_EXTENSION_CLS}
                 if int(args.tensor_parallel_size) > 1 else {}))
    doc["engine_load_s"] = round(time.perf_counter() - t_load, 2)

    # FOOTGUN 3, BY VALUE off the live config rather than off our own kwargs.
    cfg = getattr(getattr(llm, "llm_engine", llm), "vllm_config", None)
    sched = getattr(cfg, "scheduler_config", None)
    cache = getattr(cfg, "cache_config", None)
    observed = {
        "enable_chunked_prefill": getattr(sched, "chunked_prefill_enabled",
                                          getattr(sched, "enable_chunked_prefill",
                                                  None)),
        "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
        "enforce_eager": getattr(getattr(cfg, "model_config", None),
                                 "enforce_eager", None),
    }
    doc["engine_flags_observed"] = observed
    for flag in ("enable_chunked_prefill", "enable_prefix_caching"):
        if observed[flag] is None:
            raise EngineConfigRefused(
                f"could not READ {flag} off the live engine config — the span "
                "rule's correctness requirement cannot be trusted to a kwarg that "
                "was never read back (vllm "
                f"{vllm.__version__}); observed={observed}")
        if bool(observed[flag]):
            raise EngineConfigRefused(
                f"{flag} is ON. Pre-statement §1: prefix caching and chunked "
                "prefill are OFF as a CORRECTNESS requirement of the span rule — "
                "a cache hit means the prompt's leading tokens are not recomputed "
                "and the span detector would inject into the PROMPT.")

    TP = int(args.tensor_parallel_size)
    if TP > 1:
        # ---- TP>1: the instrument lives in the workers -----------------------
        from metabasis.extraction.vllm_tp_proxy import (TPCapProxy, TPStateProxy,
                                                        TPWrapperProxy)
        att = llm.collective_rpc("mb_attach", args=(site,))
        # Parsed, not indexed: a malformed or partial report from one rank read
        # positionally is how a part-steered column would look normal.
        rank_reports = [RankAttachReport.model_validate(a) for a in att]
        doc["rank_attach_reports"] = [r.model_dump() for r in rank_reports]
        if len(att) != TP:
            raise LaneColumnRefused(
                f"the worker extension answered from {len(att)} ranks but "
                f"tensor_parallel_size is {TP} -- the instrument did not reach "
                "every rank and a column would be silently part-steered")
        dims = {int(a["hidden_dim"]) for a in att}
        if len(dims) != 1:
            raise LaneColumnRefused(
                f"ranks disagree on the site's hidden width {dims}")
        hidden = dims.pop()
        if hidden != dim:
            raise LaneColumnRefused(
                f"model hidden_dim {hidden} != banked vector dim {dim}")
        wrapper = TPWrapperProxy(llm, hidden, att[0]["wrapped_type"])
        cap = TPCapProxy(llm)
        state = TPStateProxy(llm)
        model = None
        orig_forward = None
        detach_ent = lambda: llm.collective_rpc("mb_detach")
        doc["attached"] = {"site": site, "hidden_dim": hidden,
                           "wrapped_type": att[0]["wrapped_type"],
                           "attachment": "worker_extension_cls (per-rank)",
                           "ranks": [int(a["rank"]) for a in att],
                           "tensor_parallel_size": TP}
    else:
        from vllm_lane_smoke import _find_model
        model = _find_model(llm)
        wrapper = attach_steering_layer(model, site)
        if int(wrapper.hidden_dim) != dim:
            raise LaneColumnRefused(
                f"model hidden_dim {wrapper.hidden_dim} != banked vector dim {dim}")
        cap = EntropyCapture()
        detach_ent = attach_entropy_capture(model, cap)
        doc["attached"] = {"site": site, "hidden_dim": int(wrapper.hidden_dim),
                           "wrapped_type": type(wrapper.wrapped_layer).__name__}

        orig_forward = model.forward
        state: dict[str, Any] = {"mode": "generation", "prompt_len": None,
                                 "mask_true": 0, "mask_n": 0,
                                 "prompt_lengths": None}

        def shim(*a: Any, **k: Any) -> Any:
            positions = k.get("positions")
            if positions is None and len(a) >= 2 and isinstance(a[1], torch.Tensor):
                positions = a[1]
            if positions is None:
                wrapper.set_position_mask(None)
            else:
                if state["mode"] == "scoring":
                    # TEACHER-FORCED REPLAY: the generated span is the tail of the
                    # replayed sequence, so injectability is absolute position against
                    # THAT sequence's own prompt_length — the same rule generation
                    # used, applied to a scoring pass.
                    pl = state["prompt_len"]
                    if pl is None:
                        m = torch.zeros_like(positions.reshape(-1), dtype=torch.bool)
                    else:
                        m = positions.reshape(-1) >= int(pl)
                else:
                    m = span_mask_from_positions(
                        positions, torch, prompt_lengths=state["prompt_lengths"])
                state["mask_true"] += int(m.sum().item())
                state["mask_n"] += int(m.numel())
                wrapper.set_position_mask(m)
            return orig_forward(*a, **k)

        model.forward = shim

    def spec_for(vector: Optional["np.ndarray"], alpha: float,
                 alpha_frac: Optional[float], measured: float
                 ) -> Optional[MetabasisSteeringSpec]:
        if vector is None or alpha == 0.0:
            return None
        return MetabasisSteeringSpec(
            layer_idx=site,
            vector=torch.as_tensor(vector.reshape(-1), dtype=torch.float32),
            alpha=float(alpha), alpha_frac=alpha_frac, measured_norm=measured)

    def score_sequence(full_ids: list[int], prompt_len: int, n_gen: int,
                       spec: Optional[MetabasisSteeringSpec], indexing: str
                       ) -> "np.ndarray":
        """One teacher-forced scoring pass -> the generated span's entropies."""
        wrapper.set_spec(spec)
        state["mode"] = "scoring"
        state["prompt_len"] = prompt_len
        cap.reset("scoring")
        llm.generate([{"prompt_token_ids": full_ids}],
                     SamplingParams(max_tokens=1, temperature=0.0,
                                    prompt_logprobs=0))
        ent_all = cap.flat()[:len(full_ids)]
        return slice_generated(ent_all, prompt_len, n_gen, indexing=indexing)

    cells_out: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        # ---- §2.5 on the LANE'S OWN measurement (the dose invariant) ---------
        prompts = list(pool.prompts)[:n]
        tok = llm.get_tokenizer()
        rendered = [list(render_prompt(p, tok, arm)) for p in prompts]
        if not all(isinstance(t, int) for t in rendered[0]):
            raise LaneColumnRefused(
                "render_prompt did not return ints — the prompt object is not a "
                "string and this lane never hands vLLM a stringified mapping")
        doc["rendered_prompt_lens"] = [len(r) for r in rendered]
        doc["rendered_first_ids"] = rendered[0][:24]
        state["prompt_lengths"] = frozenset(len(r) for r in rendered)

        wrapper.set_spec(None)
        wrapper.capture_input_hidden_states(True)
        state["mode"] = "generation"
        llm.generate([{"prompt_token_ids": r} for r in rendered],
                     SamplingParams(max_tokens=1, temperature=0.0))
        if TP > 1:
            # the captures are (tokens x d_model) activations; the certified
            # reduction runs ON the rank and only the scalar crosses.
            med = wrapper.measure_sec25()
            wrapper.capture_input_hidden_states(False)
            import math as _math
            if med != med or _math.isnan(med):
                raise LaneColumnRefused("the wrapper captured nothing at the site")
        else:
            caps = wrapper.take_captures()
            wrapper.capture_input_hidden_states(False)
            if not caps:
                raise LaneColumnRefused("the wrapper captured nothing at the site")
            med = measure_per_token_median_resid_norm_from_captures(caps)
        doc["lane_sec_2_5_norm"] = float(med)
        doc["lane_sec_2_5_norm_hex"] = float(med).hex()
        doc["sec_2_5_provenance"] = (
            f"MEASURED in-job at L{site} over the first {n} prompts of the frozen "
            f"pool ({pool.sha256[:12]}…) rendered in the {arm} arm, per-token "
            "residual norms, median (§2.5) — the LANE'S OWN measurement, which is "
            "what §1's norm-relative dose invariant requires")
        if args.engine_sec25_norm:
            e = float(args.engine_sec25_norm)
            doc["sec_2_5_cross_lane"] = {
                "engine": e, "lane": float(med),
                "rel_delta": abs(float(med) - e) / e,
                "beside_reading": (
                    "the cross-lane §2.5 norm delta of record. B3 is computed at "
                    "MATCHED alpha_frac per §1's norm-relative invariant, so this "
                    "delta is INTERNAL to the dose definition and is quoted beside "
                    "every B3 row rather than corrected out.")}

        # ---- pin the entropy index convention BY VALUE, this column ----------
        pin_prompt = rendered[0]
        pin_frac = 0.3
        pin_alpha = resolve_alpha(pin_frac, med)
        pin_spec = spec_for(vecs[native_key], pin_alpha, pin_frac, med)
        wrapper.set_spec(pin_spec)
        state["mode"] = "generation"
        state["mask_true"] = state["mask_n"] = 0
        cap.reset("generation")
        pin_out = llm.generate([{"prompt_token_ids": pin_prompt}],
                               SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                                              max_tokens=48, seed=20260808))
        gen_ent = cap.flat()
        pin_ids = list(pin_out[0].outputs[0].token_ids)
        P, NG = len(pin_prompt), len(pin_ids)
        if gen_ent.size != NG:
            raise IndexingUnpinned(
                f"generation-time capture produced {gen_ent.size} scalars for "
                f"{NG} generated tokens; the reference for the pin must be "
                "one-per-generated-token or it is not a reference")
        wrapper.set_spec(pin_spec)
        state["mode"] = "scoring"
        state["prompt_len"] = P
        cap.reset("scoring")
        llm.generate([{"prompt_token_ids": pin_prompt + pin_ids}],
                     SamplingParams(max_tokens=1, temperature=0.0,
                                    prompt_logprobs=0))
        score_all = cap.flat()[:P + NG]
        inference = infer_indexing(score_all, gen_ent, P, NG)
        indexing = pin_indexing(inference)
        doc["indexing"] = indexing
        doc["indexing_inference"] = inference
        doc["indexing_pin_reading"] = (
            "pinned BY VALUE, per column, against a reference this module made "
            "itself: the generation-time entropies of the SAME steered sequence. "
            "Two conventions exist and they differ by exactly one; the wrong one "
            "is silent (right length, plausible means). An unknown or unseparated "
            "convention is a refusal, never a default.")
        doc["generation_vs_scoring_crosscheck"] = cross_check_generation_vs_scoring(
            gen_ent, slice_generated(score_all, P, NG, indexing=indexing),
            atol=0.25)

        # ---- the cells -------------------------------------------------------
        for ci, cell in enumerate(plan):
            cid = cell["cell_id"]
            cdir = out / "cells" / cid
            if args.resume and (cdir / "cell.json").exists():
                cells_out.append(json.loads((cdir / "cell.json").read_text()))
                print(f"[{ci+1}/{len(plan)}] {cid}: RESUMED from disk", flush=True)
                continue
            frac = float(cell["alpha_frac"])
            if cell["alpha_mode"] == "baseline":
                alpha = 0.0
            elif cell["alpha_mode"] == "absolute":
                alpha = (frac / 0.3) * float(args.matched_absolute_alpha)
            else:
                alpha = resolve_alpha(frac, med)
            # the absolute-alpha beside is NOT alpha_frac x lane norm, so the
            # spec's own §2.5 consistency check must not be handed a frac.
            spec = spec_for(cell["vector"], alpha,
                            None if cell["alpha_mode"] == "absolute" else frac,
                            med)

            wrapper.set_spec(spec)
            wrapper.reset_stats()
            state["mode"] = "generation"
            state["mask_true"] = state["mask_n"] = 0
            sps = []
            for g in range(n):
                material = (f"{args.corpus_sha}|{node_key}|{arm}|L{site}|{cid}|"
                            f"{g:03d}")
                sps.append(SamplingParams(
                    temperature=1.0, top_p=1.0, top_k=-1,
                    max_tokens=int(args.max_new_tokens),
                    seed=int(seed_int(material) % (2 ** 31 - 1))))
            t0 = time.perf_counter()
            outs = llm.generate([{"prompt_token_ids": r} for r in rendered], sps)
            gen_wall = time.perf_counter() - t0

            gen_ids = [list(o.outputs[0].token_ids) for o in outs]
            texts = [o.outputs[0].text for o in outs]
            finish = [str(getattr(o.outputs[0], "finish_reason", "")) for o in outs]
            n_gen_tokens = int(sum(len(g) for g in gen_ids))
            expect = expected_generation_tokens([len(r) for r in rendered],
                                                [len(g) for g in gen_ids], finish)
            accounting = assert_span_accounting(
                seen=int(wrapper.stats["tokens_seen"]),
                injected=(int(wrapper.stats["tokens_injected"]) if spec is not None
                          else int(state["mask_true"])),
                skipped=(int(wrapper.stats["prefill_tokens_skipped"])
                         if spec is not None
                         else int(state["mask_n"]) - int(state["mask_true"])),
                expect=expect, n_requests=n, where=f"{cid} (generation)")

            # ---- the two teacher-forced scoring passes ----------------------
            t1 = time.perf_counter()
            rows: list[dict[str, Any]] = []
            ent_s_arrays: dict[str, "np.ndarray"] = {}
            ent_u_arrays: dict[str, "np.ndarray"] = {}
            short = 0
            for g in range(n):
                ng = len(gen_ids[g])
                if ng < 1:
                    short += 1
                    continue
                full = rendered[g] + gen_ids[g]
                es = score_sequence(full, len(rendered[g]), ng, spec, indexing)
                eu = score_sequence(full, len(rendered[g]), ng, None, indexing)
                ent_s_arrays[f"steered_{g:04d}"] = es
                ent_u_arrays[f"unsteered_{g:04d}"] = eu
                rows.append({"generation_id": g, "n_positions": int(ng),
                             "mean_entropy_steered": float(np.mean(es)),
                             "mean_entropy_unsteered": float(np.mean(eu)),
                             "entropy_rise": float(np.mean(es) - np.mean(eu))})
            probe_wall = time.perf_counter() - t1
            if not rows:
                raise LaneColumnRefused(f"{cid}: no generation produced a token")

            ent = entropy_rise([r["mean_entropy_steered"] for r in rows],
                               [r["mean_entropy_unsteered"] for r in rows])
            panel = coherence_panel(texts, [f == "stop" for f in finish])
            coh = json.loads(panel.model_dump_json())

            # per-position quantiles: the BESIDE the certification carries
            all_s = np.concatenate([v for v in ent_s_arrays.values()])
            all_u = np.concatenate([v for v in ent_u_arrays.values()])
            quant = {
                "steered": {q: float(np.quantile(all_s, p)) for q, p in
                            (("median", .5), ("p90", .9), ("max", 1.0))},
                "unsteered": {q: float(np.quantile(all_u, p)) for q, p in
                              (("median", .5), ("p90", .9), ("max", 1.0))},
                "n_positions": int(all_s.size),
                "reading": ("per-position dispersion, recorded BESIDE. The "
                            "certification of record is MEAN-level (2026-08-08); "
                            "these never gate."),
            }

            cdir.mkdir(parents=True, exist_ok=True)
            np.savez(cdir / "entropy.npz", **ent_s_arrays, **ent_u_arrays)
            with open(cdir / "generations.jsonl", "w") as fh:
                for g in range(n):
                    fh.write(json.dumps({
                        "generation_id": g,
                        "prompt_id": prompts[g].prompt_id,
                        "prompt_length": len(rendered[g]),
                        "n_generated": len(gen_ids[g]),
                        "finish_reason": finish[g],
                        "token_ids": gen_ids[g], "text": texts[g]}) + "\n")

            rec = {
                "cell_id": cid, "kind": cell["kind"],
                "vector_key": cell["vector_key"],
                "band_family": cell["band_family"],
                "alpha_frac": frac, "alpha": float(alpha),
                "alpha_mode": cell["alpha_mode"],
                "n": n, "n_scored": len(rows), "n_empty_generations": short,
                "tokens_generated": n_gen_tokens,
                "generation_wall_s": round(gen_wall, 3),
                "probe_wall_s": round(probe_wall, 3),
                "generation_tok_per_s": round(n_gen_tokens / gen_wall, 1),
                "mean_generated_tokens": round(n_gen_tokens / n, 2),
                "entropy": ent,
                "per_position_quantiles_beside": quant,
                "coherence": coh,
                "span_accounting": accounting,
                "entropy_npz_sha256": sha256_file(cdir / "entropy.npz"),
                "generations_sha256": sha256_file(cdir / "generations.jsonl"),
                "grade": C8,
            }
            if "beside_note" in cell:
                rec["beside_note"] = cell["beside_note"]
            rec["probe_rows"] = rows
            (cdir / "cell.json").write_text(
                json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
            cells_out.append(rec)
            print(f"[{ci+1}/{len(plan)}] {cid}: rise={ent['entropy_rise']:+.4f} "
                  f"dwr={coh['distinct_word_ratio']:.4f} "
                  f"gen={gen_wall:.1f}s probe={probe_wall:.1f}s "
                  f"tok/s={n_gen_tokens/gen_wall:.0f}", flush=True)
    except Exception as exc:                                      # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
        print(f"LANE-COLUMN-ERROR {errors[-1]}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if TP > 1:
            detach_ent()
        else:
            model.forward = orig_forward
            detach_ent()

    doc["cells"] = cells_out
    doc["n_cells_banked"] = len(cells_out)
    doc["errors"] = errors

    # ---- the bar INGREDIENTS, assembled (no bar is scored) ------------------
    by_id = {c["cell_id"]: c for c in cells_out}

    def rise_of(cid: str) -> Optional[float]:
        c = by_id.get(cid)
        return None if c is None else float(c["entropy"]["entropy_rise"])

    def ladder(prefix_fn) -> dict[str, Optional[float]]:
        return {f"{f:+g}": rise_of(prefix_fn(f)) for f in DOSE_LADDER}

    ing: dict[str, Any] = {"role_boundary": doc["role_boundary"]}

    if "calibration" in blocks:
        rises = ladder(lambda f: f"{native_key}_L{site}_a{f:+.2f}")
        bands = {}
        for f in DOSE_LADDER:
            members = [rise_of(f"Rband{i}_L{site}_a{f:+.2f}") for i in (1, 2, 3)]
            mm = band_min_max(members)
            bands[f"{f:+g}"] = {
                "band_members": members,
                "band_min": None if mm is None else mm[0],
                "band_max": None if mm is None else mm[1],
                "entropy_rise": rises[f"{f:+g}"],
                "strictly_outside_own_lane_Rband": (
                    None if mm is None or rises[f"{f:+g}"] is None
                    else bool(rises[f"{f:+g}"] < mm[0] or rises[f"{f:+g}"] > mm[1])),
                "lever_distinct_word_ratio": (
                    by_id.get(f"{native_key}_L{site}_a{f:+.2f}", {})
                    .get("coherence", {}).get("distinct_word_ratio")),
            }
        base = by_id.get(f"baseline_L{site}_a+0.00", {})
        scoring = by_id.get(f"{native_key}_L{site}_a+0.30", {})
        seven = [0.0] + [f for f in DOSE_LADDER]
        seven_means = ([base.get("entropy", {}).get("mean_entropy_steered")]
                       + [by_id.get(f"{native_key}_L{site}_a{f:+.2f}", {})
                          .get("entropy", {}).get("mean_entropy_steered")
                          for f in DOSE_LADDER])
        ing["a_dose_ordering"] = {
            "entropy_rise_by_dose": rises,
            "doses_present": sum(1 for v in rises.values() if v is not None),
            "doses_expected": len(DOSE_LADDER),
            "spearman_rho_dose_vs_entropy_rise": spearman_rho(
                [f for f in DOSE_LADDER if rises[f"{f:+g}"] is not None],
                [rises[f"{f:+g}"] for f in DOSE_LADDER
                 if rises[f"{f:+g}"] is not None]),
            "sign_flips_through_zero": (
                any((v or 0) < 0 for k, v in rises.items() if k.startswith("-"))
                and any((v or 0) > 0 for k, v in rises.items()
                        if k.startswith("+"))),
            "desk_context_floor": 0.80,
        }
        ing["b_band_separation"] = {
            "band_family_of_record": "Rband",
            "band_convention": ("strictly outside [min, max] over the dose's three "
                                "OWN-Rband members (score_entropy_writes.band's "
                                "arithmetic); an absent band is a hole, never a pass"),
            "lane_band_note": ("this column's Rband is a LANE-TAGGED draw under the "
                               "ruled recipe (§3), not the banked members — §4.2(b) "
                               "is a within-column control"),
            "per_dose": bands,
            "n_doses_strictly_outside": sum(
                1 for v in bands.values()
                if v["strictly_outside_own_lane_Rband"]),
            "of_doses": len(DOSE_LADDER),
            "outside_at_both_scoring_doses": all(
                bands[f"{d:+g}"]["strictly_outside_own_lane_Rband"]
                for d in SCORING_DOSES),
            "desk_context_min_doses": 4,
        }
        ing["c_coherence"] = {
            "distinct_word_ratio_at_scoring_dose":
                scoring.get("coherence", {}).get("distinct_word_ratio"),
            "baseline_distinct_word_ratio":
                base.get("coherence", {}).get("distinct_word_ratio"),
            "band_distinct_word_ratio_at_scoring_dose": [
                by_id.get(f"Rband{i}_L{site}_a+0.30", {})
                .get("coherence", {}).get("distinct_word_ratio") for i in (1, 2, 3)],
            "four_gram_repetition_rate":
                scoring.get("coherence", {}).get("four_gram_repetition_rate"),
            "eos_termination_rate":
                scoring.get("coherence", {}).get("eos_termination_rate"),
            "mean_generation_length_words":
                scoring.get("coherence", {}).get("mean_generation_length_words"),
            "scoring_dose": 0.3, "desk_context_floor": 0.45,
            "provenance": "metabasis.scripts.capability_battery.coherence_panel",
        }
        ing["b2_seven_point_ladder"] = {
            "doses": seven, "mean_entropy_steered": seven_means,
            "reading": ("B2's ladder: the α=0 baseline plus the frozen signed "
                        "6-dose ladder, MEAN entropy (not rise), for the "
                        "cross-lane Spearman the desk computes per side."),
        }
        ing["baseline"] = {
            "mean_entropy_unsteered":
                base.get("entropy", {}).get("mean_entropy_unsteered"),
            "mean_entropy_steered":
                base.get("entropy", {}).get("mean_entropy_steered"),
        }

    if "transported" in blocks:
        trises = ladder(lambda f: f"gentropy_gradient_L{site}_a{f:+.2f}")
        gbands = {}
        for f in DOSE_LADDER:
            members = [rise_of(f"gRband{i}_L{site}_a{f:+.2f}") for i in (1, 2, 3)]
            mm = band_min_max(members)
            gbands[f"{f:+g}"] = {
                "gRband_members": members,
                "gRband_min": None if mm is None else mm[0],
                "gRband_max": None if mm is None else mm[1],
                "transported_entropy_rise": trises[f"{f:+g}"],
                "strictly_outside_lane_gRband": (
                    None if mm is None or trises[f"{f:+g}"] is None
                    else bool(trises[f"{f:+g}"] < mm[0]
                              or trises[f"{f:+g}"] > mm[1])),
            }
        ing["b3_transported"] = {
            "entropy_rise_by_dose": trises,
            "at_scoring_doses": {f"{d:+g}": trises[f"{d:+g}"] for d in SCORING_DOSES},
            "matched_at": "alpha_frac (the stamped §1 norm-relative invariant)",
            "per_dose_vs_gRband": gbands,
            "beside_required": ("quote the cross-lane §2.5 norm delta beside every "
                                "row of this block"),
        }
        ing["b4_naive"] = {
            f"{d:+g}": {
                "cell_id": f"naive_entropy_gradient_L{site}_a{d:+.2f}",
                "entropy_rise": rise_of(f"naive_entropy_gradient_L{site}_a{d:+.2f}"),
                "distinct_word_ratio": by_id.get(
                    f"naive_entropy_gradient_L{site}_a{d:+.2f}", {})
                    .get("coherence", {}).get("distinct_word_ratio"),
                "lane_gRband_min": gbands[f"{d:+g}"]["gRband_min"],
                "lane_gRband_max": gbands[f"{d:+g}"]["gRband_max"],
                "transported_entropy_rise": trises[f"{d:+g}"],
            } for d in SCORING_DOSES}

    if "absbeside" in blocks:
        ing["matched_absolute_alpha_beside"] = {
            "label": "BESIDE — never gates, never pooled (ruling ②)",
            "engine_absolute_alpha_at_+0.30": args.matched_absolute_alpha,
            "entropy_rise_by_dose": ladder(
                lambda f: f"{native_key}_L{site}_a{f:+.2f}@absalpha"),
        }

    tot_tok = sum(c["tokens_generated"] for c in cells_out)
    tot_gen = sum(c["generation_wall_s"] for c in cells_out)
    tot_probe = sum(c["probe_wall_s"] for c in cells_out)
    ing["throughput_length_matched"] = {
        "total_generated_tokens": tot_tok,
        "generation_wall_s": round(tot_gen, 2),
        "probe_wall_s": round(tot_probe, 2),
        "generation_tok_per_s": (round(tot_tok / tot_gen, 1) if tot_gen else None),
        "mean_generated_tokens_per_generation": (
            round(tot_tok / sum(c["n"] for c in cells_out), 2) if cells_out else None),
        "per_cell": {c["cell_id"]: {
            "tokens_generated": c["tokens_generated"],
            "generation_wall_s": c["generation_wall_s"],
            "generation_tok_per_s": c["generation_tok_per_s"],
            "mean_generated_tokens": c["mean_generated_tokens"]}
            for c in cells_out},
        "reading": ("LENGTH-MATCHED by construction: per-cell tokens AND wall are "
                    "both reported, so a rate is only ever read against the token "
                    "count that produced it. The banked column's per-cell "
                    "tokens_generated/elapsed_s pair with these cell-for-cell. The "
                    "raw 2200-vs-292.5 headline stays unquotable."),
    }
    doc["ingredients"] = ing

    (out / "column_ingredients.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"LANE-COLUMN-DONE cells={len(cells_out)}/{len(plan)} "
          f"errors={len(errors)} -> {out/'column_ingredients.json'}")
    return 0 if not errors else 2


# ── selftest (desk-side, CPU, no vLLM, no weights) ───────────────────────────

def selftest() -> int:                                            # noqa: C901
    checks = 0
    fails: list[str] = []

    def ok(cond: bool, name: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(f"{name}{(' — ' + detail) if detail else ''}")

    import numpy as np

    class _T:
        bool = bool
        @staticmethod
        def zeros(n, dtype=None, device=None):
            return np.zeros(n, dtype=np.bool_)

    class _Pos:
        def __init__(self, vals): self.v = np.asarray(vals)
        def reshape(self, *a): return self
        def numel(self): return int(self.v.size)
        def detach(self): return self
        def to(self, *a, **k): return self
        def tolist(self): return list(int(x) for x in self.v)
        @property
        def device(self): return "cpu"

    # -- span mask ------------------------------------------------------------
    m = span_mask_from_positions(_Pos([0, 1, 2, 3, 0, 1, 2]), _T)
    ok(not m.any(), "a two-request all-prefill forward injects nowhere",
       f"{m.tolist()}")
    m = span_mask_from_positions(_Pos([4, 9, 3]), _T)
    ok(bool(m.all()), "an all-decode forward injects everywhere", f"{m.tolist()}")
    m = span_mask_from_positions(_Pos([0, 1, 2, 1]), _T)
    ok(m.tolist() == [False, False, False, True],
       "a prefill run followed by a lower-starting decode token splits correctly",
       f"{m.tolist()}")
    try:
        span_mask_from_positions(_Pos([0, 1, 2, 5]), _T,
                                 prompt_lengths=frozenset({3}))
        ok(False, "the ABSORPTION hazard is refused when prompt lengths are known")
    except SpanAccountingRefused:
        ok(True, "the ABSORPTION hazard is refused when prompt lengths are known")
    m = span_mask_from_positions(_Pos([0, 1, 2, 0, 1, 2]), _T,
                                 prompt_lengths=frozenset({3}))
    ok(not m.any(), "two same-length prefills still pass the length guard")
    try:
        span_mask_from_positions(_Pos([0]), _T)
        ok(False, "a lone token at position 0 REFUSES")
    except SpanAccountingRefused:
        ok(True, "a lone token at position 0 REFUSES")

    # -- span accounting ------------------------------------------------------
    exp = expected_generation_tokens([10, 10], [5, 5], ["length", "length"])
    ok(exp["generated_tokens_fed_back"] == 8 and exp["expected_seen"] == 28,
       "length-terminated: the last sampled token is not fed back", f"{exp}")
    e2 = expected_generation_tokens([10, 10], [5, 5], ["stop", "stop"])
    ok(e2["generated_tokens_sampled"] == 12 and e2["generated_tokens_fed_back"] == 10,
       "EOS-terminated: the unreported EOS was sampled, so n tokens ARE fed back",
       f"{e2}")
    e3 = expected_generation_tokens([10, 10], [5, 5], ["stop", "length"])
    ok(e3["generated_tokens_fed_back"] == 9 and e3["n_stopped_on_token"] == 1,
       "a mixed cell splits per request, never per cell", f"{e3}")
    # the four measured runs, replayed as arithmetic
    for name, P, seen_, sig_n, n_stop, n_req in (
            ("3b", 3216, 43109, 39962, 11, 80),
            ("qwen-7b", 3216, 43203, 40049, 18, 80),
            ("7b-base", 896, 31516, 30656, 44, 80),
            ("smoke", 291, 795, 512, 0, 8)):
        per = sig_n // n_req
        lens = [per] * n_req
        lens[-1] += sig_n - per * n_req
        frs = ["stop"] * n_stop + ["length"] * (n_req - n_stop)
        e = expected_generation_tokens([P // n_req] * n_req, lens, frs)
        ok(e["expected_seen"] - (P // n_req) * n_req + P == seen_,
           f"the measured run {name} closes under the EOS rule",
           f"expected_seen(decode)={e['generated_tokens_fed_back']} "
           f"vs measured {seen_ - P}")
    got = assert_span_accounting(seen=28, injected=8, skipped=20, expect=exp,
                                 n_requests=2, where="t")
    ok(got["tokens_injected"] == 8 and got["bookkeeping_residual_tokens"] == 0,
       "accounting passes when the counts agree")
    got = assert_span_accounting(seen=27, injected=7, skipped=20, expect=exp,
                                 n_requests=2, where="t")
    ok(got["bookkeeping_residual_tokens"] == -1,
       "a residual inside one-per-request passes and is RECORDED, not hidden",
       f"{got['bookkeeping_residual_tokens']}")
    for bad, why in (
            (dict(seen=28, injected=8, skipped=21), "STRUCTURAL: prompt token injected"),
            (dict(seen=28, injected=7, skipped=20), "STRUCTURAL: generated position skipped"),
            (dict(seen=25, injected=5, skipped=20), "residual beyond one per request")):
        try:
            assert_span_accounting(expect=exp, n_requests=2, where="t", **bad)
            ok(False, f"accounting REFUSES: {why}")
        except SpanAccountingRefused:
            ok(True, f"accounting REFUSES: {why}")
    # the structural gates are exact no matter how large the request count is
    try:
        assert_span_accounting(seen=28, injected=8, skipped=21, expect=exp,
                               n_requests=10_000, where="t")
        ok(False, "a huge n_requests does not soften the STRUCTURAL gates")
    except SpanAccountingRefused:
        ok(True, "a huge n_requests does not soften the STRUCTURAL gates")
    try:
        expected_generation_tokens([1, 2], [1], ["length"])
        ok(False, "ragged prompt/generation lists REFUSE")
    except SpanAccountingRefused:
        ok(True, "ragged prompt/generation lists REFUSE")

    # -- the indexing pin -----------------------------------------------------
    ok(pin_indexing({"position": {"max_abs_diff": 6.25},
                     "produced": {"max_abs_diff": 0.05}}) == "produced",
       "a 125x separation pins 'produced'")
    ok(pin_indexing({"position": {"max_abs_diff": 0.02},
                     "produced": {"max_abs_diff": 6.2}}) == "position",
       "the pin is not hardcoded to 'produced'")
    for bad, why in (({"position": {"max_abs_diff": 0.10},
                       "produced": {"max_abs_diff": 0.12}}, "unseparated"),
                     ({"position": {"max_abs_diff": 9.0},
                       "produced": {"max_abs_diff": 1.5}}, "winner too large"),
                     ({"produced": {"max_abs_diff": 0.05}}, "only one convention")):
        try:
            pin_indexing(bad)
            ok(False, f"the pin REFUSES: {why}")
        except IndexingUnpinned:
            ok(True, f"the pin REFUSES: {why}")

    # -- band arithmetic ------------------------------------------------------
    ok(band_min_max([0.1, -0.2, 0.05]) == (-0.2, 0.1), "band is [min, max]")
    ok(band_min_max([None, None, None]) is None, "an absent band is None, not zero")
    ok(band_min_max([0.1, None, 0.3]) == (0.1, 0.3), "a partial band uses what is there")

    # -- spearman -------------------------------------------------------------
    rho = spearman_rho([-0.3, -0.1, -0.03, 0.03, 0.1, 0.3],
                       [-0.5, -0.3, -0.1, 0.1, 0.4, 1.1])
    ok(rho is not None and abs(rho - 1.0) < 1e-12, "a monotone ladder gives rho=1",
       f"{rho}")
    ok(spearman_rho([1, 2], [1, 2]) is None, "fewer than 3 points gives None")
    ok(spearman_rho([1, 2, 3], [5, 5, 5]) is None,
       "a constant ladder gives None, never a spurious 0")
    # CROSS-CHECK against the frozen reader's own implementation where it exists.
    # The lane venv has no scipy; the desk does, and this is where the two are
    # proven to agree rather than assumed to.
    try:
        from scipy.stats import spearmanr                      # type: ignore
    except ImportError:
        print("  (scipy absent — the scipy cross-check is SKIPPED, not passed)")
    else:
        cases = [
            ([-.3, -.1, -.03, .03, .1, .3], [-.5283, -.2924, -.1087, .1244, .4819, 1.1399]),
            ([-.3, -.1, -.03, .03, .1, .3], [-.3398, -.1518, -.0524, .0603, .2459, 1.1908]),
            ([-.3, -.1, -.03, .03, .1, .3], [-.5044, -.281, -.1065, .1199, .4002, .5672]),
            ([-.3, -.1, -.03, .03, .1, .3], [-.2057, -.1332, -.0539, .0673, .3089, 2.1211]),
            ([1., 2., 3., 4., 5.], [2., 2., 3., 1., 9.]),       # ties
            ([1., 2., 3., 4.], [4., 3., 2., 1.]),               # anti-monotone
        ]
        for xs, ys in cases:
            mine = spearman_rho(xs, ys)
            theirs = float(spearmanr(xs, ys).statistic)
            ok(mine is not None and abs(mine - theirs) < 1e-12,
               f"rho matches scipy on {ys[:3]}…", f"{mine} vs {theirs}")

    # -- the role boundary, as a property of the SOURCE ------------------------
    # Scanned ABOVE this function only: a checklist that quotes the names it
    # forbids would otherwise fail against itself.
    runner_src = Path(__file__).read_text().split("\ndef selftest(")[0]
    ok('"verdict"' not in runner_src and "'verdict'" not in runner_src,
       "no emitted key is called `verdict` — the desk scores (C§8)")
    ok("evaluate_actuation" not in runner_src,
       "the §4.2 verdict function is never called from this runner (C§8)")
    ok("gate_transported_cells" not in runner_src,
       "no gate is applied here; this runner produces inputs, not consequences")
    ok(C8.startswith("UNSTAMPED (C§8)"),
       "every artifact this runner writes carries the C§8 grade")

    # ══ re-freeze #4: the TP attachment + the repo-neutral column tool ═══════
    #
    # Everything below is new at re-freeze #4 (Luxia's ruling 2026-08-20 §6).
    # It is CPU-only, data-independent, and torch-free; the two blocks that
    # cannot run without an importable sibling degrade to NAMED SKIPS that say
    # what evidence is missing (rakes M44 + M59(3)).
    import ast
    import inspect
    import re as _re

    skips: list[str] = []

    def skip(name: str, why: str) -> None:
        nonlocal checks
        checks += 1
        skips.append(f"{name} — {why}")

    # RAKE M59(2): the node runs this file as `python <code-root>/metabasis/
    # scripts/vllm_lane_column.py --selftest`, so `sys.path[0]` is the SCRIPTS
    # directory and `import metabasis` fails — four of the checks below would
    # degrade to named skips exactly where the columns actually fire, which is
    # the deployment-disarms-the-proof failure M59 is named for. The package
    # root is resolved off THIS file and PROVEN to exist before it is trusted.
    _root = package_root()
    if (_root / "metabasis" / "__init__.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        ok(True, "(M59) the package root resolved off __file__ and exists, so "
                 "the metabasis-dependent checks below RUN in the node layout "
                 "rather than skipping", str(_root))
    else:
        ok(False, "(M59) the package root resolved off __file__ exists",
           f"{_root} has no metabasis/__init__.py — the checks that need the "
           "engine's frozen ladder and band template will now SKIP")

    _src = Path(__file__).read_text()
    _tree = ast.parse(_src)
    _main = next(n for n in _tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "main")

    def _strip_docstrings(node: ast.AST) -> ast.AST:
        for n in ast.walk(node):
            body = getattr(n, "body", None)
            if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef))
                    and isinstance(body, list) and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                if len(body) > 1:
                    del body[0]
                else:
                    body[0] = ast.Pass()
        return node

    def _digest(stmts: list[ast.stmt]) -> str:
        mod = _strip_docstrings(
            ast.parse(ast.unparse(ast.Module(body=stmts, type_ignores=[]))))
        return hashlib.sha256(ast.unparse(mod).encode()).hexdigest()

    def _is_tp_gt_1(node: Any) -> bool:
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "TP"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Gt)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == 1)

    # ── 1. THE TP=1 IDENTITY, PINNED BY AST (M58: AST, never strings) ────────
    _branches = sorted((n for n in ast.walk(_main) if _is_tp_gt_1(n)),
                       key=lambda n: n.lineno)
    ok(len(_branches) == 3,
       "the TP fork happens in exactly THREE places (attachment, §2.5 "
       "reduction, teardown) — a fourth would be an unpinned divergence",
       f"{len(_branches)}")
    if len(_branches) == 3:
        for (name, want), br in zip(TP1_BRANCH_DIGESTS, _branches):
            got = _digest(br.orelse) if br.orelse else "(no else arm)"
            ok(got == want,
               f"the TP=1 arm of `{name}` is AST-identical to the certified "
               f"tool's own block (source sha {CERTIFIED_TOOL_SHA256[:12]}…) — "
               "`ast.unparse` normalises the extra indentation the `else:` adds, "
               "so this compares CODE, not layout",
               f"{got} vs {want}")

    _shared = {n.name: n for n in _tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for hname, want in sorted(CERTIFIED_HELPER_DIGESTS.items()):
        node = _shared.get(hname)
        if node is None:
            ok(False, f"the shared helper `{hname}` still exists", "missing")
            continue
        ok(_digest([node]) == want,
           f"the shared helper `{hname}` is AST-identical to the certified "
           "tool's — the pure arithmetic the whole lane rests on did not move "
           "when the TP branch was added",
           f"{_digest([node])} vs {want}")

    _llm_calls = [n for n in ast.walk(_main)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "LLM"]
    ok(len(_llm_calls) == 1, "there is exactly ONE engine construction")
    if _llm_calls:
        _star = [k for k in _llm_calls[0].keywords if k.arg is None]
        ok(len(_star) == 1 and isinstance(_star[0].value, ast.IfExp)
           and isinstance(_star[0].value.orelse, ast.Dict)
           and not _star[0].value.orelse.keys,
           "the TP-only engine kwargs are ONE conditional whose TP=1 arm is the "
           "EMPTY dict — at TP=1 the engine is constructed with exactly the "
           "certified kwargs and nothing else",
           ast.unparse(_star[0].value) if _star else "(absent)")
        _named = sorted(k.arg for k in _llm_calls[0].keywords if k.arg)
        ok(_named == sorted(CERTIFIED_ENGINE_KWARGS),
           "the named engine kwargs are exactly the certified set",
           f"{_named}")

    # ── 2. the argparse contract ─────────────────────────────────────────────
    _p = build_arg_parser()
    _opts = {s for a in _p._actions for s in a.option_strings} - {"-h", "--help"}
    ok(_opts == ARGPARSE_CONTRACT,
       "the option set is exactly the pinned contract — a flag added or "
       "removed without updating the contract is caught here, not by a runner "
       "that silently stops passing it",
       f"missing={sorted(ARGPARSE_CONTRACT - _opts)} extra={sorted(_opts - ARGPARSE_CONTRACT)}")
    ok("--allow-any-host" not in _opts,
       "`--allow-any-host` is RETIRED (Luxia's ruling §6(b)) — it existed only "
       "to escape a hostname compiled into the tool, and the tool no longer "
       "carries one")
    _by_dest = {a.dest: a for a in _p._actions}
    for dest in ("allowed_host", "require_env"):
        ok(_by_dest[dest].default == [] and _by_dest[dest].nargs is None,
           f"`{dest}` is a repeatable list that defaults to EMPTY — an empty "
           "guard is the repo default and is STAMPED, never assumed",
           f"{_by_dest[dest].default!r}")
    ok(_by_dest["code_root"].default is None,
       "`--code-root` has no default path — a repo module names no absolute path")
    ok(_by_dest["tensor_parallel_size"].default == 1,
       "`--tensor-parallel-size` still defaults to 1, so every existing runner "
       "invocation means exactly what it meant before re-freeze #4",
       f"{_by_dest['tensor_parallel_size'].default}")
    _required = {a.dest for a in _p._actions if a.required}
    ok(_required == REQUIRED_ARGS,
       "every path and identity the column needs is a REQUIRED argument — "
       "nothing defaults to a location",
       f"{sorted(_required)}")
    ok(tuple(_by_dest["arm"].choices) == ("native", "raw"),
       "`--arm` is still explicit and closed — a hardcoded native rendering is "
       "the addendum's named refusal case")
    for dest in ("vectors_npz", "prompt_pool", "out_dir", "transported_npz",
                 "code_root"):
        ok(_by_dest[dest].type is Path or _by_dest[dest].default is None,
           f"`{dest}` is typed as a Path rather than a bare string")

    # ── 3. the host guard is repo-neutral, and PROVEN so ─────────────────────
    # PROPERTY, NOT A NEEDLE (M58/M51(b)): rather than grepping this file for a
    # node name — which would put the node name IN this file — assert that the
    # module never compares `socket.gethostname()` against a string constant.
    _hostname_calls = 0
    _hostname_compared = 0
    for n in ast.walk(_tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "gethostname"):
            _hostname_calls += 1
        if isinstance(n, ast.Compare):
            parts = [n.left, *n.comparators]
            has_host = any(isinstance(p, ast.Call)
                           and isinstance(p.func, ast.Attribute)
                           and p.func.attr == "gethostname" for p in parts)
            has_str = any(isinstance(p, ast.Constant)
                          and isinstance(p.value, str) for p in parts)
            if has_host and has_str:
                _hostname_compared += 1
    ok(_hostname_calls >= 1,
       "the module still READS the hostname — it is stamped on every column")
    ok(_hostname_compared == 0,
       "the hostname is never compared against a STRING LITERAL anywhere in "
       "this module: the allow-list is data the caller supplies, so no node "
       "name is compiled into the repo",
       f"{_hostname_compared} literal host comparisons")

    _g = apply_deployment_guard([], [])
    ok(_g.blocked_reason is None and _g.enforced_by_this_tool is False,
       "an EMPTY guard passes and says so — the runner's guard is the guard, "
       "and the column records that this tool enforced none")
    ok(_g.host == socket.gethostname(),
       "…and it still records the host it observed")
    _g2 = apply_deployment_guard(["a-host-this-is-not"], [])
    ok(_g2.blocked_reason is not None and _g2.enforced_by_this_tool
       and "allow-list" in _g2.blocked_reason,
       "a host outside the caller's allow-list is BLOCKED, and the refusal "
       "names the allow-list", str(_g2.blocked_reason)[:100])
    _g3 = apply_deployment_guard([socket.gethostname()], [])
    ok(_g3.blocked_reason is None and _g3.enforced_by_this_tool,
       "a host INSIDE the allow-list passes, and the guard is recorded as "
       "enforced")
    _missing_var = "MB_SELFTEST_VAR_THAT_IS_NOT_SET"
    os.environ.pop(_missing_var, None)
    _g4 = apply_deployment_guard([], [_missing_var])
    ok(_g4.blocked_reason is not None and _missing_var in _g4.blocked_reason,
       "a required environment variable that is unset BLOCKS, and the refusal "
       "names the variable", str(_g4.blocked_reason)[:100])
    os.environ[_missing_var] = "1"
    try:
        _g5 = apply_deployment_guard([], [_missing_var])
        ok(_g5.blocked_reason is None,
           "…and a set variable passes")
        os.environ[_missing_var] = ""
        _g6 = apply_deployment_guard([], [_missing_var])
        ok(_g6.blocked_reason is not None,
           "an EMPTY variable is treated as unset — a scheduler that exports "
           "the name with no value has not made this a job")
    finally:
        os.environ.pop(_missing_var, None)
    ok(DeploymentGuard.model_config.get("frozen") is True
       and DeploymentGuard.model_config.get("extra") == "forbid",
       "the guard stamp is frozen and forbids unknown keys — a typo'd field is "
       "a refusal, never a silently dropped value")

    # ── 4. the cell-id grammar ───────────────────────────────────────────────
    _templates = set()
    for n in ast.walk(_main):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if (isinstance(k, ast.Constant) and k.value == "cell_id"
                        and isinstance(v, ast.JoinedStr)):
                    _templates.add(ast.unparse(v))
    ok(_templates == CELL_ID_TEMPLATES,
       "the cell-id grammar is exactly the pinned template set — the ids are "
       "the join key against the BANKED columns, so a changed template is a "
       "silently unmatchable column",
       f"missing={sorted(CELL_ID_TEMPLATES - _templates)} "
       f"extra={sorted(_templates - CELL_ID_TEMPLATES)}")
    _grammar = _re.compile(CELL_ID_GRAMMAR)
    _examples = [
        "baseline_L26_a+0.00", "entropy_gradient_L26_L26_a-0.30",
        "Rband2_L26_a+0.10", "gentropy_gradient_L21_a-0.03",
        "gRband3_L21_a+0.30", "naive_entropy_gradient_L21_a-0.30",
        "entropy_gradient_L26_L26_a+0.30@absalpha",
    ]
    for ex in _examples:
        ok(bool(_grammar.fullmatch(ex)), f"the grammar accepts {ex!r}")
    for bad in ("baseline_L26_a0.00", "Rband4_L26_a+0.10", "Rband2_L26_a+.10",
                "entropy_gradient_L26_L26_a+0.30@absolute", "_L26_a+0.30"):
        ok(not _grammar.fullmatch(bad), f"the grammar REJECTS {bad!r}")
    ok(len(DOSE_LADDER_FOR_GRAMMAR) == 6
       and all(_grammar.fullmatch(f"Rband1_L26_a{f:+.2f}")
               for f in DOSE_LADDER_FOR_GRAMMAR),
       "every dose on the frozen signed ladder renders to a legal cell id",
       f"{DOSE_LADDER_FOR_GRAMMAR}")

    # ── 5. the seed materials ────────────────────────────────────────────────
    _mat = {ast.unparse(n.value) for n in ast.walk(_main)
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "material"
            and isinstance(n.value, ast.JoinedStr)}
    ok(_mat == {LANE_SEED_MATERIAL_TEMPLATE},
       "the per-generation seed material template is pinned — it decides which "
       "prompt every gen_id draws, so a reordered field is a different column "
       "wearing the same id",
       f"{_mat}")
    ok(hashlib.sha256(LANE_SEED_MATERIAL_TEMPLATE.encode()).hexdigest()
       == LANE_SEED_MATERIAL_TEMPLATE_DIGEST,
       "…and its digest is the recorded one",
       LANE_SEED_MATERIAL_TEMPLATE_DIGEST[:16])
    try:
        from metabasis.scripts.build_behavioral_banks import (
            RANDOM_BAND_SEED_TEMPLATE, RANDOM_BAND_SEED_TEMPLATE_DIGEST)
        from metabasis.scripts.run_behavioral_cells import DOSE_LADDER, seed_int
    except ImportError as exc:
        skip("the RULED band seed template digest and the lane's own seed "
             "arithmetic",
             f"the metabasis package is not importable here ({exc}); evidence "
             "MISSING: that this lane's bands are drawn under the RULED "
             "template and that seed_int is reproducible")
    else:
        ok(hashlib.sha256(RANDOM_BAND_SEED_TEMPLATE.encode()).hexdigest()
           == RANDOM_BAND_SEED_TEMPLATE_DIGEST,
           "the RULED band seed template's digest is self-consistent")
        ok(RANDOM_BAND_SEED_TEMPLATE_DIGEST == RULED_BAND_TEMPLATE_DIGEST,
           "…and it is the digest re-freeze #4 pinned — the lane draws its "
           "bands under the ruled recipe, not a lookalike",
           f"{RANDOM_BAND_SEED_TEMPLATE_DIGEST} vs {RULED_BAND_TEMPLATE_DIGEST}")
        ok(tuple(DOSE_LADDER) == DOSE_LADDER_FOR_GRAMMAR,
           "the grammar's ladder IS the engine's frozen ladder, not a copy of it",
           f"{tuple(DOSE_LADDER)}")
        _fixed = ("b85f4d169ed0cb882a5690e3509308a5807e056b6a491f908014aee8e7b6085f"
                  "|a-node|native|L26|baseline_L26_a+0.00|007")
        _s1 = int(seed_int(_fixed) % (2 ** 31 - 1))
        _s2 = int(seed_int(_fixed) % (2 ** 31 - 1))
        ok(_s1 == _s2 and 0 <= _s1 < 2 ** 31 - 1,
           "the per-generation seed is deterministic and in vLLM's range",
           f"{_s1}")
        ok(int(seed_int(_fixed[:-1] + "8") % (2 ** 31 - 1)) != _s1,
           "…and a different generation_id gives a different seed")

    # ── 6. the indexing pin, at its stated 0.25 nats ─────────────────────────
    _sig = inspect.signature(pin_indexing)
    ok(_sig.parameters["max_abs_nats"].default == 0.25,
       "`IndexingUnpinned` fires above 0.25 nats — the ABSOLUTE half of the "
       "pin, pinned as a default rather than as prose",
       f"{_sig.parameters['max_abs_nats'].default}")
    ok(_sig.parameters["margin"].default == 3.0,
       "…and the SEPARATION half is 3x", f"{_sig.parameters['margin'].default}")
    ok(pin_indexing({"position": {"max_abs_diff": 2.5},
                     "produced": {"max_abs_diff": 0.24}}) == "produced",
       "0.24 nats, separated 10x, PINS — just inside the bar")
    try:
        pin_indexing({"position": {"max_abs_diff": 2.6},
                      "produced": {"max_abs_diff": 0.26}})
        ok(False, "0.26 nats REFUSES — just outside the bar")
    except IndexingUnpinned as exc:
        ok("0.25" in str(exc),
           "0.26 nats REFUSES — just outside the bar, and the refusal quotes "
           "the number", str(exc)[:100])

    # ── 7. the span-accounting bookkeeping boundary ──────────────────────────
    _exp = expected_generation_tokens([10] * 4, [5] * 4, ["length"] * 4)
    _base = _exp["expected_seen"]
    _got = assert_span_accounting(seen=_base - 4, injected=_base - 4 - 40,
                                  skipped=40, expect=_exp, n_requests=4,
                                  where="boundary")
    ok(_got["bookkeeping_residual_tokens"] == -4,
       "a residual of exactly one token per request PASSES and is recorded",
       f"{_got['bookkeeping_residual_tokens']}")
    try:
        assert_span_accounting(seen=_base - 5, injected=_base - 5 - 40,
                               skipped=40, expect=_exp, n_requests=4,
                               where="boundary")
        ok(False, "a residual of one-per-request PLUS ONE refuses")
    except SpanAccountingRefused as exc:
        ok("BOOKKEEPING" in str(exc),
           "a residual of one-per-request PLUS ONE refuses, as BOOKKEEPING "
           "rather than as a span question", str(exc)[:80])

    # ── 8. the instrument's identity pins ────────────────────────────────────
    try:
        _shas = instrument_module_sha256()
    except LaneColumnRefused as exc:
        ok(False, "every instrument module is present and hashable", str(exc))
        _shas = {}
    ok(set(_shas) == set(INSTRUMENT_MODULES),
       "the column stamp pins the sha of EVERY module of the instrument",
       f"{sorted(_shas)}")
    for rel, want in sorted(CERTIFIED_MODULE_SHA256.items()):
        ok(_shas.get(rel) == want,
           f"{rel} is BYTE-UNCHANGED at re-freeze #4 — the TP-correctness "
           "argument is that the certified arithmetic was already right",
           f"{_shas.get(rel)} vs {want}")
    _pins = ColumnIdentityPins(
        tensor_parallel_size=2, attachment="worker_extension_cls (per-rank)",
        vllm_worker_multiproc_method="spawn",
        vllm_enable_v1_multiprocessing="0",
        vllm_version="0.15.1", torch_version="2.9.1+cu128",
        module_sha256=dict(_shas))
    _dumped = _pins.model_dump(mode="json")
    for key in ("tensor_parallel_size", "attachment",
                "vllm_worker_multiproc_method", "vllm_enable_v1_multiprocessing",
                "vllm_version", "torch_version", "module_sha256"):
        ok(key in _dumped, f"the identity pins carry `{key}`")
    ok(ColumnIdentityPins.model_config.get("frozen") is True
       and ColumnIdentityPins.model_config.get("extra") == "forbid",
       "the identity pins are frozen and forbid unknown keys")
    try:
        ColumnIdentityPins(tensor_parallel_size=2, attachment="x",
                           vllm_version="a", torch_version="b",
                           module_sha256={}, surprise="?")   # type: ignore[call-arg]
        ok(False, "an UNKNOWN pin key is a refusal")
    except Exception:                                          # noqa: BLE001
        ok(True, "an UNKNOWN pin key is a refusal, never a dropped value")

    # ── 9. the rank attach reports are PARSED, not indexed ───────────────────
    _good = {"rank": 1, "hidden_dim": 3584, "site": 21,
             "wrapped_type": "Qwen2DecoderLayer", "n_layers_seen": None}
    _r = RankAttachReport.model_validate(_good)
    ok(_r.rank == 1 and _r.hidden_dim == 3584 and _r.site == 21,
       "a well-formed rank report validates")
    for bad, why in ((dict(_good, extra="?"), "an unknown key"),
                     ({k: v for k, v in _good.items() if k != "hidden_dim"},
                      "a MISSING hidden_dim"),
                     (dict(_good, hidden_dim="wide"), "a non-integer width")):
        try:
            RankAttachReport.model_validate(bad)
            ok(False, f"a rank report with {why} is REFUSED")
        except Exception:                                      # noqa: BLE001
            ok(True, f"a rank report with {why} is REFUSED — a part-steered "
                     "column must not be able to look normal")

    # ── 10. the TP>1 path only touches names the proxies actually have ───────
    _tp_only = ast.parse(ast.unparse(_main))
    for n in ast.walk(_tp_only):
        if _is_tp_gt_1(n):
            n.orelse = [ast.Pass()]
    _touched: dict[str, set[str]] = {"wrapper": set(), "cap": set()}
    for n in ast.walk(_tp_only):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id in _touched):
            _touched[n.value.id].add(n.attr)
    try:
        from metabasis.extraction.vllm_tp_proxy import TPCapProxy, TPWrapperProxy
    except ImportError as exc:
        skip("the TP>1 code path touches only names the proxies implement",
             f"vllm_tp_proxy is not importable here ({exc}); evidence MISSING: "
             "that no attribute reached at TP>1 is absent from the proxy, which "
             "would be an AttributeError on a loaded model")
    else:
        _wp = TPWrapperProxy(None, 8, "L")
        _cp = TPCapProxy(None)
        for obj, name in ((_wp, "wrapper"), (_cp, "cap")):
            # `hasattr(obj, …)` on a live proxy would CALL the property and
            # fail on the None transport, so the class is consulted first: the
            # question is whether the NAME exists, not whether an RPC works.
            missing = {a for a in _touched[name]
                       if not (hasattr(type(obj), a) or hasattr(obj, a))}
            ok(not missing,
               f"every `{name}.X` the TP>1 path reaches exists on "
               f"{type(obj).__name__} — the surface equality is checked against "
               "the code that USES it, not only against a written contract",
               f"missing: {sorted(missing)}")
        ok(bool(_touched["wrapper"]) and bool(_touched["cap"]),
           "…and the scan actually found attribute uses to check",
           f"wrapper={sorted(_touched['wrapper'])} cap={sorted(_touched['cap'])}")

    # ── 11. the worker extension path resolves ───────────────────────────────
    _mod_path, _, _cls_name = WORKER_EXTENSION_CLS.rpartition(".")
    ok(_mod_path.startswith("metabasis."),
       "`worker_extension_cls` names the REPO module, never a flat module on "
       "PYTHONPATH — a flat name resolves against whatever beside copy the "
       "deployment carries, and one stale rank is a part-steered column",
       WORKER_EXTENSION_CLS)
    try:
        import importlib
        _wext = importlib.import_module(_mod_path)
    except ImportError as exc:
        skip("`worker_extension_cls` resolves to a real class",
             f"{_mod_path} is not importable here ({exc}); evidence MISSING: "
             "that the qualified name vLLM resolves inside every worker exists")
    else:
        _cls = getattr(_wext, _cls_name, None)
        ok(isinstance(_cls, type),
           "`worker_extension_cls` resolves to a real class")
        ok(_cls is not None
           and all(n.startswith("mb_") for n in vars(_cls)
                   if not n.startswith("_")),
           "…whose every public attribute is `mb_`-prefixed (vLLM aborts engine "
           "construction on a Worker attribute collision, worker_base.py:269-275)")

    # ── 12. M59: the suite's own arithmetic ──────────────────────────────────
    SELFTEST_CHECK_FLOOR = 133
    KNOWN_SKIP_CEILING = 4
    ok(checks >= SELFTEST_CHECK_FLOOR and len(skips) <= KNOWN_SKIP_CEILING,
       f"(M59) the suite ran at least its recorded floor of "
       f"{SELFTEST_CHECK_FLOOR} checks and named no more than "
       f"{KNOWN_SKIP_CEILING} skip(s) — a block that stopped running, or "
       "started skipping, is caught here rather than read as a clean run",
       f"{checks} checks (floor {SELFTEST_CHECK_FLOOR}), "
       f"{len(skips)} named skip(s) (ceiling {KNOWN_SKIP_CEILING})")

    print(f"vllm_lane_column selftest: {checks - len(fails)}/{checks} pass "
          f"({len(skips)} named skip(s))")
    for f in fails:
        print(f"  FAIL: {f}")
    for s in skips:
        print(f"  SKIPPED {s}")
    return 0 if not fails else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
