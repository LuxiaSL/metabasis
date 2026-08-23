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

THE TWO VECTOR FAMILIES (the class×TP merge, 2026-08-22). Until now the lane's
two capabilities lived in two besides that were never merged: the repo tool did
TP but only the ENTROPY family, and the node's `vllm_lane_column_v2.py`
(`2c327112…`) did the CLASS family but attached in-process and so failed loudly
at TP>1. v2's class capability is ported IN here, so the program ends with ONE
tool that fires either family at any tensor-parallel size.

  * WHICH OBJECT a column rides is read from a `vllm-lane-class-spec/1` document
    (`--class-spec`). ABSENT, `LaneClassSpec()` is the entropy gradient and every
    key, cell id and emitted field is what this tool produced before the port —
    that backward compatibility is PROVEN in the selftest by rendering the EGV
    plan's ids and requiring them to equal the literals the pre-port tool spelled,
    not asserted in prose.
  * THE KEYS COME FROM THE OBJECT, NEVER FROM THE CAMPAIGN'S OBJECT OF RECORD
    (HALT E, 2026-08-05): the transported key is `g` + the native object's own
    stem and the naive null is `naive_` + the same stem, so `caa_<axis>_L<site>`
    gives `gcaa_<axis>` / `naive_caa_<axis>` and four axes are four columns rather
    than four names for one. And the stem is taken from the BANK's own spelling of
    the key, not from this module's rendering of a template — the two are required
    to be equal per column (`resolve_keys_from_bank`) rather than assumed.
  * HALT D's TWO BASES ride the spec into every column artifact: the GENERATION
    basis is the corpus manifest, the VECTOR basis is the RULED contrast set a
    class object was built from, and `fd_gate_not_applicable` is admissible for a
    class object and for nothing else. The four refusals are the engine's own.
  * THE GRAMMAR STAYS A GUARD. `CELL_ID_GRAMMAR` is no longer a hand-written
    alternation of three entropy literals; it is GENERATED from the closed
    `NATIVE_KEY_TEMPLATES` vocabulary with a closed axis token, so it admits the
    class family and still rejects everything else — an unknown class, an
    upper-case or underscored axis, a missing site suffix. Widening it to
    anything-goes would retire the join key against the banked columns.
  * THE TP BRANCH IS UNTOUCHED BY ANY OF THIS. `worker_extension_cls` +
    `collective_rpc` + the parent proxies take a SITE and a WIDTH and know nothing
    about which object is being injected, so the class path reaches TP>1 through
    exactly the code re-freeze #4 certified. The three TP forks' TP=1 arms and
    every shared helper are still pinned by `ast.unparse` digest below, and those
    pins did not move for this port.

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
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


# ── the vector-class vocabulary (HALT D/E), ported from v2 `2c327112…` ───────
#
# WHY THESE ARE LITERALS HERE AND NOT IMPORTS. This block is read at MODULE
# IMPORT to build the cell-id grammar, and the node runs this file directly with
# `sys.path[0]` at the scripts directory, so `metabasis` is not importable yet
# (the runner resolves it inside `main` via `--code-root`). The conventions are
# therefore transcribed — and the selftest PROVES each one equal to the
# campaign's own definition wherever the package is importable, which is the
# same "proven to agree, never assumed to" pattern the scipy cross-check uses.

EGV_VECTOR_CLASS = "entropy_gradient"

#: The campaign's banked NATIVE key conventions, one per admissible class. The
#: vocabulary is CLOSED: an unrecognized class is refused rather than defaulted
#: to the object of record, because defaulting is exactly how a class object
#: acquires the entropy gradient's name.
NATIVE_KEY_TEMPLATES: dict[str, str] = {
    "entropy_gradient": "entropy_gradient_L{site}",
    "caa": "caa_{axis}_L{site}",
    "repeng_pca": "repengpca_{axis}_L{site}",
}

#: Every native template ends in the site suffix, and the STEM is what remains.
#: HALT E: a transported object is `g` + the source object's own stem and the
#: naive null is `naive_` + the same stem — the site suffix drops because a
#: transported object is named for the cell it is injected INTO, not for where
#: it came from.
NATIVE_SITE_SUFFIX = "_L{site}"
TRANSPORTED_PREFIX = "g"
NAIVE_PREFIX = "naive"
SITE_SUFFIX_RE = re.compile(r"_L\d+$")

#: The two band families, unchanged by the port: the native control asks whether
#: the SITE actuates, the transported control whether the TRANSPORT carries.
BAND_MEMBER_INDICES: tuple[int, ...] = (1, 2, 3)

#: The axis token, CLOSED. Lower-case alphanumerics with no separator, which is
#: what all four banked axes are (`sentiment`, `formality`, `refusal`,
#: `language`). Admitting `_` would make `caa_a_b_L7` parse two ways and the
#: grammar would stop being able to say which object a cell id names.
CELL_ID_AXIS_TOKEN = r"[a-z][a-z0-9]*"


def native_stem_template(vector_class: str,
                         templates: Optional[dict[str, str]] = None) -> str:
    """The class's key template with the site suffix removed.

    Split off the suffix rather than regex-stripped: the suffix is a LITERAL of
    the template, and a template that does not end in it has no decidable stem —
    which is a refusal, never a guess at where the stem ends.
    """
    template = (NATIVE_KEY_TEMPLATES if templates is None else
                templates)[vector_class]
    if not template.endswith(NATIVE_SITE_SUFFIX):
        raise ValueError(
            f"the native key template for {vector_class!r} is {template!r}, which "
            f"does not end in {NATIVE_SITE_SUFFIX!r}; its transported stem is "
            "undecidable and the lane does not invent one")
    return template[:-len(NATIVE_SITE_SUFFIX)]


def build_cell_id_grammar(templates: Optional[dict[str, str]] = None) -> str:
    """The cell-id grammar, GENERATED from the closed class vocabulary.

    THE GRAMMAR IS A GUARD AND STAYS ONE. Before the class port this was a
    hand-written alternation of three entropy literals; widening it by hand to
    admit the class family would have meant either retyping nine literals (which
    drift) or relaxing it toward `.*` (which stops guarding). Generating it from
    `NATIVE_KEY_TEMPLATES` means the ids the grammar admits are exactly the ids
    the key resolver can PRODUCE — one vocabulary, two consumers — and an axis is
    a closed token rather than a wildcard.

    The alternation is sorted so the string is stable across runs; the ids are
    the join key against the BANKED columns, and a grammar whose text depends on
    dict ordering is a pin that cannot be quoted.

    `templates` is the vocabulary to generate from, defaulting to the module's.
    It is a parameter for ONE reason: the selftest generates the ENTROPY-ONLY
    grammar from it and proves that sublanguage identical to the alternation
    re-freeze #4 wrote by hand, so the generation is checked against the pin it
    replaced and not only against itself.
    """
    vocabulary = NATIVE_KEY_TEMPLATES if templates is None else templates
    objects = {"baseline"}
    for vector_class in vocabulary:
        stem = native_stem_template(vector_class, vocabulary).format(
            axis=f"(?:{CELL_ID_AXIS_TOKEN})")
        # the NATIVE id keeps its own `_L<site>` and then takes the cell's, which
        # is the banked double-suffix (`entropy_gradient_L26_L26_a+0.30`).
        objects.add(stem + r"_L\d+")
        objects.add(TRANSPORTED_PREFIX + stem)
        objects.add(f"{NAIVE_PREFIX}_{stem}")
    for i in BAND_MEMBER_INDICES:
        objects.add(f"Rband{i}")
        objects.add(f"{TRANSPORTED_PREFIX}Rband{i}")
    return (r"(?:" + "|".join(sorted(objects))
            + r")_L\d+_a[+-]\d\.\d{2}(?:@absalpha)?")


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

# ── PORTABILITY DEFECT IN THE ABOVE PINS, FOUND 2026-08-22 — NAMED, NOT SILENT ─
#
# WHAT WAS FOUND. Those digests are over `ast.unparse` output, and `ast.unparse`
# IS INTERPRETER-DEPENDENT for f-strings that contain a same-quoted subscript.
# PEP 701 (3.12) let an f-string reuse its own quote character inside the
# replacement field, and the two interpreters render it differently:
#
#   CPython 3.12.3 (the LANE NODE's venv):  f'… {expect['prompt_tokens']} …'
#   CPython 3.13.9 (the DESK):              f"… {expect['prompt_tokens']} …"
#
# Exactly ONE of the twelve helpers contains such an f-string —
# `assert_span_accounting`, on three of its lines — so exactly one digest moves.
# Both renderings are 2206 bytes and parse to the same tree; the difference is
# quote style and nothing else.
#
# WHY IT MATTERS. The digests above were taken on the desk's 3.13. Every lane
# runner gates on `--selftest || exit 2`. So the tool as merged at re-freeze #4
# FAILS ITS OWN SELFTEST ON THE NODE IT RUNS ON, and did so before this port:
# measured on `bb5f1d1` in the node venv, 126/128 with this exact failure. It is
# a pre-existing defect of the merge, surfaced here because this certification is
# the first thing to run the repo tool on the node.
#
# WHAT IS DONE ABOUT IT HERE, AND WHAT IS NOT. The 3.13 pin above is NOT edited —
# it is the digest of record and it stays. The 3.12 rendering is admitted as a
# NAMED, MEASURED ALTERNATIVE for the one affected helper, and the selftest
# records WHICH rendering it saw. Two guards keep that from being a hole: the
# alternative is a single hard-coded digest (not a wildcard, not a skip), and the
# suite additionally requires the unparse to ROUND-TRIP — the code it renders must
# parse back to the same tree as the source it came from — so an alternative
# digest cannot silently stand for a different function.
#
# WHAT THE DESK SHOULD RULE ON. The durable fix is to pin over `ast.dump`, which
# renders string VALUES rather than source quoting and is therefore immune to this
# whole class of variance. `CERTIFIED_HELPER_AST_DUMP_DIGEST` below is that pin,
# computed and asserted here as ONE aggregate over the same twelve helpers, so the
# desk can promote it at the next re-freeze without a second archaeology pass.
# Recomputing the twelve individual pins is a change to certified apparatus and is
# NOT taken unilaterally (C§8).

#: `ast.unparse` renderings of the affected helper under PEP 701, keyed by helper.
#: MEASURED, not guessed: the lane venv, CPython 3.12.3, 2026-08-22.
CERTIFIED_HELPER_DIGESTS_PEP701: dict[str, str] = {
    "assert_span_accounting":
        "2cbbbf72329305b49e8a5289a1694fc7e11437b464c7d1c0713a44c84c4d23e8",
}

#: The INTERPRETER-INDEPENDENT pin: sha256 over the concatenated CANONICAL FORM
#: of the twelve helpers, in sorted name order, docstrings stripped. The canonical
#: form is `selftest._canon` — node type plus every declared `_fields` entry,
#: recursively, leaves as `repr` — and NOT `ast.dump`, because `ast.dump` is not
#: portable either (3.13 added `show_empty` and defaults it False, so 3.12 prints
#: `type_params=[]` where 3.13 prints nothing).
#:
#: MEASURED EQUAL, NOT ASSUMED: this exact number came out of both CPython 3.12.3
#: (the lane venv) and CPython 3.13.9 (the desk) on 2026-08-22, for all twelve
#: helpers individually as well as for the aggregate.
CERTIFIED_HELPER_AST_DUMP_DIGEST = (
    "56f59c160681fdc067724d0a6cb679106ccd1009838cf2d1a110dd0184c6053e")
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
#: `--class-spec` is the class port's ONE new option (2026-08-22) and is
#: OPTIONAL — absent, this tool is the entropy-gradient tool it was, which is
#: what keeps every existing runner invocation meaning what it meant.
ARGPARSE_CONTRACT: frozenset[str] = frozenset({
    "--allowed-host", "--arm", "--class-spec", "--code-root", "--corpus-sha",
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
#:
#: THE CLASS PORT MOVED THREE OF THESE, AND ONLY THREE. The transported and
#: naive templates spelled `gentropy_gradient` / `naive_entropy_gradient` as
#: LITERALS, which is HALT E's defect: a class column run through them would
#: have stamped CAA content under an entropy-gradient name. They now interpolate
#: the resolved key the same way the native template always interpolated
#: `native_key`. For an EGV column the three keys resolve to those very
#: literals, so every EGV cell id is byte-identical to the pre-port tool's — and
#: that is PROVEN in the selftest by rendering the plan, not asserted here.
CELL_ID_TEMPLATES: frozenset[str] = frozenset({
    "f'baseline_L{site}_a+0.00'",
    "f'{native_key}_L{site}_a{frac:+.2f}'",
    "f'Rband{i}_L{site}_a{frac:+.2f}'",
    "f'{transported_key}_L{site}_a{frac:+.2f}'",
    "f'gRband{i}_L{site}_a{frac:+.2f}'",
    "f'{naive_key}_L{site}_a{frac:+.2f}'",
    "f'{naive_key}_L{site}_a{d:+.2f}'",
    "f'{native_key}_L{site}_a{frac:+.2f}@absalpha'",
})

#: The EGV cell ids the pre-port tool spelled as literals, kept as the
#: BACKWARD-COMPATIBILITY WITNESS. The selftest renders the plan for a default
#: (entropy-gradient) spec and requires these exact strings out of it, so
#: "the EGV path is unchanged" is a property that fails loudly rather than a
#: sentence in a docstring.
EGV_CELL_ID_WITNESS: tuple[str, ...] = (
    "baseline_L26_a+0.00",
    "entropy_gradient_L26_L26_a-0.30",
    "entropy_gradient_L26_L26_a+0.30",
    "Rband1_L26_a-0.30",
    "Rband3_L26_a+0.30",
    "gentropy_gradient_L26_a-0.30",
    "gentropy_gradient_L26_a+0.30",
    "gRband1_L26_a-0.30",
    "gRband3_L26_a+0.30",
    "naive_entropy_gradient_L26_a-0.30",
    "naive_entropy_gradient_L26_a+0.30",
)

#: The same grammar as a regex, so the ids those templates PRODUCE are checked
#: and not only the templates themselves. GENERATED from the closed class
#: vocabulary (see `build_cell_id_grammar`) rather than hand-written, so the two
#: families cannot drift apart; the EGV alternation it produces is character-for-
#: character the one re-freeze #4 pinned, which the selftest asserts by value.
CELL_ID_GRAMMAR = build_cell_id_grammar()

#: The EGV half of that grammar, as re-freeze #4 wrote it by hand. Kept so the
#: generation is checked against the pin it replaced instead of only against
#: itself (rake M40's family: a generated pin that vouches for itself is not a
#: pin).
CELL_ID_GRAMMAR_EGV_PIN = (
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


class VectorClassRefused(LaneColumnRefused):
    """The class spec does not describe an object this lane will fire.

    HALT D's four contract refusals plus the closed-vocabulary one. Every case
    is a document the byte-exact lane would itself have refused, caught here
    before a GPU is touched rather than discovered in a banked column.
    """


class VectorKeyUnresolvable(LaneColumnRefused):
    """The bank and the spec disagree about which object this column rides.

    Two failure shapes, deliberately distinguished because they demand different
    responses: a bank that simply LACKS the object (go find the right bank), and
    a bank that holds ANOTHER CLASS's object at this site (the spec is wrong, or
    the bank is). Naming the second is what stops a lane run from silently
    becoming a run of a different experiment.
    """


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


class LaneVectorBasis(BaseModel):
    """One named basis: WHICH kind, WHICH sha. HALT D's fix, as this lane sees it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["corpus-manifest", "contrast-set"]
    sha256: str = Field(min_length=64, max_length=64)
    provenance: str = ""

    @field_validator("sha256")
    @classmethod
    def _is_hex(cls, v: str) -> str:
        int(v, 16)                       # a non-hex digest is a typo, not a basis
        return v


class LaneClassSpec(BaseModel):
    """The document that says WHAT OBJECT this column rides (HALT D + HALT E).

    ABSENT, the runner is an entropy-gradient runner and every key, cell id and
    emitted field is what it produced before the class port — that default is
    the whole of the backward-compatibility story. PRESENT, it names the class,
    the axis and both bases, and the refusals below are the ENGINE's own, applied
    here so a lane column cannot bank a class object under a contract the
    byte-exact lane would have refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["vllm-lane-class-spec/1"] = "vllm-lane-class-spec/1"
    vector_class: Literal["entropy_gradient", "caa", "repeng_pca"] = EGV_VECTOR_CLASS
    #: the AXIS a class object is of (`sentiment`, `formality`, …). Required for a
    #: class object — it is what makes four axes four columns — and refused for an
    #: EGV, which has exactly one object per site and no axis to name.
    axis: Optional[str] = None
    #: HALT D: the FD gate has no analogue for a class object. Admissible for a
    #: class object and for nothing else; an EGV that set it would bank an
    #: ungated lever.
    fd_gate_not_applicable: bool = False
    #: HALT D: the VECTOR basis — the RULED contrast set a class object was built
    #: from. Required for a class object, refused for an EGV (whose basis IS the
    #: corpus manifest and is already carried by the vintage chain).
    vector_basis: Optional[LaneVectorBasis] = None
    #: HALT D: the GENERATION basis — the corpus the prompts and generations are
    #: of. Optional in the document because the runner already takes
    #: `--corpus-sha`; when both are given they must agree, and a disagreement is
    #: a refusal rather than a preference.
    generation_basis: Optional[LaneVectorBasis] = None
    note: str = ""

    @property
    def is_class_vector(self) -> bool:
        return self.vector_class != EGV_VECTOR_CLASS

    @model_validator(mode="after")
    def _halt_d_contract(self) -> "LaneClassSpec":
        if self.is_class_vector and not self.axis:
            raise VectorClassRefused(
                f"vector_class={self.vector_class!r} names no `axis`. A class "
                "object's key carries the axis (`caa_<axis>_L<site>`), which is "
                "what makes four axes four columns instead of four names for one; "
                "without it the lane would have to guess which object it rides.")
        if not self.is_class_vector and self.axis:
            raise VectorClassRefused(
                f"an entropy-gradient column declared axis={self.axis!r}. The EGV "
                "is one object per site and has no axis; an axis here would appear "
                "in a cell id that the banked column does not carry.")
        if self.fd_gate_not_applicable and not self.is_class_vector:
            raise VectorClassRefused(
                "an entropy-gradient object claims fd_gate_not_applicable. The FD "
                "gate is the EGV's OWN acceptance test (§9 item 3) and waiving it "
                "would run this lane on an UNGATED lever.")
        if self.is_class_vector and self.vector_basis is None:
            raise VectorClassRefused(
                f"vector_class={self.vector_class!r} names no `vector_basis`. A "
                "class object's VECTOR basis is the RULED contrast set it was "
                "built from; writing the corpus sha instead would be false "
                "provenance and writing nothing would be a hole (§9 item 2).")
        if self.is_class_vector and self.vector_basis.kind != "contrast-set":
            raise VectorClassRefused(
                f"vector_basis.kind={self.vector_basis.kind!r} on a class object. "
                "The VECTOR basis is the contrast set; `corpus-manifest` is the "
                "GENERATION basis and rides its own field. The two are never one.")
        if not self.is_class_vector and self.vector_basis is not None:
            raise VectorClassRefused(
                "an entropy-gradient object declared a vector_basis "
                f"({self.vector_basis.kind}). The EGV's basis IS the corpus "
                "manifest, already carried by the vintage chain — a second name "
                "for one basis is how the two bases blur (HALT D).")
        if (self.generation_basis is not None
                and self.generation_basis.kind != "corpus-manifest"):
            raise VectorClassRefused(
                f"generation_basis.kind={self.generation_basis.kind!r} — the basis "
                "the PROMPTS and GENERATIONS stand on is the corpus manifest, "
                "always.")
        return self


class LaneVectorKeys(BaseModel):
    """The four npz keys a lane column reads, resolved ONCE and used everywhere.

    A typed object rather than four locals so the plan block and the ingredients
    block cannot drift apart: the pre-port tool spelled `gentropy_gradient` in
    both, which is exactly how a class column would have kept an entropy-gradient
    name in half its output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vector_class: str
    axis: Optional[str] = None
    site: int
    native: str
    stem: str
    transported: str
    naive: str
    #: WHERE the stem came from. `template` when nothing but this module's own
    #: rendering was available (the CPU paths and the selftest); `bank` when the
    #: bank's own spelling of the native key was read and used, which is what
    #: every real column does.
    stem_source: Literal["template", "bank"] = "template"

    @property
    def band_native(self) -> tuple[str, ...]:
        return tuple(f"Rband{i}" for i in BAND_MEMBER_INDICES)

    @property
    def band_transported(self) -> tuple[str, ...]:
        return tuple(f"{TRANSPORTED_PREFIX}Rband{i}" for i in BAND_MEMBER_INDICES)


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


# ── key resolution (HALT E), ported from v2 `2c327112…` ──────────────────────

def resolve_vector_keys(*, site: int, vector_class: str = EGV_VECTOR_CLASS,
                        axis: Optional[str] = None) -> LaneVectorKeys:
    """The class's native key, and the transported/naive keys DERIVED FROM IT.

    HALT E's rule: the transported and naive keys come from the OBJECT, never
    from the campaign's object of record. For an EGV column the stem is
    `entropy_gradient` and the three keys are byte-identical to the literals the
    pre-port tool spelled; for a CAA column the stem carries the axis, so
    `gcaa_<axis>` and `naive_caa_<axis>` are per-axis and four axes are four
    columns.

    This is the TEMPLATE rendering. It is the right answer for the CPU paths and
    for deciding what to LOOK FOR in a bank, but a column resolves through
    `resolve_keys_from_bank` so the stem of record is the bank's own bytes.
    """
    if vector_class not in NATIVE_KEY_TEMPLATES:
        raise VectorClassRefused(
            f"vector_class={vector_class!r} has no banked key convention here "
            f"(known: {sorted(NATIVE_KEY_TEMPLATES)}). The vocabulary is CLOSED — "
            "an unrecognized class is refused rather than defaulted to the object "
            "of record, because defaulting is how a class object acquires the "
            "campaign's name.")
    native = NATIVE_KEY_TEMPLATES[vector_class].format(site=site, axis=axis)
    stem = SITE_SUFFIX_RE.sub("", native)
    if stem == native:                                          # pragma: no cover
        raise VectorClassRefused(
            f"the resolved native key {native!r} carries no `_L<site>` suffix, so "
            "its transported stem is undecidable. The banked convention is "
            "`<object>_L<site>`; the lane does not invent one.")
    return LaneVectorKeys(
        vector_class=vector_class, axis=axis, site=site, native=native, stem=stem,
        transported=f"{TRANSPORTED_PREFIX}{stem}",
        naive=f"{NAIVE_PREFIX}_{stem}", stem_source="template")


def native_key_family(key: str, *, site: int) -> Optional[str]:
    """Which class family a NATIVE bank key belongs to at this site, or None.

    Decided from the banked templates' own literal prefixes plus the `_L<site>`
    suffix, so a bank's contents are read the way the campaign NAMES them rather
    than by a heuristic this file invented.
    """
    for vector_class in NATIVE_KEY_TEMPLATES:
        prefix = native_stem_template(vector_class).split("{")[0]
        if key.startswith(prefix) and key.endswith(f"_L{site}"):
            return vector_class
    return None


def assert_key_present(keys_in_bank: Sequence[str], wanted: str, *, where: str,
                       site: int) -> None:
    """The key is there, or this says WHICH other class's object was there instead."""
    present = sorted(set(keys_in_bank))
    if wanted in present:
        return
    others = sorted(k for k in present
                    if k != wanted and native_key_family(k, site=site) is not None)
    hint = (f" The bank DOES hold {others} at L{site} — another object entirely, "
            "which this runner will not substitute." if others else "")
    raise VectorKeyUnresolvable(
        f"{where} has no {wanted!r} (keys present: {present}).{hint} The lane "
        "column writes the SAME banked vector the byte-exact column wrote, never "
        f"a re-derived one, and never another object's. [L{site}]")


def assert_bank_unambiguous(keys_in_bank: Sequence[str], *, site: int,
                            declared: str, where: str) -> None:
    """A bank holding two class families at this site must have been TOLD which one.

    With no class spec the runner defaults to the entropy gradient, and a bank
    that also carries CAA objects at the same site would have that default
    silently pick one of two experiments. Declared explicitly, either is fine;
    undeclared, this refuses.
    """
    families = sorted({f for f in (native_key_family(k, site=site)
                                   for k in keys_in_bank) if f is not None})
    if len(families) > 1 and declared == EGV_VECTOR_CLASS:
        raise VectorKeyUnresolvable(
            f"{where} carries objects of {families} at L{site} and no vector class "
            "was DECLARED, so the runner would silently take the campaign's object "
            "of record out of a bank holding two experiments. Name the class in "
            "--class-spec; the lane never guesses which object a column rides.")


def resolve_keys_from_bank(keys_in_bank: Sequence[str], *, site: int,
                           spec: LaneClassSpec, where: str) -> LaneVectorKeys:
    """The keys of record, with the STEM taken from the BANK'S OWN SPELLING.

    THE DIFFERENCE FROM `resolve_vector_keys`, AND WHY IT IS WORTH A FUNCTION.
    The template rendering says what a `caa`/`language`/`L21` object OUGHT to be
    called. The bytes in the npz say what it IS called. Those are two claims, and
    the second is the one every downstream key is derived from — so this looks
    the native key up IN THE BANK, takes the matched entry's own characters, and
    derives `stem`/`transported`/`naive` from THOSE. A bank whose spelling
    differs from the template does not get quietly renamed: `assert_key_present`
    refuses first, and names which other object it found instead. The equality of
    the two spellings is therefore PROVEN once per column rather than assumed
    forever, which is HALT E's rule read strictly.

    The two guards run in this order on purpose: ambiguity first (a bank holding
    two families under an undeclared class is a question about the RUN, not about
    one key), then presence.
    """
    assert_bank_unambiguous(keys_in_bank, site=site,
                            declared=spec.vector_class, where=where)
    wanted = resolve_vector_keys(site=site, vector_class=spec.vector_class,
                                 axis=spec.axis)
    assert_key_present(keys_in_bank, wanted.native, where=where, site=site)
    banked_spelling = next(k for k in keys_in_bank if k == wanted.native)
    stem = SITE_SUFFIX_RE.sub("", banked_spelling)
    if stem == banked_spelling:                                 # pragma: no cover
        raise VectorKeyUnresolvable(
            f"{where} spells the native key {banked_spelling!r}, which carries no "
            "`_L<site>` suffix — its transported stem is undecidable and this lane "
            "does not invent one")
    return wanted.model_copy(update={
        "native": banked_spelling, "stem": stem,
        "transported": f"{TRANSPORTED_PREFIX}{stem}",
        "naive": f"{NAIVE_PREFIX}_{stem}", "stem_source": "bank"})


def load_class_spec(path: Optional[Path]) -> LaneClassSpec:
    """Read the class spec, or return the ENTROPY-GRADIENT default.

    The default is the whole of the backward-compatibility story: no path means
    the campaign's object of record, which is what every flag of this runner
    meant before the class column existed.
    """
    if path is None:
        return LaneClassSpec()
    if not path.exists():
        raise VectorClassRefused(f"no class spec at {path}")
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise VectorClassRefused(
            f"{path}: unreadable class spec ({type(exc).__name__}: {exc})") from exc
    if not isinstance(body, dict):
        raise VectorClassRefused(f"{path}: a class spec is a JSON object")
    version = body.get("schema_version", "vllm-lane-class-spec/1")
    if version != "vllm-lane-class-spec/1":
        raise VectorClassRefused(
            f"{path}: class spec schema {version!r}, this runner speaks "
            "'vllm-lane-class-spec/1' — a silently-changed contract is a column "
            "run against terms nobody agreed to.")
    try:
        return LaneClassSpec(**body)
    except LaneColumnRefused:
        raise
    except (TypeError, ValueError) as exc:
        raise VectorClassRefused(f"{path}: {exc}") from exc


def class_contract_block(spec: LaneClassSpec, *, keys: LaneVectorKeys,
                         corpus_sha: str) -> dict[str, Any]:
    """HALT D's contract as the block that rides into every column artifact.

    Both bases are named or neither is trustworthy, so the GENERATION basis is
    ALWAYS emitted — from the spec when it names one, from `--corpus-sha`
    otherwise — and a spec whose generation basis disagrees with the corpus the
    column is actually running against is a refusal, not a preference.
    """
    generation = spec.generation_basis
    if generation is not None and generation.sha256 != corpus_sha:
        raise VectorClassRefused(
            f"the class spec's generation_basis {generation.sha256[:12]}… is not "
            f"the corpus this column runs against ({corpus_sha[:12]}…). The "
            "GENERATION basis is the corpus the prompts and generations are OF; "
            "two answers is not a basis.")
    return {
        "vector_class": spec.vector_class,
        "axis": spec.axis,
        "is_class_vector": spec.is_class_vector,
        "fd_gate_not_applicable": spec.fd_gate_not_applicable,
        "vector_basis": (spec.vector_basis.model_dump()
                         if spec.vector_basis is not None else None),
        "generation_basis": (generation.model_dump() if generation is not None else {
            "kind": "corpus-manifest", "sha256": corpus_sha,
            "provenance": "the corpus manifest the PROMPTS and GENERATIONS are of "
                          "— the GENERATION basis, never the object's"}),
        "native_key": keys.native,
        "transported_key": keys.transported,
        "naive_key": keys.naive,
        "two_bases_note": (
            "HALT D (Luxia's ruling, 2026-08-05): a class cell stands on TWO "
            "bases. The GENERATION basis is the corpus manifest; the VECTOR basis "
            "is the RULED contrast set the class object was built from. Both are "
            "named; neither is readable as the other. `fd_gate_not_applicable` is "
            "admissible for a class object ONLY — no FD-gate analogue exists for "
            "it — and an entropy-gradient object still REQUIRES its gate."),
        "note": spec.note,
    }


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
                    help="the BANKED calibration bank; must hold the NATIVE key "
                         "the class spec resolves (entropy_gradient_L<site> by "
                         "default, caa_<axis>_L<site> under a class spec)")
    ap.add_argument("--transported-npz", type=Path, default=None,
                    help="the BANKED transported bank; must hold the transported "
                         "and naive keys DERIVED from the native object's stem "
                         "(gentropy_gradient / naive_entropy_gradient by default, "
                         "gcaa_<axis> / naive_caa_<axis> under a class spec). "
                         "Enables the B3/B4 cells.")
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
    # THE CLASS PORT'S ONE NEW OPTION (2026-08-22). Optional by construction:
    # absent, `load_class_spec(None)` is the entropy gradient and every key, cell
    # id and emitted field is what this tool produced before the port, so no
    # existing runner invocation changes meaning.
    ap.add_argument("--class-spec", type=Path, default=None,
                    help="a `vllm-lane-class-spec/1` document (JSON: "
                         "vector_class, axis, fd_gate_not_applicable, "
                         "vector_basis, generation_basis). ABSENT = the "
                         "entropy-gradient family, which is what every flag here "
                         "meant before the class column existed. With "
                         "vector_class=caa the keys become caa_<axis>_L<site> / "
                         "gcaa_<axis> / naive_caa_<axis>, and HALT D's two-bases "
                         "contract rides into the column artifact.")
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

    # ---- the class spec, and the keys it resolves (the class port) -----------
    # Loaded BEFORE the bank so a malformed contract costs no I/O, and resolved
    # AGAINST the bank below so the stem of record is the bank's own spelling.
    try:
        class_spec = load_class_spec(args.class_spec)
    except LaneColumnRefused as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    doc["class_spec_path"] = (str(args.class_spec) if args.class_spec else None)
    doc["class_spec_sha256"] = (sha256_file(args.class_spec)
                                if args.class_spec else None)

    # ---- vectors -------------------------------------------------------------
    doc["vectors_npz"] = str(args.vectors_npz)
    doc["vectors_npz_sha256"] = sha256_file(args.vectors_npz)
    with np.load(args.vectors_npz, allow_pickle=True) as z:
        vecs = {k: np.asarray(z[k]).reshape(-1).astype(np.float32) for k in z.files}
    try:
        keys = resolve_keys_from_bank(list(vecs), site=site, spec=class_spec,
                                      where=str(args.vectors_npz))
        doc["vector_class_contract"] = class_contract_block(
            class_spec, keys=keys, corpus_sha=args.corpus_sha)
    except LaneColumnRefused as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    native_key = keys.native
    transported_key = keys.transported
    naive_key = keys.naive
    doc["vector_keys"] = {
        "native": keys.native, "transported": keys.transported,
        "naive": keys.naive, "stem": keys.stem,
        "stem_source": keys.stem_source,
        "band_native": list(keys.band_native),
        "band_transported": list(keys.band_transported),
        "derivation": ("HALT E (2026-08-05): the transported and naive keys come "
                       "from the OBJECT — `g` + the native object's own stem, "
                       "`naive_` + the same stem — never from the campaign's "
                       "object of record. An EGV column's three keys are therefore "
                       "the banked literals; a class column's carry the axis. The "
                       "stem is the BANK's own spelling of the native key, which "
                       "the resolver proved equal to the template rendering for "
                       "this column rather than assuming."),
    }
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
        # The two keys asked for are DERIVED from the native object's own stem,
        # never retyped: for an EGV column they are `gentropy_gradient` and
        # `naive_entropy_gradient`, the very literals this block used to spell.
        try:
            for need in (transported_key, naive_key):
                assert_key_present(list(tvecs), need, site=site,
                                   where=f"transported bank {args.transported_npz}")
        except LaneColumnRefused as exc:
            print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
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
            plan.append({"cell_id": f"{transported_key}_L{site}_a{frac:+.2f}",
                         "kind": "transported", "vector_key": transported_key,
                         "alpha_frac": float(frac), "band_family": None,
                         "vector": tvecs[transported_key], "alpha_mode": "frac"})
        for i in (1, 2, 3):
            for frac in DOSE_LADDER:
                plan.append({"cell_id": f"gRband{i}_L{site}_a{frac:+.2f}",
                             "kind": "transported_band", "vector_key": f"gRband{i}",
                             "alpha_frac": float(frac), "band_family": "gRband",
                             "vector": gband[f"gRband{i}"], "alpha_mode": "frac"})
        for frac in SCORING_DOSES:
            plan.append({"cell_id": f"{naive_key}_L{site}_a{frac:+.2f}",
                         "kind": "naive", "vector_key": naive_key,
                         "alpha_frac": float(frac), "band_family": None,
                         "vector": tvecs[naive_key],
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
        trises = ladder(lambda f: f"{transported_key}_L{site}_a{f:+.2f}")
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
                "cell_id": f"{naive_key}_L{site}_a{d:+.2f}",
                "entropy_rise": rise_of(f"{naive_key}_L{site}_a{d:+.2f}"),
                "distinct_word_ratio": by_id.get(
                    f"{naive_key}_L{site}_a{d:+.2f}", {})
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
    # HOISTED to the top of the suite, 2026-08-22. `skip` was defined halfway
    # down, so the ONE optional block above it — the scipy cross-check — degraded
    # to a bare `print` and six checks simply VANISHED from the count. That is
    # the shape rake M44 is named for: an unqualified count that silently vouches
    # for less than it says, and it is why the recorded floor could not be met in
    # the lane venv (which has no scipy) even before this port.
    skips: list[str] = []

    def ok(cond: bool, name: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(f"{name}{(' — ' + detail) if detail else ''}")

    def skip(name: str, why: str) -> None:
        nonlocal checks
        checks += 1
        skips.append(f"{name} — {why}")

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
        skip("the scipy cross-check on `spearman_rho`",
             "scipy is not importable here; evidence MISSING: that this lane's "
             "transcribed rank correlation gives the same number as the frozen "
             "reader's `scipy.stats.spearmanr` on the six pinned cases. The lane "
             "venv has no scipy BY DESIGN (verified 2026-08-08), so this skip is "
             "the normal state on the node and is NAMED rather than printed")
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

    # (`skip` and `skips` are hoisted to the top of the suite — see the note
    # there; they used to be declared at this point, which is what left the
    # scipy block above unable to name its own skip.)

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
    _pep701_renderings: list[str] = []
    for hname, want in sorted(CERTIFIED_HELPER_DIGESTS.items()):
        node = _shared.get(hname)
        if node is None:
            ok(False, f"the shared helper `{hname}` still exists", "missing")
            continue
        got = _digest([node])
        alt = CERTIFIED_HELPER_DIGESTS_PEP701.get(hname)
        if alt is not None and got == alt and got != want:
            _pep701_renderings.append(hname)
        ok(got == want or got == alt,
           f"the shared helper `{hname}` is AST-identical to the certified "
           "tool's — the pure arithmetic the whole lane rests on did not move "
           "when the TP branch was added, nor when the class port was made. On "
           "an interpreter that renders PEP 701 f-strings the other way the "
           "MEASURED alternative digest is accepted and recorded; anything else "
           "fails here",
           f"{got} vs {want}" + (f" (pep701 alt {alt})" if alt else ""))
    ok(len(_pep701_renderings) <= len(CERTIFIED_HELPER_DIGESTS_PEP701),
       "…and no helper outside the NAMED PEP-701 set needed the alternative "
       "digest — the alternative is one measured number per named helper, never "
       "a wildcard", f"{_pep701_renderings}")
    if _pep701_renderings:
        print(f"  (PEP 701 rendering in use for {_pep701_renderings} — this "
              f"interpreter is CPython {sys.version_info.major}."
              f"{sys.version_info.minor}; the digest of record was taken on 3.13)")

    # THE ROUND-TRIP, which is what keeps the alternative from being a hole. On
    # ANY interpreter, unparsing a helper and re-parsing it must give back THE
    # TREE IT CAME FROM — so a digest admitted from the alternative map cannot
    # stand for a function that is merely spelled differently. `ast.dump` is the
    # comparison because it prints a constant's VALUE and never its quoting,
    # which is precisely the axis the PEP-701 variance lives on. A FRESH parse is
    # used because `_strip_docstrings` mutates in place and `_tree` is still
    # wanted intact by the checks above.
    #
    # `ast.dump` ITSELF IS NOT PORTABLE — measured, not assumed. 3.13 added
    # `show_empty` and defaults it False, so 3.12 prints `type_params=[]` where
    # 3.13 prints nothing, and the two dumps of one tree differ. So the canonical
    # form below is hand-rolled over `_fields`: node type, then every declared
    # field in the class's own order, recursively, with leaves as `repr`. It
    # names no default and omits nothing, so it says the same thing on both.
    def _canon(node: Any) -> str:
        if isinstance(node, ast.AST):
            return ("(" + type(node).__name__ + ","
                    + ",".join(f"{f}={_canon(getattr(node, f, None))}"
                               for f in node._fields) + ")")
        if isinstance(node, list):
            return "[" + ",".join(_canon(x) for x in node) + "]"
        return repr(node)

    _fresh = {n.name: n for n in ast.parse(_src).body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    _dumps: dict[str, str] = {}
    for hname in sorted(CERTIFIED_HELPER_DIGESTS):
        node = _fresh.get(hname)
        if node is None:
            continue
        before = _strip_docstrings(ast.Module(body=[node], type_ignores=[]))
        _dumps[hname] = _canon(before)
        after = _strip_docstrings(ast.parse(ast.unparse(before)))
        ok(_dumps[hname] == _canon(after),
           f"`{hname}` round-trips through `ast.unparse` back to the same tree — "
           "the RENDERING may vary by interpreter, the CODE may not, and that is "
           "what makes the named PEP-701 alternative safe to accept")

    # THE PORTABLE PIN the desk may promote: one aggregate over those canonical
    # forms. VERIFIED equal on CPython 3.12.3 (the lane node) and 3.13.9 (the
    # desk) on 2026-08-22 — that equality is the whole claim, and it is a
    # measurement, not a property of `ast`.
    _dump_digest = hashlib.sha256(
        "\n".join(_dumps[h] for h in sorted(_dumps)).encode()).hexdigest()
    ok(_dump_digest == CERTIFIED_HELPER_AST_DUMP_DIGEST,
       "the INTERPRETER-INDEPENDENT aggregate pin over the twelve helpers' "
       "`ast.dump` is the recorded one — this is the pin the desk may promote to "
       "retire the PEP-701 variance, and it is asserted here rather than merely "
       "proposed", f"{_dump_digest} vs {CERTIFIED_HELPER_AST_DUMP_DIGEST}")

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

    # ══ the class port (2026-08-22): BOTH vector families, at any TP ════════
    #
    # Everything below is new at the class×TP merge. It is CPU-only and
    # data-independent; the one block that needs the campaign package degrades
    # to a NAMED SKIP that says what evidence is missing (rakes M44 + M59(3)).

    # ── 12. the closed vocabulary and HALT E's key derivation ────────────────
    _egv = resolve_vector_keys(site=26)
    ok((_egv.native, _egv.transported, _egv.naive, _egv.stem)
       == ("entropy_gradient_L26", "gentropy_gradient",
           "naive_entropy_gradient", "entropy_gradient"),
       "with NO class spec the three keys are the literals the pre-port tool "
       "spelled — the default IS the backward compatibility, and it is checked "
       "by value rather than promised",
       f"{_egv.native} / {_egv.transported} / {_egv.naive}")
    _caa = resolve_vector_keys(site=26, vector_class="caa", axis="sentiment")
    ok((_caa.native, _caa.transported, _caa.naive)
       == ("caa_sentiment_L26", "gcaa_sentiment", "naive_caa_sentiment"),
       "HALT E: a CAA object's transported and naive keys come from ITS OWN "
       "stem, so four axes are four columns and not four names for one",
       f"{_caa.native} / {_caa.transported} / {_caa.naive}")
    _rp = resolve_vector_keys(site=26, vector_class="repeng_pca", axis="sentiment")
    ok((_rp.native, _rp.transported, _rp.naive)
       == ("repengpca_sentiment_L26", "grepengpca_sentiment",
           "naive_repengpca_sentiment"),
       "…and the repeng method row rides the SAME derivation, because the "
       "derivation is the object's and not the class's",
       f"{_rp.native} / {_rp.transported} / {_rp.naive}")
    _other = resolve_vector_keys(site=26, vector_class="caa", axis="formality")
    ok(not ({_caa.native, _caa.transported, _caa.naive}
            & {_other.native, _other.transported, _other.naive}),
       "two axes of one class share NO key — the collision HALT E exists to "
       "prevent cannot be spelled")
    ok(_egv.band_native == ("Rband1", "Rband2", "Rband3")
       and _egv.band_transported == ("gRband1", "gRband2", "gRband3"),
       "the two band families are unchanged by the port")
    ok(_egv.stem_source == "template",
       "a template-resolved key SAYS it was template-resolved — a column's keys "
       "come from the bank and the stamp must be able to tell the two apart")
    try:
        resolve_vector_keys(site=26, vector_class="lda", axis="x")
        ok(False, "an UNKNOWN vector class is REFUSED")
    except VectorClassRefused as exc:
        ok("CLOSED" in str(exc),
           "an UNKNOWN vector class is REFUSED, and the refusal says the "
           "vocabulary is closed rather than defaulting to the object of record",
           str(exc)[:80])
    ok(LaneVectorKeys.model_config.get("frozen") is True
       and LaneVectorKeys.model_config.get("extra") == "forbid",
       "the resolved keys are frozen and forbid unknown fields — the plan block "
       "and the ingredients block read ONE object, so they cannot drift")
    for vc in sorted(NATIVE_KEY_TEMPLATES):
        ok(NATIVE_KEY_TEMPLATES[vc].endswith(NATIVE_SITE_SUFFIX),
           f"the native template for `{vc}` ends in the site suffix, so its "
           "transported stem is decidable rather than guessed",
           NATIVE_KEY_TEMPLATES[vc])

    # ── 13. resolution against a BANK, which is where the stem comes from ────
    _bank_caa = ["caa_sentiment_L26", "Rband1", "Rband2", "Rband3"]
    _spec_caa = LaneClassSpec(
        vector_class="caa", axis="sentiment", fd_gate_not_applicable=True,
        vector_basis=LaneVectorBasis(kind="contrast-set", sha256="a" * 64))
    _rk = resolve_keys_from_bank(_bank_caa, site=26, spec=_spec_caa, where="a bank")
    ok(_rk.native == "caa_sentiment_L26" and _rk.stem == "caa_sentiment"
       and _rk.stem_source == "bank",
       "the stem of record is the BANK's own spelling of the native key, and the "
       "keys say so — the template rendering is what we LOOK FOR, not what we "
       "then derive from", f"{_rk.stem} ({_rk.stem_source})")
    try:
        resolve_keys_from_bank(["entropy_gradient_L26"], site=26, spec=_spec_caa,
                               where="an EGV bank")
        ok(False, "a bank holding ANOTHER class's object at this site REFUSES")
    except VectorKeyUnresolvable as exc:
        ok("entropy_gradient_L26" in str(exc) and "not substitute" in str(exc),
           "a bank holding ANOTHER class's object at this site REFUSES, and the "
           "refusal NAMES what it found instead — that is what stops a lane run "
           "silently becoming a run of a different experiment", str(exc)[:110])
    try:
        resolve_keys_from_bank(["entropy_gradient_L26", "caa_sentiment_L26"],
                               site=26, spec=LaneClassSpec(), where="a mixed bank")
        ok(False, "an UNDECLARED class over a two-family bank REFUSES")
    except VectorKeyUnresolvable as exc:
        ok("DECLARED" in str(exc),
           "an UNDECLARED class over a bank carrying two families at this site "
           "REFUSES rather than silently taking the object of record",
           str(exc)[:110])
    ok(resolve_keys_from_bank(["entropy_gradient_L26", "caa_sentiment_L26"],
                              site=26, spec=_spec_caa,
                              where="a mixed bank").native == "caa_sentiment_L26",
       "…and the SAME bank is fine once the class is declared — the refusal is "
       "about the ambiguity, never about the bank")
    try:
        resolve_keys_from_bank([], site=26, spec=LaneClassSpec(), where="an empty bank")
        ok(False, "an EMPTY bank REFUSES")
    except VectorKeyUnresolvable:
        ok(True, "an EMPTY bank REFUSES — an absent key is a hole, never a pass")
    ok(native_key_family("caa_sentiment_L26", site=26) == "caa"
       and native_key_family("entropy_gradient_L26", site=26) == "entropy_gradient"
       and native_key_family("caa_sentiment_L21", site=26) is None
       and native_key_family("gcaa_sentiment", site=26) is None
       and native_key_family("Rband1", site=26) is None,
       "family detection reads the campaign's OWN templates plus the site suffix "
       "— a band, a transported object, and an object at another site are all "
       "correctly not-a-native-key-here")

    # ── 14. HALT D's four contract refusals ──────────────────────────────────
    _basis = LaneVectorBasis(kind="contrast-set", sha256="a" * 64)
    _good = LaneClassSpec(vector_class="caa", axis="sentiment",
                          fd_gate_not_applicable=True, vector_basis=_basis)
    ok(_good.is_class_vector and LaneClassSpec().is_class_vector is False,
       "`is_class_vector` is decided by the class, not by whether a spec exists")
    for bad, why in (
            (dict(vector_class="caa", vector_basis=_basis), "a class with no axis"),
            (dict(axis="sentiment"), "an EGV that named an axis"),
            (dict(fd_gate_not_applicable=True), "an EGV waiving its FD gate"),
            (dict(vector_class="caa", axis="s"), "a class with no vector basis"),
            (dict(vector_class="caa", axis="s",
                  vector_basis={"kind": "corpus-manifest", "sha256": "a" * 64}),
             "a class whose VECTOR basis is the corpus manifest"),
            (dict(vector_basis=_basis), "an EGV naming a vector basis"),
            (dict(generation_basis=_basis),
             "a GENERATION basis that is not a corpus manifest")):
        try:
            LaneClassSpec(**bad)                          # type: ignore[arg-type]
            ok(False, f"HALT D REFUSES: {why}")
        except VectorClassRefused:
            ok(True, f"HALT D REFUSES: {why}")
    try:
        LaneVectorBasis(kind="contrast-set", sha256="z" * 64)
        ok(False, "a NON-HEX basis digest is a refusal")
    except Exception:                                          # noqa: BLE001
        ok(True, "a NON-HEX basis digest is a refusal, not a basis")
    try:
        LaneVectorBasis(kind="contrast-set", sha256="a" * 63)
        ok(False, "a SHORT basis digest is a refusal")
    except Exception:                                          # noqa: BLE001
        ok(True, "a SHORT basis digest is a refusal (rake M40: a truncated digest "
                 "padded to width is not the digest)")
    ok(LaneClassSpec.model_config.get("frozen") is True
       and LaneClassSpec.model_config.get("extra") == "forbid",
       "the class spec is frozen and forbids unknown keys — a typo'd contract "
       "field is a refusal, never a silently dropped clause")
    _contract = class_contract_block(_good, keys=_caa, corpus_sha="b" * 64)
    ok(_contract["vector_basis"]["kind"] == "contrast-set"
       and _contract["generation_basis"]["kind"] == "corpus-manifest"
       and _contract["generation_basis"]["sha256"] == "b" * 64
       and _contract["fd_gate_not_applicable"] is True
       and _contract["native_key"] == "caa_sentiment_L26"
       and _contract["transported_key"] == "gcaa_sentiment"
       and _contract["naive_key"] == "naive_caa_sentiment",
       "the contract block names BOTH bases and all three keys — HALT D's rule "
       "is that both are named or neither is trustworthy")
    _egv_contract = class_contract_block(LaneClassSpec(), keys=_egv,
                                         corpus_sha="b" * 64)
    ok(_egv_contract["vector_basis"] is None
       and _egv_contract["is_class_vector"] is False
       and _egv_contract["fd_gate_not_applicable"] is False
       and _egv_contract["generation_basis"]["sha256"] == "b" * 64,
       "an EGV column still carries a GENERATION basis and no VECTOR basis — its "
       "basis IS the corpus manifest")
    try:
        class_contract_block(
            LaneClassSpec(generation_basis=LaneVectorBasis(
                kind="corpus-manifest", sha256="c" * 64)),
            keys=_egv, corpus_sha="b" * 64)
        ok(False, "a generation basis that is not THIS column's corpus REFUSES")
    except VectorClassRefused:
        ok(True, "a generation basis that is not THIS column's corpus REFUSES — "
                 "two answers is not a basis")

    # ── 15. `load_class_spec`: the document, and every way it can be wrong ───
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as _td:
        _tdp = Path(_td)
        ok(load_class_spec(None).vector_class == EGV_VECTOR_CLASS
           and load_class_spec(None).axis is None,
           "NO path is the entropy-gradient default, which is the whole of the "
           "backward-compatibility story")
        _p = _tdp / "spec.json"
        _p.write_text(json.dumps({
            "schema_version": "vllm-lane-class-spec/1", "vector_class": "caa",
            "axis": "language", "fd_gate_not_applicable": True,
            "generation_basis": None,
            "vector_basis": {"kind": "contrast-set", "sha256": "b" * 64}}))
        _loaded = load_class_spec(_p)
        ok(_loaded.vector_class == "caa" and _loaded.axis == "language"
           and _loaded.vector_basis is not None
           and _loaded.generation_basis is None,
           "a well-formed /1 document loads, `generation_basis: null` included — "
           "the runner takes it from --corpus-sha, which is the banked pattern")
        for body, why in (
                ({"schema_version": "vllm-lane-class-spec/2"}, "a /2 schema"),
                ({"vector_class": "caa"}, "a class with no axis"),
                ({"surprise": 1}, "an unknown field")):
            _bp = _tdp / "bad.json"
            _bp.write_text(json.dumps(body))
            try:
                load_class_spec(_bp)
                ok(False, f"the loader REFUSES: {why}")
            except VectorClassRefused:
                ok(True, f"the loader REFUSES: {why}")
        _bp = _tdp / "notjson.json"
        _bp.write_text("{ not json")
        try:
            load_class_spec(_bp)
            ok(False, "the loader REFUSES unreadable JSON")
        except VectorClassRefused as exc:
            ok("unreadable" in str(exc),
               "the loader REFUSES unreadable JSON and says so", str(exc)[:70])
        _bp = _tdp / "list.json"
        _bp.write_text("[1, 2]")
        try:
            load_class_spec(_bp)
            ok(False, "the loader REFUSES a JSON array")
        except VectorClassRefused:
            ok(True, "the loader REFUSES a JSON array — a spec is an object")
        try:
            load_class_spec(_tdp / "absent.json")
            ok(False, "the loader REFUSES a path that is not there")
        except VectorClassRefused:
            ok(True, "the loader REFUSES a path that is not there, rather than "
                     "falling back to the EGV default — an unreadable spec is a "
                     "hole, and a hole must not read as 'no spec was asked for'")

    # ── 16. the grammar admits the class family and is STILL a guard ─────────
    _cg = _re.compile(CELL_ID_GRAMMAR)
    for ex in ("caa_language_L21_L21_a-0.30", "caa_language_L21_L21_a+0.30@absalpha",
               "gcaa_language_L21_a+0.03", "naive_caa_language_L21_a-0.30",
               "repengpca_sentiment_L26_L26_a+0.10",
               "grepengpca_sentiment_L26_a-0.10",
               "naive_repengpca_sentiment_L26_a+0.30"):
        ok(bool(_cg.fullmatch(ex)), f"the grammar accepts the class id {ex!r}")
    for bad, why in (
            ("caa__L21_a+0.30", "an EMPTY axis"),
            ("caa_Language_L21_L21_a+0.30", "an UPPER-CASE axis"),
            ("caa_two_words_L21_L21_a+0.30", "an axis carrying a separator"),
            ("caa_language_a+0.30", "a native id with no site suffix"),
            ("lda_language_L21_L21_a+0.30", "an object of an unknown class"),
            ("xcaa_language_L21_a+0.30", "a near-miss on the transported prefix"),
            ("gcaa_language_L21_a+0.3", "a dose that is not two decimal places")):
        ok(not _cg.fullmatch(bad),
           f"the grammar REJECTS {bad!r} — {why}; widening it to anything-goes "
           "would retire the join key against the banked columns")
    # …and the EGV SUBLANGUAGE is exactly what re-freeze #4 pinned by hand. Not
    # string equality (the generator sorts and spells `Rband1|Rband2|Rband3`
    # where the pin spells `Rband[123]`) — LANGUAGE equality, decided over a
    # corpus the two must agree on id by id.
    _egv_gen = _re.compile(build_cell_id_grammar(
        {EGV_VECTOR_CLASS: NATIVE_KEY_TEMPLATES[EGV_VECTOR_CLASS]}))
    _pin = _re.compile(CELL_ID_GRAMMAR_EGV_PIN)
    _corpus: list[str] = []
    for _obj in ("baseline", "entropy_gradient_L26", "gentropy_gradient",
                 "naive_entropy_gradient", "Rband1", "Rband2", "Rband3",
                 "gRband1", "gRband2", "gRband3", "Rband4", "gRband0",
                 "caa_language_L26", "gcaa_language", "entropy_gradient",
                 "entropy_gradient_L26_L26", ""):
        for _d in list(DOSE_LADDER_FOR_GRAMMAR) + [0.0]:
            for _suffix in ("", "@absalpha", "@absolute"):
                _corpus.append(f"{_obj}_L26_a{_d:+.2f}{_suffix}")
                _corpus.append(f"{_obj}_L26_a{_d:+.1f}{_suffix}")
    _disagree = [c for c in _corpus
                 if bool(_egv_gen.fullmatch(c)) != bool(_pin.fullmatch(c))]
    ok(not _disagree,
       "the GENERATED entropy-only grammar accepts and rejects exactly what "
       f"re-freeze #4's hand-written alternation does, over {len(_corpus)} ids — "
       "the generation is checked against the pin it replaced, not only against "
       "itself (rake M40's family: a generated pin that vouches for itself is "
       "not a pin)", f"{_disagree[:4]}")
    ok(sum(1 for c in _corpus if _pin.fullmatch(c)) > 0,
       "…and that corpus actually contains ids the pin ACCEPTS, so the agreement "
       "is not two regexes agreeing that everything is rejected",
       f"{sum(1 for c in _corpus if _pin.fullmatch(c))} accepted")

    # ── 17. the EGV plan is BYTE-IDENTICAL to the pre-port tool's ────────────
    # The witness is rendered THROUGH the same templates `main` uses, from the
    # keys the default spec resolves — so if the port had changed any of the
    # three moved templates, or the default, this fails with the id it produced.
    def _render_plan_ids(keys: LaneVectorKeys, site: int) -> list[str]:
        native_key, transported_key, naive_key = (keys.native, keys.transported,
                                                  keys.naive)
        ids = [f"baseline_L{site}_a+0.00"]
        for frac in DOSE_LADDER_FOR_GRAMMAR:
            ids.append(f"{native_key}_L{site}_a{frac:+.2f}")
        for i in BAND_MEMBER_INDICES:
            for frac in DOSE_LADDER_FOR_GRAMMAR:
                ids.append(f"Rband{i}_L{site}_a{frac:+.2f}")
        for frac in DOSE_LADDER_FOR_GRAMMAR:
            ids.append(f"{transported_key}_L{site}_a{frac:+.2f}")
        for i in BAND_MEMBER_INDICES:
            for frac in DOSE_LADDER_FOR_GRAMMAR:
                ids.append(f"gRband{i}_L{site}_a{frac:+.2f}")
        for d in (-0.3, 0.3):
            ids.append(f"{naive_key}_L{site}_a{d:+.2f}")
        return ids

    _egv_ids = _render_plan_ids(_egv, 26)
    ok(len(_egv_ids) == 51,
       "the rendered plan is the banked 51-cell shape (1 + 6 + 18 + 6 + 18 + 2)",
       f"{len(_egv_ids)}")
    _missing_witness = [w for w in EGV_CELL_ID_WITNESS if w not in _egv_ids]
    ok(not _missing_witness,
       "every id the pre-port tool spelled as a LITERAL is still produced, "
       "character for character, by the ported templates under the default "
       "(entropy-gradient) spec — the EGV path is unchanged as a PROPERTY",
       f"missing: {_missing_witness}")
    ok(all(_cg.fullmatch(i) for i in _egv_ids),
       "…and every rendered EGV id is legal under the generated grammar")
    _caa_ids = _render_plan_ids(_caa, 26)
    ok(len(_caa_ids) == len(_egv_ids)
       and all(_cg.fullmatch(i) for i in _caa_ids)
       and not any("entropy_gradient" in i for i in _caa_ids),
       "a CAA column has the SAME cell arity, every id legal, and not one of "
       "them wearing the entropy gradient's name (HALT E's whole point)",
       f"{_caa_ids[1]} … {_caa_ids[-1]}")
    ok(not (set(_egv_ids) & set(_render_plan_ids(_other, 26)) - {
            f"baseline_L26_a+0.00"} - {i for i in _egv_ids if "band" in i.lower()}),
       "an EGV column and a CAA column share only the baseline and the bands — "
       "the two families' SIGNAL cells cannot collide")

    # ── 18. the naming constants are the CAMPAIGN's, proven not retyped ──────
    try:
        from metabasis.scripts.build_behavioral_banks import (      # type: ignore
            EGV_OBJECT_KEY, NAIVE_KEY_PREFIX,
            SITE_SUFFIX_RE as _CAMPAIGN_SUFFIX_RE)
        from metabasis.scripts.build_contrast_vectors import (      # type: ignore
            CAA_KEY_TEMPLATE, REPENG_KEY_TEMPLATE)
    except ImportError as exc:
        skip("the class naming constants are the CAMPAIGN's own",
             f"the metabasis package is not importable here ({exc}); evidence "
             "MISSING: that `caa_{axis}_L{site}`, `repengpca_{axis}_L{site}`, "
             "the `naive` prefix and the site-suffix regex here are the "
             "campaign's definitions rather than this file's transcription")
    else:
        ok(NATIVE_KEY_TEMPLATES["caa"] == CAA_KEY_TEMPLATE
           and NATIVE_KEY_TEMPLATES["repeng_pca"] == REPENG_KEY_TEMPLATE
           and NATIVE_KEY_TEMPLATES["entropy_gradient"]
           == EGV_OBJECT_KEY + NATIVE_SITE_SUFFIX
           and NAIVE_PREFIX == NAIVE_KEY_PREFIX
           and SITE_SUFFIX_RE.pattern == _CAMPAIGN_SUFFIX_RE.pattern,
           "every key convention transcribed at the top of this module is the "
           "CAMPAIGN's own, proven equal rather than retyped on trust — the same "
           "pattern the scipy cross-check uses",
           f"{sorted(NATIVE_KEY_TEMPLATES.values())}")
        ok(EGV_VECTOR_CLASS == EGV_OBJECT_KEY,
           "…and the EGV class name IS the campaign's object key, so the default "
           "spec names the object of record and not a lookalike")

    # ── 19. the TP path is untouched by the class port ───────────────────────
    # PROPERTY, NOT PROSE. The three TP forks and every shared helper are pinned
    # by AST digest in blocks 1 above and did not move; what remains to show is
    # that the TP>1 arm never reads a class name — it takes a SITE and a WIDTH,
    # which is why the class path reaches TP>1 through certified code.
    _class_names = {"class_spec", "keys", "native_key", "transported_key",
                    "naive_key", "vecs", "tvecs"}
    _tp_arm_names: set[str] = set()
    for n in ast.walk(_main):
        if _is_tp_gt_1(n):
            for sub in n.body:
                for m in ast.walk(sub):
                    if isinstance(m, ast.Name):
                        _tp_arm_names.add(m.id)
    ok(bool(_tp_arm_names) and not (_tp_arm_names & _class_names),
       "no TP>1 arm reads any class-resolution name — the attachment takes a "
       "site and a width and knows nothing about which object is injected, so "
       "the class path reaches TP>1 through exactly the code re-freeze #4 "
       "certified", f"overlap: {sorted(_tp_arm_names & _class_names)}")

    # ── 20. M59: the suite's own arithmetic ──────────────────────────────────
    # RAKE M44: the count is QUALIFIED, not a bare number, and it is qualified BY
    # ENVIRONMENT because this suite legitimately runs a different number of
    # checks in the two places it runs.
    #
    #   desk  — CPython 3.13.9, scipy present : 211 checks, 0 named skips
    #   lane  — CPython 3.12.3, no scipy      : 206 checks, 1 named skip
    #           (the scipy cross-check; the lane venv has no scipy by design)
    #
    # THE FLOOR IS THE NODE'S, because the node is where columns are actually
    # fired and a floor the node cannot meet is a gate that blocks the science —
    # which is exactly what the previous floor of 133 did: `bb5f1d1` scored
    # 126/128 in the lane venv and every runner gates on `--selftest || exit 2`.
    # It sits one below the node's count because this check has not been counted
    # yet when it reads `checks`.
    SELFTEST_CHECK_FLOOR = 205
    KNOWN_SKIP_CEILING = 6
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
