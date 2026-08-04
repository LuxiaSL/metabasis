"""A8 Leg-0 — T4+T5: Phase A paired collection (GPU) + bitwise spot-replay (CP-1 gate).

Forced replay of every corpus text through ONE model, both template arms (never mixed),
capturing the per-text MEAN residual state over completion positions at the site grid.
Banks raw fp32 means + per-site median norms via fit_transport_maps.save_state_bank (the one
contract). Pre-normalizes NOTHING (normalization happens at fit time, in the fit stamp).

Conventions mirrored from the a5 machinery (the banked vectors' native space):
  - site L = forward_pre_hook on decoder_layers(model)[L] — the residual ENTERING
    layer L; batch-1; use_cache=False; .float() cast from the bf16 forward.
  - model load: AutoModelForCausalLM(dtype=bfloat16, attn_implementation="eager").
    bf16 forward + fp32 banked means per Luxia's dtype ruling 2026-07-22 — CERTIFIED
    BY the CP-1 bitwise spot-replay gate, not by assumption. Pin ONE card per model
    for the whole session (launcher sets CUDA_VISIBLE_DEVICES; determinism is
    within-model/within-card).
  - native arm: chat template from the manifest's system/user prompts,
    add_generation_prompt=True, date_string=VMB_CANONICAL_DATE (Llama strftime_now
    pin); completion ids = tokenize(text, no specials) appended after the prompt.
  - raw arm: tokenize(text, with specials); completion positions = everything after
    the specials prefix (computed, not assumed).

Re-tokenization caveat (GP-Q9, rake-report item): the corpus banks TEXT, not token
ids — fresh tokenization is the defined object for cross-replay, and Leg-0's shared
tokenizer guarantees both models see IDENTICAL token sequences per (text, arm).
Cross-tokenizer legs (Leg 1+) revisit this explicitly.

Sharded path (prereg §4, big-rung engineering): models too large for one card
(70B/405B class) load across the node with an accelerate `device_map` instead of
`.to(device)`. OPT-IN ONLY — `--device-map` / `--shard-across` unset reproduces the
single-card path byte for byte (same loader, same hooks, same reductions, same
stamp keys). The realized sharding layout goes into the trunk stamp; cross-layout
determinism is certified by the collect+spot-replay-in-one-job gate, never assumed.

Node of origin (rake M41, 2026-07-29): the trunk stamp carries `hostname`, so every
banked artifact can be attributed to the machine that produced it. Before this, no
bank identified its node and "where does artifact X live" was answerable only by
filesystem probe — which is how a 405B readout came to expect scan fits on one node
that lived on the other. The field is UNCONDITIONAL (a hostname is an identity, not
a deviation); it can never fail a run (rake M19 — it degrades to the named sentinel
`HOSTNAME_UNRESOLVED`); and READERS tolerate its absence through `stamp_hostname`,
which returns None for every pre-M41 stamp and keeps that case distinct from a
collector that ran and could not name its host.

Dtype regime (prereg ADDENDUM 2026-07-26-A, roster row 21): a checkpoint whose own
config carries a block-wise FP8 `quantization_config` would load its NATIVE FP8
forward — a different object from every other bank, and not differentiable. The
addendum rules DeepSeek-V3 into the standard bf16 regime via dequantize-on-load;
`--dequantize-fp8` is that, OPT-IN, with a post-load witness (no FP8 modules survive,
every float parameter is bfloat16) asserted and banked in the trunk stamp. Unset =
the historical from_pretrained call, byte for byte.

CPU self-test (no weights, no GPU, no data tree):

  python -m metabasis.scripts.collect_mean_states --selftest

RAKE M44 — THE CONFIGURATION MATRIX IS THE MERGE BAR. Two things the self-test
reaches for are properties OF THE ENVIRONMENT rather than of the code under test,
and each is present in some legitimate configuration and absent in another: the
deep-learning stack (torch + transformers, which `trunk_stamp` and the post-load
witness need, and which the desk's own repo `.venv` does not carry), and a WORKING
fp8 dequantize (transformers 4.51.3's `FineGrainedFP8Config` accepts
`dequantize=True` and silently drops it, so `fp8_dequantize_config()` refuses —
correctly, at runtime). Neither absence is a test failure. Each is therefore
branched on ONE availability probe and degrades to a NAMED skip counted in the
tail, so coverage is REPORTED per configuration rather than inferred, and the
self-test exits 0 in all four cells of {stack, no stack} x {fp8, no fp8}.

Modes (one model per invocation; run once per model on the assigned card):
  --collect              full pass over the corpus, both arms (or --arms native)
  --spot-replay K        FRESH-PROCESS re-run of K stratified texts per arm, byte-
                         compared against the banked means (the CP-1 gate; K>=3)
  --cp1-summary          the compact CP-1 table from banked stamps + spot results
                         (run locally after rsyncing states/ back; needs no GPU)

⚠ --collect AND --spot-replay IN ONE INVOCATION IS A REFUSAL (desk ruling
2026-08-03). It used to be accepted and then IGNORED — the dispatch returned
after collecting, so the gate silently never ran and the job's JOB-OK line
pointed at a spot report that was never written. The gate is TWO PROCESSES IN
ONE JOB, because a replay inside the collecting process would certify a model
object that never left memory.

Launch template (node-side, after venv activation, from the deploy root) — build
the flags ONCE so the two invocations provably cannot drift apart:
  ARGS=(--model 3b --model-path <LOCAL_WEIGHTS_DIR> --arm-root <ARM_ROOT>)
  python -m metabasis.scripts.collect_mean_states --collect "${ARGS[@]}" \
  && python -m metabasis.scripts.collect_mean_states --spot-replay 3 "${ARGS[@]}"
  # big rung, whole node (CUDA_VISIBLE_DEVICES left unpinned):
  OMP_NUM_THREADS=1 python -m metabasis.scripts.collect_mean_states --collect \
    --model 70b --model-path <LOCAL_WEIGHTS_DIR> --arm-root <ARM_ROOT> \
    --device-map auto --max-memory 0=170GiB,1=170GiB,...,cpu=0GiB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from metabasis.scripts.fit_transport_maps import (
    ARMS, DEFAULT_ARM_ROOT, MODEL_KEYS, StateBank, StrataDerivationError,
    derive_strata, load_state_bank, save_state_bank, sites_for, stratum_counts)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("collect_mean_states")

# THE STRATUM VOCABULARY IS BASIS-DERIVED (desk ruling 2026-08-03; brief
# BRIEF-strata-basis-fix-2026-08-03). This module used to carry
#
#     STRATA = ("S1", "S2", "S3", "S5")
#
# and `pick_spot_ids` keyed a dict off it, so the CP-1 spot-replay gate died with
# `KeyError: 'wikitext'` the first time it met the webtext-v3 basis (canary,
# 2026-08-03) — AFTER a full collection pass, with the bank already written.
# `derive_strata` (defined once, in `fit_transport_maps`, so the collector and
# the fitter can never disagree about the vocabulary) reads the names off the
# pinned manifest's per-entry `stratum` field in FIRST-OCCURRENCE order, which
# makes them a pure function of the manifest bytes — deterministic under the
# `corpus_manifest_sha256` every stamp already carries.
#
# THE GATE'S IDENTITY ON A v2.1 CORPUS DOES NOT MOVE, and that is proven rather
# than argued (selftest 8): the round-robin skips an empty stratum WITHOUT
# consuming a pick, so cycling the legacy 4-tuple over a manifest that holds only
# S1/S2/S3 emits exactly the sequence cycling the derived 3-tuple does. The
# recorded pre-change selection is asserted byte-exact against the derived path.

#: What this module hardcoded before 2026-08-03 — DOCUMENTATION ONLY, and the
#: provenance of selftest 8's recorded expectations. Nothing reads it to make a
#: decision, and nothing may start to.
LEGACY_V21_STRATA: tuple[str, ...] = ("S1", "S2", "S3", "S5")

# The Llama strftime_now chat-template date pin. Historically imported from the
# anamnesis stage-0 generator (`vmb_stage0_generate.VMB_CANONICAL_DATE`) — that
# import eagerly dragged the entire 2,000+-line extraction stack in for one
# string, so the metabasis lift hardcodes it (pull-package bootstrap resolution,
# 2026-07-26). Project DATA, not logic: changing it changes native-arm token
# sequences and breaks bitwise parity with every banked state collection.
VMB_CANONICAL_DATE = "12 Jul 2026"


# ---------------------------------------------------------------- corpus + ids
def load_corpus(arm_root: Path) -> tuple[list[dict], str]:
    path = arm_root / "corpus" / "corpus_manifest.json"
    with open(path) as f:
        entries = json.load(f)["entries"]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if not entries:
        raise RuntimeError(f"{path}: empty corpus")
    return entries, h


def build_ids(tok, entry: dict, arm: str, date_string: str,
              max_length: int | None = None) -> tuple[list[int], int]:
    """Returns (ids, P) — full token ids and the completion start position.

    `max_length` is the prereg ADDENDUM 2026-07-27-B position-ceiling deviation:
    per-text truncation to the FIRST `max_length` tokens, for architectures whose
    LEARNED absolute position embeddings cannot represent a longer sequence
    (GPT-2: n_positions=1024, a hard ceiling on every variant). Default None =
    the historical path, untouched.

    Truncation is a pure suffix-drop applied AFTER the arm's ids are built, so:
      * a text already at or under the ceiling is returned byte-identical to the
        untruncated path (the slice never runs), and
      * because collection is batch-1 with use_cache=False, truncating one text
        cannot perturb any other text's forward — the unaffected texts stay
        byte-identical to the frozen objects, per the addendum.
    """
    text = entry["text"]
    comp = tok.encode(text, add_special_tokens=False)
    if not comp:
        raise RuntimeError(f"{entry['text_id']}: text tokenizes to nothing")
    if max_length is not None and max_length < 2:
        raise ValueError(f"max_length={max_length}: need room for a completion position")
    if arm == "native":
        msgs = []
        if entry.get("system_prompt"):
            msgs.append({"role": "system", "content": entry["system_prompt"]})
        msgs.append({"role": "user", "content": entry["user_prompt"]})
        try:
            res = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          date_string=date_string)
        except TypeError:      # date-free templates (e.g. Qwen) reject the kwarg
            res = tok.apply_chat_template(msgs, add_generation_prompt=True)
        prompt_ids = list(res["input_ids"] if hasattr(res, "keys") else res)
        return _truncate(prompt_ids + comp, len(prompt_ids), max_length, entry)
    if arm == "raw":
        ids = tok.encode(text, add_special_tokens=True)
        p = len(ids) - len(comp)
        if p < 0 or ids[p:] != comp:
            # specials interleaved unexpectedly — fall back to explicit BOS prefix
            bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
            ids, p = bos + comp, len(bos)
        return _truncate(ids, p, max_length, entry)
    raise ValueError(f"unknown arm {arm!r}")


def _truncate(ids: list[int], p: int, max_length: int | None,
              entry: dict) -> tuple[list[int], int]:
    """ADDENDUM 2026-07-27-B suffix-drop. No-op when the text already fits."""
    if max_length is None or len(ids) <= max_length:
        return ids, p
    if p >= max_length:
        raise RuntimeError(
            f"{entry['text_id']}: truncating to {max_length} would leave no "
            f"completion positions (prompt/specials prefix is {p} tokens). The "
            "mean state is defined over completion positions only, so this text "
            "has no banked object under the deviation — surface, do not silently "
            "bank a prompt-only mean.")
    return ids[:max_length], p


# ---------------------------------------------------------------- state capture
class SiteCapture:
    """forward_pre_hooks on the site layers; grabs input hidden_states per forward."""

    def __init__(self, model, sites: tuple[int, ...]):
        from metabasis.extraction.hooks import decoder_layers
        layers = decoder_layers(model)
        self.grab: dict[int, "object"] = {}
        self.handles = []
        for s in sites:
            def hook(module, hook_args, hook_kwargs, _s=s):
                hs = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
                self.grab[_s] = hs.detach()
                return None
            self.handles.append(
                layers[s].register_forward_pre_hook(hook, with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()


def compute_means(model, tok, entries: list[dict], sites: tuple[int, ...], arm: str,
                  date_string: str, device: str, log_every: int = 50,
                  max_length: int | None = None
                  ) -> tuple[dict[int, np.ndarray], dict[int, list[float]], list[int],
                             list[str]]:
    """One forward per text; per-text fp32 mean over completion positions per site.
    Also returns per-position residual norms (a5 dose-currency comparability),
    sequence lengths, and the ids of any texts the position ceiling truncated."""
    import torch

    cap = SiteCapture(model, sites)
    means: dict[int, list[np.ndarray]] = {s: [] for s in sites}
    tok_norms: dict[int, list[float]] = {s: [] for s in sites}
    seq_lens: list[int] = []
    truncated: list[str] = []
    t0 = time.time()
    try:
        for i, e in enumerate(entries):
            ids, p = build_ids(tok, e, arm, date_string, max_length=max_length)
            # Hitting the ceiling is *necessary* for truncation but not sufficient —
            # a text whose natural length is exactly max_length was not truncated.
            # Re-tokenize only in that rare boundary case, so the recorded count is
            # exact rather than an upper bound.
            if max_length is not None and len(ids) == max_length:
                if len(build_ids(tok, e, arm, date_string)[0]) > max_length:
                    truncated.append(e["text_id"])
            t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(t, use_cache=False, return_dict=True)
            for s in sites:
                h = cap.grab[s][0, p:, :].float()          # [n_comp, d] fp32
                if h.shape[0] == 0:
                    raise RuntimeError(f"{e['text_id']}: no completion positions")
                means[s].append(h.mean(dim=0).cpu().numpy().astype(np.float32))
                tok_norms[s].append(float(h.norm(dim=-1).median().cpu()))
            seq_lens.append(len(ids))
            if (i + 1) % log_every == 0:
                logger.info("[%s/%s] %d/%d texts (%.2fs/text)", arm, sites, i + 1,
                            len(entries), (time.time() - t0) / (i + 1))
    finally:
        cap.close()
    return ({s: np.stack(v) for s, v in means.items()},
            {s: v for s, v in tok_norms.items()}, seq_lens, truncated)


def load_model_and_tok(model_path: str, device: str, dequantize_fp8: bool = False):
    """The single-card loader. `dequantize_fp8=False` (the default) reproduces the
    historical call byte for byte — `_fp8_kwargs` contributes NO kwargs at all."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        **_fp8_kwargs(dequantize_fp8),
    ).to(device).eval()
    if dequantize_fp8:
        assert_dequantized(model, model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok


# ---------------------------------------------------------------- sharded load (big rungs)
# Prereg §4 (big-rung engineering): ">=70B-class nodes collect via the collector's
# sharded path (device_map across the 8-GPU node; determinism certified by the same
# collect+spot-replay-in-one-job gate — sharding layout recorded in the trunk stamp)".
#
# Everything below is OPT-IN. With neither --device-map nor --shard-across the loader,
# the capture hooks and compute_means are the byte-identical single-card path that
# banked the smalls. The sharded branch changes exactly two things: HOW weights are
# placed (accelerate device_map, never `.to(device)` on a dispatched model) and WHERE
# the input id tensor is built (the input-embedding execution device instead of
# --device). SiteCapture and compute_means are untouched; nothing is pre-normalized.
#
# Hook ordering (VERIFIED against accelerate 1.14 hooks.py + torch nn/modules/module.py,
# not assumed): accelerate does NOT dispatch via torch forward hooks. add_hook_to_module()
# REPLACES `module.forward` with a wrapper that runs hook.pre_forward() (weight on-load +
# send_to_device of args/kwargs) and only then the original forward. torch's
# nn.Module._call_impl resolves `forward_call = self.forward` and runs
# `self._forward_pre_hooks` BEFORE invoking it. So SiteCapture's forward_pre_hook fires
# FIRST and sees the real hidden_states exactly as decoder layer L-1 emitted them — on
# L-1's card, before accelerate's cross-card copy. Device-to-device `.to()` is a bit-exact
# copy for bf16, so the captured values equal what layer L consumes; and our hook returns
# None, so it never perturbs the args accelerate subsequently dispatches. (The only torch
# forward_pre_hook accelerate registers anywhere is _attach_context_parallel_hooks, which
# is context-parallel-only and never fires on a device_map load.)
#
# Layout-dependence, stated precisely: under a pure pipeline-parallel device_map every
# module — hence every matmul and every norm reduction — stays WHOLE on one card, so no
# reduction is split or re-ordered by sharding; only which card runs it changes. The
# per-text reductions in compute_means (`h.mean(dim=0)`, `h.norm(dim=-1).median()`) run on
# whatever card holds the site layer's input, i.e. sharding can move a reduction from card
# a to card b. Identical GPU model + identical kernels => expected bitwise identical, but
# EXPECTED IS NOT CERTIFIED: the sharded-vs-single-device byte compare in one job is what
# certifies it. Tensor-parallel / split-module maps WOULD change reduction order and are
# out of scope here (device_map is pipeline-parallel by construction: no_split_module_classes
# keeps each decoder layer whole).


# ---------------------------------------------------------------- dtype regime (row 21)
# Prereg ADDENDUM 2026-07-26-A, BINDING for roster row 21 (DeepSeek-V3): a checkpoint
# whose own config.json carries `quantization_config: {quant_method: "fp8", ...}` would
# otherwise load through the native-FP8 forward, which is a DIFFERENT object from every
# other bank (raw Triton block-wise kernels) and — per the FP8 lane design report — is
# not differentiable, so the entropy-gradient target build would silently return a
# residual-highway-only vector. The ruling is that DSV3 collects and builds in bf16 via
# dequantize-on-load, i.e. the standard roster regime.
#
# OPT-IN ONLY, exactly like the sharded path: with --dequantize-fp8 unset, from_pretrained
# is called with the identical kwargs it has always been called with, so every existing
# bank's loader path is untouched byte for byte. The flag is also a no-op-with-a-loud-
# failure on a checkpoint that carries no FP8 quantization_config: passing it there is a
# staging error, and `assert_dequantized` surfaces it rather than banking a surprise.
DEQUANTIZED_STAMP_KEY = "fp8_dequantize"

# RAKE M41 (2026-07-29): the trunk stamp records the NODE OF ORIGIN. Before this,
# no banked artifact identified the machine that produced it, so "where does
# artifact X live" was answerable only by filesystem probe — and the 405B readout
# block paid for it, expecting scan fits on one node that lived on the other, with
# an enactor's plausible inference retracted after a cross-node probe found them.
#
# The field is UNCONDITIONAL. Every other stamp deviation (`sharding`,
# `fp8_dequantize`, `truncation`) is conditional so that an absent key means "the
# deviation was not taken" and historical stamps stay comparable key-for-key. A
# hostname is not a deviation — it is the identity of the machine — and making it
# conditional would leave exactly the gap M41 was filed for.
#
# BACKWARD COMPATIBILITY is the READER's contract, not the writer's: every stamp
# banked before this date has no hostname, and `stamp_hostname` distinguishes
# "predates the field" (None) from "the collector could not resolve one"
# (HOSTNAME_UNRESOLVED). Named per rake M26 so a mechanical sweep can see both.
HOSTNAME_STAMP_KEY = "hostname"
#: Recorded when the collector ran but could not name its host. Deliberately a
#: NAMED sentinel rather than None or "": degraded instrumentation and an absent
#: field demand different responses, and a reader must be able to tell them apart.
HOSTNAME_UNRESOLVED = "HOSTNAME_UNRESOLVED"


class Fp8RegimeError(RuntimeError):
    """The requested FP8 dtype regime could not be established — never a silent fallback.

    Raised when --dequantize-fp8 is asked for and transformers cannot provide it, or
    when the LOADED model still carries FP8 modules/dtypes after the request. Both are
    fatal by construction: banking states from the wrong forward is exactly the silent
    wrong answer ADDENDUM 2026-07-26-A exists to prevent.
    """


def fp8_dequantize_config() -> Any:
    """`FineGrainedFP8Config(dequantize=True)`, or a loud refusal.

    Imported lazily and by name so the single-card/no-FP8 paths never depend on the
    quantization stack being present in the environment.
    """
    try:
        from transformers import FineGrainedFP8Config
    except ImportError as exc:                      # noqa: TRY003 — message is the point
        raise Fp8RegimeError(
            "--dequantize-fp8 needs transformers' FineGrainedFP8Config (the "
            "dequantize-on-load path of prereg ADDENDUM 2026-07-26-A); this "
            "transformers build does not expose it") from exc
    try:
        cfg = FineGrainedFP8Config(dequantize=True)
    except TypeError as exc:
        raise Fp8RegimeError(
            "FineGrainedFP8Config in this transformers build has no `dequantize` "
            "parameter — the addendum's regime cannot be established here") from exc
    if not getattr(cfg, "dequantize", False):
        raise Fp8RegimeError(
            "FineGrainedFP8Config(dequantize=True) did not set dequantize — refusing "
            "to load, because the difference is a native-FP8 forward vs a bf16 one")
    return cfg


# The module types transformers substitutes for `nn.Linear` / fused expert containers
# when a block-wise FP8 checkpoint is loaded NATIVELY. Their ABSENCE after a
# dequantize-on-load is the post-load witness that the regime really is bf16 — checked
# by name (not by import) so a transformers reorganization degrades to "no witness
# found", which `assert_dequantized` treats as a failure, not a pass.
FP8_MODULE_TYPE_NAMES = ("FP8Linear", "FP8Expert")


def assert_dequantized(model: Any, model_path: str) -> dict:
    """Post-load certification that the FP8 checkpoint really loaded as bf16.

    Returns the witness record banked in the trunk stamp. Raises `Fp8RegimeError` on
    any surviving FP8 module or non-bf16 floating parameter — the design report's
    "post-load certification assertions", as code rather than as a checklist.
    """
    import torch

    offenders = sorted({type(m).__name__ for m in model.modules()
                        if type(m).__name__ in FP8_MODULE_TYPE_NAMES})
    if offenders:
        raise Fp8RegimeError(
            f"{model_path}: {offenders} modules survived a --dequantize-fp8 load — the "
            "forward is still native FP8, which is NOT the regime ADDENDUM "
            "2026-07-26-A ruled for this node")
    dtypes: dict[str, int] = {}
    bad: list[str] = []
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        key = str(p.dtype)
        dtypes[key] = dtypes.get(key, 0) + 1
        if p.is_floating_point() and p.dtype is not torch.bfloat16 and len(bad) < 8:
            bad.append(f"{name}:{p.dtype}")
    if bad:
        raise Fp8RegimeError(
            f"{model_path}: floating parameters that are not bfloat16 after "
            f"--dequantize-fp8: {bad} (dtype census {dtypes})")
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    return {"requested": "FineGrainedFP8Config(dequantize=True)",
            "prereg": "ADDENDUM 2026-07-26-A",
            "fp8_modules_remaining": 0,
            "param_dtype_census": dtypes,
            "config_quantization_config": (dict(qcfg.to_dict())
                                           if hasattr(qcfg, "to_dict") else str(qcfg))}


def _fp8_kwargs(dequantize_fp8: bool) -> dict[str, Any]:
    """The from_pretrained kwargs the dtype regime adds — EMPTY unless opted in."""
    return {"quantization_config": fp8_dequantize_config()} if dequantize_fp8 else {}


class ShardedLoadError(RuntimeError):
    """Base class for every failure specific to the sharded (device_map) load path."""


class AccelerateUnavailableError(ShardedLoadError):
    """A sharded load was requested but `accelerate` is not importable."""


class DeviceMapSpecError(ShardedLoadError):
    """--device-map / --shard-across / --max-memory could not be parsed or is inconsistent."""


class ShardLayoutError(ShardedLoadError):
    """The realized layout is unusable: weights spilled to cpu/disk (model too big for the
    given --max-memory), or a single device where a multi-device layout was required."""


class SiteLayerUnresolvableError(ShardedLoadError):
    """A requested site has no decoder layer, or its device is not resolvable in the layout."""


DeviceKey = Union[int, str]


@dataclass(frozen=True)
class ShardSpec:
    """The opt-in sharded-load request, fully resolved from the CLI.

    Exactly one of `device_map` / `shard_across` is set. `device_map` is either an
    accelerate strategy string ("auto" | "balanced" | "balanced_low_0" | "sequential")
    or an explicit {module_name: device} dict. `shard_across` is a device list from
    which an explicit EVEN decoder-layer split is built (the forced multi-device layout
    the certification gate needs on a model that would otherwise fit one card).
    """

    device_map: Optional[Union[str, dict[str, DeviceKey]]] = None
    shard_across: Optional[list[DeviceKey]] = None
    max_memory: Optional[dict[DeviceKey, Union[int, str]]] = None
    offload_folder: Optional[str] = None
    allow_offload: bool = False
    require_multi_device: bool = False
    spec_echo: str = ""

    def __post_init__(self) -> None:
        if (self.device_map is None) == (self.shard_across is None):
            raise DeviceMapSpecError(
                "ShardSpec needs exactly one of device_map / shard_across")


def _normalize_device(dev: Any) -> str:
    """int | str | torch.device -> canonical torch device string; 'disk' passes through."""
    import torch
    if isinstance(dev, str) and dev.strip().lower() == "disk":
        return "disk"
    try:
        return str(dev if isinstance(dev, torch.device) else torch.device(dev))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DeviceMapSpecError(f"unusable device {dev!r} in a device map") from exc


def _device_key(tokenstr: str) -> DeviceKey:
    """'0' -> 0 (accelerate's GPU-ordinal form); 'cpu'/'disk'/'cuda:1' stay strings.
    Mixing int 0 and 'cuda:0' in ONE map would look like two devices to accelerate's
    `len(set(device_map.values())) > 1` dispatch test, so the two forms are never mixed."""
    t = tokenstr.strip()
    if not t:
        raise DeviceMapSpecError("empty device token")
    return int(t) if t.isdigit() else t


def _json_arg(spec: str, flag: str) -> Any:
    """A CLI value that is either '@path/to.json' or an inline JSON document."""
    raw = spec.strip()
    try:
        if raw.startswith("@"):
            return json.loads(Path(raw[1:]).read_text())
        return json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeviceMapSpecError(f"{flag}: cannot read {spec!r} as JSON ({exc})") from exc


ACCELERATE_STRATEGIES = ("auto", "balanced", "balanced_low_0", "sequential")


def parse_device_map(spec: str) -> Union[str, dict[str, DeviceKey]]:
    """--device-map: an accelerate strategy name, or '@map.json' / inline JSON dict."""
    if spec in ACCELERATE_STRATEGIES:
        return spec
    payload = _json_arg(spec, "--device-map")
    if not isinstance(payload, dict) or not payload:
        raise DeviceMapSpecError(
            f"--device-map: expected one of {ACCELERATE_STRATEGIES} or a non-empty "
            f"{{module: device}} JSON object, got {type(payload).__name__}")
    return {str(k): (v if isinstance(v, int) else _device_key(str(v)))
            for k, v in payload.items()}


def parse_shard_across(spec: str) -> list[DeviceKey]:
    """--shard-across: 'N' (=> GPU ordinals 0..N-1) or an explicit comma device list
    ('0,1,2,3' | 'cuda:0,cuda:1' | 'cpu,disk' for the no-GPU smoke)."""
    s = spec.strip()
    if s.isdigit():
        n = int(s)
        if n < 1:
            raise DeviceMapSpecError("--shard-across N: need N >= 1")
        return list(range(n))
    devices = [_device_key(t) for t in s.split(",") if t.strip()]
    if not devices:
        raise DeviceMapSpecError(f"--shard-across: no devices parsed from {spec!r}")
    if len(set(map(str, devices))) != len(devices):
        raise DeviceMapSpecError(f"--shard-across: duplicate devices in {spec!r}")
    return devices


def parse_max_memory(spec: str) -> dict[DeviceKey, Union[int, str]]:
    """--max-memory: '0=170GiB,1=170GiB,cpu=0GiB' (accelerate size strings), or
    '@mm.json' / inline JSON. cpu=0GiB is the usual way to forbid silent CPU offload."""
    s = spec.strip()
    if s.startswith("@") or s.startswith("{"):
        payload = _json_arg(s, "--max-memory")
        if not isinstance(payload, dict) or not payload:
            raise DeviceMapSpecError("--max-memory: expected a non-empty JSON object")
        return {_device_key(str(k)): v for k, v in payload.items()}
    out: dict[DeviceKey, Union[int, str]] = {}
    for item in s.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise DeviceMapSpecError(
                f"--max-memory: {item!r} is not DEVICE=SIZE (e.g. 0=170GiB, cpu=0GiB)")
        key, _, val = item.partition("=")
        out[_device_key(key)] = val.strip()
    if not out:
        raise DeviceMapSpecError(f"--max-memory: nothing parsed from {spec!r}")
    return out


def build_even_layer_device_map(model_path: str, devices: list[DeviceKey]
                                ) -> dict[str, DeviceKey]:
    """Explicit device_map splitting the decoder layers EVENLY over `devices`.

    Built on a meta-device skeleton (no weights read), so it is architecture-generic:
    the decoder-layer container is resolved with the same `decoder_layers()` the capture
    hooks use, and every non-layer tensor (embeddings, rotary buffers, final norm, head)
    goes to devices[0]. Placement order is irrelevant to correctness — accelerate moves
    activations between cards as needed — but keeping the whole layer list contiguous
    and each layer WHOLE on one card is what makes the map pipeline-parallel (no split
    matmuls, no changed reduction order).

    This is the "forced multi-device layout" the certification gate runs a small model
    through; `--device-map auto` would put an 8B on a single card and certify nothing.
    """
    _require_accelerate()
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from metabasis.extraction.hooks import decoder_layers

    try:
        cfg = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(cfg)
    except Exception as exc:                       # arch/config we cannot instantiate
        raise DeviceMapSpecError(
            f"--shard-across: cannot build a meta skeleton for {model_path!r} ({exc}); "
            "pass an explicit map with --device-map @map.json instead") from exc

    layers = decoder_layers(skeleton)
    prefix = next((name for name, mod in skeleton.named_modules() if mod is layers), None)
    if prefix is None:
        raise DeviceMapSpecError(
            "--shard-across: decoder-layer container has no module path; "
            "pass --device-map @map.json instead")
    n_layers = len(layers)
    if len(devices) > n_layers:
        raise DeviceMapSpecError(
            f"--shard-across: {len(devices)} devices for {n_layers} decoder layers")

    dmap: dict[str, DeviceKey] = {
        f"{prefix}.{i}": devices[(i * len(devices)) // n_layers] for i in range(n_layers)}
    # Cover every remaining state_dict tensor at its owning-module path (accelerate's
    # check_device_map raises on any parameter no key covers).
    tensors = list(skeleton.named_parameters(remove_duplicate=False))
    tensors += list(skeleton.named_buffers(remove_duplicate=False))
    for name, _ in tensors:
        if name.startswith(f"{prefix}."):
            continue
        dmap.setdefault(name.rsplit(".", 1)[0] if "." in name else name, devices[0])
    logger.info("forced shard map: %d decoder layers over %s (+%d non-layer modules)",
                n_layers, [str(d) for d in devices], len(dmap) - n_layers)
    return dmap


def _require_accelerate() -> str:
    """Returns the accelerate version; raises the named error if it is not importable."""
    try:
        import accelerate
    except ImportError as exc:                     # noqa: TRY003 - message is the point
        raise AccelerateUnavailableError(
            "the sharded path needs `accelerate` (device_map dispatch). Install it into "
            "the node venv (`uv pip install accelerate`) — the single-card path does not "
            "need it, so this import is deliberately lazy.") from exc
    return str(accelerate.__version__)


def _module_device(module: Any) -> Optional[str]:
    """Where a submodule's compute actually happens, or None if unresolvable.

    Prefers a real (non-meta) tensor device; falls back to the accelerate hook's
    `execution_device`, which is the truth for modules whose weights are offloaded
    (their parameters sit on meta between forwards)."""
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.device.type != "meta":
            return str(tensor.device)
    hook = getattr(module, "_hf_hook", None)
    candidates = [hook, *getattr(hook, "hooks", ())] if hook is not None else []
    for h in candidates:
        exec_dev = getattr(h, "execution_device", None)
        if exec_dev is not None:
            return _normalize_device(exec_dev)
    return None


def input_device_of(model: Any) -> str:
    """The device the input id tensor must be built on: the input embedding's execution
    device. NEVER `.to(device)` a dispatched model — accelerate wraps `.to` with a warning
    and raises outright once anything is offloaded."""
    emb = model.get_input_embeddings()
    if emb is None:
        raise ShardLayoutError(
            "model exposes no input embedding; cannot place inputs for the sharded path")
    dev = _module_device(emb)
    if dev is None or dev == "disk":
        raise ShardLayoutError(f"input embedding device unresolvable (got {dev!r})")
    return dev


def describe_shard_layout(model: Any, sites: tuple[int, ...], spec: ShardSpec,
                          accelerate_version: str, resolved_map: Any) -> dict:
    """The sharding record the prereg requires in the trunk stamp.

    `hf_device_map` is accelerate's own map — present only when it actually dispatched
    (transformers skips dispatch when the map resolves to one device), so the layout is
    ALSO derived from the live module tree, which is always true and version-independent.
    """
    import torch

    from metabasis.extraction.hooks import decoder_layers

    layers = decoder_layers(model)
    layer_devices: dict[str, str] = {}
    for i, layer in enumerate(layers):
        dev = _module_device(layer)
        if dev is None:
            raise ShardLayoutError(f"decoder layer {i}: device unresolvable in the layout")
        layer_devices[str(i)] = dev            # COMPUTE device (offloaded weights report
    for s in sites:                            # their accelerate execution_device, not
        if not 0 <= s < len(layers):           # the meta/cpu/disk they are stored on)
            raise SiteLayerUnresolvableError(
                f"site L{s} out of range: model has {len(layers)} decoder layers")
        if layer_devices[str(s)] == "disk":
            raise SiteLayerUnresolvableError(
                f"site L{s} resolves to 'disk'; its capture would have no compute device")

    compute_devices = sorted(set(layer_devices.values()))
    # An EXPLICIT map that misses part of the model is silently completed with "cpu" by
    # transformers (and, if it resolves to one device, never reaches accelerate's own
    # coverage check because dispatch is skipped) — so run that check ourselves.
    if isinstance(resolved_map, dict):
        from accelerate.utils import check_device_map
        try:
            check_device_map(model, resolved_map)
        except ValueError as exc:
            raise DeviceMapSpecError(f"device map does not cover the model: {exc}") from exc
    # STORAGE spill is a different question from compute placement, and it is the one
    # "model too big for max_memory" actually asks: read it off the map accelerate used.
    # Fold in any decoder layer whose COMPUTE landed on cpu — that is the same failure
    # wearing a different hat (a card ran out of room, or the map named a device wrong).
    hf_map = getattr(model, "hf_device_map", None)
    storage_map: dict[str, str] = {}
    if isinstance(hf_map, dict):
        storage_map = {k: _normalize_device(v) for k, v in hf_map.items()}
    elif isinstance(resolved_map, dict):
        storage_map = {k: _normalize_device(v) for k, v in resolved_map.items()}
    spilled = sorted({d for d in storage_map.values()
                      if d == "disk" or d.startswith("cpu")}
                     | {d for d in compute_devices if d.startswith("cpu")})
    if spilled and not spec.allow_offload:
        raise ShardLayoutError(
            f"weights spilled to {spilled} — the model does not fit the given "
            "--max-memory on the requested devices. Raise --max-memory / add devices, or "
            "pass --allow-offload to accept the (slow, compute-still-on-device) spill.")
    if spec.require_multi_device and len(compute_devices) < 2:
        raise ShardLayoutError(
            f"--assert-multi-device: realized layout computes on {compute_devices} — a "
            "single-device layout certifies nothing about sharding.")

    return {
        "mode": "sharded",
        "spec": spec.spec_echo,
        "device_map_requested": (resolved_map if isinstance(resolved_map, str)
                                 else {k: str(v) for k, v in resolved_map.items()}),
        "max_memory_requested": ({str(k): str(v) for k, v in spec.max_memory.items()}
                                 if spec.max_memory else None),
        "hf_device_map": ({k: str(v) for k, v in hf_map.items()}
                          if isinstance(hf_map, dict) else None),
        "accelerate_dispatched": isinstance(hf_map, dict),
        "decoder_layer_devices": layer_devices,
        "site_layer_devices": {f"L{s}": layer_devices[str(s)] for s in sites},
        "input_embedding_device": input_device_of(model),
        "compute_devices": compute_devices,
        "storage_spilled_to": spilled,
        "n_compute_devices": len(compute_devices),
        "offload_allowed": spec.allow_offload,
        "offload_folder": spec.offload_folder,
        "accelerate": accelerate_version,
        "cuda_device_names": {str(i): torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())},
        "parallelism": "pipeline (device_map; each decoder layer whole on one device)",
    }


def load_model_and_tok_sharded(model_path: str, spec: ShardSpec, sites: tuple[int, ...],
                               dequantize_fp8: bool = False
                               ) -> tuple[Any, Any, str, dict]:
    """The opt-in twin of load_model_and_tok: same dtype/attn/eval, device_map placement.

    Returns (model, tokenizer, input_device, sharding_record). Everything downstream —
    SiteCapture, compute_means, save_state_bank — is the unchanged single-card machinery.

    `dequantize_fp8` is the row-21 dtype regime (ADDENDUM 2026-07-26-A) and is
    orthogonal to placement: it changes what the weights ARE (bf16 reconstructed from
    the block-wise FP8 checkpoint), never where they go. The `fp8_dequantize` witness
    is folded into the sharding record so one stamp carries both facts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerate_version = _require_accelerate()
    resolved_map: Union[str, dict[str, DeviceKey]]
    if spec.shard_across is not None:
        resolved_map = build_even_layer_device_map(model_path, spec.shard_across)
    else:
        assert spec.device_map is not None       # guarded in ShardSpec.__post_init__
        resolved_map = spec.device_map

    needs_offload_dir = (not isinstance(resolved_map, str)
                         and any(str(v) == "disk" for v in resolved_map.values()))
    if needs_offload_dir and not spec.offload_folder:
        raise DeviceMapSpecError(
            "the requested device map offloads to 'disk' but --offload-folder is unset")

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16,
                              "attn_implementation": "eager",
                              "device_map": resolved_map,
                              **_fp8_kwargs(dequantize_fp8)}
    if spec.max_memory is not None:
        kwargs["max_memory"] = spec.max_memory
    if spec.offload_folder is not None:
        kwargs["offload_folder"] = spec.offload_folder
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ValueError, RuntimeError, OSError) as exc:
        raise ShardedLoadError(
            f"sharded load of {model_path!r} failed ({type(exc).__name__}: {exc}); "
            "check --device-map / --max-memory against the visible devices") from exc
    model.eval()                                   # NOT .to(device): the model is dispatched
    if dequantize_fp8:                             # fail at LOAD, not after a 26-min pass
        assert_dequantized(model, model_path)

    sharding = describe_shard_layout(model, sites, spec, accelerate_version, resolved_map)
    device = sharding["input_embedding_device"]
    logger.info("sharded load: %d compute device(s) %s; inputs on %s; layer map %s",
                sharding["n_compute_devices"], sharding["compute_devices"], device,
                json.dumps(sharding["site_layer_devices"]))
    return model, AutoTokenizer.from_pretrained(model_path), device, sharding


def collecting_hostname() -> str:
    """The node this process is running on — `HOSTNAME_UNRESOLVED` if it cannot say.

    RAKE M41: `trunk_stamp` recorded no hostname, so NO banked artifact identified
    its node of origin and every "where does artifact X live" question had to be
    settled by filesystem probe. The 405B readout block was exactly that: the scan
    fits were expected on one node's arm and lived on the other, the stamps could
    confirm the scan lane's 8-card provenance but NOT the node, and a plausible
    "pull-then-clean" inference had to be retracted after a cross-node probe found
    the tree.

    Rake M19: instrumentation that only DESCRIBES a run must never be able to FAIL
    it. A hostname is pure description — it cannot change one banked number — so
    every way of failing to get one degrades to a NAMED sentinel rather than
    raising. `HOSTNAME_UNRESOLVED` is deliberately not an empty string and not
    `None`: a reader must be able to tell "this collector could not resolve a
    hostname" from "this stamp predates the field" (rake M41's backward
    compatibility, from the other side).

    `socket.gethostname()` is the node's own name as the kernel knows it, which is
    what the scheduler's job records and the /net paths agree with. FQDN
    resolution is deliberately NOT attempted: it can block on DNS, and a stamp
    writer that can hang is worse than one that is terse.
    """
    import socket
    try:
        name = socket.gethostname()
    except OSError:                                # pragma: no cover — degraded
        return HOSTNAME_UNRESOLVED
    return name.strip() or HOSTNAME_UNRESOLVED


def trunk_stamp(model_path: str, tok, device: str,
                sharding: Optional[dict] = None,
                fp8: Optional[dict] = None) -> dict:
    import torch
    import transformers
    cfg = Path(model_path) / "config.json"
    tpl = tok.chat_template or ""
    dev_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    stamp = {"model_path": str(model_path),
             "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()
             if cfg.exists() else None,
             "chat_template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
             "transformers": transformers.__version__,
             "torch": torch.__version__,
             "dtype_forward": "bfloat16", "dtype_banked": "float32",
             "attn_implementation": "eager", "device": device,
             "cuda_device_name": dev_name,
             "cuda_visible_devices":
                 os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
             # RAKE M41: the node of origin. APPENDED at the end of the mapping so
             # a stamp read as an ordered document still opens with the historical
             # keys in their historical order; readers key by NAME, and a reader
             # that must tolerate its absence does so with
             # `stamp["trunk"].get("hostname")` — see `stamp_hostname`.
             HOSTNAME_STAMP_KEY: collecting_hostname()}
    # Single-card stamps keep EXACTLY the historical key set (the banked smalls compare
    # against them) EXCEPT for the M41 hostname, which is unconditional by design:
    # the whole point is that EVERY artifact can be attributed to its node, so a
    # conditional hostname would leave precisely the gap M41 was filed for. The
    # sharding record is appended only on the opt-in sharded path, and the
    # dtype-regime witness only on the opt-in --dequantize-fp8 path. Same
    # discipline as `truncation` in the collection stamp: an absent key means the
    # deviation was not taken, so historical stamps stay comparable key-for-key.
    if sharding is not None:
        stamp["sharding"] = sharding
    if fp8 is not None:
        stamp[DEQUANTIZED_STAMP_KEY] = fp8
    return stamp


def stamp_hostname(stamp: dict) -> Optional[str]:
    """The node a stamp was written on, or None for a PRE-M41 stamp.

    THE BACKWARD-COMPATIBLE READER, and the only one any consumer should use.
    Every artifact banked before 2026-07-29 has no hostname at all, and the three
    states a reader must keep apart are:

        None                    the stamp PREDATES the field — the node is
                                unknown and must be established by probing BOTH
                                nodes' stores (rake M41(a)), never inferred
        HOSTNAME_UNRESOLVED     the collector RAN but could not name its host —
                                degraded instrumentation, not a missing field
        "<name>"                the node of origin, as the kernel knew it

    Accepts either a trunk stamp or a whole collection stamp (which nests the
    trunk under `trunk`), because both are handed around and a reader that only
    understood one shape would silently return None for the other — which is the
    "unknown node" answer, i.e. the wrong one for the commonest input.
    """
    if not isinstance(stamp, dict):
        return None
    trunk = stamp.get("trunk")
    if isinstance(trunk, dict) and HOSTNAME_STAMP_KEY in trunk:
        value = trunk.get(HOSTNAME_STAMP_KEY)
    else:
        value = stamp.get(HOSTNAME_STAMP_KEY)
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------- collect mode
def collect(arm_root: Path, model: str, model_path: str, arms: list[str],
            device: str, sites_override: tuple[int, ...] | None = None,
            shard: Optional[ShardSpec] = None,
            max_length: int | None = None,
            dequantize_fp8: bool = False) -> None:
    entries, manifest_sha = load_corpus(arm_root)
    #  THE BASIS NAMES ITS OWN STRATA, and it does so BEFORE the weights load: a
    #  manifest whose entries cannot be read for a vocabulary is a staging defect,
    #  and finding that out after a 26-minute collection pass is how the canary
    #  of 2026-08-03 burned a bank.
    strata = derive_strata(entries)
    counts = stratum_counts(entries)
    logger.info("basis: %d strata %s (manifest %s…), counts %s",
                len(strata), list(strata), manifest_sha[:12], counts)
    sites = sites_override or sites_for(model)
    states_dir = arm_root / "states"
    logs_dir = arm_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if shard is None:                              # the unchanged single-card path
        model_obj, tok = load_model_and_tok(model_path, device, dequantize_fp8)
        sharding = None
    else:                                          # opt-in: device_map across the node
        model_obj, tok, device, sharding = load_model_and_tok_sharded(
            model_path, shard, sites, dequantize_fp8)
    fp8 = assert_dequantized(model_obj, model_path) if dequantize_fp8 else None
    trunk = trunk_stamp(model_path, tok, device, sharding, fp8)
    logger.info("trunk: %s", json.dumps(trunk)[:200])

    for arm in arms:
        t0 = time.time()
        means, tok_norms, seq_lens, truncated = compute_means(
            model_obj, tok, entries, sites, arm, VMB_CANONICAL_DATE, device,
            max_length=max_length)
        text_ids = [e["text_id"] for e in entries]
        median_norms = {s: float(np.median(np.linalg.norm(means[s], axis=1)))
                        for s in sites}
        bank = StateBank(model=model, arm=arm, text_ids=text_ids,
                         states=means, median_norms=median_norms)
        npz_path, norms_path = save_state_bank(states_dir, bank)
        stamp = {
            "arm_dir": "A8_conjugation", "leg": arm_root.name, "builder": "collect_mean_states.py",
            "prereg_tag": "prereg-arm8-v1", "model": model, "template_arm": arm,
            "sites": list(sites), "n_texts": len(entries),
            "counts_per_stratum": counts,
            "corpus_manifest_sha256": manifest_sha,
            "date_string": VMB_CANONICAL_DATE,
            "median_mean_state_norms": {f"L{s}": v for s, v in median_norms.items()},
            "median_token_resid_norms": {
                f"L{s}": float(np.median(tok_norms[s])) for s in sites},
            "seq_len": {"min": int(min(seq_lens)), "max": int(max(seq_lens)),
                        "mean": round(float(np.mean(seq_lens)), 1)},
            "wall_seconds": round(time.time() - t0, 1),
            "trunk": trunk,
            "capture_convention": "forward_pre_hook on decoder_layers[L] "
                                  "(residual ENTERING layer L), completion "
                                  "positions >= P, fp32 mean; batch-1, no cache",
            # ── THE STRATUM BASIS (desk ruling 2026-08-03) ────────────────────
            # APPENDED, so every historical key keeps its historical position in
            # a stamp read as an ordered document (the M41 hostname discipline).
            # UNCONDITIONAL, for the same reason the hostname is: the vocabulary
            # a bank was collected under is part of its IDENTITY, not a
            # deviation, and `counts_per_stratum` above is unreadable without it
            # on any basis whose names a reader does not already know.
            "strata_basis": {
                "rule": "corpus_manifest.json per-entry `stratum`, "
                        "first-occurrence order (deterministic under the "
                        "manifest sha recorded above)",
                "strata": list(strata),
                "n_strata": len(strata),
                "counts": counts,
            },
        }
        # ADDENDUM 2026-07-27-B: the deviation is recorded in EVERY stamp of a
        # truncated node. Absent on every other node, so historical stamps keep
        # exactly their key set (same discipline as the `sharding` record).
        if max_length is not None:
            stamp["truncation"] = f"first-{max_length}"
            stamp["truncation_prereg"] = "ADDENDUM 2026-07-27-B"
            stamp["n_truncated"] = len(truncated)
            stamp["truncated_text_ids"] = truncated
        stamp_path = states_dir / f"collection_stamp_{model}_{arm}.json"
        with open(stamp_path, "w") as f:
            json.dump(stamp, f, indent=1)
        logger.info("banked %s (%s) -> %s + %s + %s", model, arm, npz_path.name,
                    norms_path.name, stamp_path.name)


# ---------------------------------------------------------------- spot replay (T5)
def pick_spot_ids(entries: list[dict], k: int,
                  strata: Optional[Sequence[str]] = None) -> list[str]:
    """Deterministic stratified pick: k texts spread across strata by id-hash order.

    THE CP-1 GATE'S SELECTION RULE, unchanged in every respect except where the
    stratum vocabulary comes from. `strata` defaults to `derive_strata(entries)`
    — the pinned manifest's own names, in first-occurrence order — so a basis
    with one stratum, four, or a vocabulary nobody has seen works by
    construction, and no manifest can make this function raise `KeyError`.

    BYTE-IDENTICAL ON A v2.1 CORPUS, by construction rather than by luck: the
    round-robin below advances `i` on every turn but pops only from a NON-EMPTY
    stratum, so a stratum absent from the manifest contributes nothing and costs
    nothing. Cycling the legacy ("S1","S2","S3","S5") tuple over a manifest that
    holds S1/S2/S3 therefore emits exactly what cycling the derived ("S1","S2",
    "S3") tuple emits — the same ids, in the same order, for every k. Selftest 8
    asserts that against selections RECORDED FROM THE PRE-CHANGE CODE.
    """
    order = tuple(strata) if strata is not None else derive_strata(entries)
    if not order:
        raise StrataDerivationError(
            "pick_spot_ids: no strata to spread the CP-1 gate's picks across")
    if k < 1:
        raise ValueError(f"pick_spot_ids: k={k} — the gate picks at least one text")
    if k > len(entries):
        #  Historically this looped FOREVER (the while-loop has no other exit),
        #  which on a node is a job that burns its slot and reports nothing.
        raise ValueError(
            f"pick_spot_ids: k={k} exceeds the {len(entries)} texts the corpus "
            f"holds — the gate cannot replay more texts than were banked")
    by_stratum: dict[str, list[str]] = {s: [] for s in order}
    unknown: dict[str, int] = {}
    for e in entries:
        bucket = by_stratum.get(str(e["stratum"]))
        if bucket is None:                      # only reachable via an explicit `strata`
            unknown[str(e["stratum"])] = unknown.get(str(e["stratum"]), 0) + 1
            continue
        bucket.append(e["text_id"])
    if unknown:
        raise StrataDerivationError(
            f"pick_spot_ids: {sum(unknown.values())} manifest entries carry "
            f"stratum(s) {sorted(unknown)} that the requested vocabulary "
            f"{list(order)} does not name. Those texts could never be picked, so "
            f"the gate would silently under-cover the corpus")
    for s in order:
        by_stratum[s].sort(key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    picked, i = [], 0
    while len(picked) < k:
        s = order[i % len(order)]
        if by_stratum[s]:
            picked.append(by_stratum[s].pop(0))
        i += 1
    return picked


def spot_replay(arm_root: Path, model: str, model_path: str, arms: list[str],
                device: str, k: int, sites_override: tuple[int, ...] | None = None,
                shard: Optional[ShardSpec] = None,
                max_length: int | None = None,
                dequantize_fp8: bool = False) -> int:
    entries, manifest_sha = load_corpus(arm_root)
    strata = derive_strata(entries)
    counts = stratum_counts(entries)
    logger.info("basis: %d strata %s (manifest %s…), counts %s",
                len(strata), list(strata), manifest_sha[:12], counts)
    by_id = {e["text_id"]: e for e in entries}
    sites = sites_override or sites_for(model)
    states_dir = arm_root / "states"
    if shard is None:                              # the unchanged single-card path
        model_obj, tok = load_model_and_tok(model_path, device, dequantize_fp8)
        sharding = None
    else:                                          # opt-in: device_map across the node
        model_obj, tok, device, sharding = load_model_and_tok_sharded(
            model_path, shard, sites, dequantize_fp8)
    fp8 = assert_dequantized(model_obj, model_path) if dequantize_fp8 else None
    spot_ids = pick_spot_ids(entries, k, strata)
    ok_all = True
    report: dict = {"model": model, "k": k, "spot_ids": spot_ids,
                    # The vocabulary the picks were spread across, recorded WITH
                    # the picks: which texts this gate replayed is the gate's
                    # identity, and that identity is a function of the basis.
                    "strata_basis": {
                        "rule": "corpus_manifest.json per-entry `stratum`, "
                                "first-occurrence order",
                        "corpus_manifest_sha256": manifest_sha,
                        "strata": list(strata),
                        "n_strata": len(strata),
                        "counts": counts,
                        "picked_per_stratum": {
                            s: sum(1 for t in spot_ids
                                   if by_id[t]["stratum"] == s) for s in strata},
                    },
                    "trunk": trunk_stamp(model_path, tok, device, sharding, fp8),
                    "results": {}}
    if max_length is not None:                 # ADDENDUM 2026-07-27-B, per-use record
        report["truncation"] = f"first-{max_length}"
        report["truncation_prereg"] = "ADDENDUM 2026-07-27-B"
    for arm in arms:
        bank = load_state_bank(states_dir, model, arm)
        idx = {t: i for i, t in enumerate(bank.text_ids)}
        sub = [by_id[t] for t in spot_ids]
        means, _, _, _ = compute_means(model_obj, tok, sub, sites, arm,
                                       VMB_CANONICAL_DATE, device, log_every=1000,
                                       max_length=max_length)
        res = {}
        for j, t in enumerate(spot_ids):
            per_site = {}
            for s in sites:
                banked = bank.states[s][idx[t]]
                fresh = means[s][j]
                exact = bool(np.array_equal(banked, fresh))
                per_site[f"L{s}"] = {
                    "bitwise_exact": exact,
                    "max_abs_diff": float(np.max(np.abs(
                        banked.astype(np.float64) - fresh.astype(np.float64))))}
                ok_all &= exact
            res[t] = per_site
        report["results"][arm] = res
        n_exact = sum(d["bitwise_exact"] for r in res.values() for d in r.values())
        n_tot = sum(len(r) for r in res.values())
        logger.info("spot-replay %s/%s: %d/%d site-cells bitwise exact",
                    model, arm, n_exact, n_tot)
    out = states_dir / f"spot_replay_{model}.json"
    with open(out, "w") as f:
        json.dump({**report, "all_bitwise_exact": ok_all}, f, indent=1)
    logger.info("spot report: %s  ALL_EXACT=%s", out, ok_all)
    return 0 if ok_all else 1


# ---------------------------------------------------------------- CP-1 summary
def cp1_summary(arm_root: Path, models: list[str]) -> int:
    states_dir = arm_root / "states"
    lines = ["CP-1 COLLECTION QC (UNSTAMPED, C§8)"]
    norms: dict[tuple[str, str], dict] = {}
    for model in models:
        for arm in ARMS:
            sp = states_dir / f"collection_stamp_{model}_{arm}.json"
            if not sp.exists():
                lines.append(f"  {model}/{arm}: MISSING")
                continue
            st = json.loads(sp.read_text())
            norms[(model, arm)] = st["median_mean_state_norms"]
            #  RAKE M41: the node of origin, read through the tolerant reader —
            #  every bank predating 2026-07-29 has no hostname, and a QC table
            #  that crashed on one would be unusable on the whole existing tree.
            host = stamp_hostname(st)
            lines.append(
                f"  {model}/{arm}: n={st['n_texts']} strata={st['counts_per_stratum']}"
                f" norms={ {k: round(v, 2) for k, v in norms[(model, arm)].items()} }"
                f" seq_len={st['seq_len']['mean']} wall={st['wall_seconds']}s"
                f" host={host or 'PRE-M41 (unstamped node; probe BOTH stores)'}")
        spr = states_dir / f"spot_replay_{model}.json"
        if spr.exists():
            r = json.loads(spr.read_text())
            lines.append(f"  {model} spot-replay k={r['k']}: "
                         f"ALL_BITWISE_EXACT={r['all_bitwise_exact']}")
        else:
            lines.append(f"  {model} spot-replay: NOT RUN")
    if len(models) == 2 and all((m, a) in norms for m in models for a in ARMS):
        m0, m1 = models
        for arm in ARMS:
            r = {f"{k0}->{k1}": round(norms[(m1, arm)][k1] / norms[(m0, arm)][k0], 3)
                 for k0 in norms[(m0, arm)] for k1 in norms[(m1, arm)]}
            lines.append(f"  norm ratio {m1}/{m0} ({arm}): {r}")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------- selftest (CPU)
#  THE GATE-IDENTITY FIXTURES. Shapes, not data: `pick_spot_ids` reads exactly
#  two fields, so these carry exactly two fields. They are built the same way the
#  real manifests are — one stratum block after another, in the order the corpus
#  builders emit them (`build_uncapped_corpus` appends S5 to a byte-identical
#  v2.1 body) — because FIRST-OCCURRENCE ORDER is what the derivation reads.
def _v21_fixture_entries() -> list[dict]:
    """A v2.1-shaped manifest: S1 (model-authored) · S2 (carrier) · S3 (scaffolded)."""
    e: list[dict] = []
    for voice in ("3b", "8b"):
        for t in range(20):
            for r in (0, 1):
                e.append({"text_id": f"S1-{voice}-t{t:02d}-s0-r{r}", "stratum": "S1"})
    for r in range(40):
        e.append({"text_id": f"S2-wikitext-{r:03d}", "stratum": "S2"})
    for voice in ("3b", "8b"):
        for mode in ("linear", "socratic", "contrastive"):
            for t in range(5):
                e.append({"text_id": f"S3-{voice}-{mode}-t{t:02d}", "stratum": "S3"})
    return e


def _v21_s5_fixture_entries() -> list[dict]:
    """The Leg-4 augmentation: the v2.1 rows byte-identical, S5 appended."""
    e = _v21_fixture_entries()
    for voice in ("3b", "8b"):
        for g in range(20):
            e.append({"text_id": f"S5-{voice}-g{g:03d}", "stratum": "S5"})
    return e


def _webtext_v3_fixture_entries() -> list[dict]:
    """A webtext-v3-shaped manifest: the four strata frozen prereg §3.4 names."""
    return [{"text_id": f"{s}-{i:04d}", "stratum": s}
            for s in ("wikitext", "c4", "pg19", "stackexchange")
            for i in range(30)]


#  RECORDED FROM THE PRE-CHANGE CODE (2026-08-03, before a line of this file
#  moved), by running the then-current `pick_spot_ids` — module constant
#  ("S1","S2","S3","S5") — on the fixtures above. Raw log:
#  /tmp/claude-output/strata-fix-BEFORE-pick-spot-ids.log.
#
#  These are the CP-1 gate's IDENTITY on a v2.1-shaped corpus: which texts it
#  replays. The gate is a certified instrument, so that identity must provably
#  not move. `pick_spot_ids` is incremental (each turn appends one id and never
#  reorders), so pick(k) is a PREFIX of pick(k') for k < k' — one recorded
#  20-long selection therefore pins every K the gate is ever run at.
LEGACY_PICK_V21_K20: tuple[str, ...] = (
    "S1-3b-t19-s0-r0", "S2-wikitext-023", "S3-8b-contrastive-t00",
    "S1-8b-t16-s0-r1", "S2-wikitext-001", "S3-3b-socratic-t00",
    "S1-3b-t18-s0-r1", "S2-wikitext-027", "S3-8b-linear-t02",
    "S1-3b-t12-s0-r0", "S2-wikitext-036", "S3-8b-socratic-t02",
    "S1-3b-t12-s0-r1", "S2-wikitext-014", "S3-3b-contrastive-t03",
    "S1-3b-t14-s0-r1", "S2-wikitext-004", "S3-3b-contrastive-t00",
    "S1-8b-t18-s0-r1", "S2-wikitext-013")
LEGACY_PICK_V21_S5_K20: tuple[str, ...] = (
    "S1-3b-t19-s0-r0", "S2-wikitext-023", "S3-8b-contrastive-t00", "S5-8b-g018",
    "S1-8b-t16-s0-r1", "S2-wikitext-001", "S3-3b-socratic-t00", "S5-3b-g009",
    "S1-3b-t18-s0-r1", "S2-wikitext-027", "S3-8b-linear-t02", "S5-8b-g001",
    "S1-3b-t12-s0-r0", "S2-wikitext-036", "S3-8b-socratic-t02", "S5-3b-g017",
    "S1-3b-t12-s0-r1", "S2-wikitext-014", "S3-3b-contrastive-t03", "S5-3b-g018")
#: The same record against the REAL v2.1 fitting manifest of record (775 texts) —
#: the corpus every certified v2.1 bank was collected on, not a fixture. Guarded
#: by a NAMED SKIP (rake M44) because it is a repo data artifact and a deployed
#: code tree has none.
LEGACY_PICK_REAL_V21_K20: tuple[str, ...] = (
    "S1-dsv2-lite-t02-s3-r1", "S2-wt-094", "S3-dsv2-lite-analogical-t03-r1",
    "S1-dsv2-lite-t05-s1-r1", "S2-wt-093", "S3-8b-dialectical-t08-r1",
    "S1-dsv2-lite-t06-s0-r1", "S2-wt-101", "S3-dsv2-lite-analogical-t05-r1",
    "S1-8b-t16-s0-r1", "S2-wt-055", "S3-8b-socratic-t05-r0",
    "S1-dsv2-lite-t03-s0-r0", "S2-wt-097", "S3-dsv2-lite-socratic-t17-r0",
    "S1-dsv2-lite-t16-s0-r1", "S2-wt-089", "S3-8b-dialectical-t17-r0",
    "S1-dsv2-lite-t06-s1-r0", "S2-wt-005")
REAL_V21_MANIFEST = Path("corpus/fitting-v21/corpus_manifest.meta.json")


def selftest() -> int:
    """CPU-only gates on the OPT-IN dtype regime and its wiring.

    Everything here is about one property: with `--dequantize-fp8` unset the loader
    call is byte-identical to the one that produced every banked node, and with it set
    the bf16 regime is either established or the run DIES. No GPU, no weights.

    RAKE M44: the two environment axes (deep-learning stack, working fp8 dequantize)
    are branched on the availability probes below and degrade to NAMED skips. A skip
    is a THIRD state, distinct from pass and fail, and never a non-zero exit code —
    nothing was asserted and found wanting. See the module docstring for the matrix.
    """
    fails: list[str] = []
    checks: list[str] = []
    skips: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(name)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    def skip(name: str, why: str) -> None:
        """A NAMED skip (rake M44): a block that cannot run in THIS configuration.

        Recorded as run-and-absent rather than crashed or failed, counted in the tail
        so a sweep can report coverage per configuration, and NEVER a non-zero exit
        code. The reason names the TRIGGERING CONDITION: a skip that does not say why
        is the defect coming back in a quieter form.
        """
        skips.append(name)
        checks.append(f"SKIPPED: {name}")
        print(f"  [SKIP] {name} — {why}")

    # ---- the availability probes (rake M44) ----------------------------------
    #  AXIS 1 — the deep-learning stack. `trunk_stamp` reads torch.__version__ and
    #  transformers.__version__, and the post-load witness builds real nn.Modules, so
    #  blocks 3, 3b(stamp) and 4-6 need it. Blocks 1, 2, 3b(reader/M19) and 7 do NOT,
    #  and must stay provable in a configuration with no deep-learning stack at all —
    #  which the desk's own repo .venv (numpy/scipy/pydantic only) is.
    torch: Any = None
    nn: Any = None
    try:
        import torch as _torch_mod
        import torch.nn as _nn_mod
        import transformers as _transformers_mod
        torch, nn = _torch_mod, _nn_mod
        stack_ok, stack_why = True, ""
        print(f"[config] deep-learning stack: torch {_torch_mod.__version__} + "
              f"transformers {_transformers_mod.__version__}")
    except ImportError as exc:
        stack_ok = False
        stack_why = (f"this interpreter has no deep-learning stack "
                     f"({type(exc).__name__}: {exc})")
        print(f"[config] deep-learning stack: ABSENT ({type(exc).__name__}: {exc})")

    def _fp8_environment() -> str:
        """Name the transformers build the fp8 verdict is ABOUT, and how it falls short.

        NEVER raises: a probe that can itself raise is the same defect one level up.
        """
        try:
            import transformers
            return (f"transformers {transformers.__version__} lacks a working fp8 "
                    f"dequantize")
        except Exception as exc:                    # noqa: BLE001 — a probe never raises
            return f"transformers is not importable here ({type(exc).__name__}: {exc})"

    #  AXIS 2 — the fp8 dequantize regime. THE DEFECT THIS BLOCK CLOSES: transformers
    #  4.51.3's FineGrainedFP8Config accepts `dequantize=True` and does not set it, so
    #  `fp8_dequantize_config()` refuses (correctly — the difference is a native-FP8
    #  forward vs a bf16 one). A build that cannot offer the regime is a legitimate
    #  environment, not a test failure, and every non-FP8 node is one.
    try:
        fp8_dequantize_config()
        fp8_ok, fp8_why = True, ""
        print("[config] fp8 dequantize: available")
    except Fp8RegimeError as exc:
        fp8_ok = False
        fp8_why = f"{_fp8_environment()} (Fp8RegimeError: {exc})"
        print(f"[config] fp8 dequantize: ABSENT — {fp8_why}")
    except Exception as exc:                        # noqa: BLE001 — a probe never raises
        fp8_ok = False
        fp8_why = (f"{_fp8_environment()}; FineGrainedFP8Config raised "
                   f"{type(exc).__name__}: {exc}")
        print(f"[config] fp8 dequantize: ABSENT — {fp8_why}")

    #  AXIS 3 — the DATA TREE (rake M44(a): the count names its configuration,
    #  and M44(c): a fixture that resolves a real artifact relative to cwd is
    #  itself a defect). The gate-identity blocks below are proved on SYNTHETIC
    #  fixtures that need no tree at all; the REAL v2.1 fitting manifest is an
    #  additional, cwd-dependent pass that degrades to a named skip.
    print(f"[config] cwd={Path.cwd()}  v2.1 fitting manifest "
          f"{'PRESENT' if REAL_V21_MANIFEST.exists() else 'ABSENT'} at "
          f"{REAL_V21_MANIFEST}")

    print("== selftest 1: the historical loader call is untouched when opted out ==")
    check("_fp8_kwargs(False) contributes NO kwargs", _fp8_kwargs(False) == {},
          repr(_fp8_kwargs(False)))

    print("== selftest 2: opting in produces FineGrainedFP8Config(dequantize=True) ==")
    if fp8_ok:
        kw = _fp8_kwargs(True)
        check("exactly one kwarg, named quantization_config", list(kw) == ["quantization_config"],
              str(list(kw)))
        check("dequantize is True",
              getattr(kw.get("quantization_config"), "dequantize", None) is True)
    else:
        skip("fp8-dequantize: --dequantize-fp8 adds exactly one kwarg, "
             "FineGrainedFP8Config(dequantize=True)", fp8_why)

    print("== selftest 3: stamp key sets — absent deviation means absent key ==")

    class _Tok:
        chat_template = "{{ 'x' }}"

    base: dict = {}
    with_fp8: dict = {}
    if stack_ok:
        base = trunk_stamp(".", _Tok(), "cpu")
        with_fp8 = trunk_stamp(".", _Tok(), "cpu", None, {"requested": "probe"})
        check("plain stamp carries no fp8 key", DEQUANTIZED_STAMP_KEY not in base)
        check("fp8 stamp adds exactly that one key",
              set(with_fp8) - set(base) == {DEQUANTIZED_STAMP_KEY},
              str(sorted(set(with_fp8) - set(base))))
    else:
        skip("trunk-stamp key sets: an absent deviation means an absent key",
             stack_why + " — trunk_stamp reads torch.__version__ and "
                         "transformers.__version__")

    print("== selftest 3b: RAKE M41 — the trunk stamp names its NODE ==")
    import socket as _socket

    if stack_ok:
        check("every trunk stamp carries a hostname, UNCONDITIONALLY",
              HOSTNAME_STAMP_KEY in base and HOSTNAME_STAMP_KEY in with_fp8
              and HOSTNAME_STAMP_KEY in trunk_stamp(
                  ".", _Tok(), "cpu", {"n_compute_devices": 2}),
              f"{HOSTNAME_STAMP_KEY}={base[HOSTNAME_STAMP_KEY]!r} on the plain, "
              f"sharded and fp8 paths alike — an identity, not a deviation, so it is "
              f"never conditional")
        check("the hostname is this machine's own, as the kernel knows it",
              base[HOSTNAME_STAMP_KEY] == _socket.gethostname()
              and base[HOSTNAME_STAMP_KEY] != HOSTNAME_UNRESOLVED,
              "socket.gethostname() — no FQDN resolution, which could block on DNS; "
              "a stamp writer that can hang is worse than one that is terse")
        check("the hostname is APPENDED, so the historical keys keep their order",
              list(base)[:-1] == [k for k in base if k != HOSTNAME_STAMP_KEY]
              and list(base)[-1] == HOSTNAME_STAMP_KEY,
              f"last key is {list(base)[-1]!r}")
        check("and the historical key SET is otherwise untouched",
              set(base) - {HOSTNAME_STAMP_KEY} == {
                  "model_path", "config_sha256", "chat_template_sha256",
                  "transformers", "torch", "dtype_forward", "dtype_banked",
                  "attn_implementation", "device", "cuda_device_name",
                  "cuda_visible_devices"},
              str(sorted(set(base) - {HOSTNAME_STAMP_KEY})))
    else:
        skip("the WRITER's half of M41: a real trunk stamp names its node "
             "unconditionally, appends the key last, and leaves the historical key "
             "set otherwise untouched", stack_why)

    #  RAKE M19: instrumentation that only DESCRIBES a run must never FAIL it.
    #  The failing branch is EXERCISED here — an unexercised degradation path is
    #  not known to work (M19(c) in miniature).
    real_gethostname = _socket.gethostname
    try:
        _socket.gethostname = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            OSError("selftest: name resolution unavailable"))
        degraded = collecting_hostname()
        check("a hostname lookup that RAISES degrades to the named sentinel",
              degraded == HOSTNAME_UNRESOLVED,
              f"{degraded!r} — the stamp still writes, because a description can "
              f"never fail a collection (rake M19)")
        _socket.gethostname = lambda: "   "     # type: ignore[assignment]
        blank = collecting_hostname()
        check("and a BLANK hostname degrades the same way, never to ''",
              blank == HOSTNAME_UNRESOLVED,
              f"{blank!r} — an empty string would read as a missing field")
    finally:
        _socket.gethostname = real_gethostname  # type: ignore[assignment]

    #  THE READER'S CONTRACT: three states, kept apart. Every artifact banked
    #  before 2026-07-29 has no hostname at all, and "unknown node" demands the
    #  M41(a) response (probe BOTH stores) while "unresolved" does not.
    if stack_ok:
        check("stamp_hostname reads a trunk stamp directly",
              stamp_hostname(base) == _socket.gethostname())
        check("and a whole COLLECTION stamp, which nests the trunk",
              stamp_hostname({"model": "3b", "trunk": base})
              == _socket.gethostname(),
              "both shapes are handed around; a reader that understood only one "
              "would silently answer 'unknown node' for the commonest input")
        pre_m41 = {k: v for k, v in base.items() if k != HOSTNAME_STAMP_KEY}
        check("a PRE-M41 stamp reads as None — backward compatible, never a raise",
              stamp_hostname(pre_m41) is None
              and stamp_hostname({"model": "3b", "trunk": pre_m41}) is None,
              "the whole existing bank tree has no hostname; a reader that crashed "
              "on one would be unusable on every artifact the campaign holds")
    else:
        skip("stamp_hostname against a REAL trunk stamp: the direct shape, the "
             "nested collection shape, and the pre-M41 shape", stack_why)
    check("an UNRESOLVED hostname is distinguishable from an absent one",
          stamp_hostname({HOSTNAME_STAMP_KEY: HOSTNAME_UNRESOLVED})
          == HOSTNAME_UNRESOLVED,
          "degraded instrumentation and a missing field demand different "
          "responses, so the sentinel is a value and not None")
    check("and a malformed stamp reads as None rather than raising",
          stamp_hostname({}) is None and stamp_hostname(None) is None  # type: ignore[arg-type]
          and stamp_hostname({"trunk": "not a dict"}) is None,
          "a stamp reader is instrumentation too (rake M19)")

    print("== selftest 4: the post-load witness passes on a clean bf16 model ==")
    if stack_ok:
        class _Clean(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4).to(torch.bfloat16)

        rec = assert_dequantized(_Clean(), "<stub>")
        check("witness records zero surviving FP8 modules", rec["fp8_modules_remaining"] == 0)
        check("dtype census is bf16-only",
              set(rec["param_dtype_census"]) == {"torch.bfloat16"},
              str(rec["param_dtype_census"]))
    else:
        skip("the post-load witness PASSES on a clean bf16 model (zero surviving FP8 "
             "modules, a bf16-only dtype census)",
             stack_why + " — the witness walks real nn.Module parameters")

    print("== selftest 5: a surviving FP8 module is FATAL, not a warning ==")
    if stack_ok:
        class FP8Linear(nn.Module):        # name is the witness, per FP8_MODULE_TYPE_NAMES
            pass

        class _Native(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q = FP8Linear()

        try:
            assert_dequantized(_Native(), "<stub>")
            check("raises Fp8RegimeError on a surviving FP8Linear", False, "no raise")
        except Fp8RegimeError as exc:
            check("raises Fp8RegimeError on a surviving FP8Linear", True, str(exc)[:60])
    else:
        skip("a surviving FP8 module is FATAL, not a warning", stack_why)

    print("== selftest 6: a non-bf16 float parameter is FATAL ==")
    if stack_ok:
        class _Mixed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(4, 4).to(torch.bfloat16)
                self.b = nn.Linear(4, 4).to(torch.float16)

        try:
            assert_dequantized(_Mixed(), "<stub>")
            check("raises Fp8RegimeError on an fp16 parameter", False, "no raise")
        except Fp8RegimeError as exc:
            check("raises Fp8RegimeError on an fp16 parameter", True, str(exc)[:60])
    else:
        skip("a non-bf16 float parameter is FATAL", stack_why)

    print("== selftest 7: CLI wiring — the flag reaches BOTH modes, and only if passed ==")
    real_collect, real_spot, real_argv = collect, spot_replay, sys.argv
    seen: dict = {}
    try:
        globals()["collect"] = lambda *a, **kw: seen.update(collect_fp8=kw.get("dequantize_fp8"))
        globals()["spot_replay"] = lambda *a, **kw: (
            seen.update(spot_fp8=kw.get("dequantize_fp8"), spot_shard=kw.get("shard")), 0)[1]
        for mode in (["--collect"], ["--spot-replay", "3"]):
            sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                        "--arm-root", "/nonexistent-selftest", "--sites", "1",
                        *mode, "--dequantize-fp8", "--shard-across", "2"]
            main()
            sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                        "--arm-root", "/nonexistent-selftest", "--sites", "1", *mode]
            main()
            key = "collect_fp8" if mode[0] == "--collect" else "spot_fp8"
            check(f"{mode[0]}: default is False", seen.get(key) is False, repr(seen.get(key)))
        sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                    "--arm-root", "/nonexistent-selftest", "--sites", "1",
                    "--collect", "--dequantize-fp8"]
        main()
        check("--collect: flag forwarded", seen.get("collect_fp8") is True)
        sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                    "--arm-root", "/nonexistent-selftest", "--sites", "1",
                    "--spot-replay", "3", "--dequantize-fp8",
                    "--shard-across", "2", "--assert-multi-device"]
        main()
        check("--spot-replay: flag forwarded", seen.get("spot_fp8") is True)
        sh = seen.get("spot_shard")
        check("--spot-replay: shard spec survives beside it",
              sh is not None and sh.shard_across == [0, 1] and sh.require_multi_device,
              str(sh))
    finally:
        globals()["collect"], globals()["spot_replay"] = real_collect, real_spot
        sys.argv = real_argv

    print("== selftest 8: THE CP-1 GATE'S IDENTITY ON A v2.1 CORPUS DOES NOT MOVE ==")
    #  The whole admissibility of this change rests here. `pick_spot_ids` decides
    #  WHICH texts the certified gate replays; the v3 rebank window is the only
    #  reason a certified-gate config may be touched at all, and it is only
    #  admissible if the v2.1 behaviour is provably unchanged. So: recorded from
    #  the pre-change code, asserted byte-exact against the derived path.
    v21 = _v21_fixture_entries()
    v21_s5 = _v21_s5_fixture_entries()
    check("this file names no stratum outside the legacy documentation constant",
          LEGACY_V21_STRATA == ("S1", "S2", "S3", "S5"),
          "the constant is kept as PROVENANCE for the recordings below and is "
          "read by nothing that makes a decision")
    check("a v2.1 manifest derives S1,S2,S3 — the legacy tuple minus its "
          "never-present S5", derive_strata(v21) == ("S1", "S2", "S3"),
          str(derive_strata(v21)))
    check("the S5-augmented manifest derives the legacy tuple EXACTLY",
          derive_strata(v21_s5) == LEGACY_V21_STRATA, str(derive_strata(v21_s5)))
    for name, entries, recorded in (
            ("v2.1-shaped fixture", v21, LEGACY_PICK_V21_K20),
            ("S5-augmented fixture", v21_s5, LEGACY_PICK_V21_S5_K20)):
        derived = pick_spot_ids(entries, len(recorded))
        check(f"{name}: the derived path reproduces the RECORDED pre-change "
              f"selection byte-exact at K={len(recorded)}",
              tuple(derived) == recorded,
              f"first three {derived[:3]}" if tuple(derived) == recorded
              else f"got {derived!r} vs recorded {list(recorded)!r}")
        prefix_ok = all(tuple(pick_spot_ids(entries, k)) == recorded[:k]
                        for k in range(3, len(recorded) + 1))
        check(f"{name}: and at EVERY K from 3 to {len(recorded)} — the pick is "
              f"incremental, so one recorded selection pins every K the gate "
              f"runs at", prefix_ok)
    if REAL_V21_MANIFEST.exists():
        real_entries = json.loads(REAL_V21_MANIFEST.read_text())["entries"]
        check("the REAL v2.1 fitting manifest of record derives S1,S2,S3",
              derive_strata(real_entries) == ("S1", "S2", "S3"),
              f"{len(real_entries)} entries, counts {stratum_counts(real_entries)}")
        got = pick_spot_ids(real_entries, len(LEGACY_PICK_REAL_V21_K20))
        check("and its gate selection is byte-exact against the recording — the "
              "corpus every certified v2.1 bank was collected on",
              tuple(got) == LEGACY_PICK_REAL_V21_K20,
              f"K=3 picks {got[:3]}" if tuple(got) == LEGACY_PICK_REAL_V21_K20
              else f"got {got!r}")
    else:
        skip("the gate-identity proof against the REAL v2.1 fitting manifest",
             f"{REAL_V21_MANIFEST} is absent from this tree (a repo data "
             f"artifact; a deployed code tree carries none). The two "
             f"v2.1-shaped fixtures above prove the same claim on synthetic "
             f"manifests of the same shape")

    print("== selftest 8b: the vocabulary is the BASIS's — v3, single, unknown ==")
    v3 = _webtext_v3_fixture_entries()
    check("a webtext-v3 manifest derives its four strata in manifest order",
          derive_strata(v3) == ("wikitext", "c4", "pg19", "stackexchange"),
          str(derive_strata(v3)))
    picks3 = pick_spot_ids(v3, 8)
    check("and pick_spot_ids RUNS on it — this is the canary's KeyError, closed",
          [e["stratum"] for e in v3 if e["text_id"] in picks3] and len(picks3) == 8,
          f"{picks3[:4]}… (the old code raised KeyError: 'wikitext' here, after "
          f"a full collection pass, with the bank already written)")
    check("the picks round-robin the four strata, two apiece at K=8",
          sorted({s: sum(1 for t in picks3
                         if t.startswith(s + "-")) for s in derive_strata(v3)
                  }.values()) == [2, 2, 2, 2],
          str({s: sum(1 for t in picks3 if t.startswith(s + "-"))
               for s in derive_strata(v3)}))
    check("it is DETERMINISTIC (same manifest, same picks, no seed anywhere)",
          pick_spot_ids(v3, 8) == picks3)
    single = [{"text_id": f"only-{i:03d}", "stratum": "only"} for i in range(10)]
    check("a SINGLE-stratum manifest works by construction",
          len(pick_spot_ids(single, 3)) == 3
          and derive_strata(single) == ("only",),
          str(pick_spot_ids(single, 3)))
    exotic = [{"text_id": f"{s}-{i}", "stratum": s}
              for s in ("zeta", "alpha") for i in range(4)]
    check("so does a vocabulary NOTHING in this campaign has seen, in "
          "first-occurrence (never sorted) order",
          derive_strata(exotic) == ("zeta", "alpha")
          and len(pick_spot_ids(exotic, 5)) == 5,
          f"{derive_strata(exotic)} -> {pick_spot_ids(exotic, 5)}")
    for entries_, k_, why in (
            (v21, len(v21) + 1,
             "K larger than the corpus is refused (it used to loop FOREVER, "
             "burning a scheduler slot and reporting nothing)"),
            (v21, 0, "K < 1 is refused")):
        try:
            pick_spot_ids(entries_, k_)
            check(why, False, "no raise")
        except (ValueError, StrataDerivationError) as exc:
            check(why, True, str(exc)[:70])
    try:
        pick_spot_ids(v21, 3, strata=("S1", "S2"))
        check("an explicit vocabulary that MISSES a stratum is refused", False,
              "no raise")
    except StrataDerivationError as exc:
        check("an explicit vocabulary that MISSES a stratum is refused — its "
              "texts could never be picked and the gate would silently "
              "under-cover the corpus", True, str(exc)[:70])

    print("== selftest 9: --collect --spot-replay in ONE invocation is REFUSED ==")
    #  RAKE M45: SystemExit is not Exception. It is caught HERE, recorded as this
    #  block's result, and the suite continues to its TOTAL line.
    real_argv2 = sys.argv
    try:
        sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                    "--arm-root", "/nonexistent-selftest", "--sites", "1",
                    "--collect", "--spot-replay", "3"]
        try:
            rc = main()
            check("both flags together raise SystemExit", False,
                  f"main() returned {rc!r} — the silent-skip class is back")
        except SystemExit as exc:
            message = str(exc.code if isinstance(exc.code, str) else exc)
            check("both flags together are REFUSED at the CLI", True,
                  message.splitlines()[0])
            check("and the refusal NAMES the two-invocation chain",
                  "--collect" in message and "--spot-replay" in message
                  and "&&" in message,
                  "a refusal that does not say what to run instead is a "
                  "riddle; this one prints the chain")
            check("and says WHY one process cannot certify the gate",
                  "never left memory" in message or "TWO PROCESSES" in message,
                  "the claim is that a SEPARATE process reproduces the bank")
        #  The refusal must not have cost either mode its own path.
        sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                    "--arm-root", "/nonexistent-selftest", "--sites", "1",
                    "--collect"]
        seen.clear()
        globals()["collect"] = lambda *a, **kw: seen.update(ran="collect")
        globals()["spot_replay"] = lambda *a, **kw: (seen.update(ran="spot"), 0)[1]
        main()
        check("--collect ALONE still dispatches to collect", seen.get("ran") == "collect")
        sys.argv = ["collect_mean_states", "--model", "8b", "--model-path", ".",
                    "--arm-root", "/nonexistent-selftest", "--sites", "1",
                    "--spot-replay", "3"]
        main()
        check("--spot-replay ALONE still dispatches to the gate",
              seen.get("ran") == "spot")
    finally:
        globals()["collect"], globals()["spot_replay"] = real_collect, real_spot
        sys.argv = real_argv2

    print(f"\nselftest: {len(fails)} failures")
    for f in fails:
        print(f"  FAILING CHECK: {f}")
    # RAKE M44: coverage is part of the verdict, per configuration. A bare pass count
    # cannot be read without knowing which cell of the matrix produced it.
    print(f"selftest checks run: {len(checks)} ({len(skips)} named skip(s))")
    for name in skips:
        print(f"  SKIPPED {name}")
    # RAKE M45 rule (b): a sweep log without its terminal TOTAL line is a FAILING
    # sweep, never a quiet pass. This line is that terminus for this module.
    print(f"TOTAL collect_mean_states: {len(checks)} check(s), {len(fails)} "
          f"failure(s), {len(skips)} skip(s)")
    return 1 if fails else 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--model", choices=MODEL_KEYS, help="one model per invocation; "
                    "keys outside the fixed-grid registry (SITES) are scan-registry "
                    "models (metabasis.roster) and REQUIRE --sites")
    ap.add_argument("--model-path", help="local weights dir (node-side)")
    ap.add_argument("--arms", default="native,raw",
                    help="comma list; strata/arms never mixed in banks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default=None,
                    help="comma-separated site override (default = the model's SITES grid); "
                         "used by Leg-4F to bank Qwen L18, where Vdiverge lives")
    ap.add_argument("--max-seq-len", type=int, default=None, metavar="N",
                    help="prereg ADDENDUM 2026-07-27-B position-ceiling deviation: "
                         "truncate every text to its FIRST N tokens. Required for "
                         "architectures with learned absolute position embeddings "
                         "(gpt2-xl: 1024). MUST be passed identically to --collect and "
                         "--spot-replay, or the replay compares different objects and "
                         "the bitwise gate fails. Unset = the historical untruncated "
                         "path, byte for byte.")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--spot-replay", type=int, metavar="K", default=None)
    ap.add_argument("--cp1-summary", nargs="*", metavar="MODEL", default=None,
                    help="e.g. --cp1-summary 3b 8b (local, no GPU)")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only gates on the --dequantize-fp8 regime and its wiring")
    # --- sharded path (prereg §4 big rungs); OPT-IN, none of these => single card ---
    sh = ap.add_argument_group(
        "sharded load (>=70B class)",
        "opt-in accelerate device_map placement across the node. Unset => the unchanged "
        "single-card path. --device is ignored when sharding (inputs go to the "
        "input-embedding device). Certify with collect+spot-replay in ONE job, plus a "
        "byte-compare against a fresh single-device collection of a small model.")
    sh.add_argument("--device-map", default=None, metavar="SPEC",
                    help=f"one of {ACCELERATE_STRATEGIES}, or '@map.json' / inline JSON "
                         "{module: device}")
    sh.add_argument("--shard-across", default=None, metavar="SPEC",
                    help="build an EXPLICIT even decoder-layer split: 'N' (GPU ordinals "
                         "0..N-1) or a device list ('0,1,2,3' | 'cpu,disk'). The forced "
                         "multi-device layout the certification gate needs on a small "
                         "model; mutually exclusive with --device-map")
    sh.add_argument("--max-memory", default=None, metavar="SPEC",
                    help="'0=170GiB,1=170GiB,cpu=0GiB' or '@mm.json'; cpu=0GiB forbids "
                         "silent CPU offload")
    sh.add_argument("--offload-folder", default=None, metavar="DIR",
                    help="required if any device resolves to 'disk'")
    sh.add_argument("--allow-offload", action="store_true",
                    help="accept a layout that spills weights to cpu/disk instead of "
                         "failing fast (slow; compute still runs on the execution device)")
    sh.add_argument("--assert-multi-device", action="store_true",
                    help="fail unless the realized layout computes on >=2 devices — set "
                         "this on the certification run so it cannot pass vacuously")
    # --- dtype regime (prereg ADDENDUM 2026-07-26-A, roster row 21); OPT-IN ---------
    ap.add_argument("--dequantize-fp8", action="store_true",
                    help="load a block-wise-FP8 checkpoint as bf16 via "
                         "FineGrainedFP8Config(dequantize=True) — prereg ADDENDUM "
                         "2026-07-26-A, BINDING for roster row 21 (DeepSeek-V3). Unset "
                         "= the historical call, byte for byte, and an FP8 checkpoint "
                         "would load its NATIVE FP8 forward instead (a different object, "
                         "and not differentiable). MUST be passed identically to "
                         "--collect and --spot-replay, or the replay compares different "
                         "objects and the bitwise gate fails for the wrong reason. The "
                         "post-load witness (no FP8 modules, every float parameter "
                         "bfloat16) is asserted and banked in the trunk stamp.")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    #  THE SILENT-SKIP CLASS DIES AT THE CLI (desk ruling 2026-08-03).
    #  `--collect --spot-replay K` in ONE invocation used to be accepted and then
    #  IGNORED: the dispatch below returns after `--collect`, so the replay never
    #  ran, no spot report was written, and the job's own JOB-OK line pointed at
    #  an artifact that did not exist. Runtime-proven on the node, 2026-08-03.
    #
    #  A REFUSAL, not a silent re-order into the right thing: the gate's whole
    #  claim is that a SEPARATE PROCESS reproduces the bank bit for bit, and one
    #  process that collects and then replays in-memory would certify a model
    #  object that never left memory. Two processes in one job is the only shape
    #  that certifies what the gate says it certifies, and only the caller can
    #  arrange that — so the caller is told, in the words of the chain.
    if args.collect and args.spot_replay is not None:
        raise SystemExit(
            "--collect and --spot-replay in ONE invocation is REFUSED. This "
            "process would collect and then EXIT, silently skipping the gate "
            "(no spot report is written, and a job that prints JOB-OK on the "
            "exit code alone points at an artifact that does not exist).\n"
            "The CP-1 gate is TWO PROCESSES IN ONE JOB — a genuinely fresh "
            "replay of the bank the first process wrote. Run the chain:\n"
            "  python -m metabasis.scripts.collect_mean_states --collect "
            "<ARGS> \\\n"
            "  && python -m metabasis.scripts.collect_mean_states "
            f"--spot-replay {args.spot_replay} <ARGS>\n"
            "with <ARGS> built ONCE (a shell array) so the two invocations "
            "cannot drift apart — identical --sites/--arms/--max-seq-len/"
            "--shard-across, or the replay compares different objects and the "
            "bitwise gate fails for the wrong reason.")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; valid: {ARMS}")
    sites_override = (tuple(int(x) for x in args.sites.split(','))
                      if args.sites else None)

    shard: Optional[ShardSpec] = None
    if args.device_map is not None and args.shard_across is not None:
        raise SystemExit("--device-map and --shard-across are mutually exclusive")
    if args.device_map is not None or args.shard_across is not None:
        try:
            shard = ShardSpec(
                device_map=(parse_device_map(args.device_map)
                            if args.device_map is not None else None),
                shard_across=(parse_shard_across(args.shard_across)
                              if args.shard_across is not None else None),
                max_memory=(parse_max_memory(args.max_memory)
                            if args.max_memory else None),
                offload_folder=args.offload_folder,
                allow_offload=args.allow_offload,
                require_multi_device=args.assert_multi_device,
                spec_echo=(f"--device-map {args.device_map}" if args.device_map
                           else f"--shard-across {args.shard_across}")
                + (f" --max-memory {args.max_memory}" if args.max_memory else ""))
        except ShardedLoadError as exc:
            raise SystemExit(f"{type(exc).__name__}: {exc}") from exc
    elif any((args.max_memory, args.offload_folder, args.allow_offload,
              args.assert_multi_device)):
        raise SystemExit("--max-memory/--offload-folder/--allow-offload/"
                         "--assert-multi-device need --device-map or --shard-across")

    if args.cp1_summary is not None:
        return cp1_summary(args.arm_root, args.cp1_summary or ["3b", "8b"])
    if not args.model or not args.model_path:
        raise SystemExit("--model and --model-path required for GPU modes")
    if args.max_seq_len is not None and args.max_seq_len < 2:
        raise SystemExit("--max-seq-len must be >= 2 (room for a completion position)")

    if args.collect:
        collect(args.arm_root, args.model, args.model_path, arms, args.device,
                sites_override=sites_override, shard=shard,
                max_length=args.max_seq_len,
                dequantize_fp8=args.dequantize_fp8)
        return 0
    if args.spot_replay is not None:
        if args.spot_replay < 3:
            raise SystemExit("CP-1 gate requires K >= 3")
        # --sites forwards on BOTH paths (desk unification, 2026-07-26): a bank
        # collected with an overridden grid must spot-replay against that same
        # grid; the registry lookup (`sites_for(model)`, which refuses loudly for
        # an un-ratified key) stays the default when --sites is absent, so
        # banked-model replays are unchanged.
        return spot_replay(args.arm_root, args.model, args.model_path, arms,
                           args.device, args.spot_replay,
                           sites_override=sites_override,
                           shard=shard, max_length=args.max_seq_len,
                           dequantize_fp8=args.dequantize_fp8)
    raise SystemExit("pick a mode: --collect / --spot-replay K / --cp1-summary")


if __name__ == "__main__":
    sys.exit(main())
