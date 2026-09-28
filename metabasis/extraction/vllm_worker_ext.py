"""The TP-CAPABLE ATTACHMENT — the metabasis instrument inside every vLLM rank.

Lifted BY VALUE from the beside tool `metabasis_worker_ext.py`
(sha256 c24547985ca1790c941fd8ef3ac7aadf9b2126d68ff0ef186d7dce70e5666c87) at
re-freeze #4, 2026-08-20. UNSTAMPED (C§8): this module files ingredients and
never scores a bar.

WHY THIS EXISTS. The certified lane attaches its hooks by reaching into the
engine IN-PROCESS (`vllm_lane_smoke._find_model`, which walks
`llm_engine.…driver_worker.model_runner.model`). That path exists only under
vLLM's UniProcExecutor, i.e. tensor_parallel_size == 1. At TP>1 the model lives
in separate Worker processes and the parent holds no model object at all
(`MultiprocExecutor` has no `driver_worker`), so the steering write and the
entropy probe would silently never attach. It fails loudly rather than quietly,
which is why no unsteered column could ever have banked at TP>1.

WHAT REPLACES IT. vLLM 0.15.1 supports `worker_extension_cls`
(vllm/v1/worker/worker_base.py:262-287): a qualified class name whose methods are
dynamically mixed into the Worker class INSIDE every worker process and are then
callable by name through `LLM.collective_rpc`. `self` is the Worker, so
`self.model_runner.model` is the real, sharded, loaded module on that rank. The
reference implementation is NVIDIA NeMo-RL's `VllmInternalWorkerExtension`
(a mixin with no `__init__`, proven at TP=4); this module follows that shape.

THE HANG THAT HID ALL OF THIS, AND ITS FIX. Every TP>1 launch of the lane hung
before a single `Worker_TP` line appeared. `VLLM_WORKER_MULTIPROC_METHOD`
defaults to `fork` (vllm/envs.py:62, :738-739) and `MultiprocExecutor` forks via
`get_mp_context()` (vllm/v1/executor/multiproc_executor.py:138); vLLM's
`_maybe_force_spawn()` (vllm/utils/system_utils.py:120-121) escalates to spawn
only when CUDA is already initialised, which the lane's CPU-only preflight never
does. The lane's runner exports `OMP_NUM_THREADS=8` — which also suppresses
vLLM's own `torch.set_num_threads(1)` — and the column tool runs ~98 CPU torch
selftest checks BEFORE `LLM()`. Forking worker processes from a parent already
dirtied by an OpenMP thread pool is the hang. Discriminated by value: plain vLLM
with the same `OMP_NUM_THREADS=8` + CPU torch warmup hangs under fork and passes
under spawn (jobs `2b709ccc886c`, `cb93dd7f4919`). **The fix is
`VLLM_WORKER_MULTIPROC_METHOD=spawn`**, exported by the runner; it is INERT at
TP=1 because `UniProcExecutor` forks nothing. The precise C-level mechanism
(OpenMP thread-pool mutex inheritance across fork) is inferred, not captured;
the fix does not depend on it.

WHY THE CERTIFIED ARITHMETIC IS UNCHANGED AT TP>1 (both proven from vLLM source;
`vllm_residual_write.py` and `vllm_entropy_probe.py` are used UNEDITED and their
shas are pinned in this module's selftest):
  * RESIDUAL WRITE. A decoder layer's hidden_states in/out is the FULL d_model
    stream, REPLICATED on every rank: attention `o_proj` and MLP `down_proj` are
    RowParallelLinear (vllm/model_executor/models/qwen2.py:162, :94) whose default
    reduce_results=True all-reduces before the layer returns. The site-s layer
    INPUT the lane writes is therefore post-all-reduce and unsharded, so every
    rank applies the SAME full-width delta and the streams stay consistent. No
    sharding arithmetic is introduced.
  * ENTROPY PROBE. `compute_logits` all-gathers across TP
    (LogitsProcessor.use_all_gather is True -- vllm/platforms/interface.py:563)
    and strips vocab padding (`logits[..., :org_vocab_size]`,
    vllm/model_executor/layers/logits_processor.py:102). Every rank therefore
    sees the identical FULL-vocab 2-D tensor, so the certified reduction is
    already correct; the desk reads rank 0 and uses the other ranks as a free
    cross-rank agreement check — a check the parent-side proxies ENFORCE rather
    than observe (`metabasis.extraction.vllm_tp_proxy`).

THE `mb_` PREFIX IS A CONTRACT, NOT A STYLE. `worker_base.py:269-275` asserts the
extension class shares no attribute name with the Worker class and ABORTS engine
construction on a collision, so an unprefixed `model`, `rank` or `attach` would
take the whole engine down at load. Every public name here is `mb_`-prefixed and
a selftest asserts it.

NOTHING HERE IS CERTIFIED. This files ingredients only; it never scores a bar.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

C8 = "UNSTAMPED (C§8) -- the TP attachment. No verdicts here."

#: The `mb_` methods the parent may reach by `collective_rpc`. Pinned so a
#: rename cannot silently strand a proxy call at TP>1 (the proxies in
#: `metabasis.extraction.vllm_tp_proxy` are checked against this exact set).
RPC_SURFACE: frozenset[str] = frozenset({
    "mb_attach", "mb_detach", "mb_set_spec", "mb_set_mode", "mb_capture_inputs",
    "mb_entropy_mode", "mb_take_entropy", "mb_take_span_stats", "mb_probe",
    "mb_reset_stats", "mb_stats_now", "mb_reset_mask", "mb_mask_counters",
    "mb_cap_reset", "mb_cap_flat", "mb_measure_sec25",
})

#: sha256 of the two CERTIFIED extraction modules this attachment drives. They
#: are byte-unchanged by re-freeze #4 — the TP-correctness argument is that they
#: were already right — and the selftest refuses if either moves.
CERTIFIED_MODULE_SHA256: dict[str, str] = {
    "vllm_residual_write.py":
        "fafc07d78201b5e8ca909244dff419a892daf0fd316eb3520f648887a05d514f",
    "vllm_entropy_probe.py":
        "e9902a21e4ef65b961ebf49b71687ecfe38e2632155af90d8f2737ff2bdcddc0",
}


class WorkerAttachmentRefused(RuntimeError):
    """The rank could not attach the instrument, and says which prerequisite failed."""


def _mb_code_on_path() -> None:
    """Put the metabasis package on sys.path inside the worker process."""
    code = os.environ.get("MB_CODE")
    if code and code not in sys.path:
        sys.path.insert(0, code)


class MetabasisWorkerExtension:
    """Mixed into vLLM's Worker in EVERY TP rank; methods reached by collective_rpc.

    Every public attribute is `mb_`-prefixed: worker_base.py:269-275 asserts that
    the extension shares no attribute name with the Worker class, and an
    unprefixed name (`model`, `rank`, `attach`) would abort engine construction.
    """

    # class-level default so the attribute always resolves, even before mb_attach
    _mb: Optional[dict[str, Any]] = None

    # ---- attachment ---------------------------------------------------------

    def mb_attach(self, site: int, expect_hidden_dim: Optional[int] = None
                  ) -> dict[str, Any]:
        """Attach the certified steering wrapper + entropy capture on THIS rank."""
        _mb_code_on_path()
        import torch
        from metabasis.extraction.vllm_residual_write import attach_steering_layer
        from metabasis.extraction.vllm_entropy_probe import (EntropyCapture,
                                                             attach_entropy_capture)

        if self._mb is not None:
            raise RuntimeError("metabasis instrument already attached on this rank "
                               "-- double attachment would double the dose")

        model = self.model_runner.model
        wrapper = attach_steering_layer(model, int(site))
        hidden = int(wrapper.hidden_dim)
        if expect_hidden_dim is not None and hidden != int(expect_hidden_dim):
            raise RuntimeError(
                f"rank hidden_dim {hidden} != expected {expect_hidden_dim}; the "
                "site's residual stream is not the width the banked vector assumes")

        cap = EntropyCapture()
        detach_ent = attach_entropy_capture(model, cap)

        state: dict[str, Any] = {
            "wrapper": wrapper, "cap": cap, "detach_ent": detach_ent,
            "model": model, "orig_forward": model.forward, "site": int(site),
            "mode": "generation", "prompt_len": None, "prompt_lengths": None,
            "mask_true": 0, "mask_n": 0,
        }

        # The span shim: exactly the runner shim the certified TP=1 tool installs,
        # moved inside the worker. The segmentation function is IMPORTED FROM THE
        # COLUMN TOOL rather than re-derived, so the span rule cannot drift
        # between the TP=1 and TP>1 lanes.
        #
        # RE-FREEZE #4 DELTA, AND THE ONLY SEMANTIC ONE IN THIS LIFT. The beside
        # tool imported the flat node-side module (`from vllm_lane_column import
        # …`), which resolves against the runner's `PYTHONPATH=$TOOLS`. In the
        # repo the column tool IS `metabasis.scripts.vllm_lane_column`, and
        # keeping the flat import would let a redeployed tree silently take its
        # span rule from a STALE beside copy while the parent took it from the
        # repo — two span rules in one column, which is precisely the drift this
        # import exists to prevent. The repo path is therefore named explicitly
        # and there is no fallback: an unresolvable span rule is a refusal.
        try:
            from metabasis.scripts.vllm_lane_column import span_mask_from_positions
        except ImportError as exc:                                   # no fallback
            raise WorkerAttachmentRefused(
                "the worker cannot import the span rule from "
                "`metabasis.scripts.vllm_lane_column`; MB_CODE must point at the "
                "code root that carries the merged repo tree (got MB_CODE="
                f"{os.environ.get('MB_CODE')!r}). Falling back to a flat "
                "`vllm_lane_column` on PYTHONPATH would risk taking the span rule "
                "from a stale beside copy, so this refuses instead") from exc

        def shim(*a: Any, **k: Any) -> Any:
            positions = k.get("positions")
            if positions is None and len(a) >= 2 and isinstance(a[1], torch.Tensor):
                positions = a[1]
            if positions is None:
                wrapper.set_position_mask(None)
            else:
                if state["mode"] == "scoring":
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
            return state["orig_forward"](*a, **k)

        model.forward = shim
        self._mb = state
        return {"rank": int(getattr(self, "rank", -1)),
                "hidden_dim": hidden,
                "site": int(site),
                "wrapped_type": type(wrapper.wrapped_layer).__name__,
                "n_layers_seen": None}

    def mb_detach(self) -> dict[str, Any]:
        """Restore the rank to stock. A survivor across columns is contamination."""
        _mb_code_on_path()
        from metabasis.extraction.vllm_residual_write import detach_steering_layer
        st = self._mb
        if st is None:
            return {"detached": False}
        st["model"].forward = st["orig_forward"]
        st["detach_ent"]()
        detach_steering_layer(st["model"], st["site"])
        self._mb = None
        return {"detached": True}

    # ---- per-cell control ---------------------------------------------------

    def mb_set_spec(self, vector: Optional[Any], alpha: float,
                    alpha_frac: Optional[float], measured_norm: float) -> None:
        """Install (or clear) the dose on this rank. `vector` is a numpy array."""
        _mb_code_on_path()
        import numpy as np
        import torch
        from metabasis.extraction.vllm_residual_write import MetabasisSteeringSpec
        st = self._mb
        if st is None:
            raise RuntimeError("mb_set_spec before mb_attach")
        if vector is None or float(alpha) == 0.0:
            st["wrapper"].set_spec(None)
            return
        # collective_rpc marshals its arguments, and a numpy array arrives on the
        # far side as a plain nested list. Normalise BY VALUE rather than trusting
        # the type: a silently-wrong dtype here would be a silently-wrong dose.
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        expect = int(st["wrapper"].hidden_dim)
        if arr.shape[0] != expect:
            raise RuntimeError(
                f"steering vector has {arr.shape[0]} elements but this rank's "
                f"residual stream is {expect} wide")
        st["wrapper"].set_spec(MetabasisSteeringSpec(
            layer_idx=int(st["site"]),
            vector=torch.as_tensor(arr, dtype=torch.float32),
            alpha=float(alpha), alpha_frac=alpha_frac,
            measured_norm=float(measured_norm)))

    def mb_set_mode(self, mode: str, prompt_len: Optional[int] = None,
                    prompt_lengths: Optional[Any] = None) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_set_mode before mb_attach")
        if mode not in ("generation", "scoring"):
            raise ValueError(f"unknown mode {mode!r}")
        st["mode"] = mode
        st["prompt_len"] = prompt_len
        # collective_rpc marshals a frozenset to a list; the certified
        # span_mask_from_positions wants a frozenset of ints.
        st["prompt_lengths"] = (None if prompt_lengths is None
                                else frozenset(int(x) for x in prompt_lengths))

    def mb_capture_inputs(self, enabled: bool) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_capture_inputs before mb_attach")
        st["wrapper"].capture_input_hidden_states(bool(enabled))

    def mb_entropy_mode(self, mode: str) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_entropy_mode before mb_attach")
        st["cap"].mode = mode

    # ---- readout ------------------------------------------------------------

    def mb_take_entropy(self) -> dict[str, Any]:
        """Drain this rank's entropy rows. Rank 0 is the read; others cross-check."""
        st = self._mb
        if st is None:
            raise RuntimeError("mb_take_entropy before mb_attach")
        cap = st["cap"]
        rows = [r.tolist() for r in cap.rows]
        out = {"rank": int(getattr(self, "rank", -1)), "rows": rows,
               "counts": list(cap.counts), "calls": int(cap.calls),
               "rows_seen": int(cap.rows_seen)}
        cap.rows = []
        cap.counts = []
        cap.calls = 0
        cap.rows_seen = 0
        return out

    def mb_take_span_stats(self) -> dict[str, Any]:
        """The exact token accounting FOOTGUN-5 gate, per rank."""
        st = self._mb
        if st is None:
            raise RuntimeError("mb_take_span_stats before mb_attach")
        w = st["wrapper"]
        stats = {"rank": int(getattr(self, "rank", -1)),
                 "mask_true": int(st["mask_true"]), "mask_n": int(st["mask_n"]),
                 "wrapper_stats": dict(getattr(w, "stats", {}))}
        st["mask_true"] = 0
        st["mask_n"] = 0
        try:
            w.reset_stats()
        except Exception:                                    # noqa: BLE001
            pass
        return stats

    def mb_probe(self) -> dict[str, Any]:
        """Cheap liveness/identity read: proves the mixin reached this rank."""
        st = self._mb
        return {"rank": int(getattr(self, "rank", -1)),
                "pid": os.getpid(),
                "attached": st is not None,
                "site": (None if st is None else int(st["site"]))}

    # ---- column-runner surface (drives the certified tool at TP>1) ----------

    def mb_reset_stats(self) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_reset_stats before mb_attach")
        st["wrapper"].reset_stats()

    def mb_stats_now(self) -> dict[str, Any]:
        """The wrapper's span counters WITHOUT resetting them."""
        st = self._mb
        if st is None:
            raise RuntimeError("mb_stats_now before mb_attach")
        return dict(st["wrapper"].stats)

    def mb_reset_mask(self) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_reset_mask before mb_attach")
        st["mask_true"] = 0
        st["mask_n"] = 0

    def mb_mask_counters(self) -> dict[str, int]:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_mask_counters before mb_attach")
        return {"mask_true": int(st["mask_true"]), "mask_n": int(st["mask_n"])}

    def mb_cap_reset(self, mode: str) -> None:
        st = self._mb
        if st is None:
            raise RuntimeError("mb_cap_reset before mb_attach")
        st["cap"].reset(mode)

    def mb_cap_flat(self) -> list[float]:
        """The pass's entropy scalars, in order. Scalars only -- never vocab-sized."""
        st = self._mb
        if st is None:
            raise RuntimeError("mb_cap_flat before mb_attach")
        return [float(x) for x in st["cap"].flat()]

    def mb_measure_sec25(self) -> dict[str, Any]:
        """§2.5 median residual norm, reduced ON THE RANK.

        The captures are (tokens x d_model) activations; shipping them to the
        parent would move hundreds of MB per column for a single scalar, so the
        certified reduction runs here and only the scalar crosses.
        """
        _mb_code_on_path()
        from metabasis.extraction.vllm_residual_write import (
            measure_per_token_median_resid_norm_from_captures)
        st = self._mb
        if st is None:
            raise RuntimeError("mb_measure_sec25 before mb_attach")
        caps = st["wrapper"].take_captures()
        if not caps:
            return {"rank": int(getattr(self, "rank", -1)), "n_captures": 0,
                    "median": None}
        med = measure_per_token_median_resid_norm_from_captures(caps)
        return {"rank": int(getattr(self, "rank", -1)), "n_captures": len(caps),
                "median": float(med)}


# ── selftest (desk-side, CPU, no vLLM, no weights, torch OPTIONAL) ────────────

def selftest() -> int:                                            # noqa: C901
    """Every load-bearing property of the attachment, provable with no GPU.

    TORCH IS OPTIONAL BY DESIGN (rake M44). The desk's own repo .venv has no
    torch, and the properties that make this attachment legitimate — the `mb_`
    prefix contract, the RPC surface, the no-module-scope-import rule, the
    certified modules' shas, the pre-attach refusals — are all provable without
    a deep-learning stack. The methods that genuinely need torch (`mb_attach`,
    `mb_detach`, `mb_set_spec`, `mb_measure_sec25`) are exercised behind ONE
    availability probe and recorded as NAMED SKIPS otherwise, never as passes.
    """
    import ast
    import inspect

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def ok(cond: bool, name: str, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    def skip(name: str, why: str) -> None:
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))

    from pathlib import Path as _Path
    import hashlib as _hashlib

    src = _Path(__file__).read_text()
    tree = ast.parse(src)

    # -- 1. the `mb_` prefix contract (worker_base.py:269-275) ----------------
    public = [n for n in vars(MetabasisWorkerExtension) if not n.startswith("_")]
    ok(bool(public) and all(n.startswith("mb_") for n in public),
       "every PUBLIC attribute of the extension is `mb_`-prefixed — vLLM's "
       "worker_base.py:269-275 aborts engine construction on a name the Worker "
       "already owns, so an unprefixed `model`/`rank`/`attach` would take the "
       "engine down at load",
       f"public={sorted(public)}")
    nonpublic = [n for n in vars(MetabasisWorkerExtension)
                 if not n.startswith("_") and not n.startswith("mb_")]
    ok(not nonpublic, "no public non-`mb_` name exists at all", f"{nonpublic}")
    ok("_mb" in vars(MetabasisWorkerExtension),
       "the one non-prefixed name (`_mb`) is underscore-PRIVATE, so the prefix "
       "rule reads 'public implies mb_' rather than 'mostly mb_'")

    # The collision property, stated against vLLM's real Worker attribute names
    # rather than against a hope. These are names worker_base's mixin check
    # would trip on; the extension must own none of them.
    WORKER_NAMES = {
        "model", "rank", "local_rank", "device", "model_runner", "vllm_config",
        "model_config", "parallel_config", "cache_config", "load_model",
        "init_device", "execute_model", "determine_available_memory",
        "initialize_cache", "compile_or_warm_up_model", "get_model",
        "profile", "check_health", "add_lora", "sleep", "wake_up",
    }
    ok(not (set(vars(MetabasisWorkerExtension)) & WORKER_NAMES),
       "the extension collides with NO known vLLM Worker attribute name",
       f"{sorted(set(vars(MetabasisWorkerExtension)) & WORKER_NAMES)}")

    # -- 2. the RPC surface is pinned -----------------------------------------
    live = {n for n in vars(MetabasisWorkerExtension) if n.startswith("mb_")}
    ok(live == RPC_SURFACE,
       "the live `mb_` method set is EXACTLY `RPC_SURFACE` — a rename that "
       "stranded a `collective_rpc` call at TP>1 would otherwise only surface "
       "on a loaded model",
       f"missing={sorted(RPC_SURFACE - live)} extra={sorted(live - RPC_SURFACE)}")
    ok(all(callable(getattr(MetabasisWorkerExtension, n)) for n in RPC_SURFACE),
       "every pinned name is callable")

    # -- 3. the mixin shape: NO __init__ (the NeMo-RL pattern) ----------------
    ok("__init__" not in vars(MetabasisWorkerExtension),
       "the extension defines no `__init__` — it is MIXED INTO an already-"
       "constructed Worker class, and an initialiser would never be called")
    ok(MetabasisWorkerExtension._mb is None,
       "`_mb` has a CLASS-LEVEL default so `mb_probe` resolves on a rank that "
       "never attached, instead of raising AttributeError through the RPC")

    # -- 4. AST: nothing heavy is imported at MODULE scope --------------------
    # PROPERTY, NOT A STRING NEEDLE (rake M58): the module must not import the
    # extraction modules (or torch, or numpy) at module scope, because MB_CODE
    # is only on `sys.path` after `_mb_code_on_path()` runs inside the worker.
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    ok(not (top_imports & {"torch", "numpy", "metabasis", "vllm"}),
       "no torch/numpy/metabasis/vllm import at MODULE scope — MB_CODE reaches "
       "sys.path only inside the worker, so a module-scope import would fail "
       "the mixin at engine construction",
       f"top-level imports: {sorted(top_imports)}")

    # -- 5. the two CERTIFIED modules are byte-unchanged ----------------------
    here = _Path(__file__).resolve().parent
    for fname, want in sorted(CERTIFIED_MODULE_SHA256.items()):
        p = here / fname
        if not p.exists():
            ok(False, f"the certified module {fname} is present beside this one",
               f"missing: {p}")
            continue
        got = _hashlib.sha256(p.read_bytes()).hexdigest()
        ok(got == want,
           f"{fname} is BYTE-UNCHANGED at re-freeze #4 — the TP-correctness "
           "argument is that the certified arithmetic was already right, and a "
           "single edited byte would void it",
           f"{got} vs {want}")

    # -- 6. the docstring carries the diagnosis and the source citations ------
    # Tested against `__doc__` (prose, an ARTIFACT) rather than by grepping this
    # file, so the check's own text cannot satisfy it — rake M58(2).
    doc = __doc__ or ""
    for needle, what in (
            ("VLLM_WORKER_MULTIPROC_METHOD=spawn", "the fix"),
            ("fork", "the fork diagnosis"),
            ("OMP_NUM_THREADS", "the dirtied-parent ingredient"),
            ("qwen2.py:162", "RowParallelLinear reduce_results=True"),
            ("platforms/interface.py:563", "use_all_gather"),
            ("logits_processor.py:102", "the vocab-padding strip"),
            ("worker_base.py:269-275", "the attribute-collision abort"),
            ("worker_base.py:262-287", "the worker_extension_cls contract")):
        ok(needle in doc, f"the module docstring carries {what} ({needle})")

    # -- 7. the pre-attach refusals, on a fake rank (no torch needed) ---------
    class _FakeRank(MetabasisWorkerExtension):
        rank = 0

    bare = _FakeRank()
    for meth, margs in (("mb_set_mode", ("generation",)),
                        ("mb_capture_inputs", (True,)),
                        ("mb_entropy_mode", ("generation",)),
                        ("mb_take_entropy", ()),
                        ("mb_take_span_stats", ()),
                        ("mb_reset_stats", ()),
                        ("mb_stats_now", ()),
                        ("mb_reset_mask", ()),
                        ("mb_mask_counters", ()),
                        ("mb_cap_reset", ("off",)),
                        ("mb_cap_flat", ())):
        try:
            getattr(bare, meth)(*margs)
            ok(False, f"{meth} REFUSES before mb_attach")
        except RuntimeError as exc:
            ok("before mb_attach" in str(exc),
               f"{meth} REFUSES before mb_attach, and the message says so",
               str(exc))
    probe = bare.mb_probe()
    ok(probe["attached"] is False and probe["site"] is None and probe["rank"] == 0,
       "mb_probe answers on an UNATTACHED rank — it is the liveness read that "
       "proves the mixin arrived, so it must never depend on attachment",
       f"{probe}")

    # -- 8. the attached surface, on a fake rank ------------------------------
    class _FakeWrapper:
        def __init__(self) -> None:
            self.stats = {"tokens_seen": 7, "tokens_injected": 3,
                          "prefill_tokens_skipped": 4}
            self.reset_calls = 0

        def reset_stats(self) -> None:
            self.reset_calls += 1
            self.stats = {"tokens_seen": 0, "tokens_injected": 0,
                          "prefill_tokens_skipped": 0}

    class _FakeCap:
        def __init__(self) -> None:
            self.mode = "off"
            self.rows: list[Any] = []
            self.counts: list[int] = []
            self.calls = 0
            self.rows_seen = 0
            self.reset_modes: list[str] = []

        def reset(self, mode: str) -> None:
            self.reset_modes.append(mode)

    w, c = _FakeWrapper(), _FakeCap()
    att = _FakeRank()
    att._mb = {"wrapper": w, "cap": c, "site": 26, "mode": "generation",
               "prompt_len": None, "prompt_lengths": None,
               "mask_true": 11, "mask_n": 20}

    ok(att.mb_stats_now() == {"tokens_seen": 7, "tokens_injected": 3,
                              "prefill_tokens_skipped": 4},
       "mb_stats_now reads the wrapper's counters WITHOUT resetting them",
       f"reset_calls={w.reset_calls}")
    ok(w.reset_calls == 0, "…and it really did not reset them")
    ok(att.mb_mask_counters() == {"mask_true": 11, "mask_n": 20},
       "mb_mask_counters reports the shim's own independent count")
    span = att.mb_take_span_stats()
    ok(span["mask_true"] == 11 and span["mask_n"] == 20
       and span["wrapper_stats"]["tokens_seen"] == 7,
       "mb_take_span_stats DRAINS: it returns the counts and then zeroes them",
       f"{span}")
    ok(att._mb["mask_true"] == 0 and att._mb["mask_n"] == 0 and w.reset_calls == 1,
       "…and the drain zeroed both the shim counters and the wrapper stats")
    att._mb["mask_true"] = 5
    att.mb_reset_mask()
    ok(att._mb["mask_true"] == 0, "mb_reset_mask zeroes the shim counters")
    c.rows = [1.5, 2.5]
    c.flat = lambda: [1.5, 2.5]                        # type: ignore[assignment]
    ok(att.mb_cap_flat() == [1.5, 2.5],
       "mb_cap_flat ships SCALARS — never a vocab-sized tensor across the RPC")
    att.mb_cap_reset("scoring")
    ok(c.reset_modes == ["scoring"], "mb_cap_reset forwards the mode verbatim")
    att.mb_entropy_mode("generation")
    ok(c.mode == "generation", "mb_entropy_mode sets the capture mode")
    att.mb_set_mode("scoring", 12, [3, 1, 2, 1])
    ok(att._mb["mode"] == "scoring" and att._mb["prompt_len"] == 12,
       "mb_set_mode installs the scoring mode and its prompt length")
    ok(att._mb["prompt_lengths"] == frozenset({1, 2, 3}),
       "mb_set_mode re-freezes prompt_lengths — collective_rpc marshals a "
       "frozenset to a LIST, and the certified span rule wants a frozenset",
       f"{att._mb['prompt_lengths']!r}")
    ok(isinstance(att._mb["prompt_lengths"], frozenset),
       "…and the TYPE, not just the contents, is restored")
    try:
        att.mb_set_mode("verdict")
        ok(False, "mb_set_mode REFUSES an unknown mode")
    except ValueError as exc:
        ok("unknown mode" in str(exc),
           "mb_set_mode REFUSES an unknown mode", str(exc))
    c.rows, c.counts, c.calls, c.rows_seen = [], [], 3, 4
    ent = att.mb_take_entropy()
    ok(ent["calls"] == 3 and c.calls == 0 and c.rows_seen == 0,
       "mb_take_entropy DRAINS the capture and leaves it empty", f"{ent}")

    # -- 9. torch-dependent: mb_attach's refusals -----------------------------
    try:
        import torch                                              # noqa: F401
        _have_torch = True
    except ImportError:
        _have_torch = False
    if not _have_torch:
        skip("mb_attach's refusal pair (double attachment; hidden width != the "
             "banked vector's) exercised on a live rank",
             "torch is absent from this interpreter; the refusals are exercised "
             "on the node, where the runner's CPU preflight imports torch "
             "before any GPU. Evidence MISSING here: that the guards fire on a "
             "real attach, not merely that they are present in the source")
    else:
        atree = ast.parse(inspect.cleandoc(
            inspect.getsource(MetabasisWorkerExtension.mb_attach)))
        raises = [n for n in ast.walk(atree) if isinstance(n, ast.Raise)]
        ok(len(raises) >= 2,
           "mb_attach carries BOTH refusals (double attachment, and a hidden "
           "width that is not the banked vector's)", f"{len(raises)} raises")

    # -- 10. M59: the suite's own arithmetic ----------------------------------
    SELFTEST_CHECK_FLOOR = 46
    KNOWN_SKIP_CEILING = 1
    ok(len(checks) >= SELFTEST_CHECK_FLOOR and len(skips) <= KNOWN_SKIP_CEILING,
       f"(M59) the suite ran at least its recorded floor of "
       f"{SELFTEST_CHECK_FLOOR} checks and named no more than "
       f"{KNOWN_SKIP_CEILING} skip(s) — a block that stopped running, or "
       "started skipping, is caught here rather than read as a clean run",
       f"{len(checks)} checks (floor {SELFTEST_CHECK_FLOOR}), "
       f"{len(skips)} named skip(s) (ceiling {KNOWN_SKIP_CEILING})")

    fails = [c for c in checks if not c[1]]
    print(f"vllm_worker_ext selftest checks run: {len(checks)} "
          f"({len(skips)} named skip(s)), {len(fails)} failure(s)")
    for name, _, detail in fails:
        print(f"  MISS {name} — {detail}")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
