"""Parent-side proxies for the tensor-parallel vLLM lane.

Lifted BY VALUE from the beside tool `metabasis_tp_proxy.py`
(sha256 a14cb464d2cf1e7298ba63c9ce002b5bca5b865178de9985be10b9a0db2b805e) at
re-freeze #4, 2026-08-20. UNSTAMPED (C§8): nothing here scores a bar. The
objects below are the ones the TP=2 bridge certification ran on (jobs
`70b952964a3f`, `38ae1163fd61`; 152 cells, four rungs, zero errors), lifted with
exactly ONE non-prose edit: the two gate tolerances that were inline literals
(`1e-4`, `1e-6`) are now the module constants `ENTROPY_CROSS_RANK_TOL` and
`SEC25_CROSS_RANK_TOL`, of the same value, so the selftest can pin the NUMBER
the gate uses instead of a re-typed copy of it (rake M58). No other executable
line differs from the certified bytes.

WHAT THEY ARE FOR. The certified column runner drives three in-process objects:
the steering `wrapper` (`MetabasisSteeringLayer`), the entropy `cap`
(`EntropyCapture`), and a `state` dict that the forward shim reads. At TP>1 all
three live inside the worker processes, reached only by name through
`LLM.collective_rpc` against
`metabasis.extraction.vllm_worker_ext.MetabasisWorkerExtension`. These proxies
present the SAME attribute surface in the parent and forward every operation
over that RPC, so the ~700 lines of column orchestration — the refusals, the
span-accounting gate, the index pinning — stay byte-identical and keep their
meaning. Only the transport changes.

RANK POLICY, STATED ONCE. Every rank runs the identical replicated arithmetic
(the residual stream at a layer boundary is post-all-reduce — `o_proj`/
`down_proj` are RowParallelLinear with reduce_results=True, qwen2.py:162, :94 —
and `compute_logits` all-gathers full vocab, platforms/interface.py:563, with
padding stripped at logits_processor.py:102), so rank 0 is THE read. Where a
disagreement between ranks would be silent and consequential, these proxies
REFUSE rather than pick.

**CROSS-RANK AGREEMENT IS A STRUCTURAL GATE, NOT AN OBSERVATION.** It fires on
every scoring pass of every cell of every column, and it aborts the column:
  * `TPCapProxy.flat()` refuses if any rank's entropy vector differs from
    rank 0 by more than **1e-4**, or if the ranks captured different row counts
    at all;
  * `_one()` refuses on ANY disagreement in `wrapper.stats`
    (`tokens_seen` / `tokens_injected` / `prefill_tokens_skipped`), in the §2.5
    median (tolerance 1e-6, a hair of slack rather than exact float equality on
    a median), and in the shim's mask counters (`mask_true` / `mask_n`).
A violation aborts the column; it is never silently averaged. On the four
certification rungs the gate held across 152 cells, MoE path included
(`Qwen3MoeDecoderLayer`).

THE SURFACE IS NOT A FULL IMPERSONATION, AND THE DIFFERENCES ARE NAMED.
`PROXY_SURFACE_CONTRACT` below records, name by name, what the proxies share
with the in-process objects and the three places they deliberately substitute
(the shim moved into the worker; the §2.5 captures reduced on the rank because
shipping hundreds of MB of activations for one scalar is not a transport
detail; the wrapped layer reported by TYPE NAME because the module itself
cannot cross an RPC). The selftest asserts that mapping against the certified
classes' real AST, so a drift on either side is a refusal rather than a
surprise at TP>1.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: The gate's tolerances, named once so the selftest pins the NUMBER and not a
#: re-typed copy of it. Both are the values the TP=2 bridge certification ran.
ENTROPY_CROSS_RANK_TOL: float = 1e-4
SEC25_CROSS_RANK_TOL: float = 1e-6

#: The parent-side surface, name by name, against the in-process objects the
#: certified TP=1 runner drives. `shared` names must exist on BOTH the proxy and
#: the certified class; `substituted` names are the three places the transport
#: genuinely changes the object, each with the reason it could not be a
#: pass-through. A drift on either side is a refusal (see `selftest`).
PROXY_SURFACE_CONTRACT: dict[str, dict[str, object]] = {
    "MetabasisSteeringLayer": {
        "proxy": "TPWrapperProxy",
        "shared": ("set_spec", "capture_input_hidden_states", "reset_stats",
                   "stats", "hidden_dim"),
        "substituted": {
            "take_captures -> measure_sec25": (
                "the captures are (tokens x d_model) activations; shipping them "
                "to the parent would move hundreds of MB per column for a single "
                "scalar, so the certified reduction "
                "(measure_per_token_median_resid_norm_from_captures) runs ON the "
                "rank and only the median crosses — cross-checked between ranks "
                "at 1e-6"),
            "set_position_mask -> (worker-side shim)": (
                "the span mask is derived from `positions` inside the forward, "
                "which only exists in the worker; the shim moved there with it "
                "(MetabasisWorkerExtension.mb_attach) and the parent drives it "
                "through TPStateProxy instead"),
            "wrapped_layer -> wrapped_type_name": (
                "an nn.Module cannot cross a collective_rpc; the runner only "
                "ever reads `type(wrapper.wrapped_layer).__name__` for the "
                "column stamp, so the TYPE NAME is what the rank returns"),
        },
        "not_driven_from_the_parent": (
            # `forward` is vLLM's call path into the wrapper, invoked by the
            # model inside the worker and never by the runner; `layer_idx` is
            # the wrapper's own copy of the site, which the runner already holds
            # as `site` and never reads back off the object.
            "forward", "layer_idx",
        ),
    },
    "EntropyCapture": {
        "proxy": "TPCapProxy",
        "shared": ("reset", "flat"),
        "substituted": {
            "mode -> mb_entropy_mode / reset(mode)": (
                "the capture's mode is worker-side state; the parent sets it "
                "through the RPC rather than by attribute assignment"),
            "rows/counts/calls/rows_seen -> mb_take_entropy": (
                "the raw rows are drained on the rank; only scalars cross"),
        },
    },
}


class TPRankDisagreement(RuntimeError):
    """Ranks returned different values for a quantity that must be replicated."""


def _one(results: list[Any], what: str, *, tol: float = 0.0) -> Any:
    """Rank 0's value, refusing if a rank disagrees beyond `tol`."""
    if not results:
        raise TPRankDisagreement(f"{what}: no rank answered")
    r0 = results[0]
    for i, r in enumerate(results[1:], start=1):
        same = (abs(float(r) - float(r0)) <= tol
                if isinstance(r0, (int, float)) and not isinstance(r0, bool)
                else r == r0)
        if not same:
            raise TPRankDisagreement(
                f"{what}: rank {i} returned {r!r} but rank 0 returned {r0!r} -- a "
                "quantity this lane assumes is replicated across tensor-parallel "
                "ranks is not, and picking one would be a silent choice")
    return r0


class TPWrapperProxy:
    """Stands in for `MetabasisSteeringLayer` in the parent process."""

    def __init__(self, llm: Any, hidden_dim: int, wrapped_type_name: str) -> None:
        self._llm = llm
        self.hidden_dim = int(hidden_dim)
        self.wrapped_type_name = str(wrapped_type_name)

    def set_spec(self, spec: Optional[Any]) -> None:
        if spec is None:
            self._llm.collective_rpc("mb_set_spec", args=(None, 0.0, None, 0.0))
            return
        vec = np.asarray(spec.vector.detach().cpu().numpy()
                         if hasattr(spec.vector, "detach") else spec.vector,
                         dtype=np.float32).reshape(-1)
        self._llm.collective_rpc(
            "mb_set_spec",
            args=(vec.tolist(), float(spec.alpha),
                  None if spec.alpha_frac is None else float(spec.alpha_frac),
                  float(spec.measured_norm)))

    def capture_input_hidden_states(self, enabled: bool) -> None:
        self._llm.collective_rpc("mb_capture_inputs", args=(bool(enabled),))

    def reset_stats(self) -> None:
        self._llm.collective_rpc("mb_reset_stats")

    @property
    def stats(self) -> dict[str, int]:
        res = self._llm.collective_rpc("mb_stats_now")
        for k in ("tokens_seen", "tokens_injected", "prefill_tokens_skipped"):
            _one([r[k] for r in res], f"wrapper.stats[{k}]")
        return dict(res[0])

    def measure_sec25(self) -> float:
        """§2.5 median, reduced on each rank; refuses on cross-rank disagreement."""
        res = self._llm.collective_rpc("mb_measure_sec25")
        if any(r["n_captures"] == 0 for r in res):
            return float("nan")
        # bf16 activations are bitwise replicated across ranks, but allow a hair
        # of slack rather than asserting exact float equality on a median.
        return float(_one([r["median"] for r in res], "§2.5 median",
                          tol=SEC25_CROSS_RANK_TOL))

    def n_captures(self) -> int:
        res = self._llm.collective_rpc("mb_measure_sec25")
        return int(res[0]["n_captures"])


class TPCapProxy:
    """Stands in for `EntropyCapture` in the parent process."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def reset(self, mode: str = "off") -> None:
        self._llm.collective_rpc("mb_cap_reset", args=(str(mode),))

    def flat(self) -> np.ndarray:
        res = self._llm.collective_rpc("mb_cap_flat")
        lens = {len(r) for r in res}
        if len(lens) != 1:
            raise TPRankDisagreement(
                f"ranks captured different entropy row counts {lens} -- the "
                "all-gathered logits are supposed to be identical on every rank")
        a0 = np.asarray(res[0], dtype=np.float32)
        for i, r in enumerate(res[1:], start=1):
            ai = np.asarray(r, dtype=np.float32)
            if (a0.size and float(np.max(np.abs(ai - a0)))
                    > ENTROPY_CROSS_RANK_TOL):
                raise TPRankDisagreement(
                    f"rank {i} entropy differs from rank 0 by "
                    f"{float(np.max(np.abs(ai - a0)))} -- `compute_logits` did not "
                    "deliver the same full-vocab tensor to every rank")
        return a0


class TPStateProxy(dict):
    """The runner's `state` dict, with the shim-facing keys pushed to the workers."""

    _PUSH = ("mode", "prompt_len", "prompt_lengths")
    _PULL = ("mask_true", "mask_n")

    def __init__(self, llm: Any) -> None:
        super().__init__(mode="generation", prompt_len=None, prompt_lengths=None,
                         mask_true=0, mask_n=0)
        self._llm = llm
        self._push()

    def _push(self) -> None:
        pl = dict.__getitem__(self, "prompt_lengths")
        self._llm.collective_rpc(
            "mb_set_mode",
            args=(dict.__getitem__(self, "mode"),
                  dict.__getitem__(self, "prompt_len"),
                  None if pl is None else sorted(int(x) for x in pl)))

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        if key in self._PUSH:
            self._push()
        elif key in self._PULL and value == 0:
            self._llm.collective_rpc("mb_reset_mask")

    def __getitem__(self, key: str) -> Any:
        if key in self._PULL:
            res = self._llm.collective_rpc("mb_mask_counters")
            return int(_one([r[key] for r in res], f"state[{key}]"))
        return dict.__getitem__(self, key)


# ── selftest (desk-side, CPU, no vLLM, no torch, no weights) ─────────────────

class _FakeRankFleet:
    """A CPU stand-in for `LLM` whose `collective_rpc` answers as N ranks.

    THE HARNESS IS THE POINT. The cross-rank gate is the one new correctness
    claim the TP attachment makes, and it can only be exercised by a caller that
    can make the ranks DISAGREE on demand. `answers` maps an RPC method name to
    the per-rank return list; `calls` records everything the proxies sent, so a
    push that never left the parent is caught as easily as a wrong value.
    """

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> list[Any]:
        self.calls.append((method, tuple(args)))
        if method not in self.answers:
            raise AssertionError(
                f"the harness was not told what rank should answer for {method!r}")
        return self.answers[method]


def _certified_class_surface(path: "Path", class_name: str) -> set[str]:
    """The PUBLIC surface of a certified class, read from its AST.

    AST, NOT IMPORT, AND NOT A STRING NEEDLE (rakes M58 + M44). Both certified
    modules import torch at module scope, so the desk .venv cannot import them
    at all — and a surface test that only runs where torch happens to exist is
    a proof that disarms itself in the configuration where the lift is reviewed.
    Parsing the source gives the same answer everywhere: public method names,
    plus dataclass fields, plus `self.X = …` assignments in `__init__`.
    """
    import ast as _ast
    if not path.exists():
        raise FileNotFoundError(
            f"the certified module {path} is not beside this one; the proxy "
            "surface cannot be proven against a class that is not there")
    tree = _ast.parse(path.read_text())
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == class_name:
            break
    else:
        raise LookupError(f"{class_name} is not defined in {path}")
    names: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if not stmt.name.startswith("_"):
                names.add(stmt.name)
            if stmt.name == "__init__":
                # BOTH assignment forms. `self.stats: dict[…] = {…}` is an
                # AnnAssign, not an Assign, and a walker that only knows Assign
                # silently drops exactly the annotated (i.e. most carefully
                # written) attributes — which is the opposite of what a surface
                # contract wants to miss.
                for sub in _ast.walk(stmt):
                    tgt = None
                    if isinstance(sub, _ast.Assign) and len(sub.targets) == 1:
                        tgt = sub.targets[0]
                    elif isinstance(sub, _ast.AnnAssign):
                        tgt = sub.target
                    if (isinstance(tgt, _ast.Attribute)
                            and isinstance(tgt.value, _ast.Name)
                            and tgt.value.id == "self"
                            and not tgt.attr.startswith("_")):
                        names.add(tgt.attr)
        elif isinstance(stmt, _ast.AnnAssign) and isinstance(stmt.target, _ast.Name):
            if not stmt.target.id.startswith("_"):
                names.add(stmt.target.id)
    return names


def selftest() -> int:                                            # noqa: C901
    """The gate semantics and the surface contract, both provable on CPU."""
    import sys
    from pathlib import Path

    # RAKE M59(2): when this module is exercised from a deployment whose
    # sys.path[0] is not the code root, `import metabasis…` below would skip
    # rather than run. The root is resolved off THIS file and proven to exist.
    _root = Path(__file__).resolve().parents[2]
    if (_root / "metabasis" / "__init__.py").exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    checks: list[tuple[str, bool, str]] = []
    skips: list[str] = []

    def ok(cond: bool, name: str, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    def skip(name: str, why: str) -> None:
        skips.append(name)
        checks.append((f"SKIPPED: {name}", True, why))

    def refuses(fn: Any, what: str) -> None:
        try:
            fn()
            ok(False, f"REFUSES: {what}", "no exception raised")
        except TPRankDisagreement as exc:
            ok(True, f"REFUSES: {what}", str(exc)[:120])
        except Exception as exc:                                   # noqa: BLE001
            ok(False, f"REFUSES: {what}",
               f"wrong class {type(exc).__name__}: {exc}")

    # -- 1. `_one`: rank 0 is the read, and a disagreement is a refusal -------
    ok(_one([3, 3, 3], "x") == 3, "_one returns rank 0's value when ranks agree")
    ok(_one(["a", "a"], "x") == "a", "_one compares non-numerics by equality")
    refuses(lambda: _one([3, 4], "tokens_seen"), "a bare integer disagreement")
    refuses(lambda: _one([], "anything"), "no rank answered at all")
    ok(_one([1.0, 1.0 + 5e-7], "median", tol=1e-6) == 1.0,
       "_one honours a tolerance rather than demanding float equality")
    refuses(lambda: _one([1.0, 1.0 + 5e-6], "median", tol=1e-6),
            "a numeric disagreement OUTSIDE the tolerance")
    try:
        _one([3, 4], "tokens_seen")
        ok(False, "the refusal message names the quantity and both values")
    except TPRankDisagreement as exc:
        msg = str(exc)
        ok("tokens_seen" in msg and "rank 1" in msg and "4" in msg and "3" in msg,
           "the refusal message names the quantity, the rank, and both values",
           msg[:140])
    ok(issubclass(TPRankDisagreement, RuntimeError),
       "TPRankDisagreement is a RuntimeError — the lane's refusals are one family")

    # -- 2. TPCapProxy.flat(): the 1e-4 entropy gate --------------------------
    ok(ENTROPY_CROSS_RANK_TOL == 1e-4,
       "the cross-rank entropy tolerance is pinned at 1e-4 — the value the TP=2 "
       "bridge certification ran", f"{ENTROPY_CROSS_RANK_TOL}")
    ok(SEC25_CROSS_RANK_TOL == 1e-6,
       "the cross-rank §2.5 median tolerance is pinned at 1e-6",
       f"{SEC25_CROSS_RANK_TOL}")

    agree = TPCapProxy(_FakeRankFleet({"mb_cap_flat": [[1.0, 2.0], [1.0, 2.0]]}))
    got = agree.flat()
    ok(list(got) == [1.0, 2.0] and got.dtype == np.float32,
       "agreeing ranks give rank 0's vector, float32", f"{got!r}")

    inside = TPCapProxy(_FakeRankFleet(
        {"mb_cap_flat": [[1.0, 2.0], [1.0, 2.0 + 9e-5]]}))
    ok(list(inside.flat()) == [1.0, 2.0],
       "a divergence INSIDE 1e-4 passes and rank 0 is still the read — bitwise "
       "identity is not required (§2.3's statistical vocabulary)")

    outside = TPCapProxy(_FakeRankFleet(
        {"mb_cap_flat": [[1.0, 2.0], [1.0, 2.0 + 1.1e-4]]}))
    refuses(outside.flat,
            "an entropy divergence just OUTSIDE 1e-4 — the gate boundary, "
            "exercised from both sides")

    ragged = TPCapProxy(_FakeRankFleet(
        {"mb_cap_flat": [[1.0, 2.0], [1.0]]}))
    refuses(ragged.flat,
            "ranks that captured DIFFERENT ROW COUNTS (compute_logits is "
            "supposed to all-gather the identical full-vocab tensor)")

    empty = TPCapProxy(_FakeRankFleet({"mb_cap_flat": [[], []]}))
    ok(empty.flat().size == 0,
       "an empty pass is empty on every rank and is not an error")

    fleet = _FakeRankFleet({"mb_cap_reset": [None, None]})
    TPCapProxy(fleet).reset("scoring")
    ok(fleet.calls == [("mb_cap_reset", ("scoring",))],
       "TPCapProxy.reset pushes the mode to EVERY rank", f"{fleet.calls}")

    # -- 3. TPWrapperProxy: stats, §2.5, and the pushes -----------------------
    good_stats = [{"tokens_seen": 10, "tokens_injected": 4,
                   "prefill_tokens_skipped": 6}] * 2
    wp = TPWrapperProxy(_FakeRankFleet({"mb_stats_now": good_stats}), 2048, "Qwen2DecoderLayer")
    ok(wp.stats == good_stats[0], "agreeing ranks give rank 0's stats dict")
    ok(wp.hidden_dim == 2048 and wp.wrapped_type_name == "Qwen2DecoderLayer",
       "the proxy carries the width and the wrapped TYPE NAME the stamp needs")
    for key in ("tokens_seen", "tokens_injected", "prefill_tokens_skipped"):
        bad = [dict(good_stats[0]), dict(good_stats[0])]
        bad[1][key] = bad[1][key] + 1
        bad_proxy = TPWrapperProxy(_FakeRankFleet({"mb_stats_now": bad}), 8, "L")
        refuses(lambda p=bad_proxy: p.stats,
                f"a cross-rank disagreement in wrapper.stats[{key}]")

    sec_ok = [{"rank": 0, "n_captures": 3, "median": 80.5476},
              {"rank": 1, "n_captures": 3, "median": 80.5476}]
    ok(TPWrapperProxy(_FakeRankFleet({"mb_measure_sec25": sec_ok}), 8, "L")
       .measure_sec25() == 80.5476,
       "the §2.5 median is reduced ON the rank and only the scalar crosses")
    sec_none = [{"rank": 0, "n_captures": 0, "median": None},
                {"rank": 1, "n_captures": 3, "median": 1.0}]
    val = TPWrapperProxy(_FakeRankFleet({"mb_measure_sec25": sec_none}), 8, "L").measure_sec25()
    ok(val != val,
       "a rank that captured NOTHING yields NaN, which the runner turns into "
       "the certified 'the wrapper captured nothing at the site' refusal",
       f"{val}")
    sec_bad = [{"rank": 0, "n_captures": 3, "median": 80.5476},
               {"rank": 1, "n_captures": 3, "median": 80.5476 + 1e-4}]
    refuses(TPWrapperProxy(_FakeRankFleet({"mb_measure_sec25": sec_bad}), 8, "L").measure_sec25,
            "a cross-rank disagreement in the §2.5 median beyond 1e-6")

    fleet = _FakeRankFleet({"mb_set_spec": [None, None]})
    TPWrapperProxy(fleet, 4, "L").set_spec(None)
    ok(fleet.calls == [("mb_set_spec", (None, 0.0, None, 0.0))],
       "clearing the dose pushes an explicit (None, 0.0, None, 0.0) to every "
       "rank — never a skipped call that would leave a stale dose installed",
       f"{fleet.calls}")

    class _Spec:
        vector = np.asarray([1.0, -2.0, 3.0, 4.0], dtype=np.float64)
        alpha = 0.5
        alpha_frac = 0.3
        measured_norm = 80.5

    fleet = _FakeRankFleet({"mb_set_spec": [None, None]})
    TPWrapperProxy(fleet, 4, "L").set_spec(_Spec())
    sent = fleet.calls[0][1]
    ok(sent[0] == [1.0, -2.0, 3.0, 4.0] and sent[1] == 0.5
       and sent[2] == 0.3 and sent[3] == 80.5,
       "a real dose crosses as a plain float32 LIST plus three floats — "
       "collective_rpc marshals arguments, so nothing tensor-shaped is assumed",
       f"{sent}")
    ok(all(isinstance(x, float) for x in sent[0]),
       "…and every element is a plain float, not a numpy scalar")

    fleet = _FakeRankFleet({"mb_capture_inputs": [None, None],
                            "mb_reset_stats": [None, None]})
    p = TPWrapperProxy(fleet, 4, "L")
    p.capture_input_hidden_states(True)
    p.reset_stats()
    ok(fleet.calls == [("mb_capture_inputs", (True,)), ("mb_reset_stats", ())],
       "capture toggling and stat resets both reach every rank", f"{fleet.calls}")

    # -- 4. TPStateProxy: what is PUSHED and what is PULLED -------------------
    fleet = _FakeRankFleet({"mb_set_mode": [None, None],
                            "mb_reset_mask": [None, None],
                            "mb_mask_counters": [{"mask_true": 5, "mask_n": 9},
                                                 {"mask_true": 5, "mask_n": 9}]})
    st = TPStateProxy(fleet)
    ok(fleet.calls[0] == ("mb_set_mode", ("generation", None, None)),
       "constructing the state pushes the initial mode immediately — a rank "
       "left in a stale mode would mask the wrong tokens", f"{fleet.calls[0]}")
    st["mode"] = "scoring"
    ok(fleet.calls[-1] == ("mb_set_mode", ("scoring", None, None)),
       "setting `mode` pushes")
    st["prompt_len"] = 12
    ok(fleet.calls[-1] == ("mb_set_mode", ("scoring", 12, None)),
       "setting `prompt_len` pushes the whole triple, not a partial update")
    st["prompt_lengths"] = frozenset({7, 3, 5})
    ok(fleet.calls[-1] == ("mb_set_mode", ("scoring", 12, [3, 5, 7])),
       "a frozenset is pushed as a SORTED list of ints — collective_rpc cannot "
       "marshal a frozenset and an unordered list would make the RPC payload "
       "non-deterministic", f"{fleet.calls[-1]}")
    ok(st["mask_true"] == 5 and st["mask_n"] == 9,
       "reading a mask counter PULLS it from the ranks rather than trusting a "
       "parent-side copy that no forward ever incremented")
    n_before = len(fleet.calls)
    st["mask_true"] = 0
    ok(fleet.calls[-1] == ("mb_reset_mask", ()) and len(fleet.calls) == n_before + 1,
       "zeroing a mask counter is the runner's per-cell reset and reaches the ranks",
       f"{fleet.calls[-1]}")
    ok(st["mode"] == "scoring",
       "a non-counter key still reads locally (the dict half of the proxy)")
    disagree = _FakeRankFleet({"mb_set_mode": [None, None],
                               "mb_mask_counters": [{"mask_true": 5, "mask_n": 9},
                                                    {"mask_true": 6, "mask_n": 9}]})
    st2 = TPStateProxy(disagree)
    refuses(lambda: st2["mask_true"],
            "ranks that disagree on the shim's own mask counters")
    ok(isinstance(st2, dict),
       "TPStateProxy IS a dict, so the certified runner's `state[...]` lines "
       "need no edit at all")

    # -- 5. the surface contract, against the certified classes' real AST -----
    here = Path(__file__).resolve().parent
    sources = {"MetabasisSteeringLayer": here / "vllm_residual_write.py",
               "EntropyCapture": here / "vllm_entropy_probe.py"}
    for cls_name, entry in sorted(PROXY_SURFACE_CONTRACT.items()):
        path = sources[cls_name]
        try:
            certified = _certified_class_surface(path, cls_name)
        except (FileNotFoundError, LookupError) as exc:
            ok(False, f"the certified surface of {cls_name} is readable", str(exc))
            continue
        # The proxy's surface is read the SAME way as the certified class's —
        # from the AST — because both carry public state set in `__init__`
        # (`hidden_dim`, `wrapped_type_name`) that `dir()` on the class cannot
        # see. Two surfaces compared by two different rules is not a comparison.
        proxy_name = str(entry["proxy"])
        proxy_names = _certified_class_surface(Path(__file__).resolve(), proxy_name)
        shared = set(entry["shared"])                     # type: ignore[arg-type]
        substituted: set[str] = set()
        for lhs in entry["substituted"]:                  # type: ignore[union-attr]
            substituted.update(p.strip()
                               for p in str(lhs).split("->")[0].strip().split("/"))
        unused = set(entry.get("not_driven_from_the_parent", ()))  # type: ignore[arg-type]

        ok(shared <= certified,
           f"every SHARED name of {cls_name} really exists on the certified "
           "class — the contract is checked against the class, not against a "
           "memory of it",
           f"missing from {cls_name}: {sorted(shared - certified)}")
        ok(shared <= proxy_names,
           f"every SHARED name of {cls_name} exists on {proxy_name} — this "
           "is the attribute-by-attribute surface equality the TP path needs",
           f"missing from {proxy_name}: {sorted(shared - proxy_names)}")
        ok(substituted <= certified,
           f"every SUBSTITUTED name of {cls_name} is a name the certified class "
           "actually has (a substitution for a name that does not exist would "
           "be documenting a fiction)",
           f"not on {cls_name}: {sorted(substituted - certified)}")
        unaccounted = certified - shared - substituted - unused
        ok(not unaccounted,
           f"NO public name of {cls_name} is unaccounted for — a method added "
           "to the certified class later shows up HERE as a refusal instead of "
           "silently having no TP counterpart",
           f"unaccounted: {sorted(unaccounted)}")
        ok(not (shared & substituted),
           f"{cls_name}: no name is both shared and substituted")
        for lhs, why in sorted(entry["substituted"].items()):   # type: ignore[union-attr]
            ok(isinstance(why, str) and len(why) > 40,
               f"the substitution {lhs} carries a REASON, not just a mapping")

    # -- 6. the proxies reach only names the worker extension exposes ---------
    try:
        from metabasis.extraction.vllm_worker_ext import RPC_SURFACE
    except ImportError as exc:
        skip("every collective_rpc method name the proxies send exists on the "
             "worker extension",
             f"vllm_worker_ext is not importable here ({exc}); evidence "
             "MISSING: that no proxy call is stranded at TP>1")
    else:
        import ast as _ast
        sent_names: set[str] = set()
        for node in _ast.walk(_ast.parse(Path(__file__).read_text())):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "collective_rpc"
                    and node.args
                    and isinstance(node.args[0], _ast.Constant)
                    and isinstance(node.args[0].value, str)):
                sent_names.add(node.args[0].value)
        ok(bool(sent_names) and sent_names <= RPC_SURFACE,
           "every `collective_rpc` method name these proxies send is on the "
           "worker extension's pinned RPC_SURFACE — a rename on either side is "
           "caught here instead of on a loaded 405B model",
           f"sent={sorted(sent_names)} stranded={sorted(sent_names - RPC_SURFACE)}")

    # -- 7. M59: the suite's own arithmetic -----------------------------------
    SELFTEST_CHECK_FLOOR = 53
    KNOWN_SKIP_CEILING = 1
    ok(len(checks) >= SELFTEST_CHECK_FLOOR and len(skips) <= KNOWN_SKIP_CEILING,
       f"(M59) the suite ran at least its recorded floor of "
       f"{SELFTEST_CHECK_FLOOR} checks and named no more than "
       f"{KNOWN_SKIP_CEILING} skip(s) — a block that stopped running, or "
       "started skipping, is caught here rather than read as a clean run",
       f"{len(checks)} checks (floor {SELFTEST_CHECK_FLOOR}), "
       f"{len(skips)} named skip(s) (ceiling {KNOWN_SKIP_CEILING})")

    fails = [c for c in checks if not c[1]]
    print(f"vllm_tp_proxy selftest checks run: {len(checks)} "
          f"({len(skips)} named skip(s)), {len(fails)} failure(s)")
    for name, _, detail in fails:
        print(f"  MISS {name} — {detail}")
    for name in skips:
        print(f"  SKIPPED {name}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
