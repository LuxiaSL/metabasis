"""The EFFECTIVE thread configuration, read at runtime, as instrument identity.

THE RULING (Luxia, 2026-08-01, on the opt-pass item-3 measurement):
`OMP_NUM_THREADS=8` becomes the standing default everywhere, and the thread
count is recorded in every build stamp as part of instrument identity. Basis:
eigh is bitwise-deterministic at any FIXED thread count (repeat-identical, both
rungs measured) and bytes differ across counts; Σ construction is bitwise
stable regardless. The byte-comparability break with the banked single-threaded
v2.1 vectors is characterized and taken knowingly: banked artifacts stay as
they are; v3 rebuilds all vectors =8-stamped from birth; any comparison against
a banked artifact quotes the count mismatch.

────────────────────────────────────────────────────────────────────────────────
WHY THE ENV VARIABLE IS NOT THE ANSWER
────────────────────────────────────────────────────────────────────────────────
`os.environ["OMP_NUM_THREADS"]` is what was REQUESTED. What decides the
reduction order — and therefore the bytes — is what the BLAS runtime actually
came up with, and the two can disagree in ways that are invisible from the
env alone:

  * OpenBLAS reads its thread count when the shared object is LOADED. A script
    that sets `os.environ["OMP_NUM_THREADS"]` AFTER `import numpy` changes the
    string and nothing else. Every builder in this package uses
    `os.environ.setdefault(...)` above its numpy import precisely so the two
    agree — but a stamp that recorded the request would be unable to say so.
  * `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` override `OMP_NUM_THREADS` for
    their own runtime, so the effective count can differ per loaded library.
  * A cgroup/affinity mask can hold the runtime below the requested number.

So this module READS THE THREADPOOL: it enumerates the BLAS/OpenMP shared
objects already mapped into the process and calls each one's own
`get_num_threads` entry point through `ctypes`. The requested env is recorded
BESIDE that, never instead of it, so a stamp shows both the intent and the
instrument.

────────────────────────────────────────────────────────────────────────────────
RAKE M19 — INSTRUMENTATION THAT ONLY DESCRIBES A RUN MUST NEVER FAIL IT
────────────────────────────────────────────────────────────────────────────────
A thread count cannot change one banked number. Every way of failing to read
one therefore degrades to a NAMED sentinel rather than raising:
`effective_num_threads=None` with `consensus="unresolved"` and a `note` saying
what went wrong. `THREADS_UNRESOLVED` is deliberately distinguishable from "the
stamp predates the field" — see `stamp_thread_config`, which returns `None` for
a PRE-RULING stamp and a `ThreadConfig` with `consensus="unresolved"` for a
degraded probe (the M41 hostname discipline, one field over).

Run (repo root, PYTHONPATH=.):
  python -m metabasis.threads --selftest
  python -m metabasis.threads --show [--json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

#: THE RULED DEFAULT (Luxia 2026-08-01). Job templates export it; builders
#: `setdefault` it; stamps record what was actually in effect and compare.
RULED_OMP_NUM_THREADS: int = 8

#: The stamp key every builder writes. One name, so a reader that wants the
#: thread count of an artifact never has to know which builder made it.
THREAD_STAMP_KEY: str = "thread_config"

#: The probe ran and could not name a count. NOT the same as "this stamp
#: predates the field" (that is `stamp_thread_config(...) is None`) and not the
#: same as "no BLAS runtime is loaded" (that is an empty `pools` with
#: `consensus="unresolved"` and a note saying so).
THREADS_UNRESOLVED: str = "THREADS_UNRESOLVED"

#: What a PRE-RULING artifact's thread count reads as in a comparison. Every
#: vector banked before 2026-08-01 was built single-threaded by the deployed
#: job scripts' `export OMP_NUM_THREADS=1`, but the stamp does not SAY so, and
#: an inference is not a record (rake M41(a), from the other side).
PRE_RULING_UNRECORDED: str = "unrecorded (stamp predates the 2026-08-01 ruling)"

#: The label a comparison prints when two thread counts differ. Constant, so a
#: log sweep can grep for it and a reader never has to recognise prose.
THREAD_COUNT_MISMATCH_LABEL: str = "THREAD-COUNT MISMATCH"

#: The env vars that decide a BLAS/OpenMP runtime's thread count, in the order
#: a reader should think about them. Recorded as REQUESTED values only.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)

#: `(basename substring, kind)` — how a mapped shared object is classified.
#: Substrings rather than exact names because the wheels vendor hashed copies
#: (`libscipy_openblas64_-017048f4.so`) and the hash changes per build.
_LIBRARY_KINDS: tuple[tuple[str, str], ...] = (
    ("openblas", "openblas"),
    ("mkl_rt", "mkl"),
    ("libmkl_core", "mkl"),
    ("blis", "blis"),
    ("libgomp", "openmp"),
    ("libomp", "openmp"),
    ("libiomp", "openmp"),
)

#: Candidate `get_num_threads` entry points, per kind. Several per kind because
#: the 64-bit-integer builds suffix their symbols and the scipy/numpy wheels
#: additionally PREFIX them (`scipy_openblas_get_num_threads64_`) so two
#: vendored OpenBLASes can coexist in one process without colliding.
_GET_SYMBOLS: dict[str, tuple[str, ...]] = {
    "openblas": ("openblas_get_num_threads", "openblas_get_num_threads64_",
                 "scipy_openblas_get_num_threads64_",
                 "scipy_openblas_get_num_threads_64_"),
    "mkl": ("MKL_Get_Max_Threads", "mkl_get_max_threads"),
    "blis": ("bli_thread_get_num_threads",),
    "openmp": ("omp_get_max_threads",),
}


class ThreadProbeError(RuntimeError):
    """A thread probe could not be performed.

    Raised only INSIDE the probe and always caught by it: nothing in this
    module lets a description of a run fail that run (rake M19). It exists so
    the degraded paths carry a typed reason rather than a bare string.
    """


class ThreadPool(BaseModel):
    """One BLAS/OpenMP runtime mapped into this process, and its thread count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    library: str = Field(
        description="the shared object's BASENAME, never its path — a stamp "
                    "travels into the repo and into reports, and an absolute "
                    "path carries a username and a store layout with it")
    kind: Literal["openblas", "mkl", "blis", "openmp", "unknown"]
    symbol: Optional[str] = Field(
        default=None,
        description="the entry point that answered; None when none did")
    num_threads: Optional[int] = Field(
        default=None,
        description="what the runtime says it will use. None = the library is "
                    "loaded but would not answer — a NAMED degradation, never "
                    "a silent 1")
    note: str = ""


class ThreadConfig(BaseModel):
    """The effective thread configuration of THIS process, as one document.

    `effective_num_threads` is the number that decides reduction order, and it
    is populated ONLY when every BLAS pool that answered agrees. Disagreement
    is reported as disagreement (`consensus="disagreed"`, the per-pool counts
    intact) rather than resolved by a rule nobody ruled: two BLAS runtimes at
    different thread counts is a fact about the environment that a stamp must
    surface, not average away.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ruled_default: int = Field(
        default=RULED_OMP_NUM_THREADS,
        description="the standing default of record (Luxia 2026-08-01), "
                    "written into every stamp so an artifact says what the "
                    "convention WAS when it was built")
    effective_num_threads: Optional[int] = Field(
        default=None,
        description="the agreed BLAS thread count; None when the pools "
                    "disagree or none answered")
    consensus: Literal["agreed", "disagreed", "unresolved"] = "unresolved"
    matches_ruled_default: Optional[bool] = Field(
        default=None,
        description="None when the count is unresolved — 'we could not tell' "
                    "and 'it does not match' are different facts")
    env_requested: dict[str, str] = Field(
        default={},
        description="what was ASKED for, '(unset)' where absent. Recorded "
                    "beside the measurement, never instead of it")
    pools: list[ThreadPool] = Field(
        default=[], description="one row per mapped BLAS/OpenMP runtime")
    cpu_count: Optional[int] = None
    affinity_count: Optional[int] = Field(
        default=None,
        description="len(os.sched_getaffinity(0)) — the ceiling a cgroup or a "
                    "taskset actually leaves, which can sit below cpu_count "
                    "and below the requested thread count")
    probe: str = Field(
        default="",
        description="HOW the count was read, so a reader can judge it")
    note: str = Field(
        default="",
        description="why the probe degraded, when it did. Empty on a clean "
                    "read; never empty when consensus is 'unresolved'")

    @property
    def quoted(self) -> str:
        """The count as a comparison should print it — never a bare integer."""
        if self.effective_num_threads is None:
            return f"{THREADS_UNRESOLVED} ({self.consensus})"
        return str(self.effective_num_threads)


# ---------------------------------------------------------------- the probe
def _mapped_libraries() -> list[str]:
    """Absolute paths of the shared objects mapped into this process.

    Linux-only by construction (`/proc/self/maps`). Anywhere else this returns
    an empty list and the caller degrades to a NAMED unresolved config rather
    than guessing — the campaign runs on Linux nodes and a wrong answer would
    be worse than an honest absence.
    """
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise ThreadProbeError(
            f"/proc/self/maps unreadable ({exc}) — the loaded-library "
            f"enumeration is Linux-only") from exc
    seen: list[str] = []
    for line in lines:
        parts = line.rstrip("\n").split(" ", 5)
        if len(parts) != 6:
            continue
        path = parts[5].strip()
        if path.startswith("/") and path not in seen:
            seen.append(path)
    return seen


def _classify(basename: str) -> Optional[str]:
    low = basename.lower()
    for needle, kind in _LIBRARY_KINDS:
        if needle in low:
            return kind
    return None


def _probe_library(path: str, kind: str) -> ThreadPool:
    """Ask ONE loaded runtime for its thread count. Never raises."""
    basename = os.path.basename(path)
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:                                    # pragma: no cover
        return ThreadPool(library=basename, kind=kind,  # type: ignore[arg-type]
                          note=f"cannot dlopen ({exc})")
    for symbol in _GET_SYMBOLS.get(kind, ()):
        try:
            fn = getattr(lib, symbol)
        except AttributeError:
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = []
        try:
            value = int(fn())
        except Exception as exc:                              # noqa: BLE001 — M19
            return ThreadPool(library=basename, kind=kind,  # type: ignore[arg-type]
                              symbol=symbol,
                              note=f"{symbol} raised {type(exc).__name__}: {exc}")
        if value <= 0:
            return ThreadPool(library=basename, kind=kind,  # type: ignore[arg-type]
                              symbol=symbol,
                              note=f"{symbol} returned {value}, which is not a "
                                   f"thread count")
        return ThreadPool(library=basename, kind=kind,  # type: ignore[arg-type]
                          symbol=symbol, num_threads=value)
    return ThreadPool(library=basename, kind=kind,  # type: ignore[arg-type]
                      note="loaded, but exports none of the known "
                           f"get_num_threads entry points "
                           f"({', '.join(_GET_SYMBOLS.get(kind, ())) or 'none known'})")


def _requested_env() -> dict[str, str]:
    return {name: os.environ.get(name, "(unset)") for name in THREAD_ENV_VARS}


def _affinity_count() -> Optional[int]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:                                        # pragma: no cover
        return None
    try:
        return len(getter(0))
    except OSError:                                           # pragma: no cover
        return None


def effective_thread_config() -> ThreadConfig:
    """Read the EFFECTIVE thread configuration. Never raises (rake M19).

    The BLAS pools decide reduction order, so `effective_num_threads` is agreed
    over BLAS runtimes only; an OpenMP runtime is reported for the reader but
    does not vote, because torch's intra-op pool and numpy's BLAS pool are
    routinely set to different sizes on purpose and letting the OpenMP row veto
    the consensus would report a disagreement that is not one.
    """
    env = _requested_env()
    cpu = os.cpu_count()
    affinity = _affinity_count()
    base = dict(ruled_default=RULED_OMP_NUM_THREADS, env_requested=env,
                cpu_count=cpu, affinity_count=affinity)
    try:
        paths = _mapped_libraries()
    except ThreadProbeError as exc:
        return ThreadConfig(
            **base, pools=[], consensus="unresolved",
            probe="ctypes over the mapped BLAS/OpenMP runtimes (unavailable)",
            note=f"{THREADS_UNRESOLVED}: {exc}")
    except Exception as exc:                                  # noqa: BLE001 — M19
        return ThreadConfig(
            **base, pools=[], consensus="unresolved",
            probe="ctypes over the mapped BLAS/OpenMP runtimes (failed)",
            note=f"{THREADS_UNRESOLVED}: unexpected {type(exc).__name__}: {exc}")

    pools: list[ThreadPool] = []
    for path in paths:
        kind = _classify(os.path.basename(path))
        if kind is None:
            continue
        pools.append(_probe_library(path, kind))

    probe = ("ctypes over the BLAS/OpenMP runtimes already mapped into this "
             "process (/proc/self/maps + dlopen + each runtime's own "
             "get_num_threads); the env is recorded beside it as the REQUEST")
    voting = [p.num_threads for p in pools
              if p.kind in ("openblas", "mkl", "blis") and p.num_threads is not None]
    if not voting:
        why = ("no BLAS runtime is mapped into this process yet — import numpy "
               "before probing" if not pools else
               "every mapped BLAS runtime declined to answer: "
               + "; ".join(f"{p.library}: {p.note}" for p in pools if p.note))
        return ThreadConfig(**base, pools=pools, consensus="unresolved",
                            probe=probe, note=f"{THREADS_UNRESOLVED}: {why}")
    if len(set(voting)) != 1:
        return ThreadConfig(
            **base, pools=pools, consensus="disagreed", probe=probe,
            note="the mapped BLAS runtimes report DIFFERENT thread counts "
                 + ", ".join(f"{p.library}={p.num_threads}" for p in pools
                             if p.num_threads is not None)
                 + " — reported as a disagreement rather than resolved by a "
                   "rule nobody ruled")
    count = voting[0]
    return ThreadConfig(**base, pools=pools, consensus="agreed",
                        effective_num_threads=count,
                        matches_ruled_default=(count == RULED_OMP_NUM_THREADS),
                        probe=probe)


#: The one-line STATUS every stamped thread block opens with, so a reader who
#: has never seen the field knows in one line what it is and why it is there.
THREAD_STAMP_STATUS: str = (
    "INSTRUMENT IDENTITY (Luxia ruling 2026-08-01): eigh is bitwise-"
    "deterministic at a FIXED thread count and its bytes differ ACROSS counts, "
    "so the effective thread configuration is part of what produced these "
    "numbers. Read from the threadpool at stamp time, not from the request. A "
    "rebuild compared against an artifact banked at another count is a "
    "characterized break, never a corruption.")


def thread_config_stamp(config: Optional[ThreadConfig] = None) -> dict:
    """The `thread_config` block a builder writes into its stamp/sidecar.

    Plain JSON-able types only, `STATUS` first, so the block reads correctly in
    a stamp opened by a human with no access to this module.
    """
    cfg = config if config is not None else effective_thread_config()
    return {"STATUS": THREAD_STAMP_STATUS, **cfg.model_dump(mode="json")}


def stamp_thread_config(stamp: Optional[dict]) -> Optional[ThreadConfig]:
    """The thread configuration a stamp records — `None` for a PRE-RULING stamp.

    THE BACKWARD-COMPATIBLE READER, and the only one any consumer should use.
    Three states a reader must keep apart (the M41 hostname discipline):

        None                            the stamp PREDATES the field. The
                                        artifact's thread count is UNRECORDED
                                        and must never be inferred, even though
                                        every deployed job script of that
                                        vintage exported OMP_NUM_THREADS=1
        consensus="unresolved"          the builder RAN and could not name a
                                        count — degraded instrumentation
        effective_num_threads=<n>       the count the artifact was built at

    Accepts a whole build stamp, a nested `trunk`, or the block itself, because
    all three get handed around and a reader that only ever sees one of them
    would silently return None for the other two.
    """
    if not isinstance(stamp, dict):
        return None
    block: object = stamp.get(THREAD_STAMP_KEY)
    if block is None and isinstance(stamp.get("trunk"), dict):
        block = stamp["trunk"].get(THREAD_STAMP_KEY)
    if block is None and "consensus" in stamp and "ruled_default" in stamp:
        block = stamp                       # the block itself was handed in
    if not isinstance(block, dict):
        return None
    payload = {k: v for k, v in block.items() if k != "STATUS"}
    try:
        return ThreadConfig(**payload)
    except Exception:                                         # noqa: BLE001 — M19
        #  A block this module cannot parse is still EVIDENCE that the field is
        #  present, so it must not read as "pre-ruling". It reads as a probe
        #  that produced something unusable, which is what it is.
        return ThreadConfig(
            consensus="unresolved",
            note=f"{THREADS_UNRESOLVED}: the stamp carries a {THREAD_STAMP_KEY} "
                 f"block this reader cannot parse (keys "
                 f"{sorted(str(k) for k in block)}) — recorded as an "
                 f"unresolved probe, NOT as a pre-ruling absence")


def quote_thread_count(config: Optional[ThreadConfig]) -> str:
    """How a thread count is printed in a comparison. Never a bare integer."""
    return PRE_RULING_UNRECORDED if config is None else config.quoted


def thread_count_mismatch(banked: Optional[ThreadConfig],
                          rebuilt: Optional[ThreadConfig],
                          *, banked_label: str = "banked",
                          rebuilt_label: str = "rebuild") -> Optional[str]:
    """A LABELED count mismatch, or None when both sides are known and equal.

    Scope 3 of the ruling, in code: wherever a rebuild is compared against a
    banked pre-ruling artifact, the comparison quotes BOTH thread counts when
    they differ. A byte difference that a count mismatch explains is a
    characterized break; the same difference reported bare reads as corruption,
    and the two demand opposite responses.

    Returns None ONLY when both sides resolved to the same count — an
    unrecorded side is a mismatch to be quoted, not an agreement, because the
    whole point is that a pre-ruling artifact cannot vouch for its own count.
    """
    a = banked.effective_num_threads if banked is not None else None
    b = rebuilt.effective_num_threads if rebuilt is not None else None
    if a is not None and b is not None and a == b:
        return None
    return (f"{THREAD_COUNT_MISMATCH_LABEL}: {banked_label} "
            f"{quote_thread_count(banked)} vs {rebuilt_label} "
            f"{quote_thread_count(rebuilt)}. eigh is bitwise-deterministic at a "
            f"FIXED thread count and its bytes differ across counts (Luxia "
            f"ruling 2026-08-01), so a byte difference between these two is "
            f"EXPECTED and characterized — adjudicate the count before reading "
            f"it as corruption.")


# ---------------------------------------------------------------- selftest
def selftest() -> int:                                        # noqa: C901 — a checklist
    """CPU-only, numpy-only, no torch: the probe, the stamp, the reader, the label.

    Named configurations (rake M44): every count below says which environment
    it was measured in. The one axis that matters here is whether a BLAS
    runtime is mapped into the process at all — this module deliberately does
    NOT import numpy at module scope (a stamp helper that dragged in numpy
    would be unusable from a pure-config context), so the selftest exercises
    BOTH the un-mapped and the mapped case, in that order, in one process.
    """
    failures: list[str] = []
    skips: list[str] = []
    checks = 0

    def check(cond: bool, msg: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            failures.append(msg)

    def skip(msg: str) -> None:
        skips.append(msg)
        print(f"  [SKIP] {msg}")

    print("== selftest 1: the probe describes, and never fails, its own run ==")
    before = effective_thread_config()
    check(isinstance(before, ThreadConfig),
          "the probe returns a ThreadConfig in every environment")
    check(before.ruled_default == RULED_OMP_NUM_THREADS == 8,
          "the ruled default of record travels in every config", "8")
    check(before.env_requested.keys() == set(THREAD_ENV_VARS)
          and all(isinstance(v, str) for v in before.env_requested.values()),
          "every thread env var is recorded, '(unset)' where absent",
          f"{len(THREAD_ENV_VARS)} vars")
    check(before.consensus == "unresolved" or before.note == "",
          "an unresolved config ALWAYS carries a note saying why (M19: a "
          "degraded read is described, never silent)", before.note[:60])
    check(before.effective_num_threads is None
          or before.matches_ruled_default is not None,
          "matches_ruled_default is None only when the count is unresolved — "
          "'could not tell' and 'does not match' are different facts")

    print("== selftest 2: with a BLAS runtime mapped, the count is READ ==")
    try:
        import numpy as _np                                   # noqa: F401
        _np.linalg.eigh(_np.eye(4))          # force the LAPACK path to load
        blas = True
    except Exception as exc:                                  # noqa: BLE001
        blas = False
        skip(f"NAMED SKIP — BLAS probe: numpy is not importable in this "
             f"configuration ({type(exc).__name__}: {exc}), so the mapped-BLAS "
             f"half of the grid cannot run here. It runs in the project venv "
             f"with: python -m metabasis.threads --selftest")
    if blas:
        after = effective_thread_config()
        blas_pools = [p for p in after.pools if p.kind in ("openblas", "mkl", "blis")]
        check(bool(blas_pools),
              "a BLAS runtime is mapped into the process once numpy has run",
              ", ".join(f"{p.library}({p.kind})" for p in blas_pools) or "none")
        check(all("/" not in p.library for p in after.pools),
              "every pool records a BASENAME, never a path — a stamp travels "
              "into the repo and a path carries a username with it")
        check(after.consensus in ("agreed", "disagreed"),
              "the count RESOLVES once a BLAS runtime is loaded", after.consensus)
        if after.consensus == "agreed":
            check(isinstance(after.effective_num_threads, int)
                  and after.effective_num_threads >= 1,
                  "the effective count is a positive integer read from the "
                  "runtime, not parsed from the env",
                  str(after.effective_num_threads))
            check(after.matches_ruled_default
                  == (after.effective_num_threads == RULED_OMP_NUM_THREADS),
                  "matches_ruled_default agrees with the count it describes",
                  f"{after.effective_num_threads} vs ruled {RULED_OMP_NUM_THREADS}")
            check(all(p.symbol for p in blas_pools if p.num_threads is not None),
                  "every answered pool names the SYMBOL that answered, so the "
                  "measurement is auditable",
                  ", ".join(str(p.symbol) for p in blas_pools if p.symbol))
        else:
            check(THREADS_UNRESOLVED in after.note or "DIFFERENT" in after.note,
                  "a disagreement is REPORTED as one, with the per-pool counts "
                  "intact, rather than averaged away", after.note[:70])
        check(after.quoted != "" and (after.effective_num_threads is None)
              == (THREADS_UNRESOLVED in after.quoted),
              "`quoted` prints the sentinel exactly when there is no count",
              after.quoted)

    print("== selftest 3: the stamp block is plain JSON and opens with STATUS ==")
    block = thread_config_stamp()
    check(list(block)[0] == "STATUS" and THREAD_STAMP_STATUS == block["STATUS"],
          "the block opens with the STATUS line, so a reader who has never "
          "seen the field knows in one line what it is")
    try:
        round_trip = json.loads(json.dumps(block))
        check(round_trip == block,
              "the block is JSON-round-trippable as written — a stamp that "
              "cannot be re-read is not a record")
    except (TypeError, ValueError) as exc:
        check(False, f"the block must be JSON-serializable ({exc})")
    check("ruled_default" in block and block["ruled_default"] == 8,
          "the block states the ruled default, so an artifact says what the "
          "convention WAS when it was built")

    print("== selftest 4: the reader keeps PRE-RULING apart from DEGRADED ==")
    fixed = ThreadConfig(effective_num_threads=8, consensus="agreed",
                         matches_ruled_default=True,
                         env_requested=_requested_env(), probe="fixture")
    stamped = {"builder": "x", THREAD_STAMP_KEY: thread_config_stamp(fixed)}
    check(stamp_thread_config({"builder": "x"}) is None,
          "a PRE-RULING stamp reads as None — the count is UNRECORDED and must "
          "never be inferred, even though every job script of that vintage "
          "exported OMP_NUM_THREADS=1")
    check(stamp_thread_config(None) is None and stamp_thread_config([]) is None,  # type: ignore[arg-type]
          "a missing or non-mapping stamp reads as pre-ruling rather than raising")
    got = stamp_thread_config(stamped)
    check(got is not None and got.effective_num_threads == 8
          and got.consensus == "agreed",
          "a stamped block round-trips through the reader with its count intact")
    nested = {"builder": "x", "trunk": {THREAD_STAMP_KEY: thread_config_stamp(fixed)}}
    check((stamp_thread_config(nested) or ThreadConfig()).effective_num_threads == 8,
          "a block nested under `trunk` is found too — all three shapes get "
          "handed around and a reader that saw one would silently return None "
          "for the others")
    bare = stamp_thread_config(thread_config_stamp(fixed))
    check(bare is not None and bare.effective_num_threads == 8,
          "and the block handed in on its own is read as itself")
    broken = stamp_thread_config({THREAD_STAMP_KEY: {"nonsense": 1}})
    check(broken is not None and broken.consensus == "unresolved"
          and THREADS_UNRESOLVED in broken.note,
          "an UNPARSEABLE block is an unresolved probe, NEVER a pre-ruling "
          "absence — the field's presence is itself evidence", broken.note[:60])

    print("== selftest 5: the count mismatch is LABELED, and quotes both sides ==")
    eight = ThreadConfig(effective_num_threads=8, consensus="agreed",
                         matches_ruled_default=True)
    one = ThreadConfig(effective_num_threads=1, consensus="agreed",
                       matches_ruled_default=False)
    unres = ThreadConfig(consensus="unresolved",
                         note=f"{THREADS_UNRESOLVED}: fixture")
    check(thread_count_mismatch(eight, eight) is None,
          "two artifacts at the SAME known count raise nothing")
    note = thread_count_mismatch(one, eight)
    check(note is not None and THREAD_COUNT_MISMATCH_LABEL in note
          and "banked 1" in note and "rebuild 8" in note,
          "a real mismatch is LABELED and quotes BOTH counts", (note or "")[:72])
    pre = thread_count_mismatch(None, eight)
    check(pre is not None and PRE_RULING_UNRECORDED in pre and "rebuild 8" in pre,
          "a PRE-RULING banked side is quoted as unrecorded, not silently "
          "treated as agreeing", (pre or "")[:72])
    check(thread_count_mismatch(None, None) is not None,
          "two unrecorded sides are still a quoted mismatch: neither can vouch "
          "for its own count, and 'both unknown' is not 'both equal'")
    deg = thread_count_mismatch(unres, eight)
    check(deg is not None and THREADS_UNRESOLVED in deg,
          "a DEGRADED probe on one side is quoted as degraded, distinctly from "
          "a pre-ruling absence", (deg or "")[:72])
    check(all(THREAD_COUNT_MISMATCH_LABEL in n for n in (note, pre, deg)
              if n is not None),
          "every mismatch carries the same greppable label, so a log sweep "
          "never has to recognise prose")
    check("characterized" in (note or ""),
          "and it says the difference is CHARACTERIZED, so a reader does not "
          "adjudicate it as corruption")

    print(f"\nselftest: TOTAL — checks run: {checks} "
          f"({len(skips)} named skip(s)), {len(failures)} failure(s)")
    for line in skips:
        print(f"  SKIPPED: {line.splitlines()[0]}")
    for line in failures:
        print(f"  FAILED: {line}")
    return 1 if failures else 0


# ---------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only verification of the probe, the stamp block, "
                         "the backward-compatible reader and the mismatch label")
    ap.add_argument("--show", action="store_true",
                    help="print THIS process's effective thread configuration")
    ap.add_argument("--json", action="store_true",
                    help="with --show: the stamp block as a builder writes it")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.show:
        import numpy as _np                                   # noqa: F401
        _np.linalg.eigh(_np.eye(4))          # load the runtime we are describing
        cfg = effective_thread_config()
        if args.json:
            print(json.dumps(thread_config_stamp(cfg), indent=1))
        else:
            print(f"effective BLAS threads : {cfg.quoted}  ({cfg.consensus})")
            print(f"ruled default          : {cfg.ruled_default}  "
                  f"(matches: {cfg.matches_ruled_default})")
            for pool in cfg.pools:
                print(f"  pool {pool.library} [{pool.kind}] -> "
                      f"{pool.num_threads} via {pool.symbol}"
                      + (f"  ({pool.note})" if pool.note else ""))
            for name, value in cfg.env_requested.items():
                print(f"  env  {name} = {value}")
            if cfg.note:
                print(f"  note: {cfg.note}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
