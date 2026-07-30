"""ONE per-card VRAM estimator for every preflight that reserves capacity.

WHY THIS MODULE EXISTS. Three node-side vector-build scripts carried an ad-hoc
`weights × 1.6 / n_cards` heuristic that had diverged from the M19-derived
estimator the big-model lane uses, and two mirrored scan scripts carried an older
`× 1.25` of the same shape. Five copies of one arithmetic, in shell heredocs, none
of them selftested, deciding whether a job runs. The desk's flag (ledgered
2026-07-29) was "rule before a third rung"; the rung arrived, and the rule is that
there is one estimator and it lives here.

────────────────────────────────────────────────────────────────────────────────
THE TWO BOUNDS, AND WHY BOTH ARE COMPUTED
────────────────────────────────────────────────────────────────────────────────
**The FLAT bound** — `weights_gib × multiplier / n_cards`. Crude and deliberately
so: it needs nothing but the size of the checkpoint on disk, and its multiplier
covers activations, workspace and allocator fragmentation in one number.

**The STRUCTURAL bound** (M19-derived, the big-model lane's) — the worst CARD's
resident weight from the ACTUAL per-layer sizes distributed over the shard
fan-out, plus a load TRANSIENT that is measured where a measurement is available
and a conservative analytic figure where it is not, plus an explicit margin. It
is the honest one whenever layers are heterogeneous: the DSV3 lane's 3 dense
layers at 1.09 GiB beside 58 MoE layers at 21.43 GiB cannot be described by any
average, and an even split by layer INDEX forces at least two cards to hold 8 MoE
layers — 171.5 GiB where `weights / n_cards` says 156.2.

**The estimator returns the MAXIMUM of the two, and names which one bound.**
Neither dominates. The flat bound is larger on a homogeneous dense checkpoint
(its multiplier is generous); the structural bound is larger wherever layer sizes
are skewed or the load transient is big. Taking the max is what "preserve the more
conservative behaviour where they disagree" means as code rather than as a habit.

────────────────────────────────────────────────────────────────────────────────
WHICH ONE WAS MORE CONSERVATIVE, AND THE RULING THAT SETTLED IT
────────────────────────────────────────────────────────────────────────────────
On the case that raised the question — the 405B v2.1 re-bank's sharded collection
— the FLAT ×1.6 bound was the STRICTER of the two, and the desk ruled
(2026-07-29, ledgered): **KEEP ×1.6.** Its reasoning is preserved here because it
is the reasoning, not the number, that generalizes:

    a PASS under the stricter bound is sound (it reserved more than needed);
    a BLOCK under the stricter bound escalates to the structural estimator
    rather than to a lowered multiplier.

So `FLAT_MULTIPLIER = 1.6` is the ruled value and `SUPERSEDED_FLAT_MULTIPLIER =
1.25` is recorded beside it — named, not deleted, because five deployed scripts
carry one of the two and a mechanical sweep must be able to SEE both by value
(rake M26). The 1.25 scripts are not "wrong"; they are less conservative, and a
job that passed under 1.25 and would block under 1.6 is a job whose margin was
never verified.

Rake M19 governs the transient. `TransientEstimate.measured is False` does not
lower the bound: the ANALYTIC figure is used and the consumer BLOCKS on it,
because a block is cheap and a silent risk is not. Instrumentation that only
describes a run must never be able to fail it — so a caller that could not
measure passes `None` and gets the conservative path, never an exception.

────────────────────────────────────────────────────────────────────────────────
NO TORCH
────────────────────────────────────────────────────────────────────────────────
This module imports numpy/pydantic and nothing heavier: it is arithmetic over
file sizes and caller-supplied measurements, so it belongs to the CPU spine and a
preflight can consult it before a single weight is read. A MEASURED transient and
the free-VRAM readings are the CALLER's to obtain (they need torch, and the
caller already has it); this module only decides what the numbers mean.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.capacity --selftest
  python -m metabasis.capacity --checkpoint <LOCAL_WEIGHTS_DIR> --n-cards 8 \\
      [--min-free-gib 40] [--margin-gib 2] [--transient-gib <measured>] \\
      [--free-gib 178.2,177.9,...] [--json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("capacity")

# ---------------------------------------------------------------- constants
#: bytes per GiB. Every figure in this module is GiB, never GB — the scheduler,
#: `nvidia-smi` and `torch.cuda.mem_get_info` all speak GiB, and mixing the two
#: units is a 7% error in the direction that matters.
GIB = 1 << 30

#: THE RULED FLAT MULTIPLIER (Luxia/desk ruling 2026-07-29, ledgered): weights ×
#: this, split across the fan-out, covers activations + workspace +
#: fragmentation. Kept because it was the STRICTER bound on the case that raised
#: the question; a block under it escalates to the structural estimator rather
#: than to a smaller multiplier.
FLAT_MULTIPLIER = 1.6
#: The value the two mirrored scan scripts still carry. NAMED, not deleted (rake
#: M26): a mechanical sweep must be able to see both multipliers BY VALUE, and a
#: job that passed under 1.25 and would block under 1.6 never verified its margin.
SUPERSEDED_FLAT_MULTIPLIER = 1.25
#: The conservative load-transient reservation used when nothing was measured.
#: Derived in the DSV3 lane from the largest single fused tensor transformers
#: materializes (one `gate_up_proj` at 14.0 GiB) plus slack. Rake M19: a degraded
#: measurement uses THIS and the consumer blocks on it, rather than gambling.
DEFAULT_TRANSIENT_ANALYTIC_GIB = 16.0
#: Headroom above the computed need, for allocator behaviour nobody models.
DEFAULT_MARGIN_GIB = 2.0
#: Per-card free-VRAM floor a tiny model must still refuse a starved card at.
DEFAULT_MIN_FREE_GIB = 40.0
#: Weight-file suffixes, in preference order. `.safetensors` first because every
#: roster checkpoint uses it; `.bin` is the legacy fallback. A directory holding
#: both is a staging error and is REFUSED rather than summed twice.
WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")


class CapacityError(RuntimeError):
    """A capacity question cannot be answered as asked.

    Only ever raised for a MALFORMED question — no weight files, a non-positive
    card count, a layer table that does not sum to the checkpoint. Never for
    "the model does not fit": that is a VERDICT, returned as data, because a
    preflight must be able to report a block without an exception unwinding
    through it.
    """


class CheckpointFootprint(BaseModel):
    """What a checkpoint weighs on disk, and how it is shaped.

    `layer_bytes` is the per-decoder-layer breakdown when the caller can supply
    one (from a safetensors index, or measured off a skeleton). Its ABSENCE is
    recorded rather than papered over: without it the structural bound must
    assume an even byte split, which under-reserves on a heterogeneous stack, and
    a reader has to be able to tell an assumption from a measurement.
    """
    path: str
    total_bytes: int = Field(gt=0)
    n_files: int = Field(gt=0)
    suffix: str
    layer_bytes: list[int] = Field(
        default=[], description="per-decoder-layer byte sizes, in LAYER ORDER; "
                                "empty => not supplied, and the structural "
                                "bound falls back to an even byte split")

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def layers_supplied(self) -> bool:
        return bool(self.layer_bytes)

    @property
    def non_layer_bytes(self) -> int:
        """Embeddings, the LM head, norms — everything outside the layer stack.

        Charged to the FIRST card, which is where `device_map` places the
        embeddings and where the input tensor lands.
        """
        if not self.layer_bytes:
            return 0
        return max(0, self.total_bytes - sum(self.layer_bytes))


class TransientEstimate(BaseModel):
    """The load transient, and whether anybody actually measured it.

    RAKE M19 in one object. `measured=False` does not soften the number — the
    analytic figure is used and the consumer blocks on it. The distinction is
    reported so a log says "we reserved 16.0 GiB because we could not measure"
    rather than presenting a guess as a reading.
    """
    gib: float = Field(ge=0.0)
    measured: bool
    note: str = ""

    @classmethod
    def analytic(cls, gib: float = DEFAULT_TRANSIENT_ANALYTIC_GIB,
                 why: str = "") -> "TransientEstimate":
        return cls(gib=gib, measured=False,
                   note=("DEGRADED INSTRUMENTATION — nothing was measured, so "
                         "the conservative analytic bound is reserved and the "
                         "consumer BLOCKS on it rather than gambling (rake M19). "
                         + why).strip())

    @classmethod
    def from_measurement(cls, gib: float) -> "TransientEstimate":
        if gib < 0.0:
            raise CapacityError(
                f"a measured transient cannot be negative (got {gib} GiB). A "
                f"`peak - resident` reading below zero means the peak was taken "
                f"before the load, or in a process that had already peaked "
                f"(rake M29) — the measurement is invalid, not the model")
        return cls(gib=gib, measured=True,
                   note="MEASURED in this process, artifact loaded first "
                        "(rake M29: one transient measurement per fresh "
                        "process; a second in the same process is an upper "
                        "bound contaminated by the first)")


class CapacityEstimate(BaseModel):
    """The per-card reservation, both bounds, and which one binds."""
    checkpoint: str
    n_cards: int = Field(gt=0)
    weights_gib: float
    #: `weights_gib × FLAT_MULTIPLIER / n_cards`, floored by `min_free_gib`.
    flat_bound_gib: float
    flat_multiplier: float
    #: worst-card resident + transient + margin, from the real layer shape.
    structural_bound_gib: float
    worst_card_resident_gib: float
    per_card_resident_gib: list[float]
    transient: TransientEstimate
    margin_gib: float
    min_free_gib: float
    #: max(flat, structural) — the reservation of record.
    need_per_card_gib: float
    binding_bound: Literal["flat", "structural", "floor"]
    layer_shape: Literal["measured", "assumed-even"]
    notes: list[str] = []

    @property
    def disagreement_gib(self) -> float:
        """How far apart the two bounds are. Reported, never averaged."""
        return abs(self.flat_bound_gib - self.structural_bound_gib)


class CapacityVerdict(BaseModel):
    """Whether the cards on offer can host the reservation. A RESULT, not a raise."""
    estimate: CapacityEstimate
    free_gib: list[float]
    least_free_gib: float
    slack_gib: float
    fits: bool
    blocked_reason: str = ""

    @model_validator(mode="after")
    def _reason_iff_blocked(self) -> "CapacityVerdict":
        if self.fits == bool(self.blocked_reason):
            raise ValueError(
                "a verdict must carry a reason exactly when it blocks: a block "
                "with no reason cannot be acted on, and a pass with one reads as "
                "a block to anybody scanning the log")
        return self


def checkpoint_footprint(path: Path,
                         layer_bytes: Optional[Sequence[int]] = None
                         ) -> CheckpointFootprint:
    """Sum a checkpoint's weight files. `.safetensors` preferred, `.bin` fallback.

    Both suffixes present is a HALT, not a sum: a directory holding a
    safetensors set AND a legacy bin set describes ONE model twice, and adding
    them reserves double while looking like a careful over-estimate.

    Symlinked weight files are followed (`Path.stat()` does, and rake M38's whole
    lesson is that a store reached through a symlink is the normal case under the
    /models placement rule). A symlink whose target is gone is a HALT: a
    checkpoint that cannot be measured must not be silently measured as smaller.
    """
    root = Path(path)
    if not root.is_dir():
        raise CapacityError(
            f"{root} is not a directory — the footprint is measured over a "
            f"checkpoint DIRECTORY's weight files, never over one file")
    found: dict[str, list[Path]] = {}
    for suffix in WEIGHT_SUFFIXES:
        hits = sorted(p for p in root.glob(f"*{suffix}"))
        if hits:
            found[suffix] = hits
    if not found:
        raise CapacityError(
            f"{root} holds no {' or '.join(WEIGHT_SUFFIXES)} files. A capacity "
            f"estimate over an empty directory would return a comfortable number "
            f"for a model that is not there — this is the case where 'cannot "
            f"find' and 'does not match' must read differently (rake M31(b))")
    if len(found) > 1:
        raise CapacityError(
            f"{root} holds BOTH "
            + " and ".join(f"{len(v)} {k}" for k, v in found.items())
            + " weight files. That is one model described twice; summing them "
              "would reserve double while looking like caution. Name the "
              "intended set and re-ask")
    suffix, files = next(iter(found.items()))
    total = 0
    for file in files:
        try:
            total += file.stat().st_size          # follows symlinks (rake M38)
        except OSError as exc:
            raise CapacityError(
                f"{file} cannot be measured ({exc}). Under the /models placement "
                f"rule a weight file is often a symlink into a store; a DANGLING "
                f"one must halt, because measuring the checkpoint as smaller "
                f"than it is under-reserves silently") from exc
    if total <= 0:
        raise CapacityError(f"{root}: {len(files)} {suffix} file(s) totalling 0 bytes")
    layers = [int(b) for b in (layer_bytes or [])]
    if any(b < 0 for b in layers):
        raise CapacityError(f"{root}: a layer size cannot be negative: {layers}")
    if layers and sum(layers) > total:
        raise CapacityError(
            f"{root}: the supplied layer table sums to {sum(layers) / GIB:.2f} "
            f"GiB but the checkpoint weighs {total / GIB:.2f} GiB. A layer table "
            f"larger than its own checkpoint describes a different model — rake "
            f"M23(c)'s rule at footprint grain: when two counts that should agree "
            f"disagree, assert the identity rather than record the defect")
    return CheckpointFootprint(path=str(root), total_bytes=total,
                               n_files=len(files), suffix=suffix,
                               layer_bytes=layers)


def flat_bound_gib(weights_gib: float, n_cards: int,
                   multiplier: float = FLAT_MULTIPLIER) -> float:
    """`weights × multiplier / n_cards`, the crude bound, rounded as deployed.

    Rounded to 1 dp because every deployed copy of this arithmetic rounds that
    way and a preflight's printed number must be the number it compares against.
    """
    if n_cards <= 0:
        raise CapacityError(f"n_cards must be positive, got {n_cards}")
    if multiplier <= 0.0:
        raise CapacityError(f"multiplier must be positive, got {multiplier}")
    return round(weights_gib * multiplier / n_cards, 1)


def per_card_resident_gib(footprint: CheckpointFootprint, n_cards: int
                          ) -> tuple[list[float], Literal["measured", "assumed-even"]]:
    """Resident weight per card under an even split BY LAYER INDEX.

    This is the layout `--shard-across N` builds and the one every big-rung bank
    collected under, so it is the layout a preflight must reserve for. The split
    is by layer INDEX, not by bytes: `accelerate` places whole layers, so a stack
    of 58 heterogeneous layers over 8 cards forces some cards to hold 8 of them
    however large those 8 are. That is precisely where `weights / n_cards`
    under-reserves, and where the flat bound's multiplier is doing the work.

    Non-layer weights (embeddings, LM head, final norm) are charged to card 0,
    which is where `device_map` places the embeddings and where inputs land.
    """
    if n_cards <= 0:
        raise CapacityError(f"n_cards must be positive, got {n_cards}")
    if not footprint.layers_supplied:
        even = footprint.total_gib / n_cards
        return [round(even, 3)] * n_cards, "assumed-even"
    sizes = footprint.layer_bytes
    per_card = [0.0] * n_cards
    for i, size in enumerate(sizes):
        per_card[(i * n_cards) // len(sizes)] += size / GIB
    per_card[0] += footprint.non_layer_bytes / GIB
    return [round(x, 3) for x in per_card], "measured"


def estimate_capacity(footprint: CheckpointFootprint, n_cards: int, *,
                      transient: Optional[TransientEstimate] = None,
                      margin_gib: float = DEFAULT_MARGIN_GIB,
                      min_free_gib: float = DEFAULT_MIN_FREE_GIB,
                      multiplier: float = FLAT_MULTIPLIER) -> CapacityEstimate:
    """The per-card reservation: max(flat bound, structural bound), floored.

    Both bounds are computed and BOTH are reported; the larger binds. Neither
    dominates in general (the flat multiplier is generous on a homogeneous dense
    stack; the structural bound is larger on a skewed one or under a big load
    transient), so taking the max is how "preserve the more conservative
    behaviour" becomes arithmetic instead of a habit — and `binding_bound` says
    which one it was, so a surprising reservation can be explained without
    re-deriving it.
    """
    if margin_gib < 0.0:
        raise CapacityError(f"margin must be non-negative, got {margin_gib} GiB")
    if min_free_gib < 0.0:
        raise CapacityError(f"floor must be non-negative, got {min_free_gib} GiB")
    trans = transient if transient is not None else TransientEstimate.analytic(
        why="no measurement was supplied to estimate_capacity()")
    resident, shape = per_card_resident_gib(footprint, n_cards)
    worst = max(resident)
    structural = round(worst + trans.gib + margin_gib, 1)
    flat = flat_bound_gib(footprint.total_gib, n_cards, multiplier)
    need = max(flat, structural, min_free_gib)
    #  A TIE is reported as `flat`, deliberately and in this order: the flat bound
    #  is the RULED one, so when two bounds agree the ruling is what a reader
    #  should see named rather than whichever branch happened to be tested first.
    binding: Literal["flat", "structural", "floor"] = (
        "flat" if need == flat
        else "structural" if need == structural else "floor")

    notes: list[str] = []
    if shape == "assumed-even":
        notes.append(
            "LAYER SHAPE ASSUMED EVEN — no per-layer table was supplied, so the "
            "structural bound splits BYTES evenly instead of LAYERS. That "
            "UNDER-reserves on a heterogeneous stack (the DSV3 case: 3 dense "
            "layers beside 58 MoE layers cannot be described by an average), "
            "which is exactly when the flat bound must be the one that binds. "
            "Supply layer_bytes for a checkpoint whose layers differ in size")
    if not trans.measured:
        notes.append(
            f"TRANSIENT NOT MEASURED — reserving the conservative analytic "
            f"{trans.gib} GiB. Rake M19: a degraded measurement sets "
            f"measured=false and the consumer BLOCKS on the bound rather than "
            f"gambling; a block is cheap, a silent risk is not")
    if binding == "structural" and shape == "measured":
        notes.append(
            f"THE STRUCTURAL BOUND BINDS at {structural} GiB against the flat "
            f"bound's {flat} GiB. The layer stack is skewed enough (or the "
            f"transient large enough) that weights/n_cards × {multiplier} "
            f"under-describes the worst card — this is the case the M19-derived "
            f"estimator exists for")
    if binding == "flat":
        notes.append(
            f"THE FLAT BOUND BINDS at {flat} GiB against the structural bound's "
            f"{structural} GiB — the ruled ×{multiplier} multiplier is the "
            f"stricter reading here, which is the 2026-07-29 ruling's own case. A "
            f"PASS under it is sound; a BLOCK under it escalates to the "
            f"structural estimator, never to a smaller multiplier")
    if binding == "floor":
        notes.append(
            f"THE FREE-VRAM FLOOR BINDS at {min_free_gib} GiB — both computed "
            f"bounds are below it, which is the small-model case the floor exists "
            f"for: a tiny checkpoint must still refuse a starved card")
    return CapacityEstimate(
        checkpoint=footprint.path, n_cards=n_cards,
        weights_gib=round(footprint.total_gib, 2),
        flat_bound_gib=flat, flat_multiplier=multiplier,
        structural_bound_gib=structural,
        worst_card_resident_gib=round(worst, 3),
        per_card_resident_gib=resident, transient=trans,
        margin_gib=margin_gib, min_free_gib=min_free_gib,
        need_per_card_gib=need, binding_bound=binding, layer_shape=shape,
        notes=notes)


def capacity_verdict(estimate: CapacityEstimate,
                     free_gib: Sequence[float]) -> CapacityVerdict:
    """Does the LEAST-FREE card clear the reservation? A verdict, never a raise.

    The comparison is against the least-free card, not the mean and not card 0:
    a whole-layer pipeline map cannot go below its worst card, so one starved
    card blocks the job however comfortable the others are.
    """
    frees = [float(f) for f in free_gib]
    if len(frees) != estimate.n_cards:
        raise CapacityError(
            f"{len(frees)} free-VRAM reading(s) for a {estimate.n_cards}-card "
            f"estimate. A count mismatch here means the reading and the plan "
            f"describe different node states, and comparing them would answer a "
            f"question nobody asked")
    least = min(frees)
    slack = round(least - estimate.need_per_card_gib, 1)
    fits = least >= estimate.need_per_card_gib
    reason = "" if fits else (
        f"least-free card {least:.1f} GiB < {estimate.need_per_card_gib} GiB "
        f"needed ({estimate.binding_bound} bound: worst-card resident "
        f"{estimate.worst_card_resident_gib:.1f} + transient "
        f"{estimate.transient.gib} [measured={estimate.transient.measured}] + "
        f"margin {estimate.margin_gib}; flat bound "
        f"{estimate.flat_bound_gib}). A whole-layer pipeline map cannot go below "
        f"its worst card, so this is a BLOCK and not a tuning knob")
    return CapacityVerdict(estimate=estimate, free_gib=frees,
                           least_free_gib=round(least, 1), slack_gib=slack,
                           fits=fits, blocked_reason=reason)


# ---------------------------------------------------------------- selftest
def selftest() -> int:                                   # noqa: C901 — a checklist
    """CPU-only, no weights: file-size arithmetic and every refusal."""
    import tempfile

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    def _ckpt(root: Path, sizes_gib: Sequence[float],
              suffix: str = ".safetensors") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for i, gib in enumerate(sizes_gib):
            (root / f"model-{i:05d}{suffix}").write_bytes(b"\0" * int(gib * GIB))
        return root

    print("== selftest 1: the footprint is measured, never assumed ==")
    with tempfile.TemporaryDirectory(prefix="capacity_") as td:
        root = Path(td)
        #  Small real files; the arithmetic is exact in bytes, so a fixture in
        #  MiB proves the same identities a 750 GiB checkpoint would.
        ck = _ckpt(root / "dense", [1 / 1024, 2 / 1024, 1 / 1024])
        fp = checkpoint_footprint(ck)
        check(fp.n_files == 3 and fp.total_bytes == int(4 / 1024 * GIB)
              and fp.suffix == ".safetensors",
              f"three shards sum exactly: {fp.total_bytes} bytes "
              f"({fp.total_gib:.6f} GiB), suffix {fp.suffix}")
        check(not fp.layers_supplied and fp.non_layer_bytes == 0,
              "with no layer table supplied the footprint SAYS so rather than "
              "inventing one")

        binfp = checkpoint_footprint(_ckpt(root / "legacy", [1 / 1024], ".bin"))
        check(binfp.suffix == ".bin",
              "a legacy .bin checkpoint measures through the fallback suffix")
        _ckpt(root / "both", [1 / 1024])
        _ckpt(root / "both", [1 / 1024], ".bin")
        for bad, needle, why in (
                (root / "both", "BOTH",
                 "a directory holding both suffixes HALTs — one model described "
                 "twice, and summing it reserves double while looking careful"),
                (root / "empty", "no ",
                 "a directory with no weight files HALTs rather than returning a "
                 "comfortable number for a model that is not there"),
                (root / "dense" / "model-00000.safetensors", "not a directory",
                 "a single FILE is refused: the footprint is over a checkpoint "
                 "directory")):
            (root / "empty").mkdir(exist_ok=True)
            try:
                checkpoint_footprint(bad)
                check(False, f"{why} — must HALT")
            except CapacityError as exc:
                check(needle in str(exc), f"{why} [{needle!r} in the halt]")
        dangling = root / "dangling"
        dangling.mkdir()
        (dangling / "model-00000.safetensors").symlink_to(root / "gone.safetensors")
        try:
            checkpoint_footprint(dangling)
            check(False, "a DANGLING weight symlink must HALT")
        except CapacityError as exc:
            check("cannot be measured" in str(exc),
                  "a dangling weight symlink HALTs — under the /models placement "
                  "rule a weight file is often a symlink into a store, and "
                  "measuring the checkpoint as smaller under-reserves silently")
        try:
            checkpoint_footprint(ck, layer_bytes=[fp.total_bytes * 2])
            check(False, "a layer table bigger than its checkpoint must HALT")
        except CapacityError as exc:
            check("describes a different model" in str(exc),
                  "a layer table that sums ABOVE its own checkpoint HALTs (rake "
                  "M23(c): assert the identity, never record the defect)")

    print("== selftest 2: the flat bound is the deployed arithmetic, exactly ==")
    check(flat_bound_gib(755.96, 8) == round(755.96 * 1.6 / 8, 1) == 151.2,
          f"the 405B case: 755.96 GiB × {FLAT_MULTIPLIER} / 8 = "
          f"{flat_bound_gib(755.96, 8)} GiB per card")
    check(flat_bound_gib(755.96, 8, SUPERSEDED_FLAT_MULTIPLIER) == 118.1,
          f"and under the SUPERSEDED ×{SUPERSEDED_FLAT_MULTIPLIER} the same "
          f"checkpoint reserves {flat_bound_gib(755.96, 8, SUPERSEDED_FLAT_MULTIPLIER)} "
          f"GiB — 33.1 GiB per card less, which is what 'stricter' means in "
          f"figures")
    check(FLAT_MULTIPLIER > SUPERSEDED_FLAT_MULTIPLIER,
          f"the RULED multiplier is the larger one ({FLAT_MULTIPLIER} > "
          f"{SUPERSEDED_FLAT_MULTIPLIER}) — the ruling kept the more "
          f"conservative bound, and this asserts the direction rather than "
          f"trusting the comment")
    for bad_n, bad_m in ((0, FLAT_MULTIPLIER), (-1, FLAT_MULTIPLIER), (8, 0.0)):
        try:
            flat_bound_gib(100.0, bad_n, bad_m)
            check(False, f"flat_bound_gib({bad_n}, {bad_m}) must be refused")
        except CapacityError:
            check(True, f"a non-positive card count or multiplier is refused "
                        f"({bad_n} cards, ×{bad_m})")

    print("== selftest 3: the structural bound splits LAYERS, not bytes ==")
    #  The DSV3 shape, scaled: 3 small layers + 58 large ones over 8 cards. An
    #  even BYTE split says total/8; the real layout forces some cards to hold 8
    #  large layers. This is the case the M19-derived estimator exists for.
    small, large = 1.09, 21.43
    sizes = [small] * 3 + [large] * 58
    layer_bytes = [int(g * GIB) for g in sizes]
    total_bytes = sum(layer_bytes) + int(3.0 * GIB)   # + embeddings/head
    fp3 = CheckpointFootprint(path="<dsv3-shape>", total_bytes=total_bytes,
                              n_files=163, suffix=".safetensors",
                              layer_bytes=layer_bytes)
    resident, shape = per_card_resident_gib(fp3, 8)
    even = fp3.total_gib / 8
    check(shape == "measured" and len(resident) == 8,
          f"per-card resident from the real layer table: "
          f"{[round(x, 1) for x in resident]} GiB")
    check(max(resident) > even,
          f"the worst card holds {max(resident):.1f} GiB where an even BYTE "
          f"split says {even:.1f} GiB — a {max(resident) - even:.1f} GiB "
          f"under-reservation the flat multiplier would have to cover")
    check(abs(sum(resident) - fp3.total_gib) < 1e-6,
          f"and the split conserves the checkpoint: Σ per-card "
          f"{sum(resident):.3f} == {fp3.total_gib:.3f} GiB")
    check(resident[0] >= fp3.non_layer_bytes / GIB,
          "non-layer weights (embeddings, LM head, final norm) are charged to "
          "card 0, where device_map places them and where inputs land")
    flat3, shape3 = per_card_resident_gib(
        CheckpointFootprint(path="<flat>", total_bytes=total_bytes, n_files=1,
                            suffix=".safetensors"), 8)
    check(shape3 == "assumed-even" and len(set(flat3)) == 1,
          f"with no layer table the split is EVEN and SAYS so "
          f"({shape3}) — an assumption a reader can see")

    print("== selftest 4: the estimate takes the MAX and names which bound ==")
    est3 = estimate_capacity(fp3, 8, transient=TransientEstimate.analytic())
    check(est3.need_per_card_gib
          == max(est3.flat_bound_gib, est3.structural_bound_gib,
                 est3.min_free_gib),
          f"need = max(flat {est3.flat_bound_gib}, structural "
          f"{est3.structural_bound_gib}, floor {est3.min_free_gib}) = "
          f"{est3.need_per_card_gib} GiB")
    #  THE RULING, MEASURED. On the campaign's largest and most heterogeneous
    #  checkpoint — the one whose skew motivated the structural estimator at all —
    #  the RULED flat ×1.6 bound is still the STRICTER of the two, by 60 GiB per
    #  card. That is the 2026-07-29 ruling's own claim ("×1.6 is stricter than
    #  v3's"), and it is asserted here rather than believed.
    check(est3.binding_bound == "flat"
          and est3.flat_bound_gib > est3.structural_bound_gib,
          f"on the DSV3 shape the RULED FLAT bound is the stricter one: flat "
          f"{est3.flat_bound_gib} > structural {est3.structural_bound_gib} GiB "
          f"(they differ by {est3.disagreement_gib:.1f} GiB) — the ruling's claim, "
          f"measured on the case that raised it")
    check(any("FLAT BOUND BINDS" in n and "escalates to the structural" in n
              for n in est3.notes),
          "and its note carries the ruling's reasoning: a PASS under the stricter "
          "bound is sound, a BLOCK escalates to the structural estimator rather "
          "than to a smaller multiplier")
    #  THE STRUCTURAL BOUND MUST ALSO BE ABLE TO BIND, or "take the max" is a
    #  branch nobody has run (rake M19(c)). Two mechanisms, both real:
    #  (a) a load TRANSIENT large relative to the per-card weight — the small-model
    #      case, where 0.6 × weights/n_cards does not cover one fused tensor;
    even_layers = [int(1.0 * GIB)] * 80
    fp4 = CheckpointFootprint(path="<80-GiB-even-stack>",
                              total_bytes=sum(even_layers), n_files=20,
                              suffix=".safetensors", layer_bytes=even_layers)
    est4 = estimate_capacity(fp4, 8, transient=TransientEstimate.analytic(),
                             margin_gib=2.0, min_free_gib=8.0)
    check(est4.binding_bound == "structural"
          and est4.structural_bound_gib > est4.flat_bound_gib,
          f"a big TRANSIENT flips it: structural {est4.structural_bound_gib} > "
          f"flat {est4.flat_bound_gib} GiB, because the analytic "
          f"{est4.transient.gib} GiB transient exceeds what 0.6 × "
          f"{fp4.total_gib / 8:.1f} GiB/card of multiplier headroom covers")
    check(any("STRUCTURAL BOUND BINDS" in n for n in est4.notes),
          "and its note names the structural bound and why, so a surprising "
          "reservation is explainable without re-deriving it")
    #  (b) EXTREME SKEW — one layer far larger than the rest. Whole-layer
    #      placement puts it on ONE card, and no average can describe that.
    skewed = [int(0.1 * GIB)] * 100 + [int(50.0 * GIB)]
    fp4b = CheckpointFootprint(path="<one-huge-layer>", total_bytes=sum(skewed),
                               n_files=8, suffix=".safetensors",
                               layer_bytes=skewed)
    est4b = estimate_capacity(fp4b, 8,
                              transient=TransientEstimate.from_measurement(0.0),
                              margin_gib=2.0, min_free_gib=8.0)
    check(est4b.binding_bound == "structural"
          and est4b.worst_card_resident_gib >= 50.0,
          f"and so does extreme SKEW: one 50 GiB layer lands whole on one card "
          f"({est4b.worst_card_resident_gib:.1f} GiB resident), so structural "
          f"{est4b.structural_bound_gib} > flat {est4b.flat_bound_gib} GiB — "
          f"weights/n_cards is off by {est4b.structural_bound_gib - est4b.flat_bound_gib:.1f} "
          f"GiB and no multiplier chosen in advance would close it")
    tiny = CheckpointFootprint(path="<3b>", total_bytes=int(6.0 * GIB),
                               n_files=2, suffix=".safetensors")
    est5 = estimate_capacity(tiny, 1,
                             transient=TransientEstimate.from_measurement(0.5),
                             margin_gib=1.0)
    check(est5.binding_bound == "floor"
          and est5.need_per_card_gib == DEFAULT_MIN_FREE_GIB,
          f"and a tiny checkpoint is floored at {est5.need_per_card_gib} GiB — a "
          f"small model must still refuse a starved card")
    check(len({est3.binding_bound, est4.binding_bound, est5.binding_bound}) == 3,
          f"all THREE binding branches are exercised — flat, structural, floor — "
          f"because a max whose branches nobody ran is not known to take the "
          f"larger (rake M19(c))")
    check(est5.transient.measured
          and "MEASURED in this process" in est5.transient.note,
          "a measured transient is recorded AS measured, with rake M29's "
          "one-measurement-per-fresh-process rule named on it")
    check(not est3.transient.measured
          and "BLOCKS on it rather than gambling" in est3.transient.note
          and any("TRANSIENT NOT MEASURED" in n for n in est3.notes),
          "and an unmeasured one is recorded as DEGRADED with rake M19's rule — "
          "the number is not softened, the consumer blocks on it")
    try:
        TransientEstimate.from_measurement(-0.1)
        check(False, "a negative measured transient must be refused")
    except CapacityError as exc:
        check("rake M29" in str(exc),
              "a negative `peak - resident` is refused and named as rake M29 (a "
              "peak taken before the load, or in an already-peaked process)")
    for kwargs in ({"margin_gib": -1.0}, {"min_free_gib": -1.0}):
        try:
            estimate_capacity(tiny, 1, **kwargs)     # type: ignore[arg-type]
            check(False, f"{kwargs} must be refused")
        except CapacityError:
            check(True, f"a negative {list(kwargs)[0]} is refused")

    print("== selftest 5: the verdict is data, and blocks on the WORST card ==")
    ok = capacity_verdict(est4, [est4.need_per_card_gib + 3.0] * 8)
    check(ok.fits and ok.slack_gib == 3.0 and not ok.blocked_reason,
          f"eight comfortable cards fit with {ok.slack_gib} GiB slack")
    starved = [est4.need_per_card_gib + 3.0] * 8
    starved[5] = est4.need_per_card_gib - 0.5
    bad = capacity_verdict(est4, starved)
    check(not bad.fits and bad.least_free_gib == round(starved[5], 1)
          and "BLOCK and not a tuning knob" in bad.blocked_reason,
          f"ONE starved card blocks the job however comfortable the others are: "
          f"least-free {bad.least_free_gib} GiB, slack {bad.slack_gib} GiB")
    check(f"measured={est4.transient.measured}" in bad.blocked_reason
          and str(est4.flat_bound_gib) in bad.blocked_reason
          and str(est4.worst_card_resident_gib)[:4] in bad.blocked_reason
          and est4.binding_bound in bad.blocked_reason,
          f"and the block message carries EVERY term of the reservation — which "
          f"bound bound, worst-card resident, transient with its measured flag, "
          f"margin and the flat bound beside it — so the operator can act on it "
          f"without re-running the estimator: {bad.blocked_reason[:96]}…")
    check(isinstance(bad, CapacityVerdict),
          "a block is RETURNED, never raised — a preflight must be able to "
          "report one without an exception unwinding through it")
    try:
        capacity_verdict(est4, [100.0, 100.0])
        check(False, "a free-VRAM reading count mismatch must be refused")
    except CapacityError as exc:
        check("different node states" in str(exc),
              "a reading count that disagrees with the plan is refused, not "
              "reconciled")
    try:
        CapacityVerdict(estimate=est4, free_gib=[1.0] * 8, least_free_gib=1.0,
                        slack_gib=-1.0, fits=False, blocked_reason="")
        check(False, "a block with no reason must be refused")
    except Exception:                               # noqa: BLE001
        check(True, "a blocked verdict with an EMPTY reason is refused by the "
                    "model itself — a block nobody can act on is not a block")

    print(f"\nselftest: {len(failures)} failure(s)")
    for failure in failures:
        print(f"  FAILED: {failure}")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only verification of the arithmetic and every "
                         "refusal; no weights, no GPU")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="local weights directory (node-side)")
    ap.add_argument("--n-cards", type=int, default=None,
                    help="the shard fan-out this job will use")
    ap.add_argument("--layer-gib", default=None,
                    help="comma list of per-decoder-layer sizes in GiB, in LAYER "
                         "ORDER. Supply it for a heterogeneous stack (MoE): "
                         "without it the structural bound assumes an even BYTE "
                         "split, which under-reserves, and the output says so")
    ap.add_argument("--transient-gib", type=float, default=None,
                    help="a MEASURED load transient (peak - resident, one "
                         "measurement per fresh process with the artifact loaded "
                         "FIRST — rake M29). Omit it and the conservative "
                         f"analytic {DEFAULT_TRANSIENT_ANALYTIC_GIB} GiB is "
                         f"reserved and the verdict BLOCKS on that (rake M19)")
    ap.add_argument("--margin-gib", type=float, default=DEFAULT_MARGIN_GIB)
    ap.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB,
                    help="per-card free-VRAM floor, so a tiny model still "
                         "refuses a starved card")
    ap.add_argument("--multiplier", type=float, default=FLAT_MULTIPLIER,
                    help=f"the flat bound's multiplier (ruled {FLAT_MULTIPLIER} "
                         f"2026-07-29; the superseded value is "
                         f"{SUPERSEDED_FLAT_MULTIPLIER}). Lowering it is NOT the "
                         f"response to a block — escalating to the structural "
                         f"bound is")
    ap.add_argument("--free-gib", default=None,
                    help="comma list of per-card free VRAM in GiB (the caller "
                         "reads these with torch.cuda.mem_get_info). With it, a "
                         "VERDICT is emitted and a block exits nonzero")
    ap.add_argument("--json", action="store_true",
                    help="emit the estimate/verdict as JSON for a job script to "
                         "parse, instead of the human table")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    missing = [f"--{n.replace('_', '-')}" for n in ("checkpoint", "n_cards")
               if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)} "
                         f"(or --selftest)")
    try:
        layers = ([int(float(x) * GIB) for x in args.layer_gib.split(",") if x.strip()]
                  if args.layer_gib else None)
        footprint = checkpoint_footprint(args.checkpoint, layers)
        transient = (TransientEstimate.from_measurement(args.transient_gib)
                     if args.transient_gib is not None
                     else TransientEstimate.analytic(
                         why="no --transient-gib was supplied"))
        estimate = estimate_capacity(
            footprint, args.n_cards, transient=transient,
            margin_gib=args.margin_gib, min_free_gib=args.min_free_gib,
            multiplier=args.multiplier)
    except (CapacityError, ValueError) as exc:
        #  An EXPECTED halt with a meaningful message: the question was malformed.
        #  A preflight must be able to tell "cannot find / cannot measure" from
        #  "does not fit" — they demand opposite responses (rake M31(b)).
        print(f"CAPACITY-QUESTION-MALFORMED: {exc}", file=sys.stderr)
        return 2

    verdict: Optional[CapacityVerdict] = None
    if args.free_gib:
        try:
            verdict = capacity_verdict(
                estimate, [float(x) for x in args.free_gib.split(",") if x.strip()])
        except (CapacityError, ValueError) as exc:
            print(f"CAPACITY-QUESTION-MALFORMED: {exc}", file=sys.stderr)
            return 2

    if args.json:
        payload: dict[str, Any] = (verdict.model_dump() if verdict is not None
                                   else estimate.model_dump())
        print(json.dumps(payload, indent=1))
    else:
        print(f"checkpoint {estimate.checkpoint}")
        print(f"  weights {estimate.weights_gib} GiB over {estimate.n_cards} "
              f"card(s); layer shape {estimate.layer_shape}")
        print(f"  per-card resident "
              f"{[round(x, 1) for x in estimate.per_card_resident_gib]} GiB; "
              f"worst {estimate.worst_card_resident_gib:.1f} GiB")
        print(f"  flat bound       {estimate.flat_bound_gib:8.1f} GiB "
              f"(weights × {estimate.flat_multiplier} / {estimate.n_cards})")
        print(f"  structural bound {estimate.structural_bound_gib:8.1f} GiB "
              f"(worst card + transient {estimate.transient.gib} "
              f"[measured={estimate.transient.measured}] + margin "
              f"{estimate.margin_gib})")
        print(f"  NEED PER CARD    {estimate.need_per_card_gib:8.1f} GiB "
              f"({estimate.binding_bound} bound binds; the two differ by "
              f"{estimate.disagreement_gib:.1f} GiB)")
        for note in estimate.notes:
            print(f"  NOTE: {note}")
        if verdict is not None:
            print(f"  least-free card {verdict.least_free_gib:.1f} GiB, slack "
                  f"{verdict.slack_gib:.1f} GiB")
            print("  VERDICT: "
                  + ("FITS" if verdict.fits
                     else f"PREFLIGHT-BLOCKED — {verdict.blocked_reason}"))
    if verdict is not None and not verdict.fits:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
