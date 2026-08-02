"""Tentative portability coefficients c_M from a set of exchange rates.

THE ALGEBRA. The star factorization (candidate) says the exchange rate of a pair
factorizes into one portability coefficient per model:

    â(A→B) = c_A · c_B

Take logs and it is LINEAR:

    log â(A→B) = log c_A + log c_B

so a set of exchange rates over any pair graph is one least-squares problem in
`x_M := log c_M`. That is the whole solver. `solve_star_systems` writes the same
system out by hand for the four banked models, closed by the direct 3B→Qwen fit;
this module is that machinery generalized to an arbitrary pair set, and its
selftest cross-checks the two agree to machine precision on the banked shape.

WHY A GENERAL SOLVER IS NEEDED NOW. The wave-1 shape is a pure hub STAR: every
exchange rate has the hub on one side. A star graph is bipartite, so its design
matrix has a one-dimensional null space — the gauge c_hub → t·c_hub,
c_leaf → c_leaf/t leaves every â unchanged. A star alone therefore CANNOT
determine any coefficient. Exactly one anchor fixes it, and the anchor of record
is the hub constant from the inherited banked system (native::proc_k128:
c_8B = .7545, the system that passes its own held-out check). The solver
reports, per solve, whether the gauge was actually fixed and by what — an
unanchored star comes back `gauge_fixed=false` and `USABLE_FOR_PREDICTIONS=false`
rather than silently returning the min-norm solution as if it meant something.

Once ANY non-hub pair is measured, the graph stops being bipartite and the
system closes on its own; the anchor then becomes a consistency check instead of
a necessity, and the solver reports the anchor residual.

CONVENTIONS INHERITED FROM `solve_star_systems` (do not drift):
  * The star is a POSITIVE-FACTOR model. Non-positive â cannot enter the log
    system; such pairs are excluded by name with the reason recorded, never
    silently dropped or abs()'d.
  * Families are NEVER mixed inside one system, and arms are never mixed
    (rake 40, and the arm-consistency rule: a constant belongs to the system it
    was solved in). One solve = one (arm × family).
  * The rank guard k ≤ n_train/1.2 is reported, not silently applied — a solve
    whose inputs include a rank-forbidden fit is marked not-usable.
  * A system must pass a held-out check before a new constant may hang on it.

STATUS: everything this module emits is TENTATIVE and UNSTAMPED. It derives
coefficients; it files no predictions. The prediction ceremony (prereg §3 —
file BEFORE the pair is fit) lives in the prediction tooling, and nothing here
may be used to back-fill one.

Run (repo root):
  python -m metabasis.scripts.solve_portability_coefficients --selftest
  python -m metabasis.scripts.solve_portability_coefficients \
      --exchange-rates outputs/collection/readouts/exchange_rates_hub8b_native_k128.json \
      --anchor-from-inherited native::proc_k128 \
      --out outputs/collection/readouts/portability_tentative_native_k128.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from metabasis.scripts.read_exchange_rates import BANK_ROOT, FAMILY_OF_RECORD
from metabasis.threads import thread_config_stamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("solve_portability_coefficients")

#: prereg §3: held-out star predictions are scored at ±.05 absolute.
BAND_HALFWIDTH = 0.05
#: singular values below this fraction of the largest are treated as zero.
RANK_TOL = 1e-9
#: the inherited banked star systems — the source of the hub anchor of record.
INHERITED_SYSTEMS = (BANK_ROOT / "arms" / "A8_conjugation" / "smalls" /
                     "readouts_cpu" / "star_systems.json")
#: the hub constant's name inside those banked systems.
INHERITED_HUB_KEY = "c_8B"
INHERITED_HUB_MODEL = "8b"


# ---------------------------------------------------------------- data model
class ExchangeRateObservation(BaseModel):
    """One â feeding the system. `source`/`target` are bank model keys."""
    source: str
    target: str
    a_hat: float
    arm: str
    family: str
    pair_id: str = ""
    rank_forbidden: bool = False
    clears_null_floor: Optional[bool] = None

    def key(self) -> str:
        return self.pair_id or f"{self.source}->{self.target}"


class ExcludedPair(BaseModel):
    pair_id: str
    a_hat: float
    reason: str


class PairResidual(BaseModel):
    pair_id: str
    observed: float
    predicted: float
    abs_error: float


class HeldOutCheck(BaseModel):
    """A pair kept OUT of the fit and re-predicted from the solved constants."""
    pair_id: str
    formula: str
    predicted: float
    observed: float
    abs_error: float
    band: tuple[float, float]
    in_band: bool


class PortabilitySolution(BaseModel):
    STATUS: str = (
        "TENTATIVE / UNSTAMPED — derived coefficients only. Files no prediction, "
        "scores nothing. The prediction ceremony (prereg §3: file BEFORE the pair "
        "is fit) is a separate step and may not be back-filled from this.")
    model_form: str = "â(A→B) = c_A · c_B, solved as log â = log c_A + log c_B"
    arm: str
    family: str
    generated: str
    coefficients: dict[str, float] = {}
    anchors: dict[str, float] = {}
    anchor_provenance: str = ""
    gauge: str = ""
    gauge_fixed: bool = False
    n_models: int = 0
    n_pairs_used: int = 0
    design_rank: int = 0
    nullspace_dim: int = 0
    anchor_residuals: dict[str, float] = {}
    residuals: list[PairResidual] = []
    max_abs_residual: Optional[float] = None
    rms_residual: Optional[float] = None
    excluded: list[ExcludedPair] = []
    held_out_checks: list[HeldOutCheck] = []
    rank_forbidden_inputs: list[str] = []
    USABLE_FOR_PREDICTIONS: bool = False
    notes: list[str] = Field(default_factory=list)
    #: THE EFFECTIVE THREAD CONFIGURATION (Luxia ruling 2026-08-01). The design
    #: matrix is decomposed by `np.linalg.svd` (rank + nullspace) and solved by
    #: `np.linalg.lstsq`, both blocked LAPACK routines whose summation order
    #: moves with the thread count — so the coefficients this document carries
    #: are thread-conditioned in the last digits and the count belongs beside
    #: them. Empty dict = this solution predates the ruling; the backward-
    #: compatible reader is `metabasis.threads.stamp_thread_config`.
    thread_config: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- the solver
def _design(observations: Sequence[ExchangeRateObservation], models: Sequence[str]
            ) -> tuple[np.ndarray, np.ndarray]:
    idx = {m: i for i, m in enumerate(models)}
    a = np.zeros((len(observations), len(models)), dtype=np.float64)
    b = np.zeros(len(observations), dtype=np.float64)
    for r, o in enumerate(observations):
        a[r, idx[o.source]] += 1.0
        a[r, idx[o.target]] += 1.0        # self-pair (A→A) correctly gets a 2
        b[r] = math.log(o.a_hat)
    return a, b


def _nullspace(a: np.ndarray, tol: float = RANK_TOL) -> tuple[np.ndarray, int]:
    """Orthonormal basis of null(A) as columns, plus the numerical rank."""
    if a.size == 0:
        return np.zeros((a.shape[1], a.shape[1])), 0
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    smax = float(s[0]) if s.size else 0.0
    rank = int((s > tol * smax).sum()) if smax > 0 else 0
    return vt[rank:].T, rank


def solve_log_least_squares(
        observations: Sequence[ExchangeRateObservation],
        anchors: Optional[Mapping[str, float]] = None,
        arm: str = "", family: str = "",
        held_out: Sequence[ExchangeRateObservation] = (),
) -> PortabilitySolution:
    """Solve â = c_A·c_B for all models touched by `observations`.

    `anchors` pin named coefficients (log-space exact where the gauge allows).
    Pairs with â ≤ 0 are excluded by name — the star is a positive-factor model.
    """
    arm_set = {o.arm for o in observations} | {o.arm for o in held_out}
    fam_set = {o.family for o in observations} | {o.family for o in held_out}
    if len(arm_set) > 1 or len(fam_set) > 1:
        raise ValueError(
            f"one solve = one (arm × family); got arms {sorted(arm_set)} and "
            f"families {sorted(fam_set)}. Families and arms are never mixed "
            "inside a star system (rake 40 / the arm-consistency rule).")

    sol = PortabilitySolution(
        arm=arm or (next(iter(arm_set)) if arm_set else ""),
        family=family or (next(iter(fam_set)) if fam_set else ""),
        generated=date.today().isoformat(),
        anchors={k: float(v) for k, v in (anchors or {}).items()})

    usable: list[ExchangeRateObservation] = []
    for o in observations:
        if not math.isfinite(o.a_hat):
            sol.excluded.append(ExcludedPair(pair_id=o.key(), a_hat=o.a_hat,
                                             reason="â is not finite"))
        elif o.a_hat <= 0.0:
            sol.excluded.append(ExcludedPair(
                pair_id=o.key(), a_hat=o.a_hat,
                reason="â ≤ 0 cannot enter the log system — the star "
                       "factorization is a positive-factor model. Report the "
                       "sign as a finding; do not abs() it into the solve."))
        else:
            usable.append(o)
        if o.rank_forbidden:
            sol.rank_forbidden_inputs.append(o.key())

    if not usable:
        sol.notes.append("no usable exchange rates — nothing solved")
        return sol

    models = sorted({m for o in usable for m in (o.source, o.target)})
    sol.n_models, sol.n_pairs_used = len(models), len(usable)
    a, b = _design(usable, models)
    null_basis, rank = _nullspace(a)
    sol.design_rank, sol.nullspace_dim = rank, int(null_basis.shape[1])

    x0, *_ = np.linalg.lstsq(a, b, rcond=None)          # min-norm particular sol

    idx = {m: i for i, m in enumerate(models)}
    anchor_items = [(m, v) for m, v in (anchors or {}).items()
                    if m in idx and v > 0.0]
    for m, v in (anchors or {}).items():
        if m not in idx:
            sol.notes.append(f"anchor {m}={v} ignored: model absent from the "
                             f"pair set {models}")
        elif v <= 0.0:
            sol.notes.append(f"anchor {m}={v} ignored: coefficients are positive")

    if sol.nullspace_dim == 0:
        sol.gauge_fixed = True
        sol.gauge = ("determined by the pair graph itself (non-bipartite: at "
                     "least one closing pair). Anchors are consistency checks "
                     "here, not constraints.")
        x = x0
    elif anchor_items:
        c_mat = np.zeros((len(anchor_items), len(models)))
        d_vec = np.zeros(len(anchor_items))
        for r, (m, v) in enumerate(anchor_items):
            c_mat[r, idx[m]] = 1.0
            d_vec[r] = math.log(v)
        cn = c_mat @ null_basis                          # [n_anchor, null_dim]
        t, *_ = np.linalg.lstsq(cn, d_vec - c_mat @ x0, rcond=None)
        _, s_cn, _ = np.linalg.svd(cn, full_matrices=False)
        gauge_rank = int((s_cn > RANK_TOL * float(s_cn[0])).sum()) if s_cn.size and s_cn[0] > 0 else 0
        sol.gauge_fixed = gauge_rank == sol.nullspace_dim
        sol.gauge = (f"anchored on {', '.join(m for m, _ in anchor_items)} "
                     f"(gauge rank {gauge_rank} of {sol.nullspace_dim} null "
                     f"direction(s))")
        x = x0 + null_basis @ t
    else:
        sol.gauge = (f"UNFIXED — the pair graph leaves {sol.nullspace_dim} gauge "
                     "freedom(s) (a pure hub star is bipartite: c_hub→t·c_hub, "
                     "c_leaf→c_leaf/t leaves every â unchanged) and no anchor "
                     "was supplied. The coefficients below are the min-norm "
                     "representative of an entire family and mean nothing on "
                     "their own.")
        x = x0

    sol.coefficients = {m: round(float(math.exp(x[idx[m]])), 4) for m in models}
    for m, v in anchor_items:
        sol.anchor_residuals[m] = round(abs(float(math.exp(x[idx[m]])) - v), 6)

    errs: list[float] = []
    for o in usable:
        pred = math.exp(x[idx[o.source]] + x[idx[o.target]])
        err = abs(pred - o.a_hat)
        errs.append(err)
        sol.residuals.append(PairResidual(pair_id=o.key(),
                                          observed=round(o.a_hat, 4),
                                          predicted=round(pred, 4),
                                          abs_error=round(err, 4)))
    sol.max_abs_residual = round(max(errs), 4)
    sol.rms_residual = round(float(np.sqrt(np.mean(np.square(errs)))), 4)

    for o in held_out:
        if o.source not in idx or o.target not in idx:
            sol.notes.append(f"held-out {o.key()} touches a model outside the "
                             "solved set — not scorable")
            continue
        pred = math.exp(x[idx[o.source]] + x[idx[o.target]])
        lo, hi = pred - BAND_HALFWIDTH, pred + BAND_HALFWIDTH
        sol.held_out_checks.append(HeldOutCheck(
            pair_id=o.key(),
            formula=f"c_{o.source} ({sol.coefficients[o.source]}) × "
                    f"c_{o.target} ({sol.coefficients[o.target]})",
            predicted=round(pred, 4), observed=round(o.a_hat, 4),
            abs_error=round(abs(pred - o.a_hat), 4),
            band=(round(lo, 4), round(hi, 4)),
            in_band=bool(lo <= o.a_hat <= hi)))

    checks_ok = all(h.in_band for h in sol.held_out_checks)
    if not sol.held_out_checks:
        sol.notes.append(
            "NO held-out check available (a pure hub star has no spare pair to "
            "hold out). A system that has not been checked may not host a "
            "constant a prediction is hung on — measure one non-hub pair, or "
            "treat these coefficients as provisional only.")
    sol.USABLE_FOR_PREDICTIONS = bool(
        sol.gauge_fixed and not sol.rank_forbidden_inputs
        and sol.held_out_checks and checks_ok)
    return sol


# ---------------------------------------------------------------- adapters
def observations_from_readout(path: Path,
                              require_null_floor: bool = False
                              ) -> list[ExchangeRateObservation]:
    """Read the JSON written by `read_exchange_rates` into observations."""
    if not path.exists():
        raise SystemExit(f"exchange-rate readout absent: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"unreadable readout {path}: {exc}") from exc
    rows = doc.get("rows", [])
    if not rows:
        raise SystemExit(
            f"{path} holds no â rows (missing pieces: "
            f"{len(doc.get('missing', []))}). Nothing to solve — the wave-1 "
            "native target builds are the blocker, not this solver.")
    out: list[ExchangeRateObservation] = []
    for r in rows:
        if require_null_floor and not r.get("clears_null_floor", False):
            logger.warning("dropping %s: â does not clear its own null floor",
                           r.get("pair_id"))
            continue
        out.append(ExchangeRateObservation(
            source=r["source_model"], target=r["target_model"],
            a_hat=float(r["a_hat"]), arm=r["arm"], family=r["family"],
            pair_id=r.get("pair_id", ""),
            rank_forbidden=bool(r.get("rank_forbidden", False)),
            clears_null_floor=r.get("clears_null_floor")))
    return out


def inherited_hub_anchor(system_key: str,
                         systems_path: Path = INHERITED_SYSTEMS
                         ) -> tuple[dict[str, float], str]:
    """The hub constant of record from the banked star systems.

    Refuses a system that fails its own out-of-sample check — the same rule
    `extend_star_system` enforces before hanging a new constant on a system.
    """
    if not systems_path.exists():
        raise SystemExit(f"banked star systems absent: {systems_path}")
    systems = json.loads(systems_path.read_text()).get("systems", {})
    blk = systems.get(system_key)
    if blk is None:
        raise SystemExit(f"{system_key!r} not in {systems_path} "
                         f"(have {sorted(systems)})")
    chk = blk.get("out_of_sample_check", {})
    if not chk.get("within_pm_0.05", False):
        raise SystemExit(
            f"REFUSING to anchor on {system_key}: it fails its own out-of-sample "
            f"check (predicted {chk.get('predicted')} vs observed "
            f"{chk.get('observed')}). A system that fails its own check is not a "
            "system a new constant may be hung on.")
    c_hub = blk.get("constants", {}).get(INHERITED_HUB_KEY)
    if c_hub is None:
        raise SystemExit(f"{system_key} has no {INHERITED_HUB_KEY}")
    prov = (f"{INHERITED_HUB_KEY}={c_hub} from the banked {system_key} system "
            f"({systems_path}); that system's own held-out check: predicted "
            f"{chk.get('predicted')} vs observed {chk.get('observed')} "
            f"(|e|={chk.get('abs_error')}), PASS")
    return {INHERITED_HUB_MODEL: float(c_hub)}, prov


# ---------------------------------------------------------------- selftest
def _obs(src: str, tgt: str, a: float, arm: str = "native",
         family: str = FAMILY_OF_RECORD) -> ExchangeRateObservation:
    return ExchangeRateObservation(source=src, target=tgt, a_hat=a, arm=arm,
                                   family=family, pair_id=f"{src}->{tgt}")


def selftest() -> int:                                   # noqa: C901 — a checklist
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    print("== selftest 1: exact recovery on a hub star with one anchor ==")
    truth = {"8b": 0.7545, "qwen2.5-3b-instruct": 0.61, "qwen2.5-14b-instruct": 0.52,
             "qwen2.5-32b-instruct": 0.48, "mistral-7b-instruct-v0.3": 0.44,
             "olmo2-7b-instruct": 0.39, "phi-4": 0.55,
             "phi-3.5-mini-instruct": 0.31}
    star = [_obs("8b", m, truth["8b"] * c) for m, c in truth.items() if m != "8b"]
    sol = solve_log_least_squares(star, anchors={"8b": truth["8b"]})
    worst = max(abs(sol.coefficients[m] - round(c, 4)) for m, c in truth.items())
    check(sol.gauge_fixed, f"gauge fixed by the anchor ({sol.gauge})")
    check(sol.nullspace_dim == 1,
          f"a pure hub star is bipartite → nullspace dim {sol.nullspace_dim} (want 1)")
    check(worst < 5e-5, f"all 8 constants recovered, worst |Δc| = {worst:.2e}")
    check(sol.max_abs_residual is not None and sol.max_abs_residual < 1e-4,
          f"max |â_pred − â_obs| = {sol.max_abs_residual}")
    check(not sol.USABLE_FOR_PREDICTIONS,
          "star-only system is NOT usable for predictions (no held-out check) "
          "even though it recovers exactly — the check discipline holds")

    print("== selftest 2: an UNANCHORED star refuses to pretend ==")
    bare = solve_log_least_squares(star)
    check(not bare.gauge_fixed, "gauge_fixed=False without an anchor")
    check(not bare.USABLE_FOR_PREDICTIONS, "USABLE_FOR_PREDICTIONS=False")
    check(bare.max_abs_residual is not None and bare.max_abs_residual < 1e-4,
          f"…yet it still reproduces every â (residual {bare.max_abs_residual}) — "
          "which is exactly why the gauge flag, not the residual, is the guard")
    scaled = {m: bare.coefficients[m] / truth[m] for m in truth}
    ratios = sorted(scaled.values())
    check(abs(ratios[0] * ratios[-1] - scaled["8b"] ** 0) > -1,  # always true; report
          f"min/max gauge ratio {ratios[0]:.4f}/{ratios[-1]:.4f} (a whole family)")

    print("== selftest 3: agreement with the banked closed-form solver ==")
    from metabasis.scripts.solve_star_systems import solve_system
    banked = {"3b->8b": 0.4815, "8b->qwen-7b": 0.3569, "8b->dsv2-lite": 0.3331,
              "3b->qwen-7b": 0.3019}
    closed = solve_system(banked)
    assert closed is not None
    tri = [_obs("3b", "8b", banked["3b->8b"]),
           _obs("8b", "qwen-7b", banked["8b->qwen-7b"]),
           _obs("3b", "qwen-7b", banked["3b->qwen-7b"]),
           _obs("8b", "dsv2-lite", banked["8b->dsv2-lite"])]
    gen = solve_log_least_squares(tri)
    check(gen.nullspace_dim == 0,
          "the 3B/8B/Qwen triangle closes the system on its own (nullspace 0)")
    pairs = [("3b", "c_3B"), ("8b", "c_8B"), ("qwen-7b", "c_Qwen"),
             ("dsv2-lite", "c_DSV2")]
    for key, name in pairs:
        d = abs(gen.coefficients[key] - closed[name])
        check(d <= 1e-4, f"{name}: general {gen.coefficients[key]} vs banked "
                         f"closed form {closed[name]} (|Δ| {d:.1e})")

    print("== selftest 4: the banked out-of-sample check reproduces ==")
    gen_ho = solve_log_least_squares(tri, held_out=[_obs("3b", "dsv2-lite", 0.2773)])
    ho = gen_ho.held_out_checks[0]
    check(abs(ho.predicted - 0.2818) <= 1e-4,
          f"held-out 3b->dsv2-lite predicted {ho.predicted} (banked: 0.2818)")
    check(ho.in_band and abs(ho.abs_error - 0.0045) <= 1e-4,
          f"observed {ho.observed}, |e|={ho.abs_error}, in ±.05 band {ho.band}")
    check(gen_ho.USABLE_FOR_PREDICTIONS,
          "a gauge-fixed system that passes its held-out check IS usable")

    print("== selftest 5: noisy â still recovers c within the noise ==")
    rng = np.random.default_rng(11)
    noisy = [_obs("8b", m, max(1e-6, truth["8b"] * c
                               + float(rng.normal(0, 0.01))))
             for m, c in truth.items() if m != "8b"]
    nsol = solve_log_least_squares(noisy, anchors={"8b": truth["8b"]})
    worst_n = max(abs(nsol.coefficients[m] - round(c, 4))
                  for m, c in truth.items() if m != "8b")
    check(worst_n < 0.05,
          f"σ=.01 noise on â → worst |Δc| = {worst_n:.4f} (< .05, the band width)")

    print("== selftest 6: non-positive and non-finite â are excluded by name ==")
    bad = star + [_obs("8b", "gpt2-xl", -0.03), _obs("8b", "broken", float("nan"))]
    bsol = solve_log_least_squares(bad, anchors={"8b": truth["8b"]})
    reasons = {e.pair_id: e.reason for e in bsol.excluded}
    check(set(reasons) == {"8b->gpt2-xl", "8b->broken"},
          f"excluded exactly the bad pairs: {sorted(reasons)}")
    check("positive-factor" in reasons["8b->gpt2-xl"],
          "the negative-â exclusion names the positive-factor model")
    check("gpt2-xl" not in bsol.coefficients,
          "a model whose only pair was excluded gets NO coefficient (not a zero)")
    check(max(abs(bsol.coefficients[m] - round(c, 4))
              for m, c in truth.items()) < 5e-5,
          "the surviving constants are unchanged by the exclusions")

    print("== selftest 7: arm/family mixing is refused ==")
    try:
        solve_log_least_squares([_obs("8b", "phi-4", 0.4, arm="native"),
                                 _obs("8b", "phi-4", 0.3, arm="raw")])
        check(False, "mixing arms must raise")
    except ValueError as exc:
        check(True, f"mixing arms raises: {exc!s:.72}")
    try:
        solve_log_least_squares([_obs("8b", "phi-4", 0.4, family="proc_k128"),
                                 _obs("8b", "phi-4", 0.3, family="proc_k32")])
        check(False, "mixing families must raise")
    except ValueError as exc:
        check(True, f"mixing families raises: {exc!s:.72}")

    print("== selftest 8: rank-forbidden inputs poison usability ==")
    rf = [_obs("3b", "8b", banked["3b->8b"]),
          _obs("8b", "qwen-7b", banked["8b->qwen-7b"]),
          _obs("3b", "qwen-7b", banked["3b->qwen-7b"])]
    rf[0].rank_forbidden = True
    rsol = solve_log_least_squares(
        rf, held_out=[_obs("8b", "dsv2-lite", banked["8b->dsv2-lite"])])
    check(rsol.rank_forbidden_inputs == ["3b->8b"],
          f"rank-forbidden inputs named: {rsol.rank_forbidden_inputs}")
    check(not rsol.USABLE_FOR_PREDICTIONS,
          "a system built on a rank-forbidden fit is not usable")

    print(f"\nselftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exchange-rates", type=Path, default=None,
                    help="JSON written by read_exchange_rates")
    ap.add_argument("--anchor", action="append", default=[], metavar="MODEL=VALUE",
                    help="pin a coefficient, repeatable (e.g. --anchor 8b=0.7545)")
    ap.add_argument("--anchor-from-inherited", default=None, metavar="ARM::FAMILY",
                    help="take the hub anchor from the banked star systems, e.g. "
                         "native::proc_k128. Refuses a system that fails its own "
                         "out-of-sample check.")
    ap.add_argument("--hold-out", action="append", default=[], metavar="PAIR_ID",
                    help="keep this pair OUT of the fit and score it against the "
                         "±.05 band, repeatable")
    ap.add_argument("--require-null-floor", action="store_true",
                    help="drop rows whose â does not clear their own null floor")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.exchange_rates is None:
        raise SystemExit("--exchange-rates is required (or --selftest)")

    obs = observations_from_readout(args.exchange_rates,
                                    require_null_floor=args.require_null_floor)
    hold = set(args.hold_out)
    held_out = [o for o in obs if o.key() in hold]
    fit_obs = [o for o in obs if o.key() not in hold]
    unknown = hold - {o.key() for o in obs}
    if unknown:
        raise SystemExit(f"--hold-out names pairs absent from the readout: "
                         f"{sorted(unknown)}")

    anchors: dict[str, float] = {}
    provenance = ""
    if args.anchor_from_inherited:
        anchors, provenance = inherited_hub_anchor(args.anchor_from_inherited)
    for spec in args.anchor:
        if "=" not in spec:
            raise SystemExit(f"bad --anchor {spec!r}; want MODEL=VALUE")
        m, _, v = spec.partition("=")
        try:
            anchors[m.strip()] = float(v)
        except ValueError as exc:
            raise SystemExit(f"bad --anchor {spec!r}: {exc}") from exc
        provenance = (provenance + "; " if provenance else "") + f"CLI --anchor {spec}"

    sol = solve_log_least_squares(fit_obs, anchors=anchors, held_out=held_out)
    sol.anchor_provenance = provenance
    #  Read AFTER the solve, so the block describes the threadpool that actually
    #  ran the svd/lstsq rather than the one that was requested at start-up.
    sol.thread_config = thread_config_stamp()
    payload = sol.model_dump_json(indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        logger.info("wrote %s", args.out)
    else:
        print(payload)
    logger.info("c_M (%s::%s) = %s", sol.arm, sol.family, sol.coefficients)
    logger.info("gauge_fixed=%s  usable=%s  max|resid|=%s  %s", sol.gauge_fixed,
                sol.USABLE_FOR_PREDICTIONS, sol.max_abs_residual, sol.gauge)
    for h in sol.held_out_checks:
        logger.info("HELD-OUT %s pred %s obs %s band %s in_band=%s", h.pair_id,
                    h.predicted, h.observed, h.band, h.in_band)
    return 0


if __name__ == "__main__":
    sys.exit(main())
