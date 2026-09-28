"""THE STAR SYSTEMS for the extension pairs, one per (template-arm x family).

Arm consistency: a star constant belongs to the system it was solved in. The native-arm constants of record (c_3B=.6382, c_8B=.7545,
c_Qwen=.4730, c_DSV2=.4415 at k128) are NATIVE-ARM numbers. Any model whose fits are
raw-arm-only — OLMo has no chat template, so its native arm does not exist — must take
its constant from a PARALLEL RAW-ARM SYSTEM, derived from the already-banked raw-arm
a_hat rows. Slotting a raw-arm a_hat into the native-arm system would be the
family-mixing sin one level up: an equation that mixes fit families mixes estimators
with different attenuation, and its answer scores nothing real.

This script builds every system that closes, from banked fits only. ZERO GPU COMPUTE.

The algebra. Star model: a_hat(A->B) = c_A . c_B. With models {3B, 8B, Qwen, DSV2} the
three hub pairs (3B-8B, 8B-Qwen, 8B-DSV2) leave the system underdetermined; the direct
3B->Qwen fit (the transitivity pair) closes it:

    c_3B^2  = a(3B->8B) . a(3B->Qwen) / a(8B->Qwen)
    c_8B    = a(3B->8B)  / c_3B
    c_Qwen  = a(8B->Qwen)/ c_8B
    c_DSV2  = a(8B->DSV2)/ c_8B

3B->DSV2 is then a PREDICTION, and the fourth-node fit measures it — so every system printed here
carries its own out-of-sample check, and a system whose check fails is not a system you
may hang a new constant on. Reported per (arm, family); families are never mixed inside
one equation and the rank guard k <= n_train/1.2 is applied per pair, since n_train
differs across pairs (600 hub, 360 for the transitivity closer).

UNSTAMPED: scores nothing. This is the substrate for the extension-pair predictions,
which are filed BEFORE any new pair is fitted.

Run (repository root):

    python -m metabasis.scripts.solve_star_systems              # writes <OUT>/star_systems.json
    python -m metabasis.scripts.solve_star_systems --selftest   # synthetic, temp dir only

`--arm-root`, `--bank-root` and `--out-dir` relocate the inputs and the output;
an unrecognized argument is a usage error (exit 2), never ignored.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from metabasis.scripts.fit_transport_maps import load_transport_map
from metabasis.scripts.read_transported_axes import _unit, cos

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("solve_star_systems")

ARM = Path("outputs/battery/arms/A8_conjugation")
BANK = Path("outputs/battery")
OUT = ARM / "smalls" / "readouts_cpu"

ARMS = ("native", "raw")
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
RANK_GUARD_DIVISOR = 1.2          # rank guard: k <= n_train / 1.2

# The V7_L<site> vectors are the attenuation-class reference axis.
# a_hat(A->B) := cos(g.V7_A, V7_B).
V7: dict[str, tuple[str, str, int]] = {
    "3b": ("a5_vectors_3b_b7/a5_vectors.npz", "V7_L14", 14),
    "8b": ("a5_vectors_8b_b7/a5_vectors.npz", "V7_L16", 16),
    "qwen-7b": ("a5_vectors_qwen-7b_b7/a5_vectors.npz", "V7_L21", 21),
    "dsv2-lite": ("a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz", "V7_L22", 22),
    # gemma3-27b: V7_L36, banked by the extension-pair collection
    "gemma3-27b": ("a5_vectors_gemma3-27b_b7/a5_vectors.npz", "V7_L36", 36),
    # olmo2-7b: V7_L16 built by build_olmo_entropy_gradient.py with the SAME entropy-gradient
    # construction. RAW ARM ONLY (base model, no chat template) — its constant lives in
    # raw::proc_k128, never native. Read only once the bank exists on disk.
    "olmo2-7b": ("a5_vectors_olmo2-7b_b7/a5_vectors.npz", "V7_L16", 16),
}

# Banked pairs: label -> (src, tgt, fits dir relative to the arm root). Sites come
# from each model's banked V7_L<site>.
PAIRS: dict[str, tuple[str, str, Path]] = {
    "3b->8b": ("3b", "8b", Path("fits")),
    "8b->qwen-7b": ("8b", "qwen-7b", Path("leg1") / "fits"),
    "8b->dsv2-lite": ("8b", "dsv2-lite", Path("leg2") / "fits"),
    "3b->qwen-7b": ("3b", "qwen-7b", Path("leg4e_transitivity") / "fits"),  # the closer
    "3b->dsv2-lite": ("3b", "dsv2-lite", Path("leg6") / "fits"),            # the check
}

# The system's closing equations, in solve order.
CLOSERS = ("3b->8b", "8b->qwen-7b", "3b->qwen-7b")
DERIVED = {"c_DSV2": "8b->dsv2-lite"}
CHECK_PAIR = "3b->dsv2-lite"
CHECK_FORMULA = ("c_3B", "c_DSV2")


def _load_v7(model: str, bank: Path = BANK) -> np.ndarray:
    rel, key, _ = V7[model]
    return _unit(np.load(bank / rel, allow_pickle=True)[key].astype(np.float64))


def _n_train(fits_dir: Path) -> Optional[int]:
    p = fits_dir / "cp2_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["split"].get("n_train")


def _fit_path(fits_dir: Path, src: str, s_site: int, tgt: str, t_site: int,
              arm: str, family: str) -> Path:
    return fits_dir / f"fit_{src}L{s_site}__{tgt}L{t_site}_{arm}_{family}.npz"


def _family_k(family: str) -> Optional[int]:
    return int(family.split("_k")[1]) if family.startswith("proc_k") else None


def measure_a_hats(arm_root: Path = ARM, bank: Path = BANK) -> dict:
    """a_hat for every (pair, arm, family) whose fit exists, with the rank guard applied."""
    out: dict[str, dict] = {}
    for label, (src, tgt, fits_rel) in PAIRS.items():
        fits_dir = arm_root / fits_rel
        if src not in V7 or tgt not in V7:
            continue
        try:
            v_src, v_tgt = _load_v7(src, bank), _load_v7(tgt, bank)
        except (FileNotFoundError, KeyError) as exc:
            out[label] = {"UNAVAILABLE": f"banked V7 missing: {exc}"}
            continue
        s_site, t_site = V7[src][2], V7[tgt][2]
        n_train = _n_train(fits_dir)
        max_k = (n_train / RANK_GUARD_DIVISOR) if n_train else None
        blk: dict = {"src": src, "tgt": tgt, "site_pair": f"{src}L{s_site}->{tgt}L{t_site}",
                     "fits_dir": str(fits_dir), "n_train": n_train,
                     "rank_guard_max_k": max_k, "a_hat": {}}
        for arm in ARMS:
            for family in FAMILIES:
                p = _fit_path(fits_dir, src, s_site, tgt, t_site, arm, family)
                if not p.exists():
                    continue
                k = _family_k(family)
                forbidden = bool(max_k is not None and k is not None and k > max_k)
                tm = load_transport_map(p)
                blk["a_hat"].setdefault(arm, {})[family] = {
                    "value": round(cos(_unit(tm.transport(v_src)), v_tgt), 4),
                    "rank_forbidden": forbidden,
                }
        out[label] = blk
    return out


def solve_system(a: dict[str, float]) -> Optional[dict]:
    """Solve the 4-model star from the three closers. Returns None if ill-posed."""
    try:
        a38, a8q, a3q = a["3b->8b"], a["8b->qwen-7b"], a["3b->qwen-7b"]
    except KeyError:
        return None
    if a8q <= 0 or a38 <= 0 or a3q <= 0:
        return None                      # the star is a positive-factor model
    c3 = math.sqrt(a38 * a3q / a8q)
    if c3 <= 0:
        return None
    c8 = a38 / c3
    cq = a8q / c8
    res = {"c_3B": round(c3, 4), "c_8B": round(c8, 4), "c_Qwen": round(cq, 4)}
    if "8b->dsv2-lite" in a and c8 > 0:
        res["c_DSV2"] = round(a["8b->dsv2-lite"] / c8, 4)
    return res


def build_systems(a_hats: dict) -> dict:
    """Every (arm, family) system that closes, with its out-of-sample check."""
    systems: dict[str, dict] = {}
    for arm in ARMS:
        for family in FAMILIES:
            a: dict[str, float] = {}
            forbidden: list[str] = []
            for label, blk in a_hats.items():
                cell = blk.get("a_hat", {}).get(arm, {}).get(family)
                if cell is None:
                    continue
                a[label] = cell["value"]
                if cell["rank_forbidden"]:
                    forbidden.append(label)
            const = solve_system(a)
            if const is None:
                continue
            key = f"{arm}::{family}"
            entry: dict = {
                "arm": arm, "family": family,
                "self_consistent": True,
                "inputs_a_hat": a,
                "rank_forbidden_inputs": forbidden,
                "constants": const,
                "USABLE_FOR_NEW_CONSTANTS": not forbidden,
            }
            if CHECK_PAIR in a and all(k in const for k in CHECK_FORMULA):
                pred = const[CHECK_FORMULA[0]] * const[CHECK_FORMULA[1]]
                obs = a[CHECK_PAIR]
                entry["out_of_sample_check"] = {
                    "pair": CHECK_PAIR,
                    "predicted": round(pred, 4), "observed": round(obs, 4),
                    "abs_error": round(abs(pred - obs), 4),
                    "within_pm_0.05": bool(abs(pred - obs) <= 0.05),
                    "note": "the Leg-6 fourth node, re-derived inside THIS system. A "
                            "system that fails its own check is not a system a new "
                            "constant may be hung on.",
                }
            systems[key] = entry
    return systems


def solve(arm_root: Path = ARM, bank: Path = BANK, out_dir: Path = OUT) -> Path:
    """Measure, solve, and write `<out_dir>/star_systems.json`; returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    a_hats = measure_a_hats(arm_root, bank)
    systems = build_systems(a_hats)

    res = {
        "STATUS": "UNSTAMPED (C section 8) — scores nothing; substrate for A8-add-7",
        "leg": "A8 extension-pairs smalls — star systems by (template-arm x family)",
        "arm_consistency_rule": (
            "Luxia 2026-07-23: a star constant belongs to the system it was solved in. "
            "The constants of record are NATIVE-arm. A model whose fits are raw-arm-only "
            "(OLMo: no chat template) takes its constant from the RAW-arm system below, "
            "never from the native one. A lower a_hat measured on the raw arm is "
            "ARM-CONFOUNDED before it is a transport fact — the raw arm is known to "
            "stress Procrustes (rake 13: Qwen cross-model CKA-before .47 raw vs .97 "
            "native)."),
        "estimand": "a_hat(A->B) = cos(g . V7_A, V7_B), forward direction, banked V7 "
                    "sites, per (template-arm x family) — never mixed across families",
        "rank_guard": f"add-3: k <= n_train/{RANK_GUARD_DIVISOR}, applied per pair "
                      "(n_train differs by leg: 600 hub, 360 for the Leg-4E closer)",
        "a_hats": a_hats,
        "systems": systems,
        "olmo_note": (
            "olmo2-7b has NO banked V7 (no a5 b7 build was ever run for it — the model "
            "entered the battery for A4/A1 only). a_hat(8B->OLMo) is therefore not "
            "measurable, and c_OLMo cannot be solved in ANY system, native or raw. This "
            "is a missing-target problem, not an arm problem; the raw-arm system below "
            "is where c_OLMo will live once a V7 exists."),
    }
    out_path = out_dir / "star_systems.json"
    out_path.write_text(json.dumps(res, indent=1))

    for label, blk in a_hats.items():
        if "UNAVAILABLE" in blk:
            logger.info("%-16s UNAVAILABLE: %s", label, blk["UNAVAILABLE"])
            continue
        for arm, fams in blk.get("a_hat", {}).items():
            logger.info("%-16s %-7s n_train=%s  %s", label, arm, blk["n_train"],
                        {f: v["value"] for f, v in fams.items()})
    for key, e in systems.items():
        chk = e.get("out_of_sample_check", {})
        logger.info("SYSTEM %-18s %s  usable=%s  check pred %s vs obs %s (|e|=%s)",
                    key, e["constants"], e["USABLE_FOR_NEW_CONSTANTS"],
                    chk.get("predicted"), chk.get("observed"), chk.get("abs_error"))
    logger.info("wrote %s", out_path)
    return out_path


def _plant_star_world(root: Path, constants: dict[str, float],
                      n_train: int = 600) -> tuple[Path, Path]:
    """A synthetic arm root + bank whose a_hat(A->B) is exactly c_A * c_B.

    Each model's V7_L<site> is c_m e_0 + sqrt(1 - c_m^2) e_(i+1), so the cosine of two
    models' vectors is c_A c_B; every map is the identity, so transport leaves
    that cosine unchanged. Fits are written for proc_k128 (inside the rank guard
    at n_train=600) and proc_k512 (outside it).
    """
    from metabasis.scripts.fit_transport_maps import TransportMap, save_transport_map

    models = list(constants)
    dim = len(models) + 1
    bank, arm_root = root / "bank", root / "arm"
    for i, model in enumerate(models):
        c = constants[model]
        v = np.zeros(dim)
        v[0], v[i + 1] = c, math.sqrt(1.0 - c * c)
        rel, key, _ = V7[model]
        (bank / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez(bank / rel, **{key: v})
    eye = np.eye(dim)
    for label, (src, tgt, fits_rel) in PAIRS.items():
        fits_dir = arm_root / fits_rel
        fits_dir.mkdir(parents=True, exist_ok=True)
        (fits_dir / "cp2_summary.json").write_text(
            json.dumps({"split": {"n_train": n_train}}))
        pair = f"{src}L{V7[src][2]}->{tgt}L{V7[tgt][2]}"
        for family in ("proc_k128", "proc_k512"):
            save_transport_map(fits_dir, pair, "native", family, TransportMap(
                kind="proc", src_norm=1.0, tgt_norm=1.0, va=eye, vb=eye,
                omega=eye, scale=1.0))
    return arm_root, bank


def selftest() -> int:
    """Recover planted star constants from a synthetic world; write only to a temp dir."""
    failures: list[str] = []
    n_checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal n_checks
        n_checks += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    print("== selftest 1: the star algebra on exact inputs ==")
    c = {"3b": 0.6, "8b": 0.8, "qwen-7b": 0.5, "dsv2-lite": 0.45}
    exact = {"3b->8b": c["3b"] * c["8b"], "8b->qwen-7b": c["8b"] * c["qwen-7b"],
             "3b->qwen-7b": c["3b"] * c["qwen-7b"],
             "8b->dsv2-lite": c["8b"] * c["dsv2-lite"]}
    got = solve_system(exact)
    check(got is not None and abs(got["c_3B"] - 0.6) < 1e-4
          and abs(got["c_8B"] - 0.8) < 1e-4 and abs(got["c_Qwen"] - 0.5) < 1e-4
          and abs(got["c_DSV2"] - 0.45) < 1e-4,
          f"solve_system recovers the planted constants: {got}")
    check(solve_system({**exact, "8b->qwen-7b": -0.1}) is None,
          "a non-positive closer is refused (the star is a positive-factor model)")
    check(solve_system({"3b->8b": 0.5}) is None,
          "an underdetermined system (missing closers) returns None")

    print("== selftest 2: end to end on a synthetic tree, temp dir only ==")
    with tempfile.TemporaryDirectory(prefix="star_systems_") as td:
        root = Path(td)
        arm_root, bank = _plant_star_world(root, c)
        out_dir = root / "out"
        out_path = solve(arm_root, bank, out_dir)
        doc = json.loads(out_path.read_text())
        k128 = doc["systems"].get("native::proc_k128", {})
        k512 = doc["systems"].get("native::proc_k512", {})
        consts = k128.get("constants", {})
        check(all(abs(consts.get(name, -1.0) - val) < 2e-3 for name, val in
                  (("c_3B", 0.6), ("c_8B", 0.8), ("c_Qwen", 0.5),
                   ("c_DSV2", 0.45))),
              f"the k128 system recovers the planted constants: {consts}")
        chk = k128.get("out_of_sample_check", {})
        check(chk.get("within_pm_0.05") is True and chk.get("abs_error", 1.0) < 2e-3,
              f"its out-of-sample 3b->dsv2-lite check holds: {chk.get('predicted')} "
              f"vs {chk.get('observed')}")
        check(k128.get("USABLE_FOR_NEW_CONSTANTS") is True,
              "k128 at n_train=600 is inside the rank guard and usable")
        check(k512.get("USABLE_FOR_NEW_CONSTANTS") is False
              and len(k512.get("rank_forbidden_inputs", [])) >= 3,
              "k512 at n_train=600 (> 600/1.2) is marked unusable")
        written = sorted(p.relative_to(root).as_posix() for p in out_dir.rglob("*"))
        check(written == ["out/star_systems.json"],
              f"the only file written is the temp output: {written}")

    print("== selftest 3: the command line refuses what it does not know ==")
    import contextlib
    import io
    for argv in (["--no-such-flag"], ["--selftest", "--no-such-flag"], ["extra"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                main(argv)
                code: object = None
            except SystemExit as exc:
                code = exc.code
        check(code == 2, f"{argv} is a usage error (exit 2), never ignored")

    print(f"\nselftest: {len(failures)} failure(s) of {n_checks} check(s)")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="recover planted constants from a synthetic tree in a "
                         "temporary directory; reads and writes nothing else")
    ap.add_argument("--arm-root", type=Path, default=ARM,
                    help=f"root holding the pair fit directories (default {ARM})")
    ap.add_argument("--bank-root", type=Path, default=BANK,
                    help=f"root holding the V7 vector banks (default {BANK})")
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help=f"where star_systems.json is written (default {OUT})")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    solve(args.arm_root, args.bank_root, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
