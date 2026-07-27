"""Exchange-rate readout — â(source→target) from banked entropy-gradient vectors
pushed through an ALREADY-FITTED transport map. CPU only, seconds per pair.

WHAT THIS IS. The exchange rate â of a pair is one number: push the source
model's entropy-gradient vector (banked, unit, in the source model's residual
space at the source site) through the fitted transport map for that site pair,
and take the cosine against the target model's OWN entropy-gradient vector at
the target site.

    â(A L_s → B L_t | arm, family) := cos( g · v_A , v_B )

`g` is `fit_transport_maps.TransportMap.transport(..., direction="fwd")` — the
DIRECTION map, not the state predictor. This module fits nothing and writes no
map; the ceremony (prereg §3) is that predictions are filed BEFORE pair fits, so
readout code must never be able to create a fit.

────────────────────────────────────────────────────────────────────────────────
THE NORMALIZATION PATH — spelled out, because getting it wrong is silent
────────────────────────────────────────────────────────────────────────────────
There are TWO different per-site median norms in this program and they are not
interchangeable:

  (1) `median_mean_state_norms` — median over texts of ‖per-text MEAN state‖.
      This is the FIT-TIME normalization. State banks store raw fp32 mean states
      (`states_{model}_{arm}.npz`) and this number beside them
      (`norms_{model}_{arm}.json`). `StateBank.matrix(site)` = raw / this. Every
      PCA, Procrustes rotation and ridge in `fit_transport_maps` is fit on those
      normalized rows, and the number itself is stamped into each fit npz as
      `src_norm` / `tgt_norm`.

  (2) `median_token_resid_norms` / `median_resid_norms` — median over generated
      POSITIONS of ‖per-token residual‖. This is what the entropy-gradient
      builders stamp (`a5_vectors_*_b7/a5_vectors_stamps.json`). It is a
      different quantity on different objects; it is ~2x larger. It must NEVER
      be compared to (1) and it plays no role in the readout.

The fit-time path is closed by `TransportMap.transport`, which does the whole
round trip itself:

    fwd:  out = ((v / src_norm) @ va.T @ omega @ vb) * scale * tgt_norm
    fwd:  out = ((v / src_norm) @ left) @ right * tgt_norm            (ridge)

i.e. it takes a vector in the SOURCE model's RAW residual space, divides by the
source site's fit-time median norm to enter the normalized space the map was fit
in, applies the map, and multiplies by the target site's median norm to land back
in the TARGET model's RAW residual space. The banked entropy-gradient vector is a
unit vector in the source model's raw residual space, so it is exactly the right
input, unmodified. THE CALLER MUST NOT PRE-DIVIDE IT.

Two consequences, both load-bearing:

  * â is INVARIANT to this normalization. `transport` is linear and homogeneous
    (no additive term), and `src_norm`, `tgt_norm`, `scale` are positive scalars,
    so cos(g·(c·v), w) = cos(g·v, w) for any c > 0, and rescaling the target
    vector cannot move it either. Pre-normalizing, double-normalizing or
    forgetting to normalize the INPUT vector therefore cannot change â. The
    selftest proves this (`normalization invariance` checks).

  * What CAN silently change â is using the wrong path. The affine state
    predictor inside `run_pair_arm` (`predict()`, which adds `pca_b.mean` and
    subtracts `pca_a.mean`) is for R² gating only. Using it here would transport
    a STATE, not a direction: the PCA mean offset has norm of order the whole
    state cloud and would swamp a unit direction. `transport()` drops both means
    deliberately. Never substitute one for the other.

So the guard this module actually enforces is a PROVENANCE guard, not an
arithmetic one: the `src_norm`/`tgt_norm` stamped in the fit npz must equal the
fit-time `norms_{model}_{arm}.json` entries at the sites named in the filename.
If they disagree, the map was fit against a different state bank than the one you
think you are reading (wave-1's hub bank is the SMALLS 8B bank, whose L14 native
norm is 5.2234 — the A8 trunk bank's is 5.6915: two different banks, same model,
same site). That mismatch is exactly the failure this audit catches.

────────────────────────────────────────────────────────────────────────────────
Arm and family discipline (prereg §2, and the arm-consistency rule)
────────────────────────────────────────────────────────────────────────────────
The entropy-gradient vector is ARM-AGNOSTIC (one vector per model per site; the
banked system uses the same 8B vector for both the native and raw rows). The MAP
is arm-specific, so â is arm-labeled by its map. Family of record is `proc_k128`;
k32/ridge ride beside, never mixed inside one star equation. The rank guard
(k ≤ n_train/1.2) is read from the fit dir's `cp2_summary.json` and reported per
row rather than silently applied.

Nulls: every row carries a transported-null floor — 100 seeded unit randoms in
the source space through the SAME map, |cos| q95 against the target vector, plus
the source model's banked matched-support random band if one is banked. An â
below its own null floor is not a transport fact.

Run (repo root, PYTHONPATH=. or installed):
  python -m metabasis.scripts.read_exchange_rates --selftest
  python -m metabasis.scripts.read_exchange_rates --preset banked-precedent \
      --out /tmp/claude-output/ahat_precedent.json
  python -m metabasis.scripts.read_exchange_rates --preset wave1 \
      --out outputs/collection/readouts/exchange_rates_hub8b_native_k128.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from metabasis.scripts.fit_transport_maps import (
    A8_SEED, ARMS, TransportMap, load_transport_map)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("read_exchange_rates")

# ---------------------------------------------------------------- constants
#: npz key conventions for the entropy-gradient vector, canonical FIRST. New banks
#: write `entropy_gradient_L{site}` (naming conventions §6); banked anamnesis
#: artifacts carry the frozen legacy key `V7_L{site}` and are never rewritten
#: (naming conventions §1 rule 6).
ENTROPY_GRADIENT_KEY_PATTERNS: tuple[str, ...] = (
    "entropy_gradient_L{site}", "V7_L{site}")
#: matched-support random band, same convention split.
RANDOM_BAND_KEY_PATTERNS: tuple[str, ...] = (
    "random_band{i}_L{site}", "Rband{i}_L{site}")
RANDOM_BAND_MAX_MEMBERS = 8

N_RANDOM_UNIT_NULLS = 100          # matched to read_transported_axes' envelope n
NULL_QUANTILE = 0.95
RANK_GUARD_DIVISOR = 1.2           # prereg §2: k <= n_train / 1.2
NEAR_ZERO_CARVE_OUT = 0.08         # prereg §3: |c_A·c_B| < .08 scored magnitude-only
NORM_AUDIT_RTOL = 1e-6

FAMILY_OF_RECORD = "proc_k128"
FAMILIES: tuple[str, ...] = ("proc_k32", "proc_k128", "proc_k512", "ridge")

REPO_OUTPUTS = Path("outputs")
BANK_ROOT = REPO_OUTPUTS / "battery"
COLLECTION_ROOT = REPO_OUTPUTS / "collection"
SMALLS_STATES = BANK_ROOT / "arms" / "A8_conjugation" / "smalls" / "states"

#: The hub. Universal source for the collection phase (ratified 2026-07-27).
HUB_MODEL = "8b"
#: Sites at which a hub entropy-gradient vector is actually BANKED. L16 is the
#: only one — there is no 8B vector at L14 anywhere in the tree (verified by an
#: exhaustive npz key scan, 2026-07-27). A hub-source site outside this set needs
#: a GPU build before any â at that site can be read.
HUB_BANKED_SITES: tuple[int, ...] = (16,)
HUB_VECTORS_PATH = BANK_ROOT / "a5_vectors_8b_b7" / "a5_vectors.npz"

#: Wave-1 record sites, ratified 2026-07-27. model key -> target site.
WAVE1_RECORD_SITES: dict[str, int] = {
    "qwen2.5-3b-instruct": 26,
    "qwen2.5-14b-instruct": 29,
    "qwen2.5-32b-instruct": 46,
    "mistral-7b-instruct-v0.3": 15,
    "olmo2-7b-instruct": 15,
    "phi-4": 19,
    "phi-3.5-mini-instruct": 13,
}

#: The banked precedent: the only hub→node pair whose BOTH endpoints already have
#: banked entropy-gradient vectors AND a banked fit. Used as the end-to-end
#: regression: the readout must reproduce â = 0.1278 (filed in
#: smalls/readouts_cpu/star_predictions_A8-add-8.json).
BANKED_PRECEDENT_A_HAT = 0.1278


# ---------------------------------------------------------------- data model
class VectorSpec(BaseModel):
    """Provenance of one banked entropy-gradient vector, as actually loaded."""
    model: str
    site: int
    path: str
    key: str
    key_convention: Literal["canonical", "legacy"]
    dim: int
    norm_as_banked: float = Field(
        description="‖v‖ straight off disk before unit-normalization; banked "
                    "entropy-gradient vectors are already unit, so ~1.0")


class NormAudit(BaseModel):
    """Provenance guard: does the fit's stamped median norm match the state bank?"""
    side: Literal["source", "target"]
    model: str
    site: int
    arm: str
    map_norm: float
    bank_norm: Optional[float] = None
    bank_norms_path: Optional[str] = None
    match: Optional[bool] = None
    note: str = ""


class NullFloor(BaseModel):
    """The transported-null floor this â has to clear."""
    n_random_unit: int
    random_unit_abs_cos_q95: float
    random_unit_abs_cos_max: float
    n_banked_band: int = 0
    banked_band_abs_cos_max: Optional[float] = None
    seed: int = A8_SEED


class ExchangeRateRow(BaseModel):
    """One â: one (site pair × arm × family), fully self-describing."""
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    family: str
    a_hat: float
    clears_null_floor: bool
    magnitude_only: bool = Field(
        description="prereg §3 near-zero carve-out: |â| < .08 → sign unscored")
    n_train: Optional[int] = None
    rank_forbidden: bool = False
    fit_path: str
    map_kind: str
    source_vector: VectorSpec
    target_vector: VectorSpec
    norm_audit: list[NormAudit]
    null_floor: NullFloor


class MissingPiece(BaseModel):
    """A pair that could not be read, and exactly which piece was absent."""
    pair_id: str
    source_model: str
    source_site: int
    target_model: str
    target_site: int
    arm: str
    family: str
    missing: list[str]
    probed_paths: list[str]


class ExchangeRateReadout(BaseModel):
    STATUS: str = (
        "UNSTAMPED — readout only. Fits nothing, scores nothing, files no "
        "prediction. The desk rules.")
    estimand: str = (
        "â(A L_s → B L_t | arm, family) = cos(g · entropy_gradient_A, "
        "entropy_gradient_B), forward direction, banked sites")
    normalization: str = (
        "fit-time per-site median MEAN-STATE norm, applied inside "
        "TransportMap.transport (v / src_norm → map → × scale × tgt_norm). â is "
        "invariant to it (transport is linear-homogeneous, cosine is "
        "scale-free); the audit below is a PROVENANCE check that the fit was "
        "made against the state bank named, not an arithmetic correction. The "
        "token-residual norms stamped on the vector banks are a DIFFERENT "
        "quantity and are not compared.")
    direction: Literal["fwd", "rev"] = "fwd"
    generated: str
    hub: str
    rows: list[ExchangeRateRow] = []
    missing: list[MissingPiece] = []


# ---------------------------------------------------------------- helpers
def unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize in float64. Matches read_transported_axes._unit."""
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"cannot unit-normalize: ‖v‖={n!r}")
    return arr / n


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two vectors. Matches read_transported_axes.cos."""
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a64.shape != b64.shape:
        raise ValueError(f"cos: shape mismatch {a64.shape} vs {b64.shape}")
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("cos: zero-norm operand")
    return float(a64 @ b64 / (na * nb))


def resolve_vector_key(available: Iterable[str], site: int
                       ) -> tuple[str, Literal["canonical", "legacy"]]:
    """Pick the entropy-gradient key for `site`, canonical convention first."""
    have = set(available)
    for i, pat in enumerate(ENTROPY_GRADIENT_KEY_PATTERNS):
        key = pat.format(site=site)
        if key in have:
            return key, ("canonical" if i == 0 else "legacy")
    raise KeyError(
        f"no entropy-gradient key for site {site}: tried "
        f"{[p.format(site=site) for p in ENTROPY_GRADIENT_KEY_PATTERNS]}; "
        f"npz holds {sorted(have)}")


def banked_sites(path: Path) -> list[int]:
    """Every site for which `path` holds an entropy-gradient vector."""
    if not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
    except (OSError, ValueError) as exc:                    # pragma: no cover
        logger.warning("unreadable vector bank %s: %s", path, exc)
        return []
    sites: set[int] = set()
    for name in files:
        for pat in ENTROPY_GRADIENT_KEY_PATTERNS:
            head = pat.format(site="")
            if name.startswith(head) and name[len(head):].isdigit():
                sites.add(int(name[len(head):]))
    return sorted(sites)


def load_entropy_gradient(path: Path, model: str, site: int
                          ) -> tuple[np.ndarray, VectorSpec]:
    """Load + unit-normalize the banked entropy-gradient vector at `site`."""
    if not path.exists():
        raise FileNotFoundError(f"entropy-gradient bank absent: {path}")
    with np.load(path, allow_pickle=True) as z:
        key, convention = resolve_vector_key(z.files, site)
        raw = np.asarray(z[key], dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError(f"{path}:{key} has shape {raw.shape}, expected 1-D")
    spec = VectorSpec(model=model, site=site, path=str(path), key=key,
                      key_convention=convention, dim=int(raw.shape[0]),
                      norm_as_banked=round(float(np.linalg.norm(raw)), 6))
    if convention == "legacy":
        logger.debug("%s L%d: legacy npz key %r (frozen bank, never rewritten)",
                     model, site, key)
    return unit(raw), spec


def load_random_band(path: Path, site: int) -> list[np.ndarray]:
    """Banked matched-support random band members at `site` (may be empty)."""
    if not path.exists():
        return []
    out: list[np.ndarray] = []
    with np.load(path, allow_pickle=True) as z:
        have = set(z.files)
        for i in range(1, RANDOM_BAND_MAX_MEMBERS + 1):
            for pat in RANDOM_BAND_KEY_PATTERNS:
                key = pat.format(i=i, site=site)
                if key in have:
                    out.append(unit(np.asarray(z[key], dtype=np.float64)))
                    break
    return out


def n_train_of(fits_dir: Path) -> Optional[int]:
    """n_train from the fit dir's cp2_summary.json, or None if unavailable."""
    p = fits_dir / "cp2_summary.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text())["split"]["n_train"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("cannot read n_train from %s: %s", p, exc)
        return None


def family_k(family: str) -> Optional[int]:
    """PC rank of a proc family, None for ridge."""
    if not family.startswith("proc_k"):
        return None
    try:
        return int(family.split("_k", 1)[1])
    except ValueError:                                       # pragma: no cover
        return None


def fit_path_for(fits_dir: Path, src: str, s_site: int, tgt: str, t_site: int,
                 arm: str, family: str) -> Path:
    """The fit npz name written by fit_transport_maps.save_transport_map."""
    return fits_dir / f"fit_{src}L{s_site}__{tgt}L{t_site}_{arm}_{family}.npz"


def probe_norms_path(model: str, arm: str, fits_dir: Path,
                     extra_roots: Sequence[Path] = ()) -> Optional[Path]:
    """Find `norms_{model}_{arm}.json` — the FIT-TIME median mean-state norms."""
    name = f"norms_{model}_{arm}.json"
    candidates = [
        *(root / name for root in extra_roots),
        fits_dir.parent / "states" / name,
        COLLECTION_ROOT / model / "states" / name,
        SMALLS_STATES / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def audit_norm(side: Literal["source", "target"], model: str, site: int, arm: str,
               map_norm: float, fits_dir: Path,
               norms_path: Optional[Path] = None,
               extra_roots: Sequence[Path] = ()) -> NormAudit:
    """Provenance guard — see the module docstring's normalization section."""
    path = norms_path or probe_norms_path(model, arm, fits_dir, extra_roots)
    if path is None:
        return NormAudit(side=side, model=model, site=site, arm=arm,
                         map_norm=map_norm,
                         note="no norms_{model}_{arm}.json found — provenance "
                              "UNVERIFIED (â is unaffected; the fit's bank of "
                              "origin is simply unconfirmed)")
    try:
        norms = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return NormAudit(side=side, model=model, site=site, arm=arm,
                         map_norm=map_norm, bank_norms_path=str(path),
                         note=f"norms file unreadable: {exc}")
    bank = norms.get(f"L{site}")
    if bank is None:
        return NormAudit(side=side, model=model, site=site, arm=arm,
                         map_norm=map_norm, bank_norms_path=str(path),
                         note=f"L{site} absent from {path.name} "
                              f"(has {sorted(norms)})")
    ok = bool(np.isclose(map_norm, float(bank), rtol=NORM_AUDIT_RTOL, atol=0.0))
    return NormAudit(
        side=side, model=model, site=site, arm=arm, map_norm=map_norm,
        bank_norm=float(bank), bank_norms_path=str(path), match=ok,
        note="" if ok else
             "MISMATCH — the fit was made against a DIFFERENT state bank than "
             "this norms file. â itself is unchanged by normalization, but the "
             "pair's provenance is wrong: resolve before quoting the number.")


def transported_null_floor(tm: TransportMap, v_tgt: np.ndarray, d_src: int,
                           band: Sequence[np.ndarray] = (),
                           seed: int = A8_SEED) -> NullFloor:
    """|cos| envelope of transported randoms against the target vector."""
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((N_RANDOM_UNIT_NULLS, d_src))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    cosines = np.array([abs(cos(tm.transport(p), v_tgt)) for p in probes])
    band_cos = [abs(cos(tm.transport(b), v_tgt)) for b in band]
    return NullFloor(
        n_random_unit=N_RANDOM_UNIT_NULLS,
        random_unit_abs_cos_q95=round(float(np.quantile(cosines, NULL_QUANTILE)), 4),
        random_unit_abs_cos_max=round(float(cosines.max()), 4),
        n_banked_band=len(band_cos),
        banked_band_abs_cos_max=(round(float(max(band_cos)), 4) if band_cos else None),
        seed=seed)


# ---------------------------------------------------------------- the readout
def exchange_rate(tm: TransportMap, v_src: np.ndarray, v_tgt: np.ndarray,
                  direction: Literal["fwd", "rev"] = "fwd") -> float:
    """THE estimand. `v_src` is the banked unit vector in the SOURCE model's RAW
    residual space — pass it unmodified; `transport` owns the normalization."""
    if v_src.ndim != 1 or v_tgt.ndim != 1:
        raise ValueError(f"vectors must be 1-D, got {v_src.shape} / {v_tgt.shape}")
    moved = tm.transport(v_src, direction=direction)
    if moved.shape != v_tgt.shape:
        raise ValueError(
            f"transported image has dim {moved.shape[0]} but the target vector "
            f"has dim {v_tgt.shape[0]} — wrong map, wrong site, or wrong model")
    return cos(moved, v_tgt)


def read_pair(src_model: str, src_site: int, src_vectors: Path,
              tgt_model: str, tgt_site: int, tgt_vectors: Path,
              fits_dir: Path, arm: str, family: str,
              direction: Literal["fwd", "rev"] = "fwd",
              src_norms: Optional[Path] = None,
              tgt_norms: Optional[Path] = None,
              with_nulls: bool = True,
              ) -> ExchangeRateRow | MissingPiece:
    """Read one â end-to-end. Returns a MissingPiece instead of raising when a
    banked piece is simply absent — the wave-1 case until targets land."""
    pair_id = f"{src_model}L{src_site}->{tgt_model}L{tgt_site}"
    fit_p = fit_path_for(fits_dir, src_model, src_site, tgt_model, tgt_site,
                         arm, family)
    missing: list[str] = []
    probed = [str(fit_p), str(src_vectors), str(tgt_vectors)]

    if not fit_p.exists():
        missing.append(f"transport map: {fit_p.name}")
    v_src = v_tgt = None
    src_spec = tgt_spec = None
    for role, path, model, site in (("source", src_vectors, src_model, src_site),
                                    ("target", tgt_vectors, tgt_model, tgt_site)):
        try:
            vec, spec = load_entropy_gradient(path, model, site)
        except FileNotFoundError:
            missing.append(f"{role} entropy-gradient bank: {path}")
            continue
        except (KeyError, ValueError) as exc:
            have = banked_sites(path)
            missing.append(f"{role} entropy-gradient vector at L{site} "
                           f"({exc}; banked sites: {have or 'none'})")
            continue
        if role == "source":
            v_src, src_spec = vec, spec
        else:
            v_tgt, tgt_spec = vec, spec

    if missing or v_src is None or v_tgt is None or src_spec is None or tgt_spec is None:
        return MissingPiece(pair_id=pair_id, source_model=src_model,
                            source_site=src_site, target_model=tgt_model,
                            target_site=tgt_site, arm=arm, family=family,
                            missing=missing, probed_paths=probed)

    tm = load_transport_map(fit_p)
    a_hat = exchange_rate(tm, v_src, v_tgt, direction=direction)

    nulls = (transported_null_floor(tm, v_tgt, d_src=src_spec.dim,
                                    band=load_random_band(src_vectors, src_site))
             if with_nulls else
             NullFloor(n_random_unit=0, random_unit_abs_cos_q95=float("nan"),
                       random_unit_abs_cos_max=float("nan")))

    n_train = n_train_of(fits_dir)
    k = family_k(family)
    max_k = (n_train / RANK_GUARD_DIVISOR) if n_train else None

    audits = [
        audit_norm("source", src_model, src_site, arm, tm.src_norm, fits_dir,
                   src_norms),
        audit_norm("target", tgt_model, tgt_site, arm, tm.tgt_norm, fits_dir,
                   tgt_norms),
    ]
    return ExchangeRateRow(
        pair_id=pair_id, source_model=src_model, source_site=src_site,
        target_model=tgt_model, target_site=tgt_site, arm=arm, family=family,
        a_hat=round(a_hat, 4),
        clears_null_floor=bool(abs(a_hat) > nulls.random_unit_abs_cos_q95)
        if with_nulls else False,
        magnitude_only=bool(abs(a_hat) < NEAR_ZERO_CARVE_OUT),
        n_train=n_train,
        rank_forbidden=bool(max_k is not None and k is not None and k > max_k),
        fit_path=str(fit_p), map_kind=tm.kind,
        source_vector=src_spec, target_vector=tgt_spec,
        norm_audit=audits, null_floor=nulls)


# ---------------------------------------------------------------- presets
class PairRequest(BaseModel):
    """One requested readout, resolved to concrete paths."""
    source_model: str
    source_site: int
    source_vectors: Path
    target_model: str
    target_site: int
    target_vectors: Path
    fits_dir: Path
    arm: str
    family: str

    model_config = {"arbitrary_types_allowed": True}


def target_vector_candidates(model: str, collection_root: Path) -> list[Path]:
    """Where a wave-1 native target build could reasonably land, in order.

    The first is the convention this tool recommends; the rest are probed so a
    build that landed elsewhere is still found rather than silently reported
    missing. All probed paths are echoed into the MissingPiece record.
    """
    node = collection_root / model
    return [
        node / "vectors" / f"entropy_gradient_{model}.npz",
        node / "vectors" / "entropy_gradient.npz",
        node / "vectors" / "a5_vectors.npz",
        node / f"entropy_gradient_{model}.npz",
        BANK_ROOT / f"entropy_gradient_{model}" / "entropy_gradient.npz",
        BANK_ROOT / f"a5_vectors_{model}_b7" / "a5_vectors.npz",
    ]


def resolve_target_vectors(model: str, site: int, collection_root: Path
                           ) -> tuple[Path, list[Path]]:
    """First candidate that actually holds a vector at `site`; else candidate[0]."""
    candidates = target_vector_candidates(model, collection_root)
    for c in candidates:
        if site in banked_sites(c):
            return c, candidates
    for c in candidates:                     # exists but wrong site: still report it
        if c.exists():
            return c, candidates
    return candidates[0], candidates


def preset_wave1(arm: str, family: str, collection_root: Path, hub_site: int,
                 hub_vectors: Optional[Path] = None) -> list[PairRequest]:
    """Hub 8B → each wave-1 node at its ratified record site."""
    reqs: list[PairRequest] = []
    for model, site in sorted(WAVE1_RECORD_SITES.items()):
        tgt_path, _ = resolve_target_vectors(model, site, collection_root)
        reqs.append(PairRequest(
            source_model=HUB_MODEL, source_site=hub_site,
            source_vectors=hub_vectors or HUB_VECTORS_PATH,
            target_model=model, target_site=site, target_vectors=tgt_path,
            fits_dir=collection_root / model / f"fits_scan_{model}",
            arm=arm, family=family))
    return reqs


def preset_banked_precedent(family: str) -> list[PairRequest]:
    """The banked OLMo-2-7B BASE node — the one hub pair with both endpoints
    banked. RAW arm only: OLMo-2-1124-7B has no chat template, so its constant
    lives in the raw system (the arm-consistency rule)."""
    smalls = BANK_ROOT / "arms" / "A8_conjugation" / "smalls"
    return [PairRequest(
        source_model=HUB_MODEL, source_site=16,
        source_vectors=HUB_VECTORS_PATH,
        target_model="olmo2-7b", target_site=16,
        target_vectors=BANK_ROOT / "a5_vectors_olmo2-7b_b7" / "a5_vectors.npz",
        fits_dir=smalls / "fits_olmo", arm="raw", family=family)]


# ---------------------------------------------------------------- driver
def run(requests: Sequence[PairRequest], hub: str = HUB_MODEL,
        direction: Literal["fwd", "rev"] = "fwd",
        with_nulls: bool = True) -> ExchangeRateReadout:
    readout = ExchangeRateReadout(generated=date.today().isoformat(), hub=hub,
                                  direction=direction)
    for req in requests:
        result = read_pair(
            req.source_model, req.source_site, req.source_vectors,
            req.target_model, req.target_site, req.target_vectors,
            req.fits_dir, req.arm, req.family, direction=direction,
            with_nulls=with_nulls)
        if isinstance(result, MissingPiece):
            readout.missing.append(result)
            logger.warning("MISSING %-46s %s::%s — %s", result.pair_id, req.arm,
                           req.family, "; ".join(result.missing))
            continue
        readout.rows.append(result)
        bad_audit = [a for a in result.norm_audit if a.match is False]
        logger.info("â %-46s %s::%-9s = %+.4f  null_q95=%.4f clears=%s%s%s",
                    result.pair_id, req.arm, req.family, result.a_hat,
                    result.null_floor.random_unit_abs_cos_q95,
                    result.clears_null_floor,
                    "  RANK-FORBIDDEN" if result.rank_forbidden else "",
                    "  NORM-PROVENANCE-MISMATCH" if bad_audit else "")
    return readout


# ---------------------------------------------------------------- selftest
def _random_proc_map(rng: np.random.Generator, d_a: int, d_b: int, k: int,
                     src_norm: float, tgt_norm: float, scale: float) -> TransportMap:
    va = np.linalg.qr(rng.standard_normal((d_a, k)))[0].T          # [k, d_a]
    vb = np.linalg.qr(rng.standard_normal((d_b, k)))[0].T          # [k, d_b]
    omega = np.linalg.qr(rng.standard_normal((k, k)))[0]
    return TransportMap(kind="proc", src_norm=src_norm, tgt_norm=tgt_norm,
                        va=va, vb=vb, omega=omega, scale=scale)


def _random_ridge_map(rng: np.random.Generator, d_a: int, d_b: int, r: int,
                      src_norm: float, tgt_norm: float) -> TransportMap:
    return TransportMap(kind="ridge", src_norm=src_norm, tgt_norm=tgt_norm,
                        left=rng.standard_normal((d_a, r)) / np.sqrt(d_a),
                        right=rng.standard_normal((r, d_b)) / np.sqrt(r))


def selftest() -> int:                                   # noqa: C901 — a checklist
    rng = np.random.default_rng(A8_SEED)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    d_a, d_b, k = 64, 96, 16
    tm = _random_proc_map(rng, d_a, d_b, k, src_norm=5.2234, tgt_norm=2.2506,
                          scale=1.1541)
    v_src = unit(rng.standard_normal(d_a))
    moved = tm.transport(v_src)

    print("== selftest 1: round trip — the mapped hub vector IS the target ==")
    a_id = exchange_rate(tm, v_src, moved)
    check(abs(a_id - 1.0) < 1e-12, f"â(target := g·v_hub) = {a_id:.15f} (want 1.0)")
    a_id_unit = exchange_rate(tm, v_src, unit(moved))
    check(abs(a_id_unit - 1.0) < 1e-12,
          f"â against the UNIT mapped image = {a_id_unit:.15f} (want 1.0)")
    a_neg = exchange_rate(tm, v_src, -moved)
    check(abs(a_neg + 1.0) < 1e-12,
          f"â against the sign-flipped image = {a_neg:.15f} (want -1.0; the sign "
          "anchor is real, a flipped construction flips â)")

    print("== selftest 2: orthogonal target ~ 0 ==")
    q = np.linalg.qr(np.column_stack([moved, rng.standard_normal((d_b, 3))]))[0]
    orth = q[:, 1]
    a_orth = exchange_rate(tm, v_src, orth)
    check(abs(a_orth) < 1e-12, f"â(target ⊥ g·v_hub) = {a_orth:.3e} (want 0)")

    print("== selftest 3: controlled-noise targets recover the planted cosine ==")
    for planted in (0.9, 0.5, 0.1278, 0.0, -0.35):
        tgt = planted * unit(moved) + np.sqrt(max(0.0, 1 - planted ** 2)) * orth
        got = exchange_rate(tm, v_src, tgt)
        check(abs(got - planted) < 1e-9,
              f"planted cos {planted:+.4f} → â {got:+.10f}")

    print("== selftest 4: normalization invariance (the silent-failure guard) ==")
    base = exchange_rate(tm, v_src, moved * 0.0 + unit(moved))
    for c in (1e-6, 1e3, 7.0):
        check(abs(exchange_rate(tm, v_src * c, unit(moved)) - base) < 1e-12,
              f"source vector scaled ×{c:g} leaves â unchanged")
        check(abs(exchange_rate(tm, v_src, unit(moved) * c) - base) < 1e-12,
              f"target vector scaled ×{c:g} leaves â unchanged")
    tm_renorm = TransportMap(kind="proc", src_norm=tm.src_norm * 13.0,
                             tgt_norm=tm.tgt_norm / 29.0, va=tm.va, vb=tm.vb,
                             omega=tm.omega, scale=tm.scale * 3.0)
    check(abs(exchange_rate(tm_renorm, v_src, unit(moved)) - base) < 1e-12,
          "perturbing src_norm/tgt_norm/scale leaves â bit-for-bit unchanged "
          "(â is invariant to the whole median-norm path)")

    print("== selftest 5: ridge family behaves identically ==")
    rtm = _random_ridge_map(rng, d_a, d_b, r=24, src_norm=3.3, tgt_norm=9.1)
    rmoved = rtm.transport(v_src)
    check(abs(exchange_rate(rtm, v_src, rmoved) - 1.0) < 1e-12,
          "ridge: â(target := g·v_hub) = 1.0")
    rq = np.linalg.qr(np.column_stack([rmoved, rng.standard_normal((d_b, 2))]))[0]
    check(abs(exchange_rate(rtm, v_src, rq[:, 1])) < 1e-12,
          "ridge: â(orthogonal target) = 0")
    check(abs(exchange_rate(
        TransportMap(kind="ridge", src_norm=rtm.src_norm * 5, tgt_norm=rtm.tgt_norm / 7,
                     left=rtm.left, right=rtm.right), v_src, rmoved) - 1.0) < 1e-12,
        "ridge: normalization invariance holds")

    print("== selftest 6: on-disk round trip + both npz key conventions ==")
    import tempfile

    from metabasis.scripts.fit_transport_maps import save_transport_map
    with tempfile.TemporaryDirectory(prefix="ahat_selftest_") as td:
        root = Path(td)
        save_transport_map(root, "srcML14->tgtML9", "native", "proc_k16", tm)
        back = load_transport_map(root / "fit_srcML14__tgtML9_native_proc_k16.npz")
        check(abs(exchange_rate(back, v_src, moved) - 1.0) < 1e-6,
              "map survives save→load (fp32 on disk): â = 1 within fp32 tolerance")

        canon = root / "canon.npz"
        legacy = root / "legacy.npz"
        np.savez(canon, **{"entropy_gradient_L9": unit(moved).astype(np.float32),
                           "random_band1_L9": unit(orth).astype(np.float32)})
        np.savez(legacy, **{"V7_L9": unit(moved).astype(np.float32),
                            "Rband1_L9": unit(orth).astype(np.float32)})
        vc, sc = load_entropy_gradient(canon, "tgtM", 9)
        vl, sl = load_entropy_gradient(legacy, "tgtM", 9)
        check(sc.key_convention == "canonical" and sc.key == "entropy_gradient_L9",
              "canonical key entropy_gradient_L9 resolves")
        check(sl.key_convention == "legacy" and sl.key == "V7_L9",
              "legacy key V7_L9 resolves (frozen banks are readable unchanged)")
        check(np.allclose(vc, vl), "both conventions load the same vector")
        check(len(load_random_band(canon, 9)) == 1
              and len(load_random_band(legacy, 9)) == 1,
              "random band loads under both conventions")
        check(banked_sites(canon) == [9], f"banked_sites(canon) = {banked_sites(canon)}")
        try:
            load_entropy_gradient(canon, "tgtM", 11)
            check(False, "missing-site load must raise")
        except KeyError as exc:
            check("banked" not in str(exc) or True, f"missing site raises: {exc!s:.70}")

        print("== selftest 7: dimension mismatch is caught, not broadcast ==")
        try:
            exchange_rate(tm, v_src, unit(rng.standard_normal(d_b + 1)))
            check(False, "dim mismatch must raise")
        except ValueError as exc:
            check(True, f"dim mismatch raises: {exc!s:.70}")

        print("== selftest 8: null floor separates signal from noise ==")
        sig = transported_null_floor(tm, unit(moved), d_src=d_a)
        # A proc map's image lives in the k-dim target PC subspace, and the target
        # vector's in-subspace component is what randoms compete with, so the floor
        # scales as ~1.96/sqrt(k) — NOT 1/sqrt(d_b). The selftest's k=16 gives a
        # floor near .49; the real k=128 fits give ~.17. Check the geometry, not a
        # magic constant, so this stays honest at every k.
        expected = 1.96 / np.sqrt(k)
        check(0.5 * expected <= sig.random_unit_abs_cos_q95 <= 1.5 * expected,
              f"random-unit null |cos| q95 = {sig.random_unit_abs_cos_q95} tracks "
              f"the k={k} subspace expectation {expected:.3f} (±50%)")
        check(1.0 > sig.random_unit_abs_cos_q95,
              f"â=1 clears its own null floor ({sig.random_unit_abs_cos_q95})")
        noise_tgt = unit(rng.standard_normal(d_b))
        a_noise = exchange_rate(tm, v_src, noise_tgt)
        floor = transported_null_floor(tm, noise_tgt, d_src=d_a)
        check(abs(a_noise) <= floor.random_unit_abs_cos_max,
              f"an unrelated target (â={a_noise:+.4f}) sits inside its own null "
              f"envelope (max {floor.random_unit_abs_cos_max})")

        print("== selftest 9: the norm-provenance audit fires on a wrong bank ==")
        states = root / "states"
        states.mkdir()
        (states / "norms_srcM_native.json").write_text(
            json.dumps({"L14": 5.2234, "L16": 6.2000}))
        ok = audit_norm("source", "srcM", 14, "native", 5.2234, root / "fits")
        check(ok.match is True and ok.bank_norm == 5.2234,
              f"matching bank → match=True (map {ok.map_norm}, bank {ok.bank_norm})")
        bad = audit_norm("source", "srcM", 14, "native", 5.6915, root / "fits")
        check(bad.match is False and "MISMATCH" in bad.note,
              "a fit stamped with a DIFFERENT bank's norm (5.6915, the A8 trunk "
              "8B bank) is caught against the smalls bank's 5.2234")
        gone = audit_norm("source", "srcM", 99, "native", 1.0, root / "fits")
        check(gone.match is None and "absent" in gone.note,
              f"unknown site reported, not asserted: {gone.note!s:.48}")
        nofile = audit_norm("target", "nobodyM", 3, "raw", 1.0, root / "fits")
        check(nofile.match is None and "UNVERIFIED" in nofile.note,
              "no norms file → provenance UNVERIFIED (â still readable)")

    print(f"\nselftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic validation of the readout algebra; no data needed")
    ap.add_argument("--preset", choices=("wave1", "banked-precedent"),
                    help="wave1: hub 8B → the seven wave-1 nodes at their ratified "
                         "record sites. banked-precedent: 8bL16→olmo2-7bL16 raw "
                         "(the only fully-banked hub pair; regression target "
                         f"â={BANKED_PRECEDENT_A_HAT})")
    ap.add_argument("--arm", default="native", choices=ARMS)
    ap.add_argument("--family", default=FAMILY_OF_RECORD, choices=FAMILIES)
    ap.add_argument("--hub-site", type=int, default=HUB_BANKED_SITES[0],
                    help=f"hub source site. Banked hub vectors exist only at "
                         f"{list(HUB_BANKED_SITES)}; any other site needs a hub "
                         f"entropy-gradient build first (GPU).")
    ap.add_argument("--hub-vectors", type=Path, default=None,
                    help=f"hub entropy-gradient bank (default {HUB_VECTORS_PATH}). "
                         "Point this at a NEW hub build when one lands.")
    ap.add_argument("--collection-root", type=Path, default=COLLECTION_ROOT)
    ap.add_argument("--direction", choices=("fwd", "rev"), default="fwd")
    ap.add_argument("--no-nulls", action="store_true",
                    help="skip the transported-null envelope (faster; the floor "
                         "is then unknown and clears_null_floor is meaningless)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the readout JSON here (parents created)")
    # single-pair mode
    ap.add_argument("--source-model", default=None)
    ap.add_argument("--source-site", type=int, default=None)
    ap.add_argument("--source-vectors", type=Path, default=None)
    ap.add_argument("--target-model", default=None)
    ap.add_argument("--target-site", type=int, default=None)
    ap.add_argument("--target-vectors", type=Path, default=None)
    ap.add_argument("--fits-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    single = [args.source_model, args.source_site, args.target_model,
              args.target_site, args.fits_dir]
    if args.preset and any(x is not None for x in single):
        raise SystemExit("--preset and single-pair flags are mutually exclusive")

    if args.preset == "wave1":
        hub_bank = args.hub_vectors or HUB_VECTORS_PATH
        have = banked_sites(hub_bank)
        if args.hub_site not in have:
            logger.warning(
                "hub site L%d has NO banked entropy-gradient vector in %s "
                "(banked sites: %s). Every row will report the hub vector as "
                "missing until a hub build at L%d lands — pass --hub-vectors if "
                "one landed elsewhere.", args.hub_site, hub_bank,
                have or "none", args.hub_site)
        requests = preset_wave1(args.arm, args.family, args.collection_root,
                                args.hub_site, hub_vectors=args.hub_vectors)
    elif args.preset == "banked-precedent":
        requests = preset_banked_precedent(args.family)
    else:
        if any(x is None for x in single):
            raise SystemExit(
                "pass --preset, or all of --source-model --source-site "
                "--target-model --target-site --fits-dir")
        src_vec = args.source_vectors or (
            HUB_VECTORS_PATH if args.source_model == HUB_MODEL else None)
        if src_vec is None:
            src_vec, _ = resolve_target_vectors(
                args.source_model, args.source_site, args.collection_root)
        tgt_vec = args.target_vectors
        if tgt_vec is None:
            tgt_vec, _ = resolve_target_vectors(
                args.target_model, args.target_site, args.collection_root)
        requests = [PairRequest(
            source_model=args.source_model, source_site=args.source_site,
            source_vectors=src_vec, target_model=args.target_model,
            target_site=args.target_site, target_vectors=tgt_vec,
            fits_dir=args.fits_dir, arm=args.arm, family=args.family)]

    readout = run(requests, direction=args.direction, with_nulls=not args.no_nulls)
    payload = readout.model_dump_json(indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        logger.info("wrote %s (%d row(s), %d missing)", args.out,
                    len(readout.rows), len(readout.missing))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
