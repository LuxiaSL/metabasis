"""â-CEILING diagnostic — the hygiene layer under every exchange rate.

WHAT THIS IS. `read_exchange_rates` answers "how much of the target model's
entropy-gradient direction does the transported hub vector recover?". It does
NOT answer "how much of that direction was recoverable AT ALL through this
map?". A rank-k Procrustes map's image is a k-dimensional subspace of the
target model's residual space; anything of the target vector living outside
that subspace is invisible to â by construction. This module measures that:

    ceiling(pair, family) := ‖P_k v_target‖      (v_target unit)

where P_k is the orthogonal projector onto the map's TARGET-side image
subspace (`vb` rows for proc, the row space of `right` for ridge). It is a
hard cap: no vector the map can emit has |cos| to v_target above it.

────────────────────────────────────────────────────────────────────────────────
THE ALGEBRA — why the ceiling is exactly ‖vb·v_target‖, and what it is not
────────────────────────────────────────────────────────────────────────────────
For a proc map, `TransportMap.transport` computes

    out = ((v / src_norm) @ va.T @ omega @ vb) * scale * tgt_norm

Write a := va·v (the source vector's coordinates in the source PC basis). Then
`out` is, up to positive scalars, vbᵀ(ωᵀa) — i.e. it lives in span(vb rows),
always. With ‖v_target‖ = 1 and b := vb·v_target,

    â = cos(out, v_target) = cos(ωᵀa, b) · ‖b‖ = cos(ωᵀa, b) · ceiling

so **â/ceiling is the in-subspace alignment**, a genuine cosine in [-1, 1], and
â ≤ ceiling always. This factorization is the whole point: it separates "the
map cannot see this direction" from "the map sees it and it does not match".

Two things the ceiling is NOT:

  * It is NOT a function of the source vector. ‖a‖ cancels in the cosine, so a
    source vector poorly captured by `va` does not lower the ceiling. It does
    make â NOISIER (the surviving coordinates are a small, possibly
    noise-dominated slice of the source direction), so this module reports
    `source_capture` := ‖va·v_source‖ **beside** the ceiling as a reliability
    flag, never multiplied into it.
  * It is NOT the same object as build coherence. The ceiling is a property of
    the MAP + the target direction. Coherence
    (`per_text_band_pairwise_coherence`) is a property of the target vector's
    CONSTRUCTION: the mean pairwise cosine between per-text gradient estimates.
    Low coherence means the banked vector is the axis of a wide cone rather
    than a ray, which deflates â mechanically without touching the ceiling.

────────────────────────────────────────────────────────────────────────────────
THE INTERPRETATION CONTRACT (brief Question 2, ruled 2026-07-27)
────────────────────────────────────────────────────────────────────────────────
Given (â, ceiling, coherence, null q95), a low â sorts into exactly one of:

  RANK-LIMITED     low â, LOW ceiling. The map cannot see the target direction.
                   Not evidence against the law — evidence the map is too
                   narrow. Fix: larger k, or a better-matched site pair.
  DIFFUSE-TARGET   low â, high ceiling, LOW coherence. The target vector is a
                   cone axis; â is deflated by the construction's own spread.
                   An upper bound on what any map could score is set by how
                   reproducible the target direction is at all.
  CONSTRUCTION     â moves materially when the SAME site is re-built through a
                   different builder lineage. The number is measuring the
                   builder, not the models. Diagnosed by the lineage deconfound,
                   not by this table alone.
  GENUINE          low â, HIGH ceiling, HIGH coherence, â at/below its null
                   floor. The direction is readable and reproducible and it
                   simply does not correspond. This is the only bucket that is
                   evidence against the law.

This module reports the numbers and applies the contract as a SUGGESTED bucket
(`bucket_hint`). It never rules; the desk does.

Run (repo root, PYTHONPATH=.):
  python -m metabasis.scripts.read_ahat_ceilings --selftest
  python -m metabasis.scripts.read_ahat_ceilings --preset all \
      --out outputs/collection/readouts/ahat_ceilings_k128.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from metabasis.scripts.fit_transport_maps import (
    A8_SEED, ARMS, TransportMap, load_transport_map)
from metabasis.scripts.read_exchange_rates import (
    BANK_ROOT, COLLECTION_ROOT, FAMILIES, FAMILY_OF_RECORD, HUB_MODEL,
    ExchangeRateRow, MissingPiece, PairRequest, cos, exchange_rate,
    fit_path_for, load_entropy_gradient, read_pair, unit)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("read_ahat_ceilings")

#: Rows of a PCA basis should be orthonormal to this tolerance (fp32 on disk).
ORTHONORMAL_ATOL = 2e-5
#: Singular values below this fraction of the largest are dropped when
#: orthonormalizing a non-orthonormal image basis (the ridge case).
RANK_TOL = 1e-8

#: Interpretation-contract thresholds. Reported alongside every hint so the
#: desk can move them without re-deriving anything.
CEILING_LOW = 0.35
COHERENCE_LOW = 0.45


# ---------------------------------------------------------------- data model
class HubSource(BaseModel):
    """One hub source vector: a (site, bank) pair with a short column label."""
    label: str
    site: int
    vectors: Path
    lineage: Literal["new-builder", "legacy"]

    model_config = {"arbitrary_types_allowed": True}


class TargetSpec(BaseModel):
    """One target node at its site of record."""
    model: str
    site: int
    arm: str
    vectors: Path
    fits_dir: Path
    #: Where cp2_summary.json actually lives. Usually `fits_dir`, but a ruled
    #: scan EXTENSION writes its merged summary into a sibling dir while the
    #: fit npz's stay in the original (pythia-6.9b's L28–L31 extension).
    summary_dir: Optional[Path] = None
    stamps: Optional[Path] = None
    depth_fraction: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


class BuildDiagnostics(BaseModel):
    """The target vector's CONSTRUCTION quality, straight off its stamps json."""
    stamps_path: Optional[str] = None
    coherence: Optional[float] = Field(
        default=None,
        description="per_text_band_pairwise_coherence (legacy banks: "
                    "per_gen_band_pairwise_coherence) — mean pairwise cosine "
                    "between per-text gradient estimates")
    band_energy_fraction: Optional[float] = None
    sign_consistency: Optional[str] = None
    n_grad_texts: Optional[int] = None
    fd_gate_passes: Optional[bool] = None
    note: str = ""


class FitQuality(BaseModel):
    """The map's own scan-fit record — how good the map is, independent of â."""
    r2: Optional[float] = None
    r2_null_shuffled_q95: Optional[float] = None
    r2_null_stratum_q95: Optional[float] = None
    cka_before: Optional[float] = None
    cka_after: Optional[float] = None
    k_effective: Optional[int] = None
    pca_explained_tgt: Optional[float] = None
    pca_explained_src: Optional[float] = None
    two_arm_g_mean_cos: Optional[float] = None
    two_arm_g_min_cos: Optional[float] = None
    note: str = ""


class CeilingRow(BaseModel):
    """One â with its ceiling and everything needed to bucket it."""
    pair_id: str
    column: str = Field(description="hub source column label (L14/legacyL16/…)")
    source_model: str
    source_site: int
    source_lineage: str
    target_model: str
    target_site: int
    arm: str
    family: str

    a_hat: float
    null_q95: float
    clears_null_floor: bool
    magnitude_only: bool

    ceiling: float = Field(description="‖P_k v_target‖, the hard cap on |â|")
    a_over_ceiling: float = Field(
        description="in-subspace alignment cos(ωᵀa, vb·v_target) ∈ [-1,1]")
    source_capture: float = Field(
        description="‖va·v_source‖ — reliability flag, NOT part of the cap")
    ceiling_random_q95: float = Field(
        description="‖P_k u‖ q95 for random unit u in target space — the "
                    "ceiling a DIRECTIONLESS target would post (≈√(k/d))")
    ceiling_excess: float = Field(
        description="ceiling − ceiling_random_q95; >0 means the target "
                    "direction is preferentially inside the map's image")

    build: BuildDiagnostics
    fit: FitQuality

    map_kind: str
    image_rank: int
    fit_path: str
    source_vector_path: str
    source_vector_key: str
    target_vector_path: str
    target_vector_key: str
    norm_audit_ok: Optional[bool] = None
    n_train: Optional[int] = None
    rank_forbidden: bool = False

    bucket_hint: Literal["clears", "rank-limited", "diffuse-target",
                         "genuine-negative", "inconclusive"]
    bucket_reason: str


class CeilingReadout(BaseModel):
    STATUS: str = (
        "UNSTAMPED — diagnostic only. Fits nothing, scores nothing, files no "
        "prediction, picks no site. The desk rules.")
    estimand: str = (
        "ceiling(pair, family) = ‖P_k v_target‖ through the map's target-side "
        "image subspace; â/ceiling = in-subspace alignment")
    contract: str = (
        "low â + LOW ceiling = rank-limitation artifact (not evidence against "
        "the law); low â + high ceiling + LOW coherence = diffuse target (the "
        "target is a cone, not a ray); low â + high ceiling + high coherence + "
        "â inside its null floor = genuine negative. source_capture is a "
        "reliability flag, never multiplied into the cap.")
    thresholds: dict[str, float] = {
        "ceiling_low": CEILING_LOW, "coherence_low": COHERENCE_LOW}
    generated: str
    rows: list[CeilingRow] = []
    missing: list[MissingPiece] = []


# ---------------------------------------------------------------- geometry
def image_basis(tm: TransportMap) -> np.ndarray:
    """Orthonormal ROWS spanning the image of `tm.transport(·, 'fwd')`.

    proc: span(vb rows) exactly — `out = (…) @ vb`, so every emitted vector is
    a row-combination of vb. vb comes from an SVD and is already orthonormal;
    this is asserted, not assumed (fp32 round-trip through the npz).
    ridge: span(right rows), which is NOT orthonormal — orthonormalized here.
    """
    if tm.kind == "proc":
        if tm.vb is None:
            raise ValueError("proc map missing vb — cannot form an image basis")
        basis = np.asarray(tm.vb, dtype=np.float64)
        gram = basis @ basis.T
        off = float(np.abs(gram - np.eye(basis.shape[0])).max())
        if off > ORTHONORMAL_ATOL:
            logger.warning(
                "vb rows deviate from orthonormal by %.2e (> %.1e) — "
                "re-orthonormalizing before projecting", off, ORTHONORMAL_ATOL)
            return _orthonormal_rows(basis)
        return basis
    if tm.right is None:
        raise ValueError("ridge map missing `right` — cannot form an image basis")
    return _orthonormal_rows(np.asarray(tm.right, dtype=np.float64))


def _orthonormal_rows(m: np.ndarray) -> np.ndarray:
    """Orthonormal rows spanning the row space of `m`, rank-truncated."""
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        raise ValueError("degenerate basis: no positive singular value")
    keep = int((s > RANK_TOL * s[0]).sum())
    return vt[:keep]


def projection_norm(basis: np.ndarray, v: np.ndarray) -> float:
    """‖P v‖/‖v‖ for the orthonormal-row `basis`. Unit-safe, dimension-checked."""
    b = np.asarray(basis, dtype=np.float64)
    x = unit(v)
    if b.ndim != 2 or b.shape[1] != x.shape[0]:
        raise ValueError(
            f"projection_norm: basis {b.shape} cannot project a vector of dim "
            f"{x.shape[0]} — wrong map, wrong site, or wrong model")
    val = float(np.linalg.norm(b @ x))
    # numerical guard: an orthogonal projection of a unit vector is in [0, 1].
    if not np.isfinite(val) or val < -1e-9 or val > 1.0 + 1e-6:
        raise ValueError(f"projection_norm out of range: {val!r}")
    return min(max(val, 0.0), 1.0)


def random_ceiling_q95(basis: np.ndarray, dim: int, n: int = 200,
                       seed: int = A8_SEED) -> float:
    """‖P u‖ q95 for random unit u — what a directionless target would post.

    For a k-dim subspace of R^d this concentrates at √(k/d); reported so the
    ceiling can be read as excess over chance rather than as a bare number.
    """
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((n, dim))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    return float(np.quantile(np.linalg.norm(probes @ basis.T, axis=1), 0.95))


# ---------------------------------------------------------------- provenance
_COHERENCE_KEYS: tuple[str, ...] = (
    "per_text_band_pairwise_coherence", "per_gen_band_pairwise_coherence")
_SIGN_KEYS: tuple[str, ...] = (
    "per_text_band_sign_consistency", "per_gen_band_sign_consistency")


def read_build_diagnostics(stamps: Optional[Path]) -> BuildDiagnostics:
    """Construction quality off a stamps json; absence is reported, not raised."""
    if stamps is None:
        return BuildDiagnostics(note="no stamps path supplied for this vector")
    if not stamps.exists():
        return BuildDiagnostics(stamps_path=str(stamps),
                                note="stamps json absent — coherence UNKNOWN")
    try:
        doc = json.loads(stamps.read_text())
    except (OSError, ValueError) as exc:
        return BuildDiagnostics(stamps_path=str(stamps),
                                note=f"stamps json unreadable: {exc}")
    diag = doc.get("diagnostics") or {}
    if not isinstance(diag, dict):
        return BuildDiagnostics(stamps_path=str(stamps),
                                note="stamps json has no diagnostics block")
    coh = next((float(diag[k]) for k in _COHERENCE_KEYS
                if isinstance(diag.get(k), (int, float))), None)
    sign = next((str(diag[k]) for k in _SIGN_KEYS if diag.get(k) is not None), None)
    band = diag.get("band_energy_fraction")
    n_grad = diag.get("n_grad_texts")
    gate = doc.get("fd_gate") or {}
    return BuildDiagnostics(
        stamps_path=str(stamps),
        coherence=coh,
        band_energy_fraction=float(band) if isinstance(band, (int, float)) else None,
        sign_consistency=sign,
        n_grad_texts=int(n_grad) if isinstance(n_grad, int) else None,
        fd_gate_passes=(bool(gate["PASSES"]) if isinstance(gate, dict)
                        and "PASSES" in gate else None),
        note="" if coh is not None else
             "no coherence diagnostic in this bank (legacy stamps predate it)")


def read_fit_quality(fits_dir: Path, site_pair: str, arm: str, family: str
                     ) -> FitQuality:
    """The scan-fit record for this exact cell, from cp2_summary.json."""
    summary = fits_dir / "cp2_summary.json"
    if not summary.exists():
        return FitQuality(note=f"cp2_summary.json absent under {fits_dir}")
    try:
        doc = json.loads(summary.read_text())
    except (OSError, ValueError) as exc:
        return FitQuality(note=f"cp2_summary.json unreadable: {exc}")
    rec = next((r for r in doc.get("records", [])
                if r.get("site_pair") == site_pair and r.get("arm") == arm
                and r.get("family") == family), None)
    agree = ((doc.get("two_arm_g_agreement") or {}).get(site_pair) or {}).get(family) or {}
    if rec is None:
        return FitQuality(
            two_arm_g_mean_cos=agree.get("mean_cos"),
            two_arm_g_min_cos=agree.get("min_cos"),
            note=f"no scan record for {site_pair} {arm} {family} in {summary}")
    detail = rec.get("detail") or {}
    return FitQuality(
        r2=rec.get("r2"), r2_null_shuffled_q95=rec.get("r2_null_shuffled_q95"),
        r2_null_stratum_q95=rec.get("r2_null_stratum_q95"),
        cka_before=rec.get("cka_before"), cka_after=rec.get("cka_after"),
        k_effective=detail.get("k_effective"),
        pca_explained_tgt=detail.get("pca_explained_tgt"),
        pca_explained_src=detail.get("pca_explained_src"),
        two_arm_g_mean_cos=agree.get("mean_cos"),
        two_arm_g_min_cos=agree.get("min_cos"))


# ---------------------------------------------------------------- bucketing
def bucket(a_hat: float, ceiling: float, coherence: Optional[float],
           clears: bool) -> tuple[str, str]:
    """Apply the interpretation contract. Suggestion only — the desk rules."""
    if clears and a_hat > 0:
        return "clears", (
            f"â={a_hat:+.4f} clears its null floor; ceiling {ceiling:.3f} "
            f"leaves in-subspace alignment {a_hat / ceiling:+.3f}")
    if ceiling < CEILING_LOW:
        return "rank-limited", (
            f"ceiling {ceiling:.3f} < {CEILING_LOW}: the map's image cannot "
            f"see most of the target direction, so â={a_hat:+.4f} is capped by "
            "the map's rank, not by the models. NOT evidence against the law.")
    if coherence is None:
        return "inconclusive", (
            f"ceiling {ceiling:.3f} is high but the target vector's build "
            "coherence is unknown (no stamps diagnostic) — cannot separate a "
            "diffuse target from a genuine negative.")
    if coherence < COHERENCE_LOW:
        return "diffuse-target", (
            f"ceiling {ceiling:.3f} is high but build coherence {coherence:.3f} "
            f"< {COHERENCE_LOW}: the banked target is a cone axis, which "
            f"deflates â={a_hat:+.4f} mechanically.")
    return "genuine-negative", (
        f"ceiling {ceiling:.3f} high, coherence {coherence:.3f} high, and "
        f"â={a_hat:+.4f} does not clear its null floor — the direction is "
        "readable and reproducible and does not correspond.")


# ---------------------------------------------------------------- the readout
def ceiling_row(hub: HubSource, tgt: TargetSpec, family: str,
                collection_root: Path = COLLECTION_ROOT
                ) -> CeilingRow | MissingPiece:
    """One (hub column × target) row, â and ceiling computed side by side."""
    base = read_pair(
        HUB_MODEL, hub.site, hub.vectors, tgt.model, tgt.site, tgt.vectors,
        tgt.fits_dir, tgt.arm, family)
    if isinstance(base, MissingPiece):
        return base
    assert isinstance(base, ExchangeRateRow)

    fit_p = fit_path_for(tgt.fits_dir, HUB_MODEL, hub.site, tgt.model, tgt.site,
                         tgt.arm, family)
    tm = load_transport_map(fit_p)
    v_src, _ = load_entropy_gradient(hub.vectors, HUB_MODEL, hub.site)
    v_tgt, _ = load_entropy_gradient(tgt.vectors, tgt.model, tgt.site)

    basis = image_basis(tm)
    ceil = projection_norm(basis, v_tgt)
    if ceil < 1e-9:
        raise ValueError(
            f"{base.pair_id}: target vector is numerically orthogonal to the "
            f"map's image (ceiling {ceil:.3e}) — â carries no information")
    if abs(base.a_hat) > ceil + 1e-3:
        raise ValueError(
            f"{base.pair_id}: |â|={abs(base.a_hat):.4f} exceeds its own "
            f"ceiling {ceil:.4f} — the ceiling algebra is wrong for this map "
            f"kind ({tm.kind}); refusing to report")

    src_basis = (np.asarray(tm.va, dtype=np.float64) if tm.kind == "proc"
                 else _orthonormal_rows(np.asarray(tm.left, dtype=np.float64).T))
    capture = projection_norm(src_basis, v_src)

    site_pair = f"{HUB_MODEL}L{hub.site}->{tgt.model}L{tgt.site}"
    build = read_build_diagnostics(tgt.stamps)
    fitq = read_fit_quality(tgt.summary_dir or tgt.fits_dir, site_pair, tgt.arm,
                            family)
    hint, why = bucket(base.a_hat, ceil, build.coherence, base.clears_null_floor)
    audits = [a.match for a in base.norm_audit if a.match is not None]

    return CeilingRow(
        pair_id=base.pair_id, column=hub.label, source_model=HUB_MODEL,
        source_site=hub.site, source_lineage=hub.lineage,
        target_model=tgt.model, target_site=tgt.site, arm=tgt.arm, family=family,
        a_hat=base.a_hat, null_q95=base.null_floor.random_unit_abs_cos_q95,
        clears_null_floor=base.clears_null_floor,
        magnitude_only=base.magnitude_only,
        ceiling=round(ceil, 4), a_over_ceiling=round(base.a_hat / ceil, 4),
        source_capture=round(capture, 4),
        ceiling_random_q95=round(
            random_ceiling_q95(basis, dim=int(v_tgt.shape[0])), 4),
        ceiling_excess=round(
            ceil - random_ceiling_q95(basis, dim=int(v_tgt.shape[0])), 4),
        build=build, fit=fitq, map_kind=tm.kind, image_rank=int(basis.shape[0]),
        fit_path=str(fit_p),
        source_vector_path=base.source_vector.path,
        source_vector_key=base.source_vector.key,
        target_vector_path=base.target_vector.path,
        target_vector_key=base.target_vector.key,
        norm_audit_ok=(all(audits) if audits else None),
        n_train=base.n_train, rank_forbidden=base.rank_forbidden,
        bucket_hint=hint, bucket_reason=why)  # type: ignore[arg-type]


def run(hubs: Sequence[HubSource], targets: Sequence[TargetSpec], family: str
        ) -> CeilingReadout:
    out = CeilingReadout(generated=date.today().isoformat())
    for tgt in targets:
        for hub in hubs:
            try:
                res = ceiling_row(hub, tgt, family)
            except (OSError, ValueError, KeyError) as exc:
                logger.error("FAILED %s L%d ← hub %s: %s", tgt.model, tgt.site,
                             hub.label, exc)
                out.missing.append(MissingPiece(
                    pair_id=f"{HUB_MODEL}L{hub.site}->{tgt.model}L{tgt.site}",
                    source_model=HUB_MODEL, source_site=hub.site,
                    target_model=tgt.model, target_site=tgt.site, arm=tgt.arm,
                    family=family, missing=[str(exc)],
                    probed_paths=[str(tgt.fits_dir), str(tgt.vectors)]))
                continue
            if isinstance(res, MissingPiece):
                logger.warning("MISSING %s [%s] — %s", res.pair_id, hub.label,
                               "; ".join(res.missing))
                out.missing.append(res)
                continue
            out.rows.append(res)
            logger.info(
                "%-42s [%-10s] â=%+.4f ceil=%.3f â/ceil=%+.3f cap=%.3f "
                "coh=%s q95=%.4f %s", res.pair_id, res.column, res.a_hat,
                res.ceiling, res.a_over_ceiling, res.source_capture,
                f"{res.build.coherence:.3f}" if res.build.coherence is not None
                else "  n/a", res.null_q95, res.bucket_hint)
    return out


# ---------------------------------------------------------------- roster
def hub_sources() -> list[HubSource]:
    """The three hub source columns of the hub-site analysis."""
    return [
        HubSource(label="L14", site=14, lineage="new-builder",
                  vectors=COLLECTION_ROOT / "8b" / "vectors"
                  / "entropy_gradient_8b.npz"),
        HubSource(label="legacyL16", site=16, lineage="legacy",
                  vectors=BANK_ROOT / "a5_vectors_8b_b7" / "a5_vectors.npz"),
        HubSource(label="rebuiltL16", site=16, lineage="new-builder",
                  vectors=COLLECTION_ROOT / "8b" / "vectors"
                  / "entropy_gradient_8b_L16rebuild.npz"),
    ]


def _node(model: str, site: int, arm: str, n_layers: int,
          fits_dirname: Optional[str] = None,
          summary_dirname: Optional[str] = None,
          vectors_name: Optional[str] = None) -> TargetSpec:
    root = COLLECTION_ROOT / model
    vec = root / "vectors" / (vectors_name or f"entropy_gradient_{model}.npz")
    stem = vec.name[:-len(".npz")]
    return TargetSpec(
        model=model, site=site, arm=arm, vectors=vec,
        fits_dir=root / (fits_dirname or f"fits_scan_{model}"),
        summary_dir=(root / summary_dirname) if summary_dirname else None,
        stamps=root / "vectors" / f"{stem}_stamps.json",
        depth_fraction=round(site / n_layers, 4))


def roster_wave1() -> list[TargetSpec]:
    """The seven wave-1 nodes at their ratified record sites, native arm."""
    return [
        _node("qwen2.5-3b-instruct", 26, "native", 36),
        _node("qwen2.5-14b-instruct", 29, "native", 48),
        _node("qwen2.5-32b-instruct", 46, "native", 64),
        _node("mistral-7b-instruct-v0.3", 15, "native", 32),
        _node("olmo2-7b-instruct", 15, "native", 32),
        _node("phi-4", 19, "native", 40),
        _node("phi-3.5-mini-instruct", 13, "native", 32),
    ]


def roster_rungs2() -> list[TargetSpec]:
    """Hub rungs 2: the 70B, pythia, and all three gpt2-xl sites of evidence."""
    gpt2_fits = "fits_scan_gpt2-xl-25site"
    return [
        _node("llama-3.1-70b-instruct", 17, "native", 80),
        # pythia's ruled L28–L31 extension banked its npz's in the original
        # fits dir and its MERGED 16-site cp2_summary.json in the sibling.
        _node("pythia-6.9b", 31, "raw", 32,
              summary_dirname="fits_scan_pythia-6.9b-16site"),
        _node("gpt2-xl", 7, "raw", 48, fits_dirname=gpt2_fits),
        _node("gpt2-xl", 26, "raw", 48, fits_dirname=gpt2_fits),
        _node("gpt2-xl", 47, "raw", 48, fits_dirname=gpt2_fits),
    ]


def roster_precedent() -> list[TargetSpec]:
    """The banked precedent — OLMo-2-7B BASE, raw arm, smalls fits tree."""
    smalls = BANK_ROOT / "arms" / "A8_conjugation" / "smalls"
    return [TargetSpec(
        model="olmo2-7b", site=16, arm="raw",
        vectors=BANK_ROOT / "a5_vectors_olmo2-7b_b7" / "a5_vectors.npz",
        fits_dir=smalls / "fits_olmo",
        stamps=BANK_ROOT / "a5_vectors_olmo2-7b_b7" / "a5_vectors_stamps.json",
        depth_fraction=round(16 / 32, 4))]


ROSTERS: dict[str, list[TargetSpec]] = {}


def roster(name: str) -> list[TargetSpec]:
    if name == "wave1":
        return roster_wave1()
    if name == "rungs2":
        return roster_rungs2()
    if name == "precedent":
        return roster_precedent()
    if name == "all":
        return roster_wave1() + roster_rungs2() + roster_precedent()
    raise ValueError(f"unknown preset {name!r}")


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    """Synthetic validation of the ceiling algebra. No data needed."""
    rng = np.random.default_rng(A8_SEED)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    d_a, d_b, k = 64, 96, 16
    va = np.linalg.qr(rng.standard_normal((d_a, k)))[0].T
    vb = np.linalg.qr(rng.standard_normal((d_b, k)))[0].T
    omega = np.linalg.qr(rng.standard_normal((k, k)))[0]
    tm = TransportMap(kind="proc", src_norm=5.2234, tgt_norm=2.2506, va=va,
                      vb=vb, omega=omega, scale=1.1541)
    v_src = unit(rng.standard_normal(d_a))
    moved = tm.transport(v_src)
    basis = image_basis(tm)

    print("== selftest 1: the brief's round trip — mapped hub vector ==")
    check(basis.shape == (k, d_b), f"image basis is [{k}, {d_b}]: {basis.shape}")
    ceil_moved = projection_norm(basis, moved)
    a_moved = exchange_rate(tm, v_src, moved)
    check(abs(ceil_moved - 1.0) < 1e-12,
          f"ceiling of the mapped hub vector = {ceil_moved:.15f} (want 1.0 — it "
          "lies exactly in the image)")
    check(ceil_moved >= a_moved - 1e-12,
          f"ceiling {ceil_moved:.12f} >= â {a_moved:.12f} (the cap holds at the "
          "boundary case â = 1)")

    print("== selftest 2: the brief's orthogonal case ==")
    #  a direction orthogonal to EVERY row of vb — outside the image entirely.
    full = np.linalg.qr(rng.standard_normal((d_b, d_b)))[0]
    resid = full - vb.T @ (vb @ full)
    col = int(np.argmax(np.linalg.norm(resid, axis=0)))
    orth_to_image = unit(resid[:, col])
    ceil_orth = projection_norm(basis, orth_to_image)
    check(ceil_orth < 1e-10,
          f"ceiling of a target orthogonal to the image = {ceil_orth:.3e} (want 0)")
    check(abs(exchange_rate(tm, v_src, orth_to_image)) < 1e-10,
          "â against that same target is 0 — a zero ceiling forces a zero â")

    print("== selftest 3: planted ceilings are recovered exactly ==")
    in_img = unit(moved)
    for planted in (0.95, 0.5, 0.2, 0.05):
        tgt = planted * in_img + np.sqrt(1 - planted ** 2) * orth_to_image
        got = projection_norm(basis, tgt)
        check(abs(got - planted) < 1e-10,
              f"planted ceiling {planted:.2f} → measured {got:.12f}")

    print("== selftest 4: â = ceiling × in-subspace alignment (the factorization) ==")
    for _ in range(5):
        w = unit(rng.standard_normal(d_b))
        c = projection_norm(basis, w)
        a = exchange_rate(tm, v_src, w)
        inside = cos(basis @ unit(moved), basis @ w)
        check(abs(a - c * inside) < 1e-10,
              f"â {a:+.6f} == ceiling {c:.6f} × in-subspace cos {inside:+.6f}")
        check(abs(a) <= c + 1e-12, f"|â| {abs(a):.6f} <= ceiling {c:.6f}")

    print("== selftest 5: ceiling is INDEPENDENT of the source vector ==")
    w = unit(rng.standard_normal(d_b))
    c0 = projection_norm(basis, w)
    for _ in range(3):
        v2 = unit(rng.standard_normal(d_a))
        check(abs(projection_norm(image_basis(tm), w) - c0) < 1e-14,
              "a different source vector leaves the ceiling bit-identical")
        check(abs(exchange_rate(tm, v2, w)) <= c0 + 1e-12,
              f"…and its â still respects the same cap {c0:.6f}")

    print("== selftest 6: ceiling is scale- and normalization-invariant ==")
    tm2 = TransportMap(kind="proc", src_norm=tm.src_norm * 13.0,
                       tgt_norm=tm.tgt_norm / 29.0, va=va, vb=vb, omega=omega,
                       scale=tm.scale * 3.0)
    check(abs(projection_norm(image_basis(tm2), w) - c0) < 1e-14,
          "perturbing src_norm/tgt_norm/scale leaves the ceiling unchanged")
    check(abs(projection_norm(basis, w * 7.0) - c0) < 1e-12,
          "rescaling the target vector leaves the ceiling unchanged")

    print("== selftest 7: random-target ceiling concentrates at √(k/d) ==")
    q95 = random_ceiling_q95(basis, dim=d_b)
    expect = np.sqrt(k / d_b)
    check(0.7 * expect <= q95 <= 1.4 * expect,
          f"random-unit ceiling q95 {q95:.4f} tracks √(k/d) = {expect:.4f}")

    print("== selftest 8: ridge maps get a correct (non-orthonormal) basis ==")
    r = 24
    rtm = TransportMap(kind="ridge", src_norm=3.3, tgt_norm=9.1,
                       left=rng.standard_normal((d_a, r)) / np.sqrt(d_a),
                       right=rng.standard_normal((r, d_b)) / np.sqrt(r))
    rbasis = image_basis(rtm)
    check(rbasis.shape == (r, d_b), f"ridge image basis is [{r}, {d_b}]: {rbasis.shape}")
    check(float(np.abs(rbasis @ rbasis.T - np.eye(r)).max()) < 1e-10,
          "ridge image basis rows are orthonormal after re-orthonormalization")
    rmoved = rtm.transport(v_src)
    check(abs(projection_norm(rbasis, rmoved) - 1.0) < 1e-10,
          "ridge: ceiling of the mapped hub vector = 1.0")
    for _ in range(3):
        w2 = unit(rng.standard_normal(d_b))
        check(abs(exchange_rate(rtm, v_src, w2)) <= projection_norm(rbasis, w2) + 1e-10,
              "ridge: |â| respects its ceiling")

    print("== selftest 9: rank-deficient image is truncated, not inflated ==")
    right = rng.standard_normal((r, d_b))
    right[10:] = right[:r - 10][:len(right[10:])]        # duplicate rows → rank < r
    dtm = TransportMap(kind="ridge", src_norm=1.0, tgt_norm=1.0,
                       left=rng.standard_normal((d_a, r)), right=right)
    dbasis = image_basis(dtm)
    check(dbasis.shape[0] < r,
          f"duplicated rows give rank {dbasis.shape[0]} < r={r} (not padded)")
    check(abs(projection_norm(dbasis, dtm.transport(v_src)) - 1.0) < 1e-9,
          "…and the mapped vector still sits exactly inside the truncated image")

    print("== selftest 10: dimension mismatch raises, never broadcasts ==")
    try:
        projection_norm(basis, unit(rng.standard_normal(d_b + 1)))
        check(False, "dim mismatch must raise")
    except ValueError as exc:
        check(True, f"dim mismatch raises: {exc!s:.72}")

    print("== selftest 11: the interpretation contract routes correctly ==")
    for a, c, coh, clr, want in (
            (0.40, 0.85, 0.70, True, "clears"),
            (0.05, 0.20, 0.70, False, "rank-limited"),
            (0.05, 0.85, 0.30, False, "diffuse-target"),
            (0.05, 0.85, 0.70, False, "genuine-negative"),
            (0.05, 0.85, None, False, "inconclusive")):
        got, _ = bucket(a, c, coh, clr)
        check(got == want, f"(â={a}, ceil={c}, coh={coh}, clears={clr}) → {got}")

    print(f"\nselftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preset", choices=("wave1", "rungs2", "precedent", "all"),
                    default="all")
    ap.add_argument("--family", default=FAMILY_OF_RECORD, choices=FAMILIES)
    ap.add_argument("--columns", default="L14,legacyL16,rebuiltL16",
                    help="comma-separated hub source column labels to read")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    wanted = [c.strip() for c in args.columns.split(",") if c.strip()]
    hubs = [h for h in hub_sources() if h.label in wanted]
    unknown = sorted(set(wanted) - {h.label for h in hubs})
    if unknown:
        raise SystemExit(f"unknown hub column(s) {unknown}; known: "
                         f"{[h.label for h in hub_sources()]}")
    if not hubs:
        raise SystemExit("no hub columns selected")

    readout = run(hubs, roster(args.preset), args.family)
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
