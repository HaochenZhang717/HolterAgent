"""QRS and ST-T shape over time, and the shape of the ST segment of an averaged beat.

Two record-level shape spaces are built from the per-window feature table, one for the QRS complex
(per-lead QRS area and peak-to-peak amplitude) and one for the ST-T segment (per-lead ST at J, ST at
the measurement point, ST slope and T amplitude). Each is reduced to its first principal components
and scaled to SD units, the way LTSTDB tracks QRS and ST-segment Karhunen-Loeve coefficients:
"Axis shifts ... are usually most apparent in the QRS complexes ... when ischemic ST changes occur,
they are most apparent in the principal components of the ST segment" (PhysioNet LTSTDB page).

The ST segment of an averaged beat is described by its first three Legendre coefficients (level,
linear rise, curvature), after Amon & Jager (PLoS One 2016;11:e0148814), who used Legendre
polynomials because their shapes match the level, slope and scooping changes of ischaemic ST
morphology. They fitted a 120-ms segment starting 40 ms after their QRS fiducial point; here the
120 ms after our J point are used.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre

from .features import FeatureTable

N_COMPONENTS = 5
ST_SEGMENT_MS = 120.0
ST_SEGMENT_POINTS = 32

QRS_FEATURES = ("qrs_area_uvs", "qrs_pp_uv")
ST_T_FEATURES = ("st_j0_uv", "st_uv", "st_slope", "t_amp_uv")


def shape_scores(table: FeatureTable, features: tuple[str, ...], k: int = N_COMPONENTS) -> np.ndarray:
    """[W, k] principal-component scores in SD units (NaN for windows without usable fiducials)."""
    x = np.concatenate([table[f] for f in features], axis=1).astype(np.float64)
    ok = table["fiducials_ok"] & np.isfinite(x).all(axis=1)
    k = min(k, x.shape[1])  # a 2-lead record has 4 QRS features, so at most 4 components
    out = np.full((len(x), k), np.nan)
    if ok.sum() < 3 * k:
        return out
    med = np.median(x[ok], axis=0)
    mad = 1.4826 * np.median(np.abs(x[ok] - med), axis=0)
    mad[mad <= 0] = 1.0
    z = (x[ok] - med) / mad
    zc = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    scores = zc @ vt[:k].T
    sd = scores.std(axis=0)
    sd[sd <= 0] = 1.0
    out[np.flatnonzero(ok), :k] = scores / sd
    return out


def shape_change(scores: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance (SD units) between the median shape of windows `a` and of windows `b`."""
    sa, sb = scores[a & np.isfinite(scores).all(1)], scores[b & np.isfinite(scores).all(1)]
    if len(sa) == 0 or len(sb) == 0:
        return float("nan")
    d = np.median(sb, axis=0) - np.median(sa, axis=0)
    return float(np.sqrt(np.mean(d ** 2)))


def largest_jump(scores: np.ndarray, sel: np.ndarray, lag: int) -> float:
    """Largest RMS shape change between windows `lag` apart within `sel` (SD units)."""
    idx = np.flatnonzero(sel)
    best = np.nan
    for i in idx:
        j = i + lag
        if j < len(scores) and sel[j] and np.isfinite(scores[i]).all() and np.isfinite(scores[j]).all():
            d = float(np.sqrt(np.mean((scores[j] - scores[i]) ** 2)))
            best = d if not np.isfinite(best) else max(best, d)
    return best


def typical_jump(scores: np.ndarray, lag: int) -> float:
    """Median RMS shape change between windows `lag` apart over the whole record (SD units)."""
    a, b = scores[:-lag], scores[lag:]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if not ok.any():
        return float("nan")
    return float(np.median(np.sqrt(np.mean((b[ok] - a[ok]) ** 2, axis=1))))


def st_segment_shape(data_mv: np.ndarray, fs: float, j_index: int, iso_mv: np.ndarray) -> np.ndarray:
    """[L, 3] level, linear rise and curvature (uV) of the 120-ms segment after J of each lead.

    The segment, relative to the isoelectric level, is fitted with Legendre polynomials P0..P2 on
    [-1, 1]: level = mean value; rise = 2*c1 (linear change from J to J+120 ms); curvature = 1.5*c2
    (mean of the two ends minus the middle; positive = concave upward, e.g. a sagging or scooped ST).
    """
    n = int(round(ST_SEGMENT_MS / 1000.0 * fs))
    if j_index < 0 or j_index + n >= len(data_mv):
        return np.full((data_mv.shape[1], 3), np.nan)
    seg = data_mv[j_index:j_index + n + 1].astype(np.float64) - iso_mv[None, :]
    x_src = np.linspace(-1.0, 1.0, len(seg))
    x = np.linspace(-1.0, 1.0, ST_SEGMENT_POINTS)
    out = np.full((seg.shape[1], 3), np.nan)
    for lead in range(seg.shape[1]):
        y = np.interp(x, x_src, seg[:, lead])
        if not np.isfinite(y).all():
            continue
        c = legendre.legfit(x, y, 2)
        out[lead] = [c[0] * 1000.0, 2.0 * c[1] * 1000.0, 1.5 * c[2] * 1000.0]
    return out
