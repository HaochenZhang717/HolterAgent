"""Sudden steps in ST level and in QRS morphology.

A step in ST level that coincides with a step in QRS morphology points to an axis
or body-position shift, or to a conduction change, rather than ischemia. In the
LTSTDB, axis shifts are "characterised by a change in the Q-, R- or S-wave
amplitude" and conduction changes "by bizarre and wider QRS complexes"
(Jager et al., Med Biol Eng Comput 2003;41:172-182).

Step rule of Smrdel & Jager (Computers in Cardiology 2002;29:325-328): a flat
stretch of 216 s (mean absolute deviation below 10 uV for the ST level, below 0.33
for the distance function), a change of the 72-s moving average by at least 50 uV
(ST) or 0.5 (distance) within the next 144 s, then another flat stretch.

The distance mimics the LTSTDB "QRS complex KL Mahalanobis distance using the
first 5 KL coefficients (units of 1 SD)": principal components of the per-window
QRS features (net area and peak-to-peak of every lead), fitted on the record's own
usable windows. Its scale differs from the LTSTDB one, so the 0.33/0.5 thresholds
are a starting point to be calibrated on our data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .episodes import nan_moving_mean
from .features import FeatureTable

FLAT_S = 216.0
CHANGE_S = 144.0
MA_S = 72.0
ST_FLAT_UV, ST_STEP_UV = 10.0, 50.0
DIST_FLAT, DIST_STEP = 0.33, 0.5
N_COMPONENTS = 5


@dataclass
class Step:
    series: str  # "QRS_distance" or "ST:<lead>"
    t_start_s: float  # transition interval
    t_end_s: float
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


def qrs_distance(table: FeatureTable) -> np.ndarray:
    """Mahalanobis distance (SD units) of each window's QRS features in the record's first 5 PCs."""
    x = np.concatenate([table["qrs_area_uvs"], table["qrs_pp_uv"]], axis=1).astype(np.float64)
    ok = table["fiducials_ok"] & np.isfinite(x).all(axis=1)
    out = np.full(len(x), np.nan)
    if ok.sum() < 3 * N_COMPONENTS:
        return out
    med = np.median(x[ok], axis=0)
    mad = 1.4826 * np.median(np.abs(x[ok] - med), axis=0)
    mad[mad <= 0] = 1.0
    z = (x - med) / mad
    zc = z[ok] - z[ok].mean(axis=0)
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    k = min(N_COMPONENTS, len(s))
    scores = (z[ok] - z[ok].mean(axis=0)) @ vt[:k].T
    sd = scores.std(axis=0)
    sd[sd <= 0] = 1.0
    out[ok] = np.sqrt(((scores / sd) ** 2).sum(axis=1) / k)  # RMS over components, SD units
    return out


def detect_steps(y: np.ndarray, t: np.ndarray, window_s: float, flat_thr: float, step_thr: float,
                 name: str) -> list[Step]:
    """Flat -> change within CHANGE_S -> flat, on a window series (NaN = missing)."""
    n_flat = max(2, int(round(FLAT_S / window_s)))
    n_change = max(1, int(round(CHANGE_S / window_s)))
    ma = nan_moving_mean(y[:, None], max(1, int(round(MA_S / window_s))))[:, 0]
    hits: list[tuple[int, float, float, float]] = []
    for k in range(n_flat, len(y) - n_change - n_flat + 1):
        a = y[k - n_flat:k]
        b = y[k + n_change:k + n_change + n_flat]
        if np.isfinite(a).mean() < 0.8 or np.isfinite(b).mean() < 0.8:
            continue
        ma_a, mb_b = np.nanmean(a), np.nanmean(b)
        if np.nanmean(np.abs(a - ma_a)) >= flat_thr or np.nanmean(np.abs(b - mb_b)) >= flat_thr:
            continue
        mid = ma[k:k + n_change + 1]
        if not np.isfinite(mid).any() or np.nanmax(np.abs(mid - ma_a)) < step_thr:
            continue
        if abs(mb_b - ma_a) < step_thr:
            continue
        hits.append((k, ma_a, mb_b, abs(mb_b - ma_a)))
    steps: list[Step] = []
    i = 0
    while i < len(hits):  # keep the largest step in each run of neighbouring hits
        j = i
        while j + 1 < len(hits) and hits[j + 1][0] - hits[j][0] <= n_change:
            j += 1
        k, before, after, _ = max(hits[i:j + 1], key=lambda h: h[3])
        steps.append(Step(name, float(t[k]), float(t[min(len(t) - 1, k + n_change)]), float(before), float(after)))
        i = j + 1
    return steps


def frontal_axis_series(table: FeatureTable) -> np.ndarray:
    """Frontal QRS axis (degrees) per window from net QRS areas of leads I and aVF."""
    if "I" not in table.leads or "aVF" not in table.leads:
        return np.full(table.n_windows, np.nan)
    a_i = table["qrs_area_uvs"][:, table.lead_index("I")]
    a_f = table["qrs_area_uvs"][:, table.lead_index("aVF")]
    return np.degrees(np.arctan2(a_f, a_i))


def record_steps(table: FeatureTable) -> dict[str, list[Step]]:
    """All steps of a record: the QRS distance and the ST level of every lead."""
    t = table["t_start_s"]
    ws = table.window_s
    out = {"QRS_distance": detect_steps(qrs_distance(table), t, ws, DIST_FLAT, DIST_STEP, "QRS_distance")}
    st = np.where(table["usable"], table["st_uv"], np.nan)
    for j, lead in enumerate(table.leads):
        out[f"ST:{lead}"] = detect_steps(st[:, j], t, ws, ST_FLAT_UV, ST_STEP_UV, f"ST:{lead}")
    return out
