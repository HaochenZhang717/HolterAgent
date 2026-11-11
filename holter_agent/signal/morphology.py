"""Compare QRS morphology of two templates.

A change of QRS shape together with an ST shift points to a non-ischemic cause
(axis or body-position shift, conduction change, electrode problem). Comparing
amplitude alone misses polarity flips, so shape correlation, amplitude ratio,
polarity and the frontal QRS axis are all reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MorphologyConfig
from .templates import Template


@dataclass
class MorphologyComparison:
    leads: list[str]
    lag_ms: float  # shift applied to the second template for alignment
    qrs_corr: np.ndarray  # [L] Pearson r of the QRS window
    pp_ratio: np.ndarray  # [L] peak-to-peak (b / a)
    pp_a_uv: np.ndarray  # [L] peak-to-peak of the compared QRS window, uV
    pp_b_uv: np.ndarray
    polarity_flip: np.ndarray  # bool [L] dominant QRS deflection changed sign
    qrs_area_a_uvs: np.ndarray  # [L] net QRS area, uV*s
    qrs_area_b_uvs: np.ndarray
    qrs_ms_a: float
    qrs_ms_b: float
    axis_a_deg: float  # frontal QRS axis from net areas in I and aVF (NaN if unavailable)
    axis_b_deg: float
    changed_leads: list[str]

    @property
    def axis_change_deg(self) -> float:
        if not (np.isfinite(self.axis_a_deg) and np.isfinite(self.axis_b_deg)):
            return float("nan")
        return float((self.axis_b_deg - self.axis_a_deg + 180.0) % 360.0 - 180.0)


def frontal_axis_deg(net_area: np.ndarray, leads: list[str]) -> float:
    """Frontal QRS axis estimated from net QRS areas of leads I and aVF."""
    if "I" not in leads or "aVF" not in leads:
        return float("nan")
    a_i = net_area[leads.index("I")]
    a_f = net_area[leads.index("aVF")]
    if not (np.isfinite(a_i) and np.isfinite(a_f)) or (a_i == 0 and a_f == 0):
        return float("nan")
    return float(np.degrees(np.arctan2(a_f, a_i)))


def _best_lag(a: np.ndarray, b: np.ndarray, lo: int, hi: int, max_lag: int) -> int:
    """Lag (samples) of b that best matches a over [lo, hi), using all leads."""
    best, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lo + lag < 0 or hi + lag > len(b):
            continue
        x = np.nan_to_num(a[lo:hi]).ravel()
        y = np.nan_to_num(b[lo + lag:hi + lag]).ravel()
        score = float(np.dot(x - x.mean(), y - y.mean()))
        if score > best_score:
            best, best_score = lag, score
    return best


def compare(a: Template, b: Template, cfg: MorphologyConfig) -> MorphologyComparison:
    if a.leads != b.leads or a.fs != b.fs or a.r_index != b.r_index:
        raise ValueError("templates must share leads, sampling rate and layout")
    fs = a.fs
    ms = 1000.0 / fs
    fa, fb = a.fiducials, b.fiducials
    on = np.nanmin([fa.onset_ms, fb.onset_ms]) - 10.0
    off = np.nanmin([fa.j_ms, fb.j_ms])  # QRS only: an ST shift must not enter the QRS comparison (0.8)
    if not (np.isfinite(on) and np.isfinite(off)):
        on, off = -60.0, 80.0
    lo, hi = a.index_at(on), a.index_at(off)
    lag = _best_lag(a.data, b.data, lo, hi, int(round(cfg.max_align_ms / ms)))
    qa = a.data[lo:hi].astype(np.float64)
    qb = b.data[lo + lag:hi + lag].astype(np.float64)
    n_leads = qa.shape[1]
    corr = np.full(n_leads, np.nan)
    for j in range(n_leads):
        x, y = qa[:, j], qb[:, j]
        if np.isfinite(x).all() and np.isfinite(y).all() and x.std() > 0 and y.std() > 0:
            corr[j] = np.corrcoef(x, y)[0, 1]
    pp_a = np.ptp(qa, axis=0)
    pp_b = np.ptp(qb, axis=0)
    with np.errstate(all="ignore"):
        pp_ratio = pp_b / pp_a
    dom_a = qa[np.nanargmax(np.abs(np.nan_to_num(qa)), axis=0), np.arange(n_leads)]
    dom_b = qb[np.nanargmax(np.abs(np.nan_to_num(qb)), axis=0), np.arange(n_leads)]
    flip = np.sign(dom_a) != np.sign(dom_b)
    area_a = np.nansum(qa, axis=0) * ms  # mV*ms == uV*s
    area_b = np.nansum(qb, axis=0) * ms
    lo_r, hi_r = cfg.pp_ratio_range
    changed = (corr < cfg.min_qrs_corr) | (pp_ratio < lo_r) | (pp_ratio > hi_r) | flip
    changed &= np.isfinite(corr)
    return MorphologyComparison(
        leads=list(a.leads),
        lag_ms=lag * ms,
        qrs_corr=corr.astype(np.float32),
        pp_ratio=pp_ratio.astype(np.float32),
        pp_a_uv=(pp_a * 1000.0).astype(np.float32),
        pp_b_uv=(pp_b * 1000.0).astype(np.float32),
        polarity_flip=flip,
        qrs_area_a_uvs=area_a.astype(np.float32),
        qrs_area_b_uvs=area_b.astype(np.float32),
        qrs_ms_a=fa.qrs_ms,
        qrs_ms_b=fb.qrs_ms,
        axis_a_deg=frontal_axis_deg(area_a, a.leads),
        axis_b_deg=frontal_axis_deg(area_b, b.leads),
        changed_leads=[lead for lead, c in zip(a.leads, changed) if c],
    )
