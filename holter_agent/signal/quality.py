"""Per-lead signal quality of a window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import QualityConfig
from ..data.record import Segment


@dataclass
class LeadQuality:
    leads: list[str]
    saturation_frac: np.ndarray  # [L] fraction of samples at the ADC rail
    flat_frac: np.ndarray  # [L] fraction of samples inside flat runs
    noise_uv: np.ndarray  # [L] template residual (NaN if no template)
    wander_uv: np.ndarray  # [L] spread of isoelectric levels (NaN if no template)
    usable: np.ndarray  # bool [L]
    reasons: list[list[str]]  # per lead, why it is not usable


def flat_fraction(x: np.ndarray, min_run: int) -> np.ndarray:
    """Fraction of samples in runs of >= min_run identical values, per column."""
    out = np.zeros(x.shape[1], dtype=np.float32)
    if x.shape[0] < min_run:
        return out
    same = np.diff(x, axis=0) == 0
    for j in range(x.shape[1]):
        col = np.concatenate([[False], same[:, j], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(col))
        lengths = edges[1::2] - edges[::2] + 1  # a run of k equal diffs spans k+1 samples
        out[j] = lengths[lengths >= min_run].sum() / x.shape[0]
    return out


def assess(
    segment: Segment,
    noise_uv: np.ndarray | None,
    wander_uv: np.ndarray | None,
    cfg: QualityConfig,
) -> LeadQuality:
    n_leads = segment.signal.shape[1]
    sat = segment.saturated.mean(axis=0).astype(np.float32)
    flat = flat_fraction(segment.signal, int(round(cfg.flat_run_ms * segment.fs / 1000.0)))
    noise = np.full(n_leads, np.nan, np.float32) if noise_uv is None else np.asarray(noise_uv, np.float32)
    wander = np.full(n_leads, np.nan, np.float32) if wander_uv is None else np.asarray(wander_uv, np.float32)
    reasons: list[list[str]] = [[] for _ in range(n_leads)]
    for j in range(n_leads):
        if sat[j] > cfg.max_saturation_frac:
            reasons[j].append("saturation")
        if flat[j] > cfg.max_flat_frac:
            reasons[j].append("flat_line")
        if not np.isfinite(noise[j]):
            reasons[j].append("no_template")
        elif noise[j] > cfg.max_noise_uv:
            reasons[j].append("noise")
        if np.isfinite(wander[j]) and wander[j] > cfg.max_wander_uv:
            reasons[j].append("baseline_wander")
    usable = np.array([not r for r in reasons])
    return LeadQuality(list(segment.leads), sat, flat, noise, wander, usable, reasons)
