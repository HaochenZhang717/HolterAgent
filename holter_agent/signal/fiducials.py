"""QRS onset, J point and isoelectric point on an averaged beat.

The fiducials are global (one set for all leads), found from the spatial velocity
sqrt(sum_leads (dx/dt)^2) of the template. This mirrors the usual practice of
placing a single J point across simultaneously recorded leads. Leads chosen by the
caller (noisy ones) can be left out of the spatial velocity; the fiducials still
apply to all leads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FiducialConfig


@dataclass
class Fiducials:
    onset_ms: float  # QRS onset, ms relative to R
    j_ms: float  # J point (QRS offset), ms relative to R
    iso_ms: float  # centre of the isoelectric window in the PR segment
    ok: bool
    issues: list[str] = field(default_factory=list)
    leads_excluded: list[int] = field(default_factory=list)  # lead indices left out of the spatial velocity

    @property
    def qrs_ms(self) -> float:
        return self.j_ms - self.onset_ms


def spatial_velocity(data: np.ndarray) -> np.ndarray:
    """Spatial velocity per sample, same length as `data` (NaN leads ignored)."""
    d = np.diff(np.nan_to_num(data, nan=0.0), axis=0)
    sv = np.sqrt((d**2).sum(axis=1))
    sv = np.concatenate([sv[:1], sv])
    return np.convolve(sv, np.ones(3) / 3, mode="same")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True values."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def detect_fiducials(data: np.ndarray, fs: float, r_index: int, cfg: FiducialConfig,
                     use: np.ndarray | None = None) -> Fiducials:
    """Find QRS onset, J point and isoelectric point on a template.

    `data` is [n_samples, n_leads] in mV with the R fiducial at `r_index`. `use` is an
    optional boolean mask of the leads that enter the spatial velocity (default: all).
    """
    ms = 1000.0 / fs
    use = np.ones(data.shape[1], bool) if use is None else np.asarray(use, bool)
    excluded = [int(j) for j in np.flatnonzero(~use)]
    sv = spatial_velocity(data[:, use])
    lo = max(0, r_index + int(round(cfg.search_ms[0] / ms)))
    hi = min(len(sv), r_index + int(round(cfg.search_ms[1] / ms)))
    win = sv[lo:hi]
    issues: list[str] = []
    if win.size == 0 or not np.isfinite(win).any() or win.max() <= 0:
        return Fiducials(np.nan, np.nan, np.nan, ok=False, issues=["no_qrs_activity"], leads_excluded=excluded)
    thr = cfg.threshold_frac * win.max()
    runs = _runs(win > thr)
    gap = int(round(cfg.merge_gap_ms / ms))
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    r_rel = r_index - lo
    peak_rel = int(np.argmax(win))

    def distance(run):
        s, e = run
        if s <= r_rel < e:
            return 0
        return min(abs(r_rel - s), abs(r_rel - (e - 1)), abs(peak_rel - s))

    s, e = min(merged, key=distance)
    onset_idx, j_idx = lo + s, lo + e
    if distance((s, e)) > 0:
        issues.append("qrs_run_not_at_r")

    # Isoelectric point: of the iso_len windows before QRS onset that are nearly as flat as the
    # flattest one, the one closest to QRS onset (the end of the PR segment, not the P wave).
    iso_len = max(2, int(round(cfg.iso_len_ms / ms)))
    iso_lo = max(0, onset_idx - int(round(cfg.iso_search_ms / ms)))
    iso_hi = onset_idx - 1
    if iso_hi - iso_lo < iso_len:
        issues.append("iso_window_truncated")
        iso_center = max(iso_len // 2, onset_idx - iso_len)
    else:
        csum = np.concatenate([[0.0], np.cumsum(sv[iso_lo:iso_hi])])
        means = (csum[iso_len:] - csum[:-iso_len]) / iso_len
        near_flat = np.flatnonzero(means <= cfg.iso_flat_ratio * means.min())
        iso_center = iso_lo + int(near_flat.max()) + iso_len // 2

    onset_ms = (onset_idx - r_index) * ms
    j_ms = (j_idx - r_index) * ms
    iso_ms = (iso_center - r_index) * ms
    # Only an implausible QRS duration makes the fiducials unusable; the other
    # issues are warnings passed on to the caller.
    ok = cfg.qrs_ms_range[0] <= j_ms - onset_ms <= cfg.qrs_ms_range[1]
    if not ok:
        issues.append("qrs_duration_out_of_range")
    return Fiducials(onset_ms, j_ms, iso_ms, ok=ok, issues=issues, leads_excluded=excluded)
