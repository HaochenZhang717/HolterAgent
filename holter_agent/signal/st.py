"""ST-segment measurements on a template."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import STConfig
from .templates import Template


@dataclass
class STMeasurement:
    leads: list[str]
    hr_bpm: float
    jx_offset_ms: float  # heart-rate adjusted measurement point after J
    level_uv: np.ndarray  # [L] ST level at J + jx_offset, relative to the isoelectric level
    points_uv: dict[float, np.ndarray]  # J + offset (ms) -> [L] level
    slope_mv_per_s: np.ndarray  # [L] (level(J+X) - level(J)) / X
    t_amp_uv: np.ndarray  # [L] signed T-wave extreme in the T window after J
    valid: bool  # False when the fiducials are unusable


def jx_offset_ms(hr_bpm: float, cfg: STConfig) -> float:
    """Heart-rate adjusted ST measurement point after J (LTSTDB convention)."""
    if not np.isfinite(hr_bpm):
        return cfg.offsets_ms[0]
    for brk, off in zip(cfg.hr_breaks_bpm, cfg.offsets_ms):
        if hr_bpm < brk:
            return off
    return cfg.offsets_ms[-1]


def measure_st(tpl: Template, cfg: STConfig) -> STMeasurement:
    fid = tpl.fiducials
    n_leads = tpl.data.shape[1]
    nan = np.full(n_leads, np.nan, dtype=np.float32)
    hr = tpl.hr_bpm
    x = jx_offset_ms(hr, cfg)
    if not fid.ok or not np.isfinite(fid.j_ms):
        return STMeasurement(tpl.leads, hr, x, nan, {}, nan, nan, valid=False)
    hw = cfg.point_halfwidth_ms
    iso = tpl.mean_at(fid.iso_ms, hw)
    points = {off: (tpl.mean_at(fid.j_ms + off, hw) - iso) * 1000.0 for off in cfg.extra_points_ms}
    level = (tpl.mean_at(fid.j_ms + x, hw) - iso) * 1000.0
    slope = (level - points[0.0]) / x if 0.0 in points else nan
    lo = tpl.index_at(fid.j_ms + cfg.t_window_ms[0])
    hi = min(len(tpl.data), tpl.index_at(fid.j_ms + cfg.t_window_ms[1]))
    if hi > lo:
        seg = tpl.data[lo:hi] - iso[None, :]
        with np.errstate(all="ignore"):
            k = np.nanargmax(np.abs(np.nan_to_num(seg, nan=0.0)), axis=0)
        t_amp = seg[k, np.arange(n_leads)] * 1000.0
    else:
        t_amp = nan
    return STMeasurement(
        leads=tpl.leads,
        hr_bpm=hr,
        jx_offset_ms=x,
        level_uv=level.astype(np.float32),
        points_uv={k: v.astype(np.float32) for k, v in points.items()},
        slope_mv_per_s=np.asarray(slope, dtype=np.float32),
        t_amp_uv=np.asarray(t_amp, dtype=np.float32),
        valid=True,
    )
