"""Median beat templates with per-beat baseline correction.

Two passes:
1. Subtract a rough isoelectric level from each beat, take the median, and find
   the fiducials on that rough template. Leads whose beats scatter strongly around
   the rough template (noise) are left out of the fiducial search in both passes.
2. For each beat, draw a straight baseline from its own isoelectric point to the
   isoelectric point of the next beat, subtract it, and take the median again.
   Fiducials are re-detected on the final template.

A beat-lead window that contains a saturated sample is excluded from that lead's
median (set to NaN), so saturation cannot leak into ST levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FiducialConfig, TemplateConfig
from ..data.record import Segment
from .fiducials import Fiducials, detect_fiducials


@dataclass
class Template:
    data: np.ndarray  # float32 [n_t, n_leads], mV, isoelectric-corrected
    fs: float
    r_index: int  # index of the R fiducial in `data`
    leads: list[str]
    n_beats: np.ndarray  # int [n_leads], beats contributing to each lead
    rr_ms: float  # median RR of the template beats
    noise_uv: np.ndarray  # [n_leads] median beat-to-template RMS residual in the ST measurement region
    wander_uv: np.ndarray  # [n_leads] p95 - p5 of per-beat isoelectric levels
    fiducials: Fiducials
    beat_samples: np.ndarray  # absolute R samples used

    @property
    def hr_bpm(self) -> float:
        return 60000.0 / self.rr_ms if self.rr_ms > 0 else float("nan")

    def index_at(self, ms: float) -> int:
        return self.r_index + int(round(ms * self.fs / 1000.0))

    def mean_at(self, ms: float, halfwidth_ms: float) -> np.ndarray:
        """Mean value of each lead in [ms - halfwidth, ms + halfwidth] relative to R."""
        lo = max(0, self.index_at(ms - halfwidth_ms))
        hi = min(len(self.data), self.index_at(ms + halfwidth_ms) + 1)
        if hi <= lo:
            return np.full(self.data.shape[1], np.nan, dtype=np.float32)
        return np.nanmean(self.data[lo:hi], axis=0)


def _window_mean(x: np.ndarray, centre: np.ndarray, half: int) -> np.ndarray:
    """Mean of x[c-half : c+half+1] for each centre c; x is [n, L], returns [len(centre), L]."""
    offs = np.arange(-half, half + 1)
    idx = np.clip(centre[:, None] + offs[None, :], 0, len(x) - 1)
    return x[idx].mean(axis=1)


def _measurement_noise_uv(
    corrected: np.ndarray, tpl: np.ndarray, fid: Fiducials, fs: float, r_index: int
) -> np.ndarray:
    """Per-lead beat-to-template RMS residual where ST is measured.

    The region is the isoelectric window (from 20 ms before it up to QRS onset) plus
    J .. J+100 ms. The QRS is left out on purpose: a one-sample jitter of the R
    annotation on a steep QRS edge produces large residuals that say nothing about ST
    reliability.
    """
    t_ms = (np.arange(tpl.shape[0]) - r_index) * 1000.0 / fs
    if fid.ok and np.isfinite(fid.iso_ms) and np.isfinite(fid.j_ms):
        iso = (t_ms >= fid.iso_ms - 20.0) & (t_ms < fid.onset_ms)
        region = iso | ((t_ms >= fid.j_ms) & (t_ms <= fid.j_ms + 100.0))
    else:
        region = (t_ms < -80.0) | (t_ms > 100.0)
    resid = corrected[:, region, :] - tpl[None, region, :]
    with np.errstate(all="ignore"):
        per_beat = np.sqrt(np.nanmean(resid**2, axis=1))  # [B, L]
        return np.nanmedian(per_beat, axis=0) * 1000.0


def _outside_qrs_noise_uv(beats: np.ndarray, tpl: np.ndarray, fs: float, r_index: int,
                          fcfg: FiducialConfig) -> np.ndarray:
    """Per-lead beat-to-template RMS residual outside the QRS search window.

    Unlike `_measurement_noise_uv` it needs no fiducials, so it can decide which
    leads take part in finding them.
    """
    t_ms = (np.arange(tpl.shape[0]) - r_index) * 1000.0 / fs
    region = (t_ms < fcfg.search_ms[0]) | (t_ms > fcfg.search_ms[1])
    resid = beats[:, region, :] - tpl[None, region, :]
    with np.errstate(all="ignore"):
        per_beat = np.sqrt(np.nanmean(resid**2, axis=1))  # [B, L]
        return np.nanmedian(per_beat, axis=0) * 1000.0


def _fiducial_leads(noise_uv: np.ndarray, limit_uv: float, min_leads: int) -> np.ndarray:
    """Leads used to find the fiducials: those not above the noise limit, at least min_leads."""
    finite = np.isfinite(noise_uv)
    use = finite & (noise_uv <= limit_uv)
    if use.sum() < min_leads:
        order = np.argsort(np.where(finite, noise_uv, np.inf), kind="stable")
        use = np.zeros(len(noise_uv), bool)
        use[order[:min_leads]] = True
    return use


def _detect(tpl: np.ndarray, fs: float, r_index: int, fcfg: FiducialConfig, noise_uv: np.ndarray) -> Fiducials:
    """Fiducials with noisy leads left out; stricter noise limits only if the QRS is not found."""
    fid, tried = None, []
    for limit in fcfg.exclude_noise_uv:
        use = _fiducial_leads(noise_uv, limit, fcfg.min_leads)
        if any(np.array_equal(use, u) for u in tried):
            continue
        tried.append(use)
        fid = detect_fiducials(tpl, fs, r_index, fcfg, use)
        if fid.ok:
            break
    return fid


def build_template(
    segment: Segment,
    r_samples: np.ndarray,
    next_r_samples: np.ndarray,
    tcfg: TemplateConfig,
    fcfg: FiducialConfig,
) -> Template | None:
    """Build a template from beats with absolute R positions `r_samples`.

    `next_r_samples[i]` is the R position of the beat that follows `r_samples[i]`;
    it is used to anchor the end of the baseline line. Returns None if no beat fits.
    """
    fs = segment.fs
    pre = int(round(tcfg.pre_ms * fs / 1000.0))
    post = int(round(tcfg.post_ms * fs / 1000.0))
    rel = np.asarray(r_samples, dtype=np.int64) - segment.start
    nxt = np.asarray(next_r_samples, dtype=np.int64) - segment.start
    n = segment.signal.shape[0]
    fits = (rel - pre >= 0) & (rel + post < n)
    rel, nxt = rel[fits], nxt[fits]
    if len(rel) == 0:
        return None
    offs = np.arange(-pre, post)
    idx = rel[:, None] + offs[None, :]
    beats = segment.signal[idx].astype(np.float64)  # [B, T, L]
    sat = segment.saturated[idx].any(axis=1)  # [B, L]
    beats[np.broadcast_to(sat[:, None, :], beats.shape)] = np.nan

    # Pass 1: rough template.
    r0, r1 = (int(round(v * fs / 1000.0)) + pre for v in tcfg.rough_iso_ms)
    rough = beats - np.nanmean(beats[:, r0:r1, :], axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        tpl0 = np.nanmedian(rough, axis=0)
    lead_noise = _outside_qrs_noise_uv(rough, tpl0, fs, pre, fcfg)
    fid0 = _detect(tpl0, fs, pre, fcfg, lead_noise)
    iso_ms = fid0.iso_ms if np.isfinite(fid0.iso_ms) else float(np.mean(tcfg.rough_iso_ms))

    # Pass 2: straight baseline between this beat's and the next beat's isoelectric point.
    iso_off = int(round(iso_ms * fs / 1000.0))
    half = max(1, int(round(fcfg.iso_len_ms * fs / 2000.0)))
    sig = segment.signal.astype(np.float64)
    iso_this = _window_mean(sig, rel + iso_off, half)  # [B, L]
    next_ok = (nxt + iso_off + half < n) & (nxt > rel)
    iso_next = np.where(next_ok[:, None], _window_mean(sig, np.clip(nxt + iso_off, 0, n - 1), half), iso_this)
    span = np.where(next_ok, nxt - rel, 1).astype(np.float64)  # samples between the two iso points
    frac = (offs[None, :] - iso_off) / span[:, None]  # [B, T]
    frac = np.where(next_ok[:, None], frac, 0.0)
    baseline = iso_this[:, None, :] + frac[:, :, None] * (iso_next - iso_this)[:, None, :]
    corrected = beats - baseline
    with np.errstate(all="ignore"):
        tpl = np.nanmedian(corrected, axis=0)
        iso_valid = np.where(sat, np.nan, iso_this)
        wander = (np.nanpercentile(iso_valid, 95, axis=0) - np.nanpercentile(iso_valid, 5, axis=0)) * 1000.0
    fid = _detect(tpl, fs, pre, fcfg, lead_noise)
    noise = _measurement_noise_uv(corrected, tpl, fid, fs, pre)
    rr = np.median(nxt - rel) * 1000.0 / fs if len(rel) else float("nan")
    return Template(
        data=tpl.astype(np.float32),
        fs=fs,
        r_index=pre,
        leads=list(segment.leads),
        n_beats=(~sat).sum(axis=0),
        rr_ms=float(rr),
        noise_uv=noise.astype(np.float32),
        wander_uv=wander.astype(np.float32),
        fiducials=fid,
        beat_samples=rel + segment.start,
    )
