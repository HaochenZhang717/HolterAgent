"""Analyse one time window: template, ST measurement and lead quality."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..data.record import HolterRecord, Segment
from ..signal import (
    LeadQuality,
    STMeasurement,
    Template,
    assess,
    build_template,
    measure_st,
    select_template_beats,
)


@dataclass
class WindowAnalysis:
    start: int  # samples
    stop: int
    n_beats_total: int
    template: Template | None
    st: STMeasurement | None
    quality: LeadQuality


def read_signal(record: HolterRecord, start: int, stop: int, cfg: Config, leads: list[str] | None = None) -> Segment:
    """Signal as every measurement sees it: with the configured power-line notch filter."""
    f = cfg.filter
    return record.read(start, stop, leads, notch_hz=f.mains_hz, notch_q=f.notch_q, pad_s=f.pad_s)


def read_margin(record: HolterRecord, cfg: Config) -> int:
    """Extra samples needed around a window so every template beat fits."""
    ms = max(cfg.template.pre_ms, cfg.template.post_ms) + 2000.0  # room for the next beat
    return int(round(ms * record.fs / 1000.0))


def analyze_window(
    record: HolterRecord,
    start: int,
    stop: int,
    cfg: Config,
    segment: Segment | None = None,
) -> WindowAnalysis:
    """Analyse beats whose R falls in [start, stop).

    `segment` may be a larger pre-read segment covering the window plus margin;
    otherwise the needed signal is read from disk.
    """
    ann = record.annotations
    sl = ann.beats_between(start, stop)
    candidates = np.arange(sl.start, sl.stop)
    chosen = select_template_beats(ann.beat_samples, ann.beat_symbols, candidates, cfg.beats)
    margin = read_margin(record, cfg)
    need_lo = max(0, start - margin)
    need_hi = min(record.n_samples, stop + margin)
    if segment is None or segment.start > need_lo or segment.stop < need_hi:
        segment = read_signal(record, need_lo, need_hi, cfg)
    template = None
    if len(chosen) >= cfg.beats.min_beats:
        template = build_template(
            segment,
            ann.beat_samples[chosen],
            ann.beat_samples[chosen + 1],
            cfg.template,
            cfg.fiducial,
        )
    st = measure_st(template, cfg.st) if template is not None else None
    lo = max(0, start - segment.start)
    hi = min(segment.signal.shape[0], stop - segment.start)
    window = Segment(start, segment.fs, segment.leads, segment.signal[lo:hi], segment.saturated[lo:hi])
    quality = assess(
        window,
        template.noise_uv if template is not None else None,
        template.wander_uv if template is not None else None,
        cfg.quality,
    )
    if template is not None:
        # A lead is unusable when no beat survived for it.
        for j, n in enumerate(template.n_beats):
            if n < cfg.beats.min_beats and "too_few_beats" not in quality.reasons[j]:
                quality.reasons[j].append("too_few_beats")
                quality.usable[j] = False
    return WindowAnalysis(start, stop, int(sl.stop - sl.start), template, st, quality)
