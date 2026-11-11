"""Rhythm context and RR scatter summaries, as numbers rather than plots.

RR intervals are formed only between annotated beats that are adjacent in the
file, so removing a beat can never join two intervals that were not adjacent.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..data.record import HolterRecord

PACING_LEAD_MS = 100.0  # default: '^' this close before a beat counts as a possible pacing spike


def rr_pairs(samples: np.ndarray, symbols: np.ndarray, fs: float, max_rr_s: float = 3.0):
    """Adjacent RR pairs (RR_n, RR_n+1) in ms with the symbol of the middle beat.

    A pair is dropped when either interval exceeds `max_rr_s` (annotation gaps).
    """
    rr = np.diff(samples) * 1000.0 / fs
    x, y = rr[:-1], rr[1:]
    mid = symbols[1:-1]
    ok = (x > 0) & (y > 0) & (x <= max_rr_s * 1000) & (y <= max_rr_s * 1000)
    return x[ok], y[ok], mid[ok]


def lorenz_summary(x: np.ndarray, y: np.ndarray, mid: np.ndarray) -> dict:
    """Numbers describing a Lorenz (RR_n vs RR_n+1) scatter."""
    if len(x) < 3:
        return {"n_pairs": int(len(x))}
    rel = np.abs(y - x) / x
    nn = mid == "N"
    out = {
        "n_pairs": int(len(x)),
        "frac_near_identity_10pct": round(float((rel <= 0.10).mean()), 3),
        "median_abs_successive_diff_ms": round(float(np.median(np.abs(y - x))), 1),
    }
    if nn.sum() >= 3:
        d = (x[nn] - y[nn]) / np.sqrt(2)
        a = (x[nn] + y[nn]) / np.sqrt(2)
        out["sd1_ms_normal"] = round(float(d.std()), 1)
        out["sd2_ms_normal"] = round(float(a.std()), 1)
    # Premature pairs: a short interval followed by a longer one, the usual
    # footprint of an ectopic beat.
    out["frac_short_then_long_20pct"] = round(float(((x < 0.8 * y)).mean()), 3)
    return out


def rhythm_context(record: HolterRecord, start_s: float, end_s: float) -> dict:
    ann = record.annotations
    fs = record.fs
    a, b = record.to_samples(start_s), record.to_samples(end_s)
    sl = ann.beats_between(a, b)
    samples = ann.beat_samples[max(0, sl.start - 1):sl.stop + 1]
    symbols = ann.beat_symbols[max(0, sl.start - 1):sl.stop + 1]
    comp = Counter(ann.beat_symbols[sl].tolist())
    x, y, mid = rr_pairs(samples, symbols, fs)
    rr = np.diff(samples) * 1000.0 / fs
    rr = rr[(rr > 200) & (rr < 3000)]
    out: dict = {
        "start_s": round(start_s),
        "end_s": round(end_s),
        "n_beats": int(sl.stop - sl.start),
        "beat_types": dict(comp.most_common()),
        "hr_bpm": None,
        "lorenz": lorenz_summary(x, y, mid),
    }
    if len(rr) >= 3:
        hr = 60000.0 / rr
        out["hr_bpm"] = {"median": round(float(np.median(hr)), 1),
                         "p5": round(float(np.percentile(hr, 5)), 1),
                         "p95": round(float(np.percentile(hr, 95)), 1)}
        # Beat-to-beat irregularity over all intervals, whatever the beat labels say: a recorder can label
        # every beat of atrial fibrillation as normal.
        d = np.abs(np.diff(rr))
        out["rr_irregularity"] = {"rmssd_over_mean_rr": round(float(np.sqrt(np.mean(d ** 2)) / np.mean(rr)), 3),
                                  "frac_successive_diff_gt_50ms": round(float((d > 50).mean()), 3),
                                  "note": "high values with beats labelled normal can mean atrial fibrillation or "
                                          "flutter missed by the recorder; check view_ecg for P waves"}
    out["rhythm_labels"] = [
        {"label": r.label, "start_s": round(r.start / fs), "end_s": round(r.end / fs),
         "overlap_s": round((min(r.end, b) - max(r.start, a)) / fs), "closed": r.closed}
        for r in ann.rhythms_overlapping(a, b)
    ]
    carets = ann.events.get("^", np.array([], dtype=np.int64))
    carets = carets[(carets >= a) & (carets < b)]
    if len(carets):
        j = np.searchsorted(ann.beat_samples, carets)
        j = np.clip(j, 0, len(ann.beat_samples) - 1)
        lead_ms = (ann.beat_samples[j] - carets) * 1000.0 / fs
        near = (lead_ms >= 0) & (lead_ms <= PACING_LEAD_MS)
        out["caret_events"] = {
            "n": int(len(carets)),
            "frac_just_before_beat": round(float(near.mean()), 3),
            "beat_types_after": dict(Counter(ann.beat_symbols[j[near]].tolist()).most_common(4)),
        }
    return out
