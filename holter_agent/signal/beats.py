"""Choose beats that are safe to average into an ST template."""
from __future__ import annotations

import numpy as np

from ..config import BeatSelectionConfig


def select_template_beats(
    beat_samples: np.ndarray,
    beat_symbols: np.ndarray,
    candidates: np.ndarray,
    cfg: BeatSelectionConfig,
) -> np.ndarray:
    """Return the subset of `candidates` (indices into the beat arrays) usable for a template.

    A beat is kept when it and both neighbours carry a normal symbol, and both of its
    RR intervals lie within `rr_tolerance` of the median RR of the candidate beats.
    Beats at the first or last position of the record are dropped.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    n = len(beat_samples)
    candidates = candidates[(candidates > 0) & (candidates < n - 1)]
    if len(candidates) == 0:
        return candidates
    normal = np.isin(beat_symbols, list(cfg.normal_symbols))
    keep = normal[candidates] & normal[candidates - 1] & normal[candidates + 1]
    rr_prev = beat_samples[candidates] - beat_samples[candidates - 1]
    rr_next = beat_samples[candidates + 1] - beat_samples[candidates]
    ref = np.median(np.concatenate([rr_prev[keep], rr_next[keep]])) if keep.any() else None
    if ref is None:
        return candidates[:0]
    tol = cfg.rr_tolerance
    keep &= np.abs(rr_prev - ref) <= tol * ref
    keep &= np.abs(rr_next - ref) <= tol * ref
    return candidates[keep]
