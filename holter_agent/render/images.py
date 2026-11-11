"""Deterministic PNG images drawn from the same signal and computation as the measurement tools.

Signals are drawn on ECG paper: a 1-mm grid with a darker line every 5 mm, at a fixed paper speed
(mm/s) and gain (mm/mV), so amplitudes and times can be read from the grid. The images carry no
measured values as text, only scales, lead names, times, beat labels and marker lines, so a run that
sees only images has to read the values from the picture.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PX_PER_MM = 5.0  # rendering resolution
MM_PER_INCH = 25.4
GRID_MINOR = "#f4c7c3"
GRID_MAJOR = "#e8908a"
TRACE = "#111111"
SCALES = {
    "strip": {"mm_per_s": 25.0, "mm_per_mv": 10.0},  # standard ECG paper
    "beat": {"mm_per_s": 50.0, "mm_per_mv": 20.0},  # averaged beats, twice the standard scale
}
MARKERS = {"baseline": "#2e7d32", "j": "#e08a1e", "st": "#b03030", "usual": "#1f4e79"}


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PX_PER_MM * MM_PER_INCH, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def _figure(width_mm: float, height_mm: float):
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
    return fig


def _paper(ax, x0: float, x1: float, y0: float, y1: float) -> None:
    """ECG grid in millimetres on an axes whose data units are millimetres."""
    for v in np.arange(np.floor(x0), x1 + 1e-9, 1.0):
        ax.axvline(v, color=GRID_MAJOR if round(v) % 5 == 0 else GRID_MINOR, lw=0.6 if round(v) % 5 == 0 else 0.3,
                   zorder=0)
    for v in np.arange(np.floor(y0), y1 + 1e-9, 1.0):
        ax.axhline(v, color=GRID_MAJOR if round(v) % 5 == 0 else GRID_MINOR, lw=0.6 if round(v) % 5 == 0 else 0.3,
                   zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def ecg_strip(signal: np.ndarray, fs: float, leads: list[str], start_s: float, title: str,
              beats: list[tuple[float, str]] | None = None, row_mm: float = 20.0) -> bytes:
    """Raw ECG, one row per lead, on standard paper (25 mm/s, 10 mm/mV), with a 1-mV calibration pulse.

    `signal` is [n, L] in mV; `beats` are (time_s, label) pairs from the annotation file.
    """
    sp = SCALES["strip"]
    dur = signal.shape[0] / fs
    left, width = 18.0, dur * sp["mm_per_s"]
    top = 10.0
    height = top + row_mm * len(leads) + 4.0
    fig = _figure(left + width + 4.0, height)
    ax = fig.add_axes([0, 0, 1, 1])
    _paper(ax, 0, left + width + 4.0, 0, height)
    t_mm = left + np.arange(signal.shape[0]) / fs * sp["mm_per_s"]
    for i, lead in enumerate(leads):
        base = height - top - row_mm * (i + 0.5)
        y = signal[:, i].astype(float)
        y = y - np.nanmedian(y)
        ax.plot(t_mm, base + y * sp["mm_per_mv"], color=TRACE, lw=0.6)
        # calibration pulse: 1 mV high, 200 ms wide
        cx = 4.0
        ax.plot([cx, cx + 1, cx + 1, cx + 1 + 0.2 * sp["mm_per_s"], cx + 1 + 0.2 * sp["mm_per_s"], cx + 7],
                [base, base, base + sp["mm_per_mv"], base + sp["mm_per_mv"], base, base], color=TRACE, lw=0.6)
        ax.text(cx + 8, base + 2.5, lead, fontsize=7, va="bottom")
    for tb, lab in beats or []:
        x = left + (tb - start_s) * sp["mm_per_s"]
        if left <= x <= left + width:
            ax.text(x, height - top + 1.5, lab, fontsize=6, ha="center", color="#555555")
    for k in range(int(np.floor(dur)) + 1):
        ax.text(left + k * sp["mm_per_s"], 1.0, f"{k} s", fontsize=5.5, ha="center", color="#555555")
    ax.text(left, height - 3.0, f"{title}   25 mm/s, 10 mm/mV; small square 40 ms x 0.1 mV", fontsize=7, va="top")
    return _png(fig)


def _beat_grid(n: int) -> tuple[int, int]:
    cols = 4 if n > 4 else n
    return int(np.ceil(n / cols)), cols


def _beat_axes(fig, n: int, panel_w: float, panel_h: float, top: float, fig_w: float, fig_h: float):
    rows, cols = _beat_grid(n)
    axes = []
    for k in range(n):
        r, c = divmod(k, cols)
        x0 = 4.0 + c * (panel_w + 4.0)
        y0 = fig_h - top - (r + 1) * (panel_h + 4.0)
        axes.append(fig.add_axes([x0 / fig_w, y0 / fig_h, panel_w / fig_w, panel_h / fig_h]))
    return axes


def beat_panels(t_ms: np.ndarray, beats: np.ndarray, leads: list[str], iso_ms: float, j_ms: float,
                st_ms: float, usual_uv: np.ndarray | None, title: str, window_ms=(-250.0, 450.0),
                span_mv=(-1.0, 1.5)) -> bytes:
    """Averaged beat per lead at 50 mm/s and 20 mm/mV, baseline at zero.

    Markers: baseline level (dashed), baseline point, J point and ST point (vertical lines), and the
    lead's usual ST level (dotted), drawn at the height of that level above the baseline.
    """
    sp = SCALES["beat"]
    panel_w = (window_ms[1] - window_ms[0]) / 1000.0 * sp["mm_per_s"]
    panel_h = (span_mv[1] - span_mv[0]) * sp["mm_per_mv"]
    rows, cols = _beat_grid(len(leads))
    top = 14.0
    fig_w = 4.0 + cols * (panel_w + 4.0)
    fig_h = top + rows * (panel_h + 4.0) + 2.0
    fig = _figure(fig_w, fig_h)
    x = (t_ms - window_ms[0]) / 1000.0 * sp["mm_per_s"]
    sel = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
    y0 = -span_mv[0] * sp["mm_per_mv"]
    xm = lambda ms: (ms - window_ms[0]) / 1000.0 * sp["mm_per_s"]  # noqa: E731
    for k, ax in enumerate(_beat_axes(fig, len(leads), panel_w, panel_h, top, fig_w, fig_h)):
        _paper(ax, 0, panel_w, 0, panel_h)
        ax.axhline(y0, color=MARKERS["baseline"], lw=0.7, ls="--", zorder=1)
        for ms, key in ((iso_ms, "baseline"), (j_ms, "j"), (st_ms, "st")):
            if np.isfinite(ms):
                ax.axvline(xm(ms), color=MARKERS[key], lw=0.8, zorder=1)
        if usual_uv is not None and np.isfinite(usual_uv[k]):
            ax.axhline(y0 + usual_uv[k] / 1000.0 * sp["mm_per_mv"], color=MARKERS["usual"], lw=0.8, ls=":", zorder=1)
        ax.plot(x[sel], y0 + beats[sel, k] * sp["mm_per_mv"], color=TRACE, lw=0.9, zorder=2)
        ax.text(1.0, panel_h - 1.0, leads[k], fontsize=7, va="top")
    fig.text(4.0 / fig_w, 1 - 1.5 / fig_h,
             f"{title}\n50 mm/s, 20 mm/mV (small square 20 ms x 50 uV). Green dashed: baseline. Vertical lines: "
             "baseline point (green),\nJ point (orange), ST point (red). Blue dotted: the lead's usual ST level.",
             fontsize=6.5, va="top")
    return _png(fig)


def beat_overlay(t_ms: np.ndarray, ref: np.ndarray, cur: np.ndarray, leads: list[str], title: str,
                 window_ms=(-250.0, 450.0), span_mv=(-1.0, 1.5)) -> bytes:
    """Two averaged beats per lead, aligned at R and at their own baselines: grey = reference time."""
    sp = SCALES["beat"]
    panel_w = (window_ms[1] - window_ms[0]) / 1000.0 * sp["mm_per_s"]
    panel_h = (span_mv[1] - span_mv[0]) * sp["mm_per_mv"]
    rows, cols = _beat_grid(len(leads))
    top = 14.0
    fig_w = 4.0 + cols * (panel_w + 4.0)
    fig_h = top + rows * (panel_h + 4.0) + 2.0
    fig = _figure(fig_w, fig_h)
    x = (t_ms - window_ms[0]) / 1000.0 * sp["mm_per_s"]
    sel = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
    y0 = -span_mv[0] * sp["mm_per_mv"]
    for k, ax in enumerate(_beat_axes(fig, len(leads), panel_w, panel_h, top, fig_w, fig_h)):
        _paper(ax, 0, panel_w, 0, panel_h)
        ax.plot(x[sel], y0 + ref[sel, k] * sp["mm_per_mv"], color="#9e9e9e", lw=1.4, zorder=2)
        ax.plot(x[sel], y0 + cur[sel, k] * sp["mm_per_mv"], color=TRACE, lw=0.9, zorder=3)
        ax.text(1.0, panel_h - 1.0, leads[k], fontsize=7, va="top")
    fig.text(4.0 / fig_w, 1 - 1.5 / fig_h,
             f"{title}\n50 mm/s, 20 mm/mV (small square 20 ms x 50 uV). Grey: reference time; black: the other "
             "time. Both drawn from their own baseline.", fontsize=6.5, va="top")
    return _png(fig)


def st_trend_image(t_s: np.ndarray, dev_uv: np.ndarray, hr: np.ndarray, leads: list[str], title: str,
                   spans: list[tuple[float, float]] | None = None, clip_uv: float = 250.0) -> bytes:
    """ST deviation per lead over time (one row each, lines at 0, +-50, +-100 uV) and heart rate."""
    n = len(leads)
    fig, axes = plt.subplots(n + 1, 1, figsize=(240 / MM_PER_INCH, (17 * n + 34) / MM_PER_INCH), sharex=True,
                             gridspec_kw={"height_ratios": [1] * n + [1.6]})
    x = t_s / 3600.0
    for k, ax in enumerate(axes[:-1]):
        for v, ls in ((0, "-"), (50, ":"), (-50, ":"), (100, "--"), (-100, "--")):
            ax.axhline(v, color="#9e9e9e", lw=0.5, ls=ls)
        ax.plot(x, np.clip(dev_uv[:, k], -clip_uv, clip_uv), color=TRACE, lw=0.6)
        ax.set_ylim(-clip_uv, clip_uv)
        step = 100 if clip_uv <= 400 else 200
        top = int(clip_uv // step) * step
        ax.set_yticks(list(range(-top, top + 1, step)))
        ax.tick_params(labelsize=5.5, length=2)
        ax.set_ylabel(leads[k], fontsize=7, rotation=0, ha="right", va="center")
        for a, b in spans or []:
            ax.axvspan(a / 3600.0, b / 3600.0, color="#f3d9d9", zorder=0)
    ax = axes[-1]
    ax.plot(x, hr, color="#333333", lw=0.6)
    ax.set_ylabel("HR", fontsize=7, rotation=0, ha="right", va="center")
    ax.tick_params(labelsize=5.5, length=2)
    for a, b in spans or []:
        ax.axvspan(a / 3600.0, b / 3600.0, color="#f3d9d9", zorder=0)
    ax.set_xlabel("hours from the start of the recording", fontsize=7)
    fig.suptitle(f"{title}\nST deviation per lead in uV (dashed: +-100, dotted: +-50); bottom: heart rate in bpm; "
                 "shaded: the candidate", fontsize=7, x=0.01, ha="left")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.04, hspace=0.15)
    return _png(fig)
