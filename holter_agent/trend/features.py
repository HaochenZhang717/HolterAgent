"""Whole-record feature scan: one row per analysis window, cached to disk."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..data.record import HolterRecord, unique_lead_names
from .window import analyze_window, read_margin, read_signal

# 2: noisy leads left out of the fiducial search; adds fid_excluded. 3: 50 Hz notch filter.
# 4: isoelectric point closest to QRS onset among the nearly flattest windows; adds iso_ms.
# 5: noise region stops at QRS onset; lead exclusion threshold 50 uV (calibrated after filtering).
# 6: lead exclusion at 100 uV, falling back to 50 uV only when the QRS is not found.
FEATURE_VERSION = 6

# Per-window, per-lead arrays [W, L].
LEAD_FIELDS = (
    "st_uv",  # ST level at the heart-rate adjusted point after J
    "st_j0_uv",
    "st_slope",  # mV/s from J to J+X
    "t_amp_uv",
    "qrs_pp_uv",
    "qrs_area_uvs",
    "noise_uv",
    "wander_uv",
    "sat_frac",
    "flat_frac",
    "usable",
    "fid_excluded",  # lead left out of the QRS boundary search because it was too noisy
)
# Per-window scalars [W].
WINDOW_FIELDS = (
    "t_start_s",
    "hr_bpm",  # from all beats in the window
    "n_beats",
    "n_template_beats",
    "qrs_ms",
    "j_ms",
    "iso_ms",  # centre of the isoelectric window, ms relative to R
    "fiducials_ok",
    "n_normal",
    "n_supra",  # S A a J e j n
    "n_ventricular",  # V E F r
    "n_other",  # Q ? and anything else
)
SUPRA = frozenset("SAaJejn")
VENTRICULAR = frozenset("VEFr")


@dataclass
class FeatureTable:
    record: str
    fs: float
    leads: list[str]
    window_s: float
    duration_s: float
    arrays: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.arrays["t_start_s"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def window_index(self, t_s: float) -> int:
        return int(np.clip(t_s // self.window_s, 0, self.n_windows - 1))

    def lead_index(self, lead: str) -> int:
        return self.leads.index(lead)

    def isolated_outliers(self, threshold_uv: float = 100.0, k: int = 3, min_neighbours: int = 4) -> np.ndarray:
        """[W, L] usable ST values more than threshold_uv away from the median of the k usable
        windows on each side (quality check; a single window cannot make a candidate)."""
        st = np.where(self["usable"], self["st_uv"], np.nan).astype(np.float64)
        w = len(st)
        pad = np.vstack([np.full((k, st.shape[1]), np.nan), st, np.full((k, st.shape[1]), np.nan)])
        neigh = np.stack([pad[k + d:k + d + w] for d in range(-k, k + 1) if d], 0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(neigh, 0)
        enough = np.isfinite(neigh).sum(0) >= min_neighbours
        return np.isfinite(st) & enough & (np.abs(st - med) > threshold_uv)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta, record=self.record, fs=self.fs, leads=self.leads,
                    window_s=self.window_s, duration_s=self.duration_s, version=FEATURE_VERSION)
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **self.arrays)

    @classmethod
    def load(cls, path: str | Path, check_version: bool = True) -> "FeatureTable":
        """Load a cached table; check_version=False also loads older versions (for comparisons)."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            arrays = {k: z[k] for k in z.files if k != "__meta__"}
        if check_version and meta.get("version") != FEATURE_VERSION:
            raise ValueError(f"{path}: feature version {meta.get('version')} != {FEATURE_VERSION}")
        # tables cached before 2026-10-08 may hold a header's duplicate lead names; name them as HolterRecord does
        leads = unique_lead_names(meta["leads"])
        return cls(meta["record"], meta["fs"], leads, meta["window_s"], meta["duration_s"], arrays, meta | {"leads": leads})


def _composition(symbols: np.ndarray) -> tuple[int, int, int, int]:
    n = int((symbols == "N").sum())
    s = int(np.isin(symbols, list(SUPRA)).sum())
    v = int(np.isin(symbols, list(VENTRICULAR)).sum())
    return n, s, v, int(len(symbols) - n - s - v)


def compute_features(record: HolterRecord, cfg: Config) -> FeatureTable:
    fs = record.fs
    win = int(round(cfg.feature.window_s * fs))
    n_win = int(np.ceil(record.n_samples / win))
    n_leads = len(record.leads)
    arrays = {k: np.full((n_win, n_leads), np.nan, np.float32) for k in LEAD_FIELDS}
    arrays["usable"] = np.zeros((n_win, n_leads), bool)
    arrays["fid_excluded"] = np.zeros((n_win, n_leads), bool)
    for k in WINDOW_FIELDS:
        arrays[k] = np.full(n_win, np.nan, np.float32)
    arrays["fiducials_ok"] = np.zeros(n_win, bool)
    arrays["t_start_s"] = (np.arange(n_win) * win / fs).astype(np.float64)

    ann = record.annotations
    margin = read_margin(record, cfg)
    per_chunk = max(1, int(cfg.feature.chunk_s // cfg.feature.window_s))
    for c0 in range(0, n_win, per_chunk):
        c1 = min(n_win, c0 + per_chunk)
        seg = read_signal(record, c0 * win - margin, c1 * win + margin, cfg)
        for w in range(c0, c1):
            start, stop = w * win, min(record.n_samples, (w + 1) * win)
            sl = ann.beats_between(start, stop)
            sym = ann.beat_symbols[sl]
            n, s, v, o = _composition(sym)
            arrays["n_beats"][w] = len(sym)
            arrays["n_normal"][w], arrays["n_supra"][w] = n, s
            arrays["n_ventricular"][w], arrays["n_other"][w] = v, o
            beats = ann.beat_samples[max(0, sl.start - 1):sl.stop]
            rr = np.diff(beats) / fs
            rr = rr[(rr > 0.2) & (rr < 3.0)]
            if len(rr) >= 3:
                arrays["hr_bpm"][w] = 60.0 / np.median(rr)
            wa = analyze_window(record, start, stop, cfg, segment=seg)
            q = wa.quality
            arrays["sat_frac"][w] = q.saturation_frac
            arrays["flat_frac"][w] = q.flat_frac
            arrays["noise_uv"][w] = q.noise_uv
            arrays["wander_uv"][w] = q.wander_uv
            if wa.template is None:
                continue
            tpl, st = wa.template, wa.st
            fid = tpl.fiducials
            arrays["n_template_beats"][w] = len(tpl.beat_samples)
            arrays["qrs_ms"][w] = fid.qrs_ms
            arrays["j_ms"][w] = fid.j_ms
            arrays["iso_ms"][w] = fid.iso_ms
            arrays["fiducials_ok"][w] = fid.ok
            arrays["fid_excluded"][w, fid.leads_excluded] = True
            if fid.ok and np.isfinite(fid.onset_ms):
                lo, hi = tpl.index_at(fid.onset_ms), tpl.index_at(fid.j_ms)
                qrs = tpl.data[lo:hi]
                arrays["qrs_pp_uv"][w] = np.ptp(qrs, axis=0) * 1000.0
                arrays["qrs_area_uvs"][w] = np.nansum(qrs, axis=0) * 1000.0 / fs  # mV*ms == uV*s
            if st is not None and st.valid:
                arrays["st_uv"][w] = st.level_uv
                arrays["st_j0_uv"][w] = st.points_uv[0.0]
                arrays["st_slope"][w] = st.slope_mv_per_s
                arrays["t_amp_uv"][w] = st.t_amp_uv
                arrays["usable"][w] = q.usable
    return FeatureTable(
        record=record.name,
        fs=fs,
        leads=list(record.leads),
        window_s=cfg.feature.window_s,
        duration_s=record.duration_s,
        arrays=arrays,
        meta={"config": cfg.to_dict()},
    )


def load_or_compute(record: HolterRecord, cfg: Config, cache_dir: str | Path | None) -> FeatureTable:
    if cache_dir is None:
        return compute_features(record, cfg)
    path = Path(cache_dir) / f"{record.name}.features.npz"
    if path.exists():
        try:
            table = FeatureTable.load(path)
        except ValueError:  # written by an older feature version: recompute
            table = None
        if table is not None and table.meta.get("config") == json.loads(json.dumps(cfg.to_dict())):
            return table
    table = compute_features(record, cfg)
    table.save(path)
    return table
