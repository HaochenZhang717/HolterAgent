"""Deterministic tools the agent can call.

Every tool returns a JSON-serialisable dict of numbers and short strings. No tool
returns an image: the design keeps perception in code and reasoning in the LLM.
Times are seconds from the start of the record; ST values are in microvolts.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

from ..config import DEFAULT_CONFIG, Config
from ..data.clinical import parse_clinical
from ..data.record import HolterRecord, RecordStore
from ..rhythm.rr import rhythm_context
from ..signal.morphology import compare
from ..trend.episodes import Candidate, detect_candidates, hms, st_deviation, st_reference
from ..trend.features import FeatureTable, load_or_compute
from ..trend.shape import (QRS_FEATURES, ST_T_FEATURES, largest_jump, shape_change, shape_scores,
                           st_segment_shape, typical_jump)
from ..trend.steps import FLAT_S, Step, frontal_axis_series, record_steps
from ..trend.window import WindowAnalysis, analyze_window

# Lead groups used to describe the spatial extent of a change.
TERRITORIES = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anterior": ("V2", "V3", "V4"),
    "septal": ("V1", "V2"),
}
CRITERIA_RULES = ("acc_aha_1999", "contiguous_leads")
# Measurement warnings (toolkit 0.7). Defaults chosen from failure cases, not from the guideline.
QRS_MS_WARN_DELTA = 30.0  # QRS duration this far from the record median: J point possibly misplaced
LOW_QRS_PP_UV = 500.0  # QRS peak-to-peak below this: QRS correlation and amplitude ratio are unstable
SAME_HR_BPM = 5.0  # heart-rate tolerance of the same-heart-rate comparison in episode_dynamics


def has_contiguous_pair(leads: list[str]) -> bool:
    """Whether at least two of the record's leads fall in one standard 12-lead group (TERRITORIES)."""
    return any(sum(lead in leads for lead in members) >= 2 for members in TERRITORIES.values())


class ToolError(Exception):
    """A tool could not run with the given arguments; the message goes back to the agent."""


def _r(x: Any, nd: int = 0) -> Any:
    """Round for compact JSON; NaN/inf become None."""
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if not math.isfinite(f):
        return None
    return round(f, nd) if nd else int(round(f))


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats to plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _per_lead(leads: list[str], values: np.ndarray, nd: int = 0) -> dict[str, Any]:
    return {lead: _r(v, nd) for lead, v in zip(leads, values)}


@dataclass
class _RecordState:
    record: HolterRecord
    table: FeatureTable
    candidates: dict[tuple[str, str], list[Candidate]] = field(default_factory=dict)
    steps: dict[str, list[Step]] | None = None
    eligibility: dict | None = None
    shapes: dict[str, np.ndarray] | None = None  # "qrs", "st_t": [W, k] shape scores
    qrs_median_ms: float | None = None


class HolterToolkit:
    """Holds records, cached features and candidates; exposes the agent tools."""

    def __init__(self, data_dir: str | Path, cache_dir: str | Path | None = None, cfg: Config = DEFAULT_CONFIG):
        self.store = RecordStore(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cfg = cfg
        self._states: dict[str, _RecordState] = {}
        self.estimator = None  # IschemiaEstimator, loaded on first use of ischemia_estimate

    # ------------------------------------------------------------------ helpers
    def _state(self, record_id: str) -> _RecordState:
        name = self.store.resolve(record_id)
        if name not in self._states:
            record = self.store.get(name)
            table = load_or_compute(record, self.cfg, self.cache_dir)
            self._states[name] = _RecordState(record, table)
        return self._states[name]

    def candidates(self, record_id: str, protocol: str = "B", reference: str = "global") -> list[Candidate]:
        st = self._state(record_id)
        key = (protocol, reference)
        if key not in st.candidates:
            if protocol not in self.cfg.episode.protocols:
                raise ToolError(f"unknown protocol {protocol!r}; use one of {sorted(self.cfg.episode.protocols)}")
            st.candidates[key] = detect_candidates(st.table, self.cfg.episode, protocol, reference)
        return st.candidates[key]

    def candidate(self, record_id: str, candidate_id: str) -> Candidate:
        """Look up a candidate among the lists computed so far for this record."""
        known = [c for cands in self._state(record_id).candidates.values() for c in cands]
        for c in known:
            # Ids are "<record>-<rule>-<hhmmss>"; the record part may be left out, since the record is given.
            if candidate_id in (c.id, c.id.split("-", 1)[-1]):
                return c
        ids = ", ".join(sorted({c.id for c in known})[:10])
        raise ToolError(f"no candidate {candidate_id!r} in record {record_id!r}"
                        + (f"; known candidates: {ids}" if ids else "; call list_candidates first"))

    def _check_range(self, st: _RecordState, start_s: float, end_s: float) -> tuple[float, float]:
        dur = st.record.duration_s
        start_s, end_s = max(0.0, float(start_s)), min(dur, float(end_s))
        if end_s <= start_s:
            raise ToolError(f"empty time range; the record lasts {dur:.0f} s")
        return start_s, end_s

    @staticmethod
    def _qrs_median(st: _RecordState) -> float:
        """Median automatic QRS duration over the record's windows with valid fiducials."""
        if st.qrs_median_ms is None:
            ok = st.table["fiducials_ok"]
            with np.errstate(all="ignore"):
                st.qrs_median_ms = float(np.nanmedian(st.table["qrs_ms"][ok])) if ok.any() else float("nan")
        return st.qrs_median_ms

    def _busy(self, record_id: str, reference: str, pad_s: float = 60.0) -> np.ndarray:
        """Windows inside any candidate of the most sensitive protocol (A), padded by pad_s: ST episodes."""
        t = self._state(record_id).table["t_start_s"]
        busy = np.zeros(len(t), bool)
        for c in self.candidates(record_id, "A", reference):
            busy |= (t >= c.start_s - pad_s) & (t < c.end_s + pad_s)
        return busy

    def _qrs_warning(self, st: _RecordState, qrs_ms: float, where: str) -> list[str]:
        med = self._qrs_median(st)
        if np.isfinite(qrs_ms) and np.isfinite(med) and abs(qrs_ms - med) >= QRS_MS_WARN_DELTA:
            return [f"{where}QRS duration {qrs_ms:.0f} ms vs record median {med:.0f} ms: the automatic J point may be "
                    "misplaced (a large ST shift can move it); QRS duration, ST at J and the J-to-ST slope may then "
                    "be wrong. Check the J+40/J+80 levels and the image before relying on them."]
        return []

    def _window(self, st: _RecordState, t_s: float, duration_s: float) -> WindowAnalysis:
        """Analyse the window of `duration_s` centred on `t_s`."""
        rec = st.record
        if not 0 <= t_s < rec.duration_s:
            raise ToolError(f"t_s={t_s} outside the record (0..{rec.duration_s:.0f} s)")
        if not 5.0 <= duration_s <= 300.0:
            raise ToolError("duration_s must be between 5 and 300 s")
        start = max(0, rec.to_samples(t_s - duration_s / 2))
        stop = min(rec.n_samples, rec.to_samples(t_s + duration_s / 2))
        wa = analyze_window(rec, start, stop, self.cfg)
        if wa.template is None:
            raise ToolError(
                f"no template at t={t_s:.0f}s: too few clean normal beats "
                f"({wa.n_beats_total} beats in window); try another time or a longer window"
            )
        return wa

    def strip_range(self, record_id: str, t_s: float, duration_s: float) -> tuple[float, float]:
        """Start and end of a raw-ECG strip of duration_s (2-20 s) centred on t_s, kept inside the record."""
        dur = self._state(record_id).record.duration_s
        if not 0 <= t_s < dur:
            raise ToolError(f"t_s={t_s} outside the record (0..{dur:.0f} s)")
        duration_s = float(np.clip(duration_s, 2.0, 20.0))
        start = float(np.clip(t_s - duration_s / 2, 0.0, max(0.0, dur - duration_s)))
        return start, min(dur, start + duration_s)

    # ------------------------------------------------------------------ tools
    def record_overview(self, record_id: str) -> dict:
        st = self._state(record_id)
        rec, tab = st.record, st.table
        ann = rec.annotations
        sym, cnt = np.unique(ann.beat_symbols, return_counts=True)
        rhythms: dict[str, dict] = {}
        for r in ann.rhythms:
            d = rhythms.setdefault(r.label, {"n_intervals": 0, "hours_sum": 0.0})
            d["n_intervals"] += 1
            d["hours_sum"] += (r.end - r.start) / rec.fs / 3600
        usable = tab["usable"]
        ref = st_reference(tab, "global")[0]
        hr = tab["hr_bpm"]
        out = {
            "record": rec.name,
            "duration_h": _r(rec.duration_s / 3600, 2),
            "fs": rec.fs,
            "leads": rec.leads,
            "beat_counts": {s: int(c) for s, c in sorted(zip(sym, cnt), key=lambda t: -t[1])},
            "rhythm_labels": {k: {"n_intervals": v["n_intervals"], "hours_sum": _r(v["hours_sum"], 2)}
                              for k, v in rhythms.items()},
            "other_events": {k: int(len(v)) for k, v in ann.events.items()},
            "hr_bpm": {"median": _r(np.nanmedian(hr), 1), "p5": _r(np.nanpercentile(hr, 5), 1),
                       "p95": _r(np.nanpercentile(hr, 95), 1)},
            "usable_fraction_by_lead": _per_lead(tab.leads, usable.mean(axis=0), 2),
            "st_reference_uv_by_lead": _per_lead(tab.leads, ref),
            "window_s": tab.window_s,
            "n_candidates": {p: len(self.candidates(record_id, p)) for p in sorted(self.cfg.episode.protocols)},
        }
        return out

    def list_candidates(self, record_id: str, protocol: str = "B", reference: str = "global",
                        max_items: int = 40) -> dict:
        cands = self.candidates(record_id, protocol, reference)
        ranked = sorted(cands, key=lambda c: -abs(c.peak.peak_uv) * math.sqrt(max(c.duration_s, 1)))
        shown = ranked[:max_items]
        return {
            "record": self._state(record_id).record.name,
            "protocol": protocol,
            "vmin_uv_tmin_s": list(self.cfg.episode.protocols[protocol]),
            "reference": reference,
            "n_total": len(cands),
            "n_shown": len(shown),
            "ordering": "by |peak deviation| x sqrt(duration), largest first",
            "candidates": [c.summary() for c in sorted(shown, key=lambda c: c.start_s)],
        }

    def st_trend(self, record_id: str, start_s: float, end_s: float, leads: list[str] | None = None,
                 max_rows: int = 40, reference: str = "global") -> dict:
        st = self._state(record_id)
        tab = st.table
        start_s, end_s = self._check_range(st, start_s, end_s)
        leads = leads or tab.leads
        idx = [tab.lead_index(lead) for lead in leads]
        dev = st_deviation(tab, reference)[:, idx]
        t = tab["t_start_s"]
        sel = np.flatnonzero((t >= start_s) & (t < end_s))
        if len(sel) == 0:
            raise ToolError("no analysis windows in that range")
        bins = np.array_split(sel, min(max_rows, len(sel)))
        rows = []
        for b in bins:
            with np.errstate(all="ignore"):
                d = np.nanmedian(dev[b], axis=0)
                hr = np.nanmedian(tab["hr_bpm"][b])
            rows.append({"t_s": _r(t[b[0]]), "hms": hms(t[b[0]]), "hr": _r(hr),
                         "dev_uv": [_r(v) for v in d],
                         "usable_frac": _r(np.isfinite(dev[b]).mean(), 2)})
        return {"leads": leads, "reference": reference, "bin_s": _r(len(bins[0]) * tab.window_s),
                "note": "dev_uv = ST level minus reference; null = no usable window in the bin", "rows": rows}

    def measure_window(self, record_id: str, t_s: float, duration_s: float = 30.0,
                       reference: str = "global") -> dict:
        st = self._state(record_id)
        wa = self._window(st, t_s, duration_s)
        tpl, meas, q = wa.template, wa.st, wa.quality
        fid = tpl.fiducials
        ref = st_reference(st.table, reference)[st.table.window_index(t_s)]
        out = {
            "t_s": _r(t_s), "hms": hms(t_s), "duration_s": _r(duration_s),
            "n_beats_in_window": wa.n_beats_total, "n_template_beats": int(len(tpl.beat_samples)),
            "hr_bpm": _r(tpl.hr_bpm, 1),
            "fiducials": {"qrs_onset_ms": _r(fid.onset_ms), "j_ms": _r(fid.j_ms), "iso_ms": _r(fid.iso_ms),
                          "qrs_ms": _r(fid.qrs_ms), "qrs_ms_record_median": _r(self._qrs_median(st)),
                          "ok": fid.ok, "issues": fid.issues,
                          "leads_excluded": [tpl.leads[j] for j in fid.leads_excluded]},
            "st_point": f"J+{meas.jx_offset_ms:.0f} ms" if meas else None,
        }
        per_lead = {}
        for j, lead in enumerate(tpl.leads):
            entry = {"usable": bool(q.usable[j]), "issues": q.reasons[j],
                     "noise_uv": _r(q.noise_uv[j]), "sat_frac": _r(q.saturation_frac[j], 3)}
            if meas is not None and meas.valid:
                entry.update({
                    "st_uv": _r(meas.level_uv[j]),
                    "reference_uv": _r(ref[j]),
                    "dev_uv": _r(meas.level_uv[j] - ref[j]),
                    "j0_uv": _r(meas.points_uv[0.0][j]),
                    "j20_uv": _r(meas.points_uv[20.0][j]),
                    "j40_uv": _r(meas.points_uv[40.0][j]),
                    "j60_uv": _r(meas.points_uv[60.0][j]),
                    "j80_uv": _r(meas.points_uv[80.0][j]),
                    "slope_mv_per_s": _r(meas.slope_mv_per_s[j], 2),
                    "t_amp_uv": _r(meas.t_amp_uv[j]),
                })
            per_lead[lead] = entry
        out["leads"] = per_lead
        out["warnings"] = self._qrs_warning(st, fid.qrs_ms, "")
        if np.isfinite(tpl.hr_bpm) and tpl.hr_bpm >= 100:
            out["warnings"].append(f"heart rate {tpl.hr_bpm:.0f} bpm: the fixed J+80 point may lie on the T-wave upstroke "
                                   "and the isoelectric point may approach the P wave; judge the ST segment shape from "
                                   "J+20 to J+60 and the image, not from J+40 to J+80")
        out["notes"] = ["st_uv: ST level relative to this window's isoelectric point; reference_uv: the lead's usual "
                        "ST level; dev_uv = st_uv - reference_uv. When reference_uv is far from 0, dev_uv and st_uv "
                        "differ: say which one a conclusion rests on",
                        "slope_mv_per_s runs from J to the ST point; j20_uv to j80_uv show the ST segment itself"]
        return out

    def view_ecg(self, record_id: str, t_s: float, duration_s: float = 10.0) -> dict:
        """Numbers that go with a raw-ECG strip: time range, beat labels and heart rate."""
        st = self._state(record_id)
        rec = st.record
        start, end = self.strip_range(record_id, t_s, duration_s)
        ann = rec.annotations
        sl = ann.beats_between(rec.to_samples(start), rec.to_samples(end))
        labels, counts = np.unique(ann.beat_symbols[sl], return_counts=True)
        rr = np.diff(ann.beat_samples[sl]) / rec.fs
        return {"start_s": _r(start, 1), "end_s": _r(end, 1), "hms": hms(start), "leads": list(rec.leads),
                "beat_labels": {str(k): int(v) for k, v in zip(labels, counts)},
                "hr_bpm": _r(60.0 / np.median(rr), 1) if len(rr) else None}

    def compare_morphology(self, record_id: str, t_ref_s: float, t_s: float, duration_s: float = 30.0) -> dict:
        st = self._state(record_id)
        a = self._window(st, t_ref_s, duration_s)
        b = self._window(st, t_s, duration_s)
        mc = compare(a.template, b.template, self.cfg.morphology)
        out = {
            "t_ref_s": _r(t_ref_s), "t_s": _r(t_s),
            "hr_ref": _r(a.template.hr_bpm, 1), "hr": _r(b.template.hr_bpm, 1),
            "qrs_ms_ref": _r(mc.qrs_ms_a), "qrs_ms": _r(mc.qrs_ms_b),
            "frontal_axis_ref_deg": _r(mc.axis_a_deg), "frontal_axis_deg": _r(mc.axis_b_deg),
            "axis_change_deg": _r(mc.axis_change_deg),
            "alignment_lag_ms": _r(mc.lag_ms),
            "qrs_changed_leads": mc.changed_leads,
            "thresholds": {"min_qrs_corr": self.cfg.morphology.min_qrs_corr,
                           "pp_ratio_range": list(self.cfg.morphology.pp_ratio_range)},
            "leads": {},
        }
        warnings = self._qrs_warning(st, mc.qrs_ms_a, "reference window: ") + self._qrs_warning(st, mc.qrs_ms_b, "")
        for j, lead in enumerate(mc.leads):
            entry = {"qrs_corr": _r(mc.qrs_corr[j], 3), "qrs_pp_ratio": _r(mc.pp_ratio[j], 2),
                     "qrs_pp_ref_uv": _r(mc.pp_a_uv[j]), "qrs_pp_change_uv": _r(mc.pp_b_uv[j] - mc.pp_a_uv[j]),
                     "polarity_flip": bool(mc.polarity_flip[j])}
            st_change = np.nan
            if a.st is not None and b.st is not None and a.st.valid and b.st.valid:
                st_change = float(b.st.level_uv[j] - a.st.level_uv[j])
                entry["st_change_uv"] = _r(st_change)
                entry["t_amp_change_uv"] = _r(b.st.t_amp_uv[j] - a.st.t_amp_uv[j])
            entry["usable_both"] = bool(a.quality.usable[j] and b.quality.usable[j])
            out["leads"][lead] = entry
            if lead in mc.changed_leads:
                pp = float(mc.pp_a_uv[j])
                if np.isfinite(pp) and pp < LOW_QRS_PP_UV:
                    warnings.append(f"{lead}: QRS change flagged on a small QRS ({pp:.0f} uV peak-to-peak), where "
                                    "correlation and amplitude ratio are unstable; it does not by itself show an "
                                    "axis or conduction change")

        if "I" not in mc.leads or "aVF" not in mc.leads:
            out["axis_note"] = "frontal axis needs leads I and aVF, which this record does not have"
        out["warnings"] = warnings
        return out

    def signal_quality(self, record_id: str, start_s: float, end_s: float) -> dict:
        st = self._state(record_id)
        tab = st.table
        start_s, end_s = self._check_range(st, start_s, end_s)
        t = tab["t_start_s"]
        sel = (t >= start_s) & (t < end_s)
        if not sel.any():
            raise ToolError("no analysis windows in that range")
        with np.errstate(all="ignore"):
            out = {
                "start_s": _r(start_s), "end_s": _r(end_s), "n_windows": int(sel.sum()),
                "fiducials_ok_frac": _r(tab["fiducials_ok"][sel].mean(), 2),
                "leads": {
                    lead: {
                        "usable_frac": _r(tab["usable"][sel, j].mean(), 2),
                        "noise_uv_median": _r(np.nanmedian(tab["noise_uv"][sel, j])),
                        "noise_uv_p90": _r(np.nanpercentile(tab["noise_uv"][sel, j], 90)),
                        "saturated_window_frac": _r((tab["sat_frac"][sel, j] > 0).mean(), 2),
                        "flat_window_frac": _r((tab["flat_frac"][sel, j] > self.cfg.quality.max_flat_frac).mean(), 2),
                    }
                    for j, lead in enumerate(tab.leads)
                },
            }
        return out

    def rhythm_context(self, record_id: str, start_s: float, end_s: float) -> dict:
        st = self._state(record_id)
        start_s, end_s = self._check_range(st, start_s, end_s)
        return rhythm_context(st.record, start_s, end_s)

    def episode_dynamics(self, record_id: str, candidate_id: str, context_s: float = 600.0,
                         reference: str = "global") -> dict:
        """Heart rate around the episode and the time course of the ST change.

        Heart-rate features follow Panza et al. (J Am Coll Cardiol 1992;20:1092): 89% of
        ambulatory ischemic episodes were preceded by a heart-rate rise of >=10 bpm, largest
        in the 5 min before onset. Onset/offset durations quantify the "gradual onset and
        offset" of the ACC/AHA 1999 ambulatory ECG guideline.
        """
        st = self._state(record_id)
        tab = st.table
        cand = self.candidate(record_id, candidate_id)
        dev = st_deviation(tab, reference)
        t = tab["t_start_s"]
        ws = tab.window_s
        hr = tab["hr_bpm"]
        onset, end = cand.start_s, cand.end_s
        t_peak = cand.peak.peak_s
        inside = (t >= onset) & (t < end)
        around = (t >= onset - context_s) & (t < end + context_s)

        def hr_between(a: float, b: float, fn=np.nanmedian) -> float:
            m = (t >= a) & (t < b)
            with np.errstate(all="ignore"):
                return float(fn(hr[m])) if m.any() and np.isfinite(hr[m]).any() else float("nan")

        base_hr = hr_between(onset - 1800.0, onset - 600.0)
        max_pre5 = hr_between(onset - 300.0, onset, np.nanmax)
        win = (t >= onset - 600.0) & (t < end) & np.isfinite(hr)
        t_hr_max = float(t[win][np.argmax(hr[win])] + ws / 2) if win.any() else float("nan")
        out: dict = {
            "id": cand.id, "start_s": _r(onset), "end_s": _r(end), "duration_s": _r(cand.duration_s),
            "peak_lead": cand.peak.lead, "peak_s": _r(t_peak),
            "hr": {
                "baseline_30_to_10_min_before": _r(base_hr, 1),
                **{f"at_minus_{m}_min": _r(hr_between(onset - 60 * m - 30, onset - 60 * m + 30), 1)
                   for m in (15, 5, 1)},
                "max_in_5_min_before_onset": _r(max_pre5, 1),
                "rise_5_min_before_onset": _r(max_pre5 - base_hr, 1),
                "at_onset": _r(hr_between(onset - 30, onset + 30), 1),
                "at_st_peak": _r(hr_between(t_peak - 30, t_peak + 30), 1),
                "at_end": _r(hr_between(end - 30, end + 30), 1),
                "max_in_episode": _r(hr_between(onset, end, np.nanmax), 1),
                "hr_max_minus_st_peak_s": _r(t_hr_max - t_peak),
            },
            "leads": {},
        }
        involved = sorted({e.lead for e in cand.lead_episodes})
        idx_in = np.flatnonzero(inside)
        busy = self._busy(record_id, reference)
        for lead in involved:
            j = tab.lead_index(lead)
            d = dev[:, j]
            seg = d[inside]
            if not np.isfinite(seg).any():
                out["leads"][lead] = {"usable_frac_in_episode": 0.0}
                continue
            k = int(np.nanargmax(np.abs(seg)))
            peak = float(seg[k])
            big = np.abs(np.nan_to_num(seg)) >= 0.9 * abs(peak)
            first_big, last_big = int(np.argmax(big)), len(big) - 1 - int(np.argmax(big[::-1]))
            i0 = idx_in[0]
            path = d[max(0, i0 - 4):i0 + k + 1]
            steps_ = np.abs(np.diff(path))
            steps_ = steps_[np.isfinite(steps_)]
            span = np.nanmax(path) - np.nanmin(path) if np.isfinite(path).sum() >= 2 else np.nan
            a = around & np.isfinite(d) & np.isfinite(hr)
            rho = slope = np.nan
            if a.sum() >= 8 and np.nanstd(hr[a]) > 0:
                rho = stats.spearmanr(hr[a], d[a]).statistic
                slope = stats.theilslopes(d[a], hr[a]).slope
            # ST/HR slope by phase: from onset to this lead's extremum and from the extremum to the end
            t_lead_peak = t[idx_in[k]]
            phase = {}
            for name, sel in (("to_extremum", inside & (t <= t_lead_peak)), ("from_extremum", inside & (t >= t_lead_peak))):
                sel = sel & np.isfinite(d) & np.isfinite(hr)
                ok = sel.sum() >= 4 and np.ptp(hr[sel]) >= 3.0
                phase[name] = stats.theilslopes(d[sel], hr[sel]).slope if ok else np.nan
            # Same heart rate elsewhere: windows outside the candidate (+-60 s) whose heart rate is within
            # SAME_HR_BPM of the heart rate at this lead's extremum.
            hr_pk = hr_between(t_lead_peak - 30, t_lead_peak + 30)
            other = ~busy & np.isfinite(d) & np.isfinite(hr)  # outside every ST episode of the record
            same = other & (np.abs(hr - hr_pk) <= SAME_HR_BPM) if np.isfinite(hr_pk) else np.zeros_like(other)
            diff = other & (np.abs(hr - hr_pk) > 2 * SAME_HR_BPM) if np.isfinite(hr_pk) else np.zeros_like(other)
            same_hr = {"hr_at_extremum": _r(hr_pk, 1), "n_windows": int(same.sum()),
                       "too_few_windows": bool(same.sum() < 20)}
            if same.sum() >= 4:
                same_hr["median_dev_uv"] = _r(np.median(d[same]))
                same_hr["frac_windows_with_half_the_extremum_or_more"] = _r(
                    np.mean(np.sign(peak) * d[same] >= 0.5 * abs(peak)), 2)
            if diff.sum() >= 4:
                same_hr["median_dev_at_other_hr_uv"] = _r(np.median(d[diff]))
            # Was the change already under way when the candidate started?
            pre = (t >= onset - 300) & (t < onset) & np.isfinite(d)
            first = seg[np.isfinite(seg)][0]
            start_frac = first / peak if peak != 0 else np.nan
            out["leads"][lead] = {
                "peak_dev_uv": _r(peak),
                "time_to_peak_s": _r(k * ws),
                "onset_to_90pct_s": _r(first_big * ws),
                "90pct_to_end_s": _r((len(seg) - 1 - last_big) * ws),
                "plateau_frac_ge_80pct": _r((np.abs(np.nan_to_num(seg)) >= 0.8 * abs(peak)).mean(), 2),
                "max_single_step_frac": _r(steps_.max() / span, 2) if steps_.size and span > 0 else None,
                "st_hr_spearman": _r(rho, 2),
                "st_hr_slope_uv_per_bpm": _r(slope, 1),
                "st_hr_slope_to_extremum_uv_per_bpm": _r(phase["to_extremum"], 1),
                "st_hr_slope_from_extremum_uv_per_bpm": _r(phase["from_extremum"], 1),
                "usable_frac_in_episode": _r(np.isfinite(seg).mean(), 2),
                "dev_at_candidate_start_frac_of_peak": _r(start_frac, 2),
                "median_dev_5_min_before_start_uv": _r(np.median(d[pre])) if pre.any() else None,
                "same_hr_elsewhere_in_record": same_hr,
            }
        signs = {e.lead: e.sign for e in cand.lead_episodes}
        if has_contiguous_pair(tab.leads):
            out["territories"] = {name: [lead for lead in leads if lead in involved]
                                  for name, leads in TERRITORIES.items() if any(lead in involved for lead in leads)}
        else:
            out["territories"] = (f"not applicable: the record's leads ({', '.join(tab.leads)}) are not standard "
                                  "12-lead leads of one territory")
        out["depressed_leads"] = sorted(lead for lead, s in signs.items() if s < 0)
        out["elevated_leads"] = sorted(lead for lead, s in signs.items() if s > 0)
        out["notes"] = [
            "hr_max_minus_st_peak_s < 0: heart rate peaked before the ST extremum",
            "max_single_step_frac near 1 or short onset/offset: step-like change; long onset/offset: gradual",
            "st_hr_spearman and st_hr_slope_uv_per_bpm use the episode plus +-context_s",
            "same_hr_elsewhere_in_record: ST deviation in windows outside every ST episode of the record (+-60 s) whose heart rate is "
            f"within {SAME_HR_BPM:.0f} bpm of the heart rate at the extremum, and at heart rates more than "
            f"{2 * SAME_HR_BPM:.0f} bpm away. If ST at the same heart rate is near the reference (median far below the "
            "extremum, few windows with half of it), the heart-rate level alone does not account for the change; "
            "this supports but does not establish ischaemia. Heart-rate dependence is suggested if ST at the same "
            "heart rate is similar to the episode AND differs from ST at other heart rates; if both are similar, "
            "the lead's ST often sits at this level (other episodes or a shifted baseline), which st_trend can show",
            "dev_at_candidate_start_frac_of_peak (first usable window) near 1 or a large "
            "median_dev_5_min_before_start_uv: the change was "
            "already under way when the candidate started, so onset durations and the heart-rate rise before onset "
            "are measured from too late a start; look at st_trend before the candidate",
            "st_hr_slope_to/from_extremum: ST vs heart rate within the episode, onset to this lead's extremum "
            "and extremum to end (null if <4 windows or heart rate varies <3 bpm); in LTSTDB, ischaemic "
            "episodes had steeper ST/HR slopes than heart-rate related ones (Faganeli Pucer 2012)",
        ]
        return out

    def check_criteria(self, record_id: str, candidate_id: str, rule: str = "acc_aha_1999",
                       reference: str = "global") -> dict:
        """Apply the ACC/AHA 1999 ambulatory ischemia criterion numerically.

        Crawford et al., J Am Coll Cardiol 1999;34:912-948: "flat or downsloping ST-segment
        depression >=0.1 mV, with a gradual onset and offset that lasts for a minimum period of
        1 minute. Each episode ... must be separated by a minimum duration of at least 1 minute
        ... (1 x 1 x 1 rule) ... We recommend a 5-minute interval between episodes".
        Slope and onset shape are reported, not judged.
        """
        if rule not in CRITERIA_RULES:
            raise ToolError(f"supported rules: {list(CRITERIA_RULES)}")
        st = self._state(record_id)
        tab = st.table
        cand = self.candidate(record_id, candidate_id)
        dev = st_deviation(tab, reference)
        t = tab["t_start_s"]
        ws = tab.window_s
        if rule == "contiguous_leads":
            return self._contiguous_leads(cand, dev, t, ws, tab.leads)
        need = int(math.ceil(60.0 / ws))
        inside = (t >= cand.start_s) & (t < cand.end_s)
        per_lead = {}
        best = 0
        for j, lead in enumerate(tab.leads):
            x = np.where(inside, dev[:, j] <= -100.0, False)
            run = longest = 0
            for v in x:
                run = run + 1 if v else 0
                longest = max(longest, run)
            if longest:
                slopes = tab["st_slope"][inside & (dev[:, j] <= -100.0), j]
                per_lead[lead] = {"longest_run_s": _r(longest * ws),
                                  "median_st_slope_mv_per_s": _r(np.nanmedian(slopes), 2)}
            best = max(best, longest)
        prev = [c for c in self.candidates(record_id, cand.protocol, reference) if c.end_s <= cand.start_s]
        gap = cand.start_s - max(c.end_s for c in prev) if prev else None
        return {
            "rule": "ACC/AHA 1999: >=100 uV flat/downsloping depression lasting >=60 s; episodes separated "
                    "by >=60 s (1x1x1), 300 s recommended",
            "source": "Crawford et al., J Am Coll Cardiol 1999;34:912-948, section II.F.2",
            "depression_ge_100uv_ge_60s": best >= need,
            "separation_from_previous_s": _r(gap),
            "separation_ge_60s": gap is None or gap >= 60.0,
            "separation_ge_300s_recommended": gap is None or gap >= 300.0,
            "leads_meeting_depression": per_lead,
            "not_judged_here": ["flat or downsloping morphology (see slopes)",
                                "gradual onset and offset (see episode_dynamics)"],
            "scope": "a criterion for a positive finding of ST depression; not meeting it does not show that the "
                     "change is non-ischaemic, and it does not address ST elevation",
        }

    @staticmethod
    def _contiguous_leads(cand: Candidate, dev: np.ndarray, t: np.ndarray, ws: float, leads: list[str]) -> dict:
        """ST change of >=100 uV in >=2 contiguous leads for >=30 s (Schipaanboord et al. 2026).

        "Contiguous" is approximated by membership of the same anatomical lead group
        (TERRITORIES); the exact 12-lead definition should be agreed with clinicians.
        """
        if not has_contiguous_pair(leads):
            return {"rule": "ST change >=100 uV in >=2 contiguous leads for >=30 s", "applicable": False,
                    "reason": f"the record's leads ({', '.join(leads)}) contain no two leads of one standard 12-lead "
                              "group, so contiguity cannot be assessed; this says nothing for or against ischaemia"}
        need = int(math.ceil(30.0 / ws))
        inside = np.flatnonzero((t >= cand.start_s) & (t < cand.end_s))
        groups_idx = {g: [leads.index(x) for x in members if x in leads] for g, members in TERRITORIES.items()}
        out: dict = {
            "rule": "ST change >=100 uV in >=2 contiguous leads for >=30 s", "applicable": True,
            "source": "Schipaanboord et al., IJC Heart Vasc 2026;63:101884 (12-lead Holter; they explored "
                      "cut-offs of 0-0.5 mV, e.g. ST depression >=0.1 mV)",
            "contiguity": "same anatomical group: " + ", ".join(f"{k} {'/'.join(v)}" for k, v in TERRITORIES.items()),
        }
        for sign, name in ((-1, "depression"), (1, "elevation")):
            best_run, best_groups, run, groups_run = 0, set(), 0, set()
            for w in inside:
                hit_groups = set()
                for group, idx in groups_idx.items():
                    vals = sign * dev[w, idx]
                    if (np.nan_to_num(vals, nan=-np.inf) >= 100.0).sum() >= 2:
                        hit_groups.add(group)
                if hit_groups:
                    run += 1
                    groups_run |= hit_groups
                    if run > best_run:
                        best_run, best_groups = run, set(groups_run)
                else:
                    run, groups_run = 0, set()
            out[name] = {"met": best_run >= need, "longest_run_s": _r(best_run * ws),
                         "groups": sorted(best_groups)}
        return out

    def morphology_steps(self, record_id: str, start_s: float, end_s: float,
                         sync_tolerance_s: float = 60.0) -> dict:
        """Steps in QRS morphology and in ST level within a range, and which ones coincide."""
        st = self._state(record_id)
        tab = st.table
        start_s, end_s = self._check_range(st, start_s, end_s)
        if st.steps is None:
            st.steps = record_steps(tab)
        t = tab["t_start_s"]
        axis = frontal_axis_series(tab)

        def around(t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
            return (t >= t0 - FLAT_S) & (t < t0), (t >= t1) & (t < t1 + FLAT_S)

        def overlaps(s: Step) -> bool:
            return s.t_end_s >= start_s and s.t_start_s < end_s

        qrs_steps = []
        for s in st.steps["QRS_distance"]:
            if not overlaps(s):
                continue
            b, a = around(s.t_start_s, s.t_end_s)
            with np.errstate(all="ignore"):
                area_change = np.nanmedian(tab["qrs_area_uvs"][a], 0) - np.nanmedian(tab["qrs_area_uvs"][b], 0)
                qrs_ms = (np.nanmedian(tab["qrs_ms"][b]), np.nanmedian(tab["qrs_ms"][a]))
                ax = (np.nanmedian(axis[b]), np.nanmedian(axis[a]))
            top = np.argsort(-np.nan_to_num(np.abs(area_change)))[:4]
            qrs_steps.append({
                "t_start_s": _r(s.t_start_s), "t_end_s": _r(s.t_end_s), "hms": hms(s.t_start_s),
                "distance_change": _r(s.delta, 2),
                "frontal_axis_before_after_deg": [_r(ax[0]), _r(ax[1])],
                "axis_change_deg": _r((ax[1] - ax[0] + 180.0) % 360.0 - 180.0),
                "qrs_ms_before_after": [_r(qrs_ms[0]), _r(qrs_ms[1])],
                "largest_qrs_area_changes_uvs": {tab.leads[i]: _r(area_change[i]) for i in top},
            })
        st_steps = [{"lead": k.split(":", 1)[1], "t_start_s": _r(s.t_start_s), "t_end_s": _r(s.t_end_s),
                     "st_change_uv": _r(s.delta)}
                    for k, v in st.steps.items() if k.startswith("ST:") for s in v
                    if overlaps(s)]
        def coincide(q: dict, s: dict) -> bool:
            # Transition intervals overlap, or their starts are within the tolerance.
            overlap = min(q["t_end_s"], s["t_end_s"]) >= max(q["t_start_s"], s["t_start_s"])
            return overlap or abs(q["t_start_s"] - s["t_start_s"]) <= sync_tolerance_s

        sync = [{"qrs_step_t_s": q["t_start_s"], "st_step_lead": s["lead"], "st_step_t_s": s["t_start_s"],
                 "st_change_uv": s["st_change_uv"]}
                for q in qrs_steps for s in st_steps if coincide(q, s)]
        return {
            "start_s": _r(start_s), "end_s": _r(end_s),
            "qrs_steps": qrs_steps, "st_steps": sorted(st_steps, key=lambda s: s["t_start_s"]),
            "synchronous_st_and_qrs_steps": sync,
            "method": "flat 216 s -> change within 144 s -> flat (Smrdel & Jager, CinC 2002); QRS distance "
                      "thresholds 0.33/0.5 are uncalibrated for this distance; steps are detected on the whole "
                      "record and listed when their transition overlaps the range",
            "warnings": ([] if end_s - start_s >= 600.0 else
                         [f"range of {end_s - start_s:.0f} s: only steps whose transition falls inside it are listed, "
                          "and slower changes are not steps; an empty result here does not rule out a QRS change. "
                          "Use a range that starts >=300 s before onset and ends >=300 s after the extremum"]),
        }

    def patient_context(self, record_id: str) -> dict:
        """Clinical information about the patient from the record header (see data/clinical.py)."""
        st = self._state(record_id)
        out = {"record": st.record.name, **parse_clinical(st.record.comments)}
        if out["available"]:
            out["note"] = ("Clinical context changes how likely ischaemia is for this patient; it does not show "
                           "whether a given episode is ischaemic.")
        return out

    def shape_changes(self, record_id: str, candidate_id: str) -> dict:
        """QRS vs ST-T shape change from before onset to the extremum, and ST-segment shape per lead."""
        st = self._state(record_id)
        tab = st.table
        cand = self.candidate(record_id, candidate_id)
        if st.shapes is None:
            st.shapes = {"qrs": shape_scores(tab, QRS_FEATURES), "st_t": shape_scores(tab, ST_T_FEATURES)}
        t = tab["t_start_s"] + tab.window_s / 2
        onset, t_peak = cand.start_s, cand.peak.peak_s
        before = (t >= onset - 600.0) & (t < onset)
        at = np.abs(t - t_peak) <= 30.0
        near = (t >= onset - 120.0) & (t <= onset + 120.0)
        lag = max(1, int(round(30.0 / tab.window_s)))
        dq = shape_change(st.shapes["qrs"], before, at)
        ds = shape_change(st.shapes["st_t"], before, at)
        out: dict = {
            "id": cand.id, "before": "10 min before onset", "extremum_s": _r(t_peak),
            "qrs_shape_change_sd": _r(dq, 2), "st_t_shape_change_sd": _r(ds, 2),
            "qrs_to_st_t_change_ratio": _r(dq / (ds + 0.1), 2) if np.isfinite(dq) and np.isfinite(ds) else None,
            "largest_30s_qrs_jump_near_onset_sd": _r(largest_jump(st.shapes["qrs"], near, lag), 2),
            "typical_30s_qrs_jump_in_record_sd": _r(typical_jump(st.shapes["qrs"], lag), 2),
            "st_segment": {},
        }
        involved = sorted({e.lead for e in cand.lead_episodes})
        shapes = {}
        for label, t0s in (("before_onset", (onset - 180.0, onset - 360.0, onset - 600.0)), ("at_extremum", (t_peak,))):
            for t0 in t0s:
                if t0 < 0:
                    continue
                try:
                    wa = self._window(st, t0, 30.0)
                except ToolError:
                    continue
                tpl = wa.template
                if tpl is None or not tpl.fiducials.ok:
                    continue
                iso = tpl.mean_at(tpl.fiducials.iso_ms, self.cfg.st.point_halfwidth_ms)
                shapes[label] = (_r(t0), st_segment_shape(tpl.data, tpl.fs, tpl.index_at(tpl.fiducials.j_ms), iso),
                                 tpl.leads)
                break
        for lead in involved:
            entry = {}
            for label, (t0, sh, leads) in shapes.items():
                if lead in leads:
                    lv, rise, curv = sh[leads.index(lead)]
                    entry[label] = {"t_s": t0, "level_uv": _r(lv), "rise_uv": _r(rise), "curvature_uv": _r(curv)}
            out["st_segment"][lead] = entry
        out["notes"] = [
            "shape changes are RMS distances in record SD units between the median shape 10 min before onset "
            "and within 30 s of the extremum; QRS: per-lead QRS area and amplitude; ST-T: per-lead ST at J, ST "
            "at the measurement point, ST slope and T amplitude",
            "a QRS change large relative to the ST-T change points to an axis or body-position shift; ischaemic "
            "changes show mainly in the ST-T shape (PhysioNet LTSTDB)",
            "st_segment: 120 ms after J, relative to the isoelectric level; level = mean, rise = change from J to "
            "J+120 ms, curvature = ends minus middle (positive: concave upward, sagging or scooped)",
        ]
        return out

    def ischemia_estimate(self, record_id: str, candidate_id: str, protocol: str = "B",
                          reference: str = "global") -> dict:
        """Statistical estimate from the fixed check set (agent/estimate.py), fitted on LTSTDB development records."""
        from .baseline import collect_standard_evidence  # imports this module
        from .estimate import IschemiaEstimator, features
        from .state import CaseState
        if self.estimator is None:
            self.estimator = IschemiaEstimator()
        cand = self.candidate(record_id, candidate_id)
        summary = jsonable(cand.summary())
        state = CaseState(self._state(record_id).record.name)
        bound = {"protocol": protocol, "reference": reference}
        ev = {k: r for k, (r, _) in collect_standard_evidence(self, state, summary, bound, "estimate").items()}
        peak = max((d["peak_uv"] for d in summary["leads"].values()), key=abs)
        f = features(ev, {"peak_uv": peak, "start_s": cand.start_s, "end_s": cand.end_s})
        p, contrib = self.estimator.predict(f)
        meta = self.estimator.meta
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
        return {
            "id": cand.id, "probability_ischemic": _r(p, 2),
            "largest_contributions": {n: {"value": _r(f.get(n), 2), "log_odds": _r(c, 2)} for n, c in top},
            "model": (f"logistic regression on the outputs of a fixed check set (signal_quality, rhythm_context, "
                      f"measure_window and compare_morphology at the extremum, morphology_steps, episode_dynamics, "
                      f"check_criteria); fitted on {meta.get('fitted_on', '?')}; cross-validated AUC "
                      f"{meta.get('dev_cv_auc', float('nan')):.2f}; ischemic share there "
                      f"{meta.get('dev_ischemic_fraction', float('nan')):.0%}"),
            "notes": ["one piece of evidence, not a verdict: it reproduces that database's labels, which partly "
                      "rest on clinical information the tools do not see, and was fitted on 2-3 lead recordings",
                      "positive log_odds push towards ischemic; check whether the ECG findings agree"],
        }

    def episode_context(self, record_id: str, candidate_id: str, protocol: str = "B",
                        reference: str = "global") -> dict:
        """Where the candidate sits in the record: the full contiguous ST change around it, the local level before
        that change, both extremes of every lead, and the neighbouring candidates (toolkit 0.8).

        Candidate boundaries come from a threshold detector on deviations from a record-wide reference, so a
        candidate can be a fragment of a longer episode, the return between two episodes, or a change relative to
        a lead whose usual level is far from zero; this tool shows which.
        """
        st = self._state(record_id)
        tab = st.table
        cand = self.candidate(record_id, candidate_id)
        t, ws = tab["t_start_s"], tab.window_s
        dev = st_deviation(tab, reference)
        ref = st_reference(tab, reference)
        absolute = np.where(tab["usable"], tab["st_uv"], np.nan)
        busy = self._busy(record_id, reference)
        inside = (t >= cand.start_s) & (t < cand.end_s)
        idx_in = np.flatnonzero(inside)
        out: dict = {"id": cand.id, "start_s": _r(cand.start_s), "end_s": _r(cand.end_s), "leads": {}}
        for j, lead in enumerate(tab.leads):
            d = dev[:, j]
            seg = d[inside]
            if not np.isfinite(seg).any():
                out["leads"][lead] = {"usable_in_candidate": False}
                continue
            kx, kn = int(np.nanargmax(seg)), int(np.nanargmin(seg))
            k = kx if abs(seg[kx]) >= abs(seg[kn]) else kn
            peak, ip = float(seg[k]), int(idx_in[k])
            sign = 1.0 if peak >= 0 else -1.0
            thr = max(50.0, 0.5 * abs(peak))
            lo = hi = ip
            gap = 0
            while lo - 1 >= 0 and gap <= 2:  # extend while the change persists (short unusable gaps allowed)
                v = d[lo - 1]
                gap = gap + 1 if not np.isfinite(v) else (0 if sign * v >= thr else 99)
                if gap > 2:
                    break
                lo -= 1
            gap = 0
            while hi + 1 < len(d) and gap <= 2:
                v = d[hi + 1]
                gap = gap + 1 if not np.isfinite(v) else (0 if sign * v >= thr else 99)
                if gap > 2:
                    break
                hi += 1
            span0, span1 = float(t[lo]), float(t[hi] + ws)
            pre = (t >= span0 - 1200.0) & (t < span0 - 120.0) & np.isfinite(d)
            pre_clean = pre & ~busy
            use = pre_clean if pre_clean.sum() >= 4 else pre
            local = float(np.median(d[use])) if use.sum() >= 4 else float("nan")
            near = slice(max(0, ip - 1), ip + 2)
            with np.errstate(all="ignore"):
                abs_peak = float(np.nanmedian(absolute[near, j]))
            out["leads"][lead] = {
                "involved": lead in {e.lead for e in cand.lead_episodes},
                "max_dev_uv": _r(seg[kx]), "max_at_s": _r(t[idx_in[kx]]),
                "min_dev_uv": _r(seg[kn]), "min_at_s": _r(t[idx_in[kn]]),
                "reference_uv": _r(ref[ip, j]),
                "absolute_st_at_extremum_uv": _r(abs_peak),
                "change_span": {"start_s": _r(span0), "end_s": _r(span1), "hms": hms(span0),
                                "duration_s": _r(span1 - span0),
                                "starts_before_candidate_s": _r(max(0.0, cand.start_s - span0)),
                                "ends_after_candidate_s": _r(max(0.0, span1 - cand.end_s)),
                                "threshold_uv": _r(sign * thr)},
                "local_level_before_change_uv": _r(local),
                "n_local_windows": int(use.sum()),
                "extremum_minus_local_level_uv": _r(peak - local) if np.isfinite(local) else None,
            }
        neighbours = []
        for c in self.candidates(record_id, protocol, reference):
            if c.id == cand.id or c.end_s < cand.start_s - 3600 or c.start_s > cand.end_s + 3600:
                continue
            gap = (c.start_s - cand.end_s) if c.start_s >= cand.end_s else (c.end_s - cand.start_s
                                                                             if c.end_s <= cand.start_s else 0.0)
            neighbours.append({"id": c.id, "start_hms": hms(c.start_s), "duration_s": _r(c.duration_s),
                               "gap_s": _r(gap), "peak_lead": c.peak.lead, "peak_uv": _r(c.peak.peak_uv)})
        out["neighbouring_candidates_within_1h"] = neighbours
        out["notes"] = [
            "change_span: contiguous windows around this lead's extremum where the deviation stays beyond max(50 uV, "
            "half the extremum) with the same sign; it can extend beyond the candidate, whose boundaries come from a "
            "threshold detector",
            "local_level_before_change_uv: median deviation 2-20 min before the change span, outside ST episodes when "
            "possible; extremum_minus_local_level_uv is the change itself, which can differ from the deviation from "
            "the record-wide reference",
            "absolute_st_at_extremum_uv is relative to the isoelectric level; reference_uv is the lead's usual level",
        ]
        return out

    def rate_response(self, record_id: str, candidate_id: str, reference: str = "global",
                      min_rise_bpm: float = 15.0) -> dict:
        """How this patient's ST responds to heart-rate surges across the whole record, and where the candidate
        falls (toolkit 0.8).

        For every heart-rate surge (a local maximum at least min_rise_bpm above the minimum of the preceding
        10 min), the ST change in each involved lead from the pre-surge minimum to the largest change within
        -60..+180 s of the heart-rate peak (ST lags heart rate), in uV per bpm, with whether the surge lies in an
        ST episode. Also the median ST deviation per 10-bpm heart-rate band outside ST episodes, next to the
        candidate's own values.
        """
        st = self._state(record_id)
        tab = st.table
        cand = self.candidate(record_id, candidate_id)
        t, ws = tab["t_start_s"], tab.window_s
        dev = st_deviation(tab, reference)
        hr = tab["hr_bpm"].astype(float)
        hs = np.array([np.nanmedian(hr[max(0, i - 1):i + 2]) if np.isfinite(hr[max(0, i - 1):i + 2]).any()
                       else np.nan for i in range(len(hr))])
        busy = self._busy(record_id, reference)
        w10 = max(1, int(round(600.0 / ws)))
        lag0, lag1 = int(round(60.0 / ws)), int(round(180.0 / ws))
        surges = []
        i = 0
        while i < len(hs):
            v = hs[i]
            if np.isfinite(v):
                pre = hs[max(0, i - w10):i]
                around = hs[max(0, i - w10):i + w10 + 1]
                if pre.size and np.isfinite(pre).any() and v >= np.nanmax(around) and v - np.nanmin(pre) >= min_rise_bpm:
                    surges.append((i, max(0, i - w10) + int(np.nanargmin(pre))))
                    i += w10
                    continue
            i += 1
        involved = sorted({e.lead for e in cand.lead_episodes})
        sign = {e.lead: e.sign for e in cand.lead_episodes}
        in_cand = (t >= cand.start_s - 60) & (t < cand.end_s + 60)
        out: dict = {"id": cand.id, "min_rise_bpm": min_rise_bpm, "n_surges": len(surges), "leads": {}}
        for lead in involved:
            j = tab.lead_index(lead)
            d, sg = dev[:, j], sign[lead]
            rows = []
            for ip, ib in surges:
                base = d[max(0, ib - 1):ib + 2]
                win = d[max(0, ip - lag0):ip + lag1 + 1]
                if not (np.isfinite(base).any() and np.isfinite(win).any()):
                    continue
                st_base = float(np.nanmedian(base))
                st_peak = float(win[np.nanargmax(sg * win)])
                rise = float(hs[ip] - hs[ib])
                rows.append({"hms": hms(t[ip]), "hr_from": _r(hs[ib]), "hr_to": _r(hs[ip]),
                             "st_change_uv": _r(st_peak - st_base), "uv_per_bpm": _r((st_peak - st_base) / rise, 1),
                             "in_st_episode": bool(busy[ip]), "this_candidate": bool(in_cand[ip])})
            quiet = [r["uv_per_bpm"] for r in rows if not r["in_st_episode"] and r["uv_per_bpm"] is not None]
            # The candidate's own response: from the minimum heart rate of the 10 min before onset to its maximum
            # within the candidate, and the ST change from the local level to this lead's extremum.
            pre = (t >= cand.start_s - 600) & (t < cand.start_s) & np.isfinite(hs)
            inn = (t >= cand.start_s) & (t < cand.end_s)
            own = None
            if pre.any() and np.isfinite(hs[inn]).any() and np.isfinite(d[inn]).any():
                hr_rise = float(np.nanmax(hs[inn]) - np.nanmin(hs[pre]))
                st_pre = np.nanmedian(d[pre]) if np.isfinite(d[pre]).any() else np.nan
                st_ext = float(d[inn][np.nanargmax(sg * d[inn])])
                own = {"hr_rise_bpm": _r(hr_rise, 1), "st_change_uv": _r(st_ext - st_pre),
                       "uv_per_bpm": _r((st_ext - st_pre) / hr_rise, 1) if hr_rise >= 5 else None}
                if own["uv_per_bpm"] is not None and quiet:
                    own["percentile_among_surges_outside_episodes"] = _r(
                        100.0 * np.mean(sg * np.array(quiet) <= sg * own["uv_per_bpm"]))
            bands = {}
            for lo_b in range(40, 200, 10):
                m = (hs >= lo_b) & (hs < lo_b + 10) & np.isfinite(d)
                outside, here = m & ~busy, m & inn
                if outside.sum() >= 4 or here.any():
                    bands[f"{lo_b}-{lo_b + 10}"] = {
                        "outside_episodes_median_uv": _r(np.median(d[outside])) if outside.sum() >= 4 else None,
                        "n_outside": int(outside.sum()),
                        "this_candidate_median_uv": _r(np.median(d[here])) if here.any() else None}
            out["leads"][lead] = {"surges": rows[:30], "n_surges_outside_episodes": len(quiet),
                                  "outside_episodes_uv_per_bpm": {
                                      "median": _r(np.median(quiet), 1) if quiet else None,
                                      "p25": _r(np.percentile(quiet, 25), 1) if quiet else None,
                                      "p75": _r(np.percentile(quiet, 75), 1) if quiet else None},
                                  "this_candidate": own, "st_by_heart_rate_band": bands}
        out["notes"] = [
            "Rate-related ST change and demand ischaemia both grow with heart rate, and ST lags heart rate by tens of "
            "seconds to minutes. This tool compares the candidate with the patient's own response to other heart-rate "
            "surges outside ST episodes: a response far beyond the usual range, or ST at a given heart rate far beyond "
            "the usual level at that heart rate, is unusual for this patient's rate response; a response within the "
            "usual range fits a rate-related change but does not exclude demand ischaemia",
            "uv_per_bpm is signed like the candidate's change (negative for depression)",
        ]
        return out

    def eligibility_check(self, record_id: str) -> dict:
        """Record-level conditions that limit ST interpretation (ACC/AHA 1999, section II.F.2).

        The guideline: sinus rhythm; baseline ST deviation below 0.1 mV; R wave of the monitored
        lead >=10 mm; "Patients whose 12-lead ECG demonstrates left ventricular (LV) hypertrophy,
        preexcitation, left bundle-branch block, or nonspecific intraventricular conduction delay
        >=0.10 second are not suitable"; RBBB "may be interpretable, especially in the left
        precordial leads"; digoxin and some antidepressants distort the ST segment.
        """
        st = self._state(record_id)
        if st.eligibility is not None:
            return st.eligibility
        rec, tab = st.record, st.table
        rc = rhythm_context(rec, 0.0, rec.duration_s)
        n_beats = max(1, rc["n_beats"])
        ok = tab["fiducials_ok"]
        with np.errstate(all="ignore"):
            qrs_med = float(np.nanmedian(tab["qrs_ms"][ok])) if ok.any() else float("nan")
            qrs_wide_frac = float((tab["qrs_ms"][ok] >= 100.0).mean()) if ok.any() else float("nan")
        # R-wave amplitude per lead from a few clean windows spread over the record.
        good = np.flatnonzero(tab["usable"].mean(axis=1) >= 0.75)
        picks = good[np.linspace(0, len(good) - 1, min(12, len(good))).astype(int)] if len(good) else []
        r_amp = []
        for w in picks:
            t0 = float(tab["t_start_s"][w])
            try:
                wa = self._window(st, t0 + tab.window_s / 2, 30.0)
            except ToolError:
                continue
            tpl, fid = wa.template, wa.template.fiducials
            if not fid.ok:
                continue
            q = tpl.data[tpl.index_at(fid.onset_ms):tpl.index_at(fid.j_ms)]
            iso = tpl.mean_at(fid.iso_ms, 8.0)
            r_amp.append((np.nanmax(q, axis=0) - iso) * 1000.0)
        r_med = np.nanmedian(np.array(r_amp), axis=0) if r_amp else np.full(len(tab.leads), np.nan)
        ref = st_reference(tab, "global")[0]
        beat_types = dict(rc["beat_types"])
        normal_frac = beat_types.get("N", 0) / n_beats
        hours: dict[str, float] = {}
        for r in rec.annotations.rhythms:
            hours[r.label] = hours.get(r.label, 0.0) + (r.end - r.start) / rec.fs / 3600.0
        result = {
            "record": rec.name,
            "source": "ACC/AHA 1999 ambulatory ECG guideline (Crawford et al., JACC 34:912), section II.F.2",
            "rhythm": {
                "normal_beat_frac": _r(normal_frac, 3),
                "beat_types": beat_types,
                "rhythm_labels_hours": {k: _r(v, 2) for k, v in hours.items()},
                "possible_pacing_spikes": rc.get("caret_events", {}),
            },
            "qrs_ms_median": _r(qrs_med),
            "qrs_ge_100ms_window_frac": _r(qrs_wide_frac, 2),
            "r_wave_uv_by_lead": _per_lead(tab.leads, r_med),
            "baseline_st_uv_by_lead": _per_lead(tab.leads, ref),
            "flags": {
                "qrs_ge_100ms": bool(np.isfinite(qrs_med) and qrs_med >= 100.0),
                "leads_r_wave_below_1mv": [lead for lead, v in zip(tab.leads, r_med) if np.isfinite(v) and v < 1000.0],
                "leads_baseline_st_abs_ge_100uv": [lead for lead, v in zip(tab.leads, ref)
                                                   if np.isfinite(v) and abs(v) >= 100.0],
                # default threshold, not from the guideline (which asks for sinus rhythm)
                "non_normal_beat_frac_ge_10pct": normal_frac < 0.9,
            },
            "not_assessed": ["LV hypertrophy", "pre-excitation", "bundle-branch block type",
                             "Q waves >=0.04 s", "digoxin / antidepressants / other drugs", "clinical history"],
            "notes": ["baseline ST is measured at the heart-rate adjusted point after J; normal ST elevation "
                      "in V1-V3 can exceed 100 uV",
                      "QRS duration comes from the automatic J point and may be overestimated in noise"],
        }
        st.eligibility = jsonable(result)
        return st.eligibility

    def find_reference_window(self, record_id: str, t_s: float, max_distance_s: float = 7200.0,
                              hr_tolerance_bpm: float | None = 10.0, reference: str = "global") -> dict:
        """Nearest clean window (preferably earlier) with small ST deviation, to compare against."""
        st = self._state(record_id)
        tab = st.table
        dev = st_deviation(tab, reference)
        t = tab["t_start_s"]
        w0 = tab.window_index(t_s)
        hr0 = tab["hr_bpm"][w0]
        usable_frac = tab["usable"].mean(axis=1)
        with np.errstate(all="ignore"):
            max_abs = np.nanmax(np.abs(dev), axis=1)
        in_cand = np.zeros(len(t), bool)
        for c in self.candidates(record_id, "A", reference):
            in_cand |= (t >= c.start_s - 60) & (t < c.end_s + 60)
        ok = (usable_frac >= 0.75) & (max_abs < 50.0) & ~in_cand & (np.abs(t - t_s) <= max_distance_s)
        if hr_tolerance_bpm is not None and np.isfinite(hr0):
            ok &= np.abs(tab["hr_bpm"] - hr0) <= hr_tolerance_bpm
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            raise ToolError("no clean reference window found; widen max_distance_s or drop the HR constraint")
        # Prefer earlier windows: penalise later ones by doubling their distance.
        dist = np.abs(t[cand] - t_s) * np.where(t[cand] < t_s, 1.0, 2.0)
        w = int(cand[np.argmin(dist)])
        return {"t_s": _r(t[w]), "hms": hms(t[w]), "hr_bpm": _r(tab["hr_bpm"][w], 1),
                "hr_at_query": _r(hr0, 1), "usable_frac": _r(usable_frac[w], 2),
                "max_abs_dev_uv": _r(max_abs[w]),
                "criteria": "usable in >=75% of leads, |dev|<50 uV in all usable leads, outside candidates"
                            + ("" if hr_tolerance_bpm is None else f", HR within {hr_tolerance_bpm} bpm")}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    method: str  # HolterToolkit method name


# Arguments fixed by the run configuration, never chosen by the agent. Letting the
# agent switch the ST reference or the episode protocol would let repeated checks
# drift until ischemia "appears" (registered-analysis argument, Hou et al. 2026,
# arXiv:2607.25045).
BOUND_ARGS = ("reference", "protocol")
TOOLKIT_VERSION = "0.8.0"


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_REC = {"record_id": {"type": "string", "description": "Record id, e.g. '0049'."}}
_RANGE = {"start_s": {"type": "number"}, "end_s": {"type": "number"}}
_CAND = {"candidate_id": {"type": "string"}}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("record_overview",
             "Summary of a record: duration, beat and rhythm labels, heart rate, per-lead usable fraction and "
             "reference ST level, number of candidate episodes.",
             _obj(_REC, ["record_id"]), "record_overview"),
    ToolSpec("list_candidates",
             "All candidate ST episodes of the record found by the LTSTDB-style rule used in this run, largest "
             "first. Use it to see whether similar episodes recur.",
             _obj(_REC, ["record_id"]), "list_candidates"),
    ToolSpec("st_trend",
             "Binned ST deviation per lead (uV, ST level minus the run's reference level) and heart rate over "
             "a time range, at most 40 rows. Pass an empty leads list for all leads. Deviation is relative to "
             "a record-level reference, so a slow drift or an axis shift also shows up here.",
             _obj({**_REC, **_RANGE, "leads": {"type": "array", "items": {"type": "string"}}},
                  ["record_id", "start_s", "end_s", "leads"]), "st_trend"),
    ToolSpec("measure_window",
             "Build an averaged beat from the duration_s window centred on t_s and measure fiducials, ST level "
             "at J+0/40/80 and at the heart-rate adjusted point, ST slope (mV/s from J to the ST point), T "
             "amplitude and per-lead quality. Known failure modes: the J point is less reliable when QRS is "
             "wide (>120 ms) or the window is noisy; check fiducials.ok/issues, n_template_beats and noise_uv "
             "before relying on the numbers. fiducials.leads_excluded lists leads too noisy to take part in "
             "finding the J point; the J point found from the other leads is used for all leads. warnings flag a "
             "QRS duration far from the record median (possible J-point misplacement).",
             _obj({**_REC, "t_s": {"type": "number"}, "duration_s": {"type": "number"}},
                  ["record_id", "t_s", "duration_s"]), "measure_window"),
    ToolSpec("compare_morphology",
             "Compare the averaged beat of the window centred on t_s with the one centred on t_ref_s: per-lead "
             "QRS correlation, amplitude ratio, polarity flips, QRS duration, frontal axis, ST and T change. "
             "Known failure modes: a heart-rate difference between the windows changes QRS and T slightly; "
             "leads with usable_both=false should not be trusted; warnings flag QRS changes on small QRS complexes "
             "and ST changes large enough to distort the QRS comparison.",
             _obj({**_REC, "t_ref_s": {"type": "number"}, "t_s": {"type": "number"},
                   "duration_s": {"type": "number"}}, ["record_id", "t_ref_s", "t_s", "duration_s"]),
             "compare_morphology"),
    ToolSpec("view_ecg",
             "Show the raw ECG of all leads for duration_s seconds (2 to 20) centred on t_s, on standard ECG "
             "paper (25 mm/s, 10 mm/mV) with the beat labels of the recording system above it. The image is "
             "the result; the numbers are the time range, beat label counts and heart rate.",
             _obj({**_REC, "t_s": {"type": "number"}, "duration_s": {"type": "number"}},
                  ["record_id", "t_s", "duration_s"]), "view_ecg"),
    ToolSpec("signal_quality",
             "Per-lead usable fraction, noise, saturation and flat-line windows over a time range.",
             _obj({**_REC, **_RANGE}, ["record_id", "start_s", "end_s"]), "signal_quality"),
    ToolSpec("rhythm_context",
             "Beat types, rhythm labels, heart rate, Lorenz (RR scatter) summary numbers and possible pacing "
             "spikes over a time range. Beat and rhythm labels come from the recording system and can be wrong.",
             _obj({**_REC, **_RANGE}, ["record_id", "start_s", "end_s"]), "rhythm_context"),
    ToolSpec("episode_dynamics",
             "For a candidate: heart rate 30 min to 1 min before onset, the heart-rate rise in the 5 min "
             "before onset, heart rate at onset, ST peak and end, lag between the heart-rate maximum and the "
             "ST extremum; per lead: onset and offset durations, plateau fraction, step-likeness, ST vs "
             "heart-rate correlation and slope, and the ST/HR slope from onset to the extremum and from the "
             "extremum to the end; the ST deviation at the same heart rate elsewhere in the record; whether "
             "the change was already under way when the candidate started; lead territories (12-lead records "
             "only). Correlations use the episode plus context_s seconds on each side.",
             _obj({**_REC, **_CAND, "context_s": {"type": "number"}},
                  ["record_id", "candidate_id", "context_s"]), "episode_dynamics"),
    ToolSpec("check_criteria",
             "Check a candidate numerically. rule='acc_aha_1999': ACC/AHA 1999 ambulatory ischemia criterion "
             "(>=100 uV depression for >=60 s; separation >=60 s, 300 s recommended), with ST slopes reported "
             "for you to judge flat/downsloping morphology. rule='contiguous_leads': >=100 uV depression or "
             "elevation in >=2 contiguous leads for >=30 s (12-lead Holter rule of Schipaanboord et al. 2026); "
             "applicable=false when the record has no two leads of one standard 12-lead group.",
             _obj({**_REC, **_CAND, "rule": {"type": "string", "enum": list(CRITERIA_RULES)}},
                  ["record_id", "candidate_id", "rule"]), "check_criteria"),
    ToolSpec("morphology_steps",
             "Sudden steps in QRS morphology (with frontal-axis, QRS-duration and per-lead QRS-area change) "
             "and in ST level within a time range, and which ST steps coincide with a QRS step (overlapping "
             "transitions or starts within sync_tolerance_s). A QRS step at the time of an ST shift suggests "
             "an axis/body-position shift (QRS amplitude change) or a conduction change (wider QRS).",
             _obj({**_REC, **_RANGE, "sync_tolerance_s": {"type": "number"}},
                  ["record_id", "start_s", "end_s", "sync_tolerance_s"]), "morphology_steps"),
    ToolSpec("eligibility_check",
             "Record-level conditions that limit ST interpretation per the ACC/AHA 1999 guideline: rhythm "
             "and beat types, possible pacing, median QRS duration, R-wave amplitude and baseline ST per lead, "
             "plus the conditions this tool cannot assess.",
             _obj(_REC, ["record_id"]), "eligibility_check"),
    ToolSpec("patient_context",
             "The patient's clinical information from the record header when available: age, sex, diagnoses, "
             "treatment and medication, history (previous myocardial infarction, hypertrophy, conduction block, "
             "previous tests such as stress test and coronary angiography) and symptoms during the recording. "
             "It makes ischaemia more or less likely for this patient; it does not show whether a given episode "
             "is ischaemic. Returns available=false when the record has no such information.",
             _obj(_REC, ["record_id"]), "patient_context"),
    ToolSpec("shape_changes",
             "For a candidate: how much the QRS shape and the ST-T shape changed from 10 min before onset to "
             "the extremum (record SD units) and their ratio, the largest 30-s QRS shape jump around the onset "
             "compared with the record's typical one, and per involved lead the ST-segment shape (level, rise "
             "from J to J+120 ms, curvature) before onset and at the extremum. A QRS change that is large "
             "relative to the ST-T change points to an axis or body-position shift.",
             _obj({**_REC, **_CAND}, ["record_id", "candidate_id"]), "shape_changes"),
    ToolSpec("ischemia_estimate",
             "A statistical estimate of the probability that the candidate is ischemic: a logistic regression on "
             "the outputs of a fixed set of checks, fitted on development records of the Long-Term ST Database. "
             "Returns the probability and the features that moved it most. It reflects that database's labels, "
             "which partly rest on clinical information; use it as one piece of evidence.",
             _obj({**_REC, **_CAND}, ["record_id", "candidate_id"]), "ischemia_estimate"),
    ToolSpec("episode_context",
             "Where the candidate sits in the record: for every lead the full contiguous ST change around the "
             "extremum (which can extend beyond the candidate), the local ST level before that change and the change "
             "relative to it, the absolute ST level, both extremes within the candidate, and the neighbouring "
             "candidates within 1 h. Use it to see whether a candidate is a fragment of a longer episode, the return "
             "between two episodes, or a change of a lead whose usual level is far from zero.",
             _obj({**_REC, **_CAND}, ["record_id", "candidate_id"]), "episode_context"),
    ToolSpec("rate_response",
             "This patient's ST response to heart-rate surges across the whole record (uV per bpm for each surge, "
             "lag-tolerant), the usual response outside ST episodes, the candidate's own response and percentile, "
             "and the median ST per 10-bpm heart-rate band outside episodes next to the candidate's values.",
             _obj({**_REC, **_CAND}, ["record_id", "candidate_id"]), "rate_response"),
    ToolSpec("find_reference_window",
             "Find a nearby clean window with small ST deviation (optionally similar heart rate, null to "
             "ignore heart rate) to use as t_ref_s in compare_morphology.",
             _obj({**_REC, "t_s": {"type": "number"}, "max_distance_s": {"type": "number"},
                   "hr_tolerance_bpm": {"type": ["number", "null"]}},
                  ["record_id", "t_s", "max_distance_s", "hr_tolerance_bpm"]),
             "find_reference_window"),
]
TOOLS_BY_NAME = {s.name: s for s in TOOL_SPECS}


def _coerce(args: dict, schema: dict) -> dict:
    """Fix argument types the way strict tool use would have enforced them (servers without it, such as
    vLLM, pass e.g. "" for a null or "120" for a number): "" or "null" -> None where null is allowed,
    numeric strings -> numbers."""
    out = dict(args)
    for key, value in args.items():
        t = schema["properties"].get(key, {}).get("type")
        types = t if isinstance(t, list) else [t]
        if not isinstance(value, str):
            continue
        if value.strip().lower() in ("", "null", "none") and "null" in types:
            out[key] = None
        elif "number" in types or "integer" in types:
            try:
                out[key] = int(value) if "integer" in types and "number" not in types else float(value)
            except ValueError:
                pass
    return out


def call_tool(toolkit: HolterToolkit, name: str, args: dict, bound: dict | None = None) -> dict:
    """Run a tool by name. `bound` supplies run-level arguments (see BOUND_ARGS)."""
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}")
    fn: Callable[..., dict] = getattr(toolkit, spec.method)
    params = inspect.signature(fn).parameters
    extra = set(args) - set(spec.input_schema["properties"])
    if extra:
        raise ToolError(f"unexpected arguments {sorted(extra)}")
    # Strict tool use enforces the schema on the API side; servers without it (vLLM) do not.
    missing = [k for k in spec.input_schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing required arguments {missing}")
    kwargs = _coerce(args, spec.input_schema)
    for key, value in (bound or {}).items():
        if key in params:
            kwargs[key] = value
    try:
        return jsonable(fn(**kwargs))
    except ToolError:
        raise
    except (KeyError, ValueError, IndexError, TypeError) as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e
