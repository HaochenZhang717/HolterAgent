"""What a task agent's own deterministic tools (task_agent/tools/<name>.py) can see: the per-window feature table of a
record, the ST deviation from the run's record-wide reference, and the candidate episodes (protocol B), all computed
from the feature cache. The raw recordings, their annotation and header files and the reference labels are not
available: tools run in a sandbox (ext_worker.py) that can read only the feature cache.

    ctx.table(record_id)             FeatureTable: ctx.table(r)["st_uv"] -> [windows, leads] array; .leads, .window_s,
                                     .duration_s; columns: t_start_s, hr_bpm, st_uv, st_j0_uv, st_slope, t_amp_uv,
                                     qrs_pp_uv, qrs_area_uvs, noise_uv, wander_uv, sat_frac, flat_frac, usable,
                                     qrs_ms, j_ms, iso_ms, fiducials_ok, n_beats, n_template_beats, n_normal, n_supra,
                                     n_ventricular, n_other  (one row per 15-s window)
    ctx.deviation(record_id)         [windows, leads] ST deviation in uV (NaN where unusable)
    ctx.reference(record_id)         [leads] the record-wide reference ST level
    ctx.candidates(record_id)        list of Candidate (id, start_s, end_s, lead_episodes, peak, leads, summary())
    ctx.candidate(record_id, cid)    one Candidate
    ctx.precomputed(name, record_id) results computed beforehand for the record's candidate windows and stored next to
                                     the feature cache (<cache>/<name>/<record>.json); "fm": ECGFounder statements and
                                     NeuroKit2 delineation (sky/fm_tools_run.py, scripts/fm_to_cache.py; 2026-10-17)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holter_agent.config import DEFAULT_CONFIG

from .paths import candidate_protocol
from holter_agent.trend.episodes import Candidate, detect_candidates, st_deviation, st_reference
from holter_agent.trend.features import FeatureTable

PROTOCOL, REFERENCE = candidate_protocol(), "global"  # B unless CANDIDATE_PROTOCOL (rsi/paths.py)


class ExtContext:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._tables: dict[str, FeatureTable] = {}
        self._cands: dict[str, list[Candidate]] = {}

    def _name(self, record_id: str) -> str:
        r = str(record_id)
        for name in (r, "s" + r):
            if (self.cache_dir / f"{name}.features.npz").exists():
                return name
        hits = [f.name[:-len(".features.npz")] for f in self.cache_dir.glob(f"*{r}.features.npz")]
        if len(hits) == 1:  # a short id, as the toolkit's record store resolves it
            return hits[0]
        raise KeyError(f"no record {record_id!r}")

    def table(self, record_id: str) -> FeatureTable:
        name = self._name(record_id)
        if name not in self._tables:
            self._tables[name] = FeatureTable.load(self.cache_dir / f"{name}.features.npz")
        return self._tables[name]

    def deviation(self, record_id: str) -> np.ndarray:
        return st_deviation(self.table(record_id), REFERENCE)

    def reference(self, record_id: str) -> np.ndarray:
        return st_reference(self.table(record_id), REFERENCE)

    def candidates(self, record_id: str) -> list[Candidate]:
        name = self._name(record_id)
        if name not in self._cands:
            self._cands[name] = detect_candidates(self.table(name), DEFAULT_CONFIG.episode, PROTOCOL, REFERENCE)
        return self._cands[name]

    def precomputed(self, name: str, record_id: str) -> dict:
        p = self.cache_dir / name / f"{self._name(record_id)}.json"
        if not p.exists():
            raise KeyError(f"no precomputed {name!r} results for record {record_id!r}")
        return json.loads(p.read_text())

    def candidate(self, record_id: str, candidate_id: str) -> Candidate:
        cid = str(candidate_id)
        for c in self.candidates(record_id):
            if c.id == cid or c.id.split("-", 1)[-1] == cid:
                return c
        raise KeyError(f"no candidate {candidate_id!r} in record {record_id!r}")
