"""Per-candidate inputs of the baselines, computed once and cached.

For every development and test candidate:
- the fixed check set of the rule baseline (holter_agent/agent/baseline.py, collect_standard_evidence), i.e.
  the same tool outputs the agent can request, from our own measurements;
- ST/heart-rate features within the episode (scripts/test_candidate_features.py: Faganeli Pucer 2012,
  Faganeli & Jager 2008, exercise-test ST/HR hysteresis);
- QRS and ST-segment Karhunen-Loeve (Mahalanobis distance) features from the KLT series that LTSTDB provides
  (<record>.klt.zip; Jager et al. 1998);
- ST-segment Legendre coefficient features from the series LTSTDB provides (legendre/<record>.loc.gz; FP+40 to
  FP+168 ms; coefficients 1-3 describe level, slope and curvature: Faganeli & Jager 2008, Amon & Jager 2016);
- clinical features from the record header without the Comments field (used only by the "with clinical
  information" reference model).

    python -m baselines.evidence --data <ltstdb dir> --cache cache/features_ltstdb --workers 16

Writes outputs/baselines/_inputs/<record>.json.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baselines.common import OUT, ROOT, data_dirs, load_cases, protocol_of
from holter_agent.agent.baseline import collect_standard_evidence
from holter_agent.agent.estimate import features as tool_features
from holter_agent.agent.state import CaseState
from holter_agent.agent.toolkit import TOOLKIT_VERSION, HolterToolkit, jsonable

sys.path.insert(0, str(ROOT / "scripts"))
from test_candidate_features import clinical_features, kl_features, st_hr_features  # noqa: E402

BOUND = {"protocol": "B", "reference": "global"}
INPUTS = OUT / "_inputs"


def legendre_features(loc: np.ndarray, fs: float, lead_idx: int, start: float, peak: float) -> dict:
    """Change of ST-segment Legendre coefficients 1-3 from the 10 min before onset to within 30 s of the
    extremum, and coefficients 2-3 at the extremum (LTSTDB columns: fiducial point, then 18 per lead: 9
    coefficients x100 and 9 residuals). The documentation gives the first column in ms, but it is the sample
    index: its last value matches the record length in samples (s20011: 20594264 of 20594750 at 250 Hz)."""
    t = loc[:, 0] / fs
    c = loc[:, 1 + 18 * lead_idx:4 + 18 * lead_idx] / 100.0
    pre = (t >= start - 600) & (t < start)
    at = np.abs(t - peak) <= 30
    if pre.sum() < 20 or at.sum() < 5:
        return {}
    mp, ma = np.median(c[pre], axis=0), np.median(c[at], axis=0)
    return {"leg_d1": float(ma[0] - mp[0]), "leg_d2": float(ma[1] - mp[1]), "leg_d3": float(ma[2] - mp[2]),
            "leg_l2_peak": float(ma[1]), "leg_l3_peak": float(ma[2])}


def record_inputs(job: tuple[str, str, str, list[str]]) -> tuple[str, int]:
    data, cache, rid, cids = job
    warnings.filterwarnings("ignore")
    path = INPUTS / f"{rid}.json"
    if path.exists():
        return rid, 0
    tk = HolterToolkit(data, cache)
    bound = {**BOUND, "protocol": protocol_of(rid)}  # B, or A for the third test set (common.protocol_of)
    summaries = {c["id"]: c for c in tk.list_candidates(rid, bound["protocol"], "global", max_items=10**9)["candidates"]}
    st = tk._state(rid)
    # The KLT and Legendre series ship with LTSTDB only; other data sets (external evaluation, 2026-10-23) have no
    # such files, and those features are left out (the models impute them as for any missing feature).
    klt = loc = None
    if (Path(data) / f"{rid}.klt.zip").exists():
        with zipfile.ZipFile(Path(data) / f"{rid}.klt.zip") as z:
            klt = np.loadtxt(io.TextIOWrapper(z.open(z.namelist()[0])))
    if (Path(data) / "legendre" / f"{rid}.loc.gz").exists():
        with gzip.open(Path(data) / "legendre" / f"{rid}.loc.gz", "rt") as f:
            loc = np.loadtxt(f)
    clin = clinical_features((Path(data) / f"{rid}.hea").read_text())
    out = {"record": rid, "toolkit_version": TOOLKIT_VERSION, "leads": list(st.table.leads), "candidates": {}}
    for cid in cids:
        cand = tk.candidate(rid, cid)
        summary = summaries[cid]
        state = CaseState(st.record.name)
        ev = {k: r for k, (r, _) in collect_standard_evidence(tk, state, summary, bound, "baseline").items()}
        lead, peak = cand.peak.lead, cand.peak.peak_s
        geom = {"peak_uv": cand.peak.peak_uv, "start_s": cand.start_s, "end_s": cand.end_s}
        feats = {f"tool_{k}": v for k, v in tool_features(ev, geom).items()}
        feats.update(st_hr_features(st.table, lead, cand.start_s, cand.end_s, peak))
        if klt is not None:
            feats.update(kl_features(klt, st.record.fs, cand.start_s, cand.end_s, peak))
        if loc is not None:
            feats.update(legendre_features(loc, st.record.fs, st.table.lead_index(lead), cand.start_s, peak))
        out["candidates"][cid] = {"summary": summary, "evidence": ev, "features": feats, "clinical": clin,
                                  "peak_lead": lead, "peak_s": peak}
    path.write_text(json.dumps(jsonable(out)))
    return rid, len(cids)


def load_inputs() -> dict[str, dict]:
    """candidate id -> inputs (summary, evidence, features, clinical)."""
    out = {}
    for f in sorted(INPUTS.glob("*.json")):
        d = json.loads(f.read_text())
        for cid, v in d["candidates"].items():
            out[cid] = v | {"record": d["record"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data directory (default: by record name, common.data_dirs)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    INPUTS.mkdir(parents=True, exist_ok=True)
    by_rec: dict[str, list[str]] = {}
    for c in load_cases():
        by_rec.setdefault(c.record, []).append(c.candidate_id)
    jobs = [(args.data or str(data_dirs(r)[0]), args.cache or str(data_dirs(r)[1]), r, cids)
            for r, cids in sorted(by_rec.items())]
    with ProcessPoolExecutor(args.workers) as ex:
        for rid, n in ex.map(record_inputs, jobs):
            print(f"{rid}: {n} candidates", flush=True)


if __name__ == "__main__":
    main()
