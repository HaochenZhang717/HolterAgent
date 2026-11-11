"""Candidates and reference classes of the STAFF III pseudo-records (extra external test of ischemia sensitivity,
2026-10-21; records from scripts/staffiii_records.py).

The candidate pipeline is the one used on LTSTDB and EDB, unchanged: Step 2 candidates (protocol B rule, record-wide
reference), candidates ending in the first 300 s dropped (the 300-s baseline recording opens every pseudo-record).
Classes:
  ischemic   the candidate overlaps a balloon inflation (coronary occlusion, from the STAFF III spreadsheet)
  other      it does not (before the inflation or after deflation)
Also reports how many inflations are covered by a candidate (detector sensitivity).

    python scripts/build_features.py --data <staffiii_holter> --cache cache/features_staffiii --workers 8
    python scripts/staffiii_cases.py
Writes code/outputs/staffiii_cases/candidates.json and summary.json, and the split file RSI/splits/staffiii.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RSI"))
import validate_ltstdb as V  # noqa: E402

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.bench.metrics import SKIP_S  # noqa: E402
from holter_agent.paths import CODE_DIR, OUTPUT_DIR  # noqa: E402
from rsi.paths import STAFF_CACHE, STAFF_DIR  # noqa: E402

OUT = OUTPUT_DIR / "staffiii_cases"
SPLIT = CODE_DIR / "RSI" / "splits" / "staffiii.json"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((STAFF_DIR / "inflations.json").read_text())
    tk = HolterToolkit(str(STAFF_DIR), str(STAFF_CACHE))
    rows, per_record = [], []
    for m in meta["records"]:
        name = m["record"]
        st = tk._state(name)
        rec, tab = st.record, st.table
        cands = [c for c in V.detect_candidates(tab, V.CFG.episode, "B", "global") if c.end_s > SKIP_S]
        infl = m["inflations_s"]
        covered = [any(c.start_s < b and c.end_s > a for c in cands) for a, b in infl]
        for c in cands:
            isch = any(c.start_s < b and c.end_s > a for a, b in infl)
            peak = abs(float(c.peak.peak_uv))  # as validate_ltstdb.evaluate_candidates
            rows.append({"record": name, "candidate_id": c.id, "start_s": c.start_s, "end_s": c.end_s,
                         "reference": "ischemic" if isch else "other", "peak_uv": peak, "leads": list(rec.leads),
                         "duplicate_lead_names": False, "occluded_artery": m["occluded_artery"]})
        per_record.append({"record": name, "patient": m["patient"], "artery": m["occluded_artery"],
                           "n_candidates": len(cands), "n_inflations": len(infl), "n_covered": sum(covered)})
        print(f"{name}: {len(cands)} candidates, {sum(covered)}/{len(infl)} inflations covered", flush=True)
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=1))
    n_inf = sum(r["n_inflations"] for r in per_record)
    n_cov = sum(r["n_covered"] for r in per_record)
    summary = {"records": len(per_record), "patients": len({r["patient"] for r in per_record}),
               "candidates": len(rows), "classes": dict(Counter(r["reference"] for r in rows)),
               "inflations": n_inf, "inflations_covered": n_cov,
               "detector_sensitivity": round(n_cov / n_inf, 3) if n_inf else None, "per_record": per_record}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ("record", "candidate_id", "start_s", "end_s", "reference", "peak_uv", "leads", "duplicate_lead_names")
    cases = sorted(({k: r[k] for k in keys} for r in rows), key=lambda c: (c["record"], c["start_s"]))
    SPLIT.write_text(json.dumps({
        "note": "extra external test (2026-10-21): every Step 2 candidate of the STAFF III pseudo-records (patient's "
                "cath-lab baseline + one balloon-inflation recording; leads V2, V5, III at 250 Hz); ischemic = overlaps "
                "a balloon inflation (scripts/staffiii_records.py, staffiii_cases.py); scored once",
        "cases": cases}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_record"}, indent=1))


if __name__ == "__main__":
    main()
