"""Candidates and reference classes of the European ST-T Database (EDB), the external test set.

The pipeline is the one used on LTSTDB, unchanged: Step 2 candidates (protocol B rule, record-wide reference,
holter_agent.trend.episodes.detect_candidates), candidates ending in the first 300 s dropped (epicmp default), and
the class of validate_ltstdb.evaluate_candidates:
  ischemic             an annotated EDB ST episode confirms the candidate (epicmp rule; EDB does not separate
                       heart-rate related episodes, so every annotated ST episode counts, see holter_agent/bench/edb.py)
  axis_or_conduction   an annotated axis-shift ST interval lies within 60 s of the candidate
  noise_or_unreadable  the candidate overlaps an interval annotated noisy or unreadable in either signal
  other                none of these (not an annotated ST episode)
Also reports how many annotated EDB ST episodes are covered by a candidate (detector sensitivity, epicmp rule).

    python scripts/build_features.py --data <edb dir> --cache cache/features_edb --workers 8
    python scripts/edb_cases.py --data <edb dir> --cache cache/features_edb

Writes code/outputs/edb_cases/candidates.json and summary.json, and the split file RSI/splits/edb.json (every
candidate; scored once as the external test).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_ltstdb as V  # noqa: E402

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.bench.edb import load_edb_annotations  # noqa: E402
from holter_agent.bench.metrics import SKIP_S  # noqa: E402
from holter_agent.paths import CODE_DIR, OUTPUT_DIR  # noqa: E402

OUT = OUTPUT_DIR / "edb_cases"
SPLIT = CODE_DIR / "RSI" / "splits" / "edb.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    tk = HolterToolkit(args.data, args.cache)
    rows, per_record = [], []
    for hea in sorted(Path(args.data).glob("e*.hea")):
        name = hea.stem
        st = tk._state(name)
        rec, tab = st.record, st.table
        ann, extra = load_edb_annotations(Path(args.data) / name, rec.fs, rec.duration_s, len(rec.leads))
        ann.unreadable += extra.noisy  # noisy and unreadable intervals both make a candidate noise_or_unreadable
        cands = [c for c in V.detect_candidates(tab, V.CFG.episode, "B", "global") if c.end_s > SKIP_S]
        ev = V.evaluate_candidates(name, list(rec.leads), cands, ann, rec.duration_s / 3600, tab, "global")
        dup = len(set(rec.leads)) < len(rec.leads)
        for c, r in zip(cands, ev["cands"]):
            rows.append({"record": name, "candidate_id": c.id, "start_s": c.start_s, "end_s": c.end_s,
                         "reference": r["why"], "peak_uv": r["peak_uv"], "leads": list(rec.leads),
                         "duplicate_lead_names": dup})
        refs = ev["refs"]
        per_record.append({"record": name, "leads": list(rec.leads), "n_candidates": len(cands),
                           "n_st_episodes": len(refs), "n_detected": sum(r["any_lead"] for r in refs),
                           "n_axis_shift_episodes": len(extra.axis_shift_episodes), "unmatched": ann.unmatched})
        print(f"{name}: {len(cands)} candidates, {len(refs)} ST episodes "
              f"({sum(r['any_lead'] for r in refs)} covered)", flush=True)
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=1))
    n_ep = sum(r["n_st_episodes"] for r in per_record)
    n_det = sum(r["n_detected"] for r in per_record)
    summary = {"records": len(per_record), "candidates": len(rows),
               "classes": dict(Counter(r["reference"] for r in rows)),
               "st_episodes_after_300s": n_ep, "st_episodes_covered_by_a_candidate": n_det,
               "detector_sensitivity": round(n_det / n_ep, 3) if n_ep else None,
               "records_without_candidates": [r["record"] for r in per_record if r["n_candidates"] == 0],
               "per_record": per_record}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ("record", "candidate_id", "start_s", "end_s", "reference", "peak_uv", "leads", "duplicate_lead_names")
    cases = sorted(({k: r[k] for k in keys} for r in rows), key=lambda c: (c["record"], c["start_s"]))
    SPLIT.write_text(json.dumps({
        "note": "external test set (2026-10-16): every Step 2 candidate of the 90 European ST-T Database records, "
                "classes from the EDB ST and axis-shift annotations (scripts/edb_cases.py); scored once",
        "cases": cases}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_record"}, indent=1))


if __name__ == "__main__":
    main()
