"""Reference class of every LTSTDB candidate, and a stratified sample for agent runs.

Each Step 2 candidate (protocol B, global reference) gets the class that validate_ltstdb.py assigns:
  ischemic / heart_rate_related   an annotated ST episode of that kind confirms the candidate
  axis_or_conduction              an axis shift or conduction change marker lies within the candidate
  noise_or_unreadable             the candidate overlaps annotated noise or an unreadable segment
  other                           none of these (not an annotated ST episode)
Records whose leads share a name (22 two-lead records label both leads "ECG") are left out of the
sample: the tools address leads by name.

    python scripts/ltstdb_cases.py --data <ltstdb dir> --cache cache/features_ltstdb

Writes code/outputs/ltstdb_cases/candidates.json (all candidates) and sample.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_ltstdb as V  # noqa: E402

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.bench.ltstdb import load_st_annotations  # noqa: E402
from holter_agent.paths import OUTPUT_DIR  # noqa: E402

OUT = OUTPUT_DIR / "ltstdb_cases"
# Candidates per class in the sample (noise_or_unreadable has only 7 in the whole database).
QUOTA = {"ischemic": 12, "heart_rate_related": 8, "axis_or_conduction": 8, "noise_or_unreadable": 4, "other": 8}
PER_RECORD = 2  # at most this many sampled candidates of one class from one record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    tk = HolterToolkit(args.data, args.cache)
    rows = []
    for hea in sorted(Path(args.data).glob("s*.hea")):
        name = hea.stem
        st = tk._state(name)
        rec, tab = st.record, st.table
        ann = load_st_annotations(Path(args.data) / name, "B", rec.fs, rec.duration_s)
        cands = [c for c in V.detect_candidates(tab, V.CFG.episode, "B", "global") if c.end_s > V.SKIP_S]
        ev = V.evaluate_candidates(name, list(rec.leads), cands, ann, rec.duration_s / 3600, tab, "global")
        dup = len(set(rec.leads)) < len(rec.leads)
        for c, r in zip(cands, ev["cands"]):
            rows.append({"record": name, "candidate_id": c.id, "start_s": c.start_s, "end_s": c.end_s,
                         "reference": r["why"], "peak_uv": r["peak_uv"], "leads": list(rec.leads),
                         "duplicate_lead_names": dup})
        print(f"{name}: {len(cands)} candidates", flush=True)
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=1))

    rng = random.Random(args.seed)
    usable = [r for r in rows if not r["duplicate_lead_names"]]
    sample = []
    for cls, n in QUOTA.items():
        pool = [r for r in usable if r["reference"] == cls]
        rng.shuffle(pool)
        taken: Counter = Counter()
        for r in pool:
            if len([s for s in sample if s["reference"] == cls]) >= n:
                break
            if taken[r["record"]] < PER_RECORD:
                sample.append(r)
                taken[r["record"]] += 1
    sample.sort(key=lambda r: (r["record"], r["start_s"]))
    (OUT / "sample.json").write_text(json.dumps({"seed": args.seed, "quota": QUOTA, "per_record": PER_RECORD,
                                                 "cases": sample}, indent=1))
    print("all candidates:", dict(Counter(r["reference"] for r in rows)))
    print("usable (unique lead names):", dict(Counter(r["reference"] for r in usable)))
    print("sample:", dict(Counter(r["reference"] for r in sample)), "from", len({r["record"] for r in sample}),
          "records")


if __name__ == "__main__":
    main()
