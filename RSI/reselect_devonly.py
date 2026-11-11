"""Re-selection of an archive track's final harness without test candidates.

validation_v4 (the selection split of `cloop.py finalize`) holds 165 candidates of the 22 LTSTDB records that were
later made the test set ltstdb_unused: validation_v3 (2026-10-08) took in candidates of these records, and
validation_v4 (2026-10-18) is a subsample of validation_v3. This was found on 2026-10-24, after the p56a pick (a09) had
been evaluated on the test sets. Rule fixed before any other harness was evaluated on a test set: the selection split
is validation_v4 minus every ltstdb_unused candidate (104 candidates, 14 patients, all from the development split
dev_v07); the top-k harnesses by search AUC are those of final_selection.json (unchanged, the search does not use
validation), and the pick is the one with the highest AUC of the per-case mean score over the runs on that split.
No new runs: the validation_v4 runs of the finalists are re-scored.

    RSI_TRACK=p56a python RSI/reselect_devonly.py
Writes RSI/cloop_<track>/final_selection_devonly.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rsi import scoring  # noqa: E402
from rsi.paths import SPLITS_DIR  # noqa: E402

_spec = importlib.util.spec_from_file_location("cloop_reselect", HERE / "cloop.py")
cl = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["cloop"]
_spec.loader.exec_module(cl)
sys.argv = _argv


def main() -> None:
    old = json.loads((cl.CLOOP / "final_selection.json").read_text())
    unused = {c["candidate_id"] for c in scoring.load_cases(SPLITS_DIR / "ltstdb_unused.json")}
    v4 = scoring.load_cases(SPLITS_DIR / f"{old['select_split']}.json")
    dev = [c for c in v4 if c["candidate_id"] not in unused]
    rows = []
    for r in old["rows"]:
        runs = cl.rep_runs(old["select_split"], r["id"])
        rows.append({"id": r["id"], "s_auc": r["s_auc"], "v_auc_all": r["v_auc"],
                     "v_auc_dev": scoring.auc(*scoring.arrays(dev, cl.agg_run(runs))[:2]),
                     "v_auc_dev_per_rep": [scoring.auc(*scoring.arrays(dev, run)[:2]) for run in runs],
                     "v_auc_unused_part": scoring.auc(*scoring.arrays([c for c in v4 if c["candidate_id"] in unused],
                                                                      cl.agg_run(runs))[:2]),
                     "lineage": r["lineage"]})
    pick = max(rows, key=lambda r: r["v_auc_dev"])
    out = {"select_split": f"{old['select_split']} minus ltstdb_unused candidates",
           "n_cases": len(dev), "n_patients": len({c["record"][:-1] for c in dev}),
           "n_ischemic": sum(c["reference"] == "ischemic" for c in dev),
           "why": "validation_v4 holds 165 candidates of the ltstdb_unused test records (found 2026-10-24)",
           "old_pick": old["pick"], "rows": rows, "pick": pick["id"], "time": time.strftime("%Y-%m-%d %H:%M")}
    (cl.CLOOP / "final_selection_devonly.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['id']:6s} search {r['s_auc']:.3f}  validation all {r['v_auc_all']:.3f}  dev part {r['v_auc_dev']:.3f} "
              f"({', '.join(f'{x:.3f}' for x in r['v_auc_dev_per_rep'])})  ltstdb_unused part {r['v_auc_unused_part']:.3f}")
    print(f"{out['n_cases']} cases, {out['n_patients']} patients; pick: {pick['id']} (was {old['pick']})")


if __name__ == "__main__":
    main()
