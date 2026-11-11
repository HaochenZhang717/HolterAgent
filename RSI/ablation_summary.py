"""Aspect ablation summary (2026-11-08): per group and test set, the AUC of each of the three evaluation runs
(mean and SD over runs), the AUC of the per-case mean score with a two-level bootstrap interval (records, then
runs), and the paired difference to ABC on the same resamples. ABC = p56a's a03 runs r1-r3; the six other groups =
their single-search picks evaluated in track p56ux (the same agent loop).

    PYTHONPATH=..:. python RSI/ablation_summary.py      (run from code/RSI)
Writes cloop_p56ux/ablation_summary.json and prints a table.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rsi import scoring  # noqa: E402
from rsi.paths import SPLITS_DIR  # noqa: E402

SETS = ["ltstdb_unused", "edb", "mimic"]
GROUPS = [("ABC", "a03", "p56a", 0.776), ("A", "xa03", "p56ux", 0.683), ("B", "xb03", "p56ux", 0.726),
          ("C", "xc02", "p56ux", 0.698), ("AB", "xab03", "p56ux", 0.710), ("AC", "xac03", "p56ux", 0.714),
          ("BC", "xbc01", "p56ux", 0.725)]
N_BOOT = 2000


def runs(track: str, split: str, hid: str) -> list[dict]:
    return [scoring.load_run(HERE / f"runs_{track}" / split / f"{hid}_r{r}") for r in (1, 2, 3)]


def matrix(cases: list[dict], rs: list[dict]) -> np.ndarray:
    return np.array([[0.5 if r.get(c["candidate_id"], {}).get("p") is None else float(r[c["candidate_id"]]["p"])
                      for r in rs] for c in cases])


def main() -> None:
    out = {}
    for split in SETS:
        cases = scoring.load_cases(SPLITS_DIR / f"{split}.json")
        y = np.array([c["reference"] == "ischemic" for c in cases])
        rec = np.array([c["record"] for c in cases])
        draws = list(scoring._record_draws(rec, N_BOOT, 0))
        rng = np.random.default_rng(1)
        run_idx = [rng.integers(0, 3, (len(i), 3)) for i in draws]  # the same run draws for every group
        mats = {}
        for g, hid, track, _ in GROUPS:
            rs = runs(track, split, hid)
            missing = sum(1 for c in cases for r in rs if r.get(c["candidate_id"], {}).get("p") is None)
            mats[g] = matrix(cases, rs)
            per_run = [scoring.auc(mats[g][:, k], y) for k in range(3)]
            boot = [scoring.auc(mats[g][i[:, None], j].mean(1), y[i]) for i, j in zip(draws, run_idx)]
            boot = [b for b in boot if b == b]
            out.setdefault(g, {})[split] = {
                "auc_runs": per_run, "auc_runs_mean": float(np.mean(per_run)), "auc_runs_sd": float(np.std(per_run, ddof=1)),
                "auc_meanscore": scoring.auc(mats[g].mean(1), y),
                "auc_meanscore_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "missing_scores": missing}
        for g, *_ in GROUPS[1:]:
            d = [scoring.auc(mats[g][i[:, None], j].mean(1), y[i]) - scoring.auc(mats["ABC"][i[:, None], j].mean(1), y[i])
                 for i, j in zip(draws, run_idx)]
            d = [x for x in d if x == x]
            o = out[g][split]
            o["minus_abc"] = o["auc_meanscore"] - out["ABC"][split]["auc_meanscore"]
            o["minus_abc_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            o["minus_abc_p"] = float(np.mean(np.array(d) > 0))
    for g, *_ in GROUPS:
        ms = [out[g][s]["auc_meanscore"] for s in SETS]
        out[g]["mean_over_sets"] = float(np.mean(ms))
        out[g]["range_over_sets"] = float(max(ms) - min(ms))
    (HERE / "cloop_p56ux" / "ablation_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'group':5s} {'search':>6s} | " + " | ".join(f"{s:^30s}" for s in SETS) + " | mean  range")
    for g, hid, track, sa in GROUPS:
        cells = []
        for s in SETS:
            o = out[g][s]
            c = (f"{o['auc_runs_mean']:.3f}±{o['auc_runs_sd']:.3f} ms {o['auc_meanscore']:.3f}"
                 f" [{o['auc_meanscore_ci'][0]:.2f},{o['auc_meanscore_ci'][1]:.2f}] miss {o['missing_scores']}")
            if g != "ABC":
                c += f" d {o['minus_abc']:+.3f} [{o['minus_abc_ci'][0]:+.2f},{o['minus_abc_ci'][1]:+.2f}] P {o['minus_abc_p']:.2f}"
            cells.append(c)
        print(f"{g:5s} {sa:6.3f} | " + " | ".join(cells) + f" | {out[g]['mean_over_sets']:.3f} {out[g]['range_over_sets']:.3f}")


if __name__ == "__main__":
    main()
