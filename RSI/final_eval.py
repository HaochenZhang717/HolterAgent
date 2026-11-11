"""Final evaluation of a model track after the search (2026-10-19): the given harnesses (the seed and the best one), each
run REPS times (RSI/tracks.json) on the LTSTDB test set and the external EDB set. Neither set was used during the search.

Reported per set and harness: AUC of the per-case mean score with a 95% interval from a two-level bootstrap (records
with replacement, then the runs of each case); the per-run AUCs; the within-patient AUC (pairs of candidates of the same
patient; LTSTDB patient = record name without its last digit, EDB: the record, since EDB gives no patient ids); the
paired difference between the harnesses (two-level bootstrap); and, after Platt recalibration fitted on the same
harness's development runs (search_v2 and validation_v4 without its ltstdb_unused candidates, mean scores; from
2026-10-24), the Brier score and the decisions at p >= 0.5.

    RSI_TRACK=p56l python RSI/final_eval.py v08p p01 --sets test edb
Writes RSI/cloop_<track>/final_<harnesses>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rsi import scoring  # noqa: E402
from rsi.paths import (CODE_DIR, EDB_CACHE, EDB_DIR, MIMIC_CACHE, MIMIC_DIR, SPLITS_DIR, STAFF_CACHE, STAFF_DIR,  # noqa: E402
                       TRACK, run_dir, track_config)

EXTERNAL = {"edb": (EDB_DIR, EDB_CACHE), "staffiii": (STAFF_DIR, STAFF_CACHE),
            "mimic": (MIMIC_DIR, MIMIC_CACHE)}  # other splits are LTSTDB

_spec = importlib.util.spec_from_file_location("cloop_final", HERE / "cloop.py")
cl = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["cloop"]
_spec.loader.exec_module(cl)
sys.argv = _argv


def run_set(hid: str, split: str, reps: int, concurrency: int) -> None:
    def one(rep: int) -> None:
        cmd = [sys.executable, str(HERE / "evaluate.py"), hid, "--split", split, "--rep", str(rep),
               "--concurrency", str(concurrency)]
        if split in EXTERNAL:
            cmd += ["--data", str(EXTERNAL[split][0]), "--cache", str(EXTERNAL[split][1])]
        log = cl.CLOOP / f"final_{hid}_{split}_r{rep}.log"
        for _ in range(3):  # resume until every case has a result (pool hiccups)
            with log.open("a") as f:
                subprocess.run(cmd, cwd=CODE_DIR, stdout=f, stderr=subprocess.STDOUT)
            m = run_dir(split, hid, rep) / "metrics.json"
            n = len(scoring.load_cases(SPLITS_DIR / f"{split}.json"))
            if m.exists() and json.loads(m.read_text())["n_run"] >= n:
                return
    with ThreadPoolExecutor(reps) as ex:
        list(ex.map(one, range(1, reps + 1)))


def matrix(cases: list[dict], runs: list[dict]) -> np.ndarray:
    return np.array([[0.5 if r.get(c["candidate_id"], {}).get("p") is None else float(r[c["candidate_id"]]["p"])
                      for r in runs] for c in cases])


def auc_ci(cases: list[dict], runs: list[dict], n: int = 2000, seed: int = 0) -> tuple[float, list[float]]:
    a = matrix(cases, runs)
    y = np.array([c["reference"] == "ischemic" for c in cases])
    rec = np.array([c["record"] for c in cases])
    rng = np.random.default_rng(seed + 1)
    vals = []
    for idx in scoring._record_draws(rec, n, seed):
        k = a.shape[1]
        v = scoring.auc(a[idx[:, None], rng.integers(0, k, (len(idx), k))].mean(1), y[idx])
        if v == v:
            vals.append(v)
    return scoring.auc(a.mean(1), y), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def within(cases: list[dict], runs: list[dict], split: str) -> float | None:
    s = matrix(cases, runs).mean(1)
    y = np.array([c["reference"] == "ischemic" for c in cases])
    group = np.array([c["record"] if split == "edb" else c["record"][:-1] for c in cases])
    num = den = 0.0
    for g in set(group):
        m = group == g
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if n1 and n0:
            num += scoring.auc(s[m], y[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("harnesses", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["test", "edb"])
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    reps = int(track_config().get("reps", 1))
    conc = args.concurrency or int(track_config().get("concurrency", 16))
    if not args.score_only:
        with ThreadPoolExecutor(len(args.harnesses) * len(args.sets)) as ex:
            list(ex.map(lambda hs: run_set(hs[0], hs[1], reps, conc),
                        [(h, s) for s in args.sets for h in args.harnesses]))
    # validation_v4 holds 165 candidates of the ltstdb_unused test records (taken in by validation_v3 on 2026-10-08,
    # found 2026-10-24): they are left out of the calibration fit
    unused = {c["candidate_id"] for c in scoring.load_cases(SPLITS_DIR / "ltstdb_unused.json")}
    # a track that selects on another split (the aspect ablation, 2026-11-06: validation_v4dev) calibrates on it
    dev_split = cl.SELECT if cl.SELECT != "validation_v4" else "validation_v4"
    dev_cases = scoring.load_cases(SPLITS_DIR / "search_v2.json") + [
        c for c in scoring.load_cases(SPLITS_DIR / f"{dev_split}.json") if c["candidate_id"] not in unused]
    out = {"track": TRACK, "model": track_config().get("model"), "reps": reps, "sets": {}}
    for split in args.sets:
        cases = scoring.load_cases(SPLITS_DIR / f"{split}.json")
        res = {"n_cases": len(cases), "n_ischemic": int(sum(c["reference"] == "ischemic" for c in cases))}
        runs = {h: [scoring.load_run(run_dir(split, h, r)) for r in range(1, reps + 1)] for h in args.harnesses}
        for h in args.harnesses:
            a, ci = auc_ci(cases, runs[h])
            try:
                dev_runs = {**cl.agg_run(cl.rep_runs("search_v2", h)), **cl.agg_run(cl.rep_runs(dev_split, h))}
                cal = scoring.calibration(dev_cases, dev_runs, cases, cl.agg_run(runs[h]))
            except (FileNotFoundError, KeyError, ValueError):  # no development runs in this track (codex re-evaluation)
                cal = {k: None for k in ("brier", "brier_raw", "brier_baseline", "se_at_cal_0.5", "sp_at_cal_0.5",
                                         "ppv_at_cal_0.5", "f1_at_cal_0.5", "platt_a", "platt_b")}
            res[h] = {"auc": a, "auc_ci95": ci, "auc_per_run": [scoring.auc(*scoring.arrays(cases, r)[:2]) for r in runs[h]],
                      "within_auc": within(cases, runs[h], split),
                      "n_without_score": int(sum(any(r.get(c["candidate_id"], {}).get("p") is None for r in runs[h])
                                                 for c in cases)),
                      "calibrated": {k: cal[k] for k in ("brier", "brier_raw", "brier_baseline", "se_at_cal_0.5",
                                                         "sp_at_cal_0.5", "ppv_at_cal_0.5", "f1_at_cal_0.5",
                                                         "platt_a", "platt_b")}}
        if len(args.harnesses) >= 2:
            a, b = args.harnesses[0], args.harnesses[-1]
            res[f"{b}_minus_{a}"] = cl.paired_reps(cases, runs[a], runs[b])
        out["sets"][split] = res
    sets_tag = "" if args.sets == ["test", "edb"] else "_" + "_".join(args.sets)  # keeps earlier result files
    path = cl.CLOOP / f"final_{'_'.join(args.harnesses)}{sets_tag}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for split, res in out["sets"].items():
        print(f"== {split}: {res['n_cases']} candidates ({res['n_ischemic']} ischemic)")
        for h in args.harnesses:
            r = res[h]
            print(f"  {h}: AUC {r['auc']:.3f} [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}] runs "
                  f"{', '.join(f'{x:.3f}' for x in r['auc_per_run'])}; within {r['within_auc']}; calibrated "
                  f"Se {r['calibrated']['se_at_cal_0.5']:.2f} Sp {r['calibrated']['sp_at_cal_0.5']:.2f} "
                  f"F1 {r['calibrated']['f1_at_cal_0.5']:.2f} Brier {r['calibrated']['brier']:.3f}")
        for k, v in res.items():
            if k.endswith(f"_minus_{args.harnesses[0]}"):
                print(f"  {k}: {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], P(>0) {v['p_gt0']:.2f}")
    print(f"written {path}")


if __name__ == "__main__":
    main()
