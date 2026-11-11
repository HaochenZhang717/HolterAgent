"""Faganeli-style classifiers on our own measurements only (no series that ship with LTSTDB).

baselines/feature_ml.py's `*_faganeli` rows use the KLT and Legendre series distributed with LTSTDB. The Legendre
series is computed at ARISTOTLE's automatic fiducial point (FP+40 to FP+168 ms; legendre/doc/
ltstdb_legendre_coefficients_files.txt), but the files do not say how the KLT series was computed, so whether it
involves the manually placed isoelectric and J points cannot be checked. These rows replace both with the same feature
families from our toolkit (shape_changes):
  heart rate        tool_hr_*, st_hr_*, hr_half_* (already ours; Faganeli & Jager 2010)
  QRS / ST-T shape  qrs_shape_change_sd, st_t_shape_change_sd and their ratio, the largest 30-s QRS jump near onset
                    and its ratio to the record's typical jump (for the KLT Mahalanobis distances)
  ST-segment shape  level, rise and curvature of the 120 ms after J in the peak lead at the extremum, and their change
                    from 10 min before onset (for the Legendre coefficients 1-3; Amon & Jager 2016)
Models, grids, folds and scoring are those of feature_ml (decision tree, logistic regression, naive Bayes, kNN).

    ../.venv/bin/python -m baselines.faganeli_own features        # toolkit; writes outputs/baselines/_inputs_own/
    python -m baselines.faganeli_own fit                          # needs scikit-learn
"""
from __future__ import annotations

import os
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baselines.common import OUT, ROOT, data_dirs, evaluate, load_cases, protocol_of, save

OWN = OUT / "_inputs_own"
DATA = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data")) / "ltstdb"
CACHE = ROOT / "cache" / "features_ltstdb"
HR_PREFIX = ("tool_hr_", "st_hr_", "hr_half_")


def _shape_features(sc: dict, peak_lead: str | None) -> dict:
    f = {"own_qrs_shape_sd": sc.get("qrs_shape_change_sd"), "own_st_t_shape_sd": sc.get("st_t_shape_change_sd"),
         "own_qrs_over_st_t": sc.get("qrs_to_st_t_change_ratio"),
         "own_qrs_jump_30s_sd": sc.get("largest_30s_qrs_jump_near_onset_sd")}
    typ = sc.get("typical_30s_qrs_jump_in_record_sd")
    if f["own_qrs_jump_30s_sd"] is not None and typ:
        f["own_qrs_jump_rel"] = f["own_qrs_jump_30s_sd"] / typ
    seg = (sc.get("st_segment") or {}).get(peak_lead) if peak_lead else None
    if seg and seg.get("at_extremum") and seg.get("before_onset"):
        a, b = seg["at_extremum"], seg["before_onset"]
        for k in ("level_uv", "rise_uv", "curvature_uv"):
            if a.get(k) is not None:
                f[f"own_st_{k}_peak"] = a[k]
                if b.get(k) is not None:
                    f[f"own_st_{k}_change"] = a[k] - b[k]
    return {k: float(v) for k, v in f.items() if v is not None}


def record_features(job: tuple[str, list[str]]) -> str:
    rid, cids = job
    warnings.filterwarnings("ignore")
    from holter_agent.agent.toolkit import HolterToolkit, ToolError, jsonable
    path = OWN / f"{rid}.json"
    if path.exists():
        return rid
    data, cache = data_dirs(rid)  # by record (external evaluation, 2026-10-23); LTSTDB records as before
    tk = HolterToolkit(str(data), str(cache))
    tk.list_candidates(rid, protocol_of(rid), "global", max_items=10**9)
    inputs = json.loads((OUT / "_inputs" / f"{rid}.json").read_text())["candidates"]
    out = {}
    for cid in cids:
        try:
            sc = jsonable(tk.shape_changes(rid, cid))
        except ToolError as e:
            sc = {"error": str(e)}
        hr = {k: v for k, v in inputs[cid]["features"].items() if k.startswith(HR_PREFIX)}
        out[cid] = {"shape_changes": sc, "features": hr | _shape_features(sc, inputs[cid].get("peak_lead"))}
    path.write_text(json.dumps(out, indent=1))
    return rid


def features(workers: int) -> None:
    OWN.mkdir(parents=True, exist_ok=True)
    by_rec: dict[str, list[str]] = {}
    for c in load_cases():
        by_rec.setdefault(c.record, []).append(c.candidate_id)
    with ProcessPoolExecutor(workers) as ex:
        for rid in ex.map(record_features, sorted(by_rec.items())):
            print(rid, flush=True)


def fit() -> None:
    from baselines.common import auc
    from baselines.feature_ml import folds, models
    cases = load_cases()
    rows = {}
    for f in OWN.glob("*.json"):
        rows |= json.loads(f.read_text())
    names = sorted({k for r in rows.values() for k in r["features"]})
    dev = [c for c in cases if c.split == "dev"]
    test = [c for c in cases if c.split == "test"]
    X = lambda cs: np.array([[rows[c.candidate_id]["features"].get(n, np.nan) for n in names] for c in cs], float)  # noqa: E731
    Xd, Xt, yd = X(dev), X(test), np.array([c.ischemic for c in dev])
    fd = folds([c.record for c in dev])
    for family, name in (("tree", "tree_faganeli_own"), ("logreg", "logreg_faganeli_own"),
                         ("nb", "nb_faganeli_own"), ("knn", "knn_faganeli_own")):
        best = None
        for label, make in models()[family]:
            oof = np.full(len(dev), np.nan)
            for k in range(5):
                tr, te = fd != k, fd == k
                oof[te] = make().fit(Xd[tr], yd[tr]).predict(Xd[te])
            a = auc(oof, yd)
            if best is None or a > best[0]:
                best = (a, label, make, oof)
        a, label, make, oof = best
        pt = make().fit(Xd, yd).predict(Xt)
        dev_score = {c.candidate_id: float(p) for c, p in zip(dev, oof)}
        test_score = {c.candidate_id: float(p) for c, p in zip(test, pt)}
        r = evaluate(name, cases, score=dev_score | test_score, dev_score=dev_score,
                     notes=f"features faganeli_own ({len(names)}: heart rate + our QRS/ST-T shape + ST-segment shape, "
                           f"no LTSTDB-supplied series), model {family} {label} (best of the grid by development "
                           f"cross-validated AUC {a:.3f}); development values are out-of-fold")
        r["features"], r["hyperparameters"] = names, label
        per = {cid: {"score": s, "split": "dev (out of fold)"} for cid, s in dev_score.items()}
        per |= {cid: {"score": s, "split": "test"} for cid, s in test_score.items()}
        save(name, r, per)
        print(name, label, {s: round(d["auc"], 3) for s, d in r["splits"].items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["features", "fit"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    features(a.workers) if a.cmd == "features" else fit()
