"""Shared pieces of the baselines: cases, candidate geometry, scoring.

All baselines use the patient-level split of outputs/ltstdb_cases/split_v07.json (development: 36 records,
587 candidates; test: 25 records, 234 candidates) and score ischemic vs non-ischemic per candidate, like the
agent runs. Methods that output a score are scored by AUC (95% interval by bootstrapping records) and, at a
threshold chosen on the development set (maximum balanced accuracy), by sensitivity, specificity, PPV, F1
and balanced accuracy; binary rules are scored at their own decision.

External evaluation (2026-10-23): with BL_TEST_CASES=ltstdb_unused,edb,staffiii the test split is replaced by the
candidates of those split files in code/RSI/splits (the development split is unchanged, so every model is trained and
its threshold chosen exactly as before), and BL_OUT redirects every output (e.g. outputs/baselines_ext), so the
LTSTDB rows are not overwritten. data_dirs(record) gives each record's data and feature-cache directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "outputs" / "ltstdb_cases"
OUT = Path(os.environ.get("BL_OUT") or ROOT / "outputs" / "baselines")
TEST_SETS = [s for s in os.environ.get("BL_TEST_CASES", "").split(",") if s]
DATA_ROOT = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data"))
# record-name prefix -> (data directory, feature cache): LTSTDB s*, European ST-T e*, STAFF III pseudo-records t*,
# third test set m* (scripts/mimic_records.py, 2026-10-30)
DATA_DIRS = {"s": (DATA_ROOT / "ltstdb", ROOT / "cache" / "features_ltstdb"),
             "e": (DATA_ROOT / "edb", ROOT / "cache" / "features_edb"),
             "t": (DATA_ROOT / "staffiii_holter", ROOT / "cache" / "features_staffiii"),
             "m": (DATA_ROOT / "mimic_holter", ROOT / "cache" / "features_mimic")}


def data_dirs(record: str) -> tuple[Path, Path]:
    return DATA_DIRS[record[0]]


def protocol_of(record: str) -> str:
    """Candidate rule of a record's data set: LTSTDB protocol A for the third test set (m*), B elsewhere."""
    return "A" if record.startswith("m") else "B"


@dataclass(frozen=True)
class Case:
    candidate_id: str
    record: str
    reference: str
    split: str  # dev or test
    dataset: str = "ltstdb"  # test cases of an external evaluation: the split file they come from

    @property
    def ischemic(self) -> bool:
        return self.reference == "ischemic"


def load_cases() -> list[Case]:
    out = [Case(c["candidate_id"], c["record"], c["reference"], "dev")
           for c in json.loads((CASES / "dev_v07.json").read_text())["cases"]]
    if TEST_SETS:
        for name in TEST_SETS:
            for c in json.loads((ROOT / "RSI" / "splits" / f"{name}.json").read_text())["cases"]:
                out.append(Case(c["candidate_id"], c["record"], c["reference"], "test", name))
    else:
        out += [Case(c["candidate_id"], c["record"], c["reference"], "test")
                for c in json.loads((CASES / "test_v07.json").read_text())["cases"]]
    return out


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC; ties count one half."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks of ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(pred: np.ndarray, y: np.ndarray) -> dict:
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    return {"se": se, "sp": sp, "ppv": tp / max(1, tp + fp), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "balanced": (se + sp) / 2, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap(fn, records: list[str], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of fn(index array) over records resampled with replacement."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(records))
    idx = {r: np.flatnonzero(np.array(records) == r) for r in uniq}
    vals = []
    for _ in range(n):
        sel = np.concatenate([idx[r] for r in rng.choice(uniq, len(uniq))])
        vals.append(fn(sel))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (ties: the middle of the best run)."""
    cand = np.unique(score[np.isfinite(score)])
    if len(cand) == 0:
        return 0.5
    vals = [rates(score >= t, y)["balanced"] for t in cand]
    best = np.flatnonzero(np.isclose(vals, max(vals)))
    return float(cand[best[len(best) // 2]])


def evaluate(name: str, cases: list[Case], score: dict[str, float] | None = None,
             label: dict[str, bool] | None = None, dev_score: dict[str, float] | None = None,
             notes: str = "") -> dict:
    """Score a baseline on dev and test. `score` (higher = more ischemic) or `label` (binary).
    For a trained model, `dev_score` holds its out-of-fold development predictions and `score` its test
    predictions; the threshold comes from the development predictions."""
    res = {"name": name, "notes": notes, "splits": {}}
    thr = None
    if score is not None:
        dev = [c for c in cases if c.split == "dev"]
        ds = dev_score or score
        y = np.array([c.ischemic for c in dev])
        s = np.array([ds.get(c.candidate_id, np.nan) for c in dev], float)
        ok = np.isfinite(s)
        thr = best_threshold(s[ok], y[ok])
        res["threshold_from_dev"] = thr
    for split in ("dev", "test"):
        sub = [c for c in cases if c.split == split]
        y = np.array([c.ischemic for c in sub])
        recs = [c.record for c in sub]
        r: dict = {"n": len(sub), "n_ischemic": int(y.sum())}
        if score is not None:
            src = dev_score if (split == "dev" and dev_score is not None) else score
            s = np.array([src.get(c.candidate_id, np.nan) for c in sub], float)
            s = np.where(np.isfinite(s), s, np.nanmedian(s))  # a missing score counts as the median
            r["auc"] = auc(s, y)
            r["auc_ci"] = bootstrap(lambda i: auc(s[i], y[i]), recs)
            r["at_dev_threshold"] = rates(s >= thr, y)
            r["f1_ci"] = bootstrap(lambda i: rates(s[i] >= thr, y[i])["f1"], recs)
        if label is not None:
            pred = np.array([bool(label.get(c.candidate_id, False)) for c in sub])
            r["rule"] = rates(pred, y)
            # f1_ci stays the interval at the development threshold when there is a score; the label's own
            # interval goes to rule_f1_ci (before 2026-10-12 it overwrote f1_ci)
            ci = bootstrap(lambda i: rates(pred[i], y[i])["f1"], recs)
            r["rule_f1_ci"] = ci
            if score is None:
                r["f1_ci"] = ci
        res["splits"][split] = r
    return res


def save(name: str, result: dict, per_case: dict[str, dict]) -> None:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(result, indent=1))
    (folder / "predictions.json").write_text(json.dumps(per_case, indent=1))
