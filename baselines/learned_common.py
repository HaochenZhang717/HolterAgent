"""Shared protocol of the learned and LLM baselines (everything in baselines/ that is not a fixed rule).

Data: the fixed observation of outputs/baselines/_bundle (baselines/bundle.py): per candidate K = 8 raw segments of
10 s inside the candidate (index 0 centred on the ST extremum), one 10-s segment at the reference window, 2-3 leads
at 250 Hz in mV (lead_mask marks real leads), and the images trend / beats / overlay / strip / strip_ref.

Protocol (the same as baselines/feature_ml.py, so all baselines are comparable):
- development = 36 records / 587 candidates, test = 25 records / 234 candidates, split by patient
  (outputs/ltstdb_cases/{dev,test}_v07.json);
- trained models: record-grouped 5-fold cross-validation on the development records with the fold assignment of
  feature_ml.folds (seed 0); hyperparameters chosen by out-of-fold development AUC; development scores reported
  are out-of-fold; the test score is the mean of the 5 fold models (or a refit on all development records, say
  which in the notes); the test set is scored once, after every choice is fixed;
- untrained (zero-shot) methods: the same scoring on development and test;
- score = higher means more likely ischemic; the decision threshold is the one that maximises balanced accuracy on
  the development scores (common.evaluate); AUC with a 95% interval from bootstrapping records.

    from baselines.learned_common import load_bundle, dev_folds, finish
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from baselines.common import OUT, evaluate, load_cases, save
from baselines.feature_ml import folds

BUNDLE = OUT / "_bundle"


@dataclass
class Bundle:
    ids: list[str]           # candidate ids, dev first then test (order of common.load_cases)
    split: np.ndarray        # "dev" / "test"
    record: np.ndarray
    y: np.ndarray            # bool, ischemic
    reference: np.ndarray    # reference class (ischemic, heart_rate_related, axis_or_conduction, other, ...)
    cur: np.ndarray          # [N, 8, 3, 2500] float32, mV
    ref: np.ndarray          # [N, 3, 2500]
    lead_mask: np.ndarray    # [N, 3] bool
    meta: dict               # per-candidate metadata (leads, times, image paths relative to BUNDLE)

    def image(self, cid: str, kind: str):
        rel = self.meta[cid]["images"].get(kind)
        return BUNDLE / rel if rel else None

    @property
    def dev(self) -> np.ndarray:
        return np.flatnonzero(self.split == "dev")

    @property
    def test(self) -> np.ndarray:
        return np.flatnonzero(self.split == "test")


def load_bundle(signals: bool = True) -> Bundle:
    meta = json.loads((BUNDLE / "meta.json").read_text())
    cases = load_cases()
    ids = [c.candidate_id for c in cases]
    assert ids == meta["order"], "bundle is out of date: rerun python -m baselines.bundle"
    cur = ref = mask = None
    if signals:
        z = np.load(BUNDLE / "signals.npz")
        assert list(z["ids"]) == ids
        cur, ref, mask = z["cur"], z["ref"], z["lead_mask"]
    return Bundle(ids, np.array([c.split for c in cases]), np.array([c.record for c in cases]),
                  np.array([c.ischemic for c in cases]), np.array([c.reference for c in cases]),
                  cur, ref, mask, meta["cases"])


def dev_folds(b: Bundle) -> np.ndarray:
    """Fold index (0-4) of every development candidate, in the order of b.dev (feature_ml.folds, seed 0)."""
    return folds(list(b.record[b.dev]))


def finish(name: str, b: Bundle, dev_score: dict[str, float], test_score: dict[str, float], notes: str,
           extra: dict | None = None, label: dict[str, bool] | None = None) -> dict:
    """Score and save a baseline in the format of the other baselines (outputs/baselines/<name>/).
    `label` optionally gives the method's own binary decision (e.g. an LLM's label) on top of the score."""
    cases = load_cases()
    r = evaluate(name, cases, score=dev_score | test_score, dev_score=dev_score or None, notes=notes,
                 label=label)
    if extra:
        r.update(extra)
    per = {cid: {"score": s, "split": "dev"} for cid, s in dev_score.items()}
    per |= {cid: {"score": s, "split": "test"} for cid, s in test_score.items()}
    if label:
        for cid, v in label.items():
            per.setdefault(cid, {})["label"] = bool(v)
    save(name, r, per)
    return r
