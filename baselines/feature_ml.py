"""Classical feature-based classifiers of ischemic vs non-ischemic candidates, trained on the development
records only and evaluated once on the test records.

Feature sets (baselines/evidence.py):
  tools      features of our tool outputs (the fixed check set; holter_agent/agent/estimate.py)
  faganeli   heart rate (tool_hr_*), ST/heart-rate slopes and hysteresis (st_hr_*, hr_half_*), KLT Mahalanobis
             distances (kl_*) and ST-segment Legendre coefficients (leg_*): the feature families of
             Faganeli & Jager 2010 (Physiol Meas; "HR + Mahalanobis + Legendre", decision tree), re-implemented
             in spirit on our candidates, not their exact method
  ecg        tools + faganeli (every ECG feature)
  ecg_clin   ecg + clinical features of the record header (reference: what clinical information adds)

Models: L2 logistic regression (median imputation, standardised), decision tree, gradient boosting
(scikit-learn). Hyperparameters are chosen by record-grouped 5-fold cross-validation on the development
records; the out-of-fold development predictions give the decision threshold (maximum balanced accuracy);
the model is then refitted on all development candidates and applied to the test candidates.
"""
from __future__ import annotations

import numpy as np

from baselines.common import evaluate, load_cases, save
from baselines.evidence import load_inputs
from holter_agent.agent.estimate import fit_logreg

FAGANELI_PREFIX = ("tool_hr_", "st_hr_", "hr_half_", "kl_", "leg_")


def feature_names(rows: dict[str, dict], kind: str) -> list[str]:
    names = sorted({k for r in rows.values() for k in r["features"]})
    if kind == "tools":
        return [n for n in names if n.startswith("tool_")]
    if kind == "faganeli":
        return [n for n in names if n.startswith(FAGANELI_PREFIX)]
    if kind in ("ecg", "ecg_clin"):
        out = list(names)
        if kind == "ecg_clin":
            out += sorted({k for r in rows.values() for k in r["clinical"]})
        return out
    raise ValueError(kind)


def matrix(rows: dict[str, dict], ids: list[str], names: list[str]) -> np.ndarray:
    return np.array([[(rows[i]["features"] | rows[i]["clinical"]).get(n, np.nan) for n in names] for i in ids],
                    float)


class LogReg:
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X, y):
        med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = Xi.mean(0), Xi.std(0) + 1e-9
        self.w = fit_logreg((Xi - self.mu) / self.sd, y.astype(float), self.lam)
        return self

    def predict(self, X):
        Z = (np.where(np.isfinite(X), X, self.med) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(np.hstack([Z, np.ones((len(Z), 1))]) @ self.w)))


class Sk:
    """A scikit-learn classifier with median imputation where it cannot take NaN."""

    def __init__(self, make, impute: bool):
        self.make, self.impute = make, impute

    def fit(self, X, y):
        med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        self.m = self.make().fit(np.where(np.isfinite(X), X, self.med) if self.impute else X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(np.where(np.isfinite(X), X, self.med) if self.impute else X)[:, 1]


def models() -> dict[str, list[tuple[str, callable]]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    return {
        "logreg": [(f"lam={lam}", lambda lam=lam: LogReg(lam)) for lam in (0.3, 1.0, 3.0, 10.0)],
        "tree": [(f"depth={d},leaf={m}", lambda d=d, m=m: Sk(lambda: DecisionTreeClassifier(
            max_depth=d, min_samples_leaf=m, random_state=0), impute=True)) for d in (2, 3, 4, 6) for m in (10, 25)],
        "gbm": [(f"depth={d},iter={n}", lambda d=d, n=n: Sk(lambda: HistGradientBoostingClassifier(
            max_depth=d, max_iter=n, learning_rate=0.05, l2_regularization=1.0, random_state=0), impute=False))
            for d in (2, 3) for n in (100, 300)],
        # The other two classifiers of Faganeli's thesis (2013, Table 3.2): naive Bayes and k nearest neighbours,
        # on standardised features.
        "nb": [("gaussian", lambda: Sk(lambda: make_pipeline(StandardScaler(), GaussianNB()), impute=True))],
        "knn": [(f"k={k}", lambda k=k: Sk(lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=k)), impute=True)) for k in (5, 15, 31, 61)],
    }


def folds(records: list[str], k: int = 5) -> np.ndarray:
    uniq = sorted(set(records))
    perm = np.random.default_rng(0).permutation(uniq)
    of = {r: i % k for i, r in enumerate(perm)}
    return np.array([of[r] for r in records])


def fit_eval(name: str, kind: str, family: str, rows, cases) -> tuple[dict, dict]:
    from baselines.common import auc
    dev = [c for c in cases if c.split == "dev"]
    test = [c for c in cases if c.split == "test"]
    names = feature_names(rows, kind)
    Xd, yd = matrix(rows, [c.candidate_id for c in dev], names), np.array([c.ischemic for c in dev])
    Xt = matrix(rows, [c.candidate_id for c in test], names)
    fd = folds([c.record for c in dev])
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
                 notes=f"features {kind} ({len(names)}), model {family} {label} (best of the grid by "
                       f"development cross-validated AUC {a:.3f}); development values are out-of-fold")
    r["features"], r["hyperparameters"] = names, label
    per = {cid: {"score": s, "split": "dev (out of fold)"} for cid, s in dev_score.items()}
    per |= {cid: {"score": s, "split": "test"} for cid, s in test_score.items()}
    return r, per


RUNS = [("logreg_tools", "tools", "logreg"),            # = the ischemia_estimate tool's model
        ("tree_faganeli", "faganeli", "tree"),          # Faganeli & Jager 2010 style
        ("logreg_faganeli", "faganeli", "logreg"),
        ("nb_faganeli", "faganeli", "nb"),
        ("knn_faganeli", "faganeli", "knn"),
        ("gbm_ecg", "ecg", "gbm"),
        ("logreg_ecg", "ecg", "logreg"),
        ("gbm_ecg_clinical", "ecg_clin", "gbm")]         # reference: with clinical information


def run() -> list[dict]:
    cases = load_cases()
    rows = load_inputs()
    out = []
    for name, kind, family in RUNS:
        r, per = fit_eval(name, kind, family, rows, cases)
        save(name, r, per)
        out.append(r)
        print(name, r["notes"], {s: round(d["auc"], 3) for s, d in r["splits"].items()}, flush=True)
    return out


if __name__ == "__main__":
    run()
