"""Scoring of harness runs (frozen during the search; guard.py checks its hash).

Score of a candidate: the final p_ischemic; a candidate without a final assessment scores 0.5. Positive class:
LTSTDB reference "ischemic". Main metric: AUC. Uncertainty: bootstrap over records (the candidates of a record are
resampled together). Comparisons of two runs on the same candidates are paired.

Calibration (from 2026-10-08, after the pipeline of WaveformWorldModelInLanguage): the verbalized p_ischemic is used
only as a ranking score (AUC). Probabilities are obtained by Platt recalibration, logit(p_cal) = a * logit(p) + b,
fitted by log loss on a labelled set (in the search loop: the same agent's search-set results) and applied to another
set, with the intercept shifted from the fitting set's prevalence to the target prevalence (default: that of the set
being scored). calibration() reports Brier score and log loss before and after recalibration, calibration slope and
intercept, and the decisions at p_cal >= 0.5. Recalibration is strictly monotone and leaves AUC unchanged.

Task agents may also answer with a two-way decision (rsi/harness.py; recorded as p_ischemic 1 or 0), scored by F1 of
the decision (positive class ischemic); a candidate without a final assessment counts as non-ischemic. metrics()
reports F1 for every run (for a probability run at p >= 0.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

MISSING_SCORE = 0.5
N_BOOT = 2000


def load_cases(split_file: Path) -> list[dict]:
    return json.loads(Path(split_file).read_text())["cases"]


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_run(run_dir: Path) -> dict[str, dict]:
    """Per candidate: final p_ischemic and label, first p (before verification), credits, calls, stop reasons."""
    out = {}
    for f in Path(run_dir).glob("*.json"):
        if f.name == "metrics.json":
            continue
        d = json.loads(f.read_text())
        st = d["state"]
        for p in d["predictions"]:
            cid = p["candidate_id"]
            finals = [a for a in st["assessments"] if a["candidate_id"] == cid and a["stage"] == "final"]
            ev = st["evidence"]
            out[cid] = {
                "p": p.get("p_ischemic"),
                "first_p": p.get("first_p_ischemic"),
                "label": p.get("final_label"),
                "n_finals": len(finals),
                # ChatGPT credits (Codex CLI runs) or, for pool runs, the equivalent at the same rates
                "credits": (d.get("codex") or {}).get("credits") or (d.get("pool") or {}).get("credits_equivalent"),
                "analysis_calls": sum(1 for e in ev if e["session"] not in ("setup", "dossier")),
                "sessions": sorted(st["stop_reasons"].get(cid, {})),
                "stops": st["stop_reasons"].get(cid, {}),
                "needs_review": st["needs_review"].get(cid),
            }
    return out


def arrays(cases: list[dict], run: dict[str, dict], key: str = "p") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, labels (ischemic) and record names in the order of the split file."""
    s = np.array([MISSING_SCORE if run.get(c["candidate_id"], {}).get(key) is None else float(run[c["candidate_id"]][key])
                  for c in cases])
    y = np.array([c["reference"] == "ischemic" for c in cases])
    rec = np.array([c["record"] for c in cases])
    return s, y, rec


def _record_draws(rec: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Index sets of n bootstrap samples drawn over records."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(rec == r) for r in np.unique(rec)]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(np.concatenate([groups[i] for i in pick]))
    return draws


def auc_ci(s: np.ndarray, y: np.ndarray, rec: np.ndarray, n: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    vals = [auc(s[i], y[i]) for i in _record_draws(rec, n, seed)]
    vals = [v for v in vals if v == v]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired(cases: list[dict], run_a: dict, run_b: dict, n: int = N_BOOT, seed: int = 0) -> dict:
    """AUC(b) - AUC(a) on the same candidates, with a paired record bootstrap."""
    sa, y, rec = arrays(cases, run_a)
    sb, _, _ = arrays(cases, run_b)
    d = auc(sb, y) - auc(sa, y)
    ds = []
    for i in _record_draws(rec, n, seed):
        v = auc(sb[i], y[i]) - auc(sa[i], y[i])
        if v == v:
            ds.append(v)
    ds = np.array(ds)
    return {"auc_a": auc(sa, y), "auc_b": auc(sb, y), "delta": float(d),
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "p_gt0": float((ds > 0).mean()), "n_boot": int(len(ds))}


def rerun_noise(cases: list[dict], run_1: dict, run_2: dict, n: int = N_BOOT, q: float = 90.0, seed: int = 0) -> dict:
    """How large |AUC(A) - AUC(B)| gets between two runs of the same harness: for every candidate, its two results
    are assigned to A and B at random; delta is the q-th percentile of |dAUC| over n such draws."""
    s1, y, _ = arrays(cases, run_1)
    s2, _, _ = arrays(cases, run_2)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        swap = rng.random(len(s1)) < 0.5
        a, b = np.where(swap, s2, s1), np.where(swap, s1, s2)
        vals.append(abs(auc(a, y) - auc(b, y)))
    return {"delta": float(np.percentile(vals, q)), "quantile": q, "observed_abs_diff": abs(auc(s1, y) - auc(s2, y)),
            "auc_1": auc(s1, y), "auc_2": auc(s2, y), "mean_abs_p_diff": float(np.mean(np.abs(s1 - s2))), "n": n}


CLIP = 0.005  # verbalized probabilities can be 0 or 1: logits are taken of p clipped to [CLIP, 1 - CLIP]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def logistic_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, slope: bool = True,
                 ridge: float = 1e-3, iters: int = 100) -> tuple[float, float]:
    """Logistic regression of y on slope * x + intercept (+ offset) by Newton steps with a small ridge penalty;
    returns (slope, intercept). With slope=False the slope is fixed at 1 (calibration-in-the-large)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    off = np.zeros(len(x)) if offset is None else np.asarray(offset, float)
    X = np.c_[x, np.ones(len(x))] if slope else np.ones((len(x), 1))
    base = off if slope else off + x
    w = np.array([1.0, 0.0]) if slope else np.zeros(1)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(X @ w + base)))
        g = X.T @ (q - y) + ridge * w
        H = (X * (q * (1 - q))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return (float(w[0]), float(w[1])) if slope else (1.0, float(w[0]))


def prior_shift(pi_sample: float, pi_target: float) -> float:
    """Intercept change that moves log-odds from a sample prevalence to a target prevalence (Bayes prior adjustment)."""
    return float(np.log(pi_target / (1 - pi_target)) - np.log(pi_sample / (1 - pi_sample)))


def platt_fit(p, y) -> tuple[float, float]:
    """Platt recalibration logit(p_cal) = a * logit(p) + b fitted by log loss; returns (a, b)."""
    return logistic_fit(logit(p), np.asarray(y, float))


def platt_apply(p, ab: tuple[float, float], shift: float = 0.0) -> np.ndarray:
    a, b = ab
    return 1 / (1 + np.exp(-(a * logit(p) + b + shift)))


def proper_scores(p, y) -> dict:
    p, y = np.clip(np.asarray(p, float), CLIP, 1 - CLIP), np.asarray(y, float)
    return {"brier": float(np.mean((np.asarray(p) - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def calibration(cal_cases: list[dict], cal_run: dict[str, dict], eval_cases: list[dict], eval_run: dict[str, dict],
                target_prev: float | None = None) -> dict:
    """Platt recalibration fitted on (cal_cases, cal_run) and scored on (eval_cases, eval_run).

    Candidates without a probability are left out of the fit and score MISSING_SCORE in the evaluation. The intercept
    is shifted from the fitting set's prevalence to target_prev (default: the evaluation set's prevalence)."""
    fit = [c for c in cal_cases if cal_run.get(c["candidate_id"], {}).get("p") is not None]
    pc, yc, _ = arrays(fit, cal_run)
    ab = platt_fit(pc, yc)
    s, y, _ = arrays(eval_cases, eval_run)
    pi_s, pi_t = float(yc.mean()), float(y.mean() if target_prev is None else target_prev)
    shift = prior_shift(pi_s, pi_t)
    pcal = platt_apply(s, ab, shift)
    raw, cal = proper_scores(s, y), proper_scores(pcal, y)
    slope, intercept = logistic_fit(logit(pcal), y)
    citl = logistic_fit(logit(pcal), y, slope=False)[1]
    f = f1_stats(pcal >= 0.5, y)
    return {"platt_a": ab[0], "platt_b": ab[1], "fit_n": len(fit), "fit_prevalence": pi_s, "target_prevalence": pi_t,
            "prior_shift": shift, "brier_raw": raw["brier"], "log_loss_raw": raw["log_loss"],
            "brier": cal["brier"], "log_loss": cal["log_loss"], "brier_baseline": pi_t * (1 - pi_t),
            "calibration_slope": slope, "calibration_intercept": citl,
            "se_at_cal_0.5": f["recall"], "sp_at_cal_0.5": f["specificity"], "ppv_at_cal_0.5": f["precision"],
            "f1_at_cal_0.5": f["f1"], "auc": auc(s, y)}


def decisions(cases: list[dict], run: dict[str, dict], key: str = "p") -> tuple[np.ndarray, np.ndarray]:
    """Decided ischemic (p >= 0.5; no assessment = not ischemic) and reference labels, in split order."""
    d = np.array([run.get(c["candidate_id"], {}).get(key) is not None and float(run[c["candidate_id"]][key]) >= 0.5
                  for c in cases])
    return d, np.array([c["reference"] == "ischemic" for c in cases])


def f1_stats(d: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn, tn = int((d & y).sum()), int((d & ~y).sum()), int((~d & y).sum()), int((~d & ~y).sum())
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(cases: list[dict], run: dict[str, dict]) -> dict:
    s, y, rec = arrays(cases, run)
    done = [c["candidate_id"] for c in cases if c["candidate_id"] in run]
    missing_p = [c["candidate_id"] for c in cases if run.get(c["candidate_id"], {}).get("p") is None]
    pos = s >= 0.5
    se = float((pos & y).sum() / max(1, y.sum()))
    sp = float((~pos & ~y).sum() / max(1, (~y).sum()))
    credits = [run[c]["credits"] for c in done if run[c]["credits"] is not None]
    out = {
        "n_cases": len(cases), "n_ischemic": int(y.sum()), "n_run": len(done), "n_without_p": len(missing_p),
        "auc": auc(s, y), "auc_ci95": list(auc_ci(s, y, rec)),
        "se_at_0.5": se, "sp_at_0.5": sp, "balanced_acc_at_0.5": (se + sp) / 2,
        "n_indeterminate_data": sum(1 for c in done if run[c]["label"] == "indeterminate_data"),
        "credits_total": float(sum(credits)), "credits_per_case": float(np.mean(credits)) if credits else None,
        "analysis_calls_per_case": float(np.mean([run[c]["analysis_calls"] for c in done])) if done else None,
        "verified_fraction": float(np.mean([any(x.startswith("verify") for x in run[c]["sessions"]) for c in done]))
        if done else None,
        "median_p_ischemic": float(np.median(s[y])), "median_p_nonischemic": float(np.median(s[~y])),
    }
    out |= f1_stats(*decisions(cases, run)) | proper_scores(s, y)
    if any(run.get(c["candidate_id"], {}).get("first_p") is not None for c in cases):
        fs, _, _ = arrays(cases, run, "first_p")
        out["auc_first_assessment"] = auc(fs, y)
        out["f1_first_assessment"] = f1_stats(*decisions(cases, run, "first_p"))["f1"]
    return out
