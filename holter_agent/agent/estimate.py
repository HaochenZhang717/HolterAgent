"""A statistical estimate of how likely a candidate is ischemic, from the outputs of the fixed check set.

An L2 logistic regression on features of the tool outputs that the rule baseline collects for every
candidate (`baseline.collect_standard_evidence`). It is fitted on LTSTDB development records only
(scripts/train_estimator.py) and offered to the agent as one more tool, `ischemia_estimate`, in tool
sets ending in "+est". The estimate learns the database's labels, which partly rest on clinical
information the tools do not see, and it was fitted on 2-3 lead recordings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DEFAULT_WEIGHTS = Path(__file__).with_name("ischemia_estimator_dev.json")


def features(ev: dict, cand: dict) -> dict:
    """Features of one candidate. `ev` maps tool name -> result of the fixed check set; `cand` has
    peak_uv, start_s and end_s (a reference case or a candidate summary)."""
    f = {"depression": float(cand["peak_uv"] < 0), "abs_peak_uv": abs(cand["peak_uv"]),
         "duration_s": cand["end_s"] - cand["start_s"]}
    dyn = ev.get("episode_dynamics")
    if dyn:
        lead = dyn["peak_lead"]
        hr, L = dyn["hr"], dyn["leads"].get(lead, {})
        g = lambda d, k: np.nan if d.get(k) is None else float(d[k])  # noqa: E731
        if L.get("peak_dev_uv") is not None:  # the candidate list keeps only |peak|; the sign is here
            f["depression"] = float(L["peak_dev_uv"] < 0)
        f.update(hr_rise_5min_before=g(hr, "rise_5_min_before_onset"),
                 hr_rise_in_episode=g(hr, "max_in_episode") - g(hr, "baseline_30_to_10_min_before"),
                 hr_at_peak=g(hr, "at_st_peak"), hr_peak_lag_s=g(hr, "hr_max_minus_st_peak_s"),
                 st_hr_spearman_abs=abs(g(L, "st_hr_spearman")), st_hr_slope_abs=abs(g(L, "st_hr_slope_uv_per_bpm")),
                 onset_to_90pct_s=g(L, "onset_to_90pct_s"), offset_s=g(L, "90pct_to_end_s"),
                 plateau_frac=g(L, "plateau_frac_ge_80pct"), max_step_frac=g(L, "max_single_step_frac"))
        mw = ev.get("measure_window")
        if mw and lead in mw["leads"]:
            m = mw["leads"][lead]
            f.update(st_slope_mv_s=g(m, "slope_mv_per_s"), j0_minus_j80_uv=g(m, "j0_uv") - g(m, "j80_uv"),
                     t_amp_uv=g(m, "t_amp_uv"), noise_uv=g(m, "noise_uv"))
        mc = ev.get("compare_morphology")
        if mc and lead in mc["leads"]:
            m = mc["leads"][lead]
            f.update(qrs_corr=g(m, "qrs_corr"), qrs_pp_change=abs(g(m, "qrs_pp_ratio") - 1),
                     t_amp_change_abs=abs(g(m, "t_amp_change_uv")),
                     qrs_changed_frac=len(mc["qrs_changed_leads"]) / max(1, len(mc["leads"])))
    ms = ev.get("morphology_steps")
    if ms:
        f["sync_qrs_st_steps"] = float(len(ms["synchronous_st_and_qrs_steps"]))
        f["qrs_steps"] = float(len(ms["qrs_steps"]))
    sq = ev.get("signal_quality")
    if sq:
        f["fiducials_ok_frac"] = float(sq["fiducials_ok_frac"])
    cc = ev.get("check_criteria")
    if cc:
        f["acc_aha_met"] = float(cc["depression_ge_100uv_ge_60s"])
    rc = ev.get("rhythm_context")
    if rc:
        f["hr_median"] = float(rc["hr_bpm"]["median"])
    return f


def fit_logreg(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """L2-penalised logistic regression (intercept last, not penalised)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def loss(w):
        z = Xb @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[:-1] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y)
        g[:-1] += lam * w[:-1]
        return g

    return minimize(loss, np.zeros(Xb.shape[1]), jac=grad, method="L-BFGS-B").x


class IschemiaEstimator:
    """Standardised logistic regression with median imputation, loaded from a JSON file."""

    def __init__(self, path: str | Path = DEFAULT_WEIGHTS):
        d = json.loads(Path(path).read_text())
        self.names: list[str] = d["names"]
        self.median = np.array(d["median"], float)
        self.mu = np.array(d["mu"], float)
        self.sd = np.array(d["sd"], float)
        self.w = np.array(d["w"], float)
        self.meta: dict = d.get("meta", {})

    def z(self, f: dict) -> np.ndarray:
        x = np.array([f.get(n, np.nan) for n in self.names], float)
        x = np.where(np.isfinite(x), x, self.median)
        return (x - self.mu) / self.sd

    def predict(self, f: dict) -> tuple[float, dict[str, float]]:
        """Probability of the ischemic class and each feature's contribution to the log-odds."""
        z = self.z(f)
        contrib = z * self.w[:-1]
        p = 1.0 / (1.0 + np.exp(-(contrib.sum() + self.w[-1])))
        return float(p), {n: float(c) for n, c in zip(self.names, contrib)}
