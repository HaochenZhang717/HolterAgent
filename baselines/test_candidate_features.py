"""Do proposed new tool outputs add information beyond the current tools? (LTSTDB)

Two groups of candidate features, computed for every usable LTSTDB candidate:
  st_hr_*   ST-HR relation within the episode, by phase: ST/HR slope while ST moves away from the
            reference (onset to extremum) and while it returns (extremum to end), the steeper of the two,
            a signed hysteresis (ST at the same heart rate on the way back minus on the way out), and the
            lag of the heart-rate rise behind the ST onset. Basis: Faganeli & Jager 2008 (CinC),
            Faganeli Pucer 2012 (steeper ST/HR slopes in ischemic episodes), exercise-test ST/HR
            hysteresis (survey C, section 1.2).
  kl_*      QRS and ST-segment shape change, from the Karhunen-Loeve series LTSTDB provides (.klt):
            change of the ST-segment and QRS Mahalanobis distances from 10 min before onset to the
            extremum, their ratio, and the largest 30-s QRS jump near the onset. Basis: PhysioNet LTSTDB
            ("axis shifts ... most apparent in the QRS complexes ... ischemic ST changes ... most apparent
            in the principal components of the ST segment"), Jager 1998, Mincholé 2005.
  clin_*    clinical context from the LTSTDB record headers (diagnoses, history, previous tests,
            medications), with the #Comments field removed because it describes the record's episodes.
            Basis: Jager et al. 2003 define heart-rate related vs ischemic episodes partly by whether
            "clinical information from the subject" suggests ischaemia.
For each group: AUC per feature, and the record-grouped cross-validated logistic regression of
analyze_ltstdb_features.py with and without the group.

    python scripts/test_candidate_features.py --data <ltstdb dir>
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ltstdb_features as A  # noqa: E402

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.paths import FEATURE_CACHE_DIR, OUTPUT_DIR  # noqa: E402
from holter_agent.trend.episodes import st_deviation  # noqa: E402

FEAT = OUTPUT_DIR / "ltstdb_features"


def st_hr_features(tab, lead: str, start: float, end: float, peak: float) -> dict:
    t = tab["t_start_s"] + tab.window_s / 2
    hr = tab["hr_bpm"]
    dev = st_deviation(tab, "global")[:, tab.lead_index(lead)]
    ok = np.isfinite(hr) & np.isfinite(dev)
    up = ok & (t >= start) & (t <= peak)
    down = ok & (t >= peak) & (t <= end)
    out = {}

    def slope(sel):
        if sel.sum() < 4 or np.ptp(hr[sel]) < 3:
            return np.nan
        return float(np.polyfit(hr[sel], dev[sel], 1)[0])

    su, sd = slope(up), slope(down)
    out["st_hr_slope_up_abs"] = abs(su)
    out["st_hr_slope_down_abs"] = abs(sd)
    out["st_hr_slope_max_abs"] = np.nanmax([abs(su), abs(sd)]) if np.isfinite([su, sd]).any() else np.nan
    # signed hysteresis: on a common HR grid, ST on the way back minus on the way out, signed so that
    # positive means a larger deviation on the way back
    if up.sum() >= 4 and down.sum() >= 4:
        lo, hi = max(hr[up].min(), hr[down].min()), min(hr[up].max(), hr[down].max())
        if hi - lo >= 3:
            grid = np.linspace(lo, hi, 10)
            iu, idn = np.argsort(hr[up]), np.argsort(hr[down])
            a = np.interp(grid, hr[up][iu], dev[up][iu])
            b = np.interp(grid, hr[down][idn], dev[down][idn])
            sign = np.sign(dev[ok & (np.abs(t - peak) < 60)].mean()) if (ok & (np.abs(t - peak) < 60)).any() else 1
            out["st_hr_hysteresis_uv"] = float(np.mean((b - a) * sign))
    # lag of the heart-rate rise behind the ST onset: when HR first reaches half its rise
    pre = ok & (t >= start - 1800) & (t < start - 600)
    win = ok & (t >= start - 600) & (t <= peak)
    if pre.sum() >= 4 and win.sum() >= 4:
        base = np.median(hr[pre])
        rise = hr[win].max() - base
        if rise >= 5:
            first = t[win][np.argmax(hr[win] >= base + rise / 2)]
            out["hr_half_rise_minus_st_onset_s"] = float(first - start)
    return out


def kl_features(klt: np.ndarray, fs: float, start: float, end: float, peak: float) -> dict:
    t = klt[:, 0] / fs
    st_d, qrs_d = klt[:, 6], klt[:, 12]
    pre = (t >= start - 600) & (t < start)
    at = np.abs(t - peak) <= 30
    near = (t >= start - 120) & (t <= start + 120)
    if pre.sum() < 10 or at.sum() < 3:
        return {}
    d_st = float(np.median(st_d[at]) - np.median(st_d[pre]))
    d_qrs = float(np.median(qrs_d[at]) - np.median(qrs_d[pre]))
    out = {"kl_st_dist_change": d_st, "kl_qrs_dist_change": d_qrs,
           "kl_qrs_over_st": abs(d_qrs) / (abs(d_st) + 10.0)}
    q = qrs_d[near]
    if len(q) >= 30:  # 2-s steps: 15 points = 30 s
        out["kl_qrs_jump_30s"] = float(np.max(np.abs(q[15:] - q[:-15])))
    return out


def clinical_features(header: str) -> dict:
    h = "\n".join(line for line in header.splitlines() if line.startswith("#"))
    h = re.sub(r"#Comments:.*?(?=#Symptoms)", "", h, flags=re.S)  # describes the record's own episodes
    m = re.search(r"#Diagnoses:(.*?)#Treatment", h, re.S)
    diag = m.group(1) if m else ""
    yes = lambda key: bool(re.search(key + r":\s*Yes", h, re.I))  # noqa: E731
    stress = re.search(r"ECG stress test:\s*Yes.*?Findings:([^\n]*)", h, re.S)
    angio = re.search(r"Coronary Arteriography:\s*\n#\s*([^\n]*)", h)
    return {"clin_cad": float(bool(re.search(r"coronary|angina|ischemi|ischaemi|infarction", diag, re.I))
                              or yes("Previous Myocardial Infarction")),
            "clin_prior_mi": float(yes("Previous Myocardial Infarction")),
            "clin_lvh": float(yes("Left ventricular hypertrophy")),
            "clin_ivcd": float(yes("Intraventricular conduction block")),
            "clin_stress_st": float(bool(stress and re.search(r"ST", stress.group(1)))),
            "clin_stenosis": float(bool(angio and re.search(r"\d+\s*%", angio.group(1)))),
            "clin_beta_blocker": float(bool(re.search(r"olol\b", h, re.I)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    tk = HolterToolkit(args.data, FEATURE_CACHE_DIR.parent / "features_ltstdb")
    rows = json.loads((FEAT / "features.json").read_text())
    peaks = {}
    for f in (OUTPUT_DIR / "runs" / "ltstdb_all_rule-baseline" / "run1").glob("*.json"):
        for e in json.loads(f.read_text())["state"]["evidence"]:
            if e["tool"] == "episode_dynamics":
                r = e["result"]
                peaks[r["id"]] = (r["peak_lead"], r["peak_s"], r["start_s"], r["end_s"])
    klt_cache: dict[str, np.ndarray] = {}
    for r in rows:
        if r["candidate_id"] not in peaks:
            continue
        lead, peak, start, end = peaks[r["candidate_id"]]
        st = tk._state(r["record"])
        r.update(st_hr_features(st.table, lead, start, end, peak))
        if r["record"] not in klt_cache:
            with zipfile.ZipFile(Path(args.data) / f"{r['record']}.klt.zip") as z:
                klt_cache[r["record"]] = np.loadtxt(io.TextIOWrapper(z.open(z.namelist()[0])))
        r.update(kl_features(klt_cache[r["record"]], st.record.fs, start, end, peak))
        r.update(clinical_features((Path(args.data) / f"{r['record']}.hea").read_text()))
    y = np.array([r["reference"] == "ischemic" for r in rows])
    hr_cls = np.array([r["reference"] == "heart_rate_related" for r in rows])
    ax = np.array([r["reference"] == "axis_or_conduction" for r in rows])
    groups = {"st_hr": [k for k in rows[0].keys() | set().union(*rows) if k.startswith(("st_hr_slope_up", "st_hr_slope_down",
                                                                                          "st_hr_slope_max", "st_hr_hyst", "hr_half"))],
              "kl": [k for k in set().union(*rows) if k.startswith("kl_")],
              "clin": [k for k in set().union(*rows) if k.startswith("clin_")]}
    print(f"{len(rows)} candidates")
    print(f"{'feature':30s} {'n':>4s} {'AUC isch/rest':>13s} {'AUC isch/HR':>11s} {'AUC isch/axis':>13s}")
    for g, names in groups.items():
        for n in sorted(names):
            x = np.array([r.get(n, np.nan) for r in rows], float)
            print(f"{n:30s} {np.isfinite(x).sum():4d} {A.auc(x[y], x[~y]):13.2f} {A.auc(x[y], x[hr_cls]):11.2f} "
                  f"{A.auc(x[y], x[ax]):13.2f}")
    base = sorted({k for k in rows[0]} - {"candidate_id", "record", "reference", "hrm_resid", "hrm_frac"}
                  - set(groups["st_hr"]) - set(groups["kl"]) - set(groups["clin"]))
    recs = sorted({r["record"] for r in rows})
    perm = np.random.default_rng(0).permutation(recs)
    fold = np.array([{rec: i for i, rec in enumerate(perm)}[r["record"]] % 5 for r in rows])

    def cv(names, target):
        X = np.array([[r.get(n, np.nan) for n in names] for r in rows], float)
        prob = np.zeros(len(rows))
        for k in range(5):
            tr, te = fold != k, fold == k
            med = np.nanmedian(X[tr], 0)
            Xi = np.where(np.isfinite(X), X, med)
            mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
            Z = (Xi - mu) / sd
            w = A.fit_logreg(Z[tr], target[tr].astype(float))
            prob[te] = 1 / (1 + np.exp(-(np.hstack([Z[te], np.ones((te.sum(), 1))]) @ w)))
        return prob

    res = {}
    for label, names in (("current tools", base), ("+ ST-HR by phase", base + groups["st_hr"]),
                         ("+ QRS/ST shape (KLT)", base + groups["kl"]),
                         ("+ both", base + groups["st_hr"] + groups["kl"]),
                         ("+ clinical context", base + groups["clin"]),
                         ("+ all three", base + groups["st_hr"] + groups["kl"] + groups["clin"])):
        p = cv(names, y)
        res[label] = {"auc_isch_vs_rest": round(A.auc(p[y], p[~y]), 3),
                      "auc_isch_vs_hr": round(A.auc(p[y], p[hr_cls]), 3),
                      "auc_isch_vs_axis": round(A.auc(p[y], p[ax]), 3)}
        print(f"{label:24s} {res[label]}")
    (FEAT / "candidate_features.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
