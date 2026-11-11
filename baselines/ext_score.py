"""Per-dataset scores of the external evaluation (2026-10-23): every baseline row of BL_OUT and the three agent
versions on the test sets of BL_TEST_CASES (code/RSI/splits/<name>.json), scored separately per set.

Baselines were trained and their decision thresholds chosen on the LTSTDB development split, as for the LTSTDB rows
(common.py); their test predictions here cover all external sets at once and are split by data set. Per set:
  AUC with a 95% interval from bootstrapping records (common.bootstrap), and, at the development threshold (or the
  rule's own decision), sensitivity, specificity, PPV and F1. A missing score counts as the set's median score, as in
  common.evaluate. Reference rows: peak |ST deviation| of the candidate (no fitting) and every candidate ischemic.
Agent rows: mean p_ischemic over the 3 runs (RSI runs of tracks p56a / p56l), decision at p >= 0.5 (the model's own).
Direct-judgement rows (baselines/direct_llm.py with --rep): the same, from _runs/<tag>_r1..3.
Paired differences (table["diff_vs"][ref]): each improved agent version (REFS) minus every baseline row, per set, with a 95% interval and
the share of resamples above 0 from bootstrapping records with the same resamples for both; AUC on every set, and on
STAFF III also sensitivity at each row's own threshold (resampling patients). Agent and direct rows are resampled on
their 3-run mean scores only, so these intervals are a little narrower than the two-level ones of RSI/final_eval.py.
STAFF III has no reliable negatives (18 of its 19 non-inflation candidates start within 81 s after deflation), so
only its sensitivity is meant to be read.

    BL_OUT=outputs/baselines_ext BL_TEST_CASES=ltstdb_unused,edb,staffiii python -m baselines.ext_score
    BL_OUT=outputs/baselines_mimic BL_TEST_CASES=mimic python -m baselines.ext_score   # third test set (2026-11-02)
Writes BL_OUT/ext_table.json and ext_table.md.
"""
from __future__ import annotations

import json

import numpy as np

from baselines.common import OUT, ROOT, TEST_SETS, auc, bootstrap, load_cases, rates

# archive pick: a03 on the development part of validation_v4 (RSI/reselect_devonly.py, 2026-10-24); a09, the first pick
# on all of validation_v4 (165 of whose candidates are ltstdb_unused test candidates), is kept as a disclosed row
AGENTS = {"agent_start (v08p)": ("p56a", "v08p"), "agent_hill (p02)": ("p56l", "p02"), "agent_archive (a03)": ("p56a", "a03"),
          "agent_archive_first_pick (a09)": ("p56a", "a09")}
REFS = ("agent_hill (p02)", "agent_archive (a03)")  # the improved versions, each compared with every baseline row


def scores_of(row: str) -> tuple[dict[str, float], dict[str, bool], float | None]:
    """Candidate scores, binary labels (rules) and the development threshold of a baseline row."""
    pred = json.loads((OUT / row / "predictions.json").read_text())
    res = json.loads((OUT / row / "result.json").read_text())
    score, label = {}, {}
    for cid, v in pred.items():
        for k in ("score", "p", "p_ischemic", "longest_run_s"):
            if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                score[cid] = float(v[k])
                break
        for k in ("met", "label", "ischemic", "coronary"):
            if isinstance(v.get(k), bool):
                label[cid] = v[k]
                break
    # a test-only row (no development predictions, e.g. ecgscroll_judge_luna) has no development threshold: its own
    # decision (label) is used
    has_dev = any(str(v.get("split", "")).startswith("dev") for v in pred.values())
    return score, label, (res.get("threshold_from_dev") if has_dev or not label else None)


def agent_scores(track: str, hid: str, dataset: str) -> dict[str, float]:
    import sys
    sys.path.insert(0, str(ROOT / "RSI"))
    from rsi import scoring
    runs = [scoring.load_run(ROOT / "RSI" / f"runs_{track}" / dataset / f"{hid}_r{r}") for r in (1, 2, 3)]
    out = {}
    for cid in set().union(*runs):
        ps = [r[cid]["p"] for r in runs if cid in r and r[cid].get("p") is not None]
        if ps:
            out[cid] = float(np.mean(ps))
    return out


def direct_scores(tag: str, reps: int = 3) -> dict[str, float]:
    """Mean submitted p_ischemic over the repeated direct-judgement runs (_runs/<tag>_r1 ...)."""
    vals: dict[str, list[float]] = {}
    for r in range(1, reps + 1):
        for f in (OUT / "_runs" / f"{tag}_r{r}").glob("*.json"):
            d = json.loads(f.read_text())
            p = next((x for x in d.get("predictions", []) if x["candidate_id"] == f.stem), None)
            if p and p.get("p_ischemic") is not None:
                vals.setdefault(f.stem, []).append(float(p["p_ischemic"]))
    return {c: float(np.mean(v)) for c, v in vals.items()}


def _vec(sub, score: dict) -> np.ndarray:
    s = np.array([score.get(c.candidate_id, np.nan) for c in sub], float)
    return np.where(np.isfinite(s), s, np.nanmedian(s))


def paired_diff(sub, a: np.ndarray, b: np.ndarray, metric, groups: list[str], n: int = 2000) -> dict:
    """metric(a) - metric(b) on the same record (or patient) resamples."""
    rng = np.random.default_rng(0)
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(np.array(groups) == g) for g in uniq}
    d = [metric(a, sel) - metric(b, sel) for sel in
         (np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))]) for _ in range(n))]
    d = np.array(d, float)
    full = np.arange(len(sub))
    return {"d": float(metric(a, full) - metric(b, full)), "ci": [float(np.nanpercentile(d, 2.5)),
            float(np.nanpercentile(d, 97.5))], "p_gt0": float(np.nanmean(d > 0))}


def groups_of(sub) -> list[str]:
    """Resampling unit: the record; in the third test set the person (m<code><index>), who has several records."""
    return [c.record[:-1] if c.dataset == "mimic" else c.record for c in sub]


def score_set(sub, score: dict | None, label: dict | None, thr: float | None) -> dict:
    y = np.array([c.ischemic for c in sub])
    recs = groups_of(sub)
    out: dict = {"n": len(sub), "n_ischemic": int(y.sum())}
    if score:
        s = np.array([score.get(c.candidate_id, np.nan) for c in sub], float)
        out["n_missing"] = int((~np.isfinite(s)).sum())
        s = np.where(np.isfinite(s), s, np.nanmedian(s))
        out["auc"] = auc(s, y)
        out["auc_ci"] = bootstrap(lambda i: auc(s[i], y[i]), recs)
        if thr is not None:
            out["at_threshold"] = rates(s >= thr, y)
    if label is not None and (not score or thr is None):
        pred = np.array([bool(label.get(c.candidate_id, False)) for c in sub])
        out["at_threshold"] = rates(pred, y)
    return out


def main() -> None:
    cases = [c for c in load_cases() if c.split == "test"]
    sets = {name: [c for c in cases if c.dataset == name] for name in TEST_SETS}
    peak = {}
    for name in TEST_SETS:
        for c in json.loads((ROOT / "RSI" / "splits" / f"{name}.json").read_text())["cases"]:
            peak[c["candidate_id"]] = abs(float(c["peak_uv"]))
    table: dict = {"sets": {n: {"n": len(s), "n_ischemic": sum(c.ischemic for c in s)} for n, s in sets.items()},
                   "rows": {}}
    rows = sorted(p.name for p in OUT.iterdir() if (p / "result.json").exists() and (p / "predictions.json").exists())
    scored: dict[str, tuple[dict, dict | None, float | None]] = {}  # row -> (scores, labels, threshold) for the diffs
    for row in rows:
        score, label, thr = scores_of(row)
        table["rows"][row] = {n: score_set(s, score, label, thr) for n, s in sets.items()}
        if score:
            scored[row] = (score, label, thr)
    table["rows"]["ref_peak_deviation"] = {n: score_set(s, peak, None, None) for n, s in sets.items()}
    scored["ref_peak_deviation"] = (peak, None, None)
    table["rows"]["always_ischemic"] = {n: score_set(s, None, {c.candidate_id: True for c in s}, None)
                                        for n, s in sets.items()}
    for tag in ("direct_luna_both",):  # same model as the agent, no investigation; 3 runs, decision at p >= 0.5
        if (OUT / "_runs" / f"{tag}_r1").exists():
            sc = direct_scores(tag)
            table["rows"][tag + " (3-run mean)"] = {n: score_set(s, sc, None, 0.5) for n, s in sets.items()}
            scored[tag + " (3-run mean)"] = (sc, None, 0.5)
    agent = {}
    for name, (track, hid) in AGENTS.items():
        agent[name] = {n: agent_scores(track, hid, n) for n in TEST_SETS}
        table["rows"][name] = {n: score_set(s, agent[name][n], None, 0.5) for n, s in sets.items()}
        scored[name] = ({k: v for n in TEST_SETS for k, v in agent[name][n].items()}, None, 0.5)
    table["diff_vs"] = {}
    for ref in REFS:
        table["diff_vs"][ref] = {}
        for row, (score, label, thr) in scored.items():
            if row.startswith("agent_"):  # agent versions against each other: RSI/final_eval.py
                continue
            per = {}
            for n, sub in sets.items():
                y = np.array([c.ischemic for c in sub])
                a, b = _vec(sub, agent[ref][n]), _vec(sub, score)
                per[n] = {"auc": paired_diff(sub, a, b, lambda v, i: auc(v[i], y[i]), groups_of(sub))}
                if n == "staffiii" and (thr is not None or label):  # the peak reference has no decision
                    pa = a >= 0.5
                    pb = (b >= thr) if thr is not None else np.array([bool(label.get(c.candidate_id, False)) for c in sub])
                    se = lambda v, i: float(v[i][y[i]].mean()) if y[i].any() else np.nan
                    per[n]["se"] = paired_diff(sub, pa.astype(float), pb.astype(float), se, [c.record[:-1] for c in sub])
            table["diff_vs"][ref][row] = per
    (OUT / "ext_table.json").write_text(json.dumps(table, indent=1))
    lines = ["| row | " + " | ".join(f"{n} AUC | {n} Se / Sp" for n in TEST_SETS) + " |",
             "|---|" + "---|---|" * len(TEST_SETS)]
    for row, per in table["rows"].items():
        cells = []
        for n in TEST_SETS:
            r = per[n]
            a = f"{r['auc']:.3f} [{r['auc_ci'][0]:.2f}, {r['auc_ci'][1]:.2f}]" if "auc" in r else "-"
            t = r.get("at_threshold")
            cells += [a, f"{t['se']:.2f} / {t['sp']:.2f}" if t else "-"]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    (OUT / "ext_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
