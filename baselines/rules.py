"""Rule baselines without learning: the ACC/AHA 1999 criterion, and the fixed rule pipeline.

acc_aha_1999        ischemic if the candidate has ST depression >=100 uV lasting >=60 s in some lead
                    (check_criteria, rule acc_aha_1999: Crawford et al., J Am Coll Cardiol 1999;34:912-948,
                    "flat or downsloping ST-segment depression >=0.1 mV ... for a minimum period of 1 minute").
                    Score for the AUC: the longest such run in seconds (0 if none).
acc_aha_1999_shape  the same, and the median ST slope of the qualifying windows <= 0 mV/s (our
                    operationalisation of "flat or downsloping": the slope from J to the ST point).
rule_pipeline       holter_agent/agent/baseline.py: fixed rules with engineering-default thresholds over the
                    same check set the agent can request (signal quality, rhythm, QRS change, heart-rate
                    dependence, criteria). Score: its confidence for ischemic, 1 - confidence otherwise, 0.5 for
                    indeterminate.

Neither is fitted to data, so the development and test results are both out of sample.
"""
from __future__ import annotations

from baselines.common import evaluate, load_cases, save
from baselines.evidence import load_inputs
from holter_agent.agent.baseline import BaselineConfig, decide


def acc_aha(ev: dict) -> tuple[bool, bool, float]:
    cc = ev.get("check_criteria") or {}
    leads = cc.get("leads_meeting_depression") or {}
    longest = max((v.get("longest_run_s") or 0 for v in leads.values()), default=0)
    met = bool(cc.get("depression_ge_100uv_ge_60s"))
    shape = met and any((v.get("longest_run_s") or 0) >= 60 and (v.get("median_st_slope_mv_per_s") is not None)
                        and v["median_st_slope_mv_per_s"] <= 0 for v in leads.values())
    return met, shape, float(longest)


def run() -> list[dict]:
    cases = load_cases()
    inputs = load_inputs()
    results = []
    met, shape, longest = {}, {}, {}
    for c in cases:
        m, s, lng = acc_aha(inputs[c.candidate_id]["evidence"])
        met[c.candidate_id], shape[c.candidate_id], longest[c.candidate_id] = m, s, lng
    r = evaluate("acc_aha_1999", cases, score=longest, label=met,
                 notes="depression >=100 uV for >=60 s; score = longest qualifying run (s)")
    save("acc_aha_1999", r, {cid: {"met": met[cid], "longest_run_s": longest[cid]} for cid in met})
    results.append(r)
    r = evaluate("acc_aha_1999_shape", cases, label=shape,
                 notes="as acc_aha_1999 and median J-to-ST slope of the qualifying windows <= 0 mV/s")
    save("acc_aha_1999_shape", r, {cid: {"met": shape[cid]} for cid in shape})
    results.append(r)

    cfg = BaselineConfig()
    score, label, per = {}, {}, {}
    for c in cases:
        x = inputs[c.candidate_id]
        ev = {k: (v, f"E{i + 1}") for i, (k, v) in enumerate(x["evidence"].items())}
        lab, conf, why, _ = decide(x["summary"], ev, cfg)
        label[c.candidate_id] = lab == "ischemic"
        score[c.candidate_id] = conf if lab == "ischemic" else (0.5 if lab.startswith("indeterminate") else 1 - conf)
        per[c.candidate_id] = {"label": lab, "confidence": conf, "rationale": why}
    r = evaluate("rule_pipeline", cases, score=score, label=label,
                 notes="holter_agent/agent/baseline.py with BaselineConfig defaults")
    save("rule_pipeline", r, per)
    results.append(r)
    return results


if __name__ == "__main__":
    for r in run():
        print(r["name"], {s: {k: v for k, v in d.items() if k in ("auc", "rule")} for s, d in r["splits"].items()})
