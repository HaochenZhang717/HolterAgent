"""Fixed checks and a fixed rule pipeline over the same tools the agent uses.

`collect_standard_evidence` runs one fixed set of checks for a candidate. The rule
baseline decides from those outputs with fixed thresholds; the "oneshot" arm gives
the same outputs to an LLM for a single decision. Both therefore see exactly the
same information, which is what a fair agent-vs-pipeline comparison needs
(Xia et al. 2024 "Agentless"; Kapoor et al. 2024 "AI Agents That Matter").
All thresholds are engineering defaults (see BaselineConfig), not clinical rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import CaseState
from .toolkit import TOOLKIT_VERSION, HolterToolkit, ToolError, call_tool

StandardEvidence = dict[str, tuple[dict, str]]  # tool name -> (result, evidence id)


def peak_time(cand: dict) -> float:
    lead = max(cand["leads"], key=lambda k: abs(cand["leads"][k]["peak_uv"]))
    return float(cand["leads"][lead]["peak_s"])


def collect_standard_evidence(toolkit: HolterToolkit, state: CaseState, cand: dict, bound: dict,
                              session: str) -> StandardEvidence:
    """Run the fixed check set for one candidate and store every result in the ledger."""
    rid = state.record_id
    out: StandardEvidence = {}

    def run(name: str, **args) -> dict | None:
        try:
            result = call_tool(toolkit, name, {"record_id": rid, **args}, bound)
        except ToolError:
            return None
        state.tool_calls += 1
        out[name] = (result, state.add_evidence(name, {"record_id": rid, **args}, result, session, TOOLKIT_VERSION))
        return result

    t_peak = peak_time(cand)
    run("signal_quality", start_s=cand["start_s"], end_s=cand["end_s"])
    run("rhythm_context", start_s=cand["start_s"], end_s=cand["end_s"])
    run("measure_window", t_s=t_peak, duration_s=30.0)
    ref = run("find_reference_window", t_s=t_peak, max_distance_s=7200.0, hr_tolerance_bpm=10.0)
    if ref is None:
        ref = run("find_reference_window", t_s=t_peak, max_distance_s=7200.0, hr_tolerance_bpm=None)
    if ref is not None:
        run("compare_morphology", t_ref_s=ref["t_s"], t_s=t_peak, duration_s=30.0)
    run("morphology_steps", start_s=max(0.0, cand["start_s"] - 600.0), end_s=cand["end_s"] + 600.0,
        sync_tolerance_s=60.0)
    run("episode_dynamics", candidate_id=cand["id"], context_s=600.0)
    run("check_criteria", candidate_id=cand["id"], rule="acc_aha_1999")
    run("eligibility_check")
    return out


@dataclass(frozen=True)
class BaselineConfig:
    min_usable_frac: float = 0.5  # default: median usable fraction of involved leads
    min_changed_frac: float = 0.5  # default: share of involved leads with a QRS change
    axis_shift_deg: float = 20.0  # default
    qrs_widening_ms: float = 20.0  # default: QRS widening that indicates a conduction change
    hr_rise_bpm: float = 15.0  # default: HR rise from before to max within the episode
    min_hr_corr: float = 0.6  # default: |Spearman r| of ST vs HR in the involved leads


def decide(cand: dict, ev: StandardEvidence, cfg: BaselineConfig) -> tuple[str, float, str, list[str]]:
    """Fixed rules applied in order; returns (label, confidence, rationale, cited evidence ids)."""
    leads = list(cand["leads"])
    if "signal_quality" in ev:
        q, e = ev["signal_quality"]
        usable = float(np.median([q["leads"][lead]["usable_frac"] or 0.0 for lead in leads]))
        if usable < cfg.min_usable_frac:
            return "noise_artifact", 0.7, f"median usable fraction of involved leads {usable:.2f}", [e]
    if "morphology_steps" in ev:
        ms, e = ev["morphology_steps"]
        sync = [s for s in ms["synchronous_st_and_qrs_steps"] if s["st_step_lead"] in leads]
        if sync:
            q = next(q for q in ms["qrs_steps"] if q["t_start_s"] == sync[0]["qrs_step_t_s"])
            before, after = q["qrs_ms_before_after"]
            if before is not None and after is not None and after - before >= cfg.qrs_widening_ms:
                return "conduction_change", 0.6, f"QRS step with widening {before}->{after} ms at the ST step", [e]
            return "axis_or_position_shift", 0.6, f"QRS step coincides with ST steps in {sorted({s['st_step_lead'] for s in sync})}", [e]
    if "compare_morphology" in ev:
        mc, e = ev["compare_morphology"]
        changed = [lead for lead in leads if lead in mc["qrs_changed_leads"]]
        axis = mc["axis_change_deg"]
        if len(changed) / len(leads) >= cfg.min_changed_frac or (axis is not None and abs(axis) >= cfg.axis_shift_deg):
            widen = (mc["qrs_ms"] or 0) - (mc["qrs_ms_ref"] or 0)
            if widen >= cfg.qrs_widening_ms:
                return "conduction_change", 0.6, f"QRS widened by {widen} ms with QRS change in {changed}", [e]
            return "axis_or_position_shift", 0.6, f"QRS change in {changed} (axis change {axis} deg)", [e]
    if "episode_dynamics" in ev:
        dyn, e = ev["episode_dynamics"]
        rise = (dyn["hr"]["max_in_episode"] or 0) - (dyn["hr"]["baseline_30_to_10_min_before"] or 0)
        rhos = [abs(v["st_hr_spearman"]) for v in dyn["leads"].values() if v["st_hr_spearman"] is not None]
        if rise >= cfg.hr_rise_bpm and rhos and float(np.median(rhos)) >= cfg.min_hr_corr:
            return ("heart_rate_related", 0.55,
                    f"HR rose {rise:.0f} bpm and ST tracked HR (median |r| {np.median(rhos):.2f})", [e])
    if "check_criteria" in ev:
        crit, e = ev["check_criteria"]
        if crit["depression_ge_100uv_ge_60s"] and crit["separation_ge_60s"]:
            return "ischemic", 0.55, "meets the ACC/AHA 1999 depression criterion with no non-ischemic finding", [e]
    cited = [e for _, e in ev.values()]
    if not cited:
        return "indeterminate_data", 0.3, "no check could be run", []
    return "indeterminate_ambiguous", 0.3, "no rule matched", cited


class RuleBaseline:
    name = "rule_baseline"

    def __init__(self, toolkit: HolterToolkit, cfg: BaselineConfig | None = None,
                 protocol: str = "B", reference: str = "global"):
        self.toolkit = toolkit
        self.cfg = cfg or BaselineConfig()
        self.bound = {"protocol": protocol, "reference": reference}

    def assess(self, state: CaseState, cand: dict, session: str = "baseline") -> tuple[str, float, str, list[str]]:
        ev = collect_standard_evidence(self.toolkit, state, cand, self.bound, session)
        return decide(cand, ev, self.cfg)

    def run_record(self, record_id: str, candidate_ids: list[str] | None = None) -> CaseState:
        from .runner import open_case  # local import: runner imports this module

        state, ov_eid, cands = open_case(self.toolkit, record_id, self.bound, None)
        for cand, c_eid in cands:
            if candidate_ids is not None and cand["id"] not in candidate_ids:
                continue
            label, conf, why, eids = self.assess(state, cand)
            state.submit(cand["id"], "final", label, conf, why, [ov_eid, c_eid, *eids], session="baseline")
            state.set_stop(cand["id"], "baseline", "completed")
        return state
