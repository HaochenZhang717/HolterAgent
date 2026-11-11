"""Controlled corruption of tool outputs, to test whether the agent notices conflicting evidence.

LLMs tend to accept derived measurements as facts: in DFOT, Qwen3-8B accepted a
derived rhythm label that the reference modality contradicted in every case
(COTR = 1.000), and reliability evidence raised the repair rate only from 40.11%
to 44.83% (Guo et al., arXiv:2607.28421). A verification module that is almost
never triggered cannot be evaluated either: VitalAgent needed a second tool plan
for 0.4% of samples (Zhu et al., arXiv:2605.29483). The benchmark therefore needs
cases where one tool is wrong and others can reveal it.

`FaultInjectingToolkit` wraps a toolkit and corrupts the output of one tool when a
fault's condition holds; every injection is logged (the agent never sees the log).
Run the same configuration with and without faults and compare the labels of the
affected candidates with `fault_susceptibility`.
"""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field
from typing import Any, Callable

from .labels import PredictedEpisode


@dataclass(frozen=True)
class Fault:
    name: str
    tool: str  # toolkit method whose output is corrupted
    applies: Callable[[dict], bool]  # receives the call arguments
    mutate: Callable[[dict, dict], dict]  # (arguments, result) -> corrupted result
    description: str
    candidate_ids: tuple[str, ...] = ()  # candidates this fault is meant to affect (for scoring)


@dataclass
class InjectionEvent:
    fault: str
    tool: str
    args: dict


class FaultInjectingToolkit:
    """Proxy around a HolterToolkit that corrupts selected tool outputs."""

    def __init__(self, base: Any, faults: list[Fault]):
        self._base = base
        self._faults = list(faults)
        self.log: list[InjectionEvent] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._base, name)
        faults = [f for f in self._faults if f.tool == name]
        if not faults or not callable(attr):
            return attr

        @functools.wraps(attr)  # keeps the signature, so run-level bound arguments still reach the tool
        def wrapped(**kwargs):
            result = attr(**kwargs)
            for f in faults:
                if f.applies(kwargs):
                    result = f.mutate(kwargs, copy.deepcopy(result))
                    self.log.append(InjectionEvent(f.name, name, dict(kwargs)))
            return result

        return wrapped


def _in_range(t_key: str, start_s: float, end_s: float) -> Callable[[dict], bool]:
    return lambda args: start_s <= float(args.get(t_key, -1)) <= end_s


def st_offset_fault(lead: str, delta_uv: float, start_s: float, end_s: float,
                    candidate_ids: tuple[str, ...] = ()) -> Fault:
    """measure_window reports the ST of one lead shifted by delta_uv; st_trend does not."""
    def mutate(args: dict, result: dict) -> dict:
        entry = result.get("leads", {}).get(lead)
        if entry:
            for key in ("st_uv", "dev_uv", "j80_uv"):
                if entry.get(key) is not None:
                    entry[key] = round(entry[key] + delta_uv)
        return result

    return Fault(f"st_offset:{lead}:{delta_uv:+.0f}", "measure_window", _in_range("t_s", start_s, end_s), mutate,
                 f"measure_window shifts {lead} ST by {delta_uv:+.0f} uV in {start_s:.0f}-{end_s:.0f} s",
                 candidate_ids)


def spurious_qrs_step_fault(t_step_s: float, leads: list[str], candidate_ids: tuple[str, ...] = ()) -> Fault:
    """morphology_steps reports a QRS step with synchronous ST steps that does not exist."""
    def applies(args: dict) -> bool:
        return float(args["start_s"]) <= t_step_s < float(args["end_s"])

    def mutate(args: dict, result: dict) -> dict:
        t0 = round(t_step_s)
        result["qrs_steps"].append({"t_start_s": t0, "t_end_s": t0 + 150, "hms": "", "distance_change": 0.9,
                                    "frontal_axis_before_after_deg": [60, 20], "axis_change_deg": -40,
                                    "qrs_ms_before_after": [92, 94],
                                    "largest_qrs_area_changes_uvs": {lead: 25 for lead in leads}})
        for lead in leads:
            result["synchronous_st_and_qrs_steps"].append(
                {"qrs_step_t_s": t0, "st_step_lead": lead, "st_step_t_s": t0, "st_change_uv": -80})
        return result

    return Fault(f"spurious_qrs_step:{t_step_s:.0f}", "morphology_steps", applies, mutate,
                 f"morphology_steps adds a QRS step at {t_step_s:.0f} s synchronous with {leads}", candidate_ids)


def hidden_noise_fault(lead: str, start_s: float, end_s: float, candidate_ids: tuple[str, ...] = ()) -> Fault:
    """signal_quality reports a lead as clean although it is not."""
    def applies(args: dict) -> bool:
        return float(args["start_s"]) < end_s and float(args["end_s"]) > start_s

    def mutate(args: dict, result: dict) -> dict:
        entry = result.get("leads", {}).get(lead)
        if entry:
            entry.update({"usable_frac": 1.0, "noise_uv_median": 8, "noise_uv_p90": 12,
                          "saturated_window_frac": 0.0, "flat_window_frac": 0.0})
        return result

    return Fault(f"hidden_noise:{lead}", "signal_quality", applies, mutate,
                 f"signal_quality reports {lead} as clean in {start_s:.0f}-{end_s:.0f} s", candidate_ids)


@dataclass
class SusceptibilityReport:
    n_affected: int
    label_changed: int  # final label differs between the clean and the faulted run
    changed_ids: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        return self.label_changed / self.n_affected if self.n_affected else None


def fault_susceptibility(clean: list[PredictedEpisode], faulted: list[PredictedEpisode],
                         faults: list[Fault]) -> SusceptibilityReport:
    """Share of fault-targeted candidates whose final label changed because of the fault."""
    targets = {cid for f in faults for cid in f.candidate_ids}
    a = {p.candidate_id: p.final_label for p in clean if p.candidate_id in targets}
    b = {p.candidate_id: p.final_label for p in faulted if p.candidate_id in targets}
    common = sorted(set(a) & set(b))
    changed = [cid for cid in common if a[cid] != b[cid]]
    return SusceptibilityReport(len(common), len(changed), changed)
