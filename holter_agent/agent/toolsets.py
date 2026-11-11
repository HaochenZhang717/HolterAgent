"""Tool sets of a run, so the toolkit additions of version 0.6 can be compared with the earlier set.

  base  the 13 tools of toolkit 0.5 (episode_dynamics without the ST/HR slopes by phase)
  ecg   + shape_changes and the ST/HR slopes by phase in episode_dynamics
  full  + patient_context (clinical information from the record header)

A name ending in "+est" (e.g. "base+est") also offers ischemia_estimate, a statistical estimate fitted on
LTSTDB development records (agent/estimate.py).
"""
from __future__ import annotations

TOOLSETS = ("base", "ecg", "full")
NEW_TOOLS = {"shape_changes": "ecg", "patient_context": "full"}
PHASE_SLOPE_KEYS = ("st_hr_slope_to_extremum_uv_per_bpm", "st_hr_slope_from_extremum_uv_per_bpm")


EST_SUFFIX = "+est"
EST_TOOL = "ischemia_estimate"


def _base(toolset: str) -> str:
    return toolset[:-len(EST_SUFFIX)] if toolset.endswith(EST_SUFFIX) else toolset


def check(toolset: str) -> None:
    if _base(toolset) not in TOOLSETS:
        raise ValueError(f"toolset must be one of {TOOLSETS}, optionally with {EST_SUFFIX!r}")


# Tools added with harness 0.8; offered only to runs that list them (RunConfig.extra_tools).
EXTRA_TOOLS = ("episode_context", "rate_response")


def allows(toolset: str, tool: str, extra: tuple[str, ...] = ()) -> bool:
    """Whether the tool set (plus the run's extra tools) offers this tool."""
    if tool in EXTRA_TOOLS:
        return tool in extra
    if tool == EST_TOOL:
        return toolset.endswith(EST_SUFFIX)
    first = NEW_TOOLS.get(tool)
    return first is None or TOOLSETS.index(_base(toolset)) >= TOOLSETS.index(first)


def shown_result(toolset: str, tool: str, result: dict) -> dict:
    """The tool result as this tool set returns it (base: without the ST/HR slopes by phase)."""
    if _base(toolset) != "base" or tool != "episode_dynamics":
        return result
    out = dict(result)
    out["leads"] = {lead: {k: v for k, v in d.items() if k not in PHASE_SLOPE_KEYS}
                    for lead, d in result.get("leads", {}).items()}
    out["notes"] = [n for n in result.get("notes", []) if "to/from_extremum" not in n]
    if any("use the episode plus" in n for n in out["notes"]):  # restore the 0.5 wording
        out["notes"] = [("st_hr_* describe ST deviation vs heart rate within +-context_s of the episode"
                         if "use the episode plus" in n else n) for n in out["notes"]]
    return out
