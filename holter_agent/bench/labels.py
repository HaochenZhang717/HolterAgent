"""Benchmark label and prediction formats.

Reference labels are time intervals with a class, independent of any detector,
so the benchmark does not depend on our candidate generator. Predictions are the
reviewed candidates of a run (agent or baseline).

Label file (JSON):
{
  "records": {
    "ST-2026-9-4-0049": [
      {"start_s": 39600, "end_s": 40200, "label": "ischemic", "leads": ["V4", "V5"],
       "annotator": "A1", "note": ""}
    ]
  }
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agent.state import LABELS, CaseState

REFERENCE_LABELS = LABELS


@dataclass
class EpisodeLabel:
    record: str
    start_s: float
    end_s: float
    label: str
    leads: list[str] = field(default_factory=list)
    peak_s: float | None = None  # time of the extremum; used by the epicmp matching rule
    peak_dev_uv: float | None = None  # reference ST deviation at the extremum, if measured
    annotator: str = ""
    note: str = ""

    def __post_init__(self):
        if self.label not in REFERENCE_LABELS:
            raise ValueError(f"reference label {self.label!r} not in {REFERENCE_LABELS}")
        if self.end_s <= self.start_s:
            raise ValueError(f"empty interval {self.start_s}..{self.end_s}")


@dataclass
class PredictedEpisode:
    record: str
    candidate_id: str
    start_s: float
    end_s: float
    final_label: str | None
    peak_s: float | None = None  # time of the candidate's extremum
    initial_label: str | None = None  # blind first impression (no tools)
    first_final_label: str | None = None  # first final, before any verification round or recheck
    prior_label: str | None = None  # another system's label shown to the agent, if any
    confidence: float | None = None
    n_final_revisions: int = 0
    needs_review: bool = False
    p_ischemic: float | None = None  # graded output (harness 0.8)
    first_p_ischemic: float | None = None


def load_labels(path: str | Path) -> list[EpisodeLabel]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EpisodeLabel(record=record, **e) for record, eps in data["records"].items() for e in eps]


def save_labels(labels: list[EpisodeLabel], path: str | Path) -> None:
    by_rec: dict[str, list[dict]] = {}
    for lab in labels:
        d = asdict(lab)
        by_rec.setdefault(d.pop("record"), []).append(d)
    Path(path).write_text(json.dumps({"records": by_rec}, indent=1, ensure_ascii=False), encoding="utf-8")


def predictions_from_state(state: CaseState) -> list[PredictedEpisode]:
    """One prediction per candidate that the run reviewed."""
    cands = {e.args["candidate_id"]: e.result for e in state.evidence if e.tool == "candidate"}
    priors = {e.args["candidate_id"]: e.result["label"] for e in state.evidence if e.tool == "prior_label"}
    out = []
    for cid, c in cands.items():
        if cid not in state.stop_reasons:
            continue
        finals = state.history(cid, "final")
        initial = state.initial(cid)
        peak = max(c["leads"].values(), key=lambda v: abs(v["peak_uv"])) if c.get("leads") else None
        out.append(PredictedEpisode(
            record=state.record_id, candidate_id=cid, start_s=float(c["start_s"]), end_s=float(c["end_s"]),
            final_label=finals[-1].label if finals else None,
            peak_s=float(peak["peak_s"]) if peak else None,
            initial_label=initial.label if initial else None,
            first_final_label=finals[0].label if finals else None,
            prior_label=priors.get(cid),
            confidence=finals[-1].confidence if finals else None,
            n_final_revisions=max(0, len(finals) - 1),
            needs_review=cid in state.needs_review,
            p_ischemic=finals[-1].p_ischemic if finals else None,
            first_p_ischemic=finals[0].p_ischemic if finals else None,
        ))
    return out
