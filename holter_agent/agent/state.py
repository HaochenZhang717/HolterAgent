"""Case state: the evidence ledger and the assessment history of one record.

Every tool result is stored as numbered evidence. Every final assessment must cite
evidence ids. A later final assessment for the same candidate is a revision: it is
kept next to the earlier ones (never overwritten) and, by default, must cite at
least one evidence item obtained after the assessment it replaces. Revisions
without new evidence are the "intrinsic self-correction" that the literature finds
unreliable (Huang et al., ICLR 2024; Kumar et al., ICLR 2025).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

LABELS = (
    "ischemic",
    "heart_rate_related",
    "axis_or_position_shift",
    "conduction_change",
    "noise_artifact",
    "other_nonischemic",
    # Two kinds of abstention, kept apart as in CliniCARE-Bench (Chatrath et al., 2026):
    "indeterminate_data",  # evidence is missing or unusable (signal quality, leads)
    "indeterminate_ambiguous",  # evidence is there but ST cannot be interpreted or conflicts
)
ABSTAIN_LABELS = ("indeterminate_data", "indeterminate_ambiguous")
STAGES = ("initial", "final")


class StateError(Exception):
    """An assessment was rejected; the message goes back to the agent."""


@dataclass
class Evidence:
    id: str
    tool: str
    args: dict
    result: dict
    step: int
    session: str
    tool_version: str = ""
    images: list[str] = field(default_factory=list)  # files of the images shown with this result


@dataclass
class Assessment:
    candidate_id: str
    stage: str
    label: str
    confidence: float
    rationale: str
    evidence_ids: list[str]
    step: int
    session: str  # which session produced it: initial, investigate, verify<k>, recheck, oneshot, baseline
    revision: int  # 0 for the first assessment of this stage, then 1, 2, ...
    previous_label: str | None = None
    new_evidence_ids: list[str] = field(default_factory=list)  # cited ids obtained after the previous final
    n_evidence_at_submit: int = 0
    p_ischemic: float | None = None  # graded output (harness 0.8): probability that the change is ischaemic
    evidence_table: dict | None = None  # graded output: findings for and against ischaemia with evidence ids


@dataclass
class CaseState:
    record_id: str
    evidence: list[Evidence] = field(default_factory=list)
    assessments: list[Assessment] = field(default_factory=list)
    step: int = 0
    tool_calls: int = 0
    stop_reasons: dict[str, dict[str, str]] = field(default_factory=dict)  # candidate -> session -> reason
    needs_review: dict[str, str] = field(default_factory=dict)  # candidate -> why a human should look

    # ---------------------------------------------------------------- evidence
    def add_evidence(self, tool: str, args: dict, result: dict, session: str = "", tool_version: str = "",
                     images: list[str] | None = None) -> str:
        eid = f"E{len(self.evidence) + 1}"
        self.evidence.append(Evidence(eid, tool, dict(args), result, self.step, session, tool_version,
                                      list(images or [])))
        return eid

    def find_evidence(self, tool: str, args: dict) -> Evidence | None:
        """An earlier identical call, if any (used to answer duplicates without re-running)."""
        for e in self.evidence:
            if e.tool == tool and e.args == args:
                return e
        return None

    def _index(self, eid: str) -> int:
        return int(eid[1:]) if eid.startswith("E") and eid[1:].isdigit() else -1

    # ---------------------------------------------------------------- assessments
    def submit(self, candidate_id: str, stage: str, label: str, confidence: float, rationale: str,
               evidence_ids: list[str], session: str = "", require_new_evidence: bool = True,
               p_ischemic: float | None = None, evidence_table: dict | None = None) -> Assessment:
        if stage not in STAGES:
            raise StateError(f"stage must be one of {STAGES}")
        if label not in LABELS:
            raise StateError(f"label must be one of {LABELS}")
        if not 0.0 <= confidence <= 1.0:
            raise StateError("confidence must be between 0 and 1")
        if p_ischemic is not None:
            if not 0.0 <= p_ischemic <= 1.0:
                raise StateError("p_ischemic must be between 0 and 1")
            if label != "indeterminate_data" and (label == "ischemic") != (p_ischemic >= 0.5):
                raise StateError(f"label {label!r} contradicts p_ischemic={p_ischemic:.2f}: use 'ischemic' when "
                                 "p_ischemic >= 0.5 and the most likely non-ischemic cause otherwise")
        known = {e.id for e in self.evidence}
        unknown = [e for e in evidence_ids if e not in known]
        if unknown:
            raise StateError(f"unknown evidence ids {unknown}; cite ids returned by tool calls")
        if stage == "initial" and self.history(candidate_id, "initial"):
            raise StateError("an initial assessment was already recorded")
        if stage == "final" and not evidence_ids:
            raise StateError("a final assessment must cite at least one evidence id")
        prev = self.final(candidate_id) if stage == "final" else None
        new_ids: list[str] = []
        if prev is not None:
            new_ids = [e for e in evidence_ids if self._index(e) > prev.n_evidence_at_submit]
            if require_new_evidence and label != prev.label and not new_ids:
                raise StateError(
                    f"changing the final label from {prev.label!r} requires at least one evidence id obtained "
                    f"after that assessment (ids above E{prev.n_evidence_at_submit}); gather new evidence first "
                    "or keep the current label")
            earlier = [a.label for a in self.history(candidate_id, "final")[:-1]]
            if label != prev.label and label in earlier:
                self.needs_review[candidate_id] = f"final label oscillated back to {label!r}"
        a = Assessment(
            candidate_id=candidate_id, stage=stage, label=label, confidence=float(confidence),
            rationale=rationale, evidence_ids=list(evidence_ids), step=self.step, session=session,
            revision=len(self.history(candidate_id, stage)),
            previous_label=prev.label if prev else None, new_evidence_ids=new_ids,
            n_evidence_at_submit=len(self.evidence), p_ischemic=p_ischemic, evidence_table=evidence_table,
        )
        self.assessments.append(a)
        return a

    def history(self, candidate_id: str, stage: str | None = None) -> list[Assessment]:
        return [a for a in self.assessments
                if a.candidate_id == candidate_id and (stage is None or a.stage == stage)]

    def initial(self, candidate_id: str) -> Assessment | None:
        h = self.history(candidate_id, "initial")
        return h[0] if h else None

    def final(self, candidate_id: str) -> Assessment | None:
        h = self.history(candidate_id, "final")
        return h[-1] if h else None

    def set_stop(self, candidate_id: str, session: str, reason: str) -> None:
        self.stop_reasons.setdefault(candidate_id, {})[session] = reason

    def to_dict(self) -> dict:
        return asdict(self)
