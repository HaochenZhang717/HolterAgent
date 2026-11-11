"""The agent loop, organised as short sessions per candidate episode.

Sessions (each with a fresh message history):
- initial     : one call, no analysis tools, records a first impression. The
                investigating session does not see it, so it cannot anchor on it
                (sycophancy is ~3x higher when an answer already given is
                challenged; Ping et al. 2026, arXiv:2608.01017).
- investigate : the agent calls tools and submits a final assessment (the
                judgement). Optionally it is shown another system's label as a
                structured ledger record (task: revise someone else's output).
- verify<k>   : the verification loop. An independent session tests the current
                label with its own tool calls and either confirms it or replaces it;
                a replacement must cite evidence obtained in that session (evidence-
                free doubt harms accuracy: Almasoud 2026, arXiv:2603.06856). A
                replaced label is verified again in a new session. The loop stops
                when a round confirms the label; it flags the candidate for a human
                when the label oscillates, a round ends without an assessment, or
                the round limit is reached with the label still changing.
                `verify="ischemic"` verifies positives only (asymmetric verification:
                Kim et al. 2025, arXiv:2506.20112).
- recheck     : ablation only. One extra review with no tools, i.e. intrinsic
                self-correction (Huang et al., ICLR 2024).

The "oneshot" mode replaces investigate with a fixed set of checks followed by a
single model call, the same information the rule baseline gets.

Observations are multimodal (see observe.py): depending on the run's modality the
model gets the numbers of a tool result, an image drawn from the same data, or both.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .baseline import BaselineConfig, collect_standard_evidence, decide
from .llm import LLMBackend, ToolCall, tool_result_block, user_content
from .observe import ObservationConfig, case_image, tool_image, visible_case, visible_result
from . import toolsets
from .prompts import (
    GUIDANCE,
    SUBMIT_TOOL_GRADED,
    dossier_block,
    LAST_TURN_NOTE,
    NUDGE_FINAL,
    STALL_NOTE,
    SUBMIT_TOOL,
    case_header,
    initial_task,
    investigate_task,
    ledger_entry,
    oneshot_task,
    recheck_task,
    system,
    verify_task,
)
from .state import CaseState, StateError
from .toolkit import TOOL_SPECS, TOOLKIT_VERSION, HolterToolkit, ToolError, call_tool, jsonable

MODES = ("agent", "oneshot")
VERIFY = ("all", "ischemic", "none", "uncertain")


@dataclass
class RunConfig:
    protocol: str = "B"
    reference: str = "global"
    mode: str = "agent"
    initial: str = "separate"  # "separate": blind first impression; "none": skip
    prior_source: str = "none"  # "none" or "baseline": show the rule pipeline's label as a ledger record
    verify: str = "all"  # which final labels enter the verification loop: all, ischemic, none
    max_verify_rounds: int = 3
    intrinsic_recheck: bool = False  # ablation: extra no-tool review of the final
    max_turns: int = 24  # model calls per session (open models often call one tool per turn)
    max_tool_calls: int = 30  # analysis tool calls per session (submit_assessment is not counted)
    stall_turns: int = 3  # stop a session after this many turns without new evidence or a submission
    max_candidates: int | None = None  # review only the largest candidates
    observation: ObservationConfig = field(default_factory=ObservationConfig)  # numbers, images or both
    toolset: str = "full"  # base (toolkit 0.5 tools), ecg (+ shape_changes, ST/HR slopes by phase), full (+ patient_context)
    guidance: str = "v07b"  # rules on how to use the reference knowledge (v07b balanced, v07a one-sided); v06: before
    # Harness 0.8 (see HARNESS_V08): graded output (probability of ischaemia and findings for and against), a dossier
    # of standard checks in the first message of the judgement session, extra tools, and verification of uncertain
    # assessments only (p_ischemic inside verify_band).
    graded: bool = False
    dossier: bool = False
    extra_tools: tuple[str, ...] = ()
    verify_band: tuple[float, float] = (0.2, 0.8)

    def __post_init__(self):
        toolsets.check(self.toolset)
        if self.guidance not in GUIDANCE:
            raise ValueError(f"guidance must be one of {GUIDANCE}")

    @property
    def bound(self) -> dict:
        return {"protocol": self.protocol, "reference": self.reference}


def tool_definitions(analysis: bool = True, obs: ObservationConfig | None = None, toolset: str = "full",
                     extra: tuple[str, ...] = (), graded: bool = False) -> list[dict]:
    obs = obs or ObservationConfig()
    tools = [{"name": s.name, "description": s.description, "input_schema": s.input_schema}
             for s in TOOL_SPECS if obs.offers(s.name) and toolsets.allows(toolset, s.name, extra)] if analysis else []
    return tools + [SUBMIT_TOOL_GRADED if graded else SUBMIT_TOOL]


def harness_v08(cfg: "RunConfig") -> "RunConfig":
    """Harness 0.8 on top of a run configuration (scripts/run_review.py --harness v08)."""
    return replace(cfg, guidance="v08", graded=True, dossier=True, extra_tools=toolsets.EXTRA_TOOLS,
                   verify="uncertain", initial="none")


def open_case(toolkit: HolterToolkit, record_id: str, bound: dict, max_candidates: int | None
              ) -> tuple[CaseState, str, list[tuple[dict, str]]]:
    """Create the case state with the overview and candidate summaries as the first evidence."""
    name = toolkit.store.resolve(record_id)
    state = CaseState(name)
    overview = call_tool(toolkit, "record_overview", {"record_id": name}, bound)
    ov_eid = state.add_evidence("record_overview", {"record_id": name}, overview, "setup", TOOLKIT_VERSION)
    listing = jsonable(toolkit.list_candidates(name, bound["protocol"], bound["reference"],
                                               max_items=max_candidates or 10**9))
    cands = []
    for c in listing["candidates"]:
        eid = state.add_evidence("candidate", {"record_id": name, "candidate_id": c["id"]}, c, "setup",
                                 TOOLKIT_VERSION)
        cands.append((c, eid))
    return state, ov_eid, cands


class AgentRunner:
    def __init__(self, toolkit: HolterToolkit, backend: LLMBackend, cfg: RunConfig | None = None,
                 trace_path: str | Path | None = None, baseline_cfg: BaselineConfig | None = None):
        self.toolkit = toolkit
        self.backend = backend
        self.cfg = cfg or RunConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.cfg.verify not in VERIFY:
            raise ValueError(f"verify must be one of {VERIFY}")
        self.baseline_cfg = baseline_cfg or BaselineConfig()
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._images: dict[str, list[bytes]] = {}  # evidence id -> images shown with it
        self._case_images: list[bytes] = []  # image attached to the first message of the current case

    def _system(self, role: str) -> str:
        obs = self.cfg.observation
        return system(role, obs.modality, obs.image_kinds, self.cfg.toolset, self.cfg.guidance)

    # Hooks for subclasses that change the tools or the submission format (RSI/rsi/runner.py).
    def _tool_definitions(self, analysis: bool) -> list[dict]:
        cfg = self.cfg
        return tool_definitions(analysis, cfg.observation, cfg.toolset, cfg.extra_tools, cfg.graded)

    def _offers(self, name: str) -> bool:
        cfg = self.cfg
        return cfg.observation.offers(name) and toolsets.allows(cfg.toolset, name, cfg.extra_tools)

    def _run_tool(self, name: str, args: dict) -> dict:
        return call_tool(self.toolkit, name, args, self.cfg.bound)

    # ------------------------------------------------------------------ public
    def run_record(self, record_id: str, candidate_ids: list[str] | None = None) -> CaseState:
        state, ov_eid, cands = open_case(self.toolkit, record_id, self.cfg.bound, self.cfg.max_candidates)
        overview = state.evidence[0].result
        for cand, c_eid in cands:
            if candidate_ids is not None and cand["id"] not in candidate_ids:
                continue
            self._review_candidate(state, overview, ov_eid, cand, c_eid)
        return state

    # ------------------------------------------------------------------ per candidate
    def _review_candidate(self, state: CaseState, overview: dict, ov_eid: str, cand: dict, c_eid: str) -> None:
        cfg, cid, obs = self.cfg, cand["id"], self.cfg.observation
        ov_seen, cand_seen = visible_case(obs, overview, cand)
        header = case_header(state.record_id, ov_seen, ov_eid, cand_seen, c_eid)
        case_images = []
        if obs.images and "trend" in obs.image_kinds:
            case_images = [case_image(self.toolkit, state.record_id, cand, cfg.reference)]
            header += "\n\nAttached: the ST trend of the candidate and its surroundings."
        self._case_images = case_images
        if cfg.initial == "separate":
            self._session(state, cid, "initial", self._system("first_impression"), initial_task(header, cid),
                          analysis=False, stage="initial", max_turns=2, images=case_images)
        if cfg.mode == "agent":
            prior = None
            if cfg.prior_source == "baseline":
                label, conf, why, eids = decide(
                    cand, collect_standard_evidence(self.toolkit, state, cand, cfg.bound, "prior"), self.baseline_cfg)
                record = {"source": "rule_pipeline", "label": label, "confidence": conf, "rationale": why,
                          "evidence_ids": eids}
                state.add_evidence("prior_label", {"candidate_id": cid}, record, "prior", TOOLKIT_VERSION)
                prior = json.dumps({"ledger_record": record}, ensure_ascii=False, separators=(",", ":"))
            first, images = header, case_images
            if cfg.dossier:
                items, dossier_images = self._dossier(state, cand)
                first, images = header + dossier_block(items), case_images + dossier_images
            self._session(state, cid, "investigate", self._system("judgement"), investigate_task(first, cid, prior),
                          images=images)
        else:
            ev = collect_standard_evidence(self.toolkit, state, cand, cfg.bound, "oneshot_checks")
            items, images = [], list(case_images)
            for tool, (result, eid) in ev.items():
                items.append((eid, tool, visible_result(obs, tool, result)))
                images += self._evidence_images(state, eid)
            self._session(state, cid, "oneshot", self._system("oneshot"), oneshot_task(header, cid, items),
                          analysis=False, max_turns=2, images=images)
        self._verify_loop(state, cid, header)
        final = state.final(cid)
        if cfg.intrinsic_recheck and final is not None:
            self._session(state, cid, "recheck", self._system("recheck"),
                          recheck_task(header, cid, ledger_entry(final, "earlier_review")),
                          analysis=False, max_turns=2, require_new_evidence=False, images=case_images)

    def _dossier(self, state: CaseState, cand: dict) -> tuple[list[tuple[str, str, dict, dict]], list[bytes]]:
        """Standard checks run before the judgement session (harness 0.8): the fixed check set of the rule
        baseline plus the run's extra tools, stored in the ledger under session "dossier"."""
        cfg, obs = self.cfg, self.cfg.observation
        ev = collect_standard_evidence(self.toolkit, state, cand, cfg.bound, "dossier")
        for tool in cfg.extra_tools:
            args = {"record_id": state.record_id, "candidate_id": cand["id"]}
            try:
                result = self._run_tool(tool, args)
            except ToolError:
                continue
            ev[tool] = (result, state.add_evidence(tool, args, result, "dossier", TOOLKIT_VERSION))
        items, images = [], []
        for tool, (result, eid) in ev.items():
            e = state.evidence[int(eid[1:]) - 1]
            if not obs.offers(tool):
                continue
            items.append((eid, tool, e.args, visible_result(obs, tool, toolsets.shown_result(cfg.toolset, tool, result))))
            if tool in ("measure_window", "compare_morphology"):
                images += self._evidence_images(state, eid)
        return items, images

    def _needs_verification(self, current) -> bool:
        cfg = self.cfg
        if cfg.verify == "none":
            return False
        if cfg.verify == "ischemic":
            return current.label == "ischemic"
        if cfg.verify == "uncertain":
            p = current.p_ischemic
            return p is None or current.label == "indeterminate_data" or cfg.verify_band[0] <= p <= cfg.verify_band[1]
        return True

    def _verify_loop(self, state: CaseState, cid: str, header: str) -> None:
        """Verify the current label until a round confirms it; replaced labels are verified again."""
        cfg = self.cfg
        for k in range(1, cfg.max_verify_rounds + 1):
            current = state.final(cid)
            if current is None or not (self._needs_verification(current) or (k > 1 and cfg.verify == "uncertain")):
                return
            n_finals = len(state.history(cid, "final"))
            obs = cfg.observation
            cited_ev = [e for e in state.evidence
                        if e.id in current.evidence_ids and e.tool not in ("record_overview", "candidate")]
            cited = [(e.id, e.tool, e.args, visible_result(obs, e.tool, e.result)) for e in cited_ev]
            images = list(self._case_images) + [p for e in cited_ev for p in self._evidence_images(state, e.id)]
            self._session(state, cid, f"verify{k}", self._system("verification"),
                          verify_task(header, cid, ledger_entry(current, "earlier_review"), cited), images=images)
            if len(state.history(cid, "final")) == n_finals:
                state.needs_review.setdefault(cid, f"verification round {k} ended without an assessment")
                return
            if state.final(cid).label == current.label or cid in state.needs_review:
                return  # confirmed, or oscillation already flagged by the state
        state.needs_review.setdefault(
            cid, f"label still changing after {cfg.max_verify_rounds} verification rounds")

    # ------------------------------------------------------------------ one session
    def _session(self, state: CaseState, cid: str, session: str, system: str, first_message: str,
                 analysis: bool = True, stage: str = "final", max_turns: int | None = None,
                 require_new_evidence: bool = True, images: list[bytes] | None = None) -> str:
        cfg = self.cfg
        tools = self._tool_definitions(analysis)
        messages: list[dict] = [{"role": "user", "content": user_content(first_message, images)}]
        self._trace({"step": state.step, "candidate": cid, "session": session,
                     "prompt": {"system": system, "first_message": first_message, "n_images": len(images or []),
                                "tools": [t["name"] for t in tools]}})
        n_tools = stall = 0
        submitted = nudged = False
        stop = "max_turns"
        turns = max_turns or cfg.max_turns
        for k in range(turns):
            state.step += 1
            t0 = time.time()
            turn = self.backend.step(system, messages, tools)
            messages.append({"role": "assistant", "content": turn.content})
            trace = {"step": state.step, "candidate": cid, "session": session, "stop_reason": turn.stop_reason,
                     "text": turn.text, "calls": [asdict(c) for c in turn.tool_calls], "usage": turn.usage,
                     "seconds": round(time.time() - t0, 2)}
            if turn.reasoning:
                trace["reasoning"] = turn.reasoning
            if turn.stop_reason == "refusal":
                stop = "refusal"
                self._trace(trace)
                break
            if not turn.tool_calls:
                self._trace(trace)
                if not submitted and not nudged:
                    messages.append({"role": "user", "content": NUDGE_FINAL if stage == "final"
                                     else "Call submit_assessment with stage 'initial'."})
                    nudged = True
                    continue
                stop = "completed" if submitted else "no_assessment"
                break
            results, new_evidence, sub_now = [], 0, False
            for call in turn.tool_calls:
                if call.name == SUBMIT_TOOL["name"]:
                    block, ok = self._submit(state, call, cid, session, stage, require_new_evidence)
                    sub_now |= ok
                elif not analysis:
                    block = tool_result_block(call.id, "No analysis tools are available in this step.", is_error=True)
                elif not self._offers(call.name):
                    block = tool_result_block(call.id, f"{call.name} is not available in this run.", is_error=True)
                elif n_tools >= cfg.max_tool_calls:
                    block = tool_result_block(call.id, "Tool budget exhausted. Submit your final assessment now.",
                                              is_error=True)
                else:
                    n_tools += 1
                    block, is_new = self._analysis(state, call, session)
                    new_evidence += is_new
                results.append(block)
            submitted |= sub_now
            if sub_now and not analysis:  # a submission-only step: done once the assessment is recorded
                trace["results"] = _without_images(results)
                self._trace(trace)
                stop = "completed"
                break
            stall = 0 if (new_evidence or sub_now) else stall + 1
            content: list[dict] = list(results)
            if analysis and stall == cfg.stall_turns - 1:
                content.append({"type": "text", "text": STALL_NOTE})
            if analysis and stage == "final" and not submitted and k == turns - 2:
                content.append({"type": "text", "text": LAST_TURN_NOTE})
            trace["results"] = _without_images(results)
            self._trace(trace)
            messages.append({"role": "user", "content": content})
            if stall >= cfg.stall_turns:
                if analysis and stage == "final" and not submitted and not nudged:
                    # Stalled without a final assessment: one last turn to submit one.
                    content.append({"type": "text", "text": LAST_TURN_NOTE})
                    nudged = True
                    stall = cfg.stall_turns - 1
                    continue
                stop = "stalled"
                break
        state.set_stop(cid, session, stop)
        # the conversation as it ended, for a read-out after the session (RSI runner: logit score of open models)
        self.last_session = {"session": session, "cid": cid, "system": system, "messages": messages, "tools": tools}
        return stop

    def _submit(self, state: CaseState, call: ToolCall, cid: str, session: str, stage: str,
                require_new_evidence: bool) -> tuple[dict, bool]:
        args = dict(call.input)
        if args.get("candidate_id") == cid.split("-", 1)[-1]:  # the record part may be left out
            args["candidate_id"] = cid
        if args.get("candidate_id") != cid:
            return tool_result_block(call.id, f"This session reviews {cid} only.", is_error=True), False
        if args.get("stage") != stage:
            return tool_result_block(call.id, f"This step accepts stage {stage!r} only.", is_error=True), False
        schema = SUBMIT_TOOL_GRADED if self.cfg.graded else SUBMIT_TOOL
        missing = [k for k in schema["input_schema"]["required"] if k not in args]
        if missing:
            return tool_result_block(call.id, f"Rejected: missing required arguments {missing}", is_error=True), False
        try:
            if self.cfg.graded:
                p = float(args["p_ischemic"])
                table = {"for_ischemia": list(args.get("findings_for_ischemia") or []),
                         "against_ischemia": list(args.get("findings_against_ischemia") or [])}
                eids = list(dict.fromkeys(str(f.get("evidence_id", "")) for side in table.values() for f in side
                                          if isinstance(f, dict) and f.get("evidence_id")))
                conf = p if args["label"] == "ischemic" else (1.0 - p if args["label"] != "indeterminate_data" else 0.5)
                a = state.submit(cid, stage, args["label"], conf, args.get("rationale", ""), eids, session=session,
                                 require_new_evidence=require_new_evidence, p_ischemic=p, evidence_table=table)
            else:
                a = state.submit(cid, stage, args["label"], float(args["confidence"]), args.get("rationale", ""),
                                 list(args.get("evidence_ids", [])), session=session,
                                 require_new_evidence=require_new_evidence)
        except (StateError, KeyError, TypeError, ValueError) as e:
            return tool_result_block(call.id, f"Rejected: {e}", is_error=True), False
        return tool_result_block(call.id, {"recorded": True, "stage": a.stage, "revision": a.revision}), True

    def _analysis(self, state: CaseState, call: ToolCall, session: str) -> tuple[dict, bool]:
        args = dict(call.input)
        rid = args.get("record_id")
        if rid is not None:
            try:
                if self.toolkit.store.resolve(str(rid)) != state.record_id:
                    return tool_result_block(call.id, f"Only record {state.record_id} is in scope.", is_error=True), False
            except KeyError as e:
                return tool_result_block(call.id, str(e), is_error=True), False
        state.tool_calls += 1
        obs = self.cfg.observation
        earlier = state.find_evidence(call.name, args)
        if earlier is not None:
            payload = {"evidence_id": earlier.id, "duplicate": True,
                       "result": visible_result(obs, call.name, earlier.result)}
            if self._images.get(earlier.id):
                payload["image"] = f"as shown with {earlier.id}"
            return tool_result_block(call.id, payload), False
        try:
            result = toolsets.shown_result(self.cfg.toolset, call.name, self._run_tool(call.name, args))
        except ToolError as e:
            return tool_result_block(call.id, f"Tool error: {e}", is_error=True), False
        images = []
        if obs.image_for(call.name):
            try:
                images = [tool_image(self.toolkit, call.name, args, self.cfg.bound)]
            except (ToolError, ValueError, KeyError) as e:
                result = dict(result) | {"image_error": str(e)}
        eid = state.add_evidence(call.name, args, result, session, TOOLKIT_VERSION)
        self._store_images(state, eid, images)
        return tool_result_block(call.id, {"evidence_id": eid, "result": visible_result(obs, call.name, result)},
                                 images=images), True

    def _store_images(self, state: CaseState, eid: str, images: list[bytes]) -> None:
        if not images:
            return
        self._images[eid] = images
        if self.trace_path is not None:
            folder = self.trace_path.parent / "images"
            folder.mkdir(parents=True, exist_ok=True)
            names = []
            for k, png in enumerate(images):
                prefix = self.trace_path.name.removesuffix(".trace.jsonl")  # unique per output file
                name = f"{prefix}_{eid}_{k}.png"
                (folder / name).write_bytes(png)
                names.append(f"images/{name}")
            state.evidence[int(eid[1:]) - 1].images = names

    def _evidence_images(self, state: CaseState, eid: str) -> list[bytes]:
        """Images of an evidence item, rendered on demand for items collected outside a session."""
        if eid not in self._images:
            e = state.evidence[int(eid[1:]) - 1]
            if self.cfg.observation.image_for(e.tool):
                try:
                    self._store_images(state, eid, [tool_image(self.toolkit, e.tool, e.args, self.cfg.bound)])
                except (ToolError, ValueError, KeyError):
                    self._images[eid] = []
        return self._images.get(eid, [])

    def _trace(self, record: dict) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")


def _without_images(obj):
    """Copy of message content with image data replaced by a marker (for traces)."""
    if isinstance(obj, dict):
        if obj.get("type") == "image":
            return {"type": "image", "omitted": True}
        return {k: _without_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_without_images(v) for v in obj]
    return obj


def _jsonable(obj):
    """Serialise SDK content blocks and numpy scalars in traces."""
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
