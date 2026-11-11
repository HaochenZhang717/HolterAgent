"""Run a harness: the harness's prompts, tools, dossier and submission format on top of the project's runners.

Two backends for the inner model:
- PoolHarnessRunner (from 2026-10-06): AgentRunner's own session loop with PoolBackend, i.e. direct HTTP requests to
  a shared OpenAI Responses-API endpoint (holter_agent/agent/pool.py);
- HarnessRunner: CodexAgentRunner, i.e. one `codex exec` per session (rounds 0-5 of the search).
Everything else (the sessions and their order, the evidence ledger, the revision gate, the verification loop) is the
code of holter_agent.agent.runner, unchanged. Results of the two backends are not directly comparable.

Harnesses with sessions.verify = "independent" (from 2026-10-14, harness v0.9) replace the verification loop by the
second-reader design of ST-database annotation (European ST-T and Long-Term ST databases: two independent readers, a
third resolves disagreements): when a trigger fires (reference disagreement, measurement-point effect, signal quality
or rhythm, or an uncertain first probability), a second judgement session reads the case blind to the first one, from
the same first message and dossier; agreeing readings (same side of p = 0.5) are averaged, disagreeing ones go to an
adjudication session that sees both assessments and the evidence they cite.

Harnesses with a "knowledge" setting (from 2026-10-14, harness v1.0) show the knowledge base built beforehand
(rsi/knowledge.py: literature cards and labelled precedents of other patients) in every verification session: the
retrieved items are appended to the first message as evidence items that can be cited, and the knowledge_cards tool
returns cards on further topics. Precedents need the patients of the evaluated split (kb_exclude_patients).
"""
from __future__ import annotations

import json

from holter_agent.agent.baseline import collect_standard_evidence
from holter_agent.agent.codex_runner import CodexAgentRunner, CodexConfig
from holter_agent.agent.runner import AgentRunner
from holter_agent.agent.llm import ToolCall, tool_result_block
from holter_agent.agent.observe import visible_result
from holter_agent.agent.prompts import _j, ledger_entry
from holter_agent.agent.state import CaseState, StateError
from holter_agent.agent.toolkit import TOOLKIT_VERSION, ToolError, call_tool
from holter_agent.agent import toolsets

from holter_agent.agent.estimate import features as case_features

from .harness import Harness
from .knowledge import CARD_TOOL, KnowledgeBase

KB_VERSION = "kb-2026-10-14"

IMAGE_DOSSIER_TOOLS = ("measure_window", "compare_morphology")  # dossier items whose images are attached


class _HarnessMixin:
    """The harness-specific parts; mixed into a concrete runner."""
    harness: Harness

    def _system(self, role: str) -> str:
        return self.harness.system(role)

    def _tool_definitions(self, analysis: bool) -> list[dict]:
        tools = self.harness.tool_definitions(analysis)
        if analysis and self._kb_active and self.harness.knowledge["cards"]:
            tools = tools[:-1] + [CARD_TOOL, tools[-1]]  # submit_assessment stays last
        return tools

    def _offers(self, name: str) -> bool:
        if name == CARD_TOOL["name"]:
            return self._kb_active and bool(self.harness.knowledge["cards"])
        return name in self.harness.tool_names()

    def _run_tool(self, name: str, args: dict) -> dict:
        if name == CARD_TOOL["name"]:
            return self._card_lookup(args)
        if self.harness.is_ext(name):
            return self.harness.run_ext(self.toolkit, name, args, self.cfg.bound)
        return call_tool(self.toolkit, name, args, self.cfg.bound)

    def _dossier(self, state: CaseState, cand: dict):
        """Checks run before the judgement session, as listed in the harness ("standard" is the fixed check set
        of the rule baseline), stored in the ledger under session "dossier"."""
        cfg, obs = self.cfg, self.cfg.observation
        ev: dict = {}
        for item in self.harness.sessions["dossier"]:
            if item == "standard":
                drop = set(self.harness.sessions.get("dossier_exclude") or [])  # left out of the first message
                ev.update({k: v for k, v in collect_standard_evidence(self.toolkit, state, cand, cfg.bound,
                                                                      "dossier").items() if k not in drop})
                continue
            args = {"record_id": state.record_id, "candidate_id": cand["id"]}
            try:
                result = self._run_tool(item, args)
            except ToolError:
                continue
            ev[item] = (result, state.add_evidence(item, args, result, "dossier", TOOLKIT_VERSION))
        self._dossier_results = {tool: result for tool, (result, _) in ev.items()}
        items, images = [], []
        for tool, (result, eid) in ev.items():
            e = state.evidence[int(eid[1:]) - 1]
            items.append((eid, tool, e.args, visible_result(obs, tool, toolsets.shown_result(cfg.toolset, tool, result))))
            if tool in IMAGE_DOSSIER_TOOLS:
                images += self._evidence_images(state, eid)
        if "record_st_overview" in ev and obs.images:  # the whole-record ST trend, attached after the other images
            from holter_agent.agent.observe import trend_image
            dur = self.toolkit._state(state.record_id).record.duration_s
            images.append(trend_image(self.toolkit, state.record_id, 0.0, dur, None, cfg.reference,
                                      spans=[(cand["start_s"], cand["end_s"])], title="whole record"))
        return items, images

    # ------------------------------------------------------------- independent second reading (verify = independent)
    # Trigger thresholds are engineering defaults.
    MIN_USABLE_FRAC = 0.75
    MAX_RMSSD_OVER_RR = 0.1
    MAX_FRAC_RR_DIFF_GT_50MS = 0.2
    MAX_NON_NORMAL_FRAC = 0.1

    # Logit read-out (from 2026-10-17, open-weight task models): after the judgement session, the conversation as it
    # ended plus a fixed yes/no question, scored from the next-token log-probabilities (OpenAICompatBackend.yes_no).
    # run_case.py --logit sets logit_scores = {} and uses the read-out as the case's score.
    logit_scores: dict | None = None
    LOGIT_QUESTION = ("Based on everything above, is the ST change of candidate {cid} caused by myocardial ischemia? "
                      "Answer with one word: yes or no.")

    def _session(self, state: CaseState, cid: str, session: str, system: str, first_message: str, **kw):
        stop = self._session_kb(state, cid, session, system, first_message, **kw)
        if self.logit_scores is not None and session == "investigate":
            last = getattr(self, "last_session", None)
            if last and last["cid"] == cid:
                r = self.backend.yes_no(last["system"], last["messages"], last["tools"],
                                        self.LOGIT_QUESTION.format(cid=cid))
                self.logit_scores[cid] = r
                self._trace({"step": state.step, "candidate": cid, "session": "logit_readout", "readout": r})
        return stop

    def _session_kb(self, state: CaseState, cid: str, session: str, system: str, first_message: str, **kw):
        if session == "investigate":  # kept for a blind second reading of the same case
            self._first_inputs = (system, first_message, kw.get("images"))
        kb_sessions = (self.harness.knowledge or {}).get("sessions", ["verify"])
        if self.harness.knowledge and any(session.startswith(k) for k in kb_sessions):
            self._kb_shown = set()
            block = self._knowledge_block(state, cid, session)
            # before the closing request of verify_task / investigate_task (holter_agent/agent/prompts.py)
            i = max(first_message.rfind(f"\n\nTest this label for {cid}"), first_message.rfind(f"\n\nInvestigate {cid}"))
            first_message = first_message[:i] + block + first_message[i:] if i >= 0 else first_message + block
            self._kb_active = True
            try:
                return super()._session(state, cid, session, system, first_message, **kw)
            finally:
                self._kb_active = False
        return super()._session(state, cid, session, system, first_message, **kw)

    # ------------------------------------------------------------- knowledge base (harness key "knowledge")
    kb_exclude_patients: set[str] | None = None  # the patients of the evaluated split (run_case.py --kb-exclude-split)
    _kb_active = False
    _kb_obj = None

    def _kb(self) -> KnowledgeBase:
        if self._kb_obj is None:
            kb = self.harness.knowledge
            if kb["precedents"] and not self.kb_exclude_patients:
                raise RuntimeError("precedents need the patients of the evaluated split (run_case.py --kb-exclude-split)")
            local = self.harness.path / "knowledge" / "cards.json"  # cards added by this harness (cloop extended scope)
            self._kb_obj = KnowledgeBase(set(self.kb_exclude_patients or ()), cards=kb["cards"],
                                         precedents=kb["precedents"],
                                         extra=json.loads(local.read_text()) if local.exists() else None)
        return self._kb_obj

    def _case_features(self, state: CaseState, cid: str) -> dict:
        """The tool features of the fixed check set (as in the precedent bank), from this case's dossier."""
        c = self.toolkit.candidate(state.record_id, cid)
        geom = {"peak_uv": c.peak.peak_uv, "start_s": c.start_s, "end_s": c.end_s}
        return case_features(getattr(self, "_dossier_results", {}), geom)

    def _kb_item(self, state: CaseState, tool: str, args: dict, make, session: str) -> str:
        """One knowledge item as an evidence entry; a later verification round shows the same entry again."""
        e = state.find_evidence(tool, args)
        eid, result = (e.id, e.result) if e is not None else (None, make())
        if eid is None:
            eid = state.add_evidence(tool, args, result, session, KB_VERSION)
        if tool == CARD_TOOL["name"]:
            self._kb_shown |= {c["id"] for c in result.get("cards", [])}
        return f"{eid} {tool} {_j(args)}: {_j(result)}"

    def _knowledge_block(self, state: CaseState, cid: str, session: str) -> str:
        cfg, kb = self.harness.knowledge, self._kb()
        feats = self._case_features(state, cid)
        lines = ["", "", "Knowledge base for this review (built beforehand, not from this record; cite its items by "
                         "evidence id like any other evidence):"]
        if cfg["precedents"]:
            lines.append(self._kb_item(state, "knowledge_precedents", {"candidate_id": cid}, lambda: kb.precedents(
                feats, state.record_id, int(cfg["k_per_class"]), int(cfg["n_overall"])), session))
        if cfg["cards"]:
            topics = kb.topics_for(feats, getattr(self, "_dossier_results", {}), cfg.get("topic_rules"),
                                   cfg.get("always_topics"))
            lines.append(self._kb_item(state, CARD_TOOL["name"], {"topics": topics, "retrieved_for": cid}, lambda: {
                "cards": kb.cards(topics, limit=int(cfg["max_auto_cards"])),
                "cards_per_topic": kb.index()}, session))
        return "\n".join(lines)

    def _card_lookup(self, args: dict) -> dict:
        kb = self._kb()
        topics = [str(t) for t in (args.get("topics") or [])]
        bad = [t for t in topics if t not in kb.topics_text]
        if bad or not topics:
            raise ToolError(f"unknown topics {bad}; use {sorted(kb.topics_text)}")
        cards = kb.cards(topics, skip=self._kb_shown)
        more = max(0, len(cards) - kb.LOOKUP_LIMIT)
        cards = cards[:kb.LOOKUP_LIMIT]
        self._kb_shown |= {c["id"] for c in cards}
        if not cards:
            return {"cards": [], "note": "no further cards on these topics"}
        return {"cards": cards} | ({"note": f"{more} more cards on these topics; call again for the next ones"}
                                   if more else {})

    def _verify_loop(self, state: CaseState, cid: str, header: str) -> None:
        if self.harness.sessions["verify"] != "independent":
            return super()._verify_loop(state, cid, header)
        first = state.final(cid)
        if first is None or getattr(self, "_first_inputs", None) is None:
            return
        fired = self._triggers(state, first)
        state.add_evidence("verification_triggers", {"candidate_id": cid}, {"fired": fired}, "verification",
                           TOOLKIT_VERSION)
        if not fired:
            return
        system, first_message, images = self._first_inputs
        n0 = len(state.history(cid, "final"))
        self._session(state, cid, "reader2", system, first_message, images=images, require_new_evidence=False)
        if len(state.history(cid, "final")) == n0:
            state.needs_review.setdefault(cid, "second reading ended without an assessment")
            return
        second = state.final(cid)
        p1, p2 = first.p_ischemic, second.p_ischemic
        if p1 is None or p2 is None or (p1 >= 0.5) != (p2 >= 0.5):
            self._adjudicate(state, cid, header, first, second)
        else:
            self._combine(state, cid, first, second, "agreeing independent readings; p_ischemic is their mean")

    def _triggers(self, state: CaseState, first) -> list[str]:
        want = set(self.harness.sessions["verify_triggers"])
        res = {e.tool: e.result for e in state.evidence if e.session == "dossier"}
        fired = []
        rl = res.get("reference_levels") or {}
        if "reference_disagreement" in want and rl.get("record_local_tracked_agree") is False:
            fired.append("reference_disagreement")
        if "measurement_point" in want and (res.get("measurement_point_check") or {}).get("measurement_point_effect"):
            fired.append("measurement_point")
        if "quality_rhythm" in want and self._quality_or_rhythm(res, set(rl.get("leads") or {})):
            fired.append("quality_rhythm")
        lo, hi = self.harness.sessions["verify_band"]
        p = first.p_ischemic
        if "uncertain_p" in want and (p is None or first.label == "indeterminate_data" or lo <= p <= hi):
            fired.append("uncertain_p")
        return fired

    def _quality_or_rhythm(self, res: dict, leads: set[str]) -> bool:
        q = (res.get("signal_quality") or {}).get("leads") or {}
        fracs = [v.get("usable_frac") for k, v in q.items() if (not leads or k in leads) and isinstance(v, dict)]
        if any(f is not None and f < self.MIN_USABLE_FRAC for f in fracs):
            return True
        rc = res.get("rhythm_context") or {}
        irr = rc.get("rr_irregularity") or {}
        if (irr.get("rmssd_over_mean_rr") or 0) >= self.MAX_RMSSD_OVER_RR or \
                (irr.get("frac_successive_diff_gt_50ms") or 0) >= self.MAX_FRAC_RR_DIFF_GT_50MS:
            return True
        beats = rc.get("beat_types") or {}
        n = sum(beats.values())
        if n and 1 - beats.get("N", 0) / n >= self.MAX_NON_NORMAL_FRAC:
            return True
        return bool(((res.get("eligibility_check") or {}).get("flags") or {}).get("qrs_ge_100ms"))

    def _combine(self, state: CaseState, cid: str, a, b, why: str, session: str = "combined") -> None:
        p = (a.p_ischemic + b.p_ischemic) / 2
        if p >= 0.5:
            label = "ischemic"
        else:  # the non-ischemic cause of the reader further from 0.5
            label = (a if a.p_ischemic <= b.p_ischemic else b).label
            if label == "ischemic":
                label = "other_nonischemic"
        conf = p if label == "ischemic" else 1.0 - p
        eids = list(dict.fromkeys(list(a.evidence_ids) + list(b.evidence_ids)))
        state.submit(cid, "final", label, conf, why, eids, session=session, require_new_evidence=False, p_ischemic=p)
        if str(state.needs_review.get(cid, "")).startswith("final label oscillated"):
            state.needs_review.pop(cid)  # returning to the first reader's label is expected here

    def _adjudicate(self, state: CaseState, cid: str, header: str, first, second) -> None:
        obs = self.cfg.observation
        ids = list(dict.fromkeys(list(first.evidence_ids) + list(second.evidence_ids)))
        cited_ev = [e for e in state.evidence if e.id in ids and e.tool not in ("record_overview", "candidate")]
        cited = [f"{e.id} {e.tool} {_j(e.args)}: {_j(visible_result(obs, e.tool, e.result))}" for e in cited_ev]
        text = (f"{header}\n\nTwo independent readers disagree about this candidate. Their assessments:\n"
                f"{ledger_entry(first, 'reader_1')}\n{ledger_entry(second, 'reader_2')}")
        if cited:
            text += "\n\nEvidence they cite:\n" + "\n".join(cited)
        text += f"\n\nAdjudicate {cid} and submit a final assessment."
        images = list(self._case_images) + [p for e in cited_ev for p in self._evidence_images(state, e.id)]
        n0 = len(state.history(cid, "final"))
        self._session(state, cid, "adjudicate", self._system("adjudication"), text, images=images,
                      require_new_evidence=False)
        if len(state.history(cid, "final")) == n0:  # no verdict: fall back to the mean, flagged for a human
            self._combine(state, cid, first, second, "adjudication ended without an assessment; mean of the two "
                                                     "readings", session="combined_fallback")
            state.needs_review.setdefault(cid, "adjudication ended without an assessment")
        elif str(state.needs_review.get(cid, "")).startswith("final label oscillated"):
            state.needs_review.pop(cid)

    def _submit(self, state: CaseState, call: ToolCall, cid: str, session: str, stage: str,
                require_new_evidence: bool) -> tuple[dict, bool]:
        """A two-way decision (ischemic / non_ischemic; recorded as label "ischemic" or "other_nonischemic" with
        p_ischemic 1 or 0), or any format that gives p_ischemic and a label; evidence ids may come as a list
        (evidence_ids) and/or inside findings for and against ischaemia."""
        args = dict(call.input)
        if args.get("candidate_id") == cid.split("-", 1)[-1]:  # the record part may be left out
            args["candidate_id"] = cid
        if args.get("candidate_id") != cid:
            return tool_result_block(call.id, f"This session reviews {cid} only.", is_error=True), False
        if args.get("stage") != stage:
            return tool_result_block(call.id, f"This step accepts stage {stage!r} only.", is_error=True), False
        need = int(self.harness.limits.get("min_tool_calls", 0))
        if stage == "final" and need:
            kb_items = {"knowledge_precedents", CARD_TOOL["name"]}  # shown by code, not called by the model
            done = sum(1 for e in state.evidence if e.session == session and e.tool not in kb_items)
            if done < need:
                return tool_result_block(call.id, (
                    f"Not recorded yet: this protocol needs at least {need} analysis tool calls in this session before "
                    f"the final assessment (so far {done}). The dossier is a starting point. Call the tools that test the "
                    "two most plausible explanations against each other (results come back in the next message), then "
                    "submit."), is_error=True), False
        schema = self.harness.submit_tool["input_schema"]
        missing = [k for k in schema["required"] if k not in args]
        if missing:
            return tool_result_block(call.id, f"Rejected: missing required arguments {missing}", is_error=True), False
        try:
            if self.harness.binary:
                decision = args["decision"]
                if decision not in ("ischemic", "non_ischemic"):
                    raise StateError("decision must be 'ischemic' or 'non_ischemic'")
                label, p = ("ischemic", 1.0) if decision == "ischemic" else ("other_nonischemic", 0.0)
                cause = args.get("label")  # optional most likely explanation, kept when it agrees with the decision
                enum = (schema["properties"].get("label") or {}).get("enum", [])
                if cause in enum and (cause == "ischemic") == (decision == "ischemic") and cause != "indeterminate_data":
                    label = cause
            else:
                label = args["label"]
                if label not in schema["properties"]["label"]["enum"]:
                    raise StateError(f"label must be one of {schema['properties']['label']['enum']}")
                p = float(args["p_ischemic"])
            eids = [str(e) for e in (args.get("evidence_ids") or [])]
            table = None
            if "findings_for_ischemia" in args or "findings_against_ischemia" in args:
                table = {"for_ischemia": list(args.get("findings_for_ischemia") or []),
                         "against_ischemia": list(args.get("findings_against_ischemia") or [])}
                eids += [str(f.get("evidence_id", "")) for side in table.values() for f in side
                         if isinstance(f, dict) and f.get("evidence_id")]
            eids = list(dict.fromkeys(eids))
            conf = p if label == "ischemic" else (1.0 - p if label != "indeterminate_data" else 0.5)
            a = state.submit(cid, stage, label, conf, args.get("rationale", ""), eids, session=session,
                             require_new_evidence=require_new_evidence, p_ischemic=p, evidence_table=table)
        except (StateError, KeyError, TypeError, ValueError) as e:
            return tool_result_block(call.id, f"Rejected: {e}", is_error=True), False
        return tool_result_block(call.id, {"recorded": True, "stage": a.stage, "revision": a.revision}), True


class HarnessRunner(_HarnessMixin, CodexAgentRunner):
    """Inner model through the Codex CLI."""

    def __init__(self, toolkit, harness: Harness, codex: CodexConfig | None = None, trace_path=None):
        CodexAgentRunner.__init__(self, toolkit, harness.run_config(), codex, trace_path)
        self.harness = harness


class PoolHarnessRunner(_HarnessMixin, AgentRunner):
    """Inner model through the shared pool (direct HTTP)."""

    def __init__(self, toolkit, harness: Harness, backend, trace_path=None):
        AgentRunner.__init__(self, toolkit, backend, harness.run_config(), trace_path)
        self.harness = harness
