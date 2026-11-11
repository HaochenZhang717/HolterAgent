"""A harness as a directory: prompt parts, session settings, tools and the submission format.

    harnesses/<id>/
      harness.json   settings (below)
      prompts/*.md   text parts; the system prompt of a role is the concatenation of its parts, in order
      submit.json    the submit_assessment tool; must ask for candidate_id and stage, and either for decision (a two-way
                     choice, "ischemic" or "non_ischemic"; from 2026-10-08) or for p_ischemic and label (a probability,
                     the format of the runs up to dgmh4)
      tools/<name>.json + tools/<name>.py   optional: deterministic tools of the harness's own. The JSON (name,
                     description, input_schema) is read here; the code only ever runs in the sandboxed worker
                     (rsi/ext_worker.py), which sees nothing but the feature cache (rsi/ext_context.py)

harness.json:
    id, parent, summary
    system     {role: [part files]}; roles: judgement (always), verification (when sessions.verify is not
               "none"), first_impression (when sessions.first_impression is true)
    sessions   first_impression (bool); dossier (list of "standard", the fixed check set of the rule baseline,
               and/or tools that take only record_id and candidate_id); verify (none, all, ischemic,
               uncertain, independent); verify_band [lo, hi] (for "uncertain", and the uncertain_p trigger of
               "independent"); max_verify_rounds; verify_triggers (for "independent", from 2026-10-14: an
               independent second reading, blind to the first, when any listed trigger fires —
               reference_disagreement, measurement_point, quality_rhythm, uncertain_p; agreeing readings are
               averaged, disagreeing ones go to an adjudicator; needs the role "adjudication")
    tools      toolset (base or ecg); extra (gated tools: episode_context, rate_response); descriptions
               ({tool: replacement description})
    limits     max_tool_calls, stall_turns
    knowledge  optional (from 2026-10-14, harness v1.0): the knowledge base built beforehand (RSI/kb, rsi/knowledge.py)
               shown in every verification session: {cards: bool, precedents: bool, k_per_class, n_overall,
               max_auto_cards}; needs sessions.verify "all", "ischemic" or "uncertain". Precedents come from patients
               outside the evaluated split only (run_case.py --kb-exclude-split)

Fixed for every harness and not settable here: the inner model and its effort, the observation (numbers and
images, all four image kinds), the candidate protocol (B) and the ST reference (global). Clinical information
(patient_context) and the estimator fitted on LTSTDB labels (ischemia_estimate) are never offered.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from holter_agent.agent import toolsets
from holter_agent.agent.observe import IMAGE_KINDS, ObservationConfig
from holter_agent.agent.runner import RunConfig
from holter_agent.agent.state import LABELS
from holter_agent.agent.toolkit import TOOL_SPECS, TOOLS_BY_NAME, ToolError, jsonable

from .paths import HARNESS_DIR, candidate_protocol

OBSERVATION = ObservationConfig("both", IMAGE_KINDS)
PROTOCOL, REFERENCE = candidate_protocol(), "global"  # B unless CANDIDATE_PROTOCOL (rsi/paths.py)
FORBIDDEN_TOOLS = ("patient_context", "ischemia_estimate")
TOOLSETS = ("base", "ecg")
VERIFY = ("none", "all", "ischemic", "uncertain", "independent")
VERIFY_TRIGGERS = ("reference_disagreement", "measurement_point", "quality_rhythm", "uncertain_p")
KEYS = {"id", "parent", "summary", "system", "sessions", "tools", "limits", "knowledge"}
SESSION_KEYS = {"first_impression", "dossier", "dossier_exclude", "verify", "verify_band", "max_verify_rounds",
                "verify_triggers"}
# checks of the "standard" dossier set (holter_agent/agent/baseline.py) that sessions.dossier_exclude may leave out
STANDARD_CHECKS = ("signal_quality", "rhythm_context", "measure_window", "find_reference_window", "compare_morphology",
                   "morphology_steps", "episode_dynamics", "check_criteria", "eligibility_check")
TOOL_KEYS = {"toolset", "extra", "descriptions"}
LIMIT_KEYS = {"max_tool_calls", "stall_turns", "min_tool_calls"}  # min_tool_calls: analysis calls before a final assessment is accepted (2026-10-17)
KNOWLEDGE_KEYS = {"cards", "precedents", "k_per_class", "n_overall", "max_auto_cards", "sessions", "topic_rules",
                  "always_topics"}  # sessions: "investigate" and/or "verify" (default ["verify"])
SUBMIT_REQUIRED = {"candidate_id", "stage", "p_ischemic", "label"}  # probability format
SUBMIT_REQUIRED_BINARY = {"candidate_id", "stage", "decision"}       # two-way decision format
DECISIONS = ("ischemic", "non_ischemic")


class HarnessError(ValueError):
    pass


def resolve(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    return p if p.is_dir() else HARNESS_DIR / str(name_or_path)


def digest(path: Path) -> str:
    """Content hash of a harness directory (file names and bytes; caches excluded)."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(path)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


@dataclass
class Harness:
    path: Path
    spec: dict
    parts: dict[str, str]
    submit_tool: dict
    ext_specs: list[dict]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, name_or_path: str | Path) -> "Harness":
        path = resolve(name_or_path)
        if not (path / "harness.json").exists():
            raise HarnessError(f"no harness.json in {path}")
        spec = json.loads((path / "harness.json").read_text())
        parts = {f.name: f.read_text() for f in sorted((path / "prompts").glob("*.md"))}
        submit = json.loads((path / "submit.json").read_text())
        if (path / "ext_tools.py").exists():
            raise HarnessError("ext_tools.py is not supported: give each tool tools/<name>.json and tools/<name>.py")
        ext_specs = [json.loads(f.read_text()) for f in sorted((path / "tools").glob("*.json"))] \
            if (path / "tools").is_dir() else []
        h = cls(path, spec, parts, submit, ext_specs)
        h.validate()
        return h

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def sessions(self) -> dict:
        return {"first_impression": False, "dossier": [], "verify": "none", "verify_band": [0.2, 0.8],
                "max_verify_rounds": 3, "verify_triggers": list(VERIFY_TRIGGERS), "dossier_exclude": [],
                **self.spec.get("sessions", {})}

    @property
    def tools(self) -> dict:
        return {"toolset": "base", "extra": [], "descriptions": {}, **self.spec.get("tools", {})}

    @property
    def limits(self) -> dict:
        return {"max_tool_calls": 30, "stall_turns": 3, "min_tool_calls": 0, **self.spec.get("limits", {})}

    @property
    def knowledge(self) -> dict | None:
        if "knowledge" not in self.spec:
            return None
        return {"cards": True, "precedents": True, "k_per_class": 3, "n_overall": 10, "max_auto_cards": 10,
                **self.spec["knowledge"]}

    def validate(self) -> None:
        s = self.spec
        unknown = (set(s) - KEYS) | (set(s.get("sessions", {})) - SESSION_KEYS) | (
            set(s.get("tools", {})) - TOOL_KEYS) | (set(s.get("limits", {})) - LIMIT_KEYS) | (
            set(s.get("knowledge") or {}) - KNOWLEDGE_KEYS)
        if unknown:
            raise HarnessError(f"unknown settings {sorted(unknown)}")
        if s.get("id") != self.path.name:
            raise HarnessError(f"id {s.get('id')!r} differs from the directory name {self.path.name!r}")
        ses, tools = self.sessions, self.tools
        if ses["verify"] not in VERIFY:
            raise HarnessError(f"sessions.verify must be one of {VERIFY}")
        second = {"none": set(), "independent": {"adjudication"}}.get(ses["verify"], {"verification"})
        roles = {"judgement"} | second | ({"first_impression"} if ses["first_impression"] else set())
        bad_triggers = set(ses["verify_triggers"]) - set(VERIFY_TRIGGERS)
        if bad_triggers:
            raise HarnessError(f"sessions.verify_triggers may only contain {VERIFY_TRIGGERS}, not {sorted(bad_triggers)}")
        missing_roles = roles - set(s.get("system", {}))
        if missing_roles:
            raise HarnessError(f"system prompts missing for {sorted(missing_roles)}")
        for role, files in s["system"].items():
            absent = [f for f in files if f not in self.parts]
            if absent:
                raise HarnessError(f"prompt parts {absent} of {role!r} not found in prompts/")
        kb = self.knowledge
        if kb is not None:
            kb_sessions = kb.get("sessions", ["verify"])
            if not kb_sessions or set(kb_sessions) - {"investigate", "verify"}:
                raise HarnessError('knowledge.sessions must be a non-empty subset of ["investigate", "verify"]')
            if kb_sessions == ["verify"] and ses["verify"] not in ("all", "ischemic", "uncertain"):
                raise HarnessError("knowledge is shown in verification sessions only: sessions.verify must be all, "
                                   "ischemic or uncertain (or add \"investigate\" to knowledge.sessions)")
            if not (kb["cards"] or kb["precedents"]):
                raise HarnessError("knowledge needs cards and/or precedents")
            from .knowledge import TOPIC_INPUTS, topic_rule_problems
            probs = topic_rule_problems(kb.get("topic_rules"), kb.get("always_topics"), set(TOPIC_INPUTS))
            if probs:
                raise HarnessError("knowledge.topic_rules: " + "; ".join(probs[:5]))
            from .knowledge import CARD_TOOL
            if any(e.get("name") == CARD_TOOL["name"] for e in self.ext_specs):
                raise HarnessError(f"{CARD_TOOL['name']} is reserved for the knowledge base")
        if tools["toolset"] not in TOOLSETS:
            raise HarnessError(f"tools.toolset must be one of {TOOLSETS}")
        bad = [t for t in tools["extra"] if t not in toolsets.EXTRA_TOOLS]
        if bad:
            raise HarnessError(f"tools.extra may only contain {toolsets.EXTRA_TOOLS}, not {bad}")
        names = self.tool_names()
        if set(names) & set(FORBIDDEN_TOOLS):
            raise HarnessError(f"{FORBIDDEN_TOOLS} are never offered")
        stale = set(tools["descriptions"]) - set(names)
        if stale:
            raise HarnessError(f"descriptions given for tools not offered: {sorted(stale)}")
        bad = set(ses.get("dossier_exclude") or []) - set(STANDARD_CHECKS)
        if bad or (ses.get("dossier_exclude") and "standard" not in ses["dossier"]):
            raise HarnessError(f"sessions.dossier_exclude lists checks of the standard set {list(STANDARD_CHECKS)} and "
                               f"needs 'standard' in the dossier (unknown: {sorted(bad)})")
        if not 0 <= int(self.limits["min_tool_calls"]) < int(self.limits["max_tool_calls"]):
            raise HarnessError("limits.min_tool_calls must be >= 0 and below max_tool_calls")
        for item in ses["dossier"]:
            if item == "standard":
                continue
            props = set(self._schema(item)["properties"]) if item in names else None
            if props != {"record_id", "candidate_id"}:
                raise HarnessError(f"dossier item {item!r} must be 'standard' or an offered tool taking only "
                                   "record_id and candidate_id")
        sub = self.submit_tool
        if sub.get("name") != "submit_assessment":
            raise HarnessError("submit.json must define the tool 'submit_assessment'")
        req = set(sub["input_schema"].get("required", []))
        props = sub["input_schema"].get("properties", {})
        if self.binary:
            if not SUBMIT_REQUIRED_BINARY <= req:
                raise HarnessError(f"submit_assessment must require {sorted(SUBMIT_REQUIRED_BINARY)}")
            if set((props.get("decision") or {}).get("enum", [])) != set(DECISIONS):
                raise HarnessError(f"decision must be an enum of exactly {list(DECISIONS)}")
            if "p_ischemic" in props:
                raise HarnessError("a two-way submission must not ask for p_ischemic")
        else:
            if not SUBMIT_REQUIRED <= req:
                raise HarnessError(f"submit_assessment must require {sorted(SUBMIT_REQUIRED_BINARY)} (two-way "
                                   f"decision) or {sorted(SUBMIT_REQUIRED)} (probability)")
            if not set(props["label"]["enum"]) <= set(LABELS):
                raise HarnessError(f"labels must come from {LABELS}")
        seen = set()
        for t in self.ext_specs:
            name = t.get("name")
            if not isinstance(name, str) or not name.isidentifier():
                raise HarnessError(f"tool spec without a valid name: {t!r:.200}")
            if name in seen:
                raise HarnessError(f"tool {name!r} defined twice")
            seen.add(name)
            if not (self.path / "tools" / f"{name}.json").exists() or not (self.path / "tools" / f"{name}.py").exists():
                raise HarnessError(f"tool {name!r} needs tools/{name}.json (with that name) and tools/{name}.py")
            sch = t.get("input_schema")
            if not isinstance(t.get("description"), str) or not isinstance(sch, dict) or sch.get("type") != "object" \
                    or not isinstance(sch.get("properties"), dict):
                raise HarnessError(f"tool {name!r}: needs a description and an object input_schema with properties")
        if self.ext_specs:
            clash = {t["name"] for t in self.ext_specs} & (set(TOOLS_BY_NAME) | {"submit_assessment"})
            if clash:
                raise HarnessError(f"extension tools reuse existing names {sorted(clash)}")

    @property
    def binary(self) -> bool:
        """Whether the harness submits a two-way decision (else a probability and a label)."""
        return "decision" in self.submit_tool.get("input_schema", {}).get("properties", {})

    # ------------------------------------------------------------------ prompts
    def system(self, role: str) -> str:
        return "".join(self.parts[f] for f in self.spec["system"][role])

    # ------------------------------------------------------------------ tools
    def _ext_specs(self) -> list[dict]:
        return list(self.ext_specs)

    def _schema(self, name: str) -> dict:
        if name in TOOLS_BY_NAME:
            return TOOLS_BY_NAME[name].input_schema
        return next(t["input_schema"] for t in self._ext_specs() if t["name"] == name)

    def tool_names(self) -> list[str]:
        t = self.tools
        names = [s.name for s in TOOL_SPECS
                 if OBSERVATION.offers(s.name) and toolsets.allows(t["toolset"], s.name, tuple(t["extra"]))]
        return names + [e["name"] for e in self._ext_specs()]

    def tool_definitions(self, analysis: bool = True) -> list[dict]:
        tools = []
        if analysis:
            desc = self.tools["descriptions"]
            for s in TOOL_SPECS:
                if s.name in self.tool_names():
                    tools.append({"name": s.name, "description": desc.get(s.name, s.description),
                                  "input_schema": s.input_schema})
            for e in self._ext_specs():
                tools.append({"name": e["name"], "description": desc.get(e["name"], e["description"]),
                              "input_schema": e["input_schema"]})
        return tools + [self.submit_tool]

    def is_ext(self, name: str) -> bool:
        return any(e["name"] == name for e in self._ext_specs())

    def run_ext(self, toolkit, name: str, args: dict, bound: dict) -> dict:
        schema = self._schema(name)
        extra = set(args) - set(schema["properties"])
        if extra:
            raise ToolError(f"unexpected arguments {sorted(extra)}")
        missing = [k for k in schema.get("required", []) if k not in args]
        if missing:
            raise ToolError(f"missing required arguments {missing}")
        if getattr(self, "_worker", None) is None:
            from .ext_worker import ExtWorker

            self._worker = ExtWorker(self.path / "tools", Path(toolkit.cache_dir).resolve())
        reply = self._worker.call(name, dict(args))
        if not reply.get("ok"):
            raise ToolError(reply.get("error", "the tool failed"))
        return jsonable(reply["result"])

    # ------------------------------------------------------------------ run configuration
    def run_config(self) -> RunConfig:
        ses, t, lim = self.sessions, self.tools, self.limits
        return RunConfig(protocol=PROTOCOL, reference=REFERENCE, mode="agent",
                         initial="separate" if ses["first_impression"] else "none", prior_source="none",
                         # "independent" is run by rsi.runner (blind second reading); the base loop stays off
                         verify="none" if ses["verify"] == "independent" else ses["verify"],
                         max_verify_rounds=int(ses["max_verify_rounds"]),
                         max_tool_calls=int(lim["max_tool_calls"]), stall_turns=int(lim["stall_turns"]),
                         observation=OBSERVATION, toolset=t["toolset"], graded=True, dossier=bool(ses["dossier"]),
                         extra_tools=tuple(t["extra"]), verify_band=tuple(ses["verify_band"]))

    def digest(self) -> str:
        return digest(self.path)
