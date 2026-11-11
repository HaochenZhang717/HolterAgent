"""Run the review sessions with an OpenAI model through the Codex CLI (ChatGPT sign-in, ChatGPT credits).

The judge-verify-correct protocol, prompts, tools, evidence ledger and acceptance rules are those of
AgentRunner; only the model loop differs. Each session (first impression, judgement, verification k)
is one `codex exec` run:

- our system prompt replaces Codex's built-in instructions (`model_instructions_file`);
- Codex's own tools are switched off (shell, unified exec, image viewing, web search, sub-agents, apps,
  plugins, memories, ...), and it works in an empty directory with a read-only sandbox, so the model
  cannot read the recordings or reference annotations on disk. Code mode stays on for these models
  (switching its host off disables all tools); its `exec` runs JavaScript without file, network or
  module access (checked on 2026-10-05: require, fetch, Deno and import are unavailable), so it can
  only call the tools below;
- our tools are offered through a small MCP server (codex_mcp.py) that forwards every call to this
  process, where AgentRunner's own tool and submission handling runs against the shared ledger;
- the first message and its images are passed as the prompt and `--image` files.

Differences from the other backends, recorded in the trace: limits count tool calls, not model turns
(at most max_tool_calls analysis calls; a session that keeps calling past the budget or without new
evidence is stopped), and a session that ends without a final assessment is resumed once with the
same nudge. Codex also adds an environment message (working directory, date) and an empty skills
header; the reasoning text available is Codex's reasoning summary, not the full reasoning.
"""
from __future__ import annotations

import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .llm import ToolCall
from .prompts import LAST_TURN_NOTE, NUDGE_FINAL, STALL_NOTE, SUBMIT_TOOL
from .runner import AgentRunner, _without_images

# Codex features that would give the model tools or context beyond ours (codex-cli 0.155).
DISABLED_FEATURES = ("shell_tool", "unified_exec", "multi_agent", "apps", "plugins", "memories", "image_generation",
                     "browser_use", "browser_use_external", "computer_use", "in_app_browser", "skill_search",
                     "tool_suggest", "sleep_tool", "goals", "view_image")
# Credits per 1M tokens (input, cached input, output) from the Codex pricing page (learn.chatgpt.com/docs/pricing),
# used only to report what a run cost.
CREDIT_RATES = {"gpt-6-astra": (250.0, 25.0, 1250.0), "gpt-6-sol": (50.0, 5.0, 250.0),
                "gpt-6-luna": (2.5, 0.25, 12.5)}


@dataclass(frozen=True)
class CodexConfig:
    model: str = "gpt-6-astra"
    effort: str = "high"
    codex_bin: str = "codex"
    timeout_s: float = 2400.0  # one codex exec run
    calls_after_limit: int = 3  # calls tolerated past the budget or a stall before the session is stopped
    output_token_limit: int = 30000  # per MCP tool result (Codex truncates longer outputs)
    tool_timeout_s: float = 600.0


def toml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid TOML basic string


def codex_command(cc: CodexConfig, workdir: Path, instructions: Path, tools_json: Path, sock: Path,
                  tool_names: list[str], images: list[Path], resume_id: str | None = None) -> list[str]:
    cmd = [cc.codex_bin, "exec"] + (["resume", resume_id] if resume_id else [])
    cmd += ["--json", "--ignore-user-config", "--skip-git-repo-check", "-m", cc.model,
            "-c", f"model_reasoning_effort={toml_str(cc.effort)}",
            "-c", f"model_instructions_file={toml_str(str(instructions))}",
            "-c", 'web_search="disabled"', "-c", "tools.view_image=false", "-c", "agents.enabled=false",
            "-c", "skills.max_context_tokens=1", "-c", 'approval_policy="never"', "-c", 'sandbox_mode="read-only"',
            "-c", 'model_reasoning_summary="detailed"', "-c", "features.code_mode.enabled=false"]
    for f in DISABLED_FEATURES:
        cmd += ["-c", f"features.{f}=false"]
    srv = "mcp_servers.holter"
    cmd += ["-c", f"{srv}.command={toml_str(sys.executable)}",
            "-c", f"{srv}.args={json.dumps(['-m', 'holter_agent.agent.codex_mcp', '--tools', str(tools_json), '--socket', str(sock)])}",
            "-c", f"{srv}.cwd={toml_str(str(Path(__file__).resolve().parents[2]))}",
            "-c", f"{srv}.required=true", "-c", f"{srv}.startup_timeout_sec=60",
            "-c", f"{srv}.tool_timeout_sec={cc.tool_timeout_s:g}", "-c", f'{srv}.default_tools_approval_mode="approve"']
    for name in tool_names:
        cmd += ["-c", f"{srv}.tools.{name}.output_token_limit={cc.output_token_limit}"]
    if not resume_id:
        cmd += ["-C", str(workdir)]
    for img in images:
        cmd += ["--image", str(img)]
    return cmd + ["-"]  # the prompt is read from stdin


def mcp_content(block: dict) -> dict:
    """Our tool_result block -> MCP CallToolResult (text and PNG images)."""
    content = block["content"]
    items = [{"type": "text", "text": content}] if isinstance(content, str) else []
    if not isinstance(content, str):
        for c in content:
            if c.get("type") == "text":
                items.append({"type": "text", "text": c["text"]})
            elif c.get("type") == "image":
                items.append({"type": "image", "data": c["source"]["data"], "mimeType": c["source"]["media_type"]})
    return {"content": items, "isError": bool(block.get("is_error"))}


@dataclass
class _Session:
    cid: str
    session: str
    stage: str
    analysis: bool
    require_new_evidence: bool
    n_tools: int = 0
    stall: int = 0
    past_limit: int = 0
    submitted: bool = False
    stop: str | None = None  # set when the session must be stopped from our side
    lock: threading.Lock = field(default_factory=threading.Lock)


class CodexAgentRunner(AgentRunner):
    def __init__(self, toolkit, cfg, codex: CodexConfig | None = None, trace_path=None, baseline_cfg=None):
        super().__init__(toolkit, backend=None, cfg=cfg, trace_path=trace_path, baseline_cfg=baseline_cfg)
        self.codex = codex or CodexConfig()
        self.usage: Counter = Counter()
        self.threads: list[str] = []  # Codex thread ids of all sessions (for clean-up)

    # ------------------------------------------------------------------ tool calls from Codex
    def _handle(self, state, ss: _Session, name: str, arguments: dict) -> dict:
        cfg = self.cfg
        with ss.lock:
            state.step += 1
            call = ToolCall(id=f"codex-{state.step}", name=name, input=dict(arguments))
            new, sub_now = False, False
            if name == SUBMIT_TOOL["name"]:
                block, sub_now = self._submit(state, call, ss.cid, ss.session, ss.stage, ss.require_new_evidence)
                ss.submitted |= sub_now
            elif not ss.analysis:
                block = {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                         "content": "No analysis tools are available in this step."}
            elif not self._offers(name):
                block = {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                         "content": f"{name} is not available in this run."}
            elif ss.n_tools >= cfg.max_tool_calls:
                block = {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                         "content": "Tool budget exhausted. Submit your final assessment now."}
                ss.past_limit += 1
            else:
                ss.n_tools += 1
                block, new = self._analysis(state, call, session=ss.session)
            ss.stall = 0 if (new or sub_now) else ss.stall + 1
            result = mcp_content(block)
            if ss.analysis and ss.stage == "final" and not ss.submitted:
                if ss.stall >= cfg.stall_turns:
                    result["content"].append({"type": "text", "text": LAST_TURN_NOTE})
                    ss.past_limit += ss.stall > cfg.stall_turns
                elif ss.stall == cfg.stall_turns - 1:
                    result["content"].append({"type": "text", "text": STALL_NOTE})
                elif ss.n_tools == cfg.max_tool_calls - 1:
                    result["content"].append({"type": "text", "text": LAST_TURN_NOTE})
            if ss.past_limit > self.codex.calls_after_limit and not ss.submitted:
                ss.stop = "stalled"
            self._trace({"step": state.step, "candidate": ss.cid, "session": ss.session, "backend": "codex",
                         "calls": [{"id": call.id, "name": name, "input": arguments}],
                         "results": _without_images([block])})
            return result

    # ------------------------------------------------------------------ one session
    def _session(self, state, cid, session, system, first_message, analysis=True, stage="final", max_turns=None,
                 require_new_evidence=True, images=None) -> str:
        cfg = self.cfg
        tools = self._tool_definitions(analysis)
        ss = _Session(cid, session, stage, analysis, require_new_evidence)
        tmp = Path(tempfile.mkdtemp(prefix="hc", dir="/tmp"))  # short path: Unix socket paths are limited
        try:
            work = tmp / "w"
            work.mkdir()
            instructions = tmp / "instructions.md"
            instructions.write_text(system)
            tools_json = tmp / "tools.json"
            tools_json.write_text(json.dumps([{"name": t["name"], "description": t["description"],
                                               "inputSchema": t["input_schema"]} for t in tools]))
            img_paths = []
            for k, png in enumerate(images or []):
                p = tmp / f"image{k}.png"
                p.write_bytes(png)
                img_paths.append(p)
            self._trace({"step": state.step, "candidate": cid, "session": session, "backend": "codex",
                         "prompt": {"system": system, "first_message": first_message,
                                    "images": self._save_attachments(session, images or []),
                                    "tools": [t["name"] for t in tools]}})
            sock = tmp / "s"
            runner = self

            class Handler(socketserver.StreamRequestHandler):
                def handle(self):
                    req = json.loads(self.rfile.readline())
                    try:
                        res = runner._handle(state, ss, req["name"], req.get("arguments") or {})
                    except Exception as e:  # noqa: BLE001 - report to the model, keep the session alive
                        res = {"content": [{"type": "text", "text": f"Tool error: {type(e).__name__}: {e}"}],
                               "isError": True}
                    self.wfile.write((json.dumps(res) + "\n").encode())

            server = socketserver.ThreadingUnixStreamServer(str(sock), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                names = [t["name"] for t in tools]
                stop, thread_id = self._exec(state, ss, codex_command(self.codex, work, instructions, tools_json,
                                                                      sock, names, img_paths), first_message)
                if stop == "completed" and stage == "final" and not ss.submitted and thread_id:
                    stop, _ = self._exec(state, ss, codex_command(self.codex, work, instructions, tools_json, sock,
                                                                  names, [], resume_id=thread_id), NUDGE_FINAL)
            finally:
                server.shutdown()
                server.server_close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if stop == "completed":
            stop = "completed" if ss.submitted else "no_assessment"
        state.set_stop(cid, session, stop)
        return stop

    def _save_attachments(self, session: str, images: list[bytes]) -> list[str]:
        """Images attached to a session's first message, saved next to the trace (paths relative to it)."""
        if self.trace_path is None or not images:
            return []
        folder = self.trace_path.parent / "images"
        folder.mkdir(parents=True, exist_ok=True)
        prefix = self.trace_path.name.removesuffix(".trace.jsonl")
        names = []
        for k, png in enumerate(images):
            name = f"{prefix}_{session}_attach{k}.png"
            (folder / name).write_bytes(png)
            names.append(f"images/{name}")
        return names

    def _exec(self, state, ss: _Session, cmd: list[str], prompt: str) -> tuple[str, str | None]:
        """Run one codex exec; returns (stop reason, thread id). Events go to the trace."""
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)
        proc.stdin.write(prompt)
        proc.stdin.close()
        events: list[dict] = []
        reader = threading.Thread(target=lambda: events.extend(_read_events(proc.stdout)), daemon=True)
        reader.start()
        stop = "completed"
        while proc.poll() is None:
            if ss.stop:
                stop = ss.stop
                proc.terminate()
                break
            if time.time() - t0 > self.codex.timeout_s:
                stop = "timeout"
                proc.terminate()
                break
            time.sleep(0.5)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader.join(timeout=30)
        err = proc.stderr.read()[-4000:] if proc.stderr else ""
        if stop == "completed" and proc.returncode not in (0, None):
            stop = "error"
        thread_id, usage, items = _summarise(events)
        if thread_id:
            self.threads.append(thread_id)
        self.usage.update(usage)
        state.step += 1
        rec = {"step": state.step, "candidate": ss.cid, "session": ss.session, "backend": "codex",
               "codex": {"model": self.codex.model, "effort": self.codex.effort, "thread_id": thread_id,
                         "returncode": proc.returncode, "stop": stop, "seconds": round(time.time() - t0, 1)},
               "usage": usage, "items": items}
        if stop in ("error", "timeout") or proc.returncode:
            rec["stderr_tail"] = err
        self._trace(rec)
        return stop, thread_id


def _read_events(stream) -> list[dict]:
    out = []
    for line in stream:
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _summarise(events: list[dict]) -> tuple[str | None, dict, list[dict]]:
    """Thread id, summed token usage and the model's messages, reasoning summaries and tool calls."""
    thread_id, usage, items = None, Counter(), []
    for ev in events:
        thread_id = thread_id or ev.get("thread_id")
        u = ev.get("usage")
        if isinstance(u, dict):
            usage.update({k: v for k, v in u.items() if isinstance(v, (int, float))})
        item = ev.get("item")
        if ev.get("type") == "item.completed" and isinstance(item, dict):
            kind = item.get("type")
            if kind in ("reasoning", "agent_message"):
                items.append({"type": kind, "text": item.get("text", "")})
            elif kind == "mcp_tool_call":
                items.append({"type": kind, "tool": item.get("tool"), "status": item.get("status")})
            elif kind not in ("user_message",):
                items.append({"type": kind, "detail": {k: v for k, v in item.items() if k not in ("id",)}})
        elif ev.get("type") in ("error", "turn.failed"):
            items.append({"type": ev.get("type"), "detail": ev})
    return thread_id, dict(usage), items


def credits(model: str, usage: dict) -> float | None:
    """ChatGPT credits for the summed usage, at the rates of the Codex pricing page."""
    rates = CREDIT_RATES.get(model)
    if rates is None:
        return None
    cached = usage.get("cached_input_tokens", 0)
    fresh = max(0, usage.get("input_tokens", 0) - cached)
    return (fresh * rates[0] + cached * rates[1] + usage.get("output_tokens", 0) * rates[2]) / 1e6


