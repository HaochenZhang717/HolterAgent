"""LLM backends behind one small interface.

Messages are kept in the Anthropic Messages format (role + list of content
blocks); a backend for another provider translates at its boundary
(OpenAICompatBackend, for open-weight models served by vLLM or SGLang).
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    content: list[Any]  # provider content blocks, appended to the history unchanged
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict = field(default_factory=dict)
    reasoning: str = ""  # visible reasoning text, when the backend returns it


class LLMBackend(Protocol):
    name: str

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


def image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode("ascii")}}


def user_content(text: str, images: list[bytes] | None = None) -> str | list[dict]:
    """A user message: plain text, or text followed by images."""
    return [{"type": "text", "text": text}] + [image_block(p) for p in images] if images else text


def tool_result_block(call_id: str, payload: dict | str, is_error: bool = False,
                      images: list[bytes] | None = None) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    block = {"type": "tool_result", "tool_use_id": call_id, "content": user_content(text, images)}
    if is_error:
        block["is_error"] = True
    return block


class AnthropicBackend:
    """Claude through the official SDK: adaptive thinking, strict tools, server-side refusal fallback."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high", max_tokens: int = 16000,
                 fallbacks: bool = True, client: Any = None):
        import anthropic  # optional dependency

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.name = f"anthropic:{model}:{effort}"

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            system=system,
            tools=[{**t, "strict": True} for t in tools],
            messages=messages,
        )
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        resp = self.client.beta.messages.create(**kwargs)
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {}
        if getattr(resp, "usage", None) is not None:
            u = resp.usage
            usage = {k: getattr(u, k, None) for k in
                     ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        return Turn(resp.content, text, calls, resp.stop_reason or "", usage)


def chat_messages(system: str, messages: list[dict], tool_images: str = "inline") -> list[dict]:
    """Anthropic-format history -> OpenAI Chat Completions messages.

    tool_images="inline" keeps a tool result's images inside its tool message (the Qwen3.5 chat template
    renders them there); "user" moves them to a user message after the tool messages, for servers that
    reject images in tool messages.
    """
    def data_url(b: dict) -> str:
        return f"data:{b['source']['media_type']};base64,{b['source']['data']}"

    def parts(blocks: list[dict]) -> list[dict]:
        out = []
        for b in blocks:
            if b["type"] == "text":
                out.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                out.append({"type": "image_url", "image_url": {"url": data_url(b)}})
        return out

    def plain(p: list[dict]) -> str | list[dict]:
        return "\n".join(x["text"] for x in p) if all(x["type"] == "text" for x in p) else p

    out: list[dict] = [{"role": "system", "content": system}]
    names: dict[str, str] = {}  # tool call id -> tool name, to label moved images
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            msg: dict[str, Any] = {"role": "assistant",
                                   "content": "".join(b["text"] for b in blocks if b["type"] == "text")}
            reasoning = "".join(b["thinking"] for b in blocks if b["type"] == "thinking")
            if reasoning:
                msg["reasoning_content"] = reasoning
            calls = [b for b in blocks if b["type"] == "tool_use"]
            if calls:
                msg["tool_calls"] = [{"id": b["id"], "type": "function",
                                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                                     for b in calls]
                names.update((b["id"], b["name"]) for b in calls)
            out.append(msg)
            continue
        if isinstance(content, str):
            out.append({"role": "user", "content": content})
            continue
        rest = [b for b in content if b["type"] != "tool_result"]
        moved: list[dict] = []
        for b in content:
            if b["type"] != "tool_result":
                continue
            inner = b["content"] if isinstance(b["content"], list) else [{"type": "text", "text": b["content"]}]
            p = parts(inner)
            if tool_images == "user" and any(x["type"] == "image_url" for x in p):
                moved.append({"type": "text", "text": f"Image returned by {names.get(b['tool_use_id'], 'a tool')} "
                                                      f"(result above):"})
                moved += [x for x in p if x["type"] == "image_url"]
                p = [x for x in p if x["type"] == "text"]
            out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": plain(p)})
        user = moved + parts(rest)
        if user:
            out.append({"role": "user", "content": plain(user)})
    return out


class OpenAICompatBackend:
    """An open-weight model behind an OpenAI-compatible Chat Completions server (vLLM, SGLang).

    The history stays in the Anthropic format and is translated on every step (chat_messages). The
    model's reasoning is kept as a "thinking" block and sent back as `reasoning_content`; the chat
    template decides whether to show it. Standard library only, so no extra dependency.
    """

    # Qwen3.5 model card, "Best Practices": thinking mode for general tasks.
    QWEN_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 max_tokens: int = 32768, sampling: dict | None = None, tool_images: str = "inline",
                 timeout_s: float = 1800.0, extra_body: dict | None = None, readout_body: dict | None = None,
                 readout_top_k: int = 20, readout_max_tokens: int = 1):
        """extra_body: added to every step request (e.g. {"enable_thinking": true} on Alibaba Cloud Model Studio).
        readout_body / readout_top_k / readout_max_tokens: the yes/no read-out request (thinking off: vLLM takes
        chat_template_kwargs, Model Studio enable_thinking; Model Studio returns at most 5 top log-probabilities and
        an empty list with max_tokens 1)."""
        self.extra_body = dict(extra_body or {})
        self.readout_body = dict({"chat_template_kwargs": {"enable_thinking": False}} if readout_body is None
                                 else readout_body)
        self.readout_top_k, self.readout_max_tokens = readout_top_k, readout_max_tokens
        if tool_images not in ("inline", "user"):
            raise ValueError("tool_images must be 'inline' or 'user'")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.sampling = dict(self.QWEN_THINKING if sampling is None else sampling)
        self.tool_images = tool_images
        self.timeout_s = timeout_s
        self.name = f"openai:{model}"
        self._n = 0

    def yes_no(self, system: str, messages: list[dict], tools: list[dict], question: str, top_k: int | None = None) -> dict:
        """Score a yes/no question from the next-token log-probabilities instead of a stated probability: the
        conversation so far plus the question, thinking off, one generated token; p_yes = P(yes) / (P(yes) + P(no))
        over the top_k tokens (case and leading space ignored). Log-probabilities are vLLM's raw ones (before
        temperature), so the sampling settings do not change them."""
        import urllib.request

        msgs = chat_messages(system, messages + [{"role": "user", "content": question}], self.tool_images)
        body = {"model": self.model, "messages": msgs, "max_tokens": self.readout_max_tokens, "temperature": 0.0,
                "logprobs": True, "top_logprobs": top_k or self.readout_top_k, **self.readout_body}
        if tools:  # same tool list as the session, so the prompt prefix (and the prefix cache) stays the same
            body |= {"tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                 "parameters": t["input_schema"]}} for t in tools],
                     "tool_choice": "none"}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        top = ((choice.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
        yes, no = _yes_no_mass(top)
        u = data.get("usage") or {}
        return {"p_yes": _p_yes(top), "mass_yes": yes, "mass_no": no,
                "usage": {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")},
                "token": choice["message"].get("content"),
                "top": [(t["token"], round(t["logprob"], 3)) for t in top[:10]]}

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        import urllib.error
        import urllib.request

        body = {"model": self.model, "messages": chat_messages(system, messages, self.tool_images),
                "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                            "parameters": t["input_schema"]}} for t in tools],
                "max_tokens": self.max_tokens, **self.sampling, **self.extra_body}
        if tools:
            body["tool_choice"] = "auto"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.url} returned {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
        choice = data["choices"][0]
        msg = choice["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        text = msg.get("content") or ""
        blocks: list[dict] = ([{"type": "thinking", "thinking": reasoning}] if reasoning else []) + (
            [{"type": "text", "text": text}] if text else [])
        calls = []
        for tc in msg.get("tool_calls") or []:
            self._n += 1
            cid = tc.get("id") or f"call_{self._n}"
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {}  # the tool then reports the missing arguments to the model
            blocks.append({"type": "tool_use", "id": cid, "name": tc["function"]["name"], "input": args})
            calls.append(ToolCall(cid, tc["function"]["name"], args))
        stop = "tool_use" if calls else ("max_tokens" if choice.get("finish_reason") == "length" else "end_turn")
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
        return Turn(blocks, text, calls, stop, usage, reasoning)


def _p_yes(top: list[dict]) -> float | None:
    """P(yes) / (P(yes) + P(no)) from top log-probabilities. A side absent from the top tokens (a server that returns
    only a few, e.g. 5) is given the probability left outside them as an upper bound, so confident answers still rank."""
    yes, no = _yes_no_mass(top)
    if yes + no == 0:
        return None
    rest = max(0.0, 1.0 - sum(math.exp(t["logprob"]) for t in top))
    yes, no = (yes or rest), (no or rest)
    return yes / (yes + no) if yes + no > 0 else None


def _yes_no_mass(top: list[dict]) -> tuple[float, float]:
    """Probability mass of the yes and no tokens (any case, with or without a leading space) among top log-probs."""
    yes = no = 0.0
    for t in top:
        w = t.get("token", "").strip().lower()
        if w == "yes":
            yes += math.exp(t["logprob"])
        elif w == "no":
            no += math.exp(t["logprob"])
    return yes, no


class ScriptedBackend:
    """Offline backend driven by a Python policy; used for tests and dry runs.

    `policy(system, messages)` returns either a list of (tool_name, input) pairs or
    a string (a final text turn with no tool calls).
    """

    def __init__(self, policy: Callable[[str, list[dict]], list[tuple[str, dict]] | str], name: str = "scripted"):
        self.policy = policy
        self.name = name
        self._n = 0

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        out = self.policy(system, messages)
        if isinstance(out, str):
            return Turn([{"type": "text", "text": out}], out, [], "end_turn")
        blocks, calls = [], []
        for name, args in out:
            self._n += 1
            cid = f"call_{self._n}"
            blocks.append({"type": "tool_use", "id": cid, "name": name, "input": args})
            calls.append(ToolCall(cid, name, args))
        return Turn(blocks, "", calls, "tool_use")
