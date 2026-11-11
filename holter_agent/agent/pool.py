"""OpenAI models through an OpenAI Responses-API compatible endpoint, called directly over HTTP.

Protocol: POST {LLM_POOL_URL}{LLM_POOL_PATH} with the Responses API body, stream=True (server-sent events),
store=False. Function tools, images inside function call outputs (an array of input_text / input_image parts) and
multi-turn conversations work when every turn sends the whole history back, including the reasoning items
(include=["reasoning.encrypted_content"]).

The final response.completed event does not always repeat the output items, so the items are collected from the
response.output_item.done events.

Credentials: the API key is read from the environment variable LLM_POOL_API_KEY and never written to disk or logs.

History stays in the Anthropic format used by AgentRunner; an assistant turn keeps the provider's output items in a
{"type": "responses_items"} block, which is sent back unchanged on the next request.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import requests

from .llm import ToolCall, Turn

POOL_URL = os.environ.get("LLM_POOL_URL", "")  # base URL of an OpenAI Responses-API compatible endpoint
# path of the Responses endpoint on that server
POOL_PATH = os.environ.get("LLM_POOL_PATH", "/v1/responses")
KEY_ENV = "LLM_POOL_API_KEY"
_KEY: str | None = None


def extra_headers() -> dict:
    return {}


def pool_key() -> str:
    """The API key of the Responses endpoint, read from the environment variable LLM_POOL_API_KEY."""
    global _KEY
    if _KEY:
        return _KEY
    if not POOL_URL or not os.environ.get(KEY_ENV):
        raise RuntimeError("set LLM_POOL_URL and LLM_POOL_API_KEY to an OpenAI Responses-API compatible endpoint")
    _KEY = os.environ[KEY_ENV]
    return _KEY


def _data_url(block: dict) -> str:
    src = block["source"]
    return f"data:{src['media_type']};base64,{src['data']}"


def _parts(content) -> list[dict]:
    """Anthropic text/image blocks -> Responses input parts."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append({"type": "input_text", "text": b["text"]})
        elif b.get("type") == "image":
            out.append({"type": "input_image", "image_url": _data_url(b)})
    return out


def responses_input(messages: list[dict]) -> list[dict]:
    """The AgentRunner history (Anthropic format) as Responses API input items."""
    items: list[dict] = []
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            kept = [b for b in content if isinstance(b, dict) and b.get("type") == "responses_items"]
            if kept:
                for b in kept:
                    items.extend(b["items"])
                continue
            for b in content:  # history written by another backend (tests, prior labels)
                if b.get("type") == "text" and b.get("text"):
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": b["text"]}]})
                elif b.get("type") == "tool_use":
                    items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                  "arguments": json.dumps(b["input"])})
            continue
        if isinstance(content, str):
            items.append({"role": "user", "content": _parts(content)})
            continue
        rest = []
        for b in content:
            if b.get("type") == "tool_result":
                inner = b["content"]
                if isinstance(inner, str):
                    output = ("ERROR: " if b.get("is_error") else "") + inner
                else:
                    output = _parts(inner)
                    if b.get("is_error") and output and output[0]["type"] == "input_text":
                        output[0]["text"] = "ERROR: " + output[0]["text"]
                items.append({"type": "function_call_output", "call_id": b["tool_use_id"], "output": output})
            else:
                rest.append(b)
        if rest:
            items.append({"role": "user", "content": _parts(rest)})
    return items


def responses_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "name": t["name"], "description": t.get("description", ""),
             "parameters": t["input_schema"], "strict": False} for t in tools]


class PoolError(RuntimeError):
    pass


class PoolBackend:
    """One model call per step(), through an OpenAI Responses-API endpoint."""

    def __init__(self, model: str = "gpt-6-astra", effort: str = "high", url: str = POOL_URL,
                 read_timeout_s: float = 900.0, retries: int = 8, call_deadline_s: float = 900.0):
        self.model, self.effort, self.url = model, effort, url
        self.name = f"pool:{model}"
        self.read_timeout_s, self.retries = read_timeout_s, retries
        # The pool sends keepalive events, so a stalled generation never trips the read timeout; every call also has a
        # wall-clock deadline, after which it counts as a temporary failure and is retried.
        self.call_deadline_s = call_deadline_s
        self.session = requests.Session()
        self.session.trust_env = False
        self.usage_total: dict = {}
        # one prompt-cache key per backend (one case), so that the repeated history of later turns is served from the
        # prompt cache
        self.cache_key = str(__import__("uuid").uuid4())

    def _post(self, payload: dict) -> tuple[list[dict], dict]:
        headers = {"Authorization": "Bearer " + pool_key(), "Accept": "text/event-stream", **extra_headers()}
        payload = {**payload, "prompt_cache_key": self.cache_key}
        items: list[dict] = []
        completed: dict | None = None
        with self.session.post(self.url + POOL_PATH, json=payload, headers=headers, stream=True,
                               timeout=(15, self.read_timeout_s), allow_redirects=False) as r:
            if r.status_code != 200:
                raise PoolError(f"HTTP {r.status_code}: {r.text[:300]}")
            buf: list[str] = []

            def handle(data: str) -> None:
                nonlocal completed
                if not data.startswith("{"):
                    return  # [DONE] and keepalive markers
                ev = json.loads(data)
                kind = ev.get("type")
                if kind == "response.output_item.done":
                    items.append(ev["item"])
                elif kind in ("response.completed", "response.failed", "response.incomplete"):
                    completed = ev["response"]
                elif kind == "error":
                    err = ev.get("error") or {}  # type and code first: a long message must not hide them
                    raise PoolError(f"error event ({err.get('type')}/{err.get('code')}): "
                                    f"{str(err.get('message') or json.dumps(ev))[:300]}")

            t0 = time.time()
            for raw in r.iter_lines():
                if time.time() - t0 > self.call_deadline_s:
                    raise PoolError(f"stream ended without a completion event within {self.call_deadline_s:.0f} s "
                                    "(keepalives only)")
                line = raw.decode("utf-8")
                if not line:
                    if buf:
                        handle("\n".join(buf))
                        buf = []
                elif line.startswith("data:"):
                    buf.append(line[5:].lstrip())
            if buf:
                handle("\n".join(buf))
        if completed is None:
            raise PoolError("stream ended without a completion event")
        if completed.get("status") != "completed":
            raise PoolError(f"response {completed.get('status')}: {json.dumps(completed.get('error') or completed.get('incomplete_details'))[:300]}")
        return items or list(completed.get("output") or []), completed.get("usage") or {}

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        payload = {"model": self.model, "instructions": system, "input": responses_input(messages),
                   "reasoning": {"effort": self.effort, "summary": "auto"}, "store": False, "stream": True,
                   "include": ["reasoning.encrypted_content"]}
        if tools:
            payload |= {"tools": responses_tools(tools), "tool_choice": "auto", "parallel_tool_calls": True}
        for attempt in range(self.retries + 1):
            try:
                items, usage = self._post(payload)
                break
            except (PoolError, requests.RequestException) as e:
                transient = isinstance(e, requests.RequestException) or bool(  # rate limits, any 5xx (e.g. 520),
                    re.search(r"HTTP (429|5\d\d)\b|without a completion|response failed|response incomplete|service_unavailable|"
                              r"server_error|rate_limit|overloaded|[Pp]lease retry|can retry your request", str(e)))  # server-side, temporary
                if attempt == self.retries or not transient:
                    raise
                # 10, 20, ... 160, then 300 s (about 20 min in all) rides out the pool's server_is_overloaded spells
                # (2026-10-20: 5 retries / 4.5 min often were not enough); the random factor keeps the processes of
                # one evaluation from retrying all at the same moment (48 synchronised retries stalled for hours)
                time.sleep(min(10 * 2 ** attempt, 300) * random.uniform(0.5, 1.5))
        calls, texts, reasoning = [], [], []
        for it in items:
            if it.get("type") == "function_call":
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_unparsed_arguments": it.get("arguments")}
                calls.append(ToolCall(id=it["call_id"], name=it["name"], input=args))
            elif it.get("type") == "message":
                texts += [c.get("text", "") for c in it.get("content", []) if c.get("type") == "output_text"]
            elif it.get("type") == "reasoning":
                reasoning += [s.get("text", "") for s in it.get("summary") or []]
        u = {"input_tokens": usage.get("input_tokens", 0),
             "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
             "output_tokens": usage.get("output_tokens", 0),
             "reasoning_output_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        for k, v in u.items():
            self.usage_total[k] = self.usage_total.get(k, 0) + v
        return Turn(content=[{"type": "responses_items", "items": items}], text="\n".join(texts), tool_calls=calls,
                    stop_reason="tool_use" if calls else "end_turn", usage=u, reasoning="\n".join(reasoning))
