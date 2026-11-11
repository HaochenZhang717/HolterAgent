"""Minimal MCP stdio server that forwards tool calls to the review process (see codex_runner.py).

Codex starts this process for one session. It answers the MCP handshake and `tools/list` itself (the
tool definitions are read from a JSON file written by the review process) and forwards each
`tools/call` over a Unix socket to the review process, which runs the tool against the shared
evidence ledger exactly as in the other backends and returns MCP content (text and images).
Only JSON-RPC messages go to stdout; diagnostics go to stderr.

    python -m holter_agent.agent.codex_mcp --tools <tools.json> --socket <path>
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

PROTOCOL_FALLBACK = "2025-06-18"


def forward(sock_path: str, request: dict) -> dict:
    """Send one request to the review process and return its reply (newline-delimited JSON)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        s.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--socket", required=True)
    args = ap.parse_args()
    tools = json.loads(open(args.tools).read())
    out = sys.stdout

    def reply(msg_id, result=None, error=None) -> None:
        msg = {"jsonrpc": "2.0", "id": msg_id}
        msg.update({"error": error} if error else {"result": result})
        out.write(json.dumps(msg) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"[codex_mcp] bad JSON: {line[:200]}", file=sys.stderr)
            continue
        method, msg_id = msg.get("method"), msg.get("id")
        if msg_id is None:  # notifications (initialized, cancelled, ...) need no reply
            continue
        params = msg.get("params") or {}
        if method == "initialize":
            reply(msg_id, {"protocolVersion": params.get("protocolVersion", PROTOCOL_FALLBACK),
                           "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": "holter-tools", "version": "1"}})
        elif method == "tools/list":
            reply(msg_id, {"tools": tools})
        elif method == "tools/call":
            try:
                res = forward(args.socket, {"name": params.get("name"), "arguments": params.get("arguments") or {}})
            except OSError as e:
                res = {"content": [{"type": "text", "text": f"Tool service unavailable: {e}"}], "isError": True}
            reply(msg_id, res)
        elif method == "ping":
            reply(msg_id, {})
        else:
            reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
