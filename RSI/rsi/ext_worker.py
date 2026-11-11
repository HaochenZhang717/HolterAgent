"""Run a task agent's own tools (task_agent/tools/<name>.py) in a sandbox, one request per line on stdin.

The tool code is written by a meta agent, so it never runs in the evaluation process: the evaluation starts this
worker under sandbox-exec, where it can read only the Python environment, the holter_agent package, this package and
the feature cache (no recordings, annotations, headers, splits or results; no network; no writes).

A tool module defines   def run(ctx, args: dict) -> dict   (ctx: rsi.ext_context.ExtContext).
Request:  {"name": "<tool>", "args": {...}}      Reply: {"ok": true, "result": {...}} or {"ok": false, "error": "..."}

    ExtWorker(tools_dir, cache_dir).call(name, args)   (used by rsi.harness)
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import threading
import traceback
from pathlib import Path

RSI_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = RSI_DIR.parent
PROJECT = CODE_DIR.parent
MAX_RESULT_CHARS = 30000
CALL_TIMEOUT_S = 120.0


def _jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, float):
        return round(x, 4)
    return x


def serve(tools_dir: Path, cache_dir: Path) -> None:
    sys.path.insert(0, str(RSI_DIR))
    from rsi.ext_context import ExtContext

    ctx, modules = ExtContext(cache_dir), {}
    for line in sys.stdin:
        try:
            req = json.loads(line)
            name = req["name"]
            if name not in modules:
                path = tools_dir / f"{name}.py"
                spec = importlib.util.spec_from_file_location(f"task_tool_{name}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                modules[name] = mod
            result = _jsonable(modules[name].run(ctx, dict(req.get("args") or {})))
            if not isinstance(result, dict):
                raise TypeError(f"run() must return a dict, not {type(result).__name__}")
            text = json.dumps(result)
            if len(text) > MAX_RESULT_CHARS:
                raise ValueError(f"result too long ({len(text)} characters; limit {MAX_RESULT_CHARS})")
            reply = {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 - every failure is reported to the caller
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc(limit=3)[-1500:]}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


def profile(tools_dir: Path, cache_dir: Path, tmp: Path) -> str:
    q = lambda p: '"' + str(Path(p).resolve()) + '"'  # noqa: E731
    home = Path.home()
    lines = ["(version 1)", "(allow default)"]
    lines += [f"(deny file-read* (subpath {q(p)}))" for p in (home / "Documents", home / "Desktop", home / "Downloads",
                                                              home / ".claude", home / ".codex", home / ".sky", home / ".ssh",
                                                              PROJECT)]
    lines += [f"(allow file-read* (literal {q(p)}))" for p in (PROJECT, CODE_DIR, RSI_DIR)]  # directory entries only
    lines += [f"(allow file-read* (subpath {q(p)}))" for p in (PROJECT / ".venv", CODE_DIR / "holter_agent",
                                                               RSI_DIR / "rsi", cache_dir, tools_dir)]
    lines += ['(deny file-write* (subpath "/"))', f'(allow file-write* (subpath {q(tmp)}) (subpath "/dev"))',
              "(deny network*)"]
    return "\n".join(lines) + "\n"


class ExtWorker:
    """A sandboxed worker process, started on first use."""

    def __init__(self, tools_dir: Path, cache_dir: Path, sandboxed: bool = True):
        self.tools_dir, self.cache_dir, self.sandboxed = Path(tools_dir), Path(cache_dir), sandboxed
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="extw"))
        argv = [str(PROJECT / ".venv" / "bin" / "python"), "-m", "rsi.ext_worker", "--serve",
                str(self.tools_dir), str(self.cache_dir)]
        if self.sandboxed:
            prof = self.tmp / "worker.sb"
            prof.write_text(profile(self.tools_dir, self.cache_dir, self.tmp))
            argv = ["sandbox-exec", "-f", str(prof)] + argv
        env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(RSI_DIR), "TMPDIR": str(self.tmp)}
        self.proc = subprocess.Popen(argv, cwd=self.tmp, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env=env)

    def call(self, name: str, args: dict, timeout_s: float = CALL_TIMEOUT_S) -> dict:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps({"name": name, "args": args}) + "\n")
            self.proc.stdin.flush()
            box: list[str] = []
            t = threading.Thread(target=lambda: box.append(self.proc.stdout.readline()), daemon=True)
            t.start()
            t.join(timeout_s)
            if not box or not box[0]:
                self.close()
                return {"ok": False, "error": f"the tool did not answer within {timeout_s:.0f} s (or crashed)"}
            return json.loads(box[0])

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc = None


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--serve":
        serve(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit("usage: python -m rsi.ext_worker --serve <tools_dir> <cache_dir>")
