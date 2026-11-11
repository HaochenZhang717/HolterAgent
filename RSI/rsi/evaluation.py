"""Run a harness on a list of cases, one run_case.py process per case, in parallel; resumable.

Used by evaluate.py (fixed rounds) and by the DGM-H loop. A case with a result file is never run again, so an
interrupted evaluation (network loss, a killed process) continues where it stopped when it is called again.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .paths import CODE_DIR, LTSTDB_CACHE, LTSTDB_DIR, RSI_DIR


def pool_env() -> dict:
    """Environment for child processes with the pool key, fetched once through the shared client."""
    from holter_agent.agent.pool import KEY_ENV, pool_key

    return dict(os.environ, **{KEY_ENV: pool_key()})


def pool_reachable(timeout_s: float = 15.0) -> bool:
    """Whether the pool answers at all (any HTTP response counts; a network error does not)."""
    from holter_agent.agent.pool import POOL_URL

    try:
        s = requests.Session()
        s.trust_env = False
        s.get(POOL_URL + "/v1/models", timeout=timeout_s)
        return True
    except requests.RequestException:
        return False


def wait_for_pool(log=print, poll_s: float = 60.0) -> None:
    """Block until the pool is reachable (laptop offline, VPN down, pool restarting)."""
    waited = 0.0
    while not pool_reachable():
        if waited == 0 or waited % 600 < poll_s:
            log(f"pool not reachable; waiting (checked every {poll_s:.0f} s)")
        time.sleep(poll_s)
        waited += poll_s


def done(out: Path, cid: str) -> bool:
    return (out / f"{cid}.json").exists()


def run_cases(harness_dir: Path, cases: list[dict], out: Path, backend: str = "pool", concurrency: int = 10,
              env: dict | None = None, data: Path = LTSTDB_DIR, cache: Path = LTSTDB_CACHE, log=print,
              extra_args: list[str] | None = None) -> list[str]:
    """Run every case without a result; returns the ids that failed (no result file)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    todo = [c for c in cases if not done(out, c["candidate_id"])]

    def run(c: dict) -> str | None:
        cmd = [sys.executable, str(RSI_DIR / "run_case.py"), "--harness", str(harness_dir), "--record", c["record"],
               "--candidate", c["candidate_id"], "--out", str(out), "--data", str(data), "--cache", str(cache),
               "--backend", backend] + (extra_args or [])
        with (out / "logs" / f"{c['candidate_id']}.log").open("a") as f:
            subprocess.run(cmd, cwd=CODE_DIR, stdout=f, stderr=subprocess.STDOUT, env=env)
        return None if done(out, c["candidate_id"]) else c["candidate_id"]

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        failed = [x for x in ex.map(run, todo) if x]
    if todo:
        log(f"ran {len(todo)} cases in {(time.time() - t0) / 60:.1f} min; {len(failed)} without a result")
    return failed
