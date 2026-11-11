"""Run a harness on one split with GPT-6 Astra through the Codex CLI, in parallel, and score it.

    python RSI/evaluate.py h00 --split search --rep 1 --concurrency 10
    python RSI/evaluate.py h00 --split search --limit 2          # smoke test (first 2 cases)

- The harness directory is copied to <run>/harness/<id>/ and every case runs from that copy, so a run always
  belongs to one exact harness (its digest is checked when a run is resumed).
- Cases with a result are skipped, so an interrupted run can be resumed with the same command.
- Spending: no new case is started once the run reaches --credit-cap or the budget left for this split
  (budget.json: search and validation runs may not touch the reserve for the final evaluation).
- Output: RSI/runs/<split>/<id>_r<rep>/ with per-case results, traces, images, logs and metrics.json.
  For the search split, case summaries go to RSI/workspace/failures/<id>_r<rep>/. For every other split only
  aggregate numbers are printed; the outer loop does not read the per-case files of those splits.
- Model tracks (2026-10-17): with RSI_TRACK=<name> set, the task model, backend and servers come from
  RSI/tracks.json (e.g. q27: Qwen3.5-27B on vLLM, --backend openai, cases spread round-robin over the servers), and
  runs go to RSI/runs_<name>/ and summaries to RSI/workspace/failures_<name>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsi import scoring, summaries  # noqa: E402
from rsi.budget import allowance  # noqa: E402
from rsi.harness import Harness, digest  # noqa: E402
from rsi.paths import (CODE_DIR, FAILURES_DIR, LTSTDB_CACHE, LTSTDB_DIR, OPEN_SPLITS, RSI_DIR, SPLITS, SPLITS_DIR,  # noqa: E402
                       TRACK, run_dir, track_config)

CODEX_VERSION = "0.155.0"


def codex_version() -> str:
    out = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout.strip()
    return out.split()[-1] if out else "unknown"


def spent(out_dir: Path) -> float:
    total = 0.0
    for f in out_dir.glob("*.json"):
        if f.name != "metrics.json":
            d = json.loads(f.read_text())  # Codex CLI credits, or the credit equivalent of a pool run
            total += (d.get("codex") or {}).get("credits") or (d.get("pool") or {}).get("credits_equivalent") or 0.0
    return total


def score(split: str, out: Path, cases: list[dict], harness_id: str, rep: int, extra: dict) -> dict:
    run = scoring.load_run(out)
    m = scoring.metrics(cases, run) | {"harness": harness_id, "split": split, "rep": rep} | extra
    (out / "metrics.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("harness", help="harness id (directory under RSI/harnesses)")
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--credit-cap", type=float, default=None, help="at most this much for this run")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    ap.add_argument("--score-only", action="store_true", help="do not run cases, only (re)score")
    ap.add_argument("--backend", choices=["codex", "pool", "openai"], default="codex",
                    help="codex: the Codex CLI on ChatGPT credits (rounds 0-5); pool: a shared Responses-API endpoint over HTTP; "
                         "openai: the open-weight model of the RSI_TRACK (set automatically when RSI_TRACK is set)")
    ap.add_argument("--data", default=str(LTSTDB_DIR))
    ap.add_argument("--cache", default=str(LTSTDB_CACHE))
    args = ap.parse_args()
    from rsi.paths import SPLIT_PROTOCOL
    os.environ["CANDIDATE_PROTOCOL"] = SPLIT_PROTOCOL.get(args.split, "B")  # children inherit it (rsi/paths.py)
    track = track_config()
    if track:
        args.backend = track["backend"]
    if args.backend == "openai" and not track:
        sys.exit("the openai backend needs RSI_TRACK naming a track in RSI/tracks.json")

    version = codex_version() if args.backend == "codex" else "n/a"
    # a codex model track (2026-11-07, aspect ablation) records the installed
    # codex-cli version instead of requiring the pinned one of rounds 0-5
    if args.backend == "codex" and version != CODEX_VERSION and not (track and track.get("codex_unpinned")):
        sys.exit(f"codex-cli {version} found; the search is pinned to {CODEX_VERSION}")
    harness = Harness.load(args.harness)  # validates
    cases = scoring.load_cases(SPLITS_DIR / f"{args.split}.json")[:args.limit]
    out = run_dir(args.split, harness.id, args.rep)
    snap = out / "harness" / harness.id
    if snap.exists():
        if digest(snap) != harness.digest():
            sys.exit(f"{out} was run with a different version of {harness.id}; use another --rep")
    else:
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(harness.path, snap, ignore=shutil.ignore_patterns("__pycache__"))
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    open_split = args.split in OPEN_SPLITS
    todo = [] if args.score_only else [c for c in cases if not (out / f"{c['candidate_id']}.json").exists()]
    budget_left = allowance(args.split)
    cap = budget_left if args.credit_cap is None else min(args.credit_cap, budget_left)
    print(f"{harness.id} ({harness.digest()}) on {args.split} rep {args.rep}: {len(cases)} cases, {len(todo)} to run; "
          f"cap for this run {cap:,.0f} credits (budget left for this split {budget_left:,.0f})", flush=True)
    if todo and cap <= 0:
        sys.exit("no budget left for this split")

    urls = list(track.get("base_urls", []))
    slot = {"n": 0}

    def run(c: dict) -> tuple[str, int]:
        cmd = [sys.executable, str(RSI_DIR / "run_case.py"), "--harness", str(snap), "--record", c["record"],
               "--candidate", c["candidate_id"], "--out", str(out), "--data", args.data, "--cache", args.cache,
               "--codex-version", version, "--backend", args.backend, "--kb-exclude-split", args.split]
        if track and track["backend"] in ("pool", "codex"):  # a pool or codex model track: model and effort from the track
            cmd += ["--model", track["model"], "--effort", track.get("effort", "high")]
        elif track:  # spread cases over the servers of the track that answer now (wait while none does)
            live = urls
            if track.get("health_check", True):
                from run_case import server_up
                while True:
                    live = [u for u in urls if server_up(u, track["model"])]
                    if live:
                        break
                    time.sleep(30)
            url = live[slot["n"] % len(live)]
            slot["n"] += 1
            cmd += ["--model", track["model"], "--base-url", url, "--max-tokens", str(track.get("max_tokens", 32768)),
                    "--extra-body", json.dumps(track.get("extra_body", {})),
                    "--readout-top-k", str(track.get("readout_top_k", 20)),
                    "--readout-max-tokens", str(track.get("readout_max_tokens", 1))]
            if track.get("sampling") is not None:
                cmd += ["--sampling", json.dumps(track["sampling"])]
            if track.get("readout_body") is not None:
                cmd += ["--readout-body", json.dumps(track["readout_body"])]
            if track.get("api_key_file"):
                cmd += ["--api-key-file", track["api_key_file"]]
            if not track.get("health_check", True):
                cmd.append("--no-server-wait")
            if track.get("score") == "logit":
                cmd.append("--logit")
        with (logs / f"{c['candidate_id']}.log").open("a") as log:  # appended, so a failed attempt's error stays
            log.write(f"=== attempt {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.flush()
            return c["candidate_id"], subprocess.run(cmd, cwd=CODE_DIR, stdout=log, stderr=subprocess.STDOUT,
                                                     env=env).returncode

    env = None
    if args.backend == "pool" and todo:
        from rsi.evaluation import pool_env

        env = pool_env()  # the key is fetched once and passed to the children, never written down
    t0, done, failed = time.time(), 0, []
    start_spent = spent(out)
    with ThreadPoolExecutor(args.concurrency) as ex:
        pending, queue = set(), list(todo)
        while queue or pending:
            while queue and len(pending) < args.concurrency:
                if spent(out) - start_spent >= cap:
                    print(f"credit cap reached; not starting {len(queue)} more cases", flush=True)
                    queue = []
                    break
                pending.add(ex.submit(run, queue.pop(0)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                cid, rc = fut.result()
                done += 1
                if rc != 0:
                    failed.append(cid)
                total = spent(out)
                name = cid if open_split else "case"
                print(f"[{time.strftime('%H:%M')}] {'done' if rc == 0 else 'FAILED'} {name} ({done}/{len(todo)}); "
                      f"run credits {total:,.0f}; {(time.time() - t0) / 60:.0f} min", flush=True)
    m = score(args.split, out, cases, harness.id, args.rep,
              {"digest": harness.digest(), "codex_version": version,
               "model": track.get("model", "gpt-6-astra"), "effort": (track.get("effort", "high") if track.get("backend") == "pool" else "thinking") if track else "high",
               "track": TRACK,
               "n_failed_processes": len(failed), "minutes": round((time.time() - t0) / 60, 1),
               "time": time.strftime("%Y-%m-%d %H:%M")})
    keys = ("n_cases", "n_run", "n_without_p", "auc", "auc_ci95", "se_at_0.5", "sp_at_0.5", "n_indeterminate_data",
            "credits_total", "credits_per_case", "analysis_calls_per_case", "verified_fraction", "auc_first_assessment",
            "n_failed_processes", "minutes")
    print(json.dumps({k: m[k] for k in keys if k in m}, indent=1))
    if open_split:
        if failed:
            print("failed processes:", " ".join(failed))
        idx = summaries.write(out, cases, FAILURES_DIR / out.name)
        print(f"case summaries: {idx}")


if __name__ == "__main__":
    main()
