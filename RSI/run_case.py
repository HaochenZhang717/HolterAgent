"""Review one LTSTDB candidate with a harness (called by evaluate.py, one process per case).

Backends for the inner model (GPT-6 Astra, effort high, fixed): "pool" (default from 2026-10-06), direct HTTP requests to
a shared OpenAI Responses-API endpoint with AgentRunner's session loop; "codex", one `codex exec` per session with ChatGPT
credits (rounds 0-5); "openai", an open-weight model behind an OpenAI-compatible server (vLLM; RSI tracks
from 2026-10-17, see RSI/tracks.json), with the same session loop.

    python RSI/run_case.py --harness RSI/runs/search/h00_r1/harness/h00 --record s30771 \
        --candidate s30771-B-002330 --out RSI/runs/search/h00_r1

Writes <out>/<candidate>.json (same layout as scripts/run_review.py, plus the harness id and digest) and
<out>/<candidate>.trace.jsonl with the images under <out>/images/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.agent.codex_runner import CodexConfig, credits  # noqa: E402
from holter_agent.bench import predictions_from_state  # noqa: E402

from rsi.harness import Harness  # noqa: E402
from rsi.knowledge import patient, split_patients  # noqa: E402
from rsi.paths import LTSTDB_CACHE, LTSTDB_DIR  # noqa: E402
from rsi.runner import HarnessRunner, PoolHarnessRunner  # noqa: E402
from holter_agent.agent.pool import PoolBackend  # noqa: E402
from holter_agent.agent.llm import OpenAICompatBackend  # noqa: E402

MODEL, EFFORT = "gpt-6-astra", "high"  # fixed for the whole search
SERVER_WAIT_S = 6 * 3600  # openai backend: how long a case waits for its server to come back


def server_up(base_url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return model in r.read().decode(errors="replace")
    except (OSError, ValueError):
        return False


def wait_for_server(base_url: str, model: str, limit_s: float = SERVER_WAIT_S) -> bool:
    """Block until the server lists the model (True) or limit_s has passed (False)."""
    t0 = time.time()
    while not server_up(base_url, model):
        if time.time() - t0 > limit_s:
            return False
        time.sleep(30)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="harness directory")
    ap.add_argument("--record", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(LTSTDB_DIR))
    ap.add_argument("--cache", default=str(LTSTDB_CACHE))
    ap.add_argument("--codex-version", default=None, help="recorded with the result")
    ap.add_argument("--backend", choices=["pool", "codex", "openai"], default="pool")
    ap.add_argument("--model", default=None, help="openai backend: the served model id; pool backend: a pool model "
                                                  "(default gpt-6-astra)")
    ap.add_argument("--effort", default=None, help="pool backend: reasoning effort (default high)")
    ap.add_argument("--base-url", default=None, help="openai backend: e.g. http://localhost:8100/v1")
    ap.add_argument("--max-tokens", type=int, default=32768, help="openai backend: per model call")
    ap.add_argument("--api-key-file", default=None, help="openai backend: file holding the API key (never logged)")
    ap.add_argument("--extra-body", default="{}", help="openai backend: JSON added to every step request")
    ap.add_argument("--readout-body", default=None, help="openai backend: JSON for the yes/no read-out (thinking off)")
    ap.add_argument("--sampling", default=None, help="openai backend: JSON sampling settings replacing the Qwen "
                                                     "thinking-mode defaults (e.g. greedy decoding)")
    ap.add_argument("--readout-top-k", type=int, default=20)
    ap.add_argument("--readout-max-tokens", type=int, default=1)
    ap.add_argument("--no-server-wait", action="store_true",
                    help="openai backend: a hosted API without a health check; retry on 429/5xx instead of waiting")
    ap.add_argument("--logit", action="store_true",
                    help="openai backend: score the case by the yes/no log-probabilities read after the judgement "
                         "session (p_ischemic of the prediction; the submitted decision is kept as p_decision)")
    ap.add_argument("--kb-exclude-split", default=None,
                    help="harnesses with a knowledge base: the split under evaluation, whose patients are removed "
                         "from the precedents (required for precedents)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    harness = Harness.load(args.harness)
    toolkit = HolterToolkit(args.data, args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / f"{args.candidate}.trace.jsonl"
    model, effort = MODEL, EFFORT
    if args.backend == "pool":
        model, effort = args.model or MODEL, args.effort or EFFORT  # pool model tracks (2026-10-19)
        backend = PoolBackend(model, effort)
        runner = PoolHarnessRunner(toolkit, harness, backend, trace_path=trace)
    elif args.backend == "openai":
        if not (args.model and args.base_url):
            sys.exit("the openai backend needs --model and --base-url")
        model, effort = args.model, "thinking"
        key = Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else "EMPTY"
        backend = OpenAICompatBackend(args.model, args.base_url, api_key=key, max_tokens=args.max_tokens,
                                      extra_body=json.loads(args.extra_body),
                                      sampling=json.loads(args.sampling) if args.sampling else None,
                                      readout_body=json.loads(args.readout_body) if args.readout_body else None,
                                      readout_top_k=args.readout_top_k, readout_max_tokens=args.readout_max_tokens)
        usage_total: dict = {}
        step = backend.step
        if not args.no_server_wait:
            wait_for_server(args.base_url, args.model)

        def counted(*a, **k):  # sum the token usage of every call, like PoolBackend.usage_total
            for attempt in range(8):
                try:
                    turn = step(*a, **k)
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError, RuntimeError) as e:
                    # the server went away (preempted pod, dropped tunnel), is rate limited (429) or failed this
                    # request (5xx): wait, then repeat the same call; give up after a few failures
                    msg = str(e)
                    limited = isinstance(e, RuntimeError) and ("returned 429" in msg or "returned 5" in msg)
                    if attempt == 7 or (isinstance(e, RuntimeError) and not limited):
                        raise
                    if not isinstance(e, RuntimeError) and not args.no_server_wait and \
                            not wait_for_server(args.base_url, args.model):
                        raise
                    time.sleep(min(15 * 2 ** attempt, 300))
            for key, v in (turn.usage or {}).items():
                if v is not None:
                    usage_total[key] = usage_total.get(key, 0) + v
            return turn
        backend.step = counted
        readout = backend.yes_no

        def readout_retried(*a, **k):  # the read-out is one more request: retry it like a step
            for attempt in range(6):
                try:
                    return readout(*a, **k)
                except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                    if attempt == 5 or (isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504)):
                        raise
                    time.sleep(min(15 * 2 ** attempt, 300))
        backend.yes_no = readout_retried
        runner = PoolHarnessRunner(toolkit, harness, backend, trace_path=trace)
        if args.logit:
            runner.logit_scores = {}
    else:
        model, effort = args.model or MODEL, args.effort or EFFORT  # codex backend with another model (2026-11-07)
        runner = HarnessRunner(toolkit, harness, CodexConfig(model=model, effort=effort), trace_path=trace)
    name = toolkit.store.resolve(args.record)
    kb = harness.knowledge
    if kb is not None and kb["precedents"]:
        if not args.kb_exclude_split:
            sys.exit("this harness shows precedents: give --kb-exclude-split <the evaluated split>")
        runner.kb_exclude_patients = split_patients(args.kb_exclude_split) | {patient(name)}
    state = runner.run_record(name, [args.candidate])
    result = {
        "method": "agent",
        "harness": {"id": harness.id, "digest": harness.digest()},
        "settings": {"record": name, "candidate": args.candidate, "model": model, "effort": effort,
                     "codex_version": args.codex_version, "backend": args.backend,
                     "kb_exclude_split": args.kb_exclude_split},
        "run_config": asdict(harness.run_config()),
        "feature_config": toolkit.cfg.to_dict(),
        "record_hours": toolkit.store.get(name).duration_s / 3600.0,
        "state": state.to_dict(),
        "predictions": [p.__dict__ for p in predictions_from_state(state)],
    }
    if args.backend == "openai":  # no credits: open weights on our own GPUs; token usage is kept
        result["openai"] = {"usage": usage_total, "base_url": args.base_url}
        if args.logit:
            result["logit"] = runner.logit_scores
            for p in result["predictions"]:
                r = runner.logit_scores.get(p["candidate_id"]) or {}
                p["p_decision"], p["first_p_decision"] = p.get("p_ischemic"), p.get("first_p_ischemic")
                if r.get("p_yes") is not None:  # the score; without a read-out the decision stays the score
                    p["p_ischemic"] = p["first_p_ischemic"] = r["p_yes"]
            result["settings"]["score"] = "logit_yes_no"
    elif args.backend == "pool":  # not ChatGPT credits; the credit equivalent at Codex rates is kept for cost comparisons
        result["pool"] = {"usage": backend.usage_total, "credits_equivalent": credits(model, backend.usage_total)}
    else:
        result["codex"] = {"usage": dict(runner.usage), "credits": credits(MODEL, runner.usage), "threads": runner.threads}
    if not any(p["candidate_id"] == args.candidate for p in result["predictions"]):
        sys.exit(f"{args.candidate} is not a candidate of {name} under the run's protocol / global reference")
    (out / f"{args.candidate}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = result.get("codex", {}).get("credits") or result.get("pool", {}).get("credits_equivalent")
    print(f"{args.candidate}: {len(state.assessments)} assessments, credits (or equivalent) {cost}")


if __name__ == "__main__":
    main()
