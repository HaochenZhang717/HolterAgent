"""ECG-Scroll Judge agent and Rule Agent with signal tools only ("-dx") on the LTSTDB candidates (baseline B7).

Source: Li H, Li C, Ding Z, Li Z, Mao Y, Huang Z. "ECG-Scroll: A Long-Horizon, Streaming Benchmark and Agent
Environment for Interpretation of Ambulatory Electrocardiograms", arXiv:2609.33117 (2026). Code: github.com/CuCl-2/
ECG-Scroll (MIT), commit 4db14ef, vendored unchanged in baselines/third_party/ecg_scroll (with its LICENSE).

Their Judge agent (ecgscroll/run_llm_fc.py, paper Section 5 and Appendix C): the environment streams a recording in
30-s chunks through a monotonic cursor (no future signal), measures every chunk with the environment caliper
(ecgscroll/env.py: R peaks by scipy find_peaks, ST level per lead = mean over beats of x[R+120 ms] - x[R-40 ms]; the
ischemia cue of a chunk is st_mv = max over leads of |ST|), shows the model the chunk lines in time-ordered batches of
60 and asks for one flag_chunks(indices) call per batch (tool_choice "required", temperature 0.2, max_tokens 2048,
thinking disabled). Their Rule Agent (ecgscroll/run_stream.py) walks the same chunks and flags a chunk when
st_mv > 0.10 mV. Flagged chunks are merged into intervals (ecgscroll/chunking.assemble_answer). We use only their
"signal" tier (-dx). The diagnostic tools (+dx) read the reference annotations and are an oracle; they are replaced
here by a function that raises, so they cannot be used by accident.

What this port changes; everything else is their code, called unchanged (run_llm_fc.run_one, chunking.scan_chunks,
run_stream.stream_agent, the prompts, the flag_chunks tool, the caliper and the batch loop):
1. Data source: their reader (ecgscroll/render.py) reads our local LTSTDB records (paths.raw_path redirected for the
   pseudo database "ltstdb"); all 2-3 leads of each record, 250 Hz, no extra filtering (as in their code).
2. Framing: one stream per candidate, [start - X, end + X] clipped to the record, X = 1800 s fixed before any run
   (one Judge batch of 60 chunks of context on each side, the same +-30 min as the ST trend our agent sees). The
   sample is their ischemia_detection instance format (lead = -1: all leads, the agent picks); its answer is an
   empty placeholder, never the reference, and their end-of-stream verifier score is discarded.
3. Leads: st_mv = max |ST| over all leads of the record; their _abs_st hard-codes leads 0 and 1 of the two-lead
   European ST-T records, which would ignore the third lead of 17 of the 61 LTSTDB records. The Judge prompt itself
   defines st_mv as "max over leads".
4. Transport: their driver calls openai.OpenAI(...).chat.completions.create; the openai package is not installed
   here, so ChatClient sends the identical request body (extra_body merged as the SDK does) with requests; when the
   server is down (preempted pod, tunnel drop) it polls GET <base_url>/models every 60 s and retries the same request
   instead of silently skipping the batch as their driver does on an exception. PoolClient sends the same
   system prompt, user message and flag_chunks tool to GPT-6 Astra through a Responses-API endpoint (Responses API,
   tool_choice "required", reasoning effort "low" as the closest to "thinking disabled"; no temperature or
   max_tokens, which the reasoning API does not take in the same way).

Candidate score, fixed before any run (their method labels chunks, we score candidates):
  inside chunks = chunks whose midpoint lies in [start, end]; context chunks = chunks not overlapping (start, end).
  primary     frac_in  = fraction of inside chunks flagged;
  alternative contrast = frac_in - fraction of context chunks flagged (does the candidate stand out of its +-X
              surroundings; the paper reports that the -dx Judge flags nearly every chunk);
  the mapping with the higher development AUC is used for the test set (choose step, development labels only).
  label (the method's own decision): their merged flagged intervals contain one with temporal IoU >= 0.75 with
  [start, end], the overlap their ischemia verifier requires to count an episode as detected.

    python -m baselines.ecgscroll_port rule                                   # Rule Agent, dev + test (CPU)
    python -m baselines.ecgscroll_port judge --arm qwen27b --split dev        # Qwen3.5-27B on localhost:8000
    python -m baselines.ecgscroll_port judge --arm qwen35a3b --split dev      # Qwen3.5-35B-A3B on localhost:8001
    python -m baselines.ecgscroll_port judge --arm gpt6astra --pool-cost-cases 10   # cost probe, 10 dev cases
    python -m baselines.ecgscroll_port choose --arm qwen27b                   # mapping from development AUC
    python -m baselines.ecgscroll_port judge --arm qwen27b --split test       # test, once, after choose
    python -m baselines.ecgscroll_port score --arm qwen27b                    # outputs/baselines/ecgscroll_judge_qwen27b
    python -m baselines.ecgscroll_port pipeline --arm qwen35a3b               # judge dev, choose, judge test, score

Per-candidate outputs: outputs/baselines/_runs/ecgscroll_<arm>/<candidate>.json (chunks, flags, model calls, usage).
"""
from __future__ import os
import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import requests

from baselines.common import OUT, ROOT, TEST_SETS, auc, data_dirs, load_cases

THIRD = Path(__file__).resolve().parent / "third_party" / "ecg_scroll"
ECGSCROLL_COMMIT = "4db14ef59d8ee78de785c12a7d4644c9c8e3dc0f"
DATA = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data")) / "ltstdb"
RUNS = OUT / "_runs"
CASES = ROOT / "outputs" / "ltstdb_cases"
PN_DIR = "ltstdb"  # pseudo PhysioNet database name of our local records inside their reader

# Fixed before any run (see the module docstring).
X_S = 1800.0          # context on each side of the candidate
CHUNK_S = 30.0        # their default chunk (paper Section 3.2; run_llm_fc --chunk_s)
BATCH = 60            # their Judge batch (scripts/run_v7.py --batch 60)
TEMPERATURE = 0.2     # run_llm_fc default
TIER = "signal"       # -dx only
IOU_DETECT = 0.75     # their ischemia verifier threshold (paper Table 1, Appendix B)
POOL_MODEL, POOL_EFFORT = "gpt-6-astra", "low"
MAPPINGS = ("frac_in", "contrast")  # primary first

ARMS = {
    "qwen27b": {"backend": "openai", "model": "Qwen/Qwen3.5-27B", "base_url": "http://localhost:8000/v1",
                "row": "ecgscroll_judge_qwen27b"},
    "qwen35a3b": {"backend": "openai", "model": "Qwen/Qwen3.5-35B-A3B", "base_url": "http://localhost:8001/v1",
                  "row": "ecgscroll_judge_qwen35a3b"},
    # Test set only (user decision 2026-10-12); the mapping is pre-specified as the Rule Agent's development choice.
    "gpt6astra": {"backend": "pool", "model": POOL_MODEL, "row": "ecgscroll_judge_gpt6astra", "test_only": True},
    # External evaluation (2026-10-23): the agent's model; test sets only, mapping pre-specified as for gpt6astra.
    "luna": {"backend": "pool", "model": "gpt-5.6-luna", "row": "ecgscroll_judge_luna", "test_only": True},
    "rule": {"backend": None, "model": None, "row": "ecgscroll_rule"},
}

_INSTALLED = False


# ------------------------------------------------------------------------------------------- their code, adapted
def _abs_st_all_leads(meas: dict) -> float:
    """max |ST| (mV) over every lead the caliper measured (their _abs_st: leads 0 and 1 only)."""
    vals = [abs(float(v)) for k, v in meas.items()
            if k.startswith("st_mv_lead") and v is not None and np.isfinite(v)]
    return max(vals) if vals else 0.0


def _oracle_disabled(*_a, **_k):
    raise RuntimeError("ECG-Scroll diagnostic (+dx) tools read the reference annotations; disabled in this port")


def install(data_dir: Path = DATA) -> SimpleNamespace:
    """Import their package with the overrides of the module docstring (once per process)."""
    global _INSTALLED
    if str(THIRD) not in sys.path:
        sys.path.insert(0, str(THIRD))
    try:  # run_llm_fc imports openai.OpenAI at module level; we pass our own client to run_one
        import openai  # noqa: F401
    except ImportError:
        stub = ModuleType("openai")
        stub.OpenAI = lambda base_url, api_key="EMPTY", **_: ChatClient(base_url, api_key)
        sys.modules["openai"] = stub
    from ecgscroll import chunking, paths, render, run_llm_fc, run_stream, tools
    if not _INSTALLED:
        orig_raw_path = paths.raw_path

        def raw_path(pn_dir: str, record: str = "") -> str:
            if pn_dir != PN_DIR:
                return orig_raw_path(pn_dir, record)
            # external evaluation (2026-10-23): each record's own data directory (common.data_dirs)
            return str((data_dirs(record)[0] if TEST_SETS else Path(data_dir)) / record)

        paths.raw_path = raw_path
        chunking._abs_st = _abs_st_all_leads
        run_stream._abs_st = _abs_st_all_leads
        for name in ("classify_rhythm", "classify_beats", "flag_ischemia", "diagnostic_for",
                     "_af_eps", "_beat_ann", "_st_eps"):
            setattr(tools, name, _oracle_disabled)
        tools.DIAGNOSTIC_TOOLS = {}
        _INSTALLED = True
    return SimpleNamespace(chunking=chunking, render=render, run_llm_fc=run_llm_fc, run_stream=run_stream)


# ------------------------------------------------------------------------------------------- candidates, streams
def load_candidates() -> list[dict]:
    """Candidates of the patient-level split in load_cases order: id, record, split, start_s, end_s (no reference)."""
    files = [CASES / "dev_v07.json"] + ([ROOT / "RSI" / "splits" / f"{n}.json" for n in TEST_SETS] if TEST_SETS
                                         else [CASES / "test_v07.json"])
    raw = {c["candidate_id"]: c for f in files for c in json.loads(f.read_text())["cases"]}
    return [{"candidate_id": c.candidate_id, "record": c.record, "split": c.split,
             "start_s": float(raw[c.candidate_id]["start_s"]), "end_s": float(raw[c.candidate_id]["end_s"])}
            for c in load_cases()]


def record_duration(record: str, data_dir: Path = DATA) -> float:
    import wfdb
    h = wfdb.rdheader(str(Path(data_dir) / record))
    return h.sig_len / h.fs


def make_sample(cand: dict, duration_s: float) -> dict:
    """Their ischemia_detection instance for the stream around one candidate. The answer is a placeholder."""
    t0 = max(0.0, cand["start_s"] - X_S)
    t1 = min(duration_s, cand["end_s"] + X_S)
    return {"id": cand["candidate_id"], "task": "ischemia_detection", "record": cand["record"], "pn_dir": PN_DIR,
            "lead": -1, "t0": t0, "t1": t1, "answer": {"intervals": []},
            "verifier": {"type": "temporal_f1", "params": {"tau": IOU_DETECT}}}


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def candidate_scores(chunks: list[list[float]], flagged: set[int], start: float, end: float,
                     intervals: list[list[float]]) -> dict:
    """The pre-specified chunk-to-candidate mapping. chunks: [i, t0, t1, st_mv]."""
    inside = [int(c[0]) for c in chunks if start <= (c[1] + c[2]) / 2 <= end]
    if not inside:  # cannot happen for candidates >= 30 s inside the stream; kept for safety
        inside = [int(c[0]) for c in chunks if min(c[2], end) - max(c[1], start) > 0]
    context = [int(c[0]) for c in chunks if min(c[2], end) - max(c[1], start) <= 0]
    f_in = float(np.mean([i in flagged for i in inside])) if inside else 0.0
    f_ctx = float(np.mean([i in flagged for i in context])) if context else 0.0
    best = max((iou((a, b), (start, end)) for a, b in intervals), default=0.0)
    return {"frac_in": f_in, "frac_context": f_ctx, "contrast": f_in - f_ctx, "best_iou": round(best, 4),
            "detected": bool(best >= IOU_DETECT), "n_inside": len(inside), "n_context": len(context)}


def _chunk_rows(chunks: list[dict]) -> list[list[float]]:
    return [[c["i"], c["t0"], c["t1"], c.get("st_mv")] for c in chunks]


def _flags_from_intervals(chunks: list[dict], intervals: list[list[float]]) -> set[int]:
    """Chunk indices of flagged (t0, t1) intervals (their ledger / flagged_intervals hold the chunk bounds)."""
    out = set()
    for a, b in intervals:
        # nearest chunk within 0.1 s at both ends (2026-10-23: rounding both to one decimal failed for the last,
        # end-aligned chunk of short records, e.g. 810.448 s vs a flagged 810.45 s)
        err, i = min(((abs(c["t0"] - float(a)) + abs(c["t1"] - float(b)), c["i"]) for c in chunks),
                     default=(float("inf"), None))
        if i is None or err > 0.2:
            raise ValueError(f"flagged interval {a}-{b} is not a chunk")
        out.add(i)
    return out


# ------------------------------------------------------------------------------------------- model transport
class _Recorder:
    """Per-candidate call log shared by both clients."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):  # pragma: no cover - overridden
        raise NotImplementedError


def _response(arguments: list[str]):
    """The object shape run_llm_fc.run_one reads: r.choices[0].message.tool_calls[k].function.arguments."""
    calls = [SimpleNamespace(function=SimpleNamespace(name="flag_chunks", arguments=a)) for a in arguments]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls or None))])


def server_up(base_url: str, model: str | None = None) -> bool:
    """GET <base_url>/models answers 200 (and lists `model`, when given)."""
    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=10, proxies={"http": None, "https": None})
        return r.status_code == 200 and (model is None or any(m.get("id") == model for m in r.json().get("data", [])))
    except (requests.RequestException, ValueError):
        return False


def wait_for_server(base_url: str, model: str | None = None, poll_s: float = 60.0,
                    max_wait_s: float = 12 * 3600) -> bool:
    """Poll until the vLLM server answers again (GPU servers may be preempted and relaunched; baselines/direct_llm.py)."""
    t0 = time.time()
    while not server_up(base_url, model):
        if time.time() - t0 > max_wait_s:
            return False
        time.sleep(poll_s)
    return True


class ChatClient(_Recorder):
    """Stand-in for openai.OpenAI(base_url=...): POST /chat/completions with the same body.
    When the server is down (preempted pod, tunnel drop) it waits for it and retries the same request, so a batch is
    never skipped; errors while the server is up are retried a few times with backoff."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout_s: float = 600.0, retries: int = 5,
                 max_wait_s: float = 12 * 3600):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.timeout_s, self.retries, self.max_wait_s = timeout_s, retries, max_wait_s
        self.session = requests.Session()
        self.session.trust_env = False  # localhost tunnel: no proxy
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def create(self, model, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None,
               extra_body=None, **_):
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                "temperature": temperature, "max_tokens": max_tokens} | (extra_body or {})
        rec: dict = {"t": time.time()}
        self.calls.append(rec)
        attempt = 0
        while True:
            try:
                r = self.session.post(self.url, json=body, timeout=(15, self.timeout_s))
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.ConnectionError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    rec["error"] = f"HTTP {r.status_code}: {r.text[:300]}"
                    raise RuntimeError(rec["error"])
                d = r.json()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                rec.setdefault("transport_errors", []).append(f"{type(e).__name__}: {str(e)[:200]}")
                if not server_up(self.base_url, model):  # down: wait for it, then retry the same request
                    t_wait = time.time()
                    ok = wait_for_server(self.base_url, model, max_wait_s=self.max_wait_s)
                    rec["server_wait_s"] = round(rec.get("server_wait_s", 0.0) + time.time() - t_wait, 1)
                    if ok:
                        continue
                    rec["error"] = f"server down for more than {self.max_wait_s:.0f} s"
                    raise
                attempt += 1  # up but failing (overload, timeout): a bounded number of retries
                if attempt > self.retries:
                    rec["error"] = "transport: " + rec["transport_errors"][-1]
                    raise
                time.sleep(min(5 * 2 ** attempt, 300))
        msg = d["choices"][0]["message"]
        args = [tc["function"].get("arguments") or "{}" for tc in (msg.get("tool_calls") or [])]
        u = d.get("usage") or {}
        rec.update(latency_s=round(time.time() - rec["t"], 2), arguments=args, content=(msg.get("content") or "")[:500],
                   finish_reason=d["choices"][0].get("finish_reason"),
                   usage={"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)})
        return _response(args)


class PoolClient(_Recorder):
    """The same Judge request to GPT-6 Astra through a Responses-API endpoint (Responses API, SSE)."""

    def __init__(self, model: str = POOL_MODEL, effort: str = POOL_EFFORT, retries: int = 5):
        super().__init__()
        from holter_agent.agent.pool import PoolBackend
        self.backend = PoolBackend(model=model, effort=effort, retries=retries)
        self.model, self.effort, self.retries = model, effort, retries

    def create(self, model, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None,
               extra_body=None, **_):
        from holter_agent.agent.pool import PoolError
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = [m for m in messages if m["role"] != "system"]
        payload = {"model": self.model, "instructions": system,
                   "input": [{"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]}
                             for m in user],
                   "reasoning": {"effort": self.effort}, "store": False, "stream": True,
                   "tools": [{"type": "function", "name": t["function"]["name"],
                              "description": t["function"].get("description", ""),
                              "parameters": t["function"]["parameters"], "strict": False} for t in (tools or [])],
                   "tool_choice": tool_choice or "auto", "parallel_tool_calls": False}
        rec: dict = {"t": time.time()}
        self.calls.append(rec)
        for attempt in range(self.retries + 1):
            try:
                items, usage = self.backend._post(payload)
                break
            except (PoolError, requests.RequestException) as e:
                rec.setdefault("transport_errors", []).append(f"{type(e).__name__}: {str(e)[:300]}")
                transient = isinstance(e, requests.RequestException) or any(
                    s in str(e) for s in ("HTTP 429", "HTTP 5", "without a completion", "response failed",
                                          "server_error", "rate_limit", "overloaded", "retry"))
                if attempt == self.retries or not transient:
                    rec["error"] = rec["transport_errors"][-1]
                    raise
                time.sleep(min(10 * 2 ** attempt, 120))
        args = [it.get("arguments") or "{}" for it in items if it.get("type") == "function_call"]
        texts = [c.get("text", "") for it in items if it.get("type") == "message"
                 for c in it.get("content", []) if c.get("type") == "output_text"]
        rec.update(latency_s=round(time.time() - rec["t"], 2), arguments=args, content="\n".join(texts)[:500],
                   usage={"input_tokens": usage.get("input_tokens", 0),
                          "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                          "output_tokens": usage.get("output_tokens", 0),
                          "reasoning_output_tokens":
                              (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)})
        return _response(args)


def _sum_usage(calls: list[dict]) -> dict:
    tot: dict = {}
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            tot[k] = tot.get(k, 0) + int(v or 0)
    return tot


# ------------------------------------------------------------------------------------------- runs
def run_dir(arm: str) -> Path:
    return RUNS / f"ecgscroll_{arm}"


def judge_one(es: SimpleNamespace, cand: dict, duration_s: float, arm: str) -> dict:
    """Their Judge agent (run_llm_fc.run_one, unchanged) over the stream of one candidate."""
    cfg = ARMS[arm]
    sample = make_sample(cand, duration_s)
    client = ChatClient(cfg["base_url"]) if cfg["backend"] == "openai" else PoolClient(cfg["model"])
    t = time.time()
    row = es.run_llm_fc.run_one(sample, client, cfg["model"], tier=TIER, batch=BATCH, chunk_s=CHUNK_S,
                                temperature=TEMPERATURE)
    _, chunks = es.chunking.scan_chunks(sample, chunk_s=CHUNK_S, dx=False)  # deterministic: the chunks run_one saw
    flagged = _flags_from_intervals(chunks, row["flagged_intervals"])
    n_batches = math.ceil(len(chunks) / BATCH)
    unparsed = 0
    for c in client.calls:
        for a in c.get("arguments") or []:
            try:
                json.loads(a)
            except (TypeError, ValueError):
                unparsed += 1
    out = {"candidate_id": cand["candidate_id"], "record": cand["record"], "split": cand["split"],
           "start_s": cand["start_s"], "end_s": cand["end_s"], "stream_s": [sample["t0"], sample["t1"]],
           "arm": arm, "model": cfg["model"], "x_s": X_S, "chunk_s": CHUNK_S, "batch": BATCH,
           "n_chunks": len(chunks), "n_batches": n_batches, "n_calls_ok": row["n_calls"],
           "chunks": _chunk_rows(chunks), "flagged": sorted(flagged), "pred_intervals": row["pred"]["intervals"],
           "calls": [{k: v for k, v in c.items() if k != "t"} for c in client.calls],
           "usage": _sum_usage(client.calls), "n_unparsed_arguments": unparsed,
           "n_no_tool_call": sum(1 for c in client.calls if "error" not in c and not c.get("arguments")),
           "wall_s": round(time.time() - t, 1),
           "scores": candidate_scores(_chunk_rows(chunks), flagged, cand["start_s"], cand["end_s"],
                                      row["pred"]["intervals"])}
    # complete = every batch got a response (a batch whose arguments the model malformed still counts, as in theirs)
    out["complete"] = (len(client.calls) == n_batches and row["n_calls"] == n_batches
                       and all("latency_s" in c and "error" not in c for c in client.calls))
    return out


def rule_one(es: SimpleNamespace, cand: dict, duration_s: float) -> dict:
    """Their Rule Agent (run_stream.stream_agent, -dx, st_thr 0.10) over the stream of one candidate."""
    sample = make_sample(cand, duration_s)
    pred, env = es.run_stream.stream_agent(sample, chunk_s=CHUNK_S, dx=False)
    _, chunks = es.chunking.scan_chunks(sample, chunk_s=CHUNK_S, dx=False)
    flagged = _flags_from_intervals(chunks, [e["interval"] for e in env.ledger])
    return {"candidate_id": cand["candidate_id"], "record": cand["record"], "split": cand["split"],
            "start_s": cand["start_s"], "end_s": cand["end_s"], "stream_s": [sample["t0"], sample["t1"]],
            "arm": "rule", "x_s": X_S, "chunk_s": CHUNK_S, "st_thr_mv": 0.10, "n_chunks": len(chunks),
            "chunks": _chunk_rows(chunks), "flagged": sorted(flagged), "pred_intervals": pred["intervals"],
            "complete": True,
            "scores": candidate_scores(_chunk_rows(chunks), flagged, cand["start_s"], cand["end_s"],
                                       pred["intervals"])}


def run_arm(arm: str, splits: list[str], workers: int, ids: list[str] | None = None, limit: int | None = None,
            data_dir: Path = DATA) -> int:
    """Run an arm record by record (each record is loaded once), skipping candidates already saved.
    Returns the number of candidates left incomplete (rerun to retry them)."""
    es = install(data_dir)
    out_dir = run_dir(arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = [c for c in load_candidates() if c["split"] in splits and (ids is None or c["candidate_id"] in ids)]
    todo = [c for c in cands if not (out_dir / f"{c['candidate_id']}.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    by_rec: OrderedDict[str, list[dict]] = OrderedDict()
    for c in todo:
        by_rec.setdefault(c["record"], []).append(c)
    print(f"[{arm}] {len(cands)} candidates in {splits}, {len(todo)} to run over {len(by_rec)} records", flush=True)
    t_start, n_done, n_fail = time.time(), 0, 0
    tokens = {"input_tokens": 0, "output_tokens": 0}
    for rec, group in by_rec.items():
        dur = record_duration(rec, data_dirs(rec)[0] if TEST_SETS else data_dir)
        es.render.get_signal(rec, PN_DIR)  # load the record once, before the worker threads read it

        def work(c: dict) -> dict:
            try:
                return rule_one(es, c, dur) if arm == "rule" else judge_one(es, c, dur, arm)
            except Exception as e:  # noqa: BLE001 - recorded, the candidate is retried on the next invocation
                return {"candidate_id": c["candidate_id"], "complete": False,
                        "exception": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}

        with ThreadPoolExecutor(max(1, min(workers, len(group)))) as ex:
            results = list(ex.map(work, group))
        for r in results:
            if r.get("complete"):
                (out_dir / f"{r['candidate_id']}.json").write_text(json.dumps(r, indent=1))
                n_done += 1
                for k in tokens:
                    tokens[k] += (r.get("usage") or {}).get(k, 0)
            else:
                n_fail += 1
                fail_dir = out_dir / "_incomplete"
                fail_dir.mkdir(exist_ok=True)
                (fail_dir / f"{r['candidate_id']}.json").write_text(json.dumps(r, indent=1))
        es.render._MEM.pop(rec, None)  # free the record (24 h x 2-3 leads, float32)
        print(f"[{arm}] {rec}: {len(group)} candidates; total done {n_done}, incomplete {n_fail}, "
              f"tokens in/out {tokens['input_tokens']}/{tokens['output_tokens']}, "
              f"{time.time() - t_start:.0f} s", flush=True)
        if n_fail and arm != "rule":  # retries exhausted (server down > 12 h, or failing while up): stop, rerun later
            print(f"[{arm}] stopping after {rec}: candidates failed after all retries", flush=True)
            break
    print(f"[{arm}] finished: {n_done} saved, {n_fail} incomplete (see {out_dir / '_incomplete'})", flush=True)
    return n_fail


# ------------------------------------------------------------------------------------------- scoring
def load_runs(arm: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(run_dir(arm).glob("*.json"))
            if not p.name.startswith("_")}


def choose_mapping(arm: str) -> dict:
    """Pick the chunk-to-candidate mapping on development AUC (primary wins ties); writes _mapping_choice.json."""
    from baselines.learned_common import load_bundle
    b = load_bundle(signals=False)
    runs = load_runs(arm)
    dev = [b.ids[i] for i in b.dev]
    missing = [c for c in dev if c not in runs]
    if missing:
        raise SystemExit(f"{arm}: {len(missing)} development candidates not run yet, e.g. {missing[:3]}")
    y = b.y[b.dev]
    aucs = {m: auc(np.array([runs[c]["scores"][m] for c in dev], float), y) for m in MAPPINGS}
    chosen = max(MAPPINGS, key=lambda m: (round(aucs[m], 12), m == MAPPINGS[0]))
    res = {"arm": arm, "dev_auc": aucs, "chosen": chosen, "rule": "higher development AUC; primary (frac_in) on ties",
           "time": time.strftime("%Y-%m-%d %H:%M")}
    (run_dir(arm) / "_mapping_choice.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def usage_stats(runs: dict[str, dict], ids: list[str]) -> dict:
    rows = [runs[c] for c in ids if c in runs]
    if not rows:
        return {}
    u = [r.get("usage") or {} for r in rows]
    keys = sorted({k for x in u for k in x})
    st = {"n_candidates": len(rows), "n_calls": int(sum(len(r.get("calls", [])) for r in rows)),
          "n_chunks": int(sum(r["n_chunks"] for r in rows)),
          "calls_with_transport_retries": int(sum(1 for r in rows for c in r.get("calls", [])
                                                  if c.get("transport_errors"))),
          "no_tool_call_batches": int(sum(r.get("n_no_tool_call", 0) for r in rows)),
          "unparsed_arguments": int(sum(r.get("n_unparsed_arguments", 0) for r in rows)),
          "flagged_chunk_fraction": float(np.mean([len(r["flagged"]) / max(1, r["n_chunks"]) for r in rows]))}
    for k in keys:
        tot = int(sum(x.get(k, 0) for x in u))
        st[f"{k}_total"] = tot
        st[f"{k}_per_candidate"] = round(tot / len(rows), 1)
    lat = [c["latency_s"] for r in rows for c in r.get("calls", []) if "latency_s" in c]
    if lat:
        st["call_latency_s_median"] = float(np.median(lat))
    return st


def score_arm(arm: str) -> dict:
    """Save outputs/baselines/<row>/ with the mapping chosen on development (requires every candidate)."""
    from baselines.learned_common import finish, load_bundle
    cfg = ARMS[arm]
    if cfg["row"] is None:
        raise SystemExit(f"{arm} is a cost probe only; no row is saved")
    choice_path = run_dir(arm) / "_mapping_choice.json"
    if not choice_path.exists():
        raise SystemExit(f"run `choose --arm {arm}` first (development only)")
    choice = json.loads(choice_path.read_text())
    m = choice["chosen"]
    b = load_bundle(signals=False)
    runs = load_runs(arm)
    dev_ids, test_ids = [b.ids[i] for i in b.dev], [b.ids[i] for i in b.test]
    needed = test_ids if cfg.get("test_only") else b.ids
    missing = [c for c in needed if c not in runs]
    if missing:
        raise SystemExit(f"{arm}: {len(missing)} candidates not run yet, e.g. {missing[:3]}")
    if cfg.get("test_only"):  # no development run: AUC and the method's own label only
        dev_ids = []
    dev_score = {c: float(runs[c]["scores"][m]) for c in dev_ids}
    test_score = {c: float(runs[c]["scores"][m]) for c in test_ids}
    label = {c: bool(runs[c]["scores"]["detected"]) for c in dev_ids + test_ids}
    model = ({"name": cfg["model"], "endpoint": cfg.get("base_url"), "temperature": TEMPERATURE,
              "max_tokens": 2048, "tool_choice": "required", "thinking": "disabled (chat_template_kwargs)"}
             if arm != "rule" else None)
    extra = {
        "method": ("ECG-Scroll Rule Agent, signal tier (-dx): flag a 30-s chunk when max_lead |ST| > 0.10 mV"
                   if arm == "rule" else "ECG-Scroll Judge agent, signal tier (-dx): the model flags 30-s chunks "
                                         "from per-chunk st_mv, 60 chunks per call"),
        "source": "Li et al., arXiv:2609.33117 (2026); github.com/CuCl-2/ECG-Scroll",
        "ecgscroll_commit": ECGSCROLL_COMMIT,
        "window_x_s": X_S, "stream": "[start - X, end + X] clipped to the record, one stream per candidate",
        "chunk_s": CHUNK_S, "batch_chunks": BATCH if arm != "rule" else None, "tier": "signal (-dx)",
        "chunk_cue": "env caliper st_mv = max over all leads of |mean(x[R+120ms] - x[R-40ms])|, R by find_peaks",
        "mapping": {"primary": "frac_in = fraction of chunks with midpoint in [start, end] that are flagged",
                    "alternative": "contrast = frac_in - fraction of chunks not overlapping the candidate flagged",
                    "chosen": m, "dev_auc": choice["dev_auc"], "chosen_on": "development AUC"},
        "label_definition": f"merged flagged chunks contain an interval with temporal IoU >= {IOU_DETECT} with "
                            "[start, end] (their ischemia verifier's match)",
        "model": model,
        "usage_dev": usage_stats(runs, dev_ids) if arm != "rule" else None,
        "usage_test": usage_stats(runs, test_ids) if arm != "rule" else None,
        "flagged_chunk_fraction": float(np.mean([len(runs[c]["flagged"]) / max(1, runs[c]["n_chunks"])
                                                 for c in dev_ids + test_ids])),
        "port_changes": ["data source: their reader on local LTSTDB records (all leads, 250 Hz)",
                         "framing: one stream per candidate [start-X, end+X], X = 1800 s; placeholder answer",
                         "st_mv over all 2-3 leads (their _abs_st: leads 0 and 1)",
                         "transport: requests stand-in for openai.OpenAI with retries on transport errors"],
    }
    notes = (f"ECG-Scroll {'Rule Agent' if arm == 'rule' else 'Judge agent (' + cfg['model'] + ')'}, -dx, "
             f"stream +-{X_S:.0f} s, 30-s chunks; score = {m} (chosen on dev); label = IoU>={IOU_DETECT} detection")
    r = finish(cfg["row"], b, dev_score, test_score, notes, extra=extra, label=label)
    for split in ("dev", "test"):
        s = r["splits"][split]
        print(f"{cfg['row']} {split}: AUC {s['auc']:.3f} [{s['auc_ci'][0]:.3f}, {s['auc_ci'][1]:.3f}]  "
              f"at dev thr: se {s['at_dev_threshold']['se']:.2f} sp {s['at_dev_threshold']['sp']:.2f}  "
              f"label: se {s['rule']['se']:.2f} sp {s['rule']['sp']:.2f}")
    return r


def pool_cost_cases(n: int) -> list[str]:
    """Development candidates for the GPT-6 Astra cost probe: their load_pool stride sample (S[::len(S)//n][:n])."""
    dev = [c["candidate_id"] for c in load_candidates() if c["split"] == "dev"]
    return dev[::max(1, len(dev) // n)][:n]


def pool_cost_summary(ids: list[str]) -> dict:
    from baselines.learned_common import load_bundle
    from holter_agent.agent.codex_runner import credits
    runs = load_runs("gpt6astra")
    b = load_bundle(signals=False)
    ref = dict(zip(b.ids, b.reference))
    rows = [runs[c] for c in ids if c in runs]
    per = []
    for r in rows:
        per.append({"candidate_id": r["candidate_id"], "reference": str(ref[r["candidate_id"]]),
                    "n_calls": len(r["calls"]), "usage": r["usage"],
                    "credits_equivalent": round(credits(POOL_MODEL, r["usage"]) or 0.0, 3),
                    **{k: r["scores"][k] for k in ("frac_in", "frac_context", "contrast", "detected", "best_iou")},
                    "flagged_fraction_of_stream": round(len(r["flagged"]) / max(1, r["n_chunks"]), 3)})
    tot = _sum_usage([{"usage": r["usage"]} for r in rows])
    cr = credits(POOL_MODEL, tot) or 0.0
    out = {"model": POOL_MODEL, "effort": POOL_EFFORT, "n_candidates": len(rows), "usage_total": tot,
           "credits_equivalent_total": round(cr, 2),
           "credits_equivalent_per_candidate": round(cr / max(1, len(rows)), 3),
           "rates_per_1M": "input 250, cached input 25, output 1250 (holter_agent/agent/codex_runner.CREDIT_RATES)",
           "projected_credits_all_821": round(cr / max(1, len(rows)) * 821, 0),
           "per_candidate": per}
    (run_dir("gpt6astra") / "_cost_summary.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rule", help="their Rule Agent (-dx) on every candidate")
    p.add_argument("--split", nargs="+", default=["dev", "test"])
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("judge", help="their Judge agent (-dx) with one backend")
    p.add_argument("--arm", choices=[a for a in ARMS if a != "rule"], required=True)
    p.add_argument("--split", nargs="+", default=["dev"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="run at most this many new candidates (smoke test)")
    p.add_argument("--pool-cost-cases", type=int, default=None, help="gpt6astra: the n development cost cases")
    p = sub.add_parser("pipeline", help="judge dev -> choose -> judge test -> score (stops if anything is incomplete)")
    p.add_argument("--arm", choices=[a for a in ARMS if ARMS[a]["backend"] == "openai"], required=True)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("choose", help="choose the mapping on development AUC")
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p = sub.add_parser("score", help="save the scored row (development + test)")
    p.add_argument("--arm", choices=list(ARMS), required=True)
    a = ap.parse_args()
    if a.cmd == "rule":
        run_arm("rule", a.split, a.workers)
    elif a.cmd == "judge":
        ids = None
        if a.arm == "gpt6astra" and a.split == ["test"]:
            # Test set only, with the mapping pre-specified in _mapping_choice.json (see ARMS).
            if not (run_dir(a.arm) / "_mapping_choice.json").exists():
                raise SystemExit("gpt6astra test run needs the pre-specified _mapping_choice.json")
            run_arm(a.arm, ["test"], min(a.workers, 8))
            return
        if a.arm == "gpt6astra":
            if a.pool_cost_cases is None or a.split != ["dev"]:
                raise SystemExit("gpt6astra: --split test (pre-specified mapping) or the development cost probe "
                                 "(--pool-cost-cases N --split dev)")
            ids = pool_cost_cases(a.pool_cost_cases)
            run_arm(a.arm, ["dev"], min(a.workers, 4), ids=ids)
            print(json.dumps({k: v for k, v in pool_cost_summary(ids).items() if k != "per_candidate"}, indent=1))
            return
        if "test" in a.split and not (run_dir(a.arm) / "_mapping_choice.json").exists():
            raise SystemExit("the test set runs once, after `choose` has fixed the mapping on development")
        run_arm(a.arm, a.split, a.workers, limit=a.limit)
    elif a.cmd == "pipeline":
        cfg = ARMS[a.arm]
        if not server_up(cfg["base_url"], cfg["model"]):
            print(f"[{a.arm}] waiting for {cfg['model']} at {cfg['base_url']} (polling every 60 s)", flush=True)
            wait_for_server(cfg["base_url"], cfg["model"])
        if run_arm(a.arm, ["dev"], a.workers):
            raise SystemExit("development incomplete; rerun the same command to retry the missing candidates")
        if not (run_dir(a.arm) / "_mapping_choice.json").exists():
            choose_mapping(a.arm)  # fixed on development before the test set is run
        if run_arm(a.arm, ["test"], a.workers):
            raise SystemExit("test incomplete; rerun the same command to retry the missing candidates")
        score_arm(a.arm)
    elif a.cmd == "choose":
        choose_mapping(a.arm)
    elif a.cmd == "score":
        score_arm(a.arm)


if __name__ == "__main__":
    main()
