"""Direct judgement: the agent's model and prompts in one session, without the investigation.

Baseline 5 of the comparison: what the tool calls and the agent loop add over handing the same model the same kind
of evidence in one shot. The harness baselines/harnesses/direct is the seed harness h00 (role, quoted sources,
image legend, labels, submission format p_ischemic + label) with two sentences changed: the model sees the outputs of
a fixed set of checks instead of calling tools, and no analysis tools are offered. The checks are the fixed check set
of the rule baseline (holter_agent/agent/baseline.py: signal quality, rhythm, measure_window at the ST extremum,
reference window, compare_morphology reference vs extremum, morphology steps, episode dynamics, ACC/AHA criteria,
eligibility) plus the 10-s raw strip at the extremum (view_ecg). Three observation arms, as in the agent runs:
  numbers  the check outputs as numbers, no images
  images   the images only (trend of the first message, measure_window, compare_morphology, view_ecg) with each
           result's time and leads, no measurements
  both     numbers and images
One session, at most 2 turns, no verification. Each candidate is one model call (plus a nudge if it does not submit).

    python -m baselines.direct_llm run --backend openai --model Qwen/Qwen3.5-27B --base-url http://localhost:8000/v1 \
        --modality both --split dev test --workers 16
    python -m baselines.direct_llm run --backend pool --modality both --split test --workers 8
    python -m baselines.direct_llm score

Per-case outputs: outputs/baselines/_runs/<tag>/<candidate>.json; scored rows: outputs/baselines/direct_<model>_<arm>/.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from baselines.common import OUT, ROOT, data_dirs, load_cases

HARNESS = Path(__file__).resolve().parent / "harnesses" / "direct"
RUNS = OUT / "_runs"
DATA = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data")) / "ltstdb"
CACHE = ROOT / "cache" / "features_ltstdb"
POOL_MODEL, POOL_EFFORT = "gpt-6-astra", "high"  # as in RSI/run_case.py
IMAGE_TOOLS = ("measure_window", "compare_morphology", "view_ecg")
SHORT = {"Qwen/Qwen3.5-27B": "qwen27b", "Qwen/Qwen3.5-35B-A3B": "qwen35a3b", POOL_MODEL: "gpt6astra",
         "gpt-5.6-luna": "luna"}

sys.path.insert(0, str(ROOT / "RSI"))


def tag_of(model: str, modality: str) -> str:
    return f"direct_{SHORT.get(model, model.split('/')[-1].lower())}_{modality}"


def _runner_class():
    from holter_agent.agent import toolsets
    from holter_agent.agent.baseline import collect_standard_evidence, peak_time
    from holter_agent.agent.observe import case_image, visible_case, visible_result
    from holter_agent.agent.prompts import _j, case_header
    from holter_agent.agent.toolkit import TOOLKIT_VERSION, ToolError, call_tool
    from rsi.runner import PoolHarnessRunner

    class DirectRunner(PoolHarnessRunner):
        """PoolHarnessRunner (AgentRunner session loop + harness prompts and submission) with one direct session."""

        def _system(self, role: str) -> str:
            parts = self.harness.spec["system"]["judgement"]
            if not self.cfg.observation.images:
                parts = [p for p in parts if p != "images.md"]
            return "".join(self.harness.parts[p] for p in parts)

        def _review_candidate(self, state, overview, ov_eid, cand, c_eid) -> None:
            cfg, cid, obs = self.cfg, cand["id"], self.cfg.observation
            ov_seen, cand_seen = visible_case(obs, overview, cand)
            header = case_header(state.record_id, ov_seen, ov_eid, cand_seen, c_eid)
            images = [case_image(self.toolkit, state.record_id, cand, cfg.reference)] if obs.images else []
            ev = collect_standard_evidence(self.toolkit, state, cand, cfg.bound, "checks")
            args = {"record_id": state.record_id, "t_s": peak_time(cand), "duration_s": 10.0}
            try:
                res = call_tool(self.toolkit, "view_ecg", args, cfg.bound)
                ev["view_ecg"] = (res, state.add_evidence("view_ecg", args, res, "checks", TOOLKIT_VERSION))
            except ToolError:
                pass
            lines, attached = [], []
            for tool, (result, eid) in ev.items():
                if not obs.offers(tool):
                    continue
                e = state.evidence[int(eid[1:]) - 1]
                shown = visible_result(obs, tool, toolsets.shown_result(cfg.toolset, tool, result))
                lines.append(f"{eid} {tool} {_j(e.args)}: {_j(shown)}")
                if obs.images and tool in IMAGE_TOOLS:
                    imgs = self._evidence_images(state, eid)
                    images += imgs
                    attached += [eid] * len(imgs)
            note = ""
            if obs.images:
                note = (" Images: the ST trend first, then " + ", ".join(attached) + ", in this order."
                        if attached else " Image: the ST trend.")
            first = f"{header}\n\nCheck outputs for this candidate.{note}\n" + "\n".join(lines)
            self._session(state, cid, "direct", self._system("judgement"), first, analysis=False, max_turns=2,
                          images=images)

    return DirectRunner


def _server_up(base_url: str, model: str | None = None) -> bool:
    """The server answers and, if given, lists this model (another job's server on the same port does not count)."""
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            return r.status == 200 and (model is None or model in ids)
    except Exception:  # noqa: BLE001
        return False


def _wait_for_server(base_url: str, model: str | None = None, max_wait_s: float = 12 * 3600) -> bool:
    """Wait until the vLLM server answers again (GPU servers may be preempted and relaunched)."""
    t0 = time.time()
    while not _server_up(base_url, model):
        if time.time() - t0 > max_wait_s:
            return False
        time.sleep(60)
    return True


def run_record(job: dict) -> list[dict]:
    """Review the given candidates of one record (one process per record)."""
    warnings.filterwarnings("ignore")
    from holter_agent.agent import HolterToolkit
    from holter_agent.agent.codex_runner import credits
    from holter_agent.agent.llm import OpenAICompatBackend
    from holter_agent.agent.observe import IMAGE_KINDS, ObservationConfig
    from holter_agent.agent.pool import PoolBackend
    from holter_agent.bench import predictions_from_state
    from rsi.harness import Harness

    out = Path(job["out"])
    harness = Harness.load(HARNESS)
    data, cache = data_dirs(job["record"])  # by record (external evaluation, 2026-10-23); LTSTDB as before
    toolkit = HolterToolkit(str(data), str(cache))
    done = []
    for cid in job["candidates"]:
        path = out / f"{cid}.json"
        if path.exists():
            continue
        if job["backend"] == "pool":
            backend = PoolBackend(job["model"], POOL_EFFORT)
        else:
            backend = OpenAICompatBackend(job["model"], job["base_url"], max_tokens=job["max_tokens"])
        state, failures = None, 0
        while state is None and failures < 2:
            # A preempted or restarting server is waited for, not counted as a failure of the case.
            if job["backend"] == "openai" and not _wait_for_server(job["base_url"], job["model"]):
                return done
            trace = out / "traces" / f"{cid}.trace.jsonl"
            trace.unlink(missing_ok=True)  # one trace per completed attempt
            runner = _runner_class()(toolkit, harness, backend, trace_path=trace)
            runner.cfg = replace(runner.cfg, observation=ObservationConfig(job["modality"], IMAGE_KINDS))
            t0 = time.time()
            try:
                state = runner.run_record(job["record"], [cid])
            except Exception:  # noqa: BLE001 - recorded; the case is retried
                (out / "errors").mkdir(exist_ok=True)
                (out / "errors" / f"{cid}.txt").write_text(traceback.format_exc())
                if job["backend"] == "openai" and not _server_up(job["base_url"], job["model"]):
                    continue
                failures += 1
        if state is None:
            continue
        preds = [p.__dict__ for p in predictions_from_state(state) if p.candidate_id == cid]
        res = {"method": "direct", "harness": {"id": harness.id, "digest": harness.digest()},
               "settings": {k: job[k] for k in ("backend", "model", "modality", "record")} | {"candidate": cid},
               "seconds": round(time.time() - t0, 1), "state": state.to_dict(), "predictions": preds}
        if job["backend"] == "pool":
            try:
                eq = credits(job["model"], backend.usage_total)
            except Exception:  # noqa: BLE001 - no credit rate for this model
                eq = None
            res["pool"] = {"usage": backend.usage_total, "credits_equivalent": eq}
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        done.append({"candidate": cid, "predictions": preds})
    return done


def run(args) -> None:
    model = args.pool_model if args.backend == "pool" else args.model
    # --cases restricts the run to the candidates of a split file (e.g. RSI/splits/validation_v2.json); such runs go
    # to their own folder (<cases stem>_<tag>) so that `score` (direct_* folders, full splits) ignores them
    tag = tag_of(model, args.modality) + (f"_r{args.rep}" if args.rep else "")  # repeated runs (2026-10-23)
    out = RUNS / (f"{Path(args.cases).stem}_{tag}" if args.cases else tag)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in load_cases() if c.split in args.split]
    if args.cases:
        keep = {c["candidate_id"] for c in json.loads(Path(args.cases).read_text())["cases"]}
        cases = [c for c in cases if c.candidate_id in keep]
    if args.limit:
        cases = cases[:args.limit]
    by_rec: dict[str, list[str]] = {}
    for c in cases:
        if not (out / f"{c.candidate_id}.json").exists():
            by_rec.setdefault(c.record, []).append(c.candidate_id)
    # Records with many candidates are split into chunks so that the workers stay busy.
    jobs = []
    for rec, ids in sorted(by_rec.items()):
        for i in range(0, len(ids), args.chunk):
            jobs.append({"record": rec, "candidates": ids[i:i + args.chunk], "out": str(out), "backend": args.backend,
                         "model": model, "base_url": args.base_url, "modality": args.modality,
                         "max_tokens": args.max_tokens})
    print(f"{out.name}: {sum(len(j['candidates']) for j in jobs)} candidates to run in {len(jobs)} jobs", flush=True)
    n = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(run_record, j) for j in jobs]):
            n += len(fut.result())
            print(f"{time.strftime('%H:%M:%S')} {n} done", flush=True)


def score(args) -> None:
    from baselines.learned_common import finish, load_bundle

    b = load_bundle(signals=False)
    split = dict(zip(b.ids, b.split))
    for folder in sorted(RUNS.glob("direct_*")):
        dev, test, label, missing, cost, n = {}, {}, {}, [], 0.0, 0
        for cid in b.ids:
            f = folder / f"{cid}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            n += 1
            cost += r.get("pool", {}).get("credits_equivalent") or 0.0
            p = next((x for x in r["predictions"] if x["candidate_id"] == cid), None)
            if p is None or p.get("p_ischemic") is None:
                missing.append(cid)
                continue
            (dev if split[cid] == "dev" else test)[cid] = float(p["p_ischemic"])
            label[cid] = p.get("final_label") == "ischemic"
        splits_run = sorted({split[c] for c in dev} | {split[c] for c in test})
        n_split = sum(1 for c in b.ids if split[c] in splits_run)
        if n < n_split:
            print(f"{folder.name}: {n}/{n_split} candidates run, not scored yet")
            continue
        notes = (f"one session, harness direct (h00 prompts, no analysis tools), {folder.name.split('_')[-1]} arm; "
                 f"{len(missing)} without a submitted p_ischemic (scored as the median)")
        if not dev:
            notes += "; test only: the threshold is p >= 0.5 (no development run)"
        extra = {"missing": missing, "n_run": n, "credits_equivalent_total": round(cost, 1)}
        r = finish(folder.name, b, dev, test, notes, extra=extra, label=label)
        t = r["splits"]["test"]
        print(f"{folder.name}: test AUC {t.get('auc', float('nan')):.3f} {t.get('auc_ci')}, missing {len(missing)}, "
              f"credits {cost:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["openai", "pool"], required=True)
    r.add_argument("--pool-model", default=POOL_MODEL, help="model of the pool backend (2026-10-23)")
    r.add_argument("--rep", type=int, default=0, help="repeat index: outputs go to <tag>_r<rep>")
    r.add_argument("--model", default="Qwen/Qwen3.5-27B")
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--modality", choices=["numbers", "images", "both"], required=True)
    r.add_argument("--split", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--chunk", type=int, default=4, help="candidates per job")
    r.add_argument("--max-tokens", type=int, default=16384)
    r.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke test)")
    r.add_argument("--cases", default=None, help="only the candidates listed in this split file")
    sub.add_parser("score")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
