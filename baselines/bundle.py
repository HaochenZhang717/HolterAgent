"""One fixed observation per candidate, shared by the learned and multimodal baselines.

For every development and test candidate of the patient-level split (baselines/common.py):
- signals: K = 8 raw-ECG segments of 10 s inside the candidate (the first centred on the ST extremum that the fixed
  check set uses, the others evenly spaced from start to end) and one 10-s segment centred on the reference window
  of the fixed check set (find_reference_window at the extremum: clean, ST stable, heart rate within 10 bpm, within
  2 h; without the heart-rate condition if none; 10 min before onset if still none). Native 250 Hz, mV, the
  power-line notch filter every measurement uses; records with two leads get a zero third lead (lead_mask).
- images: the four image kinds the agent receives, for the same checks as the fixed check set:
  trend.png (ST trend +-30 min with the candidate shaded, the first message of a case), beats.png (averaged beats at
  the extremum, measure_window), overlay.png (reference vs extremum beats, compare_morphology), strip.png and
  strip_ref.png (10-s raw strips at the extremum and at the reference, view_ecg).

    python -m baselines.bundle --workers 10

Writes outputs/baselines/_bundle/{signals.npz, meta.json, images/<candidate>/<kind>.png}.
"""
from __future__ import annotations

import os
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baselines.common import OUT, TEST_SETS, data_dirs, load_cases, protocol_of
from baselines.evidence import INPUTS

BUNDLE = OUT / "_bundle"
SEG_S, K, MAX_LEADS = 10.0, 8, 3
BOUND = {"protocol": "B", "reference": "global"}


def _segment_times(start: float, end: float, peak: float) -> list[float]:
    others = np.linspace(start + SEG_S / 2, max(start + SEG_S / 2, end - SEG_S / 2), K - 1)
    return [float(peak)] + [float(t) for t in others]


def record_bundle(job: tuple[str, str, str, list[str]]) -> dict:
    data, cache, rid, cids = job
    warnings.filterwarnings("ignore")
    from holter_agent.agent.observe import case_image, tool_image
    from holter_agent.agent.toolkit import HolterToolkit
    from holter_agent.trend.window import read_signal

    tk = HolterToolkit(data, cache)
    bound = {**BOUND, "protocol": protocol_of(rid)}  # B, or A for the third test set (common.protocol_of)
    summaries = {c["id"]: c for c in tk.list_candidates(rid, bound["protocol"], "global", max_items=10**9)["candidates"]}
    st = tk._state(rid)
    rec = st.record
    n = int(round(SEG_S * rec.fs))
    inputs = json.loads((INPUTS / f"{rid}.json").read_text())["candidates"]
    out = {"record": rid, "fs": float(rec.fs), "leads": list(rec.leads), "cases": {}}
    for cid in cids:
        inp, summ = inputs[cid], summaries[cid]
        start, end, peak = float(summ["start_s"]), float(summ["end_s"]), float(inp["peak_s"])
        ref = (inp["evidence"].get("find_reference_window") or {})
        t_ref = ref.get("t_s") if isinstance(ref, dict) else None
        ref_source = "find_reference_window" if t_ref is not None else "onset_minus_600s"
        if t_ref is None:
            t_ref = max(SEG_S / 2, start - 600.0)
        t_ref = float(min(t_ref, rec.duration_s - SEG_S / 2))
        segs, seg_starts = [], []
        dur = rec.duration_s
        for t in _segment_times(start, end, peak) + [float(t_ref)]:
            t = float(np.clip(t, SEG_S / 2, dur - SEG_S / 2))  # keep the segment inside the record
            s0, _ = tk.strip_range(rid, t, SEG_S)
            a = rec.to_samples(s0)
            sig = read_signal(rec, a, a + n, tk.cfg).signal  # (n, leads), mV
            x = np.zeros((MAX_LEADS, n), np.float32)
            x[:sig.shape[1], :sig.shape[0]] = sig.T[:MAX_LEADS]
            segs.append(x)
            seg_starts.append(s0)
        folder = BUNDLE / "images" / cid
        folder.mkdir(parents=True, exist_ok=True)
        images, errors = {}, {}
        jobs = {
            "trend": lambda: case_image(tk, rid, summ, "global"),
            "beats": lambda: tool_image(tk, "measure_window", {"record_id": rid, "t_s": peak, "duration_s": 30.0},
                                        bound),
            "overlay": lambda: tool_image(tk, "compare_morphology", {"record_id": rid, "t_ref_s": float(t_ref),
                                                                      "t_s": peak, "duration_s": 30.0}, bound),
            "strip": lambda: tool_image(tk, "view_ecg", {"record_id": rid, "t_s": peak, "duration_s": SEG_S}, bound),
            "strip_ref": lambda: tool_image(tk, "view_ecg", {"record_id": rid, "t_s": float(t_ref),
                                                             "duration_s": SEG_S}, bound),
        }
        for kind, make in jobs.items():
            path = folder / f"{kind}.png"
            try:
                if not path.exists():
                    path.write_bytes(make())
                images[kind] = str(path.relative_to(BUNDLE))
            except Exception as e:  # noqa: BLE001 - a missing image is recorded, not fatal
                errors[kind] = f"{type(e).__name__}: {e}"
        out["cases"][cid] = {"start_s": start, "end_s": end, "peak_s": peak, "t_ref_s": float(t_ref),
                             "ref_source": ref_source, "seg_start_s": seg_starts,
                             "signals": np.stack(segs), "images": images, "image_errors": errors}
    return out


def build(data: str, cache: str, workers: int) -> None:
    cases = load_cases()
    by_rec: dict[str, list[str]] = {}
    for c in cases:
        by_rec.setdefault(c.record, []).append(c.candidate_id)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    # external evaluation (2026-10-23): each record's own data and cache directory (common.data_dirs)
    jobs = [((str(data_dirs(r)[0]), str(data_dirs(r)[1])) if TEST_SETS else (data, cache)) + (r, ids)
            for r, ids in sorted(by_rec.items())]
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(record_bundle, jobs):
            results[res["record"]] = res
            print(f"{res['record']}: {len(res['cases'])} candidates", flush=True)
    order = [c.candidate_id for c in cases]
    meta, sig = {}, []
    for c in cases:
        r = results[c.record]
        e = r["cases"][c.candidate_id]
        nl = min(len(r["leads"]), MAX_LEADS)
        meta[c.candidate_id] = {
            "record": c.record, "split": c.split, "dataset": c.dataset, "reference": c.reference, "ischemic": c.ischemic,
            "fs": r["fs"], "leads": r["leads"][:MAX_LEADS], "n_leads": nl,
            **{k: v for k, v in e.items() if k != "signals"},
        }
        sig.append(e["signals"])
    sig = np.stack(sig)  # [N, K+1, MAX_LEADS, n]
    mask = np.array([[i < meta[cid]["n_leads"] for i in range(MAX_LEADS)] for cid in order])
    np.savez_compressed(BUNDLE / "signals.npz", ids=np.array(order), cur=sig[:, :K], ref=sig[:, K],
                        lead_mask=mask, fs=np.float32(250.0))
    (BUNDLE / "meta.json").write_text(json.dumps({"seg_s": SEG_S, "k": K, "max_leads": MAX_LEADS,
                                                  "order": order, "cases": meta}, indent=1))
    n_err = sum(len(m["image_errors"]) for m in meta.values())
    print(f"wrote {BUNDLE}: {len(order)} candidates, signals {sig.shape}, image errors {n_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", default=str(Path(os.environ.get("HOLTER_DATA_ROOT", root / "data")) / "ltstdb"))
    ap.add_argument("--cache", default=str(root / "cache" / "features_ltstdb"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.data, a.cache, a.workers)
