"""Third test set: balloon-occlusion ischemia (STAFF III) against non-ischemic ST changes, all as
single-lead pseudo-Holter records of the same form ("<300-s lead-in><event part>", one lead named "ECG", 250 Hz).

Ischemic source: STAFF III (scripts/staffiii_records.py), one pseudo-record per balloon-inflation file = the patient's
  5-min cath-lab baseline (BC1) followed by the inflation recording; same patient exclusions as staffiii_records.py
  (possible lead reversal; baseline < 240 s). One lead per record, drawn at random from V2, V5 and III (2026-10-30, after V5 alone turned out to show only 22 of the 144 inflations; seeded per record).
Non-ischemic sources (downloads, 2026-10-30):
  cvd  "Cerebral Vasoregulation in Diabetes" (PhysioNet cerebral-vasoreg-diabetes 1.0.0, Novak & Mendez 2020), Day-2
       lab recordings (Data/Labview/Converted/Head-up-tilt_Day2/S####DB, one ECG lead, 45-60 min): 10-min supine rest,
       Valsalva maneuvers, hyperventilation and CO2 rebreathing supine, supine rest, head-up tilt, the breathing
       maneuvers repeated upright. The headers say 1000 Hz, but the recordings are sampled at 500 Hz: the release's
       GE-71_File_and_channels.csv gives 500 Hz for Day 2, and only at 500 Hz are the resting heart rates (49-101 bpm,
       median 67), the breathing rates from the CO2 channel (6-23 /min) and the protocol phase durations from the
       markers physiological (checked on all 57 records). The study excluded "clinically important cardiac disease,
       arrhythmia", heart failure and arterial stents (Novak et al., Diabetes Care 2006;29:1529); type 2 diabetes is
       kept. Lead-in = the first 300 s (supine rest); event parts = back-to-back windows after the 10-min rest.
       Protocol phases from the marker channel (k-th pulse = marker number k of
       GE-71_Head-up-tilt-Day2_Markers_per_subject.csv; 28 subjects have no marker numbers) only name what was going
       on during a candidate.
  nsr  MIT-BIH Normal Sinus Rhythm Database (PhysioNet nsrdb 1.0.0): 18 ~24-h ambulatory recordings, 2 leads, 128 Hz,
       of people aged 20-50 referred to an arrhythmia laboratory and found to have no significant arrhythmia. Each
       recording is cut into back-to-back pseudo-records (lead-in = the 300 s before the event part; one of the two
       leads at random per pseudo-record). Selection, fixed before any candidate was seen: per subject, at most 9
       pseudo-records drawn at random among those with at least one candidate (so that 18 subjects do not dominate).
  ex   EPHNOGRAM (PhysioNet ephnogram 1.0.0; Kazemnejad et al.): the 21 exercise stress tests (11 modified-Bruce
       treadmill, 10 bicycle; 30 min each) of healthy men aged 23-29 with no reported cardiovascular disorder; ECG
       channel, 8 kHz. Lead-in = the first 300 s of the session; event parts = back-to-back windows after it; at most
       9 pseudo-records per subject (random). Rate-related ST changes (added 2026-10-30).
Candidate rule of this set (2026-10-30, after the protocol-B version gave only 89 candidates): LTSTDB protocol A
(>= 75 uV for >= 30 s) instead of B; the NSR selection uses the same rule.
Event-part lengths of the non-ischemic pseudo-records are drawn (with replacement, seeded) from the STAFF III
inflation-recording lengths, so that record length does not tell the source.
All records: 16-bit, no header comments, beats from wfdb's XQRS on the record's lead (a record whose detected rate is
outside 30-200 bpm is left out); names m<3-digit patient code><index 1-9>, patient codes a random permutation over all
sources (seed 2026). Sources and protocol times: code/outputs/mimic_cases/sources.json (not in the data directory).

    python scripts/mimic_records.py
"""
from __future__ import annotations

import os
import csv
import json
import random
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from wfdb import processing

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staffiii_records as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data"))
CVD = DATA / "cvd_diabetes"
CVD_REC = CVD / "Data" / "Labview" / "Converted" / "Head-up-tilt_Day2"
NSR = DATA / "nsrdb"
EX = DATA / "ephnogram"
EX_RECORDS = [f"ECGPCG{n:04d}" for n in (35, 36, 37, 38, 39, 46, 47, 52, 54, 55, 56, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69)]
PROTOCOL = "A"  # candidate rule of this set (see the docstring)
DST = DATA / "mimic_holter"
STAGE = DATA / "mimic_nsr_stage"  # all NSR pseudo-records before the selection (never given to a model)
META = ROOT / "code" / "outputs" / "mimic_cases" / "sources.json"
FS = 250
LEAD_IN_S = 300.0
CVD_FS = 500  # true sampling rate of the Day-2 recordings (see the docstring)
REST_S = 600.0
NSR_PER_SUBJECT = 9
SEED = 2026
STAFF_LEADS = ("V2", "V5", "III")  # column order of staffiii_records.read


def beats_of(x: np.ndarray) -> np.ndarray | None:
    b = processing.xqrs_detect(x, fs=FS, verbose=False)
    rate = len(b) / (len(x) / FS / 60)
    if len(b) < 10 or not 30 <= rate <= 200:
        return None
    b = processing.correct_peaks(x, np.asarray(b, dtype=int), search_radius=int(0.05 * FS), smooth_window_size=20)
    return np.unique(b[(b > 0) & (b < len(x))])


def write(name: str, x: np.ndarray, beats: np.ndarray, where: Path) -> None:
    wfdb.wrsamp(name, fs=FS, units=["mV"], sig_name=["ECG"], p_signal=x[:, None].astype(np.float64), fmt=["16"],
                write_dir=str(where))
    wfdb.wrann(name, "atr", beats, symbol=["N"] * len(beats), write_dir=str(where))


def _beats_job(args: tuple[str, np.ndarray]) -> tuple[str, np.ndarray | None]:
    warnings.filterwarnings("ignore")
    return args[0], beats_of(args[1])


def staff_records(skipped: list[dict]) -> list[dict]:
    out = []
    for p in S.patients():
        if p["patient"] in S.EXCLUDE_PATIENTS:
            skipped.append({"source": "staffiii", "subject": p["patient"], "why": "possible lead or sign reversal"})
            continue
        try:
            base = S.read(p["bc1"])[0]
        except (FileNotFoundError, ValueError) as e:
            skipped.append({"source": "staffiii", "subject": p["patient"], "why": str(e)[:120]})
            continue
        if len(base) / FS < S.MIN_BASELINE_S:
            skipped.append({"source": "staffiii", "subject": p["patient"], "why": f"baseline {len(base) / FS:.0f} s"})
            continue
        by_file: dict[str, list[dict]] = {}
        for inf in p["inflations"]:
            by_file.setdefault(inf["file"], []).append(inf)
        for file_label, infs in by_file.items():
            try:
                sig = S.read(file_label)[0]
            except (FileNotFoundError, ValueError) as e:
                skipped.append({"source": "staffiii", "subject": p["patient"], "file": file_label, "why": str(e)[:120]})
                continue
            lead = random.Random(f"{SEED}:staffiii:{p['patient']}:{file_label}").choice(STAFF_LEADS)
            j = STAFF_LEADS.index(lead)
            off = len(base) / FS
            out.append({"source": "staffiii", "subject": f"staffiii:{p['patient']}", "order": infs[0]["bi"],
                        "x": np.concatenate([base[:, j], sig[:, j]]), "lead_source": lead,
                        "inflation_file": file_label, "baseline_file": p["bc1"], "occluded_artery": infs[0]["artery"],
                        "baseline_s": off, "event_s": len(sig) / FS,
                        "inflations_s": [[off + i["d0"], off + i["d0"] + i["d1"]] for i in infs]})
    return out


def cvd_markers() -> dict[str, dict]:
    rows = list(csv.reader((CVD / "Data_Description" / "GE-71_Head-up-tilt-Day2_Markers_per_subject.csv").open()))
    head, out = rows[0], {}
    for r in rows[1:]:
        if r and r[0].strip():
            cells = [c.strip() for c in r]
            out[cells[0].upper()] = {"group": cells[1], "cells": cells[2:-1], "notes": cells[-1], "head": head[2:-1]}
    return out


def marker_times(m: np.ndarray, fs: float) -> np.ndarray:
    """Onset times (s) of the marker pulses (rising edges above half the marker range)."""
    hi = m > (np.nanmin(m) + 0.5 * (np.nanmax(m) - np.nanmin(m)))
    on = np.flatnonzero(hi[1:] & ~hi[:-1]) + 1
    keep = [on[0]] if len(on) else []
    for t in on[1:]:  # a pulse can show two edges a few samples apart
        if t - keep[-1] > 0.05 * fs:
            keep.append(t)
    return np.array(keep, float) / fs


def windows(t0: float, t_end: float, lengths: list[float], rng: random.Random, lead_in: bool) -> list[tuple[float, float]]:
    """Back-to-back event parts with lengths drawn from `lengths`; with lead_in, each is preceded by its own 300 s."""
    out, a = [], t0
    while True:
        L = rng.choice(lengths)
        start = a + (LEAD_IN_S if lead_in else 0.0)
        if start + L > t_end:
            return out
        out.append((start, start + L))
        a = start + L


def cvd_records(skipped: list[dict], lengths: list[float]) -> list[dict]:
    marks = cvd_markers()
    out = []
    for hea in sorted(CVD_REC.glob("*DB.hea")):
        name = hea.stem
        sid = name[:-2].upper()  # s0030DB -> S0030
        info = marks.get(sid, {"group": "", "cells": [], "notes": "", "head": []})
        rec = wfdb.rdrecord(str(CVD_REC / name), channel_names=["marker", "ecg"])
        pulses = marker_times(rec.p_signal[:, 0], CVD_FS)
        x = resample_poly(np.nan_to_num(rec.p_signal[:, 1]), FS, CVD_FS)
        dur = len(x) / FS
        named = [(lab.strip(), int(c)) for lab, c in zip(info["head"], info["cells"]) if c.isdigit() and 0 < int(c) <= len(pulses)]
        phases = sorted((float(pulses[k - 1]), lab) for lab, k in named if not lab.lower().startswith("end"))
        wins = windows(REST_S, dur, lengths, random.Random(f"{SEED}:cvd:{sid}"), lead_in=False)
        if not wins:
            skipped.append({"source": "cvd", "subject": sid, "file": name, "why": f"recording {dur:.0f} s, no window"})
            continue
        n0 = int(LEAD_IN_S * FS)
        for k, (a, b) in enumerate(wins, 1):
            out.append({"source": "cvd", "subject": f"cvd:{sid}", "group": info["group"], "file": name, "order": k,
                        "x": np.concatenate([x[:n0], x[int(a * FS):int(b * FS)]]), "lead_source": "lab ECG",
                        "baseline_s": LEAD_IN_S, "event_s": b - a, "window_in_file_s": [a, b],
                        "file_offset_s": a - LEAD_IN_S,  # pseudo-record time t >= 300 s is file time t + offset
                        "phase_onsets_file_s": phases, "file_duration_s": dur, "marker_notes": info["notes"]})
    return out


def nsr_stage(lengths: list[float], skipped: list[dict]) -> list[dict]:
    """Every NSR pseudo-record (lead-in + event part, contiguous in the recording), written to STAGE one recording
    at a time (metadata returned, signals stay on disk)."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    out = []
    with ProcessPoolExecutor(8) as ex:
        for name in (NSR / "RECORDS").read_text().split():
            rec = wfdb.rdrecord(str(NSR / name))
            fs = int(rec.fs)
            sig = (resample_poly(np.nan_to_num(rec.p_signal), FS, fs, axis=0) if fs != FS else rec.p_signal).astype(np.float32)
            del rec
            dur = len(sig) / FS
            rng = random.Random(f"{SEED}:nsr:{name}")
            metas, jobs = [], []
            for k, (a, b) in enumerate(windows(0.0, dur, lengths, rng, lead_in=True), 1):
                j = rng.randrange(sig.shape[1])
                stage_name = f"w{name}_{k:03d}"
                metas.append({"source": "nsr", "subject": f"nsr:{name}", "order": k, "lead_source": f"ECG{j + 1}",
                              "baseline_s": LEAD_IN_S, "event_s": b - a, "window_in_file_s": [a - LEAD_IN_S, b],
                              "stage_name": stage_name})
                jobs.append((stage_name, sig[int((a - LEAD_IN_S) * FS):int(b * FS), j].astype(np.float64)))
            beats = dict(ex.map(_beats_job, jobs, chunksize=8))
            for m, (stage_name, x) in zip(metas, jobs):
                if beats[stage_name] is None:
                    skipped.append({"source": "nsr", "subject": m["subject"], "stage": stage_name,
                                    "why": "implausible beat rate"})
                    continue
                write(stage_name, x, beats[stage_name], STAGE)
                out.append(m)
            del sig, jobs
            print(f"nsr {name}: {len(metas)} pseudo-records staged", flush=True)
    return out


def nsr_select(recs: list[dict], skipped: list[dict]) -> list[dict]:
    """Per subject, at most NSR_PER_SUBJECT pseudo-records at random among those with a candidate."""
    import validate_ltstdb as V  # noqa: E402  (scripts/)
    from holter_agent import DEFAULT_CONFIG
    from holter_agent.bench.metrics import SKIP_S
    from holter_agent.data import RecordStore
    from holter_agent.trend.features import load_or_compute
    store, cache = RecordStore(str(STAGE)), STAGE / "_cache"
    with_cand: dict[str, list[dict]] = {}
    for r in recs:
        tab = load_or_compute(store.get(r["stage_name"]), DEFAULT_CONFIG, str(cache))
        if any(c.end_s > SKIP_S for c in V.detect_candidates(tab, V.CFG.episode, PROTOCOL, "global")):
            with_cand.setdefault(r["subject"], []).append(r)
    chosen = []
    for subj in sorted({r["subject"] for r in recs}):
        pool = sorted(with_cand.get(subj, []), key=lambda r: r["order"])
        pick = random.Random(f"{SEED}:nsr-pick:{subj}").sample(pool, min(NSR_PER_SUBJECT, len(pool)))
        for k, r in enumerate(sorted(pick, key=lambda r: r["order"]), 1):
            x = wfdb.rdrecord(str(STAGE / r["stage_name"])).p_signal[:, 0]
            chosen.append({**r, "order": k, "stage_order": r["order"], "x": x})
        skipped.append({"source": "nsr", "subject": subj, "why": f"{len(pool)} of {sum(x['subject'] == subj for x in recs)} "
                        f"pseudo-records with a candidate, {len(pick)} drawn (selection, not an exclusion)"})
    return chosen


def ex_records(skipped: list[dict], lengths: list[float]) -> list[dict]:
    sheet = {r["Record Name"].strip(): r for r in csv.DictReader((EX / "ECGPCGSpreadsheet.csv").open(encoding="utf-8"))}
    by_subject: dict[str, list[dict]] = {}
    for name in EX_RECORDS:
        info = sheet[name]
        rec = wfdb.rdrecord(str(EX / "WFDB" / name), channel_names=["ECG"])
        fs = int(rec.fs)
        x = resample_poly(np.nan_to_num(rec.p_signal[:, 0]), FS, fs)
        del rec
        dur = len(x) / FS
        subj = f"ex:{info['Subject ID'].strip()}"
        n0 = int(LEAD_IN_S * FS)
        for a, b in windows(LEAD_IN_S, dur, lengths, random.Random(f"{SEED}:ex:{name}"), lead_in=False):
            by_subject.setdefault(subj, []).append({
                "source": "ex", "subject": subj, "file": name, "scenario": info["Recording Scenario"].strip(),
                "x": np.concatenate([x[:n0], x[int(a * FS):int(b * FS)]]), "lead_source": "ECG",
                "baseline_s": LEAD_IN_S, "event_s": b - a, "window_in_file_s": [a, b], "file_offset_s": a - LEAD_IN_S,
                "file_duration_s": dur, "ecg_notes": info.get("ECG Notes", "").strip()})
    out = []
    for subj, recs in sorted(by_subject.items()):
        pick = recs if len(recs) <= NSR_PER_SUBJECT else random.Random(f"{SEED}:ex-pick:{subj}").sample(recs, NSR_PER_SUBJECT)
        if len(pick) < len(recs):
            skipped.append({"source": "ex", "subject": subj, "why": f"{len(pick)} of {len(recs)} pseudo-records drawn"})
        out += [{**r, "order": k} for k, r in enumerate(pick, 1)]
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    skipped: list[dict] = []
    staff = staff_records(skipped)
    lengths = sorted(round(r["event_s"], 1) for r in staff)
    cvd = cvd_records(skipped, lengths)
    ex = ex_records(skipped, lengths)
    nsr = nsr_select(nsr_stage(lengths, skipped), skipped)
    allrec = staff + cvd + ex + nsr
    subjects = sorted({r["subject"] for r in allrec})
    codes = random.Random(SEED).sample(range(100, 1000), len(subjects))
    code_of = dict(zip(subjects, codes))
    named = []
    for subj in subjects:
        for idx, r in enumerate(sorted((r for r in allrec if r["subject"] == subj), key=lambda r: r["order"]), 1):
            assert idx <= 9, subj
            named.append((f"m{code_of[subj]:03d}{idx}", r))
    with ProcessPoolExecutor(8) as ex:
        beats = dict(ex.map(_beats_job, [(n, r["x"]) for n, r in named], chunksize=4))
    meta = []
    for name, r in named:
        if beats[name] is None:
            skipped.append({"source": r["source"], "subject": r["subject"], "record": name, "why": "implausible beat rate"})
            continue
        write(name, r["x"], beats[name], DST)
        meta.append({"record": name, "patient_code": code_of[r["subject"]], "duration_s": len(r["x"]) / FS,
                     "n_beats": int(len(beats[name])), **{k: v for k, v in r.items() if k not in ("x", "order")}})
    META.write_text(json.dumps({"fs": FS, "seed": SEED, "records": meta, "skipped": skipped}, indent=1))
    n = {s: sum(m["source"] == s for m in meta) for s in ("staffiii", "cvd", "ex", "nsr")}
    print(f"{len(meta)} records ({n}) -> {DST}; sources in {META}")


if __name__ == "__main__":
    sys.exit(main())
