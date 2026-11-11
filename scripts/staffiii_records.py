"""STAFF III (PhysioNet staffiii 1.0.0) as an extra external test of ischemia sensitivity.

STAFF III holds 12-lead ECGs (V1-V6, I, II, III stored; 1000 Hz) of 104 patients recorded before and during
prolonged PTCA balloon inflation, with the inflation onset and duration of every inflation recording in the
annotation spreadsheet (D0: s from file onset to inflation, D1: inflation duration). During inflation the coronary
artery is occluded, so the myocardium it supplies is ischemic: a ground truth set by the procedure, not by a reader.

Our agent reads Holter-like records, so for every annotated inflation we build one pseudo-record:
  the patient's 5-min cath-lab baseline recording (BC1) followed directly by the inflation recording (BIk),
  three leads V2, V5 and III (close to a three-channel Holter: anterior, lateral, inferior), resampled to 250 Hz,
  beats from wfdb's XQRS detector on lead II, or V5 / V2 / III when lead II gives an implausible rate (all labelled N;
  STAFF III has no beat annotations).
The baseline comes first and makes up most of the record, so the record-wide reference of our pipeline lies near the
patient's resting ST level. The inflation interval(s) in pseudo-record time are written to inflations.json (one pseudo-record per inflation
file; a file with two inflations gives one record with both intervals).

    python scripts/staffiii_records.py      # reads $HOLTER_DATA_ROOT/staffiii, writes .../staffiii_holter
"""
from __future__ import os
import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import wfdb
from scipy.signal import resample_poly
from wfdb import processing

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data")) / "staffiii"
DST = Path(os.environ.get("HOLTER_DATA_ROOT", ROOT / "data")) / "staffiii_holter"
LEADS = ["V2", "V5", "III"]
# left out before any result was seen: the spreadsheet notes possible lead or sign reversal for patients 1, 4, 5, 6
# and 89; and a pseudo-record needs a baseline of at least 240 s (one BC1 is 101 s long)
EXCLUDE_PATIENTS = {1, 4, 5, 6, 89}
MIN_BASELINE_S = 240.0
FS_IN, FS_OUT = 1000, 250


def patients() -> list[dict]:
    rows = list(openpyxl.load_workbook(SRC / "STAFF-III-Database-Annotations.xlsx", read_only=True)
                .worksheets[0].iter_rows(values_only=True))
    hdr = rows[9]
    out = []
    for r in rows[10:]:
        if not isinstance(r[0], (int, float)):
            continue
        d = dict(zip(hdr, r))
        infl = []
        for i in range(1, 6):
            rec, dd = d.get(f"BI{i}"), d.get(f"BI{i}:D0;D1;D2")
            if rec and dd:
                d0, d1, d2 = (float(x) for x in str(dd).split(";"))
                infl.append({"bi": i, "file": str(rec).strip(), "artery": d.get(f"BI{i}:Occluded artery"),
                             "d0": d0, "d1": d1, "d2": d2})
        if d.get("BC1") and infl:
            out.append({"patient": int(r[0]), "age": r[1], "sex": r[2], "bc1": str(d["BC1"]).strip(), "inflations": infl})
    return out


def path_of(file_label: str) -> Path:
    m = re.match(r"(\d+)(\w)$", file_label)
    return SRC / "data" / f"{int(m.group(1)):03d}{m.group(2)}"


def read(file_label: str) -> tuple[np.ndarray, np.ndarray]:
    """Signals V2, V5, III and II in mV at 250 Hz."""
    rec = wfdb.rdrecord(str(path_of(file_label)))
    assert int(rec.fs) == FS_IN, rec.fs
    idx = [rec.sig_name.index(n) for n in LEADS + ["II"]]
    sig = resample_poly(rec.p_signal[:, idx], FS_OUT, FS_IN, axis=0)
    return sig[:, :3], sig[:, 3]


def detect_beats(lead_ii: np.ndarray, x: np.ndarray) -> tuple[np.ndarray | None, str]:
    """XQRS on lead II; when that gives an implausible beat count (rate outside 30-200 bpm), V5, V2 and III in turn."""
    for name, sig in (("II", lead_ii), ("V5", x[:, 1]), ("V2", x[:, 0]), ("III", x[:, 2])):
        beats = processing.xqrs_detect(sig, fs=FS_OUT, verbose=False)
        rate = len(beats) / (len(sig) / FS_OUT / 60)
        if len(beats) < 10 or not 30 <= rate <= 200:
            continue
        beats = processing.correct_peaks(sig, np.asarray(beats, dtype=int), search_radius=int(0.05 * FS_OUT),
                                         smooth_window_size=20)
        return np.unique(beats[(beats > 0) & (beats < len(sig))]), name
    return None, ""


def main() -> None:
    warnings.filterwarnings("ignore")
    DST.mkdir(parents=True, exist_ok=True)
    meta, skipped = [], []
    for p in patients():
        if p["patient"] in EXCLUDE_PATIENTS:
            skipped.append({"patient": p["patient"], "file": p["bc1"], "why": "possible lead or sign reversal (spreadsheet)"})
            continue
        try:
            base, base_ii = read(p["bc1"])
        except (FileNotFoundError, ValueError) as e:
            skipped.append({"patient": p["patient"], "file": p["bc1"], "why": str(e)[:120]})
            continue
        if base.shape[0] / FS_OUT < MIN_BASELINE_S:
            skipped.append({"patient": p["patient"], "file": p["bc1"], "why": f"baseline only {base.shape[0] / FS_OUT:.0f} s"})
            continue
        by_file: dict[str, list[dict]] = {}  # some inflation files hold two inflations (spreadsheet note)
        for inf in p["inflations"]:
            by_file.setdefault(inf["file"], []).append(inf)
        for file_label, infs in by_file.items():
            name = f"t{p['patient']:03d}{infs[0]['bi']}"  # record[:-1] is the patient, as for LTSTDB
            try:
                sig, ii = read(file_label)
            except (FileNotFoundError, ValueError) as e:
                skipped.append({"patient": p["patient"], "file": file_label, "why": str(e)[:120]})
                continue
            x = np.vstack([base, sig]).astype(np.float64)
            lead_ii = np.concatenate([base_ii, ii])
            beats, det_lead = detect_beats(lead_ii, x)
            if beats is None:
                skipped.append({"patient": p["patient"], "file": file_label, "why": "no plausible beat detection"})
                continue
            off = base.shape[0] / FS_OUT
            intervals = [[off + i["d0"], off + i["d0"] + i["d1"]] for i in infs]
            wfdb.wrsamp(name, fs=FS_OUT, units=["mV"] * 3, sig_name=LEADS, p_signal=x, fmt=["16"] * 3,
                        write_dir=str(DST), comments=[f"STAFF III patient {p['patient']}: BC1 {p['bc1']} + inflation "
                                                      f"recording {file_label} ({infs[0]['artery']})"])
            wfdb.wrann(name, "atr", beats, symbol=["N"] * len(beats), write_dir=str(DST))
            meta.append({"record": name, "patient": p["patient"], "baseline_file": p["bc1"], "inflation_file": file_label,
                         "occluded_artery": infs[0]["artery"], "baseline_s": off, "duration_s": x.shape[0] / FS_OUT,
                         "inflations_s": intervals, "n_beats": int(len(beats)), "beat_lead": det_lead})
            print(f"{name}: {x.shape[0] / FS_OUT:.0f} s, inflation(s) {[[round(a), round(b)] for a, b in intervals]} s, "
                  f"{len(beats)} beats", flush=True)
    (DST / "inflations.json").write_text(json.dumps({"leads": LEADS, "fs": FS_OUT, "records": meta, "skipped": skipped},
                                                   indent=1))
    print(f"{len(meta)} pseudo-records, {len(skipped)} skipped")


if __name__ == "__main__":
    sys.exit(main())
