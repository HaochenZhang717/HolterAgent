"""Candidates and reference classes of the third test set (scripts/mimic_records.py; 2026-10-30).

Candidate pipeline as for LTSTDB, EDB and STAFF III except the rule: Step 2 candidates with LTSTDB protocol A
(>= 75 uV for >= 30 s; 2026-10-30, after protocol B gave only 89 candidates) and the record-wide reference;
candidates ending in the first 300 s dropped (the 300-s lead-in opens every pseudo-record).
Reference classes:
  ischemic  STAFF III record, the candidate overlaps a balloon inflation (coronary occlusion)
  other     non-ischemic record: every candidate. Lab records (Cerebral Vasoregulation in Diabetes): "cause" = the
            protocol phase whose marker last preceded the candidate's onset (rest, Valsalva, hyperventilation, CO2
            rebreathing, tilt; "unknown" without marker numbers); NSR Holter records: "ambulatory"; EPHNOGRAM
            exercise tests: "exercise_treadmill" / "exercise_bicycle"
STAFF III candidates outside the inflations (mostly just after deflation, when ST is still recovering) are neither
ischemic nor clearly non-ischemic and are left out of the split (kept in candidates.json).

    python scripts/build_features.py --data <mimic_holter> --cache cache/features_mimic --workers 8
    python scripts/mimic_cases.py
Writes code/outputs/mimic_cases/candidates.json and summary.json, and the split file RSI/splits/mimic.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RSI"))
import validate_ltstdb as V  # noqa: E402

from holter_agent.agent import HolterToolkit  # noqa: E402
from holter_agent.bench.metrics import SKIP_S  # noqa: E402
from holter_agent.paths import CODE_DIR, OUTPUT_DIR  # noqa: E402
from rsi.paths import MIMIC_CACHE, MIMIC_DIR  # noqa: E402

PROTOCOL = "A"

OUT = OUTPUT_DIR / "mimic_cases"
SPLIT = CODE_DIR / "RSI" / "splits" / "mimic.json"
PHASE_NAMES = {"start": "rest", "val": "valsalva", "hyper": "hyperventilation", "hypo": "co2_rebreathing",
               "tilt": "tilt"}


def cause_of(m: dict, t_onset: float) -> str:
    """Protocol phase of a non-ischemic lab candidate from the marker onsets (pseudo-record time -> file time);
    "ambulatory" for the NSR Holter records."""
    if m["source"] == "nsr":
        return "ambulatory"
    if m["source"] == "ex":
        return "exercise_" + ("treadmill" if "treadmill" in m["scenario"].lower() else "bicycle")
    if t_onset < m["baseline_s"]:
        return "rest"
    t = t_onset + m["file_offset_s"]
    onsets = [(a, lab) for a, lab in m["phase_onsets_file_s"] if a <= t]
    if not m["phase_onsets_file_s"]:
        return "unknown"
    if not onsets:
        return "rest"
    lab = onsets[-1][1].lower()
    return next((v for k, v in PHASE_NAMES.items() if lab.startswith(k)), lab)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "sources.json").read_text())
    tk = HolterToolkit(str(MIMIC_DIR), str(MIMIC_CACHE))
    rows, per_record = [], []
    for m in meta["records"]:
        name = m["record"]
        st = tk._state(name)
        rec, tab = st.record, st.table
        cands = [c for c in V.detect_candidates(tab, V.CFG.episode, PROTOCOL, "global") if c.end_s > SKIP_S]
        infl = m.get("inflations_s", [])
        for c in cands:
            if m["source"] == "staffiii":
                ref = "ischemic" if any(c.start_s < b and c.end_s > a for a, b in infl) else "staffiii_outside_inflation"
                cause = "balloon_inflation" if ref == "ischemic" else "after_or_before_inflation"
            else:
                ref, cause = "other", cause_of(m, c.start_s)
            rows.append({"record": name, "candidate_id": c.id, "start_s": c.start_s, "end_s": c.end_s, "reference": ref,
                         "peak_uv": abs(float(c.peak.peak_uv)), "leads": list(rec.leads), "duplicate_lead_names": False,
                         "source": m["source"], "cause": cause, "patient_code": m["patient_code"],
                         "group": m.get("group", "")})
        covered = [any(c.start_s < b and c.end_s > a for c in cands) for a, b in infl]
        per_record.append({"record": name, "source": m["source"], "n_candidates": len(cands),
                           "n_inflations": len(infl), "n_covered": sum(covered)})
    (OUT / "candidates.json").write_text(json.dumps(rows, indent=1))
    inc = [r for r in rows if r["reference"] in ("ischemic", "other")]
    staff = [r for r in per_record if r["source"] == "staffiii"]
    summary = {
        "records": dict(Counter(r["source"] for r in per_record)),
        "patients": {s: len({r["patient_code"] for r in rows if r["source"] == s}) for s in ("staffiii", "cvd", "ex", "nsr")},
        "split_by_source": dict(Counter(r["source"] for r in rows if r["reference"] in ("ischemic", "other"))),
        "candidates_all": dict(Counter(r["reference"] for r in rows)),
        "split": {"n": len(inc), "ischemic": sum(r["reference"] == "ischemic" for r in inc),
                  "other": sum(r["reference"] == "other" for r in inc),
                  "patients": len({r["patient_code"] for r in inc})},
        "other_by_cause": dict(Counter(r["cause"] for r in inc if r["reference"] == "other")),
        "other_by_group": dict(Counter(r["group"] for r in inc if r["reference"] == "other")),
        "inflations": sum(r["n_inflations"] for r in staff), "inflations_covered": sum(r["n_covered"] for r in staff),
        "per_record": per_record}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ("record", "candidate_id", "start_s", "end_s", "reference", "peak_uv", "leads", "duplicate_lead_names")
    cases = sorted(({k: r[k] for k in keys} for r in inc), key=lambda c: (c["record"], c["start_s"]))
    SPLIT.write_text(json.dumps({
        "note": "third test set: single-lead pseudo-Holter records; ischemic = candidates overlapping "
                "a STAFF III balloon inflation (one of V2/V5/III at random per record), other = every candidate of "
                "the Cerebral Vasoregulation in Diabetes Day-2 lab recordings (rest, Valsalva, hyperventilation, CO2 "
                "rebreathing, head-up tilt), of EPHNOGRAM exercise stress tests and of selected MIT-BIH NSR Holter "
                "pseudo-records; candidate rule LTSTDB protocol A; "
                "scripts/mimic_records.py, mimic_cases.py; record sources in outputs/mimic_cases/sources.json",
        "cases": cases}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_record"}, indent=1))


if __name__ == "__main__":
    main()
