"""Build the precedent bank of the knowledge base: every development candidate of LTSTDB (split_v07, 36 records,
587 candidates) with its expert class and the tool features of the fixed check set.

    python RSI/kb/build_precedents.py        # writes RSI/kb/precedents.json

The features are those of baselines/evidence.py (outputs/baselines/_inputs/<record>.json, "tool_*"): the same
estimate.features() of the same check set that the agent's dossier contains, so a case's features can be recomputed
at run time from its dossier (checked equal on validation_v2 cases). The features computed from the series that LTSTDB
supplies (KLT, Legendre) and the clinical header features are not used. Test records are never in the bank; at run
time rsi/knowledge.py also removes every patient of the split being evaluated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent
CODE_DIR = KB_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

from baselines.evidence import load_inputs  # noqa: E402

SPLIT = CODE_DIR / "outputs" / "ltstdb_cases" / "split_v07.json"
OUT = KB_DIR / "precedents.json"


def main() -> None:
    split = json.loads(SPLIT.read_text())
    rows = load_inputs()
    test = set(split["test_records"])
    cases = []
    for c in split["dev_cases"]:
        assert c["record"] not in test
        f = rows[c["candidate_id"]]["features"]
        cases.append({"candidate_id": c["candidate_id"], "record": c["record"], "reference": c["reference"],
                      "start_s": c["start_s"], "end_s": c["end_s"],
                      "features": {k[len("tool_"):]: v for k, v in sorted(f.items()) if k.startswith("tool_")}})
    out = {"version": "2026-10-14",
           "note": "LTSTDB development candidates (split_v07, protocol B, record-wide reference) with their expert "
                   "class (ischemic, heart_rate_related, axis_or_conduction, noise_or_unreadable, other = no annotated "
                   "ST episode, axis/conduction marker or noise there) and the tool features of the fixed check set",
           "source_split": str(SPLIT.relative_to(CODE_DIR)), "n": len(cases), "cases": cases}
    OUT.write_text(json.dumps(out, indent=0) + "\n")
    print(f"{len(cases)} precedents from {len({c['record'] for c in cases})} records -> {OUT}")


if __name__ == "__main__":
    main()
