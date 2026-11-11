"""Run the baselines that need no LLM: inputs, rule baselines, classical classifiers, and a summary table.

    python -m baselines.run_all --data <ltstdb dir> --cache cache/features_ltstdb --workers 16

Writes outputs/baselines/<baseline>/{result,predictions}.json, outputs/baselines/summary.json and
outputs/baselines/summary.md. The trivial references (every candidate ischemic; ischemic exactly in patients
with coronary disease, which uses the header) are added to the table for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from baselines import evidence, feature_ml, rules
from baselines.common import OUT, ROOT, evaluate, load_cases, save

sys.path.insert(0, str(ROOT / "scripts"))
from test_candidate_features import clinical_features  # noqa: E402


def trivial(data: Path) -> list[dict]:
    cases = load_cases()
    allisch = {c.candidate_id: True for c in cases}
    r1 = evaluate("always_ischemic", cases, label=allisch, notes="every candidate ischemic")
    cad = {c.record: bool(clinical_features((data / f"{c.record}.hea").read_text())["clin_cad"]) for c in cases}
    r2 = evaluate("coronary_patient", cases, label={c.candidate_id: cad[c.record] for c in cases},
                  notes="ischemic exactly in patients with coronary disease, infarction or angina (header, Comments "
                        "left out); uses clinical information")
    save("always_ischemic", r1, {})
    save("coronary_patient", r2, {c.candidate_id: {"coronary": cad[c.record]} for c in cases})
    return [r1, r2]


def table(results: list[dict]) -> str:
    lines = ["| baseline | dev AUC [95% CI] | test AUC [95% CI] | test Se | test Sp | test PPV | test F1 [95% CI] | "
             "test balanced acc | notes |", "|" + "---|" * 9]
    for r in results:
        cells = [r["name"]]
        for split in ("dev", "test"):
            d = r["splits"][split]
            cells.append(f"{d['auc']:.2f} [{d['auc_ci'][0]:.2f}, {d['auc_ci'][1]:.2f}]" if "auc" in d else "-")
        t = r["splits"]["test"]
        k = t.get("rule") or t.get("at_dev_threshold")  # rules: their own decision; models: dev threshold
        cells += [f"{k['se']:.2f}", f"{k['sp']:.2f}", f"{k['ppv']:.2f}",
                  f"{k['f1']:.2f} [{t['f1_ci'][0]:.2f}, {t['f1_ci'][1]:.2f}]", f"{k['balanced']:.2f}",
                  r.get("notes", "").replace("|", "/")[:120]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    sys.argv = [sys.argv[0], "--data", args.data, "--cache", args.cache, "--workers", str(args.workers)]
    evidence.main()
    results = trivial(Path(args.data)) + rules.run() + feature_ml.run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    md = ("Baselines without an LLM on the LTSTDB development (36 records, 587 candidates) and test (25 records, "
          "234 candidates) sets, split by patient. Learned models: trained on development only; development values "
          "are record-grouped out-of-fold; thresholds chosen on development. Rules: their own decision.\n\n"
          + table(results))
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
