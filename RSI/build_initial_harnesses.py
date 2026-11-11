"""Write the two starting harness directories from holter_agent.agent.prompts (run once; kept for the record).

h00  the seed of the search: one judgement session with a one-paragraph role, the shared context with the three
     quoted sources of version 0.6 (LTSTDB classes, ACC/AHA 1999, Panza 1992) and no rules on how to use them,
     the legend of the four images, the 13 base tools, and p_ischemic + label + evidence ids as output.
v08  harness 0.8 (scripts/run_review.py --harness v08 --toolset base) as files; the hand-designed reference.
     tests/test_harness.py checks that its prompts and tools equal those of the original code.

    python RSI/build_initial_harnesses.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from holter_agent.agent import prompts as P  # noqa: E402
from holter_agent.agent import toolsets  # noqa: E402
from holter_agent.agent.observe import IMAGE_KINDS  # noqa: E402

from rsi.paths import HARNESS_DIR  # noqa: E402


def image_legend(outro: bool) -> str:
    g = P._IMAGE_GUIDE
    return g["intro"] + "".join(g[k] for k in IMAGE_KINDS) + (g["outro"] if outro else "\n")


def write(hid: str, spec: dict, parts: dict[str, str], submit: dict, force: bool) -> None:
    d = HARNESS_DIR / hid
    if d.exists():
        if not force:
            print(f"{d} exists; use --force to rewrite")
            return
        shutil.rmtree(d)
    (d / "prompts").mkdir(parents=True)
    (d / "harness.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    for name, text in parts.items():
        (d / "prompts" / name).write_text(text)
    (d / "submit.json").write_text(json.dumps(submit, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {d}")


SEED_ROLE = """\
Investigate the candidate with the tools, then record your assessment with submit_assessment (stage "final"): \
p_ischemic, your probability that this ST change is ischemic; the label that best explains it, which must agree \
with p_ischemic ("ischemic" if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the evidence ids \
you relied on; and a short rationale. Then reply with a one-paragraph summary."""

SEED_SUBMIT = {
    "name": "submit_assessment",
    "description": "Record your assessment of the candidate under review. p_ischemic is your probability that the "
                   "ST change is ischemic. label is the most likely explanation and must agree with p_ischemic: "
                   "'ischemic' when p_ischemic >= 0.5, otherwise the most likely non-ischemic cause. Cite the "
                   "evidence ids you relied on.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "stage": {"type": "string", "enum": ["initial", "final"]},
            "p_ischemic": {"type": "number", "description": "0 to 1"},
            "label": {"type": "string", "enum": list(P.GRADED_LABELS)},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["candidate_id", "stage", "p_ischemic", "label", "evidence_ids", "rationale"],
        "additionalProperties": False,
    },
}


def seed_labels() -> str:
    lines = P._LABEL_TEXT.split("\n")
    out = ["Labels (the label must agree with p_ischemic):"]
    skip = False
    for line in lines[1:]:
        if line.startswith("- indeterminate_ambiguous"):
            skip = True  # the entry and its continuation line
            continue
        if skip and not line.startswith("- "):
            continue
        skip = False
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # ---------------------------------------------------------------- h00 (seed)
    context = P._CONTEXT.replace("24-hour 12-lead Holter recordings", "24-hour Holter recordings (2 to 12 leads)")
    context = context.replace(P._SEE_NUMBERS, P._SEE_IMAGES)
    labels = seed_labels()
    assert "indeterminate_ambiguous" not in labels and labels.count("\n- ") == 7, labels
    write("h00", {
        "id": "h00", "parent": None,
        "summary": "Seed: one judgement session with a one-paragraph role; shared context with the quoted sources of "
                   "v0.6 and no rules on using them; image legend; 13 base tools; p_ischemic + label + evidence ids.",
        "system": {"judgement": ["context.md", "images.md", "judgement.md", "labels.md"]},
        "sessions": {"first_impression": False, "dossier": [], "verify": "none"},
        "tools": {"toolset": "base", "extra": [], "descriptions": {}},
        "limits": {"max_tool_calls": 30, "stall_turns": 3},
    }, {"context.md": context, "images.md": image_legend(outro=False), "judgement.md": "\n" + SEED_ROLE + "\n\n",
        "labels.md": labels}, SEED_SUBMIT, args.force)

    # ---------------------------------------------------------------- v08 (harness 0.8, reference)
    roles = P.ROLES_BY_GUIDANCE["v08"]
    write("v08", {
        "id": "v08", "parent": None,
        "summary": "Harness 0.8 (hand-designed reference, not part of the search archive).",
        "system": {"judgement": ["context.md", "images.md", "judgement.md", "labels.md"],
                   "verification": ["context.md", "images.md", "verification.md", "labels.md"]},
        "sessions": {"first_impression": False, "dossier": ["standard", *toolsets.EXTRA_TOOLS], "verify": "uncertain",
                     "verify_band": [0.2, 0.8], "max_verify_rounds": 3},
        "tools": {"toolset": "base", "extra": list(toolsets.EXTRA_TOOLS), "descriptions": {}},
        "limits": {"max_tool_calls": 30, "stall_turns": 3},
    }, {"context.md": P.context("both", "v08"), "images.md": image_legend(outro=True),
        "judgement.md": "\n" + roles["judgement"] + "\n\n", "verification.md": "\n" + roles["verification"] + "\n\n",
        "labels.md": P._LABEL_TEXT_V08}, P.SUBMIT_TOOL_GRADED, args.force)


if __name__ == "__main__":
    main()
