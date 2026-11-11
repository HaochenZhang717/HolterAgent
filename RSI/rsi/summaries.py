"""Case summaries of a search-set run, for the outer loop: what the model saw, did and concluded, next to the
LTSTDB reference class. Written only for the search split (paths.OPEN_SPLITS)."""
from __future__ import annotations

import json
from pathlib import Path

RESULT_CHARS = 700  # tool results are cut to this length in the summaries
MESSAGE_CHARS = 1500


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f" ...[{len(s) - n} more chars]"


def _trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def case_markdown(case: dict, result: dict, trace: list[dict]) -> str:
    cid = case["candidate_id"]
    st = result["state"]
    ev = {e["id"]: e for e in st["evidence"]}
    finals = [a for a in st["assessments"] if a["candidate_id"] == cid and a["stage"] == "final"]
    p = finals[-1]["p_ischemic"] if finals else None
    ref = case["reference"]
    y = ref == "ischemic"
    kind = "no assessment" if p is None else ("FN" if y and p < 0.5 else "FP" if not y and p >= 0.5 else
                                              "TP" if y else "TN")
    lines = [f"# {cid}: reference {ref}, p_ischemic {p}, {kind}", ""]
    cand = next((e for e in st["evidence"] if e["tool"] == "candidate" and e["args"].get("candidate_id") == cid), None)
    if cand:
        lines += [f"Candidate as shown ({cand['id']}): `{_cut(_j(cand['result']), 1200)}`", ""]
    cost = result.get("codex", {}).get("credits") if "codex" in result else \
        (result.get("pool") or {}).get("credits_equivalent") if "pool" in result else \
        f"tokens {_j((result.get('openai') or {}).get('usage'))}"
    lines += [f"Credits {cost}; stop reasons {_j(st['stop_reasons'].get(cid, {}))}; "
              f"needs review: {st['needs_review'].get(cid)}", ""]
    for k, a in enumerate(finals):
        lines += [f"## Assessment {k + 1} ({a['session']}): {a['label']}, p_ischemic {a['p_ischemic']}",
                  f"Rationale: {a['rationale']}"]
        if a.get("evidence_table"):
            for side, fs in a["evidence_table"].items():
                for f in (fs if isinstance(fs, list) else [fs]):  # open-weight models sometimes give plain strings
                    lines.append(f"- {side}: {f.get('evidence_id')}: {f.get('finding')}" if isinstance(f, dict)
                                 else f"- {side}: {_cut(str(f), 400)}")
        lines += [f"Cited: {', '.join(a['evidence_ids'])}", ""]
    lines += ["## Evidence (in order)"]
    for e in st["evidence"]:
        if e["session"] == "setup":
            continue
        img = f" [image {', '.join(e['images'])}]" if e.get("images") else ""
        lines.append(f"- {e['id']} ({e['session']}) {e['tool']} {_j({k: v for k, v in e['args'].items() if k != 'record_id'})}"
                     f"{img}: {_cut(_j(e['result']), RESULT_CHARS)}")
    msgs = [(r["session"], i["text"]) for r in trace if "items" in r for i in r["items"]
            if i.get("type") == "agent_message" and i.get("text")]
    if msgs:
        lines += ["", "## Model messages"]
        for session, text in msgs:
            lines.append(f"- ({session}) {_cut(text, MESSAGE_CHARS)}")
    return "\n".join(lines) + "\n"


def write(run_dir: Path, cases: list[dict], out_dir: Path) -> Path:
    """One markdown file per case and an index sorted by error |reference - p| (largest first)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(exist_ok=True)
    rows = []
    for c in cases:
        cid = c["candidate_id"]
        f = run_dir / f"{cid}.json"
        if not f.exists():
            rows.append((1.0, cid, c["reference"], None, None, "not run"))
            continue
        result = json.loads(f.read_text())
        (out_dir / "cases" / f"{cid}.md").write_text(
            case_markdown(c, result, _trace(run_dir / f"{cid}.trace.jsonl")))
        pred = next(p for p in result["predictions"] if p["candidate_id"] == cid)
        p = pred.get("p_ischemic")
        y = c["reference"] == "ischemic"
        err = 0.5 if p is None else abs(float(y) - p)
        kind = "none" if p is None else ("FN" if y and p < 0.5 else "FP" if not y and p >= 0.5 else "ok")
        rows.append((err, cid, c["reference"], p, pred.get("final_label"), kind))
    rows.sort(key=lambda r: -r[0])
    idx = ["| error | candidate | reference | p_ischemic | label | |", "|---|---|---|---|---|---|"]
    idx += [f"| {e:.2f} | [{cid}](cases/{cid}.md) | {ref} | {p} | {lab} | {k} |" for e, cid, ref, p, lab, k in rows]
    path = out_dir / "index.md"
    path.write_text(f"# Search-set cases of {run_dir.name}, largest error first\n\n" + "\n".join(idx) + "\n")
    return path
