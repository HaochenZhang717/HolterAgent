"""Credit ledger: what every run under RSI/runs has spent, and what is left (budget.json)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .paths import RSI_DIR, RUNS_DIR

SEARCH_PHASE = ("search", "validation")


def limits() -> dict:
    return json.loads((RSI_DIR / "budget.json").read_text())


def spent_by_run() -> dict[str, float]:
    """Credits per run directory (split/harness_rK), from the per-candidate results."""
    out: dict[str, float] = defaultdict(float)
    for f in RUNS_DIR.glob("*/*/*.json"):
        if f.name == "metrics.json":
            continue
        c = (json.loads(f.read_text()).get("codex") or {}).get("credits")
        if c is not None:
            out[f"{f.parent.parent.name}/{f.parent.name}"] += c
    return dict(out)


def status() -> dict:
    lim = limits()
    runs = spent_by_run()
    by_split: dict[str, float] = defaultdict(float)
    for k, v in runs.items():
        by_split[k.split("/")[0]] += v
    total = sum(runs.values())
    search_phase = sum(by_split[s] for s in SEARCH_PHASE)
    return {"cap": lim["cap"], "final_reserve": lim["final_reserve"], "spent_total": total,
            "spent_by_split": dict(by_split), "spent_by_run": runs,
            "left_for_search": lim["cap"] - lim["final_reserve"] - search_phase,
            "left_total": lim["cap"] - total}


def allowance(split: str) -> float:
    """Credits a new run on this split may still spend."""
    s = status()
    return s["left_for_search"] if split in SEARCH_PHASE else s["left_total"]
