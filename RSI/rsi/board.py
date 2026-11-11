"""The leaderboard (workspace/leaderboard.jsonl): one line per decision; the last line of a harness id is current.

status: seed | rejected_search (not promoted on the search set) | rejected_validation | accepted |
        accepted_simplification | pending_validation
The current best is the most recently accepted harness (or the seed). The archive is every harness with a
search-set result; the "second" is the one with the highest search-set AUC other than the best.
"""
from __future__ import annotations

import json
import time

from .paths import LEADERBOARD

ACCEPTED = ("seed", "accepted", "accepted_simplification")


def lines() -> list[dict]:
    if not LEADERBOARD.exists():
        return []
    return [json.loads(x) for x in LEADERBOARD.read_text().splitlines() if x.strip()]


def current() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in lines():
        out[e["id"]] = e
    return out


def best() -> dict:
    acc = [e for e in lines() if e["status"] in ACCEPTED]
    if not acc:
        raise RuntimeError("no accepted harness yet (register the seed first)")
    return acc[-1]


def second() -> dict | None:
    b = best()["id"]
    rest = [e for e in current().values() if e["id"] != b and e.get("s_auc") is not None]
    return max(rest, key=lambda e: e["s_auc"]) if rest else None


def append(entry: dict) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "time": time.strftime("%Y-%m-%d %H:%M")}
    with LEADERBOARD.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
