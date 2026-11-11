"""Money spent by a paid model track (RSI_TRACK with "price_yuan_per_m" in RSI/tracks.json), from the token usage of
every model request in the traces (judgement steps and the yes/no read-out), priced per request by its input length.

    RSI_TRACK=q37f python -m rsi.spend            # print the total
    RSI_TRACK=q37f python -m rsi.spend --check    # exit 1 when the track's budget_yuan is reached
"""
from __future__ import annotations

import json
import sys

from .paths import RUNS_DIR, track_config

TIER_TOKENS = 32_000  # Model Studio's first price tier of qwen3.7-flash: input up to 32K tokens per request


def request_yuan(price: dict, n_in: int, n_out: int) -> float:
    hi = n_in > TIER_TOKENS
    return (n_in * price["input_gt_32k" if hi else "input_le_32k"] +
            n_out * price["output_gt_32k" if hi else "output_le_32k"]) / 1e6


def spent(cfg: dict | None = None) -> float:
    cfg = cfg or track_config()
    price = cfg.get("price_yuan_per_m")
    if not price:
        return 0.0
    total = 0.0
    for trace in RUNS_DIR.glob("*/*/*.trace.jsonl"):
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            u = row.get("usage") if "calls" in row else (row.get("readout") or {}).get("usage")
            if u and u.get("input_tokens") is not None:
                total += request_yuan(price, int(u["input_tokens"]), int(u.get("output_tokens") or 0))
    return total


def main() -> None:
    cfg = track_config()
    total, budget = spent(cfg), cfg.get("budget_yuan")
    print(f"spent {total:.2f} yuan" + (f" of a {budget} yuan budget" if budget else ""))
    if "--check" in sys.argv and budget and total >= budget:
        sys.exit(1)


if __name__ == "__main__":
    main()
