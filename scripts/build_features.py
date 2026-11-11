"""Scan records and cache per-window features.

Example (defaults: records in $HOLTER_DATA_DIR, cache in code/cache/features):
    python scripts/build_features.py --workers 6
"""
from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from holter_agent import DEFAULT_CONFIG
from holter_agent.data import RecordStore
from holter_agent.paths import DATA_DIR, FEATURE_CACHE_DIR
from holter_agent.trend.features import load_or_compute


def _one(data_dir: str, name: str, cache_dir: str) -> tuple[str, float, int]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    record = RecordStore(data_dir).get(name)
    table = load_or_compute(record, DEFAULT_CONFIG, cache_dir)
    return name, time.time() - t0, table.n_windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_DIR), help="directory with .hea/.dat/.atr files")
    ap.add_argument("--cache", default=str(FEATURE_CACHE_DIR), help="output directory for feature files")
    ap.add_argument("--records", nargs="*", help="record ids (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    store = RecordStore(args.data)
    names = [store.resolve(r) for r in args.records] if args.records else store.names
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, args.data, n, args.cache) for n in names]
        for i, fut in enumerate(as_completed(futures), 1):
            name, secs, n_windows = fut.result()
            print(f"[{i}/{len(names)}] {name}: {n_windows} windows in {secs:.1f}s", flush=True)


if __name__ == "__main__":
    main()
