"""Locations of the harness search, anchored at code/RSI."""
from __future__ import annotations

import json
import os
from pathlib import Path

RSI_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = RSI_DIR.parent
HARNESS_DIR = RSI_DIR / "harnesses"
SPLITS_DIR = RSI_DIR / "splits"
WORKSPACE = RSI_DIR / "workspace"
# Model track (2026-10-17). Unset: the task model is GPT-6 Astra through a Responses-API endpoint (runs/, workspace/failures/,
# cloop/). RSI_TRACK=<name>: the task model of RSI/tracks.json[<name>] (e.g. q27 = Qwen3.5-27B served by vLLM on
# GPU server), with its own run, failure-summary and search directories, so results of different task models never mix.
TRACK = os.environ.get("RSI_TRACK", "")
TRACKS_FILE = RSI_DIR / "tracks.json"
RUNS_DIR = RSI_DIR / ("runs" if not TRACK else f"runs_{TRACK}")
FAILURES_DIR = WORKSPACE / ("failures" if not TRACK else f"failures_{TRACK}")
CLOOP_DIR = RSI_DIR / ("cloop" if not TRACK else f"cloop_{TRACK}")
LEADERBOARD = WORKSPACE / "leaderboard.jsonl"
DATA_ROOT = Path(os.environ.get("HOLTER_DATA_ROOT", CODE_DIR / "data"))  # PhysioNet databases, see README
LTSTDB_DIR = DATA_ROOT / "ltstdb"
LTSTDB_CACHE = CODE_DIR / "cache" / "features_ltstdb"
# External test set (2026-10-16): European ST-T Database, candidates from scripts/edb_cases.py.
EDB_DIR = DATA_ROOT / "edb"
EDB_CACHE = CODE_DIR / "cache" / "features_edb"
# Extra external test (2026-10-21): STAFF III balloon-inflation pseudo-records, scripts/staffiii_records.py.
STAFF_DIR = DATA_ROOT / "staffiii_holter"
STAFF_CACHE = CODE_DIR / "cache" / "features_staffiii"
# Third test set (2026-10-30): STAFF III ischemia vs lab-provoked non-ischemic ST changes, scripts/mimic_records.py.
MIMIC_DIR = DATA_ROOT / "mimic_holter"
MIMIC_CACHE = CODE_DIR / "cache" / "features_mimic"

SPLITS = ("search", "validation", "confirm", "test", "search_v2", "validation_v2", "validation_v3", "edb",
          "validation_v4", "ltstdb_unused", "staffiii", "mimic", "validation_v4dev")  # 2026-10-21: ltstdb_unused = the 22 records outside
# split_v07; staffiii = STAFF III balloon-inflation pseudo-records (external ischemia-sensitivity test);
# mimic (2026-10-30) = third test set, scripts/mimic_records.py
# Candidate rule per split (2026-10-30): LTSTDB protocol B everywhere except the third test set, which uses protocol A
# (scripts/mimic_cases.py). evaluate.py sets CANDIDATE_PROTOCOL for its children; harness.py and ext_context.py read it.
SPLIT_PROTOCOL = {"mimic": "A"}


def candidate_protocol() -> str:
    return os.environ.get("CANDIDATE_PROTOCOL", "B")


# Splits whose per-case results the outer loop may read. The others are scored in aggregate only.
OPEN_SPLITS = ("search", "search_v2")  # search_v2: the search split of the constrained search (cloop.py, 2026-10-16)


def run_dir(split: str, harness_id: str, rep: int) -> Path:
    return RUNS_DIR / split / f"{harness_id}_r{rep}"


def track_config() -> dict:
    """The task-model settings of the current track ({} for the default GPT-6 Astra track)."""
    if not TRACK:
        return {}
    tracks = json.loads(TRACKS_FILE.read_text())
    if TRACK not in tracks:
        raise KeyError(f"RSI_TRACK={TRACK!r} is not in {TRACKS_FILE}")
    return tracks[TRACK]
