"""Default locations, anchored at the code repository so scripts work from any directory.

    <project>/
      data/                  Holter records (HOLTER_DATA_DIR)
      code/                  this repository
        cache/               regenerable caches (git-ignored)
        outputs/             results of runs and evaluations (git-ignored)
      docs/                  documentation repository

Set HOLTER_DATA_DIR to use records stored elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("HOLTER_DATA_DIR", PROJECT_DIR / "data"))
CACHE_DIR = CODE_DIR / "cache"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
OUTPUT_DIR = CODE_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
EVAL_DIR = OUTPUT_DIR / "eval"
