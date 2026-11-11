# HolterAgent

Code, harnesses, data splits, and result files for the paper
**"HolterAgent: An Evidence-Traceable LLM Agent for Myocardial Ischemia Detection in Long-Term ECG Time Series"**
(submitted to ICDE 2027).

HolterAgent reduces a 24-hour ECG recording to candidate ST episodes with deterministic code, reviews each candidate
with an LLM task agent that calls deterministic tools and cites their stored results (evidence ledger identifiers
E1, E2, ...), and improves the task agent's harness (knowledge, tools and evidence, review procedure) by recursive
self-improvement with archive search.

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `holter_agent/signal`, `holter_agent/trend` | Window table: notch filter, beat selection, median beats, fiducials, rate-adjusted ST level, quality flags; LTSTDB episode rule and candidate merging | Sec. II-B, Alg. 1 |
| `holter_agent/agent` | Task agent: deterministic tool kit, evidence ledger, prompts, session loop, code checks of the assessment, model backends | Sec. II-C, Alg. 2, Table I |
| `holter_agent/render` | Images returned by the image tools | Sec. II-C |
| `RSI/cloop.py`, `RSI/rsi/` | Harness search: archive, parent selection, improver round, code checks, evaluation, final selection | Sec. II-D |
| `RSI/tool_library/`, `RSI/kb/` | Tools the improver may add, and the verified literature cards | Sec. II-D |
| `RSI/harnesses/` | Every harness of the archive search (`v08p` = initial harness, `a01`-`a15`; `a03` = HolterAgent), of hill climbing (`p01`-`p09`), and the picks of the aspect ablation | Sec. III-E/F |
| `RSI/splits/` | Candidate lists: search set (`search_v2`), validation set (`validation_v4dev`), test sets LTSTDB-22 (`ltstdb_unused`), EDB (`edb`), composite (`mimic`) | Sec. III-A |
| `scripts/` | Construction of the candidates and of the composite test set from the PhysioNet databases | Sec. III-A |
| `baselines/` | The nine baselines | Sec. III-C |
| `results/` | Archive boards, final selection, test results of the agent versions, and the aspect-ablation summary | Sec. III-D/E/F |

## Data

All databases are public on PhysioNet. Download them into one folder and set `HOLTER_DATA_ROOT` (default `./data`):

- Long-Term ST Database (LTSTDB): https://physionet.org/content/ltstdb/
- European ST-T Database (EDB): https://physionet.org/content/edb/
- STAFF III Database: https://physionet.org/content/staffiii/
- MIT-BIH Normal Sinus Rhythm Database: https://physionet.org/content/nsrdb/
- Cerebral Vasoregulation in Diabetes: https://physionet.org/content/cerebral-vasoreg-diabetes/
- EPHNOGRAM: https://physionet.org/content/ephnogram/

`scripts/mimic_records.py` and `scripts/staffiii_records.py` build the single-lead records of the composite test set;
`scripts/build_features.py` builds the window tables; `scripts/ltstdb_cases.py`, `scripts/edb_cases.py`, and
`scripts/mimic_cases.py` list the candidates (the resulting lists are in `RSI/splits/`).

## Setup

```bash
pip install -e .
```

The task agent calls an OpenAI Responses-API compatible endpoint (the paper uses gpt-5.6-luna with high reasoning
effort). Set

```bash
export LLM_POOL_URL=<base URL of the endpoint>
export LLM_POOL_API_KEY=<API key>
export LLM_POOL_PATH=/v1/responses   # path of the Responses endpoint, if different
```

The improver agent of the harness search runs Claude Code (`claude -p`) with Claude Opus 5.5.

## Running

```bash
# review the candidates of a split with one harness (one run)
python RSI/evaluate.py a03 --split edb --rep 1 --concurrency 10

# harness search: register the initial harness, run rounds, then select on the validation set
python RSI/cloop.py init v08p
python RSI/cloop.py round --focus OPEN
python RSI/cloop.py finalize --top 3

# test evaluation of selected harnesses (three runs each)
python RSI/final_eval.py v08p a03 --sets ltstdb_unused edb mimic
```

Each script documents its options in its header.

## Notes

- The candidate screening, the tools, and the code checks are deterministic; the LLM decisions vary between runs, so
  every harness is run three times and the mean probability is the score.
- The test sets were not used for the harness search, the final selection, or the training and tuning of the
  baselines.
