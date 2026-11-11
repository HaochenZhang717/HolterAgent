"""Constrained harness search (from 2026-10-16): Claude Code (`claude -p`) improves our hand-written harness; the
task agent is that harness, running GPT-6 Astra through a Responses-API endpoint.

Constraints (meeting of 2026-10-16: a small search space and explicit instructions):
- one explicit focus per round (RSI/cloop/focuses.json, or --focus-text), with the parent's failure cases on the
  search split;
- the improver may change only how the model uses its tools: prompt text, tool descriptions, which base and library
  tools go into the dossier, and the verification mode and band (improver/cloop_system_prompt.md);
- no code: the library tools it picks are copied verbatim from RSI/tool_library by this script;
- medical statements only from the verified literature cards (RSI/kb/cards.json), cited by card id with a verbatim
  quote, checked here.
Data: search on search_v2 (300 development candidates; its per-case results are open to the improver), select on
validation_v2 (124; aggregate only), never confirm or test. The development candidates outside both
(validation_v3 minus validation_v2) are kept for the final comparison outside this loop.

Model tracks (2026-10-17): with RSI_TRACK=<name> (RSI/tracks.json) the task agent runs an open-weight model (e.g. q27:
Qwen3.5-27B on vLLM) and the search lives in RSI/cloop_<name>/. A track may set "scope": "extended", which also lets the
improver (improver/cloop_extended_system_prompt.md)
- change every sessions field, the limits and the knowledge settings (which sessions see the knowledge base, cards,
  precedents, how many);
- add literature cards in the harness's knowledge/cards.json, each with a PubMed id and a quote that this script finds
  verbatim in the PubMed abstract (fetched from NCBI E-utilities);
- write new deterministic tools in tools/ (sandboxed like every extension tool; this script runs each new tool on three
  search-split candidates before the round is evaluated); the improver gets WebSearch and WebFetch for the literature.

Decision (fixed before the first round): a change is promoted when its paired AUC difference to the parent on the
search split is >= +0.01 with P(>0) >= 0.8; it is accepted when, in addition, its validation AUC is no more than 0.01
below the parent's. Otherwise it is rejected and the parent stays.

Archive search (track setting "search_mode": "archive", user 2026-10-20; parent selection of the Darwin Gödel Machine,
Zhang et al. arXiv 2505.22954, in the form of DGM-H, Zhang et al. arXiv 2603.19461, App. A.2): no acceptance rule.
Every child that passes the code checks is evaluated on the search split and joins the archive. Each round draws its
parent from the archive with p_i ∝ sigmoid(λ(α_i − α_mid)) / (1 + n_i), where α_i is the search-split AUC (mean score
over the runs), α_mid the mean of the top-m α, n_i the number of archived children of i (λ = 10, m = 3). The validation
split is not used during the search: `finalize` evaluates the top-k harnesses by search AUC on it once and picks the
best. The improver sees only the harnesses, runs and rounds of its own track.

    python RSI/cloop.py init v11s                # register a base harness (evaluates search_v2 / validation_v2 if needed)
    python RSI/cloop.py round --focus F1         # one round
    python RSI/cloop.py status
    python RSI/cloop.py finalize --top 3         # archive search: validation of the top harnesses, final pick
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improver import backend  # noqa: E402
from rsi import scoring  # noqa: E402
from rsi.harness import SESSION_KEYS, Harness, HarnessError  # noqa: E402
from rsi.paths import (CLOOP_DIR, CODE_DIR, EDB_DIR, STAFF_DIR, MIMIC_DIR, FAILURES_DIR, HARNESS_DIR, LTSTDB_DIR, RSI_DIR, RUNS_DIR,  # noqa: E402
                       SPLITS_DIR, TRACK, WORKSPACE, run_dir, track_config)

TRACK_CFG = track_config()  # {} = GPT-6 Astra through the pool; else the open-weight task model of RSI_TRACK
CLOOP = CLOOP_DIR
BOARD = CLOOP / "board.jsonl"
ROUNDS = CLOOP / "rounds"
LIBRARY = RSI_DIR / "tool_library"
CARDS = RSI_DIR / "kb" / "cards.json"
SEARCH = "search_v2"
# validation_v4 from 2026-10-18 (RSI/build_validation_v4.py); a track may name another selection split (2026-11-06: the
# aspect ablation selects on validation_v4dev, the 104 development candidates of validation_v4)
SELECT = TRACK_CFG.get("select_split", "validation_v4")
PROMOTE_DELTA, PROMOTE_P, SELECT_TOLERANCE = 0.01, 0.8, 0.01
CONCURRENCY = int(TRACK_CFG.get("concurrency", 16))
PREFIX = TRACK_CFG.get("harness_prefix", "c")
PY = sys.executable
SCOPE = TRACK_CFG.get("scope", "constrained")
EXTENDED = SCOPE == "extended"
# single_decision (track setting, user 2026-10-17): the search works on the judgement session alone; verification
# and a first impression stay off, so only the dossier may change among the session settings
SINGLE = bool(TRACK_CFG.get("single_decision"))
ALLOWED_SESSION_KEYS = {"dossier", "dossier_exclude"} if SINGLE else set(SESSION_KEYS) if EXTENDED else {
    "dossier", "verify", "verify_band", "max_verify_rounds"}
ALLOWED_TOOL_KEYS = {"descriptions", "extra"}
# lock_limits (track setting, user 2026-10-18): tool use is not constrained, so the limits stay as the seed has them
ALLOWED_TOP_KEYS = ({"knowledge"} if TRACK_CFG.get("lock_limits") else {"knowledge", "limits"}) if EXTENDED else set()
SYSTEM_PROMPT = RSI_DIR / "improver" / ("cloop_extended_system_prompt.md" if EXTENDED else "cloop_system_prompt.md")
ARCHIVE = TRACK_CFG.get("search_mode", "hill") == "archive"  # see the module docstring
ARCHIVE_CFG = {"lambda": 10.0, "top_m": 3, "seed": 2026} | TRACK_CFG.get("archive", {})
# aspects (track setting, user 2026-11-06, aspect ablation): the aspects of the OPEN focus (A knowledge, B tools and
# evidence, C reasoning procedure) that a round may change; check() rejects a child that touches another aspect's own
# resources (aspect_problems). Default: all three, as in p56a.
ASPECTS = "".join(sorted(set(TRACK_CFG.get("aspects", "ABC"))))
ASPECT_NAMES = {"A": "knowledge", "B": "tools and evidence", "C": "reasoning procedure"}
# the improver's system prompt in an archive search: there is no acceptance decision
ARCHIVE_PROMPT_EDITS = [
    ("evaluates it and\ndecides; you never run the evaluation.",
     "evaluates it\non the search set and adds it to an archive of harnesses (task.md); you never run the evaluation."),
    ("Cards you add stay\n   with the harness and its descendants when the round is accepted.",
     "Cards you add stay\n   with the harness and its descendants."),
]
# how the case score is obtained, by track (2026-10-19): logit read-out (open/API models) or the stated probability
SCORE_TEXT = {
    "logit": ("submits a two-way decision (ischemic or non_ischemic)\nwith the most likely explanation.",
              "After the session,\ncode appends one fixed question (\"is the ST change of this candidate caused by myocardial "
              "ischemia? yes or no\") and\nreads the probability of \"yes\" from the model's next-token log-probabilities; "
              "that probability is the case's score."),
    "stated": ("submits p_ischemic, its probability that the ST\nchange is ischemic, with the most likely explanation.",
               "The submitted\np_ischemic is the case's score (the model's log-probabilities are not available for this "
               "model)."),
}


ASPECT_RULE = """

## Aspects open in this search

Only {open} may be changed in this search. Code rejects a child that changes a resource of another aspect:
- A (knowledge): the "Reference knowledge" section of prompts/context.md (from its heading to the line "How to use
  this knowledge:"), harness.json "knowledge", and knowledge/cards.json;
- B (tools and evidence): harness.json "sessions" (dossier, dossier_exclude) and "tools" (descriptions, extra), library
  tools, and tools/;
- C (reasoning procedure): prompts/judgement.md, prompts/labels.md, prompts/images.md, new or removed prompt parts, and
  harness.json "system".
The rest of prompts/context.md (its opening paragraph and the "How to use this knowledge" guidance) may be edited in
any round, but only as far as the change of the open aspect needs it. Keep both headings of context.md unchanged.
"""


def system_prompt() -> str:
    text = SYSTEM_PROMPT.read_text()
    if "{SUBMISSION}" in text:
        sub, score = SCORE_TEXT[TRACK_CFG.get("score", "stated")]
        text = text.replace("{SUBMISSION}", sub).replace("{SCORE}", score)
    if ARCHIVE:
        for old, new in ARCHIVE_PROMPT_EDITS:
            if old not in text:
                raise RuntimeError(f"archive search: system prompt text to replace not found: {old[:60]!r}")
            text = text.replace(old, new)
    if ASPECTS != "ABC":
        text += ASPECT_RULE.format(open=" and ".join(f"aspect {a} ({ASPECT_NAMES[a]})" for a in ASPECTS))
    return text
ABSTRACTS = CLOOP / "abstracts"  # PubMed abstracts fetched to check the quotes of new cards
CARD_FIELDS = ("id", "topics", "title", "statement", "quote", "direction", "applies_when", "limits", "source")
# Sources a card may quote besides PubMed / PubMed Central (2026-10-19: textbooks on NCBI Bookshelf and official
# society guidelines, reliable sources only): the page must be on one of these hosts and contain the quote verbatim.
SOURCE_HOSTS = {"www.ncbi.nlm.nih.gov": "/books/",          # NCBI Bookshelf (StatPearls and other textbooks)
                "www.ahajournals.org": "", "ahajournals.org": "",   # AHA / ACC-AHA guidelines and statements
                "www.jacc.org": "", "jacc.org": "",                 # ACC / ACC-AHA guidelines
                "academic.oup.com": "/eurheartj/",                  # ESC guidelines in the European Heart Journal
                "www.escardio.org": ""}                             # ESC guidelines pages
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}&rettype=abstract&retmode=text"
# open-access full text in PubMed Central (2026-10-18): a card may quote the article body when it gives its pmcid
EFETCH_PMC = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={}&rettype=xml"
SEALED = ("validation", "validation_v2", "validation_v3", "validation_v4", "validation_v4dev", "confirm", "test", "edb",
          "ltstdb_unused", "staffiii", "mimic")  # never readable by the improver
MAX_TOOL_RESULT_CHARS = 20000

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "harness_id": {"type": "string"},
        "summary": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
        "library_tools": {"type": "array", "items": {"type": "string"}},
        "medical_statements": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"}, "card_id": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["statement", "card_id", "quote"]}},
        "expected_effect": {"type": "string"},
        "new_tools": {"type": "array", "items": {"type": "string"}},
        "new_cards": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["harness_id", "summary", "changes", "library_tools", "medical_statements", "expected_effect"],
}


# ------------------------------------------------------------------------------------------------------ board
def board() -> list[dict]:
    return [json.loads(x) for x in BOARD.read_text().splitlines()] if BOARD.exists() else []


def append(entry: dict) -> None:
    CLOOP.mkdir(exist_ok=True)
    with BOARD.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def best() -> dict:
    ok = [e for e in board() if e["status"] in ("base", "accepted")]
    if not ok:
        sys.exit("no base harness: run `cloop.py init <harness>` first")
    return ok[-1]


def archive() -> list[dict]:
    """Archive search: the base and every child that passed the checks and was evaluated on the search split."""
    return [e for e in board() if e["status"] in ("base", "archived") and e.get("s_auc") is not None]


def select_parent(rnd: int) -> tuple[dict, list[dict]]:
    """DGM-H parent selection (module docstring); the draw is reproducible from (seed, round). Returns the parent and
    the table of weights (written to the round directory)."""
    import math

    import numpy as np
    arch = archive()
    if not arch:
        sys.exit("no base harness: run `cloop.py init <harness>` first")
    lam, m = float(ARCHIVE_CFG["lambda"]), int(ARCHIVE_CFG["top_m"])
    mid = float(np.mean(sorted((e["s_auc"] for e in arch), reverse=True)[:m]))
    ids = {e["id"] for e in arch}
    table = []
    for e in arch:
        n = sum(1 for c in arch if c.get("parent") == e["id"] and c["id"] in ids)
        table.append({"id": e["id"], "s_auc": e["s_auc"], "children": n,
                      "weight": 1 / (1 + math.exp(-lam * (e["s_auc"] - mid))) / (1 + n)})
    w = np.array([t["weight"] for t in table])
    for t, p in zip(table, w / w.sum()):
        t["p"] = float(p)
    k = int(np.random.default_rng([int(ARCHIVE_CFG["seed"]), rnd]).choice(len(table), p=w / w.sum()))
    return arch[k], table


def isolation(keep: set[str]) -> tuple[list[Path], list[str]]:
    """Archive search ("from scratch", user 2026-10-20): the improver may read only the harnesses of this track's archive
    (and the new one), and nothing of other tracks (their runs, failure summaries and rounds). Returns the harness
    directories to add and the deny rules."""
    deny = [f"Read({backend.rule_path(d)}/**)" for d in sorted(HARNESS_DIR.iterdir()) if d.is_dir() and d.name not in keep]
    others = [d for d in RSI_DIR.iterdir() if d.is_dir() and re.match(r"(cloop|runs)(_|$)", d.name)
              and d.resolve() not in (CLOOP.resolve(), RUNS_DIR.resolve())]
    others += [d for d in WORKSPACE.glob("failures*") if d.is_dir() and d.resolve() != FAILURES_DIR.resolve()]
    deny += [f"Read({backend.rule_path(d)}/**)" for d in sorted(others)]
    return [HARNESS_DIR / h for h in sorted(keep)], deny


def lineage(hid: str) -> list[str]:
    by_id = {e["id"]: e for e in board()}
    out = [hid]
    while by_id.get(out[-1], {}).get("parent"):
        out.append(by_id[out[-1]]["parent"])
    return out[::-1]


def next_id() -> str:
    used = [p.name for p in HARNESS_DIR.iterdir()] + [e["id"] for e in board()]
    nums = [int(n[len(PREFIX):]) for n in used if re.fullmatch(PREFIX + r"\d{2,}", n)]
    return f"{PREFIX}{max(nums, default=0) + 1:02d}"


# ------------------------------------------------------------------------------------------------------ evaluation
class IncompleteRun(RuntimeError):
    pass


REPS = int(TRACK_CFG.get("reps", 1))  # runs per harness and split; scores are averaged over them (2026-10-18)


def evaluate(hid: str, split: str, log: Path, attempts: int = 3) -> dict:
    """All REPS runs of the harness on the split (in parallel); with REPS > 1 the metrics are those of the per-case
    mean score over the runs (user decision 2026-10-18: model sampling noise of qwen3.7-flash made single runs of the
    same harness differ by 0.11 AUC)."""
    if REPS == 1:
        return evaluate_rep(hid, split, log, 1, attempts)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(REPS) as ex:
        ms = list(ex.map(lambda r: evaluate_rep(hid, split, log, r, attempts), range(1, REPS + 1)))
    cases = scoring.load_cases(SPLITS_DIR / f"{split}.json")
    s_, y, _ = scoring.arrays(cases, agg_run(rep_runs(split, hid)))
    return {"auc": scoring.auc(s_, y), "n_run": min(m["n_run"] for m in ms), "credits_per_case": None, "reps": REPS,
            "auc_per_rep": [m["auc"] for m in ms]}


def rep_runs(split: str, hid: str) -> list[dict]:
    return [scoring.load_run(run_dir(split, hid, r)) for r in range(1, REPS + 1)]


def agg_run(runs: list[dict]) -> dict:
    """Per case the mean score over the runs that gave one."""
    out = {}
    for cid in set().union(*runs):
        ps = [r[cid]["p"] for r in runs if cid in r and r[cid].get("p") is not None]
        out[cid] = {"p": float(sum(ps) / len(ps)) if ps else None}
    return out


def paired_reps(cases: list[dict], runs_a: list[dict], runs_b: list[dict], n: int = scoring.N_BOOT, seed: int = 0) -> dict:
    """AUC(mean b) - AUC(mean a) with a two-level bootstrap: records with replacement, and within every case its runs
    with replacement (independently for a and b), so the interval includes the model's sampling noise."""
    import numpy as np

    def mat(runs):
        return np.array([[0.5 if r.get(c["candidate_id"], {}).get("p") is None else float(r[c["candidate_id"]]["p"])
                          for r in runs] for c in cases])
    a, b = mat(runs_a), mat(runs_b)
    y = np.array([c["reference"] == "ischemic" for c in cases])
    rec = np.array([c["record"] for c in cases])
    d = scoring.auc(b.mean(1), y) - scoring.auc(a.mean(1), y)
    rng = np.random.default_rng(seed + 1)
    ds = []
    for idx in scoring._record_draws(rec, n, seed):
        ka, kb = a.shape[1], b.shape[1]
        ma = a[idx[:, None], rng.integers(0, ka, (len(idx), ka))].mean(1)
        mb = b[idx[:, None], rng.integers(0, kb, (len(idx), kb))].mean(1)
        v = scoring.auc(mb, y[idx]) - scoring.auc(ma, y[idx])
        if v == v:
            ds.append(v)
    ds = np.array(ds)
    return {"auc_a": scoring.auc(a.mean(1), y), "auc_b": scoring.auc(b.mean(1), y), "delta": float(d),
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p_gt0": float((ds > 0).mean()),
            "n_boot": int(len(ds)), "reps": [a.shape[1], b.shape[1]]}


def evaluate_rep(hid: str, split: str, log: Path, rep: int, attempts: int = 3) -> dict:
    """Run (or resume) one run of the harness on the split until every case has a result; a run that stays incomplete
    raises IncompleteRun, so that no base, promotion or rejection is ever decided on missing cases (cases without a
    result would score 0.5). Added 2026-10-17 after a preempted vLLM pod failed 259 cases of a base run."""
    out = run_dir(split, hid, rep)
    n = len(scoring.load_cases(SPLITS_DIR / f"{split}.json"))
    for _ in range(attempts):
        if (out / "metrics.json").exists() and json.loads((out / "metrics.json").read_text())["n_run"] >= n:
            break
        with log.open("a") as f:
            subprocess.run([PY, str(RSI_DIR / "evaluate.py"), hid, "--split", split, "--rep", str(rep), "--backend", "pool",
                            "--concurrency", str(CONCURRENCY)], cwd=CODE_DIR, stdout=f, stderr=subprocess.STDOUT)
    m = json.loads((out / "metrics.json").read_text())
    if m["n_run"] < n:
        raise IncompleteRun(f"{hid} on {split}: {m['n_run']} of {n} cases have a result after {attempts} attempts")
    return m


def within_patient_auc(split: str, hid: str) -> float | None:
    """AUC over case pairs of the same patient (patients with both classes), a secondary metric from 2026-10-18: the
    pooled AUC also ranks patients against each other, which hid gains within patients on validation_v2."""
    import numpy as np
    cases = scoring.load_cases(SPLITS_DIR / f"{split}.json")
    s, y, rec = scoring.arrays(cases, agg_run(rep_runs(split, hid)))
    pat = np.array([r[:-1] for r in rec])
    num = den = 0.0
    for p in set(pat):
        m = pat == p
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if n1 and n0:
            num += scoring.auc(s[m], y[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else None


def paired(split: str, a: str, b: str) -> dict:
    cases = scoring.load_cases(SPLITS_DIR / f"{split}.json")
    if REPS > 1:
        return paired_reps(cases, rep_runs(split, a), rep_runs(split, b))
    return scoring.paired(cases, scoring.load_run(run_dir(split, a, 1)), scoring.load_run(run_dir(split, b, 1)))


# ------------------------------------------------------------------------------------------------------ checks
def library_names() -> list[str]:
    return sorted(p.stem for p in LIBRARY.glob("*.py"))


def sync_tools(hid: str, names: list[str]) -> list[str]:
    """Make tools/ of the new harness exactly the chosen library tools; returns problems."""
    bad = [n for n in names if n not in library_names()]
    if bad:
        return [f"library_tools not in the library: {bad} (available: {library_names()})"]
    tdir = HARNESS_DIR / hid / "tools"
    if EXTENDED:  # the improver may also have written tools of its own: copy the library picks next to them
        if names:
            tdir.mkdir(exist_ok=True)
        for n in names:
            for ext in (".py", ".json"):
                shutil.copy(LIBRARY / f"{n}{ext}", tdir / f"{n}{ext}")
        return []
    if tdir.exists():
        shutil.rmtree(tdir)
    if names:
        tdir.mkdir()
        for n in names:
            for ext in (".py", ".json"):
                shutil.copy(LIBRARY / f"{n}{ext}", tdir / f"{n}{ext}")
    return []


def tool_problems(hid: str, parent: str) -> list[str]:
    """Extended scope: tools named like a library tool must be the library copy; every new or changed tool must answer
    on three search-split candidates in the sandboxed worker."""
    tdir = HARNESS_DIR / hid / "tools"
    if not tdir.exists():
        return []
    probs, new, lib = [], [], set(library_names())
    for f in tdir.iterdir():
        if f.is_file() and f.suffix not in (".py", ".json"):
            probs.append(f"tools/{f.name}: only <name>.py and <name>.json files may be in tools/")
    for spec_f in sorted(tdir.glob("*.json")):
        name, py = spec_f.stem, tdir / f"{spec_f.stem}.py"
        if not py.exists():
            probs.append(f"tools/{name}.json has no tools/{name}.py")
            continue
        if name in lib:
            if any((tdir / f"{name}{e}").read_bytes() != (LIBRARY / f"{name}{e}").read_bytes() for e in (".py", ".json")):
                probs.append(f"tools/{name} has the name of a library tool but differs from it; give a changed tool a "
                             "new name")
            continue
        pdir = HARNESS_DIR / parent / "tools"
        if (pdir / f"{name}.py").exists() and (pdir / f"{name}.py").read_bytes() == py.read_bytes() and \
                (pdir / f"{name}.json").read_bytes() == spec_f.read_bytes():
            continue  # inherited unchanged, checked in its own round
        if re.search(r"\bs[23]\d{4}\b", py.read_text()):
            probs.append(f"tools/{name}.py names a record")
        req = set((json.loads(spec_f.read_text()).get("input_schema") or {}).get("required") or [])
        if not req <= {"record_id", "candidate_id"}:
            probs.append(f"tools/{name}: required arguments may only be record_id and candidate_id (others optional)")
            continue
        new.append(name)
    if not new or probs:
        return probs
    from rsi.ext_worker import ExtWorker
    from rsi.paths import LTSTDB_CACHE
    cases = scoring.load_cases(SPLITS_DIR / f"{SEARCH}.json")
    picks = [cases[0], cases[len(cases) // 2], cases[-1]]
    worker = ExtWorker(tdir.resolve(), LTSTDB_CACHE.resolve())
    try:
        for name in new:
            for c in picks:
                r = worker.call(name, {"record_id": c["record"], "candidate_id": c["candidate_id"]})
                if not r.get("ok"):
                    probs.append(f"tools/{name} failed on a search-split candidate: {str(r.get('error'))[:400]}")
                    break
                if len(json.dumps(r.get("result"))) > MAX_TOOL_RESULT_CHARS:
                    probs.append(f"tools/{name} returned more than {MAX_TOOL_RESULT_CHARS} characters")
                    break
    finally:
        worker.close()
    return probs


def _norm_quote(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("\u2010", "-"), ("\u2011", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"), ("\u2019", "'"),
                 ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def abstract(pmid: str) -> str:
    ABSTRACTS.mkdir(parents=True, exist_ok=True)
    f = ABSTRACTS / f"{pmid}.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text()
    for _ in range(3):
        try:
            txt = urllib.request.urlopen(EFETCH.format(pmid), timeout=30).read().decode("utf-8", "replace")
            if txt.strip():
                f.write_text(txt)
                return txt
        except OSError:
            pass
        time.sleep(3)
    return ""


def pmc_fulltext(pmcid: str) -> str:
    """Plain text of an open-access PubMed Central article (tags removed); empty when not available."""
    import html
    num = re.sub(r"(?i)^pmc", "", str(pmcid).strip())
    if not num.isdigit():
        return ""
    ABSTRACTS.mkdir(parents=True, exist_ok=True)
    f = ABSTRACTS / f"pmc_{num}.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text()
    for _ in range(3):
        try:
            xml = urllib.request.urlopen(EFETCH_PMC.format(num), timeout=60).read().decode("utf-8", "replace")
            if "<body" not in xml:  # not open access: only metadata and abstract come back
                return ""
            txt = html.unescape(re.sub(r"<[^>]+>", " ", xml))
            f.write_text(txt)
            return txt
        except OSError:
            time.sleep(3)
    return ""


def allowed_url(url: str) -> bool:
    from urllib.parse import urlparse
    u = urlparse(str(url))
    prefix = SOURCE_HOSTS.get(u.netloc)
    return u.scheme == "https" and prefix is not None and u.path.startswith(prefix)


def page_text(url: str) -> str:
    """Plain text of an allowed web page (tags removed), cached; empty when it cannot be fetched."""
    import hashlib
    import html
    ABSTRACTS.mkdir(parents=True, exist_ok=True)
    f = ABSTRACTS / f"url_{hashlib.sha256(url.encode()).hexdigest()[:16]}.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research card check)"})
    for _ in range(2):
        try:
            raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            txt = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)))
            f.write_text(txt)
            return txt
        except OSError:
            time.sleep(3)
    return ""


def local_cards(hid: str) -> dict:
    f = HARNESS_DIR / hid / "knowledge" / "cards.json"
    return json.loads(f.read_text()) if f.exists() else {"cards": []}


def card_problems(hid: str, parent: str) -> list[str]:
    """Extended scope: the harness's own cards need every field, a known topic and a quote found verbatim in the PubMed
    abstract of their PMID; passing cards get their url, access and verified fields here."""
    f = HARNESS_DIR / hid / "knowledge" / "cards.json"
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        return [f"knowledge/cards.json is not valid JSON: {e}"]
    shared = json.loads(CARDS.read_text())
    shared_ids = {c["id"] for c in shared["cards"]}
    topics = set(shared["topics"]) | set(d.get("topics", {}))
    inherited = {c["id"]: c for c in local_cards(parent).get("cards", [])}
    probs, seen = [], set()
    for c in d.get("cards", []):
        cid = c.get("id")
        missing = [k for k in CARD_FIELDS if not c.get(k)]
        if not (c.get("pmid") or c.get("source_url")):
            missing.append("pmid or source_url")
        if missing:
            probs.append(f"card {cid!r}: missing {missing}")
            continue
        if c.get("source_url") and not allowed_url(c["source_url"]):
            probs.append(f"card {cid}: source_url must be an https page on {sorted(SOURCE_HOSTS)} (Bookshelf: /books/; "
                         "European Heart Journal: /eurheartj/)")
            continue
        if cid in shared_ids or cid in seen:
            probs.append(f"card id {cid!r} is already used")
            continue
        seen.add(cid)
        if set(c["topics"]) - topics:
            probs.append(f"card {cid}: unknown topics {sorted(set(c['topics']) - topics)} (add new ones under \"topics\")")
        if inherited.get(cid) == c:
            continue
        q = _norm_quote(str(c["quote"]))
        if len(q) < 40:
            probs.append(f"card {cid}: the quote must be at least 40 characters of the abstract or full text")
        elif c.get("pmid") and q in _norm_quote(abstract(str(c["pmid"]))):
            c["_where"] = "abstract"
        elif c.get("pmcid") and q in _norm_quote(pmc_fulltext(str(c["pmcid"]))):
            c["_where"] = "full text (PubMed Central open access)"
        elif c.get("source_url") and q in _norm_quote(page_text(c["source_url"])):
            c["_where"] = "web page (" + ("NCBI Bookshelf" if "/books/" in c["source_url"] else "society guideline") + ")"
        else:
            probs.append(f"card {cid}: quote not found verbatim in "
                         + ", ".join(x for x in (f"the PubMed abstract of PMID {c['pmid']}" if c.get("pmid") else "",
                                                  f"the open-access full text of {c['pmcid']}" if c.get("pmcid") else "",
                                                  f"the page {c['source_url']}" if c.get("source_url") else "") if x)
                         + " (pages behind a paywall or a bot check cannot be verified)")
    if not probs:
        for c in d.get("cards", []):
            where = c.pop("_where", None) or c.get("access") or "abstract"
            url = EFETCH_PMC.format(re.sub(r"(?i)^pmc", "", str(c["pmcid"]))) if where.startswith("full text") \
                else c["source_url"] if where.startswith("web page") else EFETCH.format(c["pmid"])
            c |= {"url": url, "access": where, "verified": True}
        f.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    else:
        for c in d.get("cards", []):
            c.pop("_where", None)
    return probs


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def check(hid: str, parent: str, rd: Path, out: dict | None) -> list[str]:
    problems = []
    if out is None:
        return ["no structured output"]
    problems += sync_tools(hid, list(out.get("library_tools") or []))
    try:
        h = Harness.load(hid)
    except (HarnessError, KeyError, json.JSONDecodeError) as e:
        return problems + [f"the harness does not load: {e}"]
    p = Harness.load(parent)
    a, b = p.spec, h.spec
    if b.get("id") != hid or b.get("parent") != parent:
        problems.append(f"harness.json id/parent must be {hid!r}/{parent!r}")
    for key in set(a) | set(b):
        if key in ("summary", "system", "id", "parent") or key in ALLOWED_TOP_KEYS:
            continue
        if key == "sessions":
            changed = {k for k in set(a.get(key, {})) | set(b.get(key, {})) if a.get(key, {}).get(k) != b.get(key, {}).get(k)}
            if changed - ALLOWED_SESSION_KEYS:
                problems.append(f"sessions fields that may not change: {sorted(changed - ALLOWED_SESSION_KEYS)}")
        elif key == "tools":
            changed = {k for k in set(a.get(key, {})) | set(b.get(key, {})) if a.get(key, {}).get(k) != b.get(key, {}).get(k)}
            if changed - ALLOWED_TOOL_KEYS:
                problems.append(f"tools fields that may not change: {sorted(changed - ALLOWED_TOOL_KEYS)}")
        elif a.get(key) != b.get(key):
            problems.append(f"harness.json field {key!r} may not change")
    if EXTENDED and h.knowledge is not None and h.knowledge.get("precedents"):
        # 2026-10-19: no precedents in the search; the bank's patients outside search_v2 are mostly validation
        # patients, so their labels would reach the improver through the search runs
        problems.append('knowledge.precedents must be false in this search (literature cards only): set '
                        '"precedents": false')
    if SINGLE:
        if h.sessions["verify"] != "none" or h.sessions["first_impression"]:
            problems.append("this track searches the judgement session alone: sessions.verify must stay 'none' and "
                            "first_impression false")
        if h.knowledge is not None and h.knowledge.get("sessions") != ["investigate"]:
            problems.append('knowledge.sessions must be ["investigate"] in this track (there is no verification)')
    if (HARNESS_DIR / hid / "submit.json").read_text() != (HARNESS_DIR / parent / "submit.json").read_text():
        problems.append("submit.json may not change")
    extra = {f.relative_to(HARNESS_DIR / hid).as_posix() for f in (HARNESS_DIR / hid).rglob("*") if f.is_file()} - {
        "harness.json", "submit.json"}
    extra = {x for x in extra if not x.startswith("prompts/") and not x.startswith("tools/") and "__pycache__" not in x}
    if EXTENDED:
        extra -= {"knowledge/cards.json"}
        problems += tool_problems(hid, parent) + card_problems(hid, parent)
    if extra:
        problems.append(f"files that may not exist in the harness: {sorted(extra)}")
    text = "\n".join(f.read_text() for f in (HARNESS_DIR / hid / "prompts").glob("*.md")) + json.dumps(b) + \
        json.dumps(local_cards(hid))
    if re.search(r"\bs[23]\d{4}\b", text):
        problems.append("a record name appears in the harness")
    cards = {c["id"]: c for c in json.loads(CARDS.read_text())["cards"] + local_cards(hid).get("cards", [])}
    for st in out.get("medical_statements") or []:
        c = cards.get(st.get("card_id"))
        if c is None:
            problems.append(f"unknown card id {st.get('card_id')!r}")
        elif _norm(st.get("quote", "")) not in _norm(c.get("quote", "") + " " + c.get("statement", "")):
            problems.append(f"quote not found in card {c['id']}: {st.get('quote', '')[:80]!r}")
    problems += aspect_problems(hid, parent)
    if not (rd / "proposal.md").exists():
        problems.append("proposal.md is missing in the round directory")
    if b.get("summary") == a.get("summary"):
        problems.append("harness.json summary was not updated")
    return problems


REF_HEAD, USE_HEAD = "Reference knowledge", "How to use this knowledge"


def _context_parts(hid: str) -> tuple[str, str] | None:
    """(reference knowledge section, rest) of prompts/context.md, or None when a heading is missing."""
    f = HARNESS_DIR / hid / "prompts" / "context.md"
    lines = f.read_text().splitlines() if f.exists() else []
    i = next((k for k, x in enumerate(lines) if x.startswith(REF_HEAD)), None)
    j = next((k for k, x in enumerate(lines) if x.startswith(USE_HEAD)), None)
    if i is None or j is None or j < i:
        return None
    return "\n".join(lines[i:j]), "\n".join(lines[:i] + lines[j:])


def _files(hid: str, sub: str) -> dict[str, str]:
    d = HARNESS_DIR / hid / sub
    return {f.relative_to(d).as_posix(): f.read_text() for f in d.rglob("*") if f.is_file() and "__pycache__" not in f.parts} \
        if d.exists() else {}


def touched_aspects(hid: str, parent: str) -> tuple[set[str], list[str]]:
    """The aspects whose own resources differ between parent and child (see ASPECT_RULE)."""
    a, b = Harness.load(parent).spec, Harness.load(hid).spec
    touched, problems = set(), []
    pa, pb = _context_parts(parent), _context_parts(hid)
    if pb is None:
        problems.append(f'prompts/context.md must keep the headings "{REF_HEAD}" and "{USE_HEAD}"')
    elif pa is not None and pa[0] != pb[0]:
        touched.add("A")
    if a.get("knowledge") != b.get("knowledge") or _files(parent, "knowledge") != _files(hid, "knowledge"):
        touched.add("A")
    if a.get("sessions") != b.get("sessions") or a.get("tools") != b.get("tools") or _files(parent, "tools") != _files(hid, "tools"):
        touched.add("B")
    pp, pc = _files(parent, "prompts"), _files(hid, "prompts")
    if set(pp) != set(pc) or a.get("system") != b.get("system") or \
            any(pp.get(n) != pc.get(n) for n in ("judgement.md", "labels.md", "images.md")):
        touched.add("C")
    return touched, problems


def aspect_problems(hid: str, parent: str) -> list[str]:
    if ASPECTS == "ABC":
        return []
    touched, problems = touched_aspects(hid, parent)
    for x in sorted(touched - set(ASPECTS)):
        problems.append(f"aspect {x} ({ASPECT_NAMES[x]}) may not change in this search (open: {ASPECTS})")
    if not touched & set(ASPECTS) and not problems:
        problems.append(f"no resource of an open aspect ({ASPECTS}) changed")
    return problems


def diff_text(parent: str, hid: str) -> str:
    out = []
    pa, pb = HARNESS_DIR / parent, HARNESS_DIR / hid
    names = {f.relative_to(pa).as_posix() for f in pa.rglob("*") if f.is_file() and "__pycache__" not in f.parts}
    names |= {f.relative_to(pb).as_posix() for f in pb.rglob("*") if f.is_file() and "__pycache__" not in f.parts}
    for n in sorted(names):
        a = (pa / n).read_text().splitlines() if (pa / n).exists() else []
        b = (pb / n).read_text().splitlines() if (pb / n).exists() else []
        out += difflib.unified_diff(a, b, f"{parent}/{n}", f"{hid}/{n}", lineterm="")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------------ commands
def cmd_init(args) -> None:
    hid = args.harness
    Harness.load(hid)
    CLOOP.mkdir(exist_ok=True)
    log = CLOOP / f"init_{hid}.log"
    s, v = evaluate(hid, SEARCH, log), evaluate(hid, SELECT, log)
    append({"id": hid, "parent": None, "round": 0, "status": "base", "s_auc": s["auc"], "v_auc": v["auc"],
            "s_auc_within": within_patient_auc(SEARCH, hid), "v_auc_within": within_patient_auc(SELECT, hid),
            "select_split": SELECT, "reps": REPS, "s_auc_per_rep": s.get("auc_per_rep"),
            "v_auc_per_rep": v.get("auc_per_rep"),
            "s_credits_per_case": s["credits_per_case"], "summary": Harness.load(hid).spec.get("summary", ""),
            "time": time.strftime("%Y-%m-%d %H:%M")})
    print(f"base {hid}: search AUC {s['auc']:.3f}, validation AUC {v['auc']:.3f}")


def task_md(rnd: int, focus_id: str, focus: str, parent: dict, hid: str, table: list[dict] | None = None) -> str:
    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"
    if ARCHIVE:
        probs = {t["id"]: t["p"] for t in table or []}
        hist = [f"- round {e['round']}: {e['id']} (parent {e.get('parent')}): {e['status']}; search AUC {f3(e.get('s_auc'))}"
                + (f" (runs {', '.join(f'{x:.3f}' for x in e['s_auc_per_rep'])})" if e.get("s_auc_per_rep") else "")
                + (f", vs its parent {e['delta']:+.3f}" if e.get("delta") is not None else "")
                + (f", within-patient {f3(e.get('s_auc_within'))}" if e.get("s_auc_within") is not None else "")
                + (f"; chance to be drawn this round {probs[e['id']]:.2f}" if e["id"] in probs else "")
                + f"; {e.get('summary', '')}" for e in board()]
        parent_line = (f"Parent: **{parent['id']}** (search AUC {parent['s_auc']:.3f}; lineage "
                       f"{' -> '.join(lineage(parent['id']))}).\nThis search keeps an archive instead of accepting or "
                       f"rejecting changes: every harness that passes the code checks is evaluated on the search set and "
                       f"joins the archive, and each round's parent is drawn from the archive, more often for a higher "
                       f"search AUC and less often for a harness that already has many children. This round drew "
                       f"{parent['id']}. Your change is judged by its search AUC like every other archive member.")
    else:
        hist = [f"- round {e['round']}: {e['id']} (parent {e['parent']}, focus {e.get('focus')}): {e['status']}; "
                f"search AUC {e.get('s_auc')}, delta {e.get('delta')}; {e.get('summary', '')}" for e in board()]
        parent_line = f"Parent: **{parent['id']}** (search AUC {parent['s_auc']:.3f}, validation AUC {parent['v_auc']:.3f})."
    return f"""# Round {rnd}

Focus ({focus_id}): {focus}

{parent_line}
{_reps_note(parent)}
New harness: **{hid}**, already created as a copy of the parent: {HARNESS_DIR / hid}

Paths (read):
- parent harness: {HARNESS_DIR / parent['id']}
- failure summaries of the parent's search-split run: {FAILURES_DIR / (parent['id'] + '_r1')} (index.md, cases/; other runs in _r2, ...)
- full results of search-split runs (per case JSON with the evidence ledger and assessments, traces): {RUNS_DIR / SEARCH}
- tool library (tools you may add with library_tools): {LIBRARY} — available: {', '.join(library_names())}
- verified literature cards: {CARDS}
- tool code of the base tools (to check what they measure): {CODE_DIR / 'holter_agent' / 'agent' / 'toolkit.py'}
- earlier rounds: {ROUNDS}

{_scope_paths(hid)}

{'Archive (all rounds of this search; invalid rounds did not pass the checks)' if ARCHIVE else 'History'}:
{chr(10).join(hist) if hist else '(none)'}
"""


def _reps_note(parent: dict) -> str:
    if REPS == 1:
        return ""
    per = parent.get("s_auc_per_rep") or []
    cases = scoring.load_cases(SPLITS_DIR / f"{SEARCH}.json")
    runs = rep_runs(SEARCH, parent["id"])
    null = paired_reps(cases, runs, runs, n=1000, seed=7)  # the parent against itself, runs resampled independently
    return (f"Every harness runs {REPS} times per split and a case's score is the mean over the runs (parent's search "
            f"runs: AUC {', '.join(f'{x:.3f}' for x in per)}; their mean-score AUC is the one above). The parent's "
            f"runs are {parent['id']}_r1 to _r{REPS} in the search runs directory, with failure summaries for each. "
            f"Run-to-run noise: the parent against itself gives mean-score differences within "
            f"[{null['ci'][0]:+.3f}, {null['ci'][1]:+.3f}] AUC (95%).")


def _scope_paths(hid: str) -> str:
    h = HARNESS_DIR / hid
    if not EXTENDED:
        return f"You may edit only {h}/prompts/*.md and {h}/harness.json, and write proposal.md here."
    kind = "open weights" if TRACK_CFG.get("open_weights", True) else "a commercial hosted model"
    return f"""Task model: {TRACK_CFG.get('model')} ({kind}; track {TRACK}).
Scope: extended. You may edit {h}/prompts/*.md (including the "Reference knowledge" section of context.md),
{h}/harness.json ({', '.join(sorted({'summary', 'system', 'tools.descriptions', 'tools.extra'} | {'sessions.' + k for k in ALLOWED_SESSION_KEYS} | ALLOWED_TOP_KEYS))}),
add or edit cards in {h}/knowledge/cards.json, and write new tools in {h}/tools/ (<name>.py with run(ctx, args) and
<name>.json); write proposal.md here.
- knowledge base code and the context a tool sees: {RSI_DIR / 'rsi' / 'knowledge.py'}, {RSI_DIR / 'rsi' / 'ext_context.py'}
- harness settings and their validation: {RSI_DIR / 'rsi' / 'harness.py'}
- abstracts already fetched for card checks: {ABSTRACTS}"""


def cmd_round(args) -> None:
    focuses = json.loads((CLOOP / "focuses.json").read_text())["focuses"]
    focus_id = args.focus or "custom"
    focus = args.focus_text or focuses[focus_id]
    rnd = 1 + max((e["round"] for e in board()), default=0)
    table = None
    if ARCHIVE:
        parent, table = select_parent(rnd)
    else:
        parent = best()
    hid = next_id()
    rd = ROUNDS / f"round_{rnd:02d}"
    rd.mkdir(parents=True, exist_ok=True)
    if table is not None:
        (rd / "selection.json").write_text(json.dumps({"round": rnd, "parent": parent["id"], "config": ARCHIVE_CFG,
                                                       "table": table}, indent=1))
    shutil.copytree(HARNESS_DIR / parent["id"], HARNESS_DIR / hid, ignore=shutil.ignore_patterns("__pycache__"))
    spec = json.loads((HARNESS_DIR / hid / "harness.json").read_text())
    spec["id"], spec["parent"] = hid, parent["id"]
    (HARNESS_DIR / hid / "harness.json").write_text(json.dumps(spec, indent=2) + "\n")
    (rd / "task.md").write_text(task_md(rnd, focus_id, focus, parent, hid, table))
    hd = backend.rule_path(HARNESS_DIR / hid)
    sealed = [f"Read({backend.rule_path(root / sp)}/**)" for root in {RUNS_DIR, RSI_DIR / "runs"} for sp in SEALED]
    allow = [f"Edit({hd}/prompts/**)", f"Write({hd}/prompts/**)", f"Edit({hd}/harness.json)",
             f"Edit({backend.rule_path(rd)}/**)", f"Write({backend.rule_path(rd)}/**)"]
    deny = [f"Read({backend.rule_path(LTSTDB_DIR)}/**)", f"Read({backend.rule_path(EDB_DIR)}/**)",
            f"Read({backend.rule_path(STAFF_DIR)}/**)", f"Read({backend.rule_path(MIMIC_DIR)}/**)",
            f"Read({backend.rule_path(SPLITS_DIR / 'mimic.json')})",
            f"Read({backend.rule_path(SPLITS_DIR / 'edb.json')})", f"Read({backend.rule_path(SPLITS_DIR / 'ltstdb_unused.json')})", f"Read({backend.rule_path(SPLITS_DIR / 'staffiii.json')})", f"Edit({hd}/submit.json)",
            f"Read({backend.rule_path(CLOOP)}/final_*)"] + sealed  # final test/EDB results (final_eval.py)
    tools = ["Read", "Grep", "Glob", "Edit", "Write"]
    if "A" not in ASPECTS:
        deny += [f"Edit({hd}/knowledge/**)", f"Write({hd}/knowledge/**)"]
    if "B" not in ASPECTS:
        deny += [f"Edit({hd}/tools/**)", f"Write({hd}/tools/**)"]
    if EXTENDED:
        tools += ["WebSearch", "WebFetch"]
        allow += [f"Edit({hd}/tools/**)", f"Write({hd}/tools/**)", f"Edit({hd}/knowledge/**)",
                  f"Write({hd}/knowledge/**)", "WebSearch", "WebFetch"]
    else:
        deny += [f"Edit({hd}/tools/**)", f"Write({hd}/tools/**)"]
    harness_dirs = [HARNESS_DIR]
    if ARCHIVE:
        harness_dirs, isolated = isolation({e["id"] for e in board()} | {hid})
        deny += isolated
    call = backend.CallSpec(
        prompt="Carry out the round described in task.md in your working directory.", cwd=rd,
        system_prompt=system_prompt(), tools=tools,
        add_dirs=harness_dirs + [FAILURES_DIR, RUNS_DIR / SEARCH, LIBRARY, CARDS.parent, CODE_DIR / "holter_agent",
                                 RSI_DIR / "rsi", ROUNDS],
        allow=allow, deny=deny, schema=PROPOSE_SCHEMA, transcript=rd / "improver.transcript.jsonl")
    cfg = backend.ClaudeCodeConfig(model=args.model, effort="high")
    log = rd / "round.log"
    res = backend.run(cfg, call)
    out = res.structured
    problems = check(hid, parent["id"], rd, out)
    if problems and res.session_id:  # one chance to fix
        call.prompt = "Code checks of your change failed:\n- " + "\n- ".join(problems) + "\nFix them and finish again."
        call.resume = res.session_id
        res = backend.run(cfg, call)
        out = res.structured or out
        problems = check(hid, parent["id"], rd, out)
    (rd / "improver.json").write_text(json.dumps({"structured": out, "problems": problems, "cost_usd": res.cost_usd,
                                                  "seconds": res.seconds}, indent=1, ensure_ascii=False))
    (rd / "diff.txt").write_text(diff_text(parent["id"], hid))
    entry = {"id": hid, "parent": parent["id"], "round": rnd, "focus": focus_id,
             "summary": Harness.load(hid).spec.get("summary", "") if not problems else "", "time": time.strftime("%Y-%m-%d %H:%M")}
    if table is not None:
        entry["parent_p"] = next(t["p"] for t in table if t["id"] == parent["id"])
    if problems:
        append(entry | {"status": "invalid", "problems": problems})
        print("invalid:", problems)
        return
    (rd / "meta.json").write_text(json.dumps({"entry": entry, "parent": parent}, indent=1, ensure_ascii=False))
    decide(entry, parent, log)


def decide(entry: dict, parent: dict, log: Path) -> None:
    """Evaluate a checked round harness and apply the decision rule (also used by `resume`)."""
    hid = entry["id"]
    s = evaluate(hid, SEARCH, log)
    d = paired(SEARCH, parent["id"], hid)
    entry |= {"s_auc": s["auc"], "s_credits_per_case": s["credits_per_case"], "delta": d["delta"], "delta_ci": d["ci"],
              "p_gt0": d["p_gt0"], "s_auc_within": within_patient_auc(SEARCH, hid), "select_split": SELECT,
              "reps": REPS, "s_auc_per_rep": s.get("auc_per_rep")}
    if ARCHIVE:  # no decision: the child joins the archive (the paired difference is kept for information)
        entry["status"] = "archived"
    elif d["delta"] >= PROMOTE_DELTA and d["p_gt0"] >= PROMOTE_P:
        v = evaluate(hid, SELECT, log)
        entry["v_auc"] = v["auc"]
        entry["v_auc_within"] = within_patient_auc(SELECT, hid)
        entry["v_auc_per_rep"] = v.get("auc_per_rep")
        entry["status"] = "accepted" if v["auc"] >= parent["v_auc"] - SELECT_TOLERANCE else "rejected_on_validation"
    else:
        entry["status"] = "rejected"
    append(entry)
    print(json.dumps(entry, indent=1, ensure_ascii=False))


def cmd_resume(args) -> None:
    """Finish a round whose evaluation was interrupted (e.g. IncompleteRun after a preemption)."""
    if any(e["id"] == args.harness for e in board()):
        sys.exit(f"{args.harness} is already on the board")
    metas = [m for m in ROUNDS.glob("round_*/meta.json") if json.loads(m.read_text())["entry"]["id"] == args.harness]
    if not metas:
        sys.exit(f"no checked round for {args.harness} (meta.json is written after the code checks pass)")
    meta = json.loads(metas[0].read_text())
    decide(meta["entry"], meta["parent"], metas[0].parent / "round.log")


def cmd_finalize(args) -> None:
    """Archive search: evaluate the top-k archive harnesses by search AUC on the selection split (once each, REPS runs)
    and pick the one with the highest selection AUC. Written to final_selection.json (unreadable for later improvers)."""
    if not ARCHIVE:
        sys.exit("finalize is for an archive search")
    top = sorted(archive(), key=lambda e: e["s_auc"], reverse=True)[:args.top]
    log = CLOOP / "final_selection.log"
    rows = []
    for e in top:
        v = evaluate(e["id"], SELECT, log)
        rows.append({"id": e["id"], "parent": e.get("parent"), "round": e["round"], "s_auc": e["s_auc"],
                     "s_auc_per_rep": e.get("s_auc_per_rep"), "s_auc_within": e.get("s_auc_within"),
                     "v_auc": v["auc"], "v_auc_per_rep": v.get("auc_per_rep"), "v_auc_within": within_patient_auc(SELECT, e["id"]),
                     "lineage": lineage(e["id"]), "summary": e.get("summary", "")})
    pick = max(rows, key=lambda r: r["v_auc"])
    out = {"select_split": SELECT, "top": args.top, "reps": REPS, "rows": rows, "pick": pick["id"],
           "archive_size": len(archive()), "time": time.strftime("%Y-%m-%d %H:%M")}
    (CLOOP / "final_selection.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"{r['id']:6s} search {r['s_auc']:.3f} validation {r['v_auc']:.3f} ({', '.join(f'{x:.3f}' for x in r['v_auc_per_rep'] or [])})"
              f"  {' -> '.join(r['lineage'])}")
    print(f"pick: {pick['id']}")


def cmd_status(args) -> None:
    for e in board():
        nan = float("nan")
        print(f"{e['round']:>2} {e['id']:6s} {str(e.get('parent')):6s} {e.get('focus', ''):6s} {e['status']:24s} "
              f"search {e.get('s_auc') or nan:.3f} delta {e.get('delta') or 0:+.3f} "
              f"valid {e.get('v_auc') or nan:.3f}  {e.get('summary', '')[:80]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("harness")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("round")
    p.add_argument("--focus", default=None)
    p.add_argument("--focus-text", default=None)
    p.add_argument("--model", default="opus")
    p.set_defaults(fn=cmd_round)
    p = sub.add_parser("resume")
    p.add_argument("harness")
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("finalize")
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(fn=cmd_finalize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
