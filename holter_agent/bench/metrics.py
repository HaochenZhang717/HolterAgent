"""Benchmark metrics.

Episode matching follows the WFDB `epicmp` reference implementation of the
EC38/EC57 ST statistics (physionet.org/physiotools/wag/epicmp-1.htm): "To qualify a
test episode as a true positive for purposes of determining ST episode sensitivity,
it must overlap at least 50% of the reference episode, or the overlap must include
the reference-marked extremum. To qualify a test episode as a true positive for
purposes of determining ST episode positive predictivity, the reference episode
must overlap at least 50% of the test episode, or the overlap must include the
test-marked extremum". Comparison starts 5 minutes after the start of the record
(epicmp default). Gross statistics pool counts over records; average statistics
average per-record ratios, skipping records where a ratio is undefined.

Groups of metrics:
- detection of a target class: episode and duration Se/+P, gross and average,
  false alarms per 24 h, record-level bootstrap (Jager 1998; EVAL_ST, Jager,
  Smrdel & Mark, Computers in Cardiology 2004;31:585-588);
- EVAL_ST ischemic vs heart-rate related matrices and IE Se / IE +P;
- PhysioNet/CinC Challenge 2003 score (+1 match, -1 mismatch, 0 unclassified,
  at most 20 randomly chosen events per record);
- ST deviation measurement error (mean, SD, correlation, e95, share > 100 uV);
- classification of reviewed candidates, selective prediction with abstentions;
- correction between two stages (SCoRe transitions, Kumar et al. ICLR 2025; CL/CS/
  RSS, Yang et al. ACL 2025) and consistency across repeated runs.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..agent.state import ABSTAIN_LABELS
from .labels import EpisodeLabel, PredictedEpisode

NO_EVENT = "no_reference_event"
SKIP_S = 300.0  # epicmp default start of comparison
NON_ISCHEMIC_EVENT_LABELS = ("heart_rate_related", "axis_or_position_shift", "conduction_change", "other_nonischemic")


# --------------------------------------------------------------------------- matching primitives
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _qualifies(cover_start: float, cover_end: float, target_start: float, target_end: float,
               target_peak: float | None) -> bool:
    """epicmp rule: the cover overlaps >=50% of the target, or the overlap includes the target's extremum."""
    lo, hi = max(cover_start, target_start), min(cover_end, target_end)
    if hi <= lo:
        return False
    if hi - lo >= 0.5 * (target_end - target_start):
        return True
    return target_peak is not None and lo <= target_peak <= hi


def detects(test: PredictedEpisode, ref: EpisodeLabel) -> bool:
    """Test episode counts toward sensitivity for this reference episode."""
    return test.record == ref.record and _qualifies(test.start_s, test.end_s, ref.start_s, ref.end_s, ref.peak_s)


def confirms(ref: EpisodeLabel, test: PredictedEpisode) -> bool:
    """Reference episode counts toward positive predictivity for this test episode."""
    return test.record == ref.record and _qualifies(ref.start_s, ref.end_s, test.start_s, test.end_s, test.peak_s)


def _after_skip(items, skip_s: float):
    return [x for x in items if x.end_s > skip_s]


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _union_len(intervals, skip_s: float) -> float:
    return sum(max(0.0, e - max(s, skip_s)) for s, e in _union(intervals))


def _union_overlap(a, b, skip_s: float) -> float:
    return sum(_overlap(max(s0, skip_s), e0, s1, e1) for s0, e0 in _union(a) for s1, e1 in _union(b))


def _ratio(x: float, y: float) -> float | None:
    return x / y if y else None


def reference_label(p: PredictedEpisode, refs: list[EpisodeLabel]) -> str:
    """Class of the reference episode that confirms p with the largest overlap (NO_EVENT if none)."""
    best, best_ov = NO_EVENT, 0.0
    for r in refs:
        if confirms(r, p):
            ov = _overlap(p.start_s, p.end_s, r.start_s, r.end_s)
            if ov > best_ov:
                best, best_ov = r.label, ov
    return best


def _correct(pred: str, ref: str, binary: bool) -> bool:
    if binary:
        return (pred == "ischemic") == (ref == "ischemic")
    return pred == ref


# --------------------------------------------------------------------------- detection
@dataclass
class RecordCounts:
    tp_s: int = 0  # reference episodes detected
    fn: int = 0
    tp_p: int = 0  # test episodes confirmed
    fp: int = 0
    dur_ref: float = 0.0
    dur_test: float = 0.0
    dur_overlap: float = 0.0

    def rates(self) -> dict[str, float | None]:
        return {"se": _ratio(self.tp_s, self.tp_s + self.fn), "ppv": _ratio(self.tp_p, self.tp_p + self.fp),
                "dur_se": _ratio(self.dur_overlap, self.dur_ref), "dur_ppv": _ratio(self.dur_overlap, self.dur_test)}


def _counts(tests: list[PredictedEpisode], refs: list[EpisodeLabel], skip_s: float) -> RecordCounts:
    c = RecordCounts()
    for r in refs:
        hit = any(detects(t, r) for t in tests)
        c.tp_s += hit
        c.fn += not hit
    for t in tests:
        ok = any(confirms(r, t) for r in refs)
        c.tp_p += ok
        c.fp += not ok
    a = [(r.start_s, r.end_s) for r in refs]
    b = [(t.start_s, t.end_s) for t in tests]
    c.dur_ref, c.dur_test = _union_len(a, skip_s), _union_len(b, skip_s)
    c.dur_overlap = _union_overlap(a, b, skip_s)
    return c


def _gross(per_record: dict[str, RecordCounts]) -> dict[str, float | None]:
    tot = RecordCounts()
    for c in per_record.values():
        for k in tot.__dataclass_fields__:
            setattr(tot, k, getattr(tot, k) + getattr(c, k))
    return tot.rates()


def _average(per_record: dict[str, RecordCounts]) -> dict[str, float | None]:
    vals: dict[str, list[float]] = defaultdict(list)
    for c in per_record.values():
        for k, v in c.rates().items():
            if v is not None:
                vals[k].append(v)
    return {k: (float(np.mean(vals[k])) if vals[k] else None) for k in ("se", "ppv", "dur_se", "dur_ppv")}


@dataclass
class DetectionReport:
    target: str
    n_records: int
    n_reference: int
    n_predicted: int
    gross: dict
    average: dict
    candidate_coverage: float | None  # share of reference episodes detected by any reviewed candidate
    false_alarms_per_24h: float | None
    bootstrap: dict = field(default_factory=dict)  # record-level bootstrap of gross Se/+P


def detection_report(preds: list[PredictedEpisode], refs: list[EpisodeLabel], target: str = "ischemic",
                     skip_s: float = SKIP_S, record_hours: dict[str, float] | None = None,
                     n_boot: int = 0, seed: int = 0) -> DetectionReport:
    preds, refs = _after_skip(preds, skip_s), _after_skip(refs, skip_s)
    records = sorted({p.record for p in preds} | {r.record for r in refs} | set(record_hours or {}))
    per: dict[str, RecordCounts] = {}
    covered = n_ref = n_pred = 0
    for rec in records:
        tests = [p for p in preds if p.record == rec and p.final_label == target]
        rts = [r for r in refs if r.record == rec and r.label == target]
        per[rec] = _counts(tests, rts, skip_s)
        covered += sum(any(detects(p, r) for p in preds if p.record == rec) for r in rts)
        n_ref += len(rts)
        n_pred += len(tests)
    hours = sum((record_hours or {}).values())
    fa = sum(c.fp for c in per.values())
    boot: dict = {}
    if n_boot and records:
        rng = np.random.default_rng(seed)
        draws = {"se": [], "ppv": []}
        for _ in range(n_boot):
            sample = {f"{i}": per[records[j]] for i, j in enumerate(rng.integers(0, len(records), len(records)))}
            g = _gross(sample)
            for k in draws:
                if g[k] is not None:
                    draws[k].append(g[k])
        for k, v in draws.items():
            if v:
                boot[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "p5": float(np.percentile(v, 5))}
    return DetectionReport(target, len(records), n_ref, n_pred, _gross(per), _average(per),
                           _ratio(covered, n_ref), fa * 24.0 / hours if hours else None, boot)


@dataclass
class IschemiaMatrixReport:
    """EVAL_ST extended matching: reference rows vs algorithm columns (Jager et al. 2004)."""
    se_matrix: dict[str, dict[str, int]]  # reference class -> {ischemic, heart_rate_related, missed}
    ppv_matrix: dict[str, dict[str, int]]  # reference status -> algorithm class
    ie_se: float | None  # a / (a + b + c)
    ie_ppv: float | None  # g / (g + i + k)


def ischemia_matrix(preds: list[PredictedEpisode], refs: list[EpisodeLabel], skip_s: float = SKIP_S
                    ) -> IschemiaMatrixReport:
    classes = ("ischemic", "heart_rate_related")
    preds, refs = _after_skip(preds, skip_s), _after_skip(refs, skip_s)
    alg = {c: [p for p in preds if p.final_label == c] for c in classes}
    ref = {c: [r for r in refs if r.label == c] for c in classes}
    se = {c: {"ischemic": 0, "heart_rate_related": 0, "missed": 0} for c in classes}
    for c in classes:
        for r in ref[c]:
            if any(detects(p, r) for p in alg["ischemic"]):
                se[c]["ischemic"] += 1
            elif any(detects(p, r) for p in alg["heart_rate_related"]):
                se[c]["heart_rate_related"] += 1
            else:
                se[c]["missed"] += 1
    ppv = {s: {"ischemic": 0, "heart_rate_related": 0} for s in ("ischemic", "heart_rate_related", "none")}
    for c in classes:
        for p in alg[c]:
            if any(confirms(r, p) for r in ref["ischemic"]):
                ppv["ischemic"][c] += 1
            elif any(confirms(r, p) for r in ref["heart_rate_related"]):
                ppv["heart_rate_related"][c] += 1
            else:
                ppv["none"][c] += 1
    a, b, c_ = se["ischemic"]["ischemic"], se["ischemic"]["heart_rate_related"], se["ischemic"]["missed"]
    g, i, k = ppv["ischemic"]["ischemic"], ppv["heart_rate_related"]["ischemic"], ppv["none"]["ischemic"]
    return IschemiaMatrixReport(se, ppv, _ratio(a, a + b + c_), _ratio(g, g + i + k))


@dataclass
class ChallengeScore:
    """PhysioNet/CinC Challenge 2003 scoring of ischemic (I) vs non-ischemic (N) event classification."""
    n_events: int
    matches: int
    mismatches: int
    unclassified: int

    @property
    def score(self) -> int:
        return self.matches - self.mismatches

    @property
    def normalized(self) -> float | None:
        return _ratio(self.score, self.n_events)


def challenge_score(preds: list[PredictedEpisode], refs: list[EpisodeLabel], max_events_per_record: int = 20,
                    seed: int = 0) -> ChallengeScore:
    def ref_class(r: EpisodeLabel) -> str | None:
        if r.label == "ischemic":
            return "I"
        return "N" if r.label in NON_ISCHEMIC_EVENT_LABELS else None

    def pred_class(label: str | None) -> str:
        if label is None or label in ABSTAIN_LABELS:
            return "?"
        return "I" if label == "ischemic" else "N"

    rng = random.Random(seed)
    events = [r for r in refs if ref_class(r) is not None]
    by_rec: dict[str, list[EpisodeLabel]] = defaultdict(list)
    for r in events:
        by_rec[r.record].append(r)
    m = mm = u = n = 0
    for rec in sorted(by_rec):
        evs = by_rec[rec]
        if len(evs) > max_events_per_record:
            evs = rng.sample(evs, max_events_per_record)
        for r in evs:
            n += 1
            hits = [p for p in preds if detects(p, r)]
            if not hits:
                u += 1
                continue
            best = max(hits, key=lambda p: _overlap(p.start_s, p.end_s, r.start_s, r.end_s))
            pc = pred_class(best.final_label)
            if pc == "?":
                u += 1
            elif pc == ref_class(r):
                m += 1
            else:
                mm += 1
    return ChallengeScore(n, m, mm, u)


@dataclass
class STMeasurementReport:
    n: int
    mean_error_uv: float | None
    sd_error_uv: float | None
    correlation: float | None
    e95_uv: float | None  # |error| not exceeded by 95% of measurements
    p_gt_100uv: float | None  # share of |error| > 100 uV


def st_measurement_report(reference_uv: list[float], test_uv: list[float]) -> STMeasurementReport:
    ref, test = np.asarray(reference_uv, float), np.asarray(test_uv, float)
    ok = np.isfinite(ref) & np.isfinite(test)
    ref, test = ref[ok], test[ok]
    if len(ref) == 0:
        return STMeasurementReport(0, None, None, None, None, None)
    err = test - ref
    corr = float(np.corrcoef(ref, test)[0, 1]) if len(ref) > 1 and ref.std() > 0 and test.std() > 0 else None
    return STMeasurementReport(len(ref), float(err.mean()), float(err.std()), corr,
                               float(np.percentile(np.abs(err), 95)), float((np.abs(err) > 100.0).mean()))


# --------------------------------------------------------------------------- candidate-level
@dataclass
class ClassificationReport:
    n: int
    accuracy: float | None  # on candidates confirmed by a reference episode
    confusion: dict[str, dict[str, int]]  # reference -> predicted -> count
    binary_accuracy: float | None  # ischemic vs not, over all candidates (NO_EVENT counts as not ischemic)
    n_no_reference: int


def classification_report(preds: list[PredictedEpisode], refs: list[EpisodeLabel],
                          field_name: str = "final_label") -> ClassificationReport:
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = matched = bin_correct = bin_n = no_ref = 0
    for p in preds:
        pred = getattr(p, field_name)
        if pred is None:
            continue
        ref = reference_label(p, refs)
        confusion[ref][pred] += 1
        bin_n += 1
        bin_correct += _correct(pred, ref, binary=True)
        if ref == NO_EVENT:
            no_ref += 1
            continue
        matched += 1
        correct += pred == ref
    return ClassificationReport(matched, _ratio(correct, matched), {k: dict(v) for k, v in confusion.items()},
                                _ratio(bin_correct, bin_n), no_ref)


@dataclass
class SelectiveReport:
    n: int
    coverage: float | None  # share of candidates not abstained
    selective_binary_accuracy: float | None  # ischemic vs not, on non-abstained candidates
    abstained_binary_accuracy_if_forced: float | None  # accuracy of "not ischemic" on abstained ones


def selective_report(preds: list[PredictedEpisode], refs: list[EpisodeLabel]) -> SelectiveReport:
    kept = [p for p in preds if p.final_label is not None and p.final_label not in ABSTAIN_LABELS]
    abst = [p for p in preds if p.final_label in ABSTAIN_LABELS]
    n = len(kept) + len(abst)
    sel = sum(_correct(p.final_label, reference_label(p, refs), True) for p in kept)
    forced = sum(reference_label(p, refs) != "ischemic" for p in abst)
    return SelectiveReport(n, _ratio(len(kept), n), _ratio(sel, len(kept)), _ratio(forced, len(abst)))


@dataclass
class CorrectionReport:
    before: str
    after: str
    binary: bool
    n: int  # candidates with both labels (and a reference class)
    before_correct: int
    after_correct: int
    fixed: int  # wrong -> right
    broken: int  # right -> wrong
    changed_wrong_to_wrong: int
    n_changed: int

    @property
    def acc_before(self) -> float | None:
        return _ratio(self.before_correct, self.n)

    @property
    def acc_after(self) -> float | None:
        return _ratio(self.after_correct, self.n)

    @property
    def delta(self) -> float | None:  # SCoRe Delta(t1, t2)
        return None if not self.n else (self.after_correct - self.before_correct) / self.n

    @property
    def delta_i2c(self) -> float | None:  # SCoRe: fraction of all items going incorrect -> correct
        return _ratio(self.fixed, self.n)

    @property
    def delta_c2i(self) -> float | None:  # SCoRe: fraction of all items going correct -> incorrect
        return _ratio(self.broken, self.n)

    @property
    def cl(self) -> float | None:  # Yang et al. confidence level: P(right after | right before), empirical
        return _ratio(self.before_correct - self.broken, self.before_correct)

    @property
    def cs(self) -> float | None:  # Yang et al. critique score: P(right after | wrong before), empirical
        return _ratio(self.fixed, self.n - self.before_correct)

    @property
    def rss(self) -> float | None:  # Yang et al. Eq. 6: (Acc2 - Acc1^2) / (2 Acc1 - 2 Acc1^2)
        a1, a2 = self.acc_before, self.acc_after
        if a1 is None or a2 is None or a1 in (0.0, 1.0):
            return None
        return (a2 - a1 * a1) / (2 * a1 - 2 * a1 * a1)

    @property
    def mcnemar_p(self) -> float | None:  # exact two-sided test on the discordant pairs
        k = self.fixed + self.broken
        return None if k == 0 else float(stats.binomtest(self.fixed, k, 0.5).pvalue)

    def summary(self) -> dict:
        keys = ("acc_before", "acc_after", "delta", "delta_i2c", "delta_c2i", "cl", "cs", "rss", "mcnemar_p")
        return {"before": self.before, "after": self.after, "binary": self.binary, "n": self.n,
                "fixed": self.fixed, "broken": self.broken, "n_changed": self.n_changed,
                **{k: getattr(self, k) for k in keys}}


def correction_report(preds: list[PredictedEpisode], refs: list[EpisodeLabel], before: str = "initial_label",
                      after: str = "final_label", binary: bool = False) -> CorrectionReport:
    """Transitions between two stages. With binary=True correctness is ischemic vs not ischemic."""
    n = bc = ac = fixed = broken = w2w = changed = 0
    for p in preds:
        a, b = getattr(p, before), getattr(p, after)
        if a is None or b is None:
            continue
        ref = reference_label(p, refs)
        if ref == NO_EVENT and not binary:
            continue
        n += 1
        ok_a, ok_b = _correct(a, ref, binary), _correct(b, ref, binary)
        bc += ok_a
        ac += ok_b
        changed += a != b
        fixed += (not ok_a) and ok_b
        broken += ok_a and not ok_b
        w2w += (not ok_a) and (not ok_b) and a != b
    return CorrectionReport(before, after, binary, n, bc, ac, fixed, broken, w2w, changed)


@dataclass
class ConsistencyReport:
    n_runs: int
    n_candidates: int
    mean_agreement: float | None  # mean share of runs agreeing with the modal label
    fully_consistent: float | None  # share of candidates with the same label in every run
    binary_accuracy_consistent: float | None  # modal label accuracy on fully consistent candidates
    binary_accuracy_inconsistent: float | None


def consistency_report(runs: list[list[PredictedEpisode]], refs: list[EpisodeLabel] | None = None
                       ) -> ConsistencyReport:
    """Agreement of final labels across repeated runs of the same method.

    Behavioural consistency across runs was the strongest correctness signal for a
    clinical agent in Zhang et al. (Nature Medicine 2026).
    """
    by_key: dict[tuple[str, str], list[PredictedEpisode]] = defaultdict(list)
    for run in runs:
        for p in run:
            by_key[(p.record, p.candidate_id)].append(p)
    groups = [g for g in by_key.values() if len(g) == len(runs) and all(p.final_label for p in g)]
    agree, full = [], []
    acc_c = acc_i = n_c = n_i = 0
    for g in groups:
        label, count = Counter(p.final_label for p in g).most_common(1)[0]
        agree.append(count / len(g))
        full.append(count == len(g))
        if refs is not None:
            ok = _correct(label, reference_label(g[0], refs), True)
            if count == len(g):
                n_c += 1
                acc_c += ok
            else:
                n_i += 1
                acc_i += ok
    n = len(groups)
    return ConsistencyReport(len(runs), n, _ratio(sum(agree), n), _ratio(sum(full), n),
                             _ratio(acc_c, n_c) if refs is not None else None,
                             _ratio(acc_i, n_i) if refs is not None else None)


# --------------------------------------------------------------------------- process
CHECK_CATEGORIES = {
    "signal_quality": "quality", "measure_window": "measurement", "st_trend": "trend",
    "compare_morphology": "morphology", "morphology_steps": "morphology", "find_reference_window": "morphology",
    "episode_dynamics": "heart_rate", "rhythm_context": "rhythm", "eligibility_check": "eligibility",
    "check_criteria": "criteria",
}


@dataclass
class ProcessReport:
    n_candidates: int
    tool_calls_per_candidate: float | None
    category_coverage: dict[str, float]  # share of reviewed candidates whose sessions used each category
    revised_share: float | None  # share with more than one final assessment
    flagged_share: float | None  # share flagged for human review (label oscillation)
    stop_reasons: dict[str, int]


def process_report(states) -> ProcessReport:
    """What the decisions rest on: check categories cited by final assessments, revisions, stop reasons.

    category_coverage counts, for each category, the share of reviewed candidates whose
    final assessments cite at least one evidence item of that category.
    """
    n = calls = revised = flagged = 0
    cover: Counter = Counter()
    stops: Counter = Counter()
    for state in states:
        for cid, sessions in state.stop_reasons.items():
            n += 1
            for reason in sessions.values():
                stops[reason] += 1
            finals = [a for a in state.assessments if a.candidate_id == cid and a.stage == "final"]
            revised += len(finals) > 1
            flagged += cid in state.needs_review
            cited = {e for a in finals for e in a.evidence_ids}
            used = {CHECK_CATEGORIES[e.tool] for e in state.evidence if e.id in cited and e.tool in CHECK_CATEGORIES}
            for cat in used:
                cover[cat] += 1
        calls += state.tool_calls
    return ProcessReport(
        n_candidates=n,
        tool_calls_per_candidate=_ratio(calls, n),
        category_coverage={cat: cover[cat] / n for cat in sorted(set(CHECK_CATEGORIES.values()))} if n else {},
        revised_share=_ratio(revised, n),
        flagged_share=_ratio(flagged, n),
        stop_reasons=dict(stops),
    )
