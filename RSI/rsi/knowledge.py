"""The knowledge base shown in verification sessions (harness key "knowledge", from 2026-10-14, harness v1.0).

Built beforehand, not by the agent (RSI/kb/):
- cards.json: literature cards (statement, short verbatim quote, source, access level, direction, limits, topics), each
  checked against its source (RSI/kb/CARDS_VERIFICATION.md);
- precedents.json: labelled LTSTDB development candidates with the tool features of the fixed check set
  (RSI/kb/build_precedents.py).

Retrieval is deterministic. Cards: the topics whose findings are present in the case's dossier (heart-rate rise, QRS
change, ST elevation, rhythm, noise, ...; thresholds below are engineering defaults), at most `max_auto_cards`; the
reviewer can ask for other topics with the knowledge_cards tool. Precedents: the nearest labelled candidates of other
patients by a robust-z Euclidean distance over PRECEDENT_FEATURES (chosen by leave-one-patient-out nearest-neighbour
AUC on the development records outside validation_v2), `k_per_class` of each class, with the class mix of the
`n_overall` nearest and the bank's own leave-one-patient-out AUC so that the reviewer can weigh them.

Leakage: every patient of the split under evaluation is removed from the bank (and the case's own patient always);
a run without that exclusion is refused. LTSTDB patients are record names without the last digit (split_v07).
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from .paths import RSI_DIR, SPLITS_DIR

KB_DIR = RSI_DIR / "kb"
PRECEDENT_FEATURES = ("abs_peak_uv", "max_step_frac", "t_amp_change_abs", "qrs_corr", "duration_s", "st_slope_mv_s",
                      "j0_minus_j80_uv", "hr_rise_in_episode")
SHOWN_FEATURES = PRECEDENT_FEATURES + ("depression", "hr_at_peak", "hr_rise_5min_before", "hr_peak_lag_s",
                                       "st_hr_slope_abs", "qrs_pp_change", "sync_qrs_st_steps")
FEATURE_LEGEND = {
    "abs_peak_uv": "|peak ST deviation| from the record-wide reference, uV",
    "max_step_frac": "largest single 15-s step / total ST change at the peak lead (near 1 = step-like)",
    "t_amp_change_abs": "|T amplitude change| episode vs reference window, uV",
    "qrs_corr": "QRS correlation, episode vs reference window (peak lead)",
    "duration_s": "candidate duration, s",
    "st_slope_mv_s": "slope from J to the ST point at the peak, mV/s",
    "j0_minus_j80_uv": "ST at J minus ST at J+80, uV (negative = upsloping)",
    "hr_rise_in_episode": "max HR in the episode minus HR 30-10 min before, bpm",
    "depression": "1 depression, 0 elevation",
    "hr_at_peak": "HR at the ST extremum, bpm",
    "hr_rise_5min_before": "HR rise in the 5 min before onset, bpm",
    "hr_peak_lag_s": "time of HR maximum minus time of ST extremum, s (negative = HR peaked first)",
    "st_hr_slope_abs": "|ST/HR slope| in the episode, uV/bpm",
    "qrs_pp_change": "|QRS peak-to-peak ratio - 1|, episode vs reference",
    "sync_qrs_st_steps": "number of simultaneous QRS and ST steps within 10 min of the episode",
}
CLASS_TEXT = {
    "ischemic": "ischemic ST episode",
    "heart_rate_related": "heart-rate related ST episode",
    "axis_or_conduction": "non-ischemic: axis shift or conduction change marked at this time",
    "noise_or_unreadable": "non-ischemic: noise or unreadable signal",
    "other": "non-ischemic: no ST episode annotated here",
}
CARD_TOOL = {
    "name": "knowledge_cards",
    "description": "Literature cards of the knowledge base on the given topics (statement, direction, conditions, "
                   "limits, source, access level), at most 4 per call. Cards already shown in this session are not "
                   "repeated.",
    "input_schema": {"type": "object", "properties": {
        "topics": {"type": "array", "items": {"type": "string"}, "description": "topic keys, as listed in cards_per_topic"}},
        "required": ["topics"]},
}


def patient(record: str) -> str:
    """LTSTDB: the record name without its last digit (split_v07); other data: the record itself."""
    r = str(record)
    return r[:-1] if r.startswith(("s2", "s3")) and len(r) == 6 else r


def split_patients(split: str) -> set[str]:
    cases = json.loads((SPLITS_DIR / f"{split}.json").read_text())["cases"]
    return {patient(c["record"]) for c in cases}


class KnowledgeBase:
    def __init__(self, exclude_patients: set[str], kb_dir: Path = KB_DIR, cards: bool = True,
                 precedents: bool = True, extra: dict | None = None):
        """extra: cards added by a harness ({"topics": {...}, "cards": [...], "auto_priority": {...}}, the layout of
        cards.json); its cards follow the shared ones, its topics are added, its priorities replace those of the
        same topic."""
        if precedents and not exclude_patients:
            raise ValueError("precedents need the patients of the evaluated split to exclude")
        self.exclude = set(exclude_patients)
        self.card_list = json.loads((kb_dir / "cards.json").read_text())["cards"] if cards else []
        meta = json.loads((kb_dir / "cards.json").read_text()) if cards else {}
        self.topics_text = meta.get("topics", {})
        self.priority = {k: v for k, v in meta.get("auto_priority", {}).items() if not k.startswith("_")}
        if cards and extra:
            self.card_list = self.card_list + list(extra.get("cards", []))
            self.topics_text = {**self.topics_text, **extra.get("topics", {})}
            self.priority |= {k: v for k, v in extra.get("auto_priority", {}).items() if not k.startswith("_")}
        self.bank = []
        if precedents:
            self.bank = [c for c in json.loads((kb_dir / "precedents.json").read_text())["cases"]
                         if patient(c["record"]) not in self.exclude]

    # ------------------------------------------------------------------ precedents
    def _matrix(self, feats: list[dict]) -> np.ndarray:
        return np.array([[np.nan if f.get(n) is None else float(f[n]) for n in PRECEDENT_FEATURES] for f in feats])

    @cached_property
    def _scale(self) -> tuple[np.ndarray, np.ndarray]:
        x = self._matrix([c["features"] for c in self.bank])
        med = np.nanmedian(x, 0)
        iqr = np.nanpercentile(x, 75, 0) - np.nanpercentile(x, 25, 0)
        sd = np.nanstd(x, 0)
        iqr = np.where(iqr > 0, iqr, np.where(sd > 0, sd, 1.0))
        return med, iqr

    def _z(self, x: np.ndarray) -> np.ndarray:
        med, iqr = self._scale
        return np.clip((np.where(np.isfinite(x), x, med) - med) / iqr, -5, 5)

    @cached_property
    def _bank_z(self) -> np.ndarray:
        return self._z(self._matrix([c["features"] for c in self.bank]))

    @cached_property
    def bank_auc(self) -> float:
        """Leave-one-patient-out AUC of the ischemic share among the 10 nearest precedents, within the bank."""
        from .scoring import auc

        z, y = self._bank_z, np.array([c["reference"] == "ischemic" for c in self.bank])
        pats = np.array([patient(c["record"]) for c in self.bank])
        s = np.zeros(len(self.bank))
        for i in range(len(self.bank)):
            other = pats != pats[i]
            d = np.sqrt(((z[other] - z[i]) ** 2).sum(1))
            s[i] = y[other][np.argsort(d)[:10]].mean()
        return float(auc(s, y))

    def precedents(self, feats: dict, record: str, k_per_class: int = 4, n_overall: int = 10) -> dict:
        own = patient(record)
        if own not in self.exclude:
            raise ValueError(f"patient {own} of the case is not excluded from the bank")
        z = self._z(self._matrix([feats]))[0]
        d = np.sqrt(((self._bank_z - z) ** 2).sum(1))
        order = np.argsort(d)
        isch = [i for i in order if self.bank[i]["reference"] == "ischemic"][:k_per_class]
        non = [i for i in order if self.bank[i]["reference"] != "ischemic"][:k_per_class]
        rank = {int(i): r + 1 for r, i in enumerate(order)}
        near = [self.bank[i]["reference"] for i in order[:n_overall]]

        def row(i: int) -> dict:
            c = self.bank[i]
            return {"precedent": f"P{rank[int(i)]}", "class": CLASS_TEXT[c["reference"]],
                    "distance": round(float(d[i]), 2), "values": [_round(c["features"].get(n)) for n in SHOWN_FEATURES]}

        return {
            "values_are": list(SHOWN_FEATURES),
            "this_case": [_round(feats.get(n)) for n in SHOWN_FEATURES],
            "nearest_ischemic": [row(i) for i in isch],
            "nearest_non_ischemic": [row(i) for i in sorted(non, key=lambda i: d[i])],
            f"classes_of_the_{n_overall}_nearest": {k: near.count(k) for k in dict.fromkeys(near)},
            "bank": {"candidates": len(self.bank), "patients": len({patient(c["record"]) for c in self.bank}),
                     "ischemic_fraction": round(float(np.mean([c["reference"] == "ischemic" for c in self.bank])), 2),
                     "leave_one_patient_out_auc_of_10_nearest": round(self.bank_auc, 2)},
            "similarity_features": list(PRECEDENT_FEATURES),
            "precedent_names": "P<k> = k-th most similar candidate of the bank",
            "legend": FEATURE_LEGEND,
        }

    # ------------------------------------------------------------------ cards
    def topics_for(self, feats: dict, dossier: dict, rules: list | None = None,
                   always: list | None = None) -> list[str]:
        """Topics whose findings are present in the case, in rule order, then the always-shown topics. The rules are
        data (harness knowledge.topic_rules, default DEFAULT_TOPIC_RULES, the hand-set rules of 2026-10-14) so that
        the harness search can change when knowledge is retrieved."""
        x = topic_inputs(feats, dossier)
        out = []
        for rule in (DEFAULT_TOPIC_RULES if rules is None else rules):
            if rule["topic"] not in out and any(_holds(x, cond) for cond in rule["any"]):
                out.append(rule["topic"])
        return out + [t for t in (DEFAULT_ALWAYS_TOPICS if always is None else always) if t not in out]

    PER_TOPIC = {"heart_rate": 2}  # cards per topic in automatic retrieval (others: 1)
    LOOKUP_LIMIT = 4  # cards per knowledge_cards call

    def cards(self, topics: list[str], skip: set[str] = frozenset(), limit: int | None = None) -> list[dict]:
        """Cards on the topics, in card order; with a limit (automatic retrieval) at most PER_TOPIC per topic."""
        out = []
        for topic in topics:
            n = 0
            first = self.priority.get(topic, [])
            for c in sorted(self.card_list, key=lambda c: first.index(c["id"]) if c["id"] in first else len(first)):
                if limit and n >= self.PER_TOPIC.get(topic, 1):
                    break
                if topic in c["topics"] and c["id"] not in skip and c["id"] not in {x["id"] for x in out}:
                    out.append(c)
                    n += 1
        out = out[:limit] if limit else out
        # what the model sees; the verbatim quote and the URL stay in cards.json for checking by people
        return [{k: c[k] for k in ("id", "title", "statement", "direction", "applies_when", "limits", "source",
                                   "access") if k in c} for c in out]

    def index(self) -> dict[str, int]:
        """Cards per topic."""
        return {t: sum(t in c["topics"] for c in self.card_list) for t in self.topics_text}


def _round(v):
    if v is None:
        return None
    v = float(v)
    if not np.isfinite(v):
        return None
    return round(v, 3) if abs(v) < 10 else round(v, 1)


# ------------------------------------------------------------------ retrieval rules (data, 2026-10-18)
# Each rule: {"topic": t, "any": [[input, op, value, default], ...]}; the topic is retrieved when any condition holds.
# default: the value used when the input is missing. Inputs are the case features (case_features, PRECEDENT_FEATURES)
# plus those of topic_inputs(). Thresholds are the engineering defaults of the hand-set rules.
TOPIC_OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
             "==": lambda a, b: a == b}
DEFAULT_TOPIC_RULES = [
    {"topic": "heart_rate", "any": [["hr_rise_in_episode", ">=", 10, 0], ["hr_rise_5min_before", ">=", 10, 0],
                                    ["hr_at_peak", ">=", 100, 0]]},
    {"topic": "measurement", "any": [["hr_at_peak", ">=", 100, 0], ["j0_minus_j80_uv", "<=", -100, 0]]},
    {"topic": "position_axis", "any": [["qrs_pp_change", ">=", 0.15, 0], ["sync_qrs_st_steps", ">=", 1, 0],
                                       ["qrs_corr", "<", 0.95, 1], ["max_step_frac", ">=", 0.5, 0]]},
    {"topic": "st_elevation", "any": [["depression", "==", 0, None]]},
    {"topic": "rhythm", "any": [["non_normal_beat_frac", ">=", 0.1, 0], ["rmssd_over_mean_rr", ">=", 0.1, 0]]},
    {"topic": "conduction", "any": [["qrs_ge_100ms", "==", 1, 0]]},
    {"topic": "eligibility", "any": [["qrs_ge_100ms", "==", 1, 0], ["baseline_st_abs_ge_100uv", "==", 1, 0]]},
    {"topic": "artifact", "any": [["noise_uv", ">=", 50, 0], ["fiducials_ok_frac", "<", 0.9, 1]]},
    {"topic": "t_wave_qrs", "any": [["t_amp_change_abs", ">=", 150, 0]]},
]
DEFAULT_ALWAYS_TOPICS = ["st_shape", "reasoning_errors"]
DOSSIER_INPUTS = ("non_normal_beat_frac", "rmssd_over_mean_rr", "qrs_ge_100ms", "baseline_st_abs_ge_100uv")
# inputs a rule may test: the case features (holter_agent.agent.estimate.features) and the dossier inputs above
TOPIC_INPUTS = frozenset({"depression", "abs_peak_uv", "duration_s", "hr_at_peak", "hr_peak_lag_s", "hr_rise_5min_before",
                          "hr_rise_in_episode", "j0_minus_j80_uv", "max_step_frac", "noise_uv", "offset_s",
                          "onset_to_90pct_s", "plateau_frac", "qrs_changed_frac", "qrs_corr", "qrs_pp_change",
                          "st_hr_slope_abs", "st_hr_spearman_abs", "st_slope_mv_s", "t_amp_change_abs", "t_amp_uv",
                          "acc_aha_met", "fiducials_ok_frac", "hr_median", "qrs_steps", "sync_qrs_st_steps"}
                         | set(DOSSIER_INPUTS))


def topic_inputs(feats: dict, dossier: dict) -> dict:
    """The case features plus inputs read from the dossier (rhythm_context, eligibility_check)."""
    x = {k: v for k, v in feats.items() if v is not None and (not isinstance(v, float) or np.isfinite(v))}
    rc = dossier.get("rhythm_context") or {}
    beats = rc.get("beat_types") or {}
    n = sum(beats.values())
    if n:
        x["non_normal_beat_frac"] = 1 - beats.get("N", 0) / n
    irr = (rc.get("rr_irregularity") or {}).get("rmssd_over_mean_rr")
    if irr is not None:
        x["rmssd_over_mean_rr"] = irr
    flags = (dossier.get("eligibility_check") or {}).get("flags") or {}
    x["qrs_ge_100ms"] = int(bool(flags.get("qrs_ge_100ms")))
    x["baseline_st_abs_ge_100uv"] = int(bool(flags.get("leads_baseline_st_abs_ge_100uv")))
    return x


def _holds(x: dict, cond: list) -> bool:
    name, op, value = cond[0], cond[1], cond[2]
    default = cond[3] if len(cond) > 3 else None
    v = x.get(name, default)
    return v is not None and TOPIC_OPS[op](v, value)


def topic_rule_problems(rules, always, inputs: set[str]) -> list[str]:
    """Problems of harness-set retrieval rules (used by rsi.harness)."""
    probs = []
    for r in rules or []:
        if not isinstance(r, dict) or set(r) != {"topic", "any"} or not isinstance(r["any"], list) or not r["any"]:
            probs.append(f"rule {r!r:.80}: needs exactly 'topic' and a non-empty 'any' list")
            continue
        for c in r["any"]:
            if not isinstance(c, list) or len(c) not in (3, 4) or c[1] not in TOPIC_OPS or c[0] not in inputs or \
                    not isinstance(c[2], (int, float)):
                probs.append(f"rule for {r['topic']}: bad condition {c!r:.80} (input from {sorted(inputs)}, "
                             f"op from {sorted(TOPIC_OPS)}, numeric value)")
    if always is not None and not (isinstance(always, list) and all(isinstance(t, str) for t in always)):
        probs.append("always_topics must be a list of topic names")
    return probs
