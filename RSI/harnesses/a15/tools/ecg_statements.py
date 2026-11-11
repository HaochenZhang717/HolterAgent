"""ECG statements of a foundation model for this candidate: ECGFounder (Li et al., NEJM AI 2025, arXiv 2410.04133; MIT;
"trained on over 10 million ECGs with 150 label categories from the Harvard-Emory ECG Database", with annotations from
cardiology experts, and extended by its authors to arbitrary single-lead ECGs), run beforehand on 10-s windows of this
record (sky/fm_tools_run.py; read here with ctx.precomputed("fm", record)).

Windows: the dossier's reference window, and inside the candidate its onset, peak, end and up to four evenly spaced
windows. Per lead the candidate involves, the single-lead model reads each lead alone; when lead names map to 12-lead
slots (ML2 -> II, MLIII -> III, MV2 -> V2, V-leads by name) the 12-lead model reads them together with the other slots
empty. Reported per statement: the probability in the reference window, the median over the candidate's windows,
and that change; plus the five statements whose median changed most
from the reference in the candidate's peak lead.
Model outputs are uncalibrated probabilities of a model trained on clinical 10-s ECGs, not on ambulatory Holter leads,
so changes against the patient's own reference window say more than absolute values. Statements worded as comparisons ("ST now depressed in") were learned from comparisons with earlier
ECGs of the same patient and are only a proxy here. A measurement packet, not a diagnosis.
"""
import numpy as np

GROUPS = {
    "repolarization": ["NONSPECIFIC ST ABNORMALITY", "NONSPECIFIC T WAVE ABNORMALITY",
                       "NONSPECIFIC ST AND T WAVE ABNORMALITY", "WITH REPOLARIZATION ABNORMALITY", "EARLY REPOLARIZATION",
                       "ACUTE MI / STEMI"],
    "injury_pattern_max": ["LATERAL INJURY PATTERN", "INFERIOR INJURY PATTERN", "ANTERIOR INJURY PATTERN",
                           "INFEROLATERAL INJURY PATTERN", "ANTEROLATERAL INJURY PATTERN"],
    "serial_statement_proxies": ["ST NOW DEPRESSED IN", "ST MORE DEPRESSED IN", "ST ELEVATION NOW PRESENT IN",
                                 "ST MORE ELEVATED IN", "T WAVE INVERSION NOW EVIDENT IN"],
    "conduction": ["RIGHT BUNDLE BRANCH BLOCK", "LEFT BUNDLE BRANCH BLOCK", "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",
                   "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY", "WITH QRS WIDENING", "LEFT ANTERIOR FASCICULAR BLOCK"],
    "axis": ["LEFT AXIS DEVIATION", "RIGHT AXIS DEVIATION", "LEFTWARD AXIS", "RIGHTWARD AXIS"],
    "rhythm": ["SINUS RHYTHM", "SINUS TACHYCARDIA", "SINUS BRADYCARDIA", "ATRIAL FIBRILLATION", "ATRIAL FLUTTER",
               "PREMATURE VENTRICULAR COMPLEXES", "PREMATURE ATRIAL COMPLEXES"],
    "other": ["LOW VOLTAGE QRS", "LEFT VENTRICULAR HYPERTROPHY", "ABNORMAL ECG"],
}
INSIDE = ("onset", "peak", "end", "mid1", "mid2", "mid3", "mid4")


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def _summary(labels, by_window, names, combine_max=False):
    """[reference, median inside the candidate, median minus reference] per statement (or for the group maximum)."""
    idx = [labels.index(n) for n in names if n in labels]

    def vals(w):
        v = by_window.get(w)
        if v is None:
            return None
        x = [v[i] for i in idx]
        return [max(x)] if combine_max else x
    ref = vals("reference")
    inside = [vals(w) for w in INSIDE if by_window.get(w) is not None]
    med = np.median(np.array(inside), axis=0) if inside else None
    keys = ["max"] if combine_max else [labels[i] for i in idx]
    out = {}
    for k, key in enumerate(keys):
        r = ref[k] if ref is not None else None
        m = med[k] if med is not None else None
        out[key] = [_r(r), _r(m), _r(m - r) if r is not None and m is not None else None]
    return out["max"] if combine_max else out


def _block(labels, by_window):
    return {g: _summary(labels, by_window, names, combine_max=(g == "injury_pattern_max")) for g, names in GROUPS.items()}


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    c = ctx.candidate(rid, cid)
    data = ctx.precomputed("fm", rid)
    labels = data["labels"]
    wins = [w for w in data["windows"] if w["candidate_id"] == c.id]
    if not wins:
        return {"candidate_id": c.id, "error": "no model results for this candidate"}
    involved = sorted({e.lead for e in c.lead_episodes}, key=lambda x: x != c.peak.lead)  # peak lead first
    leads = [lead for lead in involved if lead in wins[0]["founder_1lead"]]
    per_lead = {}
    for lead in leads:
        by_window = {w["window"]: w["founder_1lead"].get(lead) for w in wins}
        per_lead[lead] = _block(labels, by_window)
    twelve = None
    if any(w.get("founder_12lead") for w in wins):
        by_window = {w["window"]: w.get("founder_12lead") for w in wins}
        twelve = {"slots": wins[0].get("slots"), **_block(labels, by_window)}
    peak_lead = c.peak.lead
    changes = []
    if peak_lead in leads:
        by_window = {w["window"]: w["founder_1lead"].get(peak_lead) for w in wins}
        ref = by_window.get("reference")
        inside = [by_window[w] for w in INSIDE if by_window.get(w) is not None]
        if ref is not None and inside:
            med = np.median(np.array(inside), axis=0)
            delta = med - np.array(ref)
            for i in np.argsort(-np.abs(delta))[:5]:
                changes.append({"statement": labels[i], "reference": _r(ref[i]), "median_in_candidate": _r(med[i]),
                                "change": _r(delta[i])})
    return {
        "candidate_id": c.id,
        "windows": {w["window"]: w["t_s"] for w in wins},
        "single_lead_model_per_lead": per_lead,
        "twelve_lead_model": twelve,
        "largest_changes_from_reference_in_peak_lead": {"lead": peak_lead, "statements": changes},
        "format": "each statement: [probability in the reference window, median inside the candidate, change]",
        "notes": ["ECGFounder statements (uncalibrated probabilities of a model trained on clinical 10-s ECGs, not on "
                  "Holter leads); compare with the patient's own reference window rather than reading absolute values",
                  "single-lead model: each lead read alone",
                  "twelve_lead_model: only the slots named in 'slots' hold signal, the others are empty",
                  "serial_statement_proxies were learned from comparisons with earlier ECGs; a proxy here",
                  "injury_pattern_max: the largest of the five injury-pattern statements",
                  "a measurement packet, not a diagnosis"],
    }
