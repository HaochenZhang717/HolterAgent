"""The whole record at a glance, for each lead the candidate involves: how often this lead's ST sits at the candidate's
level elsewhere in the record, where the candidate's extremum ranks, the other candidates of the same lead and sign,
and the hourly median ST deviation and heart rate. Deviations are from the record-wide reference, signed so that
positive points towards this candidate's extremum. A measurement packet, not a diagnosis.
"""
import numpy as np

MARGIN_S = 60.0


def _num(x, nd=0):
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    tab = ctx.table(rid)
    c = ctx.candidate(rid, cid)
    ws = tab.window_s
    t = tab["t_start_s"] + ws / 2
    hr = np.asarray(tab["hr_bpm"], float)
    dev_all = ctx.deviation(rid)
    cands = ctx.candidates(rid)
    busy = np.zeros(len(t), bool)
    for o in cands:
        busy |= (t >= o.start_s - MARGIN_S) & (t <= o.end_s + MARGIN_S)
    peaks = {}
    for e in c.lead_episodes:
        if e.lead not in peaks or abs(e.peak_uv) > abs(peaks[e.lead].peak_uv):
            peaks[e.lead] = e
    hours = np.floor(t / 3600).astype(int)
    leads = {}
    for lead, e in peaks.items():
        sign = 1.0 if e.peak_uv >= 0 else -1.0
        d = sign * dev_all[:, tab.lead_index(lead)]
        ext = abs(e.peak_uv)
        ok = np.isfinite(d)
        out = ok & ~busy
        same = []
        for o in cands:
            if o.id == c.id:
                continue
            for oe in o.lead_episodes:
                if oe.lead == lead and np.sign(oe.peak_uv) == sign:
                    same.append({"id": o.id, "start": _hms(o.start_s), "peak_uv": _num(oe.peak_uv),
                                 "duration_s": _num(o.end_s - o.start_s)})
                    break
        hourly = []
        for h in np.unique(hours):
            m = (hours == h) & ok
            if m.sum() >= 20:
                hourly.append({"hour": int(h), "median_dev_uv": _num(np.median(d[m])),
                               "median_hr": _num(np.nanmedian(hr[hours == h]))})
        leads[lead] = {
            "extremum_uv": _num(e.peak_uv),
            "frac_record_outside_candidates_at_half_extremum_or_more": _num((d[out] >= 0.5 * ext).mean(), 3)
            if out.any() else None,
            "frac_record_outside_candidates_at_extremum_or_more": _num((d[out] >= ext).mean(), 3) if out.any() else None,
            "extremum_percentile_in_record": _num(100.0 * (d[ok] < ext).mean(), 1) if ok.any() else None,
            "other_candidates_same_lead_and_sign": len(same),
            "largest_other_candidates_same_lead_and_sign": sorted(same, key=lambda x: -abs(x["peak_uv"] or 0))[:6],
            "hourly": hourly,
        }
    return {"candidate_id": c.id, "record_hours": _num(tab.duration_s / 3600, 1), "candidate_start": _hms(c.start_s),
            "leads": leads,
            "notes": ["deviations from the record-wide reference, signed towards this candidate's extremum",
                      "frac_record_outside_candidates_*: share of usable windows outside every candidate (+-60 s) "
                      "whose deviation reaches half of / all of this candidate's extremum",
                      "the whole-record ST trend image is attached after the other dossier images",
                      "a measurement packet, not a diagnosis"]}
