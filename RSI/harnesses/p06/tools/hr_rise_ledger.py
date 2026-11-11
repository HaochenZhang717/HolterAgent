"""The patient's own heart-rate rises compared with this candidate, and the timing of heart rate and ST around it.

At the candidate's peak lead, with the ST deviation signed so that positive points towards the candidate's extremum:
- this candidate: heart-rate baseline (median 30-10 min before onset), the rise before the extremum (size, minutes at
  >= baseline + 10 bpm), the heart-rate load of the previous hour, the time from the heart-rate maximum to the ST
  extremum, and the seconds after the extremum during which heart rate was back within 5 bpm of baseline while ST was
  still at half the extremum or more;
- comparable rises: every other heart-rate rise of >= 10 bpm within 10 min in the record (outside this candidate
  +-10 min) that starts within 10 bpm of this baseline and peaks within 10 bpm of this candidate's maximum heart rate,
  with the ST change it produced in the same lead.
Thresholds are engineering defaults. A measurement packet, not a diagnosis.
"""
import numpy as np

MIN_RISE_BPM = 10.0
MAX_RISE_S = 600.0
MATCH_BPM = 10.0
LIST_MAX = 8


def _num(x, nd=0):
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _rises(t, hr):
    ok = np.isfinite(hr)
    out, i, n = [], 0, len(hr)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j_end = np.searchsorted(t, t[i] + MAX_RISE_S, side="right")
        seg = np.flatnonzero(ok[i:j_end]) + i
        if seg.size and hr[seg].max() - hr[i] >= MIN_RISE_BPM and hr[i] <= np.nanmin(hr[max(0, i - 4):i + 1]):
            j = seg[np.argmax(hr[seg])]
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    tab = ctx.table(rid)
    c = ctx.candidate(rid, cid)
    ws = tab.window_s
    t = tab["t_start_s"] + ws / 2
    hr = np.asarray(tab["hr_bpm"], float)
    lead = c.peak.lead
    sign = 1.0 if c.peak.peak_uv >= 0 else -1.0
    d = sign * ctx.deviation(rid)[:, tab.lead_index(lead)]
    on, end, tx, ext = c.start_s, c.end_s, c.peak.peak_s, abs(c.peak.peak_uv)
    sel = lambda a, b: (t >= a) & (t <= b)  # noqa: E731
    pre = hr[sel(on - 1800, on - 600)]
    pre = pre[np.isfinite(pre)]
    base = float(np.median(pre)) if pre.size else np.nan
    w = sel(on - 600, tx) & np.isfinite(hr)
    hr_max = float(hr[w].max()) if w.any() else np.nan
    w2 = sel(on - 600, end) & np.isfinite(hr)
    lag = float(t[w2][np.argmax(hr[w2])] - tx) if w2.any() else np.nan
    cand = {"peak_lead": lead, "extremum_uv": _num(sign * ext), "extremum_t": _hms(tx),
            "hr_baseline_bpm": _num(base), "hr_max_before_extremum_bpm": _num(hr_max),
            "hr_rise_bpm": _num(hr_max - base), "hr_max_minus_st_extremum_s": _num(lag)}
    if np.isfinite(base):
        w = sel(on - 1800, tx) & np.isfinite(hr)
        cand["minutes_at_baseline_plus_10_bpm"] = _num((hr[w] >= base + 10).sum() * ws / 60, 1)
        w = sel(tx, end + 600) & np.isfinite(hr) & np.isfinite(d)
        cand["s_hr_back_to_baseline_while_st_at_half_or_more"] = _num(
            ((np.abs(hr[w] - base) <= 5) & (d[w] >= 0.5 * ext)).sum() * ws)
    p10 = np.nanpercentile(hr, 10)
    w = sel(on - 3600, on) & np.isfinite(hr)
    cand["hr_load_60_min_before_bpm_min"] = _num(np.clip(hr[w] - p10, 0, None).sum() * ws / 60)

    rises = [(i, j) for i, j in _rises(t, hr) if t[j] < on - 600 or t[i] > end + 600]
    comp = []
    if np.isfinite(base) and np.isfinite(hr_max):
        for i, j in rises:
            if abs(hr[i] - base) <= MATCH_BPM and abs(hr[j] - hr_max) <= MATCH_BPM:
                a, b = d[max(0, i - 4):i + 1], d[j:j + 5]
                if np.isfinite(a).any() and np.isfinite(b).any():
                    comp.append((t[i], hr[i], hr[j], float(np.nanmax(b) - np.nanmedian(a))))
    resp = np.array([x[3] for x in comp]) if comp else np.array([])
    summary = {"rises_elsewhere": len(rises), "comparable_rises": len(comp)}
    if comp:
        summary.update(
            comparable_with_st_change_at_least_half_this_extremum=int((resp >= 0.5 * ext).sum()),
            fraction_at_least_half=_num((resp >= 0.5 * ext).mean(), 2),
            median_st_change_of_comparable_uv=_num(np.median(resp)),
            this_extremum_minus_that_median_uv=_num(ext - np.median(resp)))
    shown = sorted(comp, key=lambda x: -x[3])[:LIST_MAX]
    return {
        "candidate_id": c.id, "this_candidate": cand, "comparable_rises_summary": summary,
        "largest_comparable_rises": [{"start": _hms(s), "hr_from": _num(a), "hr_to": _num(b), "st_change_uv": _num(r)}
                                     for s, a, b, r in shown],
        "notes": [
            "ST values are signed towards this candidate's extremum (positive = same direction), at its peak lead",
            "comparable rise: another heart-rate rise of >= 10 bpm within 10 min, outside this candidate +-10 min, "
            "starting within 10 bpm of this baseline and peaking within 10 bpm of this candidate's maximum heart rate",
            "st_change_uv of a rise: ST in the 75 s after its heart-rate peak minus ST at its start",
            "a measurement packet, not a diagnosis; thresholds are engineering defaults"]}
