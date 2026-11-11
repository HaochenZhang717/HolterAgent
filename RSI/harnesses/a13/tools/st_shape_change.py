"""How the ST segment's shape changed at the candidate's extremum, against the same lead's own beats.

Per involved lead, on the 15-s template measurements of the feature table (J point and ST point relative to the
isoelectric level, the J-to-ST-point slope, the T-wave extreme and the QRS peak-to-peak amplitude):
- extremum: median of the usable windows at the lead's extremum and the window on each side;
- before: median of the usable windows from 10 min to 30 s before the candidate's onset, outside every candidate of
  the record (30 min before when fewer than 4 such windows);
- matched_hr: median of the 12 usable windows nearest in time (within 2 h, outside every candidate) whose heart rate
  is within 5 bpm of the heart rate at the extremum.
For each reference: the change at the J point, at the ST point and of the T wave, signed towards the candidate
(positive = in the candidate's direction), the share of the ST-point change already present at the J point, and a
description of where along the segment the change sits. The ST point follows heart rate (J+80 ms below 100 bpm,
J+72, J+64, J+60 ms at 120 bpm or more, the LTSTDB convention), so it may differ between reference and extremum.
Thresholds are engineering defaults (50 uV = the LTSTDB episode start level). A measurement packet, not a diagnosis.
"""
import numpy as np

BEFORE_S, BEFORE_WIDE_S, GAP_S = 600.0, 1800.0, 30.0
MATCH_BPM, MATCH_MAX_S, MATCH_N = 5.0, 7200.0, 12
MIN_WINDOWS = 4
MIN_CHANGE_UV = 50.0
J_HEAVY, PARALLEL = 1.5, 0.67
T_RATIO = 2.0
HR_BREAKS, OFFSETS_MS = (100.0, 110.0, 120.0), (80, 72, 64, 60)
COLS = ("st_j0_uv", "st_uv", "st_slope", "t_amp_uv", "qrs_pp_uv")


def _num(x, nd=0):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _st_point_ms(hr):
    if not np.isfinite(hr):
        return None
    for brk, off in zip(HR_BREAKS, OFFSETS_MS):
        if hr < brk:
            return off
    return OFFSETS_MS[-1]


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def _summary(cols, hr, t, idx, j):
    v = {k: _med(cols[k][idx, j]) for k in COLS}
    h = _med(hr[idx])
    return v, h, {
        "time": f"{_hms(t[idx].min())}-{_hms(t[idx].max())}", "n_windows": int(len(idx)), "hr_bpm": _num(h),
        "j_point_uv": _num(v["st_j0_uv"]), "st_point_uv": _num(v["st_uv"]), "st_point_ms_after_j": _st_point_ms(h),
        "j_to_st_slope_mv_per_s": _num(v["st_slope"], 2), "t_amp_uv": _num(v["t_amp_uv"]),
        "qrs_pp_uv": _num(v["qrs_pp_uv"])}


def _pattern(jc, sc, tc, depression):
    if not (np.isfinite(jc) and np.isfinite(sc)):
        return None
    if sc < MIN_CHANGE_UV:
        if jc < MIN_CHANGE_UV:
            return "little change in the candidate's direction at the J point or the ST point against this reference"
        return "the J point moved in the candidate's direction but the ST point stayed within 50 uV of this reference"
    r = jc / sc
    if r >= J_HEAVY:
        out = ("change concentrated at the J point: the ST point moved less than J, so the segment after J slopes back "
               "towards the reference level (" + ("more upsloping" if depression else "less upsloping or more "
                                                  "downsloping") + " than in the reference)")
    elif r >= PARALLEL:
        out = "J point and ST point moved by similar amounts (parallel shift of the segment)"
    else:
        out = ("change concentrated at the ST point: J moved less than the ST point (" +
               ("the segment became less upsloping or more downsloping than in the reference" if depression else
                "the late segment rose more than J, the segment became more upsloping than in the reference") + ")")
    if np.isfinite(tc) and tc >= T_RATIO * sc:
        out += "; the T wave changed in the same direction by at least twice the ST-point change"
    return out


def _compare(cols, hr, t, idx, j, ext, sign, depression):
    if len(idx) < MIN_WINDOWS:
        return {"n_windows": int(len(idx)), "note": f"fewer than {MIN_WINDOWS} usable windows; no comparison"}
    ref, _, out = _summary(cols, hr, t, idx, j)
    jc = sign * (ext["st_j0_uv"] - ref["st_j0_uv"])
    sc = sign * (ext["st_uv"] - ref["st_uv"])
    tc = sign * (ext["t_amp_uv"] - ref["t_amp_uv"])
    out["change_towards_candidate_uv"] = {"j_point": _num(jc), "st_point": _num(sc), "t_wave": _num(tc)}
    out["share_of_st_point_change_present_at_j"] = _num(jc / sc, 2) if np.isfinite(sc) and sc >= MIN_CHANGE_UV \
        else None
    out["qrs_pp_ratio"] = _num(ext["qrs_pp_uv"] / ref["qrs_pp_uv"], 2) if ref["qrs_pp_uv"] else None
    out["pattern"] = _pattern(jc, sc, tc, depression)
    return out


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    tab = ctx.table(rid)
    c = ctx.candidate(rid, cid)
    ws = tab.window_s
    t = np.asarray(tab["t_start_s"], float) + ws / 2
    hr = np.asarray(tab["hr_bpm"], float)
    usable = np.asarray(tab["usable"]).astype(bool)
    cols = {k: np.asarray(tab[k], float) for k in COLS}
    outside = np.ones(len(t), bool)
    for o in ctx.candidates(rid):
        outside &= ~((t >= o.start_s - ws) & (t <= o.end_s + ws))
    peaks = {}
    for e in c.lead_episodes:
        if e.lead not in peaks or abs(e.peak_uv) > abs(peaks[e.lead].peak_uv):
            peaks[e.lead] = e
    leads = {}
    for lead, e in sorted(peaks.items(), key=lambda kv: -abs(kv[1].peak_uv)):
        j = tab.lead_index(lead)
        sign = 1.0 if e.peak_uv >= 0 else -1.0
        depression = sign < 0
        ok = usable[:, j] & np.isfinite(cols["st_uv"][:, j]) & np.isfinite(cols["st_j0_uv"][:, j])
        k = tab.window_index(e.peak_s)
        idx = np.array([i for i in (k - 1, k, k + 1) if 0 <= i < len(t) and ok[i]], int)
        if not idx.size:
            leads[lead] = {"candidate_sign": "depression" if depression else "elevation",
                           "note": "no usable window at the extremum"}
            continue
        ext, hr_ext, ext_out = _summary(cols, hr, t, idx, j)
        res = {"candidate_sign": "depression" if depression else "elevation", "extremum": ext_out}
        before = np.flatnonzero(ok & outside & (t >= c.start_s - BEFORE_S) & (t <= c.start_s - GAP_S))
        if before.size < MIN_WINDOWS:
            before = np.flatnonzero(ok & outside & (t >= c.start_s - BEFORE_WIDE_S) & (t <= c.start_s - GAP_S))
        res["before"] = _compare(cols, hr, t, before, j, ext, sign, depression)
        if np.isfinite(hr_ext):
            m = np.flatnonzero(ok & outside & np.isfinite(hr) & (np.abs(hr - hr_ext) <= MATCH_BPM)
                               & (np.abs(t - e.peak_s) <= MATCH_MAX_S))
            m = np.sort(m[np.argsort(np.abs(t[m] - e.peak_s))[:MATCH_N]])
        else:
            m = np.array([], int)
        res["matched_hr"] = _compare(cols, hr, t, m, j, ext, sign, depression)
        leads[lead] = res
    return {
        "candidate_id": c.id, "leads": leads,
        "notes": [
            "J and ST points are relative to the isoelectric level of each 15-s averaged beat; changes are extremum "
            "minus reference, signed towards the candidate (positive = in the candidate's direction)",
            "share_of_st_point_change_present_at_j: J-point change / ST-point change; about 1 = parallel shift, above "
            "1.5 = change concentrated at J, below 0.67 = change concentrated at the ST point (engineering defaults)",
            "before: 10 min to 30 s before onset outside all candidates (30 min when too few windows); it can be at a "
            "different heart rate and, for a fragment of a longer change, already shifted",
            "matched_hr: windows outside all candidates within 5 bpm of the extremum's heart rate, within 2 h",
            "when st_point_ms_after_j differs between reference and extremum, part of the ST-point change comes from "
            "measuring at another point of the segment (more for a steep segment)",
            "a measurement packet, not a diagnosis"]}
