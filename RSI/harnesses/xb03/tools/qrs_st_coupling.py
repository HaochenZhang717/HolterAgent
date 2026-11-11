"""How the QRS changed together with this candidate's ST change, lead by lead, from the 15-s feature table.

Per lead: QRS peak-to-peak amplitude (qrs_pp_uv: peak-to-peak of the averaged beat between our QRS onset and our J
point) and QRS area, before the candidate (2-10 min before its start), at the lead's extremum (windows centred within
30 s of it) and after it (2-10 min after its end); the ST change and the J-level (J+0) change at the extremum against
before; a reading of the QRS change; how step-like the QRS and ST changes are (largest change within 60 s as a share of
the change at the extremum); whether the QRS returns afterwards; and the rank correlation of QRS amplitude with ST level
over the candidate +-10 min. Window-level: our J point and QRS width per period.

Readings (thresholds are engineering defaults of our own tools):
  qrs_amplitude_stable       amplitude ratio (extremum / before) within compare_morphology's 0.75-1.33
  qrs_change_within_j_shift  ratio outside 0.75-1.33, but the amplitude changed by no more than the J level moved, and
                             the J level moved by >= 50 uV (the episode rule's onset threshold): the QRS window ends at
                             our J point, so a shifted J end of the QRS can by itself change the amplitude by up to about
                             the J shift; the change may then be the ST shift entering the QRS measure
  qrs_change_beyond_j_shift  ratio outside 0.75-1.33 and the amplitude changed by more than the J level moved
A measurement packet, not a diagnosis.
"""
import numpy as np

BEFORE_S = (600.0, 120.0)   # before: from 10 min to 2 min before the candidate start
WIDE_BEFORE_S = (1800.0, 60.0)
AFTER_S = (120.0, 600.0)    # after: from 2 min to 10 min after the candidate end
AT_S = 30.0                 # extremum: windows centred within 30 s of the lead's extremum
CONTEXT_S = 600.0           # coupling range: candidate +-10 min
RATIO_LO, RATIO_HI = 0.75, 1.33   # compare_morphology's QRS amplitude-ratio rule
SMALL_QRS_UV = 500.0        # compare_morphology's small-QRS warning
J_SHIFT_UV = 50.0           # the episode rule's onset threshold
STEP_S = 60.0


def _num(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _med(x, mask, need):
    v = np.asarray(x, float)[mask]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size >= need else float("nan")


def _rank(x):
    r = np.empty(len(x), float)
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x), dtype=float)
    return r


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _smooth(x):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[max(0, i - 1):i + 2]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out


def _largest_change(x, k):
    if len(x) <= k:
        return float("nan")
    d = np.abs(x[k:] - x[:-k])
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else float("nan")


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    tab = ctx.table(rid)
    c = ctx.candidate(rid, cid)
    ws = float(tab.window_s)
    t = np.asarray(tab["t_start_s"], float) + ws / 2
    fid = np.asarray(tab["fiducials_ok"], bool)
    usable = np.asarray(tab["usable"], bool)
    qpp = np.asarray(tab["qrs_pp_uv"], float)
    qarea = np.asarray(tab["qrs_area_uvs"], float)
    st = np.asarray(tab["st_uv"], float)
    j0 = np.asarray(tab["st_j0_uv"], float)
    j_ms = np.asarray(tab["j_ms"], float)
    qrs_ms = np.asarray(tab["qrs_ms"], float)

    peaks = {}
    for e in c.lead_episodes:
        if e.lead not in peaks or abs(e.peak_uv) > abs(peaks[e.lead].peak_uv):
            peaks[e.lead] = e
    t_peak = float(c.peak.peak_s)

    before_n = (t >= c.start_s - BEFORE_S[0]) & (t < c.start_s - BEFORE_S[1])
    before_w = (t >= c.start_s - WIDE_BEFORE_S[0]) & (t < c.start_s - WIDE_BEFORE_S[1])
    after = (t > c.end_s + AFTER_S[0]) & (t <= c.end_s + AFTER_S[1])
    context = (t >= c.start_s - CONTEXT_S) & (t <= c.end_s + CONTEXT_S)
    k_step = max(1, int(round(STEP_S / ws)))

    leads = {}
    changed_involved, changed_uninvolved = [], []
    for j, lead in enumerate(tab.leads):
        # averaged-beat measures need valid fiducials only; the ST quality flag (usable) is reported, not required,
        # because very large ST shifts can themselves be flagged
        ok = fid
        pp = np.where(ok, qpp[:, j], np.nan)
        ar = np.where(ok, qarea[:, j], np.nan)
        sv = np.where(ok, st[:, j], np.nan)
        jv = np.where(ok, j0[:, j], np.nan)
        e = peaks.get(lead)
        tx = float(e.peak_s) if e is not None else t_peak
        at = np.abs(t - tx) <= AT_S
        if not np.isfinite(pp[at]).any():
            at = np.abs(t - tx) <= 2 * AT_S
        before, compared = before_n, "2-10 min before the candidate start"
        if np.isfinite(pp[before]).sum() < 2:
            before, compared = before_w, "1-30 min before the candidate start"
        if np.isfinite(pp[before]).sum() < 2:
            before, compared = after, "2-10 min after the candidate end (nothing usable before it)"
        pp_b, pp_a, pp_f = _med(pp, before, 2), _med(pp, at, 1), _med(pp, after, 2)
        if not (np.isfinite(pp_b) and np.isfinite(pp_a)):
            leads[lead] = {"involved": e is not None, "usable": False}
            continue
        ar_b, ar_a, ar_f = _med(ar, before, 2), _med(ar, at, 1), _med(ar, after, 2)
        d_pp = pp_a - pp_b
        d_st = _med(sv, at, 1) - _med(sv, before, 2)
        d_j = _med(jv, at, 1) - _med(jv, before, 2)
        ratio = pp_a / pp_b if pp_b > 0 else float("nan")
        if not np.isfinite(ratio):
            reading = None
        elif RATIO_LO <= ratio <= RATIO_HI:
            reading = "qrs_amplitude_stable"
        elif np.isfinite(d_j) and abs(d_j) >= J_SHIFT_UV and abs(d_pp) <= abs(d_j):
            reading = "qrs_change_within_j_shift"
        else:
            reading = "qrs_change_beyond_j_shift"
        x_pp = _smooth(np.where(context, pp, np.nan)[context])
        x_st = _smooth(np.where(context, sv, np.nan)[context])
        big_pp, big_st = _largest_change(x_pp, k_step), _largest_change(x_st, k_step)
        returns = None
        if np.isfinite(pp_f) and abs(d_pp) > 0 and before is not after:
            returns = bool(abs(pp_f - pp_b) <= 0.5 * abs(d_pp))
        entry = {
            "involved": e is not None,
            "candidate_sign": None if e is None else ("elevation" if e.peak_uv >= 0 else "depression"),
            "compared_with": compared,
            "st_usable_share_at_extremum": _num(usable[at, j].mean(), 2) if at.any() else None,
            "qrs_pp_uv_before_extremum_after": [_num(pp_b), _num(pp_a), _num(pp_f)],
            "qrs_amplitude_ratio": _num(ratio, 2),
            "small_qrs_below_500_uv": bool(pp_b < SMALL_QRS_UV),
            "qrs_area_uvs_before_extremum_after": [_num(ar_b), _num(ar_a), _num(ar_f)],
            "qrs_pp_change_uv": _num(d_pp),
            "j_level_change_uv": _num(d_j),
            "st_change_uv": _num(d_st),
            "reading": reading,
            "qrs_largest_60s_change_share": _num(big_pp / abs(d_pp), 2)
            if reading not in (None, "qrs_amplitude_stable") and abs(d_pp) > 0 else None,
            "st_largest_60s_change_share": _num(big_st / abs(d_st), 2)
            if np.isfinite(d_st) and abs(d_st) >= J_SHIFT_UV else None,
            "qrs_back_to_before_level_after": returns,
            "qrs_vs_st_rank_corr_candidate_pm_10_min": _num(_spearman(np.where(context, pp, np.nan)[context],
                                                                    np.where(context, sv, np.nan)[context]), 2),
        }
        leads[lead] = entry
        if reading in ("qrs_change_within_j_shift", "qrs_change_beyond_j_shift"):
            (changed_involved if e is not None else changed_uninvolved).append(f"{lead}: {reading}")

    at_w = np.abs(t - t_peak) <= AT_S
    window = {
        "extremum_t": _hms(t_peak),
        "j_point_ms_before_extremum_after": [_num(_med(j_ms, before_n & fid, 2)), _num(_med(j_ms, at_w & fid, 1)),
                                             _num(_med(j_ms, after & fid, 2))],
        "qrs_width_ms_before_extremum_after": [_num(_med(qrs_ms, before_n & fid, 2)),
                                               _num(_med(qrs_ms, at_w & fid, 1)),
                                               _num(_med(qrs_ms, after & fid, 2))],
    }
    return {
        "candidate_id": c.id,
        "fiducials": window,
        "leads": leads,
        "involved_leads_with_qrs_change": changed_involved,
        "uninvolved_leads_with_qrs_change": changed_uninvolved,
        "notes": [
            "before = 2-10 min before the candidate start (wider or after when nothing usable); extremum = windows "
            "centred within 30 s of that lead's extremum; after = 2-10 min after the candidate end",
            "qrs_pp: peak-to-peak of the averaged beat from our QRS onset to our J point; j_level_change_uv: change of "
            "the level at J+0, st_change_uv: change at the ST measurement point",
            "reading: qrs_amplitude_stable = ratio within compare_morphology's 0.75-1.33; qrs_change_within_j_shift = "
            "outside it, but the amplitude changed by no more than the J level moved (J shift >= 50 uV), so the "
            "shifted J end of the QRS window can explain it; qrs_change_beyond_j_shift = the QRS itself changed",
            "*_largest_60s_change_share: largest change within 60 s (3-window median series, candidate +-10 min) "
            "divided by the change at the extremum; near or above 1 = the change happened within about a minute "
            "(step-like), well below 1 = gradual",
            "values use every window with valid fiducials; st_usable_share_at_extremum is the share of the extremum "
            "windows that also pass our ST quality check (noise), so a low share means noisier numbers",
            "a J point that moves later and a QRS that widens at the extremum mean our QRS window takes in more of the "
            "shifted ST segment",
            "a measurement packet, not a diagnosis; thresholds are engineering defaults of our own tools"],
    }
