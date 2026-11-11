"""The candidate re-judged against a time-varying ST reference built like that of Smrdel & Jager (CinC 2002;29:325-8),
as in annotator_reference (same constants, same numbers), plus how the change was absorbed.

Three steps, per lead, on the 15-s ST level series (unusable windows missing):
  1. local trend r_l = 30-min centred moving mean, global trend r_g = 4-h moving mean; r1 = r_g where
     |r_g - r_l| > 50 uV, else r_l;
  2. inside each stretch where |r_g - r_l| > 50 uV that contains an ST step of the lead or a QRS-shape step (the
     morphology_steps rule), the reference follows the ST level where it is more than 10 uV from r_g (r2);
  3. lead orientation from d2 = ST - r2 outside this candidate (E, D or U); deviations opposite to the orientation,
     or of less than 25 uV in a U lead, are absorbed, and the updated stretches are smoothed with the 30-min mean (r3).
The candidate is then re-judged with the LTSTDB rule (start beyond 50 uV, at least 30 s at or beyond V, end below
50 uV) at V = 100 and 150 uV.

Added here: which steps made the reference follow the ST level in step 2 (QRS-shape steps and ST-level steps of the
lead, and whether each step's transition lies within 144 s, the step rule's change interval, of the candidate), the
"step" absorption split by whether a QRS-shape step lies near the candidate, and whether the candidate goes against
the lead's orientation. A long change with a quick onset and a plateau is itself detected as an ST-level step, and it
makes the 30-min mean depart from the 4-h mean, so step 2 can follow the candidate's own ST level.
A measurement packet, not a diagnosis.
"""
import numpy as np

from holter_agent.trend.episodes import nan_moving_mean
from holter_agent.trend.steps import CHANGE_S, record_steps

LOCAL_S, GLOBAL_S = 1800.0, 14400.0
DIFF_UV, NEAR_UV, ORIENT_UVS, U_KEEP_UV = 50.0, 10.0, 5000.0, 25.0
START_UV, MIN_RUN_S, END_HOLD_S = 50.0, 30.0, 30.0
MAX_STEPS_LISTED = 8


def _num(x, nd=0):
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _runs(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def references(tab, cand, steps):
    ws = tab.window_s
    t = tab["t_start_s"] + ws / 2
    st = np.where(tab["usable"], tab["st_uv"], np.nan).astype(float)
    rl = nan_moving_mean(st, max(1, int(round(LOCAL_S / ws))))
    rg = nan_moving_mean(st, max(1, int(round(GLOBAL_S / ws))))
    big = np.abs(rg - rl) > DIFF_UV
    r1 = np.where(big, rg, rl)
    r2 = r1.copy()
    qrs_steps = [("qrs_shape", s) for s in steps.get("QRS_distance", [])]
    in_cand = (t >= cand.start_s) & (t <= cand.end_s)
    orient, r3, followed = {}, r2.copy(), {}
    for j, lead in enumerate(tab.leads):
        lead_steps = qrs_steps + [("st_level", s) for s in steps.get(f"ST:{lead}", [])]
        followed[lead] = []
        for a, b in _runs(big[:, j]):
            inside = [(kind, s) for kind, s in lead_steps if t[a] <= s.t_end_s and s.t_start_s <= t[b - 1]]
            if inside:
                seg = slice(a, b)
                follow = np.abs(rg[seg, j] - st[seg, j]) > NEAR_UV
                r2[seg, j] = np.where(follow & np.isfinite(st[seg, j]), st[seg, j], r1[seg, j])
                followed[lead].append((float(t[a]), float(t[b - 1]), inside))
        d2 = st[:, j] - r2[:, j]
        out = np.isfinite(d2) & ~in_cand
        psum = float(np.sum((d2[out] - DIFF_UV)[d2[out] > DIFF_UV]) * ws)
        nsum = float(np.sum(np.abs(d2[out] + DIFF_UV)[d2[out] < -DIFF_UV]) * ws)
        o = "E" if psum - nsum > ORIENT_UVS else "D" if nsum - psum > ORIENT_UVS else "U"
        orient[lead] = {"orientation": o, "excess_above_50_uv_s": _num(psum), "excess_below_minus_50_uv_s": _num(nsum)}
        keep = ((o == "E") & (d2 > 0)) | ((o == "D") & (d2 < 0)) | ((o == "U") & (np.abs(d2) > U_KEEP_UV))
        keep |= ~np.isfinite(d2)
        r3j = np.where(keep, r2[:, j], st[:, j])
        smooth = nan_moving_mean(r3j[:, None], max(1, int(round(LOCAL_S / ws))))[:, 0]
        r3[:, j] = np.where(keep, r3j, smooth)
    return t, st, r1, r2, r3, orient, followed


def episode_overlaps(t, d, sign, cand, v, ws):
    """LTSTDB rule on a signed deviation series: is there an episode at threshold v overlapping the candidate?"""
    x = sign * d
    above = np.where(np.isfinite(x), x > START_UV, False)
    hold = max(1, int(round(END_HOLD_S / ws)))
    # bridge short dips below 50 uV (shorter than the end hold)
    for a, b in _runs(~above):
        if 0 < a and b < len(above) and b - a < hold:
            above[a:b] = True
    need = max(1, int(round(MIN_RUN_S / ws)))
    for a, b in _runs(above):
        if t[b - 1] < cand.start_s or t[a] > cand.end_s:
            continue
        strong = np.where(np.isfinite(x[a:b]), x[a:b] >= v, False)
        if any(e - s >= need for s, e in _runs(strong)):
            return True
    return False


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    tab = ctx.table(rid)
    c = ctx.candidate(rid, cid)
    ws = tab.window_s
    t, st, r1, r2, r3, orient, followed = references(tab, c, record_steps(tab))
    rec = ctx.reference(rid)
    peaks = {}
    for e in c.lead_episodes:
        if e.lead not in peaks or abs(e.peak_uv) > abs(peaks[e.lead].peak_uv):
            peaks[e.lead] = e
    sel = (t >= c.start_s) & (t <= c.end_s)
    lo, hi = c.start_s - CHANGE_S, c.end_s + CHANGE_S
    leads = {}
    for lead, e in peaks.items():
        j = tab.lead_index(lead)
        sign = 1.0 if e.peak_uv >= 0 else -1.0

        def peak(ref):
            x = sign * (st[sel, j] - ref)
            return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
        d3 = st[:, j] - r3[:, j]
        p0, p1, p2, p3 = peak(rec[sel, j]), peak(r1[sel, j]), peak(r2[sel, j]), peak(r3[sel, j])
        absorbed = "none"
        for name, a, b in (("slow_trend", p0, p1), ("step", p1, p2), ("orientation", p2, p3)):
            if np.isfinite(a) and np.isfinite(b) and a >= 100 > b:
                absorbed = name
                break
        behind = []
        for a, b, inside in followed[lead]:
            if a > c.end_s or b < c.start_s:
                continue
            for kind, s in inside:
                behind.append({"kind": kind, "transition": f"{_hms(s.t_start_s)}-{_hms(s.t_end_s)}",
                               "change": _num(s.delta) if kind == "st_level" else _num(s.delta, 2),
                               "near_candidate": bool(s.t_start_s <= hi and s.t_end_s >= lo)})
        behind.sort(key=lambda x: (not x["near_candidate"], x["kind"] != "qrs_shape"))
        qrs_near = any(x["kind"] == "qrs_shape" and x["near_candidate"] for x in behind)
        if absorbed == "step":
            absorbed = ("step_with_qrs_shape_step_near_candidate" if qrs_near
                        else "step_without_qrs_shape_step_near_candidate")
        o = orient[lead]["orientation"]
        rr = r3[sel, j]
        slope = (rr[np.isfinite(rr)][-1] - rr[np.isfinite(rr)][0]) / max(ws, (c.end_s - c.start_s)) * 60 \
            if np.isfinite(rr).sum() >= 2 else np.nan
        leads[lead] = {
            "candidate_sign": "elevation" if sign > 0 else "depression",
            "peak_deviation_uv": {"record_median": _num(p0), "after_slow_trend": _num(p1), "after_steps": _num(p2),
                                  "annotator_like_reference": _num(p3)},
            "episode_at_100_uv": bool(episode_overlaps(t, d3, sign, c, 100.0, ws)),
            "episode_at_150_uv": bool(episode_overlaps(t, d3, sign, c, 150.0, ws)),
            "absorbed_by": absorbed,
            "reference_followed_st_level_during_candidate": bool(behind),
            "steps_behind_reference_following": behind[:MAX_STEPS_LISTED],
            "n_steps_behind_reference_following": len(behind),
            "qrs_shape_step_near_candidate": bool(qrs_near),
            "candidate_against_lead_orientation": bool((o == "E" and sign < 0) or (o == "D" and sign > 0)),
            "reference_slope_during_candidate_uv_per_min": _num(sign * slope, 1),
            **orient[lead],
        }
    return {"candidate_id": c.id, "leads": leads,
            "notes": ["peak deviations are signed towards the candidate (positive = same direction) and taken within "
                      "the candidate; the candidate was detected against record_median",
                      "annotator_like_reference: Smrdel & Jager 2002 three-step reference (slow trend, steps, lead "
                      "orientation); episode_at_*: the LTSTDB episode rule re-applied against it",
                      "absorbed_by: the first step that takes the peak below 100 uV (none, slow_trend, "
                      "step_with_qrs_shape_step_near_candidate, step_without_qrs_shape_step_near_candidate, "
                      "orientation); judged on the peak only",
                      "steps_behind_reference_following: steps inside the stretches (30-min mean more than 50 uV from "
                      "the 4-h mean) where the reference followed the ST level and that overlap the candidate; "
                      "near_candidate = transition within 144 s of the candidate; change in uV (st_level) or in "
                      "SD units of the QRS-shape distance (qrs_shape, thresholds uncalibrated)",
                      "a long change with a quick onset and a plateau is itself detected as an ST-level step",
                      "a measurement packet, not a diagnosis"]}
