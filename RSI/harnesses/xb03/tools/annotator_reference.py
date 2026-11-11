"""The candidate re-judged against a time-varying ST reference built like that of Smrdel & Jager (CinC 2002;29:325-8),
who approximated the Long-Term ST Database annotators' reference: annotators followed slow drifts, axis shifts and
conduction changes with the reference, so an ST change that exceeds 100 uV against the record-wide median may stay
below it against their reference.

Three steps, per lead, on the 15-s ST level series (unusable windows missing):
  1. local trend r_l = 30-min centred moving mean, global trend r_g = 4-h moving mean (the paper's 900 and 7200
     samples at 0.5 Hz); r1 = r_g where |r_g - r_l| > 50 uV, else r_l;
  2. inside each stretch where |r_g - r_l| > 50 uV that contains an ST step of the lead or a QRS-shape step (the
     morphology_steps rule), the reference follows the ST level where it is more than 10 uV from r_g (r2);
  3. lead orientation from d2 = ST - r2 outside this candidate: E (elevations) or D (depressions) when the summed
     excess beyond +-50 uV of one sign exceeds the other by more than 5000 uV*s (the paper's 2500 uV over 2-s
     samples), else U; deviations opposite to the orientation, or of less than 25 uV in a U lead, are absorbed into
     the reference, and the updated stretches are smoothed with the 30-min mean (r3).
The candidate is then re-judged with the LTSTDB rule (start beyond 50 uV, at least 30 s at or beyond V, end below
50 uV) at V = 100 uV (protocol B) and 150 uV (the threshold Smrdel & Jager needed to match the protocol B annotations
with their automatic reference: gross sensitivity 74.9%, positive predictivity 75.4%). Excluding the candidate from
the orientation sums is our adaptation. A measurement packet, not a diagnosis; constants are the paper's.
"""
import numpy as np

from holter_agent.trend.episodes import nan_moving_mean
from holter_agent.trend.steps import record_steps

LOCAL_S, GLOBAL_S = 1800.0, 14400.0
DIFF_UV, NEAR_UV, ORIENT_UVS, U_KEEP_UV = 50.0, 10.0, 5000.0, 25.0
START_UV, MIN_RUN_S, END_HOLD_S = 50.0, 30.0, 30.0


def _num(x, nd=0):
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


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
    qrs_steps = [(s.t_start_s, s.t_end_s) for s in steps.get("QRS_distance", [])]
    in_cand = (t >= cand.start_s) & (t <= cand.end_s)
    orient, r3 = {}, r2.copy()
    for j, lead in enumerate(tab.leads):
        lead_steps = qrs_steps + [(s.t_start_s, s.t_end_s) for s in steps.get(f"ST:{lead}", [])]
        for a, b in _runs(big[:, j]):
            if any(t[a] <= e and s <= t[b - 1] for s, e in lead_steps):
                seg = slice(a, b)
                follow = np.abs(rg[seg, j] - st[seg, j]) > NEAR_UV
                r2[seg, j] = np.where(follow & np.isfinite(st[seg, j]), st[seg, j], r1[seg, j])
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
    return t, st, r1, r2, r3, orient


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
    t, st, r1, r2, r3, orient = references(tab, c, record_steps(tab))
    rec = ctx.reference(rid)
    peaks = {}
    for e in c.lead_episodes:
        if e.lead not in peaks or abs(e.peak_uv) > abs(peaks[e.lead].peak_uv):
            peaks[e.lead] = e
    sel = (t >= c.start_s) & (t <= c.end_s)
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
        rr = r3[sel, j]
        slope = (rr[np.isfinite(rr)][-1] - rr[np.isfinite(rr)][0]) / max(ws, (c.end_s - c.start_s)) * 60 \
            if np.isfinite(rr).sum() >= 2 else np.nan
        leads[lead] = {
            "candidate_sign": "elevation" if sign > 0 else "depression",
            "peak_deviation_uv": {"record_median": _num(p0), "after_slow_trend": _num(p1), "after_steps": _num(p2),
                                  "annotator_like_reference": _num(p3)},
            "episode_at_100_uv": episode_overlaps(t, d3, sign, c, 100.0, ws),
            "episode_at_150_uv": episode_overlaps(t, d3, sign, c, 150.0, ws),
            "first_step_dropping_it_below_100_uv": absorbed,
            "reference_slope_during_candidate_uv_per_min": _num(sign * slope, 1),
            **orient[lead],
        }
    return {"candidate_id": c.id, "leads": leads,
            "notes": ["peak deviations are signed towards the candidate (positive = same direction) and taken within "
                      "the candidate; the candidate was detected against record_median",
                      "annotator_like_reference: Smrdel & Jager 2002 three-step reference (slow trend, steps, lead "
                      "orientation); episode_at_*: the LTSTDB episode rule re-applied against it",
                      "Smrdel & Jager needed 150 uV with their automatic reference to match the protocol B annotations; "
                      "a candidate that is an episode at 100 but not at 150 uV is in the uncertain zone",
                      "a long real episode can pull the 30-min trend along and be partly absorbed; read with the "
                      "trend image",
                      "a measurement packet, not a diagnosis"]}
