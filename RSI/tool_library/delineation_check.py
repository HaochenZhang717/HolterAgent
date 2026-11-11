"""A second, independent delineation of this candidate's beats, to check our own J point and ST level: NeuroKit2
(Makowski et al., Behav Res Methods 2021; MIT), method "prominence", run beforehand on 10-s windows of this record with
the record's R peaks (sky/fm_tools_run.py; read here with ctx.precomputed("fm", record)).

Per lead, in the dossier's reference window and at the candidate's peak: NeuroKit2's QRS onset, QRS end (J) and T end
relative to R (median over beats), QRS width, and on the mean beat the level at J+0, J+60 and J+80 ms against the PR
level (the 20 ms before QRS onset); next to it our pipeline's J point, QRS width and ST level (J+80, J+72/64/60 at
higher heart rates) of the 15-s window at the same time, and the differences. The ST change from the reference to the
peak is given by both methods: when they agree, the change does not depend on where our pipeline put J; when they
differ by much, the J point or the PR level is in question at one of the two times. NeuroKit2's J comes earlier than
ours in general (development candidates: median 24 ms earlier, interquartile range 8-40 ms), so the J difference
itself is reported but not flagged; flagged are a change of that difference between reference and peak by more than
20 ms (our J moved relative to NeuroKit2's, 2-6% of development candidates) and ST changes that differ by more than
100 uV (fewer than 10% of development candidates). Both thresholds are engineering defaults set on development data
without the reference labels. NeuroKit2 was not validated on Holter ST episodes. A measurement packet, not a diagnosis.
"""
import numpy as np

J_SHIFT_MS, ST_DIFF_UV = 20.0, 100.0


def _r(x, nd=0):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd) if nd else int(round(x))


def _ours(tab, t, lead):
    i = int(np.clip(np.searchsorted(tab["t_start_s"], t, side="right") - 1, 0, len(tab["t_start_s"]) - 1))
    j = tab.lead_index(lead)
    usable = bool(tab["usable"][i, j])
    return {"j_ms": _r(tab["j_ms"][i]), "qrs_ms": _r(tab["qrs_ms"][i]),
            "st_uv": _r(tab["st_uv"][i, j]) if usable else None, "st_j0_uv": _r(tab["st_j0_uv"][i, j]) if usable else None,
            "usable": usable}


def run(ctx, args):
    rid, cid = args["record_id"], args["candidate_id"]
    c = ctx.candidate(rid, cid)
    tab = ctx.table(rid)
    data = ctx.precomputed("fm", rid)
    wins = {w["window"]: w for w in data["windows"] if w["candidate_id"] == c.id}
    if "peak" not in wins:
        return {"candidate_id": c.id, "error": "no delineation for this candidate"}
    leads = sorted({e.lead for e in c.lead_episodes}, key=lambda x: x != c.peak.lead)
    out = {}
    for lead in leads:
        row = {}
        for kind in ("reference", "peak"):
            w = wins.get(kind)
            nk = (w or {}).get("delineation", {}).get(lead) if w else None
            if w is None:
                continue
            ours = _ours(tab, w["t_s"], lead)
            row[kind] = {"t_s": w["t_s"], "neurokit2": None if not nk or "error" in nk else {
                k: _r(nk.get(k), 1 if k.startswith("st_") else 0) for k in
                ("n_beats", "qrs_onset_ms", "j_ms", "t_end_ms", "qrs_ms", "st_j0_uv", "st_j60_uv", "st_j80_uv")},
                "ours": ours}
            if nk and "error" in nk:
                row[kind]["neurokit2_error"] = nk["error"]
            if row[kind]["neurokit2"] and ours["j_ms"] is not None and nk.get("j_ms") is not None:
                row[kind]["j_neurokit2_minus_ours_ms"] = _r(nk["j_ms"] - ours["j_ms"])
        ref, pk = row.get("reference", {}), row.get("peak", {})
        nk_ref, nk_pk = (ref.get("neurokit2") or {}), (pk.get("neurokit2") or {})
        ch_nk = (nk_pk.get("st_j80_uv") - nk_ref.get("st_j80_uv")) if nk_pk.get("st_j80_uv") is not None and \
            nk_ref.get("st_j80_uv") is not None else None
        o_ref, o_pk = ref.get("ours", {}), pk.get("ours", {})
        ch_ours = (o_pk.get("st_uv") - o_ref.get("st_uv")) if o_pk.get("st_uv") is not None and \
            o_ref.get("st_uv") is not None else None
        flags = []
        jr, jp = ref.get("j_neurokit2_minus_ours_ms"), pk.get("j_neurokit2_minus_ours_ms")
        if jr is not None and jp is not None:
            row["our_j_shift_relative_to_neurokit2_ms"] = _r(jr - jp)  # > 0: our J later at the peak than at reference
            if abs(jp - jr) > J_SHIFT_MS:
                flags.append(f"our J point moved by {_r(jr - jp)} ms relative to NeuroKit2's between reference and peak")
        if ch_nk is not None and ch_ours is not None and abs(ch_nk - ch_ours) > ST_DIFF_UV:
            flags.append(f"the ST change from reference to peak differs between the methods by {_r(ch_nk - ch_ours)} uV")
        row["st_change_reference_to_peak_uv"] = {"neurokit2_j80": _r(ch_nk), "ours": _r(ch_ours)}
        row["disagreements"] = flags
        out[lead] = row
    return {"candidate_id": c.id, "leads": out,
            "notes": ["ours: our pipeline's 15-s window at that time (J and QRS width are shared by all leads; ST at J+80, "
                      "or J+72/64/60 above 100/110/120 bpm)", "neurokit2: median over the 10-s window's beats; ST on the "
                      "mean beat against the 20 ms before QRS onset", "NeuroKit2's J is earlier than ours in general; "
                      "flags: J shift between reference and peak > 20 ms, ST changes differing by > 100 uV (engineering "
                      "defaults)", "a measurement packet, not a diagnosis"]}
