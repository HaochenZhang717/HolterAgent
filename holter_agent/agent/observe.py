"""Multimodal observations: numbers and images of the same data.

A tool with an image kind returns, next to its numbers, an image drawn from the same signal or window
table (see holter_agent.render). The modality of a run decides what the model receives:
- "numbers": the tool results only (images are not rendered; view_ecg is not offered);
- "images":  images, plus only the context of each result (times, leads), no measured values; only
             the tools that have an image (and find_reference_window, which returns a time) are offered;
- "both":    numbers and images.
Image kinds can be switched off one by one to compare their effect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..render import beat_overlay, beat_panels, ecg_strip, st_trend_image
from ..trend.episodes import hms, st_deviation, st_reference
from ..trend.window import read_signal
from .toolkit import HolterToolkit

MODALITIES = ("numbers", "images", "both")
IMAGE_KINDS = ("strip", "beats", "overlay", "trend")
IMAGE_OF_TOOL = {"view_ecg": "strip", "measure_window": "beats", "compare_morphology": "overlay", "st_trend": "trend"}
# Tools offered in an image-only run, and the keys of their results the model still sees there.
IMAGE_ONLY_TOOLS = {
    "view_ecg": ("start_s", "end_s", "hms"),
    "measure_window": ("t_s", "hms", "duration_s"),
    "compare_morphology": ("t_ref_s", "t_s"),
    "st_trend": ("leads", "reference"),
    "find_reference_window": ("t_s", "hms"),
}
CASE_CONTEXT_S = 1800.0  # the case image shows the candidate and this much before and after


@dataclass(frozen=True)
class ObservationConfig:
    modality: str = "both"
    image_kinds: tuple[str, ...] = IMAGE_KINDS

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        unknown = set(self.image_kinds) - set(IMAGE_KINDS)
        if unknown:
            raise ValueError(f"unknown image kinds {sorted(unknown)}; use {IMAGE_KINDS}")

    @property
    def numbers(self) -> bool:
        return self.modality in ("numbers", "both")

    @property
    def images(self) -> bool:
        return self.modality in ("images", "both")

    def image_for(self, tool: str) -> str | None:
        kind = IMAGE_OF_TOOL.get(tool)
        return kind if self.images and kind in self.image_kinds else None

    def offers(self, tool: str) -> bool:
        if tool == "view_ecg":
            return self.image_for(tool) is not None
        if self.modality == "images":
            return tool in IMAGE_ONLY_TOOLS and (tool not in IMAGE_OF_TOOL or self.image_for(tool) is not None)
        return True


def visible_result(cfg: ObservationConfig, tool: str, result: dict) -> dict:
    """What the model sees of a tool result in this modality."""
    if cfg.numbers:
        return result
    keep = IMAGE_ONLY_TOOLS.get(tool, ())
    return {k: result[k] for k in keep if k in result} | {"note": "numbers withheld; read the image"}


def visible_case(cfg: ObservationConfig, overview: dict, candidate: dict) -> tuple[dict, dict]:
    """Recording overview and candidate summary as the model sees them in this modality."""
    if cfg.numbers:
        return overview, candidate
    ov = {k: overview[k] for k in ("record", "duration_h", "leads") if k in overview}
    cand = {k: candidate[k] for k in ("id", "start_s", "end_s", "start_hms", "duration_s") if k in candidate}
    cand["leads"] = sorted(candidate.get("leads", {}))
    return ov, cand


# ---------------------------------------------------------------- rendering
def tool_image(tk: HolterToolkit, tool: str, args: dict, bound: dict) -> bytes:
    """The image that goes with a tool result (same data, same computation)."""
    rid = args["record_id"]
    st = tk._state(rid)
    rec = st.record
    title = f"{rec.name}"
    if tool == "view_ecg":
        start, stop = tk.strip_range(rid, args["t_s"], args.get("duration_s", 10.0))
        seg = read_signal(rec, rec.to_samples(start), rec.to_samples(stop), tk.cfg)
        ann = rec.annotations
        sl = ann.beats_between(seg.start, seg.stop)
        beats = [(s / rec.fs, str(b)) for s, b in zip(ann.beat_samples[sl], ann.beat_symbols[sl])]
        return ecg_strip(seg.signal, rec.fs, seg.leads, start, f"{title}, from {hms(start)}", beats)
    if tool == "measure_window":
        wa = tk._window(st, args["t_s"], args.get("duration_s", 30.0))
        tpl, fid = wa.template, wa.template.fiducials
        t_ms = (np.arange(tpl.data.shape[0]) - tpl.r_index) * 1000.0 / tpl.fs
        iso = tpl.mean_at(fid.iso_ms, tk.cfg.st.point_halfwidth_ms)
        st_ms = fid.j_ms + wa.st.jx_offset_ms if wa.st is not None else np.nan
        usual = st_reference(st.table, bound.get("reference", "global"))[st.table.window_index(args["t_s"])]
        return beat_panels(t_ms, tpl.data - iso[None, :], tpl.leads, fid.iso_ms, fid.j_ms, st_ms, usual,
                           f"{title}, averaged beats around {hms(args['t_s'])}")
    if tool == "compare_morphology":
        dur = args.get("duration_s", 30.0)
        a, b = tk._window(st, args["t_ref_s"], dur), tk._window(st, args["t_s"], dur)
        ta, tb = a.template, b.template
        t_ms = (np.arange(ta.data.shape[0]) - ta.r_index) * 1000.0 / ta.fs
        ia = ta.mean_at(ta.fiducials.iso_ms, tk.cfg.st.point_halfwidth_ms)
        ib = tb.mean_at(tb.fiducials.iso_ms, tk.cfg.st.point_halfwidth_ms)
        return beat_overlay(t_ms, ta.data - ia[None, :], tb.data - ib[None, :], ta.leads,
                            f"{title}, reference {hms(args['t_ref_s'])} vs {hms(args['t_s'])}")
    if tool == "st_trend":
        return trend_image(tk, rid, args["start_s"], args["end_s"], args.get("leads") or None,
                           bound.get("reference", "global"), title=title)
    raise ValueError(f"no image for tool {tool!r}")


def trend_image(tk: HolterToolkit, rid: str, start_s: float, end_s: float, leads: list[str] | None,
                reference: str, spans=None, title: str = "") -> bytes:
    st = tk._state(rid)
    tab = st.table
    leads = leads or tab.leads
    idx = [tab.lead_index(lead) for lead in leads]
    t = tab["t_start_s"] + tab.window_s / 2
    sel = (t >= start_s) & (t < end_s)
    dev = st_deviation(tab, reference)[:, idx]
    with np.errstate(all="ignore"):
        big = float(np.nanmax(np.abs(dev[sel]))) if np.isfinite(dev[sel]).any() else 0.0
    clip = float(min(1000.0, max(250.0, 50.0 * np.ceil(1.15 * big / 50.0))))  # 0.8: fit the largest change
    return st_trend_image(t[sel], dev[sel], tab["hr_bpm"][sel], leads,
                          f"{title or st.record.name}, {hms(start_s)} to {hms(end_s)}", spans, clip_uv=clip)


def case_image(tk: HolterToolkit, rid: str, candidate: dict, reference: str) -> bytes:
    """ST trend of the candidate and its surroundings, attached to the first message of a case."""
    dur = tk._state(rid).record.duration_s
    a = max(0.0, candidate["start_s"] - CASE_CONTEXT_S)
    b = min(dur, candidate["end_s"] + CASE_CONTEXT_S)
    return trend_image(tk, rid, a, b, None, reference, spans=[(candidate["start_s"], candidate["end_s"])],
                       title=f"candidate {candidate['id']}")
