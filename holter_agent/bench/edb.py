"""Reference annotations of the European ST-T Database (EDB), used as an external test set.

Formats and criteria, from the database's annotation description (physionet.org/files/edb/1.0.0/annotations.shtml):

- ST episodes (STCH annotations): "(ST<n><s>" begin, "AST<n><s><ddd>" peak with the peak |deviation| in uV,
  "ST<n><s>)" end; n = signal 0 or 1, s = "+" or "-". An episode contains >= 30 s at |ST deviation| >= 0.1 mV,
  measured 80 ms after J (60 ms above 120 bpm) against a reference waveform from the first 30 s of the record;
  begin and end are placed where |deviation| < 0.05 mV with < 0.1 mV throughout the previous / following 30 s.
- Axis shifts that look like ST changes (records e0161, e0509, e0601, e0611, e0613, e0615): comment annotations
  with the same text in lower case, "(st<n><s>", "ast<n><s><ddd>", "st<n><s>)".
- T episodes ("(T..", lower case for axis shifts) are not used here.
- Signal quality (NOISE annotations): subtype bit j set = signal j noisy, bit j + 4 set = signal j unreadable
  (table in the description: 0x01 "nc", 0x11 "uc", 0x20 "cu", ...); each annotation holds until the next one.

EDB does not type ST episodes as ischemic or heart-rate related. Every ST episode is returned with kind "ischemic"
(our labelling convention for this external test: the subjects had diagnosed or suspected myocardial ischemia and
the episodes are the ST changes the database marks; see the external-test notes before reading results).
Axis-shift intervals are returned as markers every MARKER_STEP_S seconds from begin to end, so that the LTSTDB
candidate classification (a marker within 60 s of a candidate -> axis_or_conduction) applies unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import wfdb

from .ltstdb import STAnnotations, STEpisode

MARKER_STEP_S = 30.0
_BEGIN = re.compile(r"^\((ST|st)(\d)([+-])$")
_PEAK = re.compile(r"^(AST|ast)(\d)([+-])(\d+)$")
_END = re.compile(r"^(ST|st)(\d)([+-])\)$")
NOISE_SYMBOL = "~"


@dataclass
class EDBExtra:
    axis_shift_episodes: list[STEpisode] = field(default_factory=list)  # kind "axis_shift"
    noisy: list[tuple[int, float, float]] = field(default_factory=list)  # (lead, start_s, end_s)
    n_st_annotations: int = 0


def _quality(subtype: int, lead: int) -> str:
    if subtype >> (lead + 4) & 1:
        return "u"
    return "n" if subtype >> lead & 1 else "c"


def parse_edb_annotations(samples, symbols, subtypes, aux, fs: float, end_s: float,
                          n_leads: int = 2) -> tuple[STAnnotations, EDBExtra]:
    out, extra = STAnnotations(), EDBExtra()
    open_ep: dict[tuple[bool, int], STEpisode] = {}  # (axis_shift, lead)
    quality_since: dict[int, tuple[str, float]] = {j: ("c", 0.0) for j in range(n_leads)}
    for s, sym, sub, a in zip(samples, symbols, subtypes, aux):
        t = float(s) / fs
        a = (a or "").strip("\x00").strip()
        if sym == NOISE_SYMBOL:
            for j in range(n_leads):
                q, since = quality_since[j]
                if q == "u":
                    out.unreadable.append((j, since, t))
                elif q == "n":
                    extra.noisy.append((j, since, t))
                quality_since[j] = (_quality(int(sub), j), t)
            continue
        if m := _BEGIN.match(a):
            key = (m.group(1) == "st", int(m.group(2)))
            if key in open_ep:
                out.unmatched += 1
            open_ep[key] = STEpisode(key[1], "axis_shift" if key[0] else "ischemic", t, t)
            extra.n_st_annotations += 1
        elif m := _PEAK.match(a):
            ep = open_ep.get((m.group(1) == "ast", int(m.group(2))))
            dev = float(m.group(4)) * (1.0 if m.group(3) == "+" else -1.0)
            if ep is not None and (ep.peak_dev_uv is None or abs(dev) > abs(ep.peak_dev_uv)):
                ep.peak_s, ep.peak_dev_uv = t, dev
            extra.n_st_annotations += 1
        elif m := _END.match(a):
            ep = open_ep.pop((m.group(1) == "st", int(m.group(2))), None)
            extra.n_st_annotations += 1
            if ep is None:
                out.unmatched += 1
            elif t > ep.start_s:
                ep.end_s = t
                (extra.axis_shift_episodes if ep.kind == "axis_shift" else out.episodes).append(ep)
    out.unmatched += len(open_ep)
    for j, (q, since) in quality_since.items():
        if q == "u":
            out.unreadable.append((j, since, end_s))
        elif q == "n":
            extra.noisy.append((j, since, end_s))
    for ep in extra.axis_shift_episodes:
        t = ep.start_s
        while t < ep.end_s:
            out.axis_shifts.append((ep.lead, t))
            t += MARKER_STEP_S
        out.axis_shifts.append((ep.lead, ep.end_s))
    out.episodes.sort(key=lambda e: e.start_s)
    return out, extra


def load_edb_annotations(base_path: str | Path, fs: float = 250.0, end_s: float = float("inf"),
                         n_leads: int = 2) -> tuple[STAnnotations, EDBExtra]:
    a = wfdb.rdann(str(base_path), "atr")
    return parse_edb_annotations(a.sample, a.symbol, a.subtype, a.aux_note, fs, end_s, n_leads)
