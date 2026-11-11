"""Reference annotations of the Long-Term ST Database (LTSTDB), used to validate Steps 1 and 2.

Formats, from the LTSTDB annotation code table (physionet.org/files/ltstdb/1.0.0/tables/acodes.png):

- .16a: one annotation per beat and lead with aux
  "[DMY]ST<n><aa>,<bb>,<cc>,<dd>,<ee>,<ff>,<gg>,<hh>,<ISO>,<JP>,<J80>,<NL>,<NR>":
  aa = ST level at J+80(60) ms in uV; bb..hh = amplitude at J+0, 20, 40, 60, 80, 100, 120 ms in uV;
  ISO = isoelectric point before the fiducial point (ms); JP = J point after it (ms);
  J80 = measurement point after it (ms); NL/NR = beats averaged to the left/right;
  DMY = isoelectric level and J point placed manually.
- .sta/.stb/.stc (annotation protocols A/B/C):
  "(st<n>±dd" / "ast<n>±dd" / "st<n>±dd)"  begin / extremum / end of an ischemic ST episode
  ("rt" before "st": heart-rate related episode); "sst<n>" axis shift; "sccst<n>" conduction change;
  "noi<n>±dd" noise; "(urd<n>" / "urd<n>)" unreadable interval; "GRST<n>" / "LRST<n>±ll" global and
  local reference. n is the lead number, dd a deviation and ll a level, both in uV.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wfdb

EPISODE_KINDS = {"": "ischemic", "rt": "heart_rate_related"}
_BEGIN = re.compile(r"^\((rt)?st(\d)([+-]\d+)$")
_PEAK = re.compile(r"^a(rt)?st(\d)([+-]\d+)$")
_END = re.compile(r"^(rt)?st(\d)([+-]\d+)\)$")
_SHIFT = re.compile(r"^s(cc)?st(\d)$")
_NOISE = re.compile(r"^noi(\d)([+-]\d+)$")
_URD_BEGIN = re.compile(r"^\(urd(\d)$")
_URD_END = re.compile(r"^urd(\d)\)$")
_ST16 = re.compile(r"^(DMY)?ST(\d)(.*)$")


@dataclass
class STEpisode:
    lead: int
    kind: str  # ischemic | heart_rate_related
    start_s: float
    end_s: float
    peak_s: float | None = None
    peak_dev_uv: float | None = None


@dataclass
class STAnnotations:
    episodes: list[STEpisode] = field(default_factory=list)
    axis_shifts: list[tuple[int, float]] = field(default_factory=list)  # (lead, t_s)
    conduction_changes: list[tuple[int, float]] = field(default_factory=list)
    noise: list[tuple[int, float]] = field(default_factory=list)
    unreadable: list[tuple[int, float, float]] = field(default_factory=list)  # (lead, start_s, end_s)
    unmatched: int = 0  # begin or end markers without a partner


def parse_st_annotations(samples, aux, fs: float, end_s: float) -> STAnnotations:
    """Episodes and events from .sta/.stb/.stc annotations (sample indices and aux strings)."""
    out = STAnnotations()
    open_ep: dict[tuple[str, int], STEpisode] = {}
    open_urd: dict[int, float] = {}
    for s, a in zip(samples, aux):
        t = float(s) / fs
        if m := _BEGIN.match(a):
            key = (EPISODE_KINDS[m.group(1) or ""], int(m.group(2)))
            if key in open_ep:
                out.unmatched += 1
            open_ep[key] = STEpisode(key[1], key[0], t, t)
        elif m := _PEAK.match(a):
            ep = open_ep.get((EPISODE_KINDS[m.group(1) or ""], int(m.group(2))))
            dev = float(m.group(3))
            if ep is not None and (ep.peak_dev_uv is None or abs(dev) > abs(ep.peak_dev_uv)):
                ep.peak_s, ep.peak_dev_uv = t, dev
        elif m := _END.match(a):
            ep = open_ep.pop((EPISODE_KINDS[m.group(1) or ""], int(m.group(2))), None)
            if ep is None:
                out.unmatched += 1
            elif t > ep.start_s:
                ep.end_s = t
                out.episodes.append(ep)
        elif m := _SHIFT.match(a):
            (out.conduction_changes if m.group(1) else out.axis_shifts).append((int(m.group(2)), t))
        elif m := _NOISE.match(a):
            out.noise.append((int(m.group(1)), t))
        elif m := _URD_BEGIN.match(a):
            open_urd[int(m.group(1))] = t
        elif m := _URD_END.match(a):
            lead = int(m.group(1))
            if lead in open_urd:
                out.unreadable.append((lead, open_urd.pop(lead), t))
    out.unmatched += len(open_ep)
    out.unreadable += [(lead, t, end_s) for lead, t in open_urd.items()]
    out.episodes.sort(key=lambda e: e.start_s)
    return out


@dataclass
class ST16a:
    sample: np.ndarray  # fiducial point of the measured beat
    lead: np.ndarray
    st_uv: np.ndarray  # ST level at J+80(60) ms
    iso_ms: np.ndarray  # isoelectric point, ms before the fiducial point
    j_ms: np.ndarray  # J point, ms after the fiducial point
    point_ms: np.ndarray  # ST measurement point, ms after the fiducial point
    manual: np.ndarray  # isoelectric level and J point placed manually


def parse_16a(samples, aux) -> ST16a:
    rows = []
    for s, a in zip(samples, aux):
        m = _ST16.match(a)
        if not m:
            continue
        v = [int(x) for x in m.group(3).split(",")]
        rows.append((int(s), int(m.group(2)), v[0], v[8], v[9], v[10], m.group(1) is not None))
    if not rows:
        empty = np.array([], float)
        return ST16a(empty.astype(int), empty.astype(int), empty, empty, empty, empty, empty.astype(bool))
    arr = list(zip(*rows))
    return ST16a(np.array(arr[0]), np.array(arr[1]), np.array(arr[2], float), np.array(arr[3], float),
                 np.array(arr[4], float), np.array(arr[5], float), np.array(arr[6], bool))


def load_st_annotations(base_path: str | Path, protocol: str = "B", fs: float = 250.0,
                        end_s: float = float("inf")) -> STAnnotations:
    a = wfdb.rdann(str(base_path), "st" + protocol.lower())
    return parse_st_annotations(a.sample, a.aux_note, fs, end_s)


def load_16a(base_path: str | Path) -> ST16a:
    a = wfdb.rdann(str(base_path), "16a")
    return parse_16a(a.sample, a.aux_note)
