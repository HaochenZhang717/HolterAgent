"""ST deviation and candidate ST episodes.

Candidates are produced by a fixed rule and are only a starting point: the agent
decides what each candidate is (ischemic or one of the non-ischemic causes).

Episode rule (LTSTDB description, physionet.org/content/ltstdb/1.0.0): an episode
begins when |ST deviation| first exceeds 50 uV, must reach Vmin for at least
Tmin, and ends when the deviation falls below 50 uV and does not exceed 50 uV in
the following 30 s. We apply it per lead on the window series and require the
Vmin stretch to be contiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EpisodeConfig
from .features import FeatureTable

REFERENCES = ("global", "tracked", "hr_matched")

# Heart-rate matched reference: ST deviation "relative to each patient's own baseline median
# beat matched by heart rate" (Schipaanboord et al., IJC Heart Vasc 2026;63:101884). Our
# version: per-lead median ST of usable windows in the same heart-rate bin. Removing the
# heart-rate dependence also removes demand-ischemia signal, so use it as an alternative
# view, not as the default.
HR_BIN_BPM = 5.0  # default
HR_BIN_MIN_WINDOWS = 20  # default: sparser bins fall back to the global reference

# Reference tracking of Smrdel & Jager (Computers in Cardiology 2002;29:325-328), step 1:
# local = 30-min moving average, global = 4-h moving average of the ST level; the
# reference is the global one where they differ by more than 50 uV, else the local one.
TRACK_LOCAL_S = 30 * 60.0
TRACK_GLOBAL_S = 4 * 3600.0
TRACK_SWITCH_UV = 50.0


def nan_moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean along axis 0 ignoring NaN; NaN where the window has no data."""
    valid = np.isfinite(x)
    csum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.where(valid, x, 0.0), axis=0)])
    ccnt = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(valid, axis=0)])
    n = len(x)
    half = width // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(all="ignore"):
        return np.where(count > 0, total / count, np.nan)


def st_reference(table: FeatureTable, kind: str = "global") -> np.ndarray:
    """Reference ST level [W, L] in uV.

    global  : per-lead median over usable windows of the whole record.
    tracked : Smrdel & Jager (2002) step-1 tracking of the time-varying reference; it
              follows slow drift, but a long real episode can pull it along.
    hr_matched : per-lead median over usable windows with a similar heart rate.
    """
    st = np.where(table["usable"], table["st_uv"], np.nan).astype(np.float64)
    n_w, n_l = st.shape
    if kind == "global":
        with np.errstate(all="ignore"):
            ref = np.nanmedian(st, axis=0)
        return np.broadcast_to(ref, (n_w, n_l)).copy()
    if kind == "tracked":
        local = nan_moving_mean(st, max(1, int(round(TRACK_LOCAL_S / table.window_s))))
        glob = nan_moving_mean(st, max(1, int(round(TRACK_GLOBAL_S / table.window_s))))
        return np.where(np.abs(glob - local) > TRACK_SWITCH_UV, glob, local)
    if kind == "hr_matched":
        with np.errstate(all="ignore"):
            glob = np.nanmedian(st, axis=0)
        out = np.broadcast_to(glob, (n_w, n_l)).copy()
        hr = table["hr_bpm"]
        bins = np.floor(hr / HR_BIN_BPM)
        for b in np.unique(bins[np.isfinite(bins)]):
            rows = bins == b
            block = st[rows]
            enough = np.isfinite(block).sum(axis=0) >= HR_BIN_MIN_WINDOWS
            with np.errstate(all="ignore"):
                med = np.nanmedian(block, axis=0)
            out[np.ix_(rows, enough)] = med[enough]
        return out
    raise ValueError(f"unknown reference {kind!r}; choose from {REFERENCES}")


def st_deviation(table: FeatureTable, reference: str = "global") -> np.ndarray:
    """ST deviation [W, L] in uV; NaN where the lead window is unusable."""
    ref = st_reference(table, reference)
    return np.where(table["usable"], table["st_uv"] - ref, np.nan)


@dataclass
class LeadEpisode:
    lead: str
    sign: int  # -1 depression, +1 elevation
    start_s: float
    end_s: float
    peak_uv: float  # signed extreme deviation
    peak_s: float


@dataclass
class Candidate:
    id: str
    start_s: float
    end_s: float
    protocol: str
    reference: str
    lead_episodes: list[LeadEpisode] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def leads(self) -> list[str]:
        return sorted({e.lead for e in self.lead_episodes})

    @property
    def peak(self) -> LeadEpisode:
        return max(self.lead_episodes, key=lambda e: abs(e.peak_uv))

    def summary(self) -> dict:
        per_lead: dict[str, dict] = {}
        for e in self.lead_episodes:
            cur = per_lead.get(e.lead)
            if cur is None or abs(e.peak_uv) > abs(cur["peak_uv"]):
                per_lead[e.lead] = {"sign": "depression" if e.sign < 0 else "elevation",
                                    "peak_uv": round(e.peak_uv), "peak_s": round(e.peak_s)}
        return {
            "id": self.id,
            "start_s": round(self.start_s),
            "end_s": round(self.end_s),
            "start_hms": hms(self.start_s),
            "duration_s": round(self.duration_s),
            "n_leads": len(per_lead),
            "leads": per_lead,
        }


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def lead_episodes(
    x: np.ndarray, window_s: float, vmin: float, tmin: float, cfg: EpisodeConfig
) -> list[tuple[int, int]]:
    """Episodes on one directed series x (positive = deviation in the direction of interest).

    Returns half-open window ranges [start, end). NaN windows count as unusable.
    """
    n = len(x)
    hold = int(np.ceil(cfg.end_hold_s / window_s))
    max_gap = int(np.floor(cfg.max_unusable_gap_s / window_s))
    need = int(np.ceil(tmin / window_s))
    above = np.where(np.isfinite(x), x > cfg.onset_uv, False)
    valid = np.isfinite(x)
    out: list[tuple[int, int]] = []
    w = 0
    while w < n:
        if not above[w]:
            w += 1
            continue
        start, last = w, w
        k = w + 1
        nan_run = 0
        while k < n:
            if not valid[k]:
                nan_run += 1
                if nan_run > max_gap:
                    break
            elif above[k]:
                last, nan_run = k, 0
            else:
                nan_run = 0
                nxt = above[k + 1:k + 1 + hold]
                if not nxt.any():
                    break
            k += 1
        seg = x[start:last + 1]
        strong = np.where(np.isfinite(seg), seg >= vmin, False)
        best = run = 0
        for s in strong:
            run = run + 1 if s else 0
            best = max(best, run)
        if best >= need:
            out.append((start, last + 1))
        w = last + 1
    return out


def detect_candidates(
    table: FeatureTable,
    cfg: EpisodeConfig,
    protocol: str = "B",
    reference: str = "global",
) -> list[Candidate]:
    vmin, tmin = cfg.protocols[protocol]
    dev = st_deviation(table, reference)
    t = table["t_start_s"]
    ws = table.window_s
    eps: list[LeadEpisode] = []
    for j, lead in enumerate(table.leads):
        for sign in (-1, 1):
            x = sign * dev[:, j]
            for a, b in lead_episodes(x, ws, vmin, tmin, cfg):
                seg = x[a:b]
                k = a + int(np.nanargmax(seg))
                eps.append(LeadEpisode(lead, sign, float(t[a]), float(t[b - 1] + ws),
                                       float(dev[k, j]), float(t[k] + ws / 2)))
    eps.sort(key=lambda e: e.start_s)
    groups: list[list[LeadEpisode]] = []
    end = -np.inf
    for e in eps:
        if groups and e.start_s <= end:
            groups[-1].append(e)
            end = max(end, e.end_s)
        else:
            groups.append([e])
            end = e.end_s
    # Candidates are named by record, rule and start time (e.g. 0010-B-022615), so that a name does
    # not change when candidates elsewhere in the record appear or disappear. Merged groups do not
    # overlap, so their start times are distinct.
    rec = table.record.split("-")[-1]
    tag = protocol + {"global": "", "tracked": "t", "hr_matched": "h"}[reference]
    out = []
    for g in groups:
        start = min(e.start_s for e in g)
        out.append(Candidate(f"{rec}-{tag}-{hms(start).replace(':', '')}", start, max(e.end_s for e in g),
                             protocol, reference, g))
    return out
