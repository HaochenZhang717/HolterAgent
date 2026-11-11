"""Central configuration.

Every threshold used by a tool lives here, so experiments can change them in one
place and a run can record exactly which values it used.

Two kinds of values appear below:
- Values taken from a verified source. The source is named next to the value.
- Engineering defaults. These are marked "default" and have no clinical authority;
  tune them on data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class FilterConfig:
    # Power-line interference is removed with a zero-phase notch filter before any measurement.
    # default: 50 Hz, the line frequency seen in these recordings (no 60 Hz component); None
    # disables the filter.
    mains_hz: float | None = 50.0
    # default: quality factor; the -3 dB bandwidth is mains_hz / notch_q (1.7 Hz). On 240 clean
    # windows from 6 records the filter changed ST by 1 uV (median) and 7 uV (p95), no bias.
    notch_q: float = 30.0
    # default: extra signal read on each side so that the filter's start-up transient is discarded.
    pad_s: float = 1.0


@dataclass(frozen=True)
class BeatSelectionConfig:
    # Beat symbols accepted as normally conducted for ST templates.
    normal_symbols: tuple[str, ...] = ("N",)
    # default: prev and next RR must lie within this fraction of the local median RR,
    # which drops beats around ectopy and pauses.
    rr_tolerance: float = 0.20
    # default: minimum number of beats before a template is built.
    min_beats: int = 6


@dataclass(frozen=True)
class TemplateConfig:
    pre_ms: float = 300.0
    post_ms: float = 500.0
    # default: rough isoelectric window relative to R, used before fiducials are known.
    rough_iso_ms: tuple[float, float] = (-90.0, -60.0)


@dataclass(frozen=True)
class FiducialConfig:
    # default: window (relative to R) searched for QRS activity.
    search_ms: tuple[float, float] = (-150.0, 200.0)
    # default: fraction of the peak spatial velocity that counts as QRS activity.
    threshold_frac: float = 0.10
    # default: gaps of low activity shorter than this are merged into one QRS.
    merge_gap_ms: float = 20.0
    # default: isoelectric point searched in this many ms before QRS onset.
    iso_search_ms: float = 80.0
    iso_len_ms: float = 16.0
    # default: among the iso_len windows in the search range whose mean spatial velocity is within
    # this factor of the flattest one, the one closest to QRS onset (end of the PR segment) is used.
    # Taking the flattest alone let the point jump between the PR segment and the P wave: a 50 Hz
    # notch filter then moved ST by more than 40 uV in 17-18% of the windows of records 0071 and 0014.
    iso_flat_ratio: float = 1.5
    # default: plausible QRS duration range; outside it the fiducials are flagged.
    qrs_ms_range: tuple[float, float] = (40.0, 200.0)
    # default: a lead whose beat-to-template residual outside the QRS search window exceeds the
    # limit is left out of the spatial velocity, so one noisy lead cannot hide the QRS boundaries
    # of the others. The limits are tried in order; the stricter one is used only when the first
    # gives no plausible QRS, because leads hovering around a strict limit would change the lead
    # set, and with it J, from window to window.
    # 100 uV: on record 0010 (before filtering) clean leads stayed below 78 uV (p99), the lead that
    # broke detection exceeded 197 uV in 90% of the failed windows. 50 uV: after the 50 Hz filter,
    # 95% of lead values in 300 clean windows from 12 records were below 49 uV, while in the windows
    # of record 0010 that still failed the noisy lead had 57-82 uV (p10-p90).
    exclude_noise_uv: tuple[float, ...] = (100.0, 50.0)
    # default: the fiducials always use at least this many leads (the least noisy ones).
    min_leads: int = 3


@dataclass(frozen=True)
class STConfig:
    # ST measurement point after J, adjusted for heart rate. Source: LTSTDB annotation
    # code table (physionet.org/files/ltstdb/1.0.0/tables/acodes.png):
    # J+80 ms if HR < 100, J+72 if 100-110, J+64 if 110-120, J+60 if HR >= 120 bpm.
    hr_breaks_bpm: tuple[float, ...] = (100.0, 110.0, 120.0)
    offsets_ms: tuple[float, ...] = (80.0, 72.0, 64.0, 60.0)
    # Extra fixed points reported for ST shape (J+0 ... J+80 as in the LTSTDB .16a file).
    extra_points_ms: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0)
    # default: half-width of the averaging window around each measurement point.
    point_halfwidth_ms: float = 8.0
    # default: T-wave search window after J.
    t_window_ms: tuple[float, float] = (80.0, 400.0)


@dataclass(frozen=True)
class QualityConfig:
    # default: a flat line is a run of identical samples at least this long.
    flat_run_ms: float = 500.0
    # default: a lead window is usable only if all of these hold.
    max_saturation_frac: float = 0.0
    max_flat_frac: float = 0.05
    # default: on 2178 low-noise lead-windows from 5 development records the ST-region
    # residual of clean windows was mostly below 40 uV, while visibly noisy ones
    # exceeded 100 uV.
    max_noise_uv: float = 60.0
    # default: wander is removed by the per-beat straight baseline, so only extreme
    # wander rejects a lead (99th percentile in the same sample was ~2200 uV).
    max_wander_uv: float = 2000.0


@dataclass(frozen=True)
class MorphologyConfig:
    # default: per-lead QRS correlation below this counts as a shape change.
    min_qrs_corr: float = 0.90
    # default: per-lead QRS peak-to-peak ratio outside this range counts as a change.
    pp_ratio_range: tuple[float, float] = (0.75, 1.33)
    # default: maximum lag searched when aligning two templates.
    max_align_ms: float = 20.0
    # default: frontal QRS axis change (degrees) that counts as an axis shift.
    axis_shift_deg: float = 20.0


@dataclass(frozen=True)
class EpisodeConfig:
    # Episode rule from the LTSTDB description (physionet.org/content/ltstdb/1.0.0):
    # an episode begins when |ST deviation| first exceeds 50 uV, must reach Vmin for at
    # least Tmin, and ends when the deviation falls below 50 uV and does not exceed
    # 50 uV in the following 30 s. Protocols: A (75 uV, 30 s), B (100 uV, 30 s),
    # C (100 uV, 60 s).
    onset_uv: float = 50.0
    end_hold_s: float = 30.0
    protocols: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"A": (75.0, 30.0), "B": (100.0, 30.0), "C": (100.0, 60.0)}
    )
    # default: an unusable stretch longer than this splits an episode.
    max_unusable_gap_s: float = 60.0


@dataclass(frozen=True)
class FeatureConfig:
    # default: non-overlapping analysis windows (LTSTDB used 16 s moving averages).
    window_s: float = 15.0
    # default: signal read in chunks of this length when scanning a whole record.
    chunk_s: float = 600.0


@dataclass(frozen=True)
class Config:
    filter: FilterConfig = field(default_factory=FilterConfig)
    beats: BeatSelectionConfig = field(default_factory=BeatSelectionConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    fiducial: FiducialConfig = field(default_factory=FiducialConfig)
    st: STConfig = field(default_factory=STConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = Config()
