"""Match-acceptance presets (strict / default / relaxed) for the RMS-distance score.

A track's rank-1 flight is accepted as its match when it passes three cuts:

- **Score:** the rank-1 track score is at most ``k * SIGMA_S_KM``.
- **Ambiguity gap:** the rank-2 score exceeds the rank-1 score by at least
  ``g * SIGMA_S_KM``. Tracks with a single candidate pass.
- **Age:** ``TTFD_FLOOR_MIN < mean_age <= TTFD_FLOOR_MIN + k * tau0``, where ``mean_age`` is
  the mean age of the rank-1 flight's points along the contrail in the track's first frame.

``SIGMA_S_KM`` (2.13 km) is the floor of the 3-frame weighted-mean track score and ``tau0``
the time for ERA5 wind error to grow the advection error by one floor. Below
``TTFD_FLOOR_MIN`` matches are dominated by neighbouring flights. ``k`` sets the score and
age tolerance and ``g`` the ambiguity gap:

    ========  ===  ===  ===========
    preset     k    g   tau0 (min)
    ========  ===  ===  ===========
    strict     1    3   10.48
    default    2    2   11.56
    relaxed    3    1   13.85
    ========  ===  ===  ===========

The constants were measured on 2025 GOES-East data (Contrails.org detections, tracks of at
least 3 frames) and assume the default ``MatchingConfig`` track scoring with the
``rms_distance`` scorer. Other satellites or configurations may need their own.

Usage:
    matched = match_tracks(matcher.matching_results, preset="default")

``track_match_table`` and ``matched_mask`` expose the two steps, e.g. to compare presets
on one table.
"""

import numpy as np
import pandas as pd

PRESET_NAMES = ("relaxed", "default", "strict")

SIGMA_S_KM = 2.13
TTFD_FLOOR_MIN = 5.5

# Score and gap in km, the upper age bound in minutes. Rounded from the unrounded fit. The
# names TTFD_FLOOR_MIN and ttfd_cut are kept for the window's edges, which bound the mean age.
MATCH_PRESETS: dict[str, dict[str, float]] = {
    "relaxed": dict(score_cut=6.39, gap_cut=2.13, ttfd_cut=47.1),
    "default": dict(score_cut=4.26, gap_cut=4.26, ttfd_cut=28.6),
    "strict": dict(score_cut=2.13, gap_cut=6.39, ttfd_cut=16.0),
}


def track_match_table(matching_results: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``Matcher.matching_results`` to one row per track.

    Args:
        matching_results: ``Matcher.matching_results`` from tracked matching, with
            ``t1_rms_distance``, ``t2_rms_distance``, ``t1_first_formation_point_time``
            and ``t1_mean_formation_point_age_seconds``.

    Returns:
        DataFrame indexed by ``track_id`` with columns ``t1`` and ``t2`` (rank-1 and
        rank-2 track scores, ``t2`` NaN for a single candidate), ``first_detection``,
        ``ttfd_min`` (minutes from the rank-1 flight's first formation point to the
        track's first detection) and ``mean_age_min`` (the age the presets cut on, in
        minutes).
    """
    grouped = matching_results.groupby("track_id")
    track = pd.DataFrame(
        {
            "t1": grouped["t1_rms_distance"].min(),
            "t2": grouped["t2_rms_distance"].min(),
            "first_detection": grouped["timestamp"].min(),
            "first_formation": grouped["t1_first_formation_point_time"].min(),
            "mean_age_min": grouped["t1_mean_formation_point_age_seconds"].min() / 60.0,
        }
    )
    track["ttfd_min"] = (
        pd.to_datetime(track["first_detection"]) - pd.to_datetime(track["first_formation"])
    ).dt.total_seconds() / 60.0
    return track.drop(columns="first_formation")


def matched_mask(track: pd.DataFrame, preset: str) -> pd.Series:
    """Return a boolean per-track mask for a preset.

    Args:
        track: Per-track DataFrame from ``track_match_table``.
        preset: One of ``"relaxed"``, ``"default"`` or ``"strict"``.

    Returns:
        Boolean Series aligned to ``track``, True where the track passes the score,
        ambiguity-gap and age cuts. Tracks with no rank-1 score are False.

    Raises:
        ValueError: If ``preset`` is not one of ``PRESET_NAMES``.
    """
    if preset not in MATCH_PRESETS:
        raise ValueError(f"preset must be one of {PRESET_NAMES}, got {preset!r}")
    c = MATCH_PRESETS[preset]
    gap = (track["t2"] - track["t1"]).fillna(np.inf)
    age = track["mean_age_min"]
    return (
        (track["t1"] <= c["score_cut"])
        & (gap >= c["gap_cut"])
        & (age > TTFD_FLOOR_MIN)
        & (age <= c["ttfd_cut"])
    )


def match_tracks(matching_results: pd.DataFrame, preset: str = "default") -> pd.DataFrame:
    """Return the tracks whose rank-1 flight passes a preset's cuts.

    Args:
        matching_results: ``Matcher.matching_results`` from tracked matching.
        preset: One of ``"relaxed"``, ``"default"`` or ``"strict"``.

    Returns:
        The rows of ``track_match_table(matching_results)`` that pass the preset.

    Raises:
        ValueError: If ``preset`` is not one of ``PRESET_NAMES``.
    """
    track = track_match_table(matching_results)
    return track[matched_mask(track, preset)]
