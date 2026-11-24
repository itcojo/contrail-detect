"""
Flight-contrail matching and correlation analysis.

This module provides tools for matching detected contrails with flight trajectories,
enabling analysis of which aircraft produced which contrails and supporting
contrail formation studies.
"""

from contrail_detect.matching.matching import Matcher, MatchingConfig
from contrail_detect.matching.presets import (
    MATCH_PRESETS,
    PRESET_NAMES,
    match_tracks,
    matched_mask,
    track_match_table,
)

__all__ = [
    "MATCH_PRESETS",
    "PRESET_NAMES",
    "Matcher",
    "MatchingConfig",
    "match_tracks",
    "matched_mask",
    "track_match_table",
]
