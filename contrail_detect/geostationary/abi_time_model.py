"""Per-pixel scan times for GOES ABI, from NOAA STAR's ABI time model.

In Mode 6, ABI scans the Full Disk as 22 west-to-east swaths, north to south, in between the
CONUS swaths and the Mesoscale scans, so time doesn't increase evenly down the image. The
time-model tables of Carr et al. (2020) give each 2 km pixel's time after the scene's first
Band 2 sample, so a pixel's time after the nominal frame time is

    (time_bounds[0] - nominal) + table value + band correction

The tables are from the Supplementary Materials of Carr et al. (2020), Remote Sens. 12, 3779,
https://doi.org/10.3390/rs12223779 (ABI_Time_Model_09042020, identical to NOAA STAR's
ABI-Time_Model_LUTS.tar.gz). We bundle its Mode 6A table,
ABI-Timeline05B_Mode 6A_20190612-183017.nc, transposed to (row, col).

Carr et al. give the accuracy as about 1 s, or 30 s for a pixel near a swath boundary that
falls in the wrong swath. This release puts the boundaries 12 rows north of the swath overlap
midlines.

Files only record timeline_id "ABI Mode 6", not whether 6A or 6M ran, so the table is chosen by
satellite. NOAA says GOES-16 uses 6A and GOES-17 6M, and says nothing about GOES-18 or GOES-19.
We use 6A for both: timed from the 6A table, their Full Disk, CONUS and Mesoscale swaths fit
together without overlapping (checked on 2023-06-01 and 2026-11-16), while with 6M, 48 overlap.

Only Mode 6 is supported.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

TABLE_PATH = (
    Path(__file__).parent.parent / "data" / "geostationary_data" / "abi_time_model_mode6a.npz"
)

SUPPORTED_TIMELINE_ID = "ABI Mode 6"

TIMELINE_BY_SATELLITE: dict[str, str] = {
    "GOES-16": "6A",
    "GOES-18": "6A",
    "GOES-19": "6A",
}

# CONUS1 and CONUS2 are identical in the 6A source, so only one CONUS table is kept
SCENE_BY_REGION: dict[str, str] = {"F": "FD", "C": "CONUS", "M1": "MESO", "M2": "MESO"}

# In-scan detector offset of each band, in seconds (Carr et al. 2020, Table A1)
BAND_OFFSETS_S: dict[int, float] = {
    1: 0.179,
    2: -0.055,
    3: 0.402,
    4: 0.642,
    5: -0.359,
    6: -0.642,
    7: 0.535,
    8: 0.267,
    9: 0.000,
    10: -0.267,
    11: -0.535,
    12: -0.542,
    13: 0.551,
    14: 0.319,
    15: -0.256,
    16: 0.579,
}


def band_correction_s(bands: list[int] | tuple[int, ...]) -> float:
    """Mean in-scan offset of the bands, relative to Band 2 (the tables' reference).

    Args:
        bands: ABI band numbers, 1-16.

    Returns:
        Seconds to add to the table time. -0.102 s for C11, C14 and C15.
    """
    if not bands:
        raise ValueError("bands must not be empty")
    unknown = [b for b in bands if b not in BAND_OFFSETS_S]
    if unknown:
        raise ValueError(f"Unknown ABI band(s) {unknown}; expected 1-16")
    return float(np.mean([BAND_OFFSETS_S[b] for b in bands]) - BAND_OFFSETS_S[2])


def check_timeline(timeline_id: str) -> None:
    """Raise unless the file's timeline_id is Mode 6."""
    if timeline_id.strip() != SUPPORTED_TIMELINE_ID:
        raise ValueError(
            f"No ABI time model for timeline_id {timeline_id!r}, only {SUPPORTED_TIMELINE_ID!r}"
        )


@cache
def load_table(scene: str) -> np.ndarray:
    """Load a bundled Mode 6A table.

    Args:
        scene: ``"FD"``, ``"CONUS"`` or ``"MESO"``.

    Returns:
        Read-only float32 array of seconds after the scene's first Band 2 sample, with rows
        north to south and columns west to east.
    """
    if scene not in ("FD", "CONUS", "MESO"):
        raise ValueError(f"Unknown scene {scene!r}; expected 'FD', 'CONUS' or 'MESO'")
    with np.load(TABLE_PATH) as z:
        table = z[scene]
    table.setflags(write=False)
    return table


def pixel_time_offsets_s(
    satellite: str,
    region: str,
    scan_start_s: float,
    bands: list[int] | tuple[int, ...],
    timeline_id: str | None = None,
) -> np.ndarray:
    """Per-pixel scan time of a GOES ABI scene, in seconds after its nominal time.

    Args:
        satellite: ``"GOES-16"``, ``"GOES-18"`` or ``"GOES-19"``.
        region: ``"F"``, ``"C"``, ``"M1"`` or ``"M2"``.
        scan_start_s: ``time_bounds[0]`` of the file minus the nominal time, in seconds.
        bands: Bands the product is made from.
        timeline_id: The file's ``timeline_id``, checked if given.

    Returns:
        float64 array of seconds after the nominal time, (row, col).
    """
    if satellite not in TIMELINE_BY_SATELLITE:
        raise ValueError(
            f"No ABI time model for {satellite!r}; supported: {sorted(TIMELINE_BY_SATELLITE)}"
        )
    if region not in SCENE_BY_REGION:
        raise ValueError(f"Unknown GOES region {region!r}; expected {sorted(SCENE_BY_REGION)}")
    if timeline_id is not None:
        check_timeline(timeline_id)
    table = load_table(SCENE_BY_REGION[region])
    return table.astype(np.float64) + (scan_start_s + band_correction_s(bands))
