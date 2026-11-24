"""Utility functions for flight data processing."""

import numpy as np
import pandas as pd
import xarray as xr


def intersection_with_ellipsoid(
    p0: np.ndarray,
    p1: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Find intersection of line with Earth's ellipsoid surface.

    Calculates where the line from satellite (p0) through aircraft position (p1)
    intersects with the WGS84 ellipsoid surface. Used for parallax correction
    of advected flight paths.

    Args:
        p0: Satellite position in ECEF coordinates.
        p1: Aircraft position in ECEF coordinates.
        a: Semi-major axis of the ellipsoid (meters).
        b: Semi-minor axis of the ellipsoid (meters).

    Returns:
        Array of intersection points in ECEF coordinates. Points where aircraft
        is not visible (Earth blocks view) are filled with NaN.
    """
    px, py, pz = p0
    v = p1 - p0
    vx, vy, vz = v

    A = vx**2 / a**2 + vy**2 / a**2 + vz**2 / b**2
    B = 2 * (px * vx / a**2 + py * vy / a**2 + pz * vz / b**2)
    C = px**2 / a**2 + py**2 / a**2 + pz**2 / b**2 - 1.0

    D = B**2 - 4 * A * C
    sqrtD = np.sqrt(D, where=D >= 0, out=np.full_like(D, np.nan))

    t0 = (-B + sqrtD) / (2.0 * A)
    t1 = (-B - sqrtD) / (2.0 * A)

    intersection0 = p0 + t0 * v
    intersection1 = p0 + t1 * v

    d0 = np.linalg.norm(intersection0 - p1, axis=0)
    d1 = np.linalg.norm(intersection1 - p1, axis=0)
    out = np.where(d0 < d1, intersection0, intersection1)

    # t0 < 1 means the Earth is between the satellite and the aircraft
    opposite_side = t0 < 1.0
    out[:, opposite_side] = np.nan

    return out


def merge_dataframes_with_index(
    list_of_dfs: list[pd.DataFrame],
    advection_step: int = 1,
) -> pd.DataFrame:
    """Merge list of DataFrames with frame index tracking.

    Concatenates multiple DataFrames while adding metadata columns to track
    the original frame index and advection timestep.

    Args:
        list_of_dfs: List of DataFrames to merge.
        advection_step: Advection timestep in minutes. Defaults to 1.

    Returns:
        Single merged DataFrame with ``'adv_frame'`` and ``'advection_step'`` columns.
    """
    merged_dfs = []
    for i, df in enumerate(list_of_dfs):
        if df.empty:
            continue
        df_copy = df.copy()
        df_copy["adv_frame"] = i
        df_copy["advection_step"] = advection_step
        merged_dfs.append(df_copy)

    if not merged_dfs:
        # Every frame was empty (e.g. no flight advected into any scan window), so
        # ``pd.concat([])`` would raise "No objects to concatenate". Return an empty
        # frame that still carries the metadata columns so callers can treat it as
        # "zero advected waypoints" rather than crash.
        template = list_of_dfs[0].iloc[0:0].copy() if list_of_dfs else pd.DataFrame()
        template["adv_frame"] = pd.Series([], dtype="int64")
        template["advection_step"] = pd.Series([], dtype="int64")
        return template

    return pd.concat(merged_dfs, ignore_index=True)


def validate_and_parse_frame_timestamps(
    frame_timestamps: list[np.datetime64],
    require_multiple: bool = False,
) -> tuple:
    """Sort, deduplicate, and validate frame timestamps.

    Args:
        frame_timestamps: Input sequence of frame times.
        require_multiple: If True, raise if fewer than two unique timestamps provided.

    Returns:
        Tuple of (frame_timestamps, frame_interval) where frame_timestamps is a
        sorted numpy array and frame_interval is the uniform spacing (or None if
        only one timestamp).

    Raises:
        ValueError: If timestamps cannot be parsed, are not uniformly spaced,
            or require_multiple is True and only one timestamp is given.
    """
    try:
        frame_timestamps = np.array(sorted(set(frame_timestamps)))
    except Exception as e:
        raise ValueError("frame_timestamps must be convertible to a numpy array.") from e

    if len(frame_timestamps) < 2:
        if require_multiple:
            raise ValueError("At least two frame timestamps are required.")
        return frame_timestamps, None

    frame_intervals = np.diff(frame_timestamps)
    if not np.all(frame_intervals == frame_intervals[0]):
        raise ValueError(
            "Frame times must be uniformly spaced, currently have "
            f"intervals: {np.unique(frame_intervals)}"
        )
    return frame_timestamps, frame_intervals[0]


def resolve_adv_timestep(
    adv_timestep: int | np.timedelta64 | None,
    frame_interval: np.timedelta64 | None,
) -> np.timedelta64:
    """Resolve and validate the advection timestep.

    Args:
        adv_timestep: Caller-supplied timestep (int minutes, timedelta64, or None).
        frame_interval: Uniform interval between frame times, or None for a single frame.

    Returns:
        Advection timestep as np.timedelta64.

    Raises:
        ValueError: If adv_timestep is None with no frame_interval, or if it does
            not evenly divide the frame interval.
    """
    if adv_timestep is None:
        if frame_interval is None:
            raise ValueError(
                "adv_timestep must be provided when only a single frame timestamp is given."
            )
        adv_timestep = frame_interval
    elif not isinstance(adv_timestep, np.timedelta64):
        adv_timestep = np.timedelta64(int(adv_timestep), "m")

    if frame_interval is not None:
        if np.timedelta64(frame_interval, "ns") % np.timedelta64(
            adv_timestep, "ns"
        ) != np.timedelta64(0, "ns"):
            raise ValueError("Advection timestep must divide frame time interval.")

    return adv_timestep


def validate_flight_ecmwf_domains(
    ecmwf_data: xr.Dataset,
    flight_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that flight data lies within ECMWF data domain.

    Args:
        ecmwf_data: Loaded ECMWF dataset with coordinate bounds.
        flight_data: Flight data to validate against ECMWF bounds.

    Returns:
        DataFrame showing domain limits for ECMWF and flight data.

    Raises:
        ValueError: If ecmwf_data is None or flight_data is empty.
    """
    if ecmwf_data is None:
        raise ValueError("ECMWF data is not loaded.")
    if flight_data.empty:
        raise ValueError("Flight data is not loaded.")

    ecmwf_min_lon = float(np.min(ecmwf_data.coords["longitude"].values))
    ecmwf_max_lon = float(np.max(ecmwf_data.coords["longitude"].values))
    ecmwf_min_lat = float(np.min(ecmwf_data.coords["latitude"].values))
    ecmwf_max_lat = float(np.max(ecmwf_data.coords["latitude"].values))
    ecmwf_min_alt = float(np.min(ecmwf_data.coords["altitude_ft"].values))
    ecmwf_max_alt = float(np.max(ecmwf_data.coords["altitude_ft"].values))
    ecmwf_min_time = np.min(ecmwf_data.coords["time"].values)
    ecmwf_max_time = np.max(ecmwf_data.coords["time"].values)

    flight_min_lon = float(np.min(flight_data["longitude"].values))
    flight_max_lon = float(np.max(flight_data["longitude"].values))
    flight_min_lat = float(np.min(flight_data["latitude"].values))
    flight_max_lat = float(np.max(flight_data["latitude"].values))
    flight_min_alt = float(np.min(flight_data["altitude_ft"].values))
    flight_max_alt = float(np.max(flight_data["altitude_ft"].values))
    flight_min_time = np.min(flight_data["time"].values)
    flight_max_time = np.max(flight_data["time"].values)

    return pd.DataFrame(
        {
            "Parameter": [
                "Longitude (min)",
                "Longitude (max)",
                "Latitude (min)",
                "Latitude (max)",
                "Altitude (min) [ft]",
                "Altitude (max) [ft]",
                "Time (min)",
                "Time (max)",
            ],
            "ECMWF": [
                ecmwf_min_lon,
                ecmwf_max_lon,
                ecmwf_min_lat,
                ecmwf_max_lat,
                ecmwf_min_alt,
                ecmwf_max_alt,
                ecmwf_min_time,
                ecmwf_max_time,
            ],
            "Flight": [
                flight_min_lon,
                flight_max_lon,
                flight_min_lat,
                flight_max_lat,
                flight_min_alt,
                flight_max_alt,
                flight_min_time,
                flight_max_time,
            ],
        }
    )
