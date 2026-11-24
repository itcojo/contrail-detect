"""
Interaction with and manipulation of ECMWF meteorological data.

"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.physics import units

from contrail_detect.core.config import DATA_DIR

# Pressure levels (hPa) downloaded by default, about 13.5 km down to the surface
DEFAULT_PRESSURE_LEVELS = [
    150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]  # fmt: skip


def _check_pressure_levels(pressure_levels: list[int]) -> None:
    """Raise ValueError unless every level is a whole number of hPa from 100 to 1000."""
    if not all(isinstance(p, (int, np.integer)) for p in pressure_levels):
        raise ValueError("All pressure levels must be integers (hPa)")
    if not all(100 <= p <= 1000 for p in pressure_levels):
        raise ValueError("Pressure levels must be between 100 and 1000 hPa")


def _read_cdsapirc(path: Path) -> tuple[str | None, str | None]:
    """Read the CDS API url and key from a ``.cdsapirc`` file.

    Args:
        path: Path to the ``.cdsapirc`` file.

    Returns:
        Tuple of (url, key), each None if the file does not set it.

    Raises:
        ValueError: If the file does not exist.
    """
    if not path.exists():
        raise ValueError(
            f"ECMWF API key not found at {path}. "
            f"Please set up your API key to retrieve data from ECMWF."
        )
    entries = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
    url = entries.get("url", "").strip() or None
    key = entries.get("key", "").strip() or None
    return url, key


def _download_era5(
    start_time: np.datetime64 | pd.Timestamp | str,
    end_time: np.datetime64 | pd.Timestamp | str,
    variables: list[str],
    pressure_levels: list[int] | None = None,
    path_to_key: str | Path | None = None,
) -> xr.Dataset:
    """Download ERA5 data from the Climate Data Store with the pycontrails ERA5 wrapper.

    To see available variables, see ``pycontrails.models.cocip.Cocip.met_variables``.

    NOTE: Requires CDS API credentials: a `.cdsapirc` file in the home directory or the
    ``path_to_key`` directory, or the ``CDSAPI_URL`` and ``CDSAPI_KEY`` environment variables.

    Args:
        start_time: Start time to retrieve. Must be a whole integer hour.
        end_time: End time to retrieve. Must be a whole integer hour.
        variables: Atmospheric variables to retrieve (e.g., ["temperature", "specific_humidity"]).
        pressure_levels: Pressure levels (hPa) to retrieve. Defaults to
            ``DEFAULT_PRESSURE_LEVELS``.
        path_to_key: Optional directory containing a .cdsapirc file to read credentials
            from. If None, credentials come from the ``CDSAPI_URL`` and ``CDSAPI_KEY``
            environment variables or ~/.cdsapirc.

    Returns:
        xr.Dataset containing the requested meteorological data.
    """
    # Credentials from path_to_key are passed on; otherwise pycontrails reads CDSAPI_URL and
    # CDSAPI_KEY, and cdsapi falls back to ~/.cdsapirc
    url = key = None
    if path_to_key is not None:
        url, key = _read_cdsapirc(Path(path_to_key) / ".cdsapirc")
    elif not (os.getenv("CDSAPI_URL") and os.getenv("CDSAPI_KEY")):
        _read_cdsapirc(Path.home() / ".cdsapirc")

    # Timestamp conversion (round up/down to nearest hour)
    start_ts = pd.Timestamp(start_time).floor("h")
    end_ts = pd.Timestamp(end_time).ceil("h")

    # Ensure times are whole integer hours
    for name, ts, raw in [("start_time", start_ts, start_time), ("end_time", end_ts, end_time)]:
        for attr in ("minute", "second", "microsecond", "nanosecond"):
            if getattr(ts, attr) != 0:
                raise ValueError(f"{name} must be a whole integer hour, got {raw}")

    if pressure_levels is None:
        pressure_levels = list(DEFAULT_PRESSURE_LEVELS)
    _check_pressure_levels(pressure_levels)

    logging.info(
        "Retrieving meteorological data from ECMWF servers...\nNOTE: this may "
        "take some time due to waiting in the queue and download size..."
    )

    era5 = ERA5(
        time=(start_ts, end_ts),
        variables=variables,
        pressure_levels=pressure_levels,
        url=url,
        key=key,
    )
    return era5.open_metdataset()


def get_era5_from_local(
    times: list[np.datetime64],
    met_variables: list[str],
    convert_pressure_levels: bool = True,
    file_path_met: str | Path | None = None,
    path_to_key: str | Path | None = None,
    pressure_levels: list[int] | None = None,
) -> xr.Dataset:
    """Load ECMWF ERA5 HRES meteorological data over a time period.

    Downloads data if not already present locally, one file per UTC day. A saved day that
    lacks a requested variable or pressure level is downloaded again with both its own and
    the requested ones. Every day from the hour at or before the earliest time to the hour at
    or after the latest time is loaded, so a time after 23:00 also loads the next day (its
    winds are needed to interpolate up to midnight).

    To see available variables, see ``pycontrails.models.cocip.Cocip.met_variables``.

    Args:
        times: Times the data must cover. Only the earliest and latest matter.
        met_variables: List of meteorological variables to retrieve.
        convert_pressure_levels: Whether to convert pressure levels to altitude in feet
            and set as coordinate. Defaults to True.
        file_path_met: Path to data storage location. Defaults to "data/ecmwf_data/era5"
            within the package directory.
        path_to_key: Optional directory containing a .cdsapirc file to read credentials
            from. If None, credentials come from the ``CDSAPI_URL`` and ``CDSAPI_KEY``
            environment variables or ~/.cdsapirc.
        pressure_levels: Pressure levels (hPa) to load, each a whole number from 100 to
            1000. Defaults to ``DEFAULT_PRESSURE_LEVELS`` (150 to 1000 hPa).

    Returns:
        Dataset containing ERA5 meteorological data at the requested pressure levels.
    """
    levels = sorted(DEFAULT_PRESSURE_LEVELS if pressure_levels is None else pressure_levels)
    _check_pressure_levels(levels)

    # Ensure download location exists
    if file_path_met is None:
        file_path_met = DATA_DIR / "ecmwf_data" / "era5"
    else:
        file_path_met = Path(file_path_met)

    if not file_path_met.exists():
        file_path_met.mkdir(parents=True)

    # Every day from the hour at or before the earliest time to the hour at or after the
    # latest, so that interpolation is bracketed on both sides
    times = pd.DatetimeIndex([pd.Timestamp(t) for t in np.atleast_1d(times)])
    first_day = times.min().floor("h").normalize()
    last_day = times.max().ceil("h").normalize()
    era5_days = list(pd.date_range(first_day, last_day, freq="D"))

    # Ensure ERA5 data is downloaded
    netcdf_paths = []
    for era5_day in era5_days:
        dt_str = f"ERA5-{era5_day.year:04d}-{era5_day.month:02d}-{era5_day.day:02d}.netcdf"
        day_path = file_path_met / dt_str
        netcdf_paths.append(day_path)

        variables, day_levels = list(met_variables), levels
        if day_path.exists():
            with xr.open_dataset(day_path) as cached:
                cached_vars = list(cached.data_vars)
                cached_levels = [int(p) for p in cached["level"].values]
            missing = [v for v in met_variables if v not in cached_vars]
            missing_levels = [p for p in levels if p not in cached_levels]
            if not missing and not missing_levels:
                logging.info(
                    f"ECMWF ERA5 data for {era5_day} already exists locally, skipping download."
                )
                continue
            # Re-download with the cached variables and levels too, so the day stays in one file
            logging.info(
                f"ECMWF ERA5 file for {era5_day} lacks {missing + missing_levels}; "
                "downloading again."
            )
            variables = cached_vars + missing
            day_levels = sorted(set(cached_levels) | set(levels))

        # Make request
        logging.info(f"Downloading ECMWF ERA5 data for {era5_day}...")
        met_dataset = _download_era5(
            start_time=era5_day,
            end_time=era5_day + pd.Timedelta(days=1, hours=-1),
            variables=variables,
            pressure_levels=day_levels,
            path_to_key=path_to_key,
        )
        # Write to a temporary name first, so an interrupted download is never mistaken
        # for a complete day on the next run
        part_path = day_path.with_name(day_path.name + ".part")
        met_dataset.data.to_netcdf(part_path)
        os.replace(part_path, day_path)

    # Open dataset
    met = xr.open_mfdataset(netcdf_paths).sel(level=levels)

    # Convert pressure levels to altitude in feet and set as coordinate if requested
    if convert_pressure_levels:
        met = met.assign_coords(altitude_ft=units.pl_to_ft(met.coords["level"]))
        met = met.set_index(level="altitude_ft")
        met = met.rename({"level": "altitude_ft"})

    return met


def initialize_tracking_windspeeds(
    df_contrails: pd.DataFrame,
    estimated_alt_ft: float | np.ndarray = 3.5e4,
    path_to_key: str | Path | None = None,
) -> pd.DataFrame:
    """Initialize tracking wind speeds for contrail detections.

    Estimates initial tracking wind speeds for contrail detections based on ECMWF ERA5 HRES
    data at a specified altitude. This can be used as an initial guess for tracking
    algorithms.

    Args:
        df_contrails: DataFrame containing contrail detections with ``'centroid_lon'``,
            ``'centroid_lat'`` and ``'centroid_scantime'`` columns.
        estimated_alt_ft: Estimated altitude of contrails in feet, defaults to 35,000 ft.
        path_to_key: Optional directory containing a .cdsapirc file to read credentials
            from. If None, credentials come from the ``CDSAPI_URL`` and ``CDSAPI_KEY``
            environment variables or ~/.cdsapirc.

    Returns:
        DataFrame with additional columns 'u_init' and 'v_init' for initial wind speed
        estimates. Detections the wind data cannot cover get NaN, with a warning.
    """
    df_local = df_contrails.copy()

    # Winds are looked up at each detection's scan time, so cover the scan times, not the
    # nominal frame times: a scan after 23:00 on the last day needs the next day's winds
    scan_times = pd.to_datetime(df_contrails["centroid_scantime"])
    met = get_era5_from_local(
        times=[scan_times.min(), scan_times.max()],
        met_variables=["eastward_wind", "northward_wind"],
        convert_pressure_levels=False,
        path_to_key=path_to_key,
    )

    # Set up interpolation coordinates based on contrail detection locations and times
    if isinstance(estimated_alt_ft, (int, float)):
        estimated_alt_ft = np.array([estimated_alt_ft] * len(df_contrails))
    elif isinstance(estimated_alt_ft, np.ndarray):
        if len(estimated_alt_ft) != len(df_contrails):
            raise ValueError(
                "Length of estimated_alt_ft array must match number of contrail detections."
            )

    interp_coords = {
        "longitude": xr.DataArray(df_contrails["centroid_lon"], dims="points"),
        "latitude": xr.DataArray(df_contrails["centroid_lat"], dims="points"),
        "level": xr.DataArray(units.ft_to_pl(estimated_alt_ft), dims="points"),
        "time": xr.DataArray(df_contrails["centroid_scantime"], dims="points"),
    }

    # Interpolate u and v wind components at the contrail detection points
    met = met.rename_vars({"eastward_wind": "u", "northward_wind": "v"})
    interpolated_wind = met[["u", "v"]].interp(
        **interp_coords,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    # Assign interpolated wind speeds to DataFrame
    df_local["u_init"] = interpolated_wind["u"].values
    df_local["v_init"] = interpolated_wind["v"].values

    n_missing = int((df_local["u_init"].isna() | df_local["v_init"].isna()).sum())
    if n_missing:
        logging.warning(
            f"No ERA5 wind for {n_missing} of {len(df_local)} contrail detections (outside "
            "the wind data); their u_init and v_init are NaN."
        )

    return df_local


def _bracketing_slice(coord: np.ndarray, values: np.ndarray) -> slice:
    """Smallest index slice of a monotonic ``coord`` whose cells bracket every value.

    Linear interpolation only reads the grid points either side of each value, so
    interpolating on this slice gives the same result as on the whole coordinate.
    """
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    below, above = coord[coord <= vmin], coord[coord >= vmax]
    lo = below.max() if below.size else coord.min()
    hi = above.min() if above.size else coord.max()
    idx = np.flatnonzero((coord >= lo) & (coord <= hi))
    return slice(idx[0], idx[-1] + 1)


def interpolate_era5_to_points(
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    pressure_hPa: np.ndarray,
    times: pd.DatetimeIndex,
    level_dim: str = "level",
    max_time_gap_hours: int = 1,
    time_block_hours: int = 24,
) -> pd.DataFrame:
    """Interpolate ERA5 dataset variables to a set of 4D coordinates (lat, lon, pressure, time).

    Performs 4D linear interpolation across all dimensions. To minimize I/O overhead
    when reading from remote stores (e.g., cloud storage), data is loaded in time-blocked
    batches rather than querying hourly slices individually.

    Each block loads only the hours, latitudes, longitudes and levels that bracket its
    points, so memory scales with the area the points cover rather than the dataset.

    Args:
        ds: ERA5 dataset with dimensions (time, level_dim, latitude, longitude).
            The time dimension must be sorted in ascending order.
        lats: Array of target latitudes.
        lons: Array of target longitudes.
        pressure_hPa: Array of target pressure levels (hPa).
        times: Array of target timestamps.
        level_dim: Name of the vertical coordinate dimension in ds. Defaults to "level".
        max_time_gap_hours: Maximum allowed gap (in hours) between the two ERA5 time
            steps bracketing a point. Points exceeding this gap will return NaN.
        time_block_hours: Number of ERA5 hours to load into memory per batch.

    Returns:
        pd.DataFrame: Dataframe containing the interpolated values, with one row per
            input point and one column per variable in ds.
    """
    n = len(lats)
    times = pd.DatetimeIndex(times)
    era5_times = pd.DatetimeIndex(ds["time"].values)

    # Initialize results with NaN to handle out-of-bounds or missing data points
    result = {var: np.full(n, np.nan) for var in ds.data_vars}

    # Find the lower-bounding ERA5 time index for each target point. A point exactly on
    # the last ERA5 hour is bracketed by the last two hours.
    lo = era5_times.searchsorted(times, side="right") - 1
    if len(era5_times) > 1:
        lo = np.where(np.asarray(times == era5_times[-1]), len(era5_times) - 2, lo)
    in_range = (lo >= 0) & (lo < len(era5_times) - 1)

    # Flag points that fall into data gaps exceeding max_time_gap_hours
    max_gap = pd.Timedelta(hours=max_time_gap_hours)
    valid = in_range.copy()
    lo_in = lo[in_range]
    valid[in_range] = (era5_times[lo_in + 1] - era5_times[lo_in]) <= max_gap

    if not valid.any():
        logging.warning("No valid points found within the ERA5 time range.")
        return pd.DataFrame(result)

    # Group points into time blocks to optimize chunk loading
    point_idx = np.arange(n)
    block = lo // time_block_hours

    valid_count = int(valid.sum())
    unique_blocks = len(np.unique(block[valid]))
    logging.info(f"Interpolating {valid_count}/{n} points across {unique_blocks} time blocks.")

    # Prepare DataArrays for interpolation
    da_lat = xr.DataArray(lats, dims="points")
    da_lon = xr.DataArray(lons, dims="points")
    da_time = xr.DataArray(times, dims="points")
    da_level = xr.DataArray(pressure_hPa, dims="points")

    for block_num, members in pd.Series(point_idx[valid]).groupby(block[valid]):
        sel = members.to_numpy()
        lo_sel = lo[sel]
        t0, t1 = int(lo_sel.min()), int(lo_sel.max()) + 1

        logging.info(
            f"Processing block {block_num}: loading time indices {t0} to {t1} ({len(sel)} points)."
        )

        # ``t1`` is the upper bracketing hour for the latest point in the block, so the
        # end-exclusive slice needs ``t1 + 1``.
        crop = {
            dim: _bracketing_slice(ds[dim].values, values[sel])
            for dim, values in (
                ("latitude", lats),
                ("longitude", lons),
                (level_dim, pressure_hPa),
            )
        }
        slab = ds.isel(time=slice(t0, t1 + 1), **crop).load()

        # Vectorized multi-dimensional interpolation using Xarray DataArrays
        interp = slab.interp(
            latitude=da_lat[sel],
            longitude=da_lon[sel],
            time=da_time[sel],
            **{level_dim: da_level[sel]},
            method="linear",
            kwargs={"fill_value": np.nan, "bounds_error": False},
        )

        for var in ds.data_vars:
            result[var][sel] = interp[var].values

    return pd.DataFrame(result)
