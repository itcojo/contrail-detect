"""
Base class for flight data from ADS-B sources.

"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from pycontrails import Flight
from pycontrails.physics import units

from contrail_detect.flights.flight_utils import (
    intersection_with_ellipsoid,
    merge_dataframes_with_index,
    resolve_adv_timestep,
    validate_and_parse_frame_timestamps,
)
from contrail_detect.geostationary.abc import Geostationary

# Bump when advection changes its results, so files saved by older code are not reused
_ADVECTION_CACHE_VERSION = 3

# Columns carried through advection. altitude_ft is geometric (GNSS) altitude, used for
# parallax; pressure_altitude_ft is barometric altitude, used to pick the ERA5 wind level,
# whose levels are pressure altitudes. Without a barometric source, altitude_ft serves both.
_CORE_COLS = [
    "flight_id",
    "original_time",
    "longitude",
    "latitude",
    "altitude_ft",
    "pressure_altitude_ft",
    "time",
]


def _wrap_longitude(lon: pd.Series) -> pd.Series:
    """Wrap longitudes into [-180, 180), so flights that cross the date line keep moving."""
    return (lon + 180.0) % 360.0 - 180.0


class FlightData(ABC):
    """Abstract base class for flight data from ADS-B sources.

    Provides common functionality for handling flight trajectory data from various
    ADS-B sources. Includes methods for data loading, meteorological data integration,
    flight advection, and coordinate transformations. Subclasses implement
    ``_normalize_columns`` and their own way of loading data: ``ContrailsAPIFlightData``
    downloads from the Contrails.org API, and ``GenericFlightData`` loads your own data.

    Attributes:
        adv_flight_data: DataFrame containing advected flight trajectories.
        adv_max: Maximum advection time window.
        ecmwf_data: ECMWF meteorological data for wind field calculations.
        flight_data: Raw flight trajectory data.
        req_timestamps: Array of requested timestamps, with lookback prepended.
        req_dates: Daily date values derived from req_timestamps.
        req_bbox: Requested bounding box [min_lon, max_lon, min_lat, max_lat].
        save_name: Readable prefix for the saved advected-flights file. The file name also
            carries a fingerprint of every input that changes the result.
        save_path: Path for saving/loading cached data.
    """

    def __init__(
        self: Self,
        req_timestamps: list[str | np.datetime64] | None = None,
        adv_max: int | np.timedelta64 | None = None,
        req_bbox: list[float] | None = None,
        add_bbox_buffer: float | None = None,
    ) -> None:
        """Initialize the FlightData class.

        Args:
            req_timestamps: Timestamps for which flight data is requested. Strings are
                parsed as ``numpy.datetime64``. If provided together with ``adv_max``,
                a lookback timestamp is prepended so that pre-observation waypoints are
                also available for advection.
            adv_max: Maximum look-back window for flight advection: a ``numpy.timedelta64``,
                or an integer (Python or numpy) number of hours.
            req_bbox: Bounding box ``[min_lon, max_lon, min_lat, max_lat]``.
            add_bbox_buffer: If set, expands ``req_bbox`` by this many degrees on all sides.
        """
        self.adv_flight_data: pd.DataFrame = pd.DataFrame()
        self.adv_max: np.timedelta64 | None = None
        self.ecmwf_data: xr.Dataset | None = None
        self.flight_data: pd.DataFrame = pd.DataFrame()
        self.req_timestamps: np.ndarray | None = None
        self.req_dates: list[np.datetime64] | None = None
        self.req_bbox: list[float] | None = None
        self.save_name: str | None = None
        self._adv_cache_name: str | None = None
        self.save_path: str | None = None

        if req_timestamps is not None:
            if len(req_timestamps) == 0:
                raise ValueError("req_timestamps must not be empty.")
            if not isinstance(req_timestamps[0], (str, np.datetime64)):
                raise ValueError(
                    f"req_timestamps must be a list of strings or numpy datetime64 objects. "
                    f"Got {type(req_timestamps[0])} instead."
                )
            if isinstance(req_timestamps[0], str):
                try:
                    req_timestamps = [np.datetime64(date) for date in req_timestamps]
                except Exception as e:
                    raise ValueError(f"Error converting string to datetime: {e}") from e

            # Reject date gaps: downloads and the domain both run min..max continuously,
            # so a gap would silently pull in un-requested days.
            unique_dates = np.unique(np.asarray(req_timestamps).astype("datetime64[D]"))
            if len(unique_dates) > 1 and np.any(np.diff(unique_dates) != np.timedelta64(1, "D")):
                raise ValueError(
                    "req_timestamps must cover consecutive calendar dates with no gaps. "
                    f"Got dates {[str(d) for d in unique_dates]}."
                )

            # Sort first so the lookback below attaches to the earliest timestamp,
            # not whatever the caller happened to pass first.
            req_timestamps = np.sort(np.asarray(req_timestamps))

            if adv_max is not None:
                # numpy integers too: added to a datetime64 they would otherwise be read in
                # the datetime's own unit (minutes or seconds), not hours
                # (np.timedelta64 is itself a numpy integer, so it is checked first)
                if isinstance(adv_max, np.timedelta64):
                    pass
                elif isinstance(adv_max, (int, np.integer)) and not isinstance(adv_max, bool):
                    adv_max = np.timedelta64(int(adv_max), "h")
                else:
                    raise TypeError(
                        "adv_max must be a numpy.timedelta64 or an integer number of hours, "
                        f"got {type(adv_max).__name__}."
                    )
                self.adv_max = adv_max
                req_timestamps = np.concatenate([[req_timestamps[0] - adv_max], req_timestamps])

            self.req_timestamps = np.asarray(req_timestamps)
            self.req_dates = [item.astype("datetime64[D]") for item in self.req_timestamps]

        self.req_bbox = req_bbox
        if add_bbox_buffer is not None:
            self.add_buffer_to_bbox(add_bbox_buffer)

    def __repr__(self: Self) -> str:
        """Return string representation of the flight data handler."""
        fields = ", ".join(f"{k}={v}" for k, v in self._repr_fields().items())
        return f"{type(self).__name__}({fields})"

    def _repr_fields(self: Self) -> dict:
        """Fields shown by ``__repr__``. Subclasses extend this to add their own."""
        fields = {}
        if self.req_timestamps is not None:
            fields["min_time"] = np.min(self.req_timestamps)
            fields["max_time"] = np.max(self.req_timestamps)
        fields["req_bbox"] = self.req_bbox
        fields["save_path"] = self.save_path
        return fields

    @abstractmethod
    def _normalize_columns(self: Self) -> None:
        """Normalize source-specific column names to the standard schema.

        Must rename/derive columns so that after this call self.flight_data contains at
        minimum: flight_id, time, latitude, longitude, altitude_ft.
        Called at the end of every download method.
        """
        pass

    def add_buffer_to_bbox(
        self: Self,
        buffer: float,
    ) -> None:
        """Expand bounding box by adding buffer on all sides.

        Args:
            buffer: Buffer distance to add to each side in degrees.
        """
        if self.req_bbox is None:
            logging.warning("No bounding box defined. Cannot add buffer.")
            return

        min_lon, max_lon, min_lat, max_lat = self.req_bbox
        self.req_bbox = [
            min_lon - buffer,
            max_lon + buffer,
            min_lat - buffer,
            max_lat + buffer,
        ]
        logging.info(f"Updated bounding box with buffer: {self.req_bbox}")

    def advect_flights_to_scan_times(
        self: Self,
        detector: Geostationary,
        adv_max_past: np.timedelta64,
        adv_timestep: int | np.timedelta64 | None = None,
        flight_resample_timestep: int | None = None,
        save_output: bool = False,
    ) -> None:
        """Advect flight trajectories to per-pixel satellite scan times.

        Projects historical flight positions forward in time using dry advection with
        ECMWF wind data, accounting for the per-pixel scan time offset of the satellite
        scanning pattern. Each flight waypoint is saved at the moment the satellite
        scanner passes over its advected position.

        Args:
            detector: Geostationary detector supplying frame timestamps and scan time
                offsets.
            adv_max_past: Maximum lookback time before each frame time from which to
                include flight waypoints, as a ``numpy.timedelta64``. Must be a whole number
                of ``adv_timestep`` steps, and no longer than the ``adv_max`` the flight data
                was requested with.
            adv_timestep: Time step for advection integration. If integer, interpreted
                as minutes. If None, defaults to the interval between frame times.
            flight_resample_timestep: If set, interpolate raw flight waypoints to this
                interval (in minutes) before advection using pycontrails
                ``Flight.resample_and_fill``. Useful when the raw ping interval is
                coarser than the desired spatial coverage of advected positions.
                Defaults to None (no resampling).
            save_output: Whether to save computed advected data to disk. Defaults to False.

        Raises:
            TypeError: If ``adv_max_past`` is not a ``numpy.timedelta64``.
            ValueError: If ``adv_max_past`` is not a whole number of advection steps, or is
                longer than the ``adv_max`` the flight data was requested with.
        """
        if not isinstance(adv_max_past, np.timedelta64):
            raise TypeError(
                "adv_max_past must be a numpy.timedelta64, e.g. np.timedelta64(60, 'm'); "
                f"got {type(adv_max_past).__name__}."
            )
        frame_timestamps, frame_interval = validate_and_parse_frame_timestamps(
            detector.timestamp_sequence, require_multiple=True
        )
        adv_timestep = resolve_adv_timestep(adv_timestep, frame_interval)
        if np.timedelta64(adv_max_past, "ns") % np.timedelta64(
            adv_timestep, "ns"
        ) != np.timedelta64(0, "ns"):
            # The pre-observation steps would otherwise end after the first frame, carrying
            # every earlier waypoint too far downwind
            raise ValueError(
                f"adv_max_past ({adv_max_past}) must be a whole number of advection steps "
                f"(adv_timestep = {adv_timestep})."
            )
        if self.adv_max is not None and adv_max_past > self.adv_max:
            raise ValueError(
                f"adv_max_past ({adv_max_past}) is longer than the adv_max ({self.adv_max}) "
                "the flight data was requested with, so the earliest flights were never "
                "downloaded. Request the flight data with a larger adv_max."
            )

        self._adv_cache_name = self._advection_cache_name(
            detector, adv_max_past, adv_timestep, flight_resample_timestep
        )
        loaded_data = self._load_adv_flights()
        if loaded_data is not None:
            self.adv_flight_data = loaded_data
            logging.info("Loaded existing advected flight data from disk.")
            return

        self._n_without_wind = 0

        # frame_timestamp references the start of each scan, so the upper bound
        # must extend to the end of the last frame's scan window
        flight_data = self._filter_and_prepare_flight_data(
            time_lower_bound=frame_timestamps[0] - adv_max_past,
            time_upper_bound=frame_timestamps[-1] + frame_interval,
            flight_resample_timestep=flight_resample_timestep,
        )
        # Split into pre-observation data (before first frame) and observation data
        pre_obs_mask = (flight_data["time"] >= frame_timestamps[0] - adv_max_past) & (
            flight_data["time"] < frame_timestamps[0]
        )
        flight_data_pre_obs = flight_data[pre_obs_mask].copy()
        flight_data_obs = flight_data[~pre_obs_mask].copy()

        # --- --- ---
        # Phase 1: advect pre-observation data forward to frame_timestamps[0]
        # so it can feed into the per-frame scan-time advection below.
        pre_obs_advection_timestamps = np.arange(
            frame_timestamps[0] - adv_max_past + adv_timestep,
            frame_timestamps[0] + adv_timestep,
            adv_timestep,
        )
        flight_data_pre_obs = self._bin_to_advection_times(
            flight_data_pre_obs, pre_obs_advection_timestamps
        )
        grouped_pre_obs = flight_data_pre_obs.groupby("advection_bin")
        available_bins_pre_obs = set(grouped_pre_obs.groups.keys())

        prev_snapshot = pd.DataFrame()
        for i, advection_timestamp in enumerate(pre_obs_advection_timestamps):
            if i in available_bins_pre_obs:
                new_data = grouped_pre_obs.get_group(i).copy()
            else:
                new_data = flight_data_pre_obs.iloc[:0].copy()

            logging.info(f"Pre-obs step {advection_timestamp}: {len(new_data)} new points")

            if not prev_snapshot.empty:
                prev_snapshot["time_to_next_bin"] = adv_timestep
            snapshot = pd.concat([prev_snapshot, new_data], ignore_index=True)
            if snapshot.empty:
                continue

            # Each point moves with the wind at its own time and place at the start of its
            # move, as in the per-frame steps below
            snapshot = self._add_winds(snapshot, snapshot["time"].to_numpy())
            time_to_bin_s = snapshot["time_to_next_bin"] / np.timedelta64(1, "s")
            snapshot["longitude"] = _wrap_longitude(
                snapshot["longitude"]
                + units.m_to_longitude_distance(
                    time_to_bin_s * snapshot["u_wind"], snapshot["latitude"].values
                )
            )
            snapshot["latitude"] += units.m_to_latitude_distance(time_to_bin_s * snapshot["v_wind"])
            snapshot["total_advected_time"] += snapshot["time_to_next_bin"]
            snapshot["time"] = advection_timestamp
            # Discard points that have already exceeded the lookback budget
            snapshot = snapshot[snapshot["total_advected_time"] <= adv_max_past].copy()
            prev_snapshot = snapshot.copy()

        # prev_snapshot now holds all points advected to frame_timestamps[0]
        if not prev_snapshot.empty:
            prev_snapshot = prev_snapshot.drop(
                columns=["advection_bin", "time_to_next_bin", "u_wind", "v_wind"]
            )
        prev_snapshot["saved"] = False

        # --- --- ---
        # Phase 2: bin observation data by frame time (left-referenced)
        frame_indexes_obs = (
            np.digitize(
                flight_data_obs["time"].values.astype("datetime64[ns]").view("int64"),
                frame_timestamps.astype("datetime64[ns]").view("int64"),
            )
            - 1
        )
        # Clip to valid range: waypoints in (frame_timestamps[-1], frame_timestamps[-1] +
        # frame_interval) are assigned to the last frame rather than going out of bounds
        frame_indexes_obs = np.clip(frame_indexes_obs, 0, len(frame_timestamps) - 1)
        flight_data_obs["frame_index"] = frame_indexes_obs
        flight_data_obs["frame_time"] = frame_timestamps[frame_indexes_obs]
        flight_data_obs["scan_offset"] = detector.scan_time_offset_for_position(
            flight_data_obs["latitude"].values,
            flight_data_obs["longitude"].values,
            flight_data_obs["frame_time"].values,
        )
        flight_data_obs = flight_data_obs[~flight_data_obs["scan_offset"].isna()].copy()
        flight_data_obs["total_advected_time"] = np.timedelta64(0)
        flight_data_obs["saved"] = False

        frame_data_groups = flight_data_obs.groupby("frame_index")
        frames_with_data = set(frame_data_groups.groups.keys())

        # --- --- ---
        # Phase 3: step through each frame interval, advecting to per-pixel scan times
        store_frame_snapshots = []
        for frame_index, frame_timestamp in enumerate(frame_timestamps):
            if frame_index in frames_with_data:
                frame_data = frame_data_groups.get_group(frame_index).copy()
            else:
                frame_data = flight_data_obs.iloc[:0].copy()

            frame_advection_timestamps = np.arange(
                frame_timestamp + adv_timestep,
                frame_timestamp + frame_interval + adv_timestep,
                adv_timestep,
            )
            logging.info(f"Frame {frame_timestamp}: {len(frame_data)} new points")

            frame_data = self._bin_to_advection_times(frame_data, frame_advection_timestamps)
            frame_adv_groups = frame_data.groupby("advection_bin")
            available_bins_frame = set(frame_adv_groups.groups.keys())

            frame_output = []
            for i, advection_timestamp in enumerate(frame_advection_timestamps):
                if i in available_bins_frame:
                    new_data = frame_adv_groups.get_group(i).copy()
                else:
                    new_data = frame_data.iloc[:0].copy()

                if not prev_snapshot.empty:
                    prev_snapshot["time_to_next_bin"] = adv_timestep
                    # Track the current physical time so the scan-time gate below
                    # can reject points whose scan hasn't happened yet
                    prev_snapshot["time"] = advection_timestamp - adv_timestep
                    prev_snapshot["scan_offset"] = detector.scan_time_offset_for_position(
                        prev_snapshot["latitude"].values,
                        prev_snapshot["longitude"].values,
                        frame_timestamp,
                    )
                    prev_snapshot = prev_snapshot[~prev_snapshot["scan_offset"].isna()].copy()

                if prev_snapshot.empty and new_data.empty:
                    continue

                snapshot = pd.concat([prev_snapshot, new_data], axis=0, ignore_index=True)

                # Wind at each point's own time: the step start for carried points, the
                # waypoint time for points joining mid-step
                snapshot = self._add_winds(snapshot, snapshot["time"].to_numpy())

                # Identify points whose scan window falls in this advection step
                snapshot["scan_time"] = frame_timestamp + snapshot["scan_offset"]
                snapshot["currently_scanning"] = (
                    (snapshot["scan_time"] >= advection_timestamp - adv_timestep)
                    & (snapshot["scan_time"] < advection_timestamp)
                    # Guard against points that entered the snapshot mid-step and
                    # whose waypoint time hasn't been reached yet
                    & (snapshot["scan_time"] >= snapshot["time"])
                    & (~snapshot["saved"])
                )

                saving_snapshot = snapshot[snapshot["currently_scanning"]].copy()

                # Advance saving_snapshot to its exact scan time, then record
                time_to_scan_s = (
                    saving_snapshot["scan_time"] - saving_snapshot["time"]
                ) / np.timedelta64(1, "s")
                saving_snapshot["longitude"] = _wrap_longitude(
                    saving_snapshot["longitude"]
                    + units.m_to_longitude_distance(
                        time_to_scan_s * saving_snapshot["u_wind"],
                        saving_snapshot["latitude"].values,
                    )
                )
                saving_snapshot["latitude"] += units.m_to_latitude_distance(
                    time_to_scan_s * saving_snapshot["v_wind"]
                )
                saving_snapshot["total_advected_time"] += time_to_scan_s * np.timedelta64(1, "s")
                # Record only points within the lookback at their scan time: the step-end
                # filter below lets a point run up to one step past it before being dropped
                within_lookback = saving_snapshot["total_advected_time"] <= adv_max_past
                frame_output.append(
                    saving_snapshot[within_lookback][
                        [
                            "flight_id",
                            "original_time",
                            "longitude",
                            "latitude",
                            "altitude_ft",
                            "pressure_altitude_ft",
                            "scan_time",
                            "total_advected_time",
                        ]
                    ]
                )

                # Keep advancing saving_snapshot to the bin boundary so it carries
                # the right position into the next step (it stays in prev_snapshot)
                time_from_scan_to_bin_s = (
                    advection_timestamp - saving_snapshot["scan_time"]
                ) / np.timedelta64(1, "s")
                saving_snapshot["longitude"] = _wrap_longitude(
                    saving_snapshot["longitude"]
                    + units.m_to_longitude_distance(
                        time_from_scan_to_bin_s * saving_snapshot["u_wind"],
                        saving_snapshot["latitude"].values,
                    )
                )
                saving_snapshot["latitude"] += units.m_to_latitude_distance(
                    time_from_scan_to_bin_s * saving_snapshot["v_wind"]
                )
                saving_snapshot["total_advected_time"] += time_from_scan_to_bin_s * np.timedelta64(
                    1, "s"
                )
                saving_snapshot["saved"] = True
                saving_snapshot = saving_snapshot.drop(columns=["scan_time"])

                # Advance passing_snapshot (not yet scanned) by the full advection step
                passing_snapshot = snapshot[~snapshot["currently_scanning"]].copy()
                time_to_bin_s = passing_snapshot["time_to_next_bin"] / np.timedelta64(1, "s")
                passing_snapshot["longitude"] = _wrap_longitude(
                    passing_snapshot["longitude"]
                    + units.m_to_longitude_distance(
                        time_to_bin_s * passing_snapshot["u_wind"],
                        passing_snapshot["latitude"].values,
                    )
                )
                passing_snapshot["latitude"] += units.m_to_latitude_distance(
                    time_to_bin_s * passing_snapshot["v_wind"]
                )
                passing_snapshot["total_advected_time"] += passing_snapshot["time_to_next_bin"]

                prev_snapshot = pd.concat([passing_snapshot, saving_snapshot], axis=0)
                prev_snapshot = prev_snapshot.drop(
                    columns=["currently_scanning", "u_wind", "v_wind"]
                )
                prev_snapshot = prev_snapshot.reset_index(drop=True)
                prev_snapshot = prev_snapshot[prev_snapshot["total_advected_time"] <= adv_max_past]

            if not frame_output:
                continue

            frame_snapshot = pd.concat(frame_output, axis=0, ignore_index=True)
            frame_snapshot["frame_time"] = frame_timestamp
            frame_snapshot = frame_snapshot[
                [
                    "flight_id",
                    "original_time",
                    "longitude",
                    "latitude",
                    "altitude_ft",
                    "pressure_altitude_ft",
                    "frame_time",
                    "scan_time",
                    "total_advected_time",
                ]
            ].copy()
            store_frame_snapshots.append(frame_snapshot)
            # Allow carried-over points to be scanned again in the next frame
            prev_snapshot["saved"] = False

        if self._n_without_wind:
            logging.warning(
                f"Dropped {self._n_without_wind} advected points that moved outside the wind "
                "data's area (or lack wind data), rather than holding them still."
            )

        self.adv_flight_data = merge_dataframes_with_index(
            store_frame_snapshots,
            advection_step=int(adv_timestep / np.timedelta64(1, "m")),
        )

        if save_output:
            self._save_adv_flights()

    def parallax_correct_advected_tracks(
        self: Self,
        detector: Geostationary,
        save_output: bool = False,
    ) -> None:
        """Parallax correct the position of advected flight tracks based on satellite perspective.

        This function considers the ray from the satellite to the points of interest and finds
        the intersection of this ray with the WGS84 ellipsoid. The intersection point is then
        returned as the corrected longitude and latitude coordinates::

            @ satellite
             \\
              \\
               \\
                \\
                 \\
                  * aircraft
                   \\
                    \\
                     x parallax corrected aircraft
                -------------- surface

        [modified from pycontrails.datalib.geo_utils.parallax_correct()]

        Args:
            detector: Geostationary satellite detector providing satellite position.
            save_output: If True, overwrite the advected flight data file on disk with the
                parallax-corrected coordinates (lon_pc, lat_pc) included.
        """
        # Extract flight track coordinates
        adv_lon = self.adv_flight_data["longitude"].values
        adv_lat = self.adv_flight_data["latitude"].values
        adv_alt = units.ft_to_m(self.adv_flight_data["altitude_ft"].values)

        # Establish satellite position
        sat_lon, sat_lat = detector.central_lon, 0.0
        sat_alt = detector.satellite_height  # [m]

        # Convert from WGS84 to ECEF coordinates using detector's transformers
        p0 = np.array(detector.wgs84_to_ecef_transformer.transform([sat_lon], [sat_lat], [sat_alt]))
        p1 = np.array(detector.wgs84_to_ecef_transformer.transform(adv_lon, adv_lat, adv_alt))

        # Major and minor axes of the ellipsoid
        a = detector.ecef_crs.ellipsoid.semi_major_metre
        b = detector.ecef_crs.ellipsoid.semi_minor_metre
        intersection = intersection_with_ellipsoid(p0, p1, a, b)

        # Convert back to WGS84 coordinates
        transformed_points = detector.ecef_to_wgs84_transformer.transform(*intersection)[:2]

        # Add new columns to advected flight data
        self.adv_flight_data["lon_pc"] = transformed_points[0]
        self.adv_flight_data["lat_pc"] = transformed_points[1]

        if save_output:
            self._save_adv_flights()

    def _bin_to_advection_times(
        self: Self,
        flight_data: pd.DataFrame,
        advection_timestamps: np.ndarray,
        clip_to_bounds: bool = False,
    ) -> pd.DataFrame:
        """Assign each waypoint to its next advection bin and compute time-to-bin.

        Args:
            flight_data: DataFrame with a ``'time'`` column.
            advection_timestamps: Sorted array of advection step boundaries.
            clip_to_bounds: If True, drop waypoints outside [first, last) bin.

        Returns:
            flight_data with ``'advection_bin'``, ``'time_to_next_bin'``, and
            ``'total_advected_time'`` columns added.
        """
        bin_indices = np.digitize(
            flight_data["time"].values.astype("datetime64[ns]").view("int64"),
            advection_timestamps.astype("datetime64[ns]").view("int64"),
        )
        if clip_to_bounds:
            valid_mask = (bin_indices >= 1) & (bin_indices < len(advection_timestamps))
            flight_data = flight_data[valid_mask].copy()
            bin_indices = bin_indices[valid_mask]

        flight_data["advection_bin"] = bin_indices
        flight_data["time_to_next_bin"] = advection_timestamps[bin_indices].astype(
            "datetime64[ns]"
        ) - flight_data["time"].values.astype("datetime64[ns]")
        flight_data["total_advected_time"] = np.timedelta64(0)
        return flight_data

    def _filter_and_prepare_flight_data(
        self: Self,
        time_lower_bound: np.datetime64,
        time_upper_bound: np.datetime64,
        extra_cols: list[str] | None = None,
        flight_resample_timestep: int | None = None,
    ) -> pd.DataFrame:
        """Validate, filter, and slim down flight data ready for advection.

        Checks that ECMWF data is loaded and covers the required window, then
        filters waypoints to the requested time range and ECMWF altitude bounds.
        Optionally resamples each flight to a finer temporal resolution.

        Args:
            time_lower_bound: Earliest waypoint time to keep.
            time_upper_bound: Exclusive upper bound on waypoint time.
            extra_cols: Additional columns to retain beyond the core set. Dropped
                when ``flight_resample_timestep`` is set.
            flight_resample_timestep: If set, resample each flight's waypoints to
                this interval (in minutes) using pycontrails
                ``Flight.resample_and_fill`` before returning. Defaults to None.

        Returns:
            Filtered DataFrame.

        Raises:
            ValueError: If ECMWF data is not loaded or does not cover the time range.
        """
        if self.ecmwf_data is None:
            raise ValueError(
                "ECMWF wind data is not loaded. Set ecmwf_data first, e.g. from "
                "get_era5_from_local."
            )

        if not (
            self.ecmwf_data["time"].min() <= time_lower_bound
            and self.ecmwf_data["time"].max() >= time_upper_bound
        ):
            raise ValueError(
                "ECMWF wind data does not cover the advection time range.\n"
                f"Required: {time_lower_bound} to {time_upper_bound}\n"
                f"ECMWF:    {self.ecmwf_data['time'].min().values} to "
                f"{self.ecmwf_data['time'].max().values}\n"
                "Load winds for times from the first frame minus adv_max_past to the last "
                "frame plus one frame interval; get_era5_from_local adds the next day when "
                "that runs past 23:00."
            )

        self._validate_flight_data_columns()
        flight_data = self.flight_data.copy()

        flight_data = flight_data[
            (flight_data["time"] >= time_lower_bound) & (flight_data["time"] < time_upper_bound)
        ]
        logging.info(
            f"Filtered flight data to advection timeframe: {len(flight_data)} points remain."
        )

        flight_data["original_time"] = flight_data["time"]
        if "pressure_altitude_ft" not in flight_data.columns:
            flight_data["pressure_altitude_ft"] = flight_data["altitude_ft"]
        keep_cols = _CORE_COLS + [c for c in (extra_cols or []) if c in flight_data.columns]
        flight_data = flight_data[keep_cols]

        ecmwf_min_alt = np.min(self.ecmwf_data.coords["altitude_ft"].values)
        ecmwf_max_alt = np.max(self.ecmwf_data.coords["altitude_ft"].values)
        n_before = len(flight_data)
        # ERA5 levels are pressure altitudes, so compare like with like
        flight_data = flight_data[
            (flight_data["pressure_altitude_ft"] > ecmwf_min_alt)
            & (flight_data["pressure_altitude_ft"] < ecmwf_max_alt)
        ]
        logging.info(
            f"Removed {n_before - len(flight_data)} of {n_before} points outside "
            f"ECMWF altitude bounds [{ecmwf_min_alt:.0f}, {ecmwf_max_alt:.0f}] ft."
        )

        if flight_resample_timestep is not None:
            flight_data = self._resample_flight_data(flight_data, flight_resample_timestep)
            logging.info(
                f"Resampled flight data to {flight_resample_timestep}-min intervals: "
                f"{len(flight_data)} waypoints."
            )

        return flight_data

    def _add_winds(
        self: Self,
        snapshot: pd.DataFrame,
        wind_time: np.datetime64 | np.ndarray,
    ) -> pd.DataFrame:
        """Return ``snapshot`` with ``u_wind`` and ``v_wind``, dropping points with no wind.

        A point has no wind when it is outside the wind data's area. It is dropped rather
        than given zero wind, which would hold it still, and counted for the summary
        warning at the end of the advection.
        """
        u_wind, v_wind = self._interpolate_winds(snapshot, wind_time)
        snapshot = snapshot.assign(u_wind=u_wind, v_wind=v_wind)
        no_wind = snapshot["u_wind"].isna() | snapshot["v_wind"].isna()
        if no_wind.any():
            self._n_without_wind = getattr(self, "_n_without_wind", 0) + int(no_wind.sum())
            snapshot = snapshot[~no_wind].copy()
        return snapshot

    def _wind_field(self: Self) -> xr.Dataset:
        """The u/v wind field, closed across the date line when it covers the globe.

        Global ERA5 longitudes run from -180 to 179.75, so nothing brackets a point between
        179.75 and 180. A copy of the first longitude is appended 360 degrees on.
        """
        cached = getattr(self, "_wind_field_cache", None)
        if cached is not None and cached[0] is self.ecmwf_data:
            return cached[1]

        winds = self.ecmwf_data[["u", "v"]].sortby("longitude")
        lon = winds["longitude"].values
        if lon.size > 1:
            step = float(np.median(np.diff(lon)))
            span = float(lon[-1] - lon[0])
            if span < 360.0 and span + step >= 360.0 - 1e-6:
                first = winds.isel(longitude=[0])
                first = first.assign_coords(longitude=first["longitude"] + 360.0)
                winds = xr.concat([winds, first], dim="longitude")
        self._wind_field_cache = (self.ecmwf_data, winds)
        return winds

    def _interpolate_winds(
        self: Self,
        snapshot: pd.DataFrame,
        wind_time: np.datetime64 | np.ndarray,
    ) -> tuple:
        """Interpolate ECMWF u/v winds at snapshot positions and a given time.

        Longitudes are wrapped into the wind data's own range. Points outside the data get
        NaN.

        Args:
            snapshot: DataFrame with ``'longitude'``, ``'latitude'`` and
                ``'pressure_altitude_ft'`` columns.
            wind_time: Time at which to evaluate the wind field, one for all points or one
                per point.

        Returns:
            Tuple of (u_wind, v_wind) as numpy arrays.
        """
        winds = self._wind_field()
        lon_min = float(winds["longitude"].values[0])
        lon = (snapshot["longitude"].to_numpy() - lon_min) % 360.0 + lon_min
        interp_coords = {
            "longitude": xr.DataArray(lon, dims="points"),
            "latitude": xr.DataArray(snapshot["latitude"], dims="points"),
            "altitude_ft": xr.DataArray(snapshot["pressure_altitude_ft"], dims="points"),
            "time": xr.DataArray(
                np.broadcast_to(np.asarray(wind_time, dtype="datetime64[ns]"), len(snapshot)),
                dims="points",
            ),
        }
        interpolated_wind = winds.interp(
            **interp_coords, method="linear", kwargs={"fill_value": np.nan}
        )
        return interpolated_wind["u"].values, interpolated_wind["v"].values

    def _advection_cache_name(
        self: Self,
        detector: Geostationary,
        adv_max_past: np.timedelta64,
        adv_timestep: np.timedelta64,
        flight_resample_timestep: int | None,
    ) -> str:
        """File name for the advected flights: a readable prefix plus a fingerprint.

        The fingerprint covers everything that changes the result: the flight data itself
        (so it also reflects the bounding box, addresses and source), the detector's
        satellite, region, area and frame times, the wind data's extent, and the advection
        settings. Any change gives a new name, so an old file is never reused.
        """
        digest = hashlib.sha256()

        def add(value: object) -> None:
            digest.update(repr(value).encode())
            digest.update(b"|")

        add(_ADVECTION_CACHE_VERSION)
        add(type(self).__name__)
        flight_cols = [c for c in _CORE_COLS if c in self.flight_data.columns]
        add(len(self.flight_data))
        if not self.flight_data.empty:
            digest.update(
                pd.util.hash_pandas_object(
                    self.flight_data[flight_cols].reset_index(drop=True), index=False
                ).values.tobytes()
            )
        add(type(detector).__name__)
        add(getattr(detector, "satellite", None))
        add(getattr(detector, "region", None))
        add(getattr(detector, "extent", None))
        add([str(t) for t in detector.timestamp_sequence])
        if self.ecmwf_data is not None:
            for coord in ("time", "longitude", "latitude", "altitude_ft"):
                values = self.ecmwf_data[coord].values
                add((coord, str(values.min()), str(values.max()), values.size))
        add(str(np.timedelta64(adv_max_past, "s")))
        add(str(np.timedelta64(adv_timestep, "s")))
        add(flight_resample_timestep)

        prefix = self.save_name or "flights"
        return f"{prefix}_{digest.hexdigest()[:16]}"

    def _load_adv_flights(
        self: Self,
        save_path: str | Path | None = None,
    ) -> None:
        """Load advected flight data from disk.

        Args:
            save_path: Path from where data
                should be loaded. Defaults to `data/flight_data/<source>/advected`.
        """
        if save_path is None:
            if self.save_path is None:
                return None
            save_path = Path(self.save_path) / "advected"
        else:
            save_path = Path(save_path)

        if self._adv_cache_name is None:
            return None
        load_file = save_path / f"{self._adv_cache_name}.pq"
        if load_file.exists():
            return pd.read_parquet(load_file)
        else:
            return None

    def _resample_flight_data(
        self: Self,
        flight_data: pd.DataFrame,
        resample_timestep: int,
    ) -> pd.DataFrame:
        """Resample flight waypoints to a finer temporal resolution.

        Interpolates each flight's trajectory to ``resample_timestep``-minute
        intervals using pycontrails ``Flight.resample_and_fill``; pressure altitude is
        interpolated linearly in time onto the same times. Flights with fewer than two
        waypoints, or whose duration does not span a complete resampling interval, are
        returned unchanged.

        Only the core columns are preserved; any extra columns in the input are dropped.

        Args:
            flight_data: DataFrame with at minimum the core columns (flight_id,
                original_time, longitude, latitude, altitude_ft, pressure_altitude_ft, time).
            resample_timestep: Target resampling interval in minutes.

        Returns:
            Concatenated DataFrame of resampled trajectories with the core columns.
        """
        resampled = []
        for flight_id, grp in flight_data.groupby("flight_id"):
            grp = (
                grp.sort_values("time")
                .drop_duplicates("time")
                .dropna(subset=["altitude_ft"])
                .copy()
            )
            if len(grp) < 2:
                resampled.append(grp[_CORE_COLS])
                continue

            # Check the flight spans at least one full resampling interval
            t_start = grp["time"].min().ceil(f"{resample_timestep}min")
            t_end = grp["time"].max().floor(f"{resample_timestep}min")
            if t_start > t_end:
                resampled.append(grp[_CORE_COLS])
                continue

            grp["altitude"] = units.ft_to_m(grp["altitude_ft"].values)
            try:
                df = (
                    Flight(grp[["longitude", "latitude", "altitude", "time"]])
                    .resample_and_fill(f"{resample_timestep}min")
                    .to_dataframe()
                )
                df["altitude_ft"] = units.m_to_ft(df["altitude"].values)
                df["pressure_altitude_ft"] = np.interp(
                    df["time"].values.astype("datetime64[ns]").view("int64"),
                    grp["time"].values.astype("datetime64[ns]").view("int64"),
                    grp["pressure_altitude_ft"].values,
                )
                df["flight_id"] = flight_id
                df["original_time"] = df["time"]
                resampled.append(df[_CORE_COLS])
            except Exception as e:
                logging.warning(
                    f"Resampling failed for flight {flight_id}, using original waypoints: {e}"
                )
                resampled.append(grp[_CORE_COLS])

        if not resampled:
            return flight_data[_CORE_COLS]

        return pd.concat(resampled, ignore_index=True)

    def _save_adv_flights(
        self: Self,
        save_path: str | Path | None = None,
    ) -> None:
        """Save advected flight data to disk.

        Args:
            save_path: Path to where data
                should be saved. Defaults to `data/flight_data/<source>/advected`.
                If not provided and ``self.save_path`` is also unset, logs a
                warning and returns without saving.
        """
        if save_path is None:
            if self.save_path is None:
                logging.warning("save_path is not set; advected flight data will not be saved.")
                return
            save_path = Path(self.save_path) / "advected"

        if self._adv_cache_name is None:
            logging.warning(
                "Advected flight data was not produced by advect_flights_to_scan_times on "
                "this object, so it is not saved."
            )
            return
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        save_file = save_path / f"{self._adv_cache_name}.pq"
        self.adv_flight_data.to_parquet(save_file)
        logging.info(f"Saved advected flight data to {save_file}")

    def _validate_flight_data_columns(self: Self) -> None:
        """Validate that the flight data contains the required columns."""
        required_columns = {
            "flight_id",
            "time",
            "longitude",
            "latitude",
            "altitude_ft",
            "level",
        }
        missing_columns = required_columns - set(self.flight_data.columns)

        # --- --- ---
        # Altitude conversions
        if "altitude_ft" in missing_columns and "altitude" in self.flight_data.columns:
            self.flight_data["altitude_ft"] = units.m_to_ft(self.flight_data["altitude"])
            missing_columns.remove("altitude_ft")

        if "altitude_ft" in missing_columns and "geometric_height" in self.flight_data.columns:
            # geometric_height must be in feet. Cruise in metres peaks near 13,000, so a
            # maximum below 15,000 most likely means metres, about 3.3 times too low.
            self.flight_data["altitude_ft"] = self.flight_data["geometric_height"].copy()
            missing_columns.remove("altitude_ft")
            max_height = self.flight_data["geometric_height"].max()
            if pd.notna(max_height) and 0 < max_height < 15000:
                logging.warning(
                    f"geometric_height peaks at {max_height:.0f}, which looks like metres; it "
                    "is used as altitude in feet. Convert it to feet (or provide altitude_ft)."
                )

        # Pressure level, once altitude_ft has been derived above
        if "level" in missing_columns and "altitude_ft" in self.flight_data.columns:
            self.flight_data["level"] = units.ft_to_pl(self.flight_data["altitude_ft"])
            missing_columns.remove("level")

        if "level" in missing_columns and "altitude" in self.flight_data.columns:
            self.flight_data["level"] = units.m_to_pl(self.flight_data["altitude"])
            missing_columns.remove("level")

        # --- --- ---
        if missing_columns:
            raise ValueError(f"Flight data is missing required columns: {missing_columns}")
