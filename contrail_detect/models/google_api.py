"""
Contrail detection retrieval from the Google Contrails REST API.

Provides detection masks for GOES-East, GOES-West, Himawari-9, and MTG via the
Google Contrails v2 API (https://developers.google.com/contrails/reference/v2).

Undocumented change in the API's scores, 2026-10-07
----------------------------------------------------
Masks served for frames from about 21:10-21:50 UTC on 2026-10-07 onwards differ from earlier
ones, for all four satellite origins. Measured 2026-11-17 by requesting frames either side of
the change; the API carries no model version and no announcement has been found.

- The first frame in the new form is 21:50 for GOES-East and GOES-West, and between 21:10 and
  21:40 for Himawari and MTG. Himawari alternated between the two forms from 21:10 to 21:40.
- Scores now have a lower bound (3.068e-7 for GOES, 1.249e-7 for Himawari, about 1.25e-6 for
  MTG) that a large share of pixels sit at. Earlier frames have no such bound.
- Scores above the bound changed too, so this is not only a crop. On 2026-10-07 between 21:00
  and 22:10, the number of pixels scoring >= 0.40 fell by about 15% (GOES-East) and 25%
  (Himawari) across the change.
- Earlier frames were not reprocessed: requests today for dates before the change return the
  old form.

Detections at a fixed threshold may therefore not be comparable across this date. A warning is
logged when requested timestamps fall on or after it.

"""

import io
import logging
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

from contrail_detect.core.config import GOOGLE_CONTRAILS_API_KEY
from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.models.contrail_model_base import (
    ContrailModelBase,
    get_with_retries,
    require_api_frame_times,
)

# Nominal geostationary satellite height above the surface, for converting scan angles to
# metres when a response does not carry its own
_GEOSTATIONARY_HEIGHT_M = 35786023.0

_GOOGLE_API_URL = "https://contrails.googleapis.com/v2/detections/masks"
_GOOGLE_SEGMENTS_URL = "https://contrails.googleapis.com/v2/detections"

_SATELLITE_ORIGIN_MAP = {
    "GOES-16": "GOES-EAST-FULL-DISK",
    "GOES-18": "GOES-WEST-FULL-DISK",
    "GOES-19": "GOES-EAST-FULL-DISK",
    "HIMAWARI-9": "HIMAWARI-FULL-DISK",
    "MTG": "MTG-000-FULL-DISK",
}

# Earliest frame seen in the changed score form (see the module docstring). Frames up to
# 21:50 UTC may be in either form, depending on satellite.
_SCORE_CHANGEOVER = np.datetime64("2026-10-07T21:10")


class GoogleAPIContrailModel(ContrailModelBase):
    """Contrail detection retrieval from the Google Contrails REST API.

    Retrieves contrail detection masks from Google's v2 contrails API for
    GOES-East, GOES-West, Himawari-9, and MTG Full Disk data. Requests are batched into
    ≤1-hour windows to minimise API calls. Results are cached as .npy files
    and consolidated into a zarr store.

    Attributes:
        detector: Geostationary instance providing satellite data parameters.
        satellite_origin: API satellite identifier string.
        xx: X-coordinate meshgrid for detections.
        yy: Y-coordinate meshgrid for detections.
    """

    def __init__(
        self: Self,
        detector: Geostationary,
        detection_output_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the GoogleAPIContrailModel.

        Args:
            detector: Satellite detector instance (GOES-16/18/19, HIMAWARI-9, or MTG,
                Full Disk only).
            detection_output_dir: Directory for caching detection outputs. Defaults to
                detector's output path with 'google_api' subdirectory.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.

        Raises:
            ValueError: If detector satellite is unsupported or region is not Full Disk.
        """
        valid_satellites = list(_SATELLITE_ORIGIN_MAP.keys())
        if detector.satellite not in valid_satellites:
            raise ValueError(
                f"Detector satellite must be one of {valid_satellites}, got '{detector.satellite}'"
            )
        if detector.region != "F":
            raise ValueError("Detector region must be 'F' (Full Disk)")

        super().__init__(
            detector=detector,
            detection_output_dir=detection_output_dir,
            subdirectory="google_api",
        )

        self.contrail_model_name = "Google Contrails API"
        self.timeout = timeout
        self.max_retries = max_retries
        self.satellite_origin = _SATELLITE_ORIGIN_MAP[detector.satellite]

        grid_dir = f"{self.detector.satellite}-{self.detector.region}-GRID"
        self.grid_path = self.detection_output_dir / grid_dir
        self.grid_path.mkdir(parents=True, exist_ok=True)

        self.xx, self.yy = None, None

        self._validate_request_timestamp()

    def _validate_detector(self: Self) -> None:
        """Validate detector compatibility, allowing lazily-populated detectors.

        Overrides the base class check to skip the ``lons``/``lats``/
        ``image_geo_coords`` None guards, for detectors such as ``MeteosatDetector``
        whose grid is populated lazily from detection data.
        """
        if not isinstance(self.detector, Geostationary):
            raise TypeError(
                f"Detector must be an instance of Geostationary, got {type(self.detector)} instead."
            )

        required_attrs = [
            "DETECT_PATH",
            "satellite",
            "region",
            "tile_extent",
            "lons",
            "lats",
            "timestamp_strings",
            "image_geo_coords",
        ]
        missing = [a for a in required_attrs if not hasattr(self.detector, a)]
        if missing:
            raise AttributeError(
                f"Detector {self.detector.__class__.__name__} is missing required "
                f"attributes: {', '.join(missing)}"
            )

        if not self.detector.timestamp_strings:
            raise ValueError(
                "Detector timestamp_strings is None or empty. "
                "Please ensure detector has loaded timestamp data."
            )

    def _validate_request_timestamp(self: Self) -> None:
        """Validate that detector timestamps fall within the API's data availability window.

        Raises:
            ValueError: If any timestamp predates the API's earliest available data for
                the detector's satellite.
        """
        _api_start_dates = {
            "GOES-16": np.datetime64("2024-01-01T00:00"),
            "GOES-18": np.datetime64("2024-01-01T00:00"),
            "GOES-19": np.datetime64("2024-01-01T00:00"),
            "HIMAWARI-9": np.datetime64("2024-01-01T00:00"),
            "MTG": np.datetime64("2025-01-31T00:00"),
        }
        horizon = _api_start_dates[self.detector.satellite]
        min_ts = np.nanmin(self.detector.timestamp_sequence)
        max_ts = np.nanmax(self.detector.timestamp_sequence)

        if max_ts >= _SCORE_CHANGEOVER:
            spans = (
                " The requested timestamps span the change." if min_ts < _SCORE_CHANGEOVER else ""
            )
            logging.warning(
                f"Google API scores changed without notice around {_SCORE_CHANGEOVER} UTC "
                "(lower bound added; ~15-25% fewer pixels >= 0.40). Detections from on or after "
                f"this time may not be comparable with earlier ones.{spans} See the "
                "google_api module docstring."
            )
        if min_ts < horizon:
            raise ValueError(
                f"Detector timestamp series starts ({min_ts}) before "
                f"Google API data availability for {self.detector.satellite} ({horizon})."
            )

        # GOES-West has no entry: its masks cover a fixed sub-disk window (roughly
        # 5°S–69°N, 173°W–76°W; ~44% of the disk), so it gets the partial-coverage warning.
        _full_disk_cutover = {
            "GOES-16": np.datetime64("2024-12-19T00:00"),
            "GOES-19": np.datetime64("2024-12-19T00:00"),
        }
        cutover = _full_disk_cutover.get(self.detector.satellite)
        if cutover is None:
            logging.warning(
                f"Google API provides partial coverage only for {self.detector.satellite}. "
                "Detection masks will not cover the full disk."
            )
        elif min_ts < cutover:
            logging.warning(
                f"Some requested timestamps fall before {cutover}, when the Google API "
                f"only provides partial disk coverage for {self.detector.satellite}. "
                "Full disk coverage is available from 2024-12-19 onwards."
            )

    def run_detections(
        self: Self,
        force_reprocess: bool = False,
        zarr_path: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """Retrieve contrail detection masks from the Google Contrails API.

        Fetches masks for all detector timestamps. Loads from cache first,
        downloading only what is missing. Groups timestamps into ≤1-hour
        windows for efficient batch requests.

        A timestamp with no frame in the response is archived as NaN. Timestamps in a
        failed request are returned as NaN but not archived, so are requested again.

        Args:
            force_reprocess: Re-download all detections even if cached.
            zarr_path: Optional zarr store path. Defaults to detector path.

        Returns:
            Dictionary mapping timestamps to detection mask arrays on the detector grid.
            Timestamps without a detection map to NaN arrays.

        Raises:
            ValueError: If any timestamp is off the API's 10-minute frame schedule.
        """
        timestamps = self.detector.timestamp_sequence
        require_api_frame_times(timestamps, "Google Contrails API")
        requested_timestamps = set(timestamps)

        if force_reprocess:
            detection_dict, timestamps_found = {}, set()
        else:
            detection_dict, timestamps_found = self._load_cached_detections(zarr_path)

        if timestamps_found >= requested_timestamps:
            zarr_store_path = self.detector.get_zarr_path(zarr_path)
            if not zarr_store_path.exists():
                self._detections_to_zarr(zarr_path=zarr_path)
            return self._fill_missing_detections_with_nans(detection_dict)

        missing = [ts for ts in timestamps if ts not in timestamps_found]
        windows = self._group_into_windows(missing)

        logging.info(f"Fetching {len(missing)} missing timestamps in {len(windows)} API window(s)")

        failed: set[np.datetime64] = set()
        for window_start, window_end, window_timestamps in tqdm(
            windows, desc="Requesting from Google API"
        ):
            try:
                ds = self._request_window(window_start, window_end)
                self._extract_and_cache(ds, window_timestamps, timestamps, detection_dict)
                ds.close()
            except requests.HTTPError as e:
                logging.warning(
                    f"HTTP {e.response.status_code} for window {window_start}–{window_end}"
                )
                failed.update(window_timestamps)
            except Exception as e:
                logging.error(f"Failed window {window_start}–{window_end}: {e}")
                failed.update(window_timestamps)

        # Keep any frame extracted before its window failed
        failed = {ts for ts in map(np.datetime64, failed) if detection_dict.get(ts) is None}
        for ts in failed:
            detection_dict[ts] = None

        detection_dict = self._fill_missing_detections_with_nans(detection_dict)
        self._detections_to_zarr(zarr_path=zarr_path, exclude_timestamps=failed)
        return detection_dict

    def run_segments(
        self: Self,
        bounds: list[tuple[float, float]] | None = None,
    ) -> pd.DataFrame:
        """Retrieve contrail line segment detections from the Google Contrails API.

        Makes a single request spanning the detector's full timestamp range (up to
        24 hours). Results are returned as a DataFrame in Decomposer output format.

        Args:
            bounds: Optional list of ``(latitude, longitude)`` vertex pairs defining
                a bounding polygon to spatially filter results. Minimum 3 vertices
                required if provided.

        Returns:
            DataFrame of contrail line segments in Decomposer output format, or an
            empty DataFrame if no detections are found.

        Raises:
            requests.HTTPError: If the API returns an error status.
        """
        timestamps = self.detector.timestamp_sequence
        start_time = self._fmt_iso(min(timestamps))
        end_time = self._fmt_iso(max(timestamps) + np.timedelta64(1, "m"))

        params: dict = {
            "start_time": start_time,
            "end_time": end_time,
            "satellite_origins": self.satellite_origin,
        }
        if bounds is not None:
            if len(bounds) < 3:
                raise ValueError(f"bounds must have at least 3 vertices, got {len(bounds)}")
            params["bounds"] = [f"{lat},{lon}" for lat, lon in bounds]

        headers = {"x-goog-api-key": GOOGLE_CONTRAILS_API_KEY} if GOOGLE_CONTRAILS_API_KEY else {}

        logging.info(f"Requesting Google API segments: {start_time} → {end_time}")
        resp = get_with_retries(
            _GOOGLE_SEGMENTS_URL,
            self.max_retries,
            self.timeout,
            params=params,
            headers=headers,
        )

        features = resp.json().get("features", [])
        if not features:
            logging.warning("No segment detections returned by Google API")
            return pd.DataFrame()

        rows = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]
            props = feature["properties"]
            rows.append(
                {
                    "time": pd.Timestamp(props["time"]),
                    "lon0": coords[0][0],
                    "lat0": coords[0][1],
                    "lon1": coords[-1][0],
                    "lat1": coords[-1][1],
                }
            )

        return self._refactor_segments(pd.DataFrame(rows))

    def _group_into_windows(
        self: Self,
        timestamps: list[np.datetime64],
        max_hours: int = 1,
    ) -> list[tuple[str, str, list[np.datetime64]]]:
        """Group timestamps into contiguous windows of at most max_hours.

        Each window produces a single API request. A 1-minute buffer is added
        past the last timestamp in each window so it is included in the response.

        Args:
            timestamps: Sorted or unsorted list of timestamps to group.
            max_hours: Maximum window span in hours (API limit is 1).

        Returns:
            List of (start_time_iso, end_time_iso, timestamps_in_window) tuples.
        """
        if not timestamps:
            return []

        sorted_ts = sorted(timestamps)
        window_delta = np.timedelta64(max_hours, "h")
        buffer = np.timedelta64(1, "m")

        windows = []
        window_start = sorted_ts[0]
        current_group = [sorted_ts[0]]

        for ts in sorted_ts[1:]:
            if ts - window_start < window_delta:
                current_group.append(ts)
            else:
                windows.append(
                    (
                        self._fmt_iso(window_start),
                        self._fmt_iso(current_group[-1] + buffer),
                        current_group,
                    )
                )
                window_start = ts
                current_group = [ts]

        windows.append(
            (
                self._fmt_iso(window_start),
                self._fmt_iso(current_group[-1] + buffer),
                current_group,
            )
        )

        return windows

    def _request_window(
        self: Self,
        start_time: str,
        end_time: str,
    ) -> xr.Dataset:
        """Make a single API request for a time window with exponential-backoff retry.

        Args:
            start_time: ISO 8601 start time (e.g. ``'2026-01-20T21:00:00Z'``).
            end_time: ISO 8601 end time (must be within 1 hour of start_time).

        Returns:
            xarray Dataset loaded into memory containing detection masks.

        Raises:
            requests.HTTPError: If a non-5xx error is received, or all retries fail.
            requests.Timeout: If all retries time out.
        """
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "satellite_origin": self.satellite_origin,
        }
        headers = {"x-goog-api-key": GOOGLE_CONTRAILS_API_KEY} if GOOGLE_CONTRAILS_API_KEY else {}

        logging.info(f"Requesting Google API masks: {start_time} → {end_time}")
        resp = get_with_retries(
            _GOOGLE_API_URL, self.max_retries, self.timeout, params=params, headers=headers
        )
        ds = xr.open_dataset(io.BytesIO(resp.content), engine="h5netcdf")
        return ds.load()

    def _extract_and_cache(
        self: Self,
        ds: xr.Dataset,
        window_timestamps: list[np.datetime64],
        all_timestamps: list[np.datetime64],
        detection_dict: dict[np.datetime64, np.ndarray],
    ) -> None:
        """Extract per-timestamp detection arrays from an API response and cache to disk.

        Coordinates are converted from radians to meters if the dataset uses
        GOES geostationary projection with radian units. Masks are matched to requested
        timestamps by time and stored on the detector's grid.

        Args:
            ds: Dataset returned by the API (dimensions: time, y, x).
            window_timestamps: Detector timestamps that were requested in this window.
            all_timestamps: Full detector timestamp sequence (for index lookup).
            detection_dict: Detection dict updated in-place with extracted arrays.

        Raises:
            ValueError: If the response has no time dimension or coordinate.
        """
        x_coords = ds["x"].values
        y_coords = ds["y"].values

        sat_height = ds.attrs.get("goes_imager_projection", {}).get("perspective_point_height")

        if abs(float(x_coords.max())) < 1.0:
            # Coordinates in radians — convert to metres
            sat_height = (
                sat_height
                or getattr(self.detector, "satellite_height", None)
                or _GEOSTATIONARY_HEIGHT_M
            )
            x_meters = x_coords * sat_height
            y_meters = y_coords * sat_height
        else:
            x_meters = x_coords
            y_meters = y_coords

        self._populate_lazy_grid(x_meters, y_meters, ds.attrs)

        if self.xx is None or self.yy is None:
            self.xx, self.yy = np.meshgrid(x_meters, y_meters)
            np.save(str(self.grid_path / "_x_coords.npy"), self.xx)
            np.save(str(self.grid_path / "_y_coords.npy"), self.yy)

        # Resolve the mask variable name (typically 'mask')
        mask_var = "mask" if "mask" in ds else list(ds.data_vars)[0]
        if mask_var != "mask":
            logging.debug(f"Using data variable '{mask_var}' as detection mask")

        masks = ds[mask_var]
        if "time" not in masks.dims:
            if "time" in ds.coords and ds["time"].ndim == 0:
                # Single frame labelled by a scalar time coordinate
                masks = masks.expand_dims(time=[ds["time"].values])
            else:
                raise ValueError("Google API response has no time dimension or coordinate")
        masks = masks.transpose("time", "y", "x")

        # Match API response times to requested timestamps exactly (to the minute), so a
        # stored mask is always the frame at its own timestamp, never a neighbouring one
        api_times = pd.DatetimeIndex(np.atleast_1d(masks["time"].values)).floor("min")

        all_ts_list = list(all_timestamps)

        for req_ts in window_timestamps:
            req_pd = pd.Timestamp(req_ts).floor("min")

            time_diffs = np.abs((api_times - req_pd).total_seconds().values)
            best_idx = int(np.argmin(time_diffs))

            if time_diffs[best_idx] > 0:
                logging.warning(
                    f"No API frame at {req_pd:%Y-%m-%dT%H:%M}; closest is "
                    f"{api_times[best_idx]:%Y-%m-%dT%H:%M}. Leaving it empty."
                )
                detection_dict[np.datetime64(req_ts)] = None
                continue

            mask = masks.isel(time=best_idx).values.copy()
            mask = self._to_detector_grid(mask, x_meters, y_meters)

            detection_dict[np.datetime64(req_ts)] = mask

            t_idx = all_ts_list.index(req_ts)
            self._save_detection_to_disk(
                t_idx,
                mask,
                source="Google Contrails API v2",
                satellite_origin=self.satellite_origin,
            )

    def _to_detector_grid(
        self: Self,
        mask: np.ndarray,
        x_meters: np.ndarray,
        y_meters: np.ndarray,
    ) -> np.ndarray:
        """Put a (y, x) mask on the detector's grid.

        A mask on the same grid with either axis reversed is flipped. Any other grid is
        interpolated.

        Args:
            mask: Mask with dims (y, x) on the response grid.
            x_meters: Response x coordinates, geostationary metres.
            y_meters: Response y coordinates, geostationary metres.

        Returns:
            Mask with the detector grid's shape and orientation.
        """
        detector_coords = self.detector.image_geo_coords
        if detector_coords is None:
            return mask
        det_x = np.asarray(detector_coords[0], dtype=float)
        det_y = np.asarray(detector_coords[1], dtype=float)

        def _axis_match(src: np.ndarray, dst: np.ndarray) -> str | None:
            # 'same', 'reversed' or None, to within 1% of a detector pixel
            if len(src) != len(dst):
                return None
            tol = 0.01 * float(np.median(np.abs(np.diff(dst)))) if len(dst) > 1 else 1e-6
            if np.allclose(src, dst, rtol=0.0, atol=tol):
                return "same"
            if np.allclose(src[::-1], dst, rtol=0.0, atol=tol):
                return "reversed"
            return None

        x_match, y_match = _axis_match(x_meters, det_x), _axis_match(y_meters, det_y)
        if x_match and y_match:
            if y_match == "reversed":
                mask = mask[::-1, :]
            if x_match == "reversed":
                mask = mask[:, ::-1]
            return np.ascontiguousarray(mask)

        logging.info(
            f"Google API grid {len(y_meters)}x{len(x_meters)} differs from the detector grid "
            f"{len(det_y)}x{len(det_x)}; regridding"
        )
        # Sort ascending, which every scipy version's interpolator accepts
        x_order, y_order = np.argsort(x_meters), np.argsort(y_meters)
        return self._regrid_to_detector_coordinates(
            mask[np.ix_(y_order, x_order)], x_meters[x_order], y_meters[y_order]
        )

    def _load_cached_detections(
        self: Self,
        zarr_path: str | Path | None = None,
    ) -> tuple[dict[np.datetime64, np.ndarray], set[np.datetime64]]:
        """Load cached detections from zarr store or .npy files.

        Args:
            zarr_path: Optional path to zarr store.

        Returns:
            Tuple of (detection_dict, timestamps_found_set).
        """
        detection_dict: dict[np.datetime64, np.ndarray] = {}
        timestamps = self.detector.timestamp_sequence
        requested_timestamps = set(timestamps)
        timestamps_found: set[np.datetime64] = set()

        zarr_store_path = self.detector.get_zarr_path(zarr_path)
        if zarr_store_path.exists():
            try:
                zarr_detections = self.detector.load_detections_from_zarr(zarr_path=zarr_path)
                zarr_timestamps = set(zarr_detections.keys()) & requested_timestamps
                detection_dict.update({ts: zarr_detections[ts] for ts in zarr_timestamps})
                timestamps_found.update(zarr_timestamps)
                logging.info(f"Loaded {len(zarr_timestamps)}/{len(requested_timestamps)} from zarr")
            except Exception as e:
                logging.warning(f"Zarr load failed: {e}")

        for idx, ts in enumerate(timestamps):
            if ts in timestamps_found:
                continue
            npy_path = self.detector.get_npy_path(idx, self.detection_output_dir)
            if npy_path.exists():
                detection_dict[ts] = np.load(npy_path)
                timestamps_found.add(ts)

        # Load cached grid coords (may exist even when zarr does not)
        if self.xx is None:
            xx_path = self.grid_path / "_x_coords.npy"
            if xx_path.exists():
                self.xx = np.load(str(xx_path))
        if self.yy is None:
            yy_path = self.grid_path / "_y_coords.npy"
            if yy_path.exists():
                self.yy = np.load(str(yy_path))

        if self.xx is not None and self.yy is not None:
            self._populate_lazy_grid(self.xx[0, :], self.yy[:, 0])

        return detection_dict, timestamps_found

    def _populate_lazy_grid(
        self: Self,
        x_meters: np.ndarray,
        y_meters: np.ndarray,
        attrs: dict | None = None,
    ) -> None:
        """Give a lazily-populated detector (e.g. ``MeteosatDetector``) its grid, if empty.

        Saving and decomposing detections need the detector's grid. Detectors without a
        fixed grid take it from the detection coordinates.

        Args:
            x_meters: Geostationary x coordinates of the detection grid, in metres.
            y_meters: Geostationary y coordinates of the detection grid, in metres.
            attrs: Projection attributes from the API response, if any.
        """
        if not hasattr(self.detector, "populate_metadata_from_dataset"):
            return
        if self.detector.image_geo_coords is not None:
            return
        logging.info("Populating detector grid from Google API coordinates")
        grid = xr.Dataset(coords={"x": x_meters, "y": y_meters}, attrs=dict(attrs or {}))
        self.detector.populate_metadata_from_dataset(grid)

    def _fill_missing_detections_with_nans(
        self: Self,
        detections_to_fill: dict[np.datetime64, np.ndarray],
    ) -> dict[np.datetime64, np.ndarray]:
        """Replace None entries with NaN arrays sized to match the detector grid.

        Falls back to the API grid if the detector has no grid yet.
        """
        for timestamp in list(detections_to_fill.keys()):
            if detections_to_fill[timestamp] is None:
                coords = self.detector.image_geo_coords
                if coords is not None:
                    shape = (len(coords[1]), len(coords[0]))
                elif self.xx is not None and self.yy is not None:
                    shape = self.xx.shape
                else:
                    raise ValueError("No detector or API grid to size the NaN fill")
                detections_to_fill[timestamp] = np.full(shape, np.nan, dtype=np.float32)
        return detections_to_fill

    @staticmethod
    def _fmt_iso(ts: np.datetime64) -> str:
        """Format a numpy datetime64 as an ISO 8601 UTC string."""
        return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
