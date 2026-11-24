"""
Base class for contrail detection models.

Provides common interface and shared functionality for different contrail detection
implementations including API-based retrieval and local inference.

"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from contrail_detect import __version__
from contrail_detect.core.common import Transforms
from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.geostationary.geo_utils import detection_encoding
from contrail_detect.geostationary.validation_utils import validate_extent

_transforms = Transforms()


def get_with_retries(url: str, max_retries: int, timeout: float, **kwargs) -> requests.Response:
    """GET ``url``, retrying transient failures with exponential backoff.

    Timeouts, dropped connections, 429 (too many requests) and 5xx responses are retried.

    Args:
        url: Request URL.
        max_retries: Maximum number of attempts.
        timeout: Per-request timeout in seconds.
        **kwargs: Passed to ``requests.get`` (e.g. ``params``, ``headers``).

    Returns:
        The successful response.

    Raises:
        requests.Timeout: If every attempt times out.
        requests.ConnectionError: If every attempt fails to connect.
        requests.HTTPError: On a 4xx response other than 429, or a retryable response on
            the last attempt.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if isinstance(e, requests.HTTPError):
                status = e.response.status_code
                retryable = status == 429 or status >= 500
                reason = f"HTTP {status}"
            else:
                retryable = True
                reason = type(e).__name__
            if not retryable or attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logging.warning(f"{reason} (attempt {attempt + 1}/{max_retries}), retrying in {wait}s")
            time.sleep(wait)


# Full Disk frames from both detection APIs are every 10 minutes, starting on the hour
API_FRAME_INTERVAL_MINUTES = 10


def require_api_frame_times(timestamps: list[np.datetime64], source: str) -> None:
    """Refuse timestamps that are not on the APIs' Full Disk frame schedule.

    The APIs serve one frame every ``API_FRAME_INTERVAL_MINUTES`` minutes from the hour
    (:00, :10, ...). A timestamp between frames has no frame of its own: the Contrails.org
    API still answers, without saying which frame it sent, so the result would be stored
    under a time it does not belong to. Checking before any request means every detection
    is labelled with the frame it actually is.

    Args:
        timestamps: Requested timestamps, typically the detector's timestamp sequence.
        source: API name for the error message.

    Raises:
        ValueError: If any timestamp is off the schedule, naming them and an aligned start.
    """
    times = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    interval = f"{API_FRAME_INTERVAL_MINUTES}min"
    off_schedule = times[times != times.floor(interval)]
    if len(off_schedule) == 0:
        return

    shown = ", ".join(t.strftime("%Y-%m-%dT%H:%M:%S") for t in off_schedule[:5])
    if len(off_schedule) > 5:
        shown += f" and {len(off_schedule) - 5} more"
    first = times.min()
    raise ValueError(
        f"The {source} has Full Disk frames only every {API_FRAME_INTERVAL_MINUTES} minutes "
        f"from the hour (:00, :10, ...), so these timestamps have no frame of their own: "
        f"{shown}. Start the detector on a {API_FRAME_INTERVAL_MINUTES}-minute boundary, "
        f"e.g. {first.floor(interval):%Y-%m-%dT%H:%M} or {first.ceil(interval):%Y-%m-%dT%H:%M}."
    )


class ContrailModelBase(ABC):
    """Abstract base class for contrail detection models.

    Defines the common interface and shared attributes for contrail detection
    implementations. Subclasses must implement model-specific detection logic
    while inheriting common configuration and output handling.

    Attributes:
        detector: Geostationary satellite detector instance providing data.
        detection_output_dir: Directory path for storing detection outputs.
    """

    # Details of data availability for different detection sources
    # (APIs and local inference) to inform users
    __FUTURE_BACKSTOP = np.datetime64("2100-01-01T00:00")
    DETECTION_DATES: list[dict] = [
        # Contrails API — split by GOES era: GOES-19 replaced GOES-16 as GOES-East on
        # 2025-04-07, matching Geostationary.SATELLITE_DATES
        {
            "Name": "Contrails API",
            "Type": "Masks",
            "Satellites": ["GOES-16"],
            "Start Datetime": np.datetime64("2024-01-01T00:00"),
            "End Datetime": np.datetime64("2025-04-07T00:00"),
            "Region": "Full Disk (F)",
        },
        {
            "Name": "Contrails API",
            "Type": "Masks",
            "Satellites": ["GOES-19"],
            "Start Datetime": np.datetime64("2025-04-07T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
        },
        # Google API — Detection Masks
        {
            "Name": "Google API",
            "Type": "Masks",
            "Satellites": ["GOES-16", "GOES-19"],
            "Start Datetime": np.datetime64("2024-01-01T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
            "Notes": "Partial coverage from 2024-01-01; full disk from 2024-12-19",
        },
        {
            "Name": "Google API",
            "Type": "Masks",
            "Satellites": ["HIMAWARI-9"],
            "Start Datetime": np.datetime64("2024-01-01T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
            "Notes": "Partial coverage only",
        },
        {
            "Name": "Google API",
            "Type": "Masks",
            "Satellites": ["MTG"],
            "Start Datetime": np.datetime64("2025-01-31T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
            "Notes": "Partial coverage only",
        },
        # Google API — Line Segments
        {
            "Name": "Google API",
            "Type": "Segments",
            "Satellites": ["GOES-16", "GOES-19"],
            "Start Datetime": np.datetime64("2024-01-01T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
        },
        {
            "Name": "Google API",
            "Type": "Segments",
            "Satellites": ["HIMAWARI-9"],
            "Start Datetime": np.datetime64("2024-01-01T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
        },
        {
            "Name": "Google API",
            "Type": "Segments",
            "Satellites": ["MTG"],
            "Start Datetime": np.datetime64("2025-01-31T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
        },
        # Local Inference
        {
            "Name": "Local Inference",
            "Type": "Masks",
            "Satellites": ["GOES-16", "GOES-19"],
            "Start Datetime": np.datetime64("2025-01-01T00:00"),
            "End Datetime": __FUTURE_BACKSTOP,
            "Region": "Full Disk (F)",
        },
    ]

    def __init__(
        self: Self,
        detector: Geostationary,
        detection_output_dir: str | Path | None = None,
        subdirectory: str | None = None,
    ) -> None:
        """Initialize the base contrail detection model.

        Args:
            detector: Geostationary satellite detector instance (e.g., GOESDetector,
                MeteoSatDetector) providing satellite data and metadata.
            detection_output_dir: Directory for storing detection outputs. If None, defaults
                to the detector's output path.
            subdirectory: Optional subdirectory name to append to detection_output_dir for
                organizing outputs by model type (e.g., 'contrails_api', 'google_api', 'local').
        """
        # Initialize attributes
        self.contrail_model_name: str = None
        self.detector: Geostationary = detector
        self.timestamps_processed: list[np.datetime64] = []
        self.timestamps_to_process: list[np.datetime64] = []

        # Setup output directory
        if detection_output_dir is None:
            base_path = self.detector.DETECT_PATH
        else:
            base_path = Path(detection_output_dir)

        if subdirectory is not None:
            self.detection_output_dir: Path = base_path / subdirectory
            self.detector.detection_subdirectory = subdirectory
        else:
            self.detection_output_dir: Path = base_path
            self.detector.detection_subdirectory = None

        # Validate detector compatibility
        self._validate_detector()

    def __repr__(self: Self) -> str:
        """Return string representation of the model instance.

        Returns:
            String representation including class name, detector type, and output directory.
        """
        return (
            f"{self.__class__.__name__}("
            f"detector={self.detector.__class__.__name__}, "
            f"detection_output_dir={self.detection_output_dir})"
        )

    @abstractmethod
    def run_detections(self: Self) -> None:
        """Obtain contrail detections for all timestamps.

        This method must be implemented by subclasses to perform the actual
        contrail detection using their specific methodology (e.g., API calls,
        local inference, loading from zarr store).

        """
        pass

    def _detections_to_zarr(
        self: Self,
        dataset_attrs: dict | None = None,
        zarr_path: str | Path | None = None,
        epoch_start: np.datetime64 | None = None,
        exclude_timestamps: set[np.datetime64] | None = None,
    ) -> None:
        """Take detection outputs and generates a Zarr store.

        M1/M2 sectors move between epochs, so each epoch is written to its own store, and a
        range spanning more than one is refused. See ``Geostationary.has_shifting_domain``
        and ``GOESDetector.spans_single_epoch``.

        Args:
            dataset_attrs: Optional dictionary of global attributes to include
                in the output dataset.
            zarr_path: Optional path to save the Zarr dataset. If None, defaults
                to ``data/processed_data``.
            epoch_start: Start of the observation epoch being written, used to namespace
                the store for M1/M2. Ignored for F and C. If None, the detector's first
                timestamp is used.
            exclude_timestamps: Detector timestamps to leave out of the store, e.g. frames
                whose request failed. If None, every detector timestamp is written, with NaN
                where no detection file exists.

        """
        if type(zarr_path) is str:
            zarr_path = Path(zarr_path)

        if getattr(self.detector, "has_shifting_domain", False):
            # Writing an archive whose frames span a repositioning bakes one grid into a
            # store describing two different places, with nothing downstream able to tell.
            # Refuse it: unlike plotting, this leaves a wrong artifact behind on disk.
            spans_one = getattr(self.detector, "spans_single_epoch", lambda: None)()
            if spans_one is False:
                raise ValueError(
                    f"{self.detector.satellite}-{self.detector.region} is repositioned "
                    f"between {self.detector.timestamp_sequence[0]} and "
                    f"{self.detector.timestamp_sequence[-1]}. Writing this range to one "
                    f"archive would describe frames from more than one sector position "
                    f"with a single grid. Split the range at the epoch boundaries and "
                    f"write one archive per epoch."
                )

            if epoch_start is None:
                epoch_start = self.detector.timestamp_sequence[0]
                logging.info(
                    f"No epoch_start given for {self.detector.satellite}-"
                    f"{self.detector.region}; using the detector's first timestamp "
                    f"{epoch_start}"
                )

        if self.detector.satellite not in ["GOES-16", "GOES-18", "GOES-19", "HIMAWARI-9", "MTG"]:
            raise ValueError(
                "detections_to_zarr is only implemented for detectors. "
                "[GOES-16, GOES-18, GOES-19, HIMAWARI-9, MTG]"
            )

        # Validate extent
        extent = self.detector.tile_extent
        validate_extent(
            extent=extent,
            min_longitude=self.detector.min_longitude,
            max_longitude=self.detector.max_longitude,
            min_latitude=self.detector.min_latitude,
            max_latitude=self.detector.max_latitude,
        )

        time_indices = list(range(len(self.detector.timestamp_strings)))
        if exclude_timestamps:
            excluded = {np.datetime64(ts, "s") for ts in exclude_timestamps}
            time_indices = [
                t_idx
                for t_idx in time_indices
                if np.datetime64(self.detector.timestamp_sequence[t_idx], "s") not in excluded
            ]

        # Load detection arrays from existing .npy files.
        available_files = {}  # Check which detection files exist
        for t_idx in time_indices:
            file_path = self.detector.get_npy_path(t_idx, self.detection_output_dir)
            if file_path.exists():
                available_files[t_idx] = file_path

        logging.info(
            f"Found detection files: {len(available_files)}/{len(self.detector.timestamp_strings)}"
        )
        if len(available_files) == 0:
            logging.error(f"Looking for files in: {self.detection_output_dir}")
            logging.error(
                f"First timestamp would be at: {self.detector.get_npy_path(0, self.detection_output_dir)}"
            )
            raise FileNotFoundError("No detection files found for any timestamp")

        # Determine sample shape from the first available file
        first_idx = sorted(available_files.keys())[0]
        sample_shape = np.load(available_files[first_idx]).shape

        detection_array = np.full(
            (len(time_indices), *sample_shape),
            np.nan,
            dtype=float,
        )
        time_stamps = [pd.to_datetime(self.detector.timestamp_strings[i]) for i in time_indices]
        position = {t_idx: pos for pos, t_idx in enumerate(time_indices)}

        # Minimum score persisted to the archive. The model emits a tiny positive value at
        # every on-disk pixel rather than a clean zero, so without this the store holds
        # billions of noise-floor values. See Geostationary.DEFAULT_STORAGE_THRESHOLD.
        storage_threshold = float(getattr(self.detector, "storage_threshold", 0.0) or 0.0)
        n_floored = 0

        # Fill available frames into the full array
        for t_idx, file_path in available_files.items():
            data = np.load(file_path)
            if storage_threshold > 0:
                # Floor finite values only: NaN marks off-disk pixels and gap filling
                # relies on it
                below = np.isfinite(data) & (data < storage_threshold)
                n_floored += int(below.sum())
                data = np.where(below, 0.0, data)
            detection_array[position[t_idx]] = data
            logging.debug(
                f"Loaded {file_path.name}, non-NaN: {np.isfinite(data).sum()}/{data.size}"
            )

        if storage_threshold > 0:
            n_total = detection_array.size
            logging.info(
                f"Storage threshold {storage_threshold}: floored {n_floored:,} of "
                f"{n_total:,} pixels to zero ({100 * n_floored / n_total:.2f}%)"
            )

        # Extract metadata from given attrs or create defaults
        if dataset_attrs is not None:
            ds_attrs = {
                "title": dataset_attrs.get("title", "Contrail Detection Data"),
                "authors": dataset_attrs.get("authors", None),
                "institution": dataset_attrs.get("institution", None),
                "source": dataset_attrs.get("source", f"{self.contrail_model_name} model"),
                "satellite": dataset_attrs.get("satellite", self.detector.satellite),
                "region": dataset_attrs.get("region", f"{self.detector.region}"),
                "extent": dataset_attrs.get("extent", extent),
                "description": dataset_attrs.get("description", None),
                "date_created": dataset_attrs.get("date_created", pd.Timestamp.now().isoformat()),
            }
        else:
            ds_attrs = {
                "title": "Contrail Detection Data",
                "source": f"{self.contrail_model_name} model",
                "satellite": self.detector.satellite,
                "region": f"{self.detector.region}",
                "extent": extent,
                "date_created": pd.Timestamp.now().isoformat(),
            }

        # Record the floor in both branches so a reader can tell what the store contains.
        ds_attrs["storage_threshold"] = storage_threshold
        ds_attrs["contrail_detect_version"] = __version__

        # The fixed-grid fields live in one shared store per satellite and region rather
        # than in every daily archive. Written first so they always exist alongside.
        # None for a shifting domain, whose grid fields stay in this archive
        static_path = self.detector.save_static_geometry(zarr_path)
        if static_path is not None:
            ds_attrs["static_geometry"] = str(static_path)

        zarr_path_full = self.detector.get_zarr_path(zarr_path, epoch_start=epoch_start)

        detection_ds = xr.Dataset(
            data_vars={
                "detection": (
                    ("time", "geo_y", "geo_x"),
                    detection_array,
                    {
                        "long_name": "Contrail detection score",
                        "units": "-",
                        "valid_range": [0.0, 1.0],
                        "description": (
                            "Model-predicted score of contrail presence. Scores below "
                            "storage_threshold are stored as 0.0; NaN marks pixels outside "
                            "the satellite disk or not imaged."
                        ),
                    },
                ),
                "scan_start_offset": (
                    "time",
                    self.detector.scan_start_offsets_s(time_stamps),
                    {
                        "long_name": "Scan start after time",
                        "units": "seconds",
                        "description": (
                            "A pixel's scan time is time + scan_start_offset + "
                            "scantime_offset (in the static geometry store, or in this "
                            "archive for Mesoscale). NaN where the frame's scan mode has no "
                            "time model."
                        ),
                    },
                ),
                # Fixed-grid fields live in the shared static geometry store
                # (Geostationary.STATIC_GEOMETRY_VARS)
            },
            coords={
                "time": (
                    "time",
                    time_stamps,
                    {
                        "long_name": "Image base timestamp",
                        "standard_name": "time",
                        "description": "Base timestamp when GOES starts scanning for image",
                    },
                ),
                "geo_x": (
                    "geo_x",
                    self.detector.image_geo_coords[0],
                    {
                        "long_name": "Satellite fixed grid projection x-coordinate",
                        "units": "meters",
                        "standard_name": "projection_x_coordinate",
                        "axis": "X",
                        "description": "Scanning from west to east under satellite perspective point",
                    },
                ),
                "geo_y": (
                    "geo_y",
                    self.detector.image_geo_coords[1],
                    {
                        "long_name": "Satellite fixed grid projection y-coordinate",
                        "units": "meters",
                        "standard_name": "projection_y_coordinate",
                        "axis": "Y",
                        "description": "Scanning from north to south under satellite perspective point",
                    },
                ),
            },
            attrs=ds_attrs,  # Global attributes
        )

        # Validate that coordinates are float64 before saving
        for coord in ["geo_x", "geo_y"]:
            if coord in detection_ds.coords:
                if detection_ds.coords[coord].dtype != np.float64:
                    raise TypeError(
                        f"Coordinate '{coord}' has dtype {detection_ds.coords[coord].dtype}, "
                        f"but must be float64. Fix the coordinate creation, "
                        f"don't convert at save time."
                    )

        if self.detector.has_shifting_domain:
            # No shared store for a domain that moves between epochs, so its fixed-grid
            # fields go in the archive
            detection_ds = detection_ds.assign(self.detector.static_geometry_fields())

        detection_ds.to_zarr(
            zarr_path_full,
            mode="w",
            encoding={"detection": detection_encoding(detection_array.shape)},
        )
        logging.info(f"Saved detection Zarr store to: {zarr_path_full}")

    def _save_detection_to_disk(
        self: Self,
        idx: int,
        detection_array: np.ndarray,
        **extra_metadata,
    ) -> None:
        """Save one detection array and its JSON metadata to the per-frame cache.

        Args:
            idx: Timestamp index in the detector sequence.
            detection_array: Detection mask array to persist.
            **extra_metadata: Additional metadata fields, e.g. the API source.
        """
        time_str = self.detector.timestamp_strings[idx]
        sat = self.detector.satellite
        reg = self.detector.region

        req_dir = self.detection_output_dir / f"{sat}-{reg}-{time_str}"
        req_dir.mkdir(parents=True, exist_ok=True)

        np.save(req_dir / "_prediction.npy", detection_array)

        metadata = {
            "extent": list(self.detector.tile_extent),
            "satellite": sat,
            "region": reg,
            "timestamp": time_str,
            "date_created": pd.Timestamp.now().isoformat(),
            **extra_metadata,
        }
        with open(req_dir / "_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _fill_missing_detections_with_nans(
        self: Self,
        detections_to_fill: dict[np.datetime64, np.ndarray],
    ) -> dict[np.datetime64, np.ndarray]:
        """Fill missing detection entries (None) with NaN arrays.

        This ensures that detections dictionary has the same length as the timestamp
        sequence, with missing frames represented as arrays of NaNs.
        """
        for timestamp in list(detections_to_fill.keys()):
            if detections_to_fill[timestamp] is None:
                logging.info(f"Filling missing detection for timestamp {timestamp} with NaN array.")

                # Create an array of NaNs with the same shape as other detections
                if self.xx is not None and self.yy is not None:
                    shape = self.xx.shape
                else:
                    # Fallback shape if coordinates are not set
                    logging.warning(
                        "Coordinates not set; using default shape from detector "
                        "for filling empty detections with NaNs."
                    )
                    shape = (
                        len(self.detector.image_geo_coords[1]),
                        len(self.detector.image_geo_coords[0]),
                    )

                detections_to_fill[timestamp] = np.full(
                    shape,
                    np.nan,
                    dtype=np.float32,
                )

        return detections_to_fill

    def _regrid_to_detector_coordinates(
        self: Self,
        source_mask: np.ndarray,
        source_x_coords: np.ndarray,
        source_y_coords: np.ndarray,
    ) -> np.ndarray:
        """Regrid detection mask from source coordinates to detector's native coordinate system.

        This method interpolates detection masks from external sources (the APIs)
        to match the detector's native geostationary coordinate grid. Uses bilinear
        interpolation appropriate for continuous probability scores.

        Args:
            source_mask: Detection mask at source resolution.
            source_x_coords: Source x-coordinates (geostationary projection, in meters).
            source_y_coords: Source y-coordinates (geostationary projection, in meters).

        Returns:
            Regridded mask matching detector's coordinate shape.

        Raises:
            ImportError: If scipy is not available.
        """
        # Get detector's native coordinates
        detector_x = self.detector.image_geo_coords[0]
        detector_y = self.detector.image_geo_coords[1]

        # Create interpolator for source mask (linear for continuous probability scores)
        interpolator = RegularGridInterpolator(
            (source_y_coords, source_x_coords),
            source_mask,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Create meshgrid of detector coordinates
        detector_xx, detector_yy = np.meshgrid(detector_x, detector_y)

        # Interpolate to detector grid
        points = np.column_stack([detector_yy.ravel(), detector_xx.ravel()])
        mask_regridded = interpolator(points).reshape(detector_yy.shape)

        return mask_regridded.astype(np.float32)

    def _validate_detector(self: Self) -> None:
        """Validate that the detector has all required attributes for detection operations.

        Raises:
            AttributeError: If detector is missing required attributes.
            ValueError: If detector attributes are not properly configured.
        """
        if not isinstance(self.detector, Geostationary):
            raise TypeError(
                f"Detector must be an instance of Geostationary, got {type(self.detector)} instead."
            )

        # Check required class attributes
        required_attrs = [
            "DETECT_PATH",
            "satellite",
            "region",
            "tile_extent",
            "lons",
            "lats",
            "timestamp_strings",
            "pixel_surface_area",
            "image_geo_coords",
            "scan_time_offset",
        ]

        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(self.detector, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            raise AttributeError(
                f"Detector {self.detector.__class__.__name__} is missing required "
                f"attributes: {', '.join(missing_attrs)}"
            )

        # Check required methods
        required_methods = ["get_npy_path", "get_zarr_path"]
        missing_methods = []
        for method in required_methods:
            if not hasattr(self.detector, method) or not callable(getattr(self.detector, method)):
                missing_methods.append(method)

        if missing_methods:
            raise AttributeError(
                f"Detector {self.detector.__class__.__name__} is missing required "
                f"methods: {', '.join(missing_methods)}"
            )

        # Check that critical attributes are not None
        if self.detector.timestamp_strings is None or len(self.detector.timestamp_strings) == 0:
            raise ValueError(
                "Detector timestamp_strings is None or empty. "
                "Please ensure detector has loaded timestamp data."
            )

        if self.detector.lons is None or self.detector.lats is None:
            raise ValueError(
                "Detector geographic coordinates (lons/lats) are None. "
                "Please ensure detector has loaded coordinate data."
            )

        if self.detector.image_geo_coords is None:
            raise ValueError(
                "Detector image_geo_coords is None. "
                "Please ensure detector has loaded projection coordinates."
            )

    def _refactor_segments(self: Self, df_segments: pd.DataFrame) -> pd.DataFrame:
        """Standardise a line-segment DataFrame to match ``decomposition.decompose`` output.

        Expects an input DataFrame with columns ``time``, ``lon0``, ``lat0``,
        ``lon1``, ``lat1``. Computes WGS84 geodesic geometry, scan-time offsets, and
        adds placeholder columns for fields not available from external sources.

        Args:
            df_segments: Raw segment DataFrame with endpoint coordinates and timestamps.

        Returns:
            Refactored DataFrame in Decomposer output format.
        """
        df_refactored = df_segments.copy()

        df_refactored["time"] = df_refactored.time.values.astype("datetime64[m]")
        df_refactored["originally_complex"] = False

        column_mapping = {
            "time": "timestamp",
            "lon0": "endpoint1_lon",
            "lat0": "endpoint1_lat",
            "lon1": "endpoint2_lon",
            "lat1": "endpoint2_lat",
        }
        df_refactored = df_refactored.rename(columns=column_mapping)

        coord_cols = ["endpoint1_lon", "endpoint1_lat", "endpoint2_lon", "endpoint2_lat"]
        valid_mask = df_refactored[coord_cols].notna().all(axis=1)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logging.warning(f"Filtering out {n_invalid} segments with NaN coordinates")
            df_refactored = df_refactored[valid_mask].reset_index(drop=True)

        # WGS84 geodesic, as in ``decomposition.decompose``
        lon1 = df_refactored["endpoint1_lon"].to_numpy(dtype=float)
        lat1 = df_refactored["endpoint1_lat"].to_numpy(dtype=float)
        lon2 = df_refactored["endpoint2_lon"].to_numpy(dtype=float)
        lat2 = df_refactored["endpoint2_lat"].to_numpy(dtype=float)

        forward_azimuth, _, distance_m = _transforms.geod.inv(lon1, lat1, lon2, lat2)
        df_refactored["great_circle_length_km"] = distance_m / 1000.0
        df_refactored["orientation"] = forward_azimuth

        # Centroid at the geodesic midpoint, longitude wrapped to [-180, 180)
        lon_mid, lat_mid, _ = _transforms.geod.fwd(lon1, lat1, forward_azimuth, distance_m / 2.0)
        df_refactored["centroid_lon"] = (np.asarray(lon_mid) + 180.0) % 360.0 - 180.0
        df_refactored["centroid_lat"] = lat_mid

        timestamps = pd.to_datetime(df_refactored["timestamp"])

        def scantime(point: str) -> np.ndarray:
            offsets = self.detector.scan_time_offset_for_position(
                df_refactored[f"{point}_lat"].to_numpy(dtype=float),
                df_refactored[f"{point}_lon"].to_numpy(dtype=float),
                timestamps.to_numpy(),
            )
            offsets = np.broadcast_to(np.asarray(offsets), (len(df_refactored),))
            return (timestamps + pd.to_timedelta(offsets).to_numpy()).to_numpy()

        for point in ("centroid", "endpoint1", "endpoint2"):
            df_refactored[f"{point}_scantime"] = scantime(point)

        for col in [
            "solidity",
            "eccentricity",
            "n_pixels",
            "area_km2",
            "width_km",
            "mean_score",
            "min_score",
            "max_score",
            "std_score",
        ]:
            df_refactored[col] = np.nan

        return df_refactored

    @staticmethod
    def display_detection_dates() -> None:
        """Display valid datetime ranges for detection sources in a formatted table."""
        name_w, type_w, sat_w, start_w, end_w, notes_w = 20, 15, 25, 18, 18, 45

        has_notes = any("Notes" in e for e in ContrailModelBase.DETECTION_DATES)

        print("\nValid datetime ranges for contrail detection sources:")

        header = (
            f"{'Name':<{name_w}} | {'Type':<{type_w}} | "
            f"{'Satellites':<{sat_w}} | {'Start':<{start_w}} | {'End':<{end_w}}"
        )
        if has_notes:
            header += f" | {'Notes':<{notes_w}}"

        print("\n" + header)
        print("-" * len(header))

        for entry in ContrailModelBase.DETECTION_DATES:
            d_type = entry.get("Type", "N/A")
            sats = ", ".join(entry.get("Satellites", []))
            start_str = pd.to_datetime(entry["Start Datetime"]).strftime("%Y-%m-%d %H:%M")
            if entry["End Datetime"] == ContrailModelBase.__FUTURE_BACKSTOP:
                end_str = "present"
            else:
                end_str = pd.to_datetime(entry["End Datetime"]).strftime("%Y-%m-%d %H:%M")

            row = (
                f"{entry['Name']:<{name_w}} | "
                f"{d_type:<{type_w}} | "
                f"{sats:<{sat_w}} | "
                f"{start_str:<{start_w}} | "
                f"{end_str:<{end_w}}"
            )
            if has_notes:
                row += f" | {entry.get('Notes', ''):<{notes_w}}"
            print(row)

        print("-" * len(header) + "\n")
