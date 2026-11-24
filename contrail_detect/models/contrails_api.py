"""
Contrail detection retrieval from the Contrails.org API.

Detections are generated using a single fold of the vit4 model from the
Kaggle Contrails solution implemented in `local_model.py`.

"""

import io
import logging
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import requests
import xarray as xr

from contrail_detect.core.config import CONTRAILS_API_HEADERS, CONTRAILS_API_URL_DETECTIONS
from contrail_detect.geostationary.goes import GOESDetector
from contrail_detect.models.contrail_model_base import (
    ContrailModelBase,
    get_with_retries,
    require_api_frame_times,
)


class ContrailsAPIContrailModel(ContrailModelBase):
    """Contrail detection retrieval from the Contrails.org API.

    Requests contrail detection predictions from the Contrails.org API for GOES
    satellite data, one timestamp at a time with retries, and caches them on disk to
    avoid repeat API calls.

    Attributes:
        detector: GOESDetector instance providing satellite data parameters.
        xx: X-coordinate meshgrid for detections.
        yy: Y-coordinate meshgrid for detections.
    """

    def __init__(
        self: Self,
        detector: GOESDetector,
        detection_output_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the ContrailsAPIContrailModel with detector and API parameters.

        Args:
            detector: GOES satellite detector instance (must be GOES-16/19 Full Disk).
            detection_output_dir: Directory for caching detection outputs. Defaults to
                detector's output path with 'contrails_api' subdirectory.
            timeout: API request timeout in seconds. Defaults to 120.0.
            max_retries: Maximum number of retry attempts for failed requests. Defaults to 3.

        Raises:
            ValueError: If detector is not GOES-16/19 or not Full Disk region.
        """
        # Validate detector requirements
        if detector.satellite not in ["GOES-16", "GOES-19"]:
            raise ValueError("Detector must be for GOES-16 or GOES-19 under current API outputs")
        if detector.region != "F":
            raise ValueError("Detector region must be 'F' (Full Disk) under current API outputs")

        super().__init__(
            detector=detector,
            detection_output_dir=detection_output_dir,
            subdirectory="contrails_api",
        )

        self.contrail_model_name = "Contrails.org API Contrail Model"

        # Store API-specific parameters
        self.timeout: float = timeout
        self.max_retries: int = max_retries

        # Placeholders for outputs
        self.xx, self.yy = None, None  # Meshgrids for coordinates

        # Grid directory path
        grid_dir = f"{self.detector.satellite}-{self.detector.region}-GRID"
        self.grid_path = self.detection_output_dir / grid_dir
        self.grid_path.mkdir(parents=True, exist_ok=True)

        # Verify the directory was created successfully
        if not self.grid_path.exists():
            raise RuntimeError(
                f"Failed to create grid directory: {self.grid_path}. "
                f"Check that {self.detection_output_dir} exists and is writable."
            )

    def __repr__(self: Self) -> str:
        """Return string representation of the ContrailsAPIContrailModel instance.

        Returns:
            String representation including detector type, API parameters, and output directory.
        """
        return (
            f"{self.__class__.__name__}("
            f"detector={self.detector.__class__.__name__}, "
            f"detection_output_dir={self.detection_output_dir})"
        )

    def request_detections(
        self: Self,
        force_reprocess: bool = False,
        zarr_path: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """Request contrail detections from API.

        Fetches detection predictions for all timestamps in the detector's sequence.
        Automatically loads existing cached files and only downloads missing detections.
        Results are returned as a dictionary and saved to disk for future use.

        Args:
            force_reprocess: Whether to ignore existing cached files and re-download
                all detections from the API.
            zarr_path: Optional path to zarr store. If None, uses default location.

        Returns:
            Dictionary mapping timestamps to detection arrays.

        Raises:
            ValueError: If any timestamp is off the API's 10-minute frame schedule. The
                API's response does not say which frame it is, so off-schedule times are
                refused rather than labelled with a frame they do not belong to.
        """
        require_api_frame_times(self.detector.timestamp_sequence, "Contrails.org API")

        # Check for cached data (zarr, .npy)
        if not force_reprocess:
            detection_dict, timestamps_found = self._load_cached_detections(
                zarr_path=zarr_path,
            )
        else:
            detection_dict, timestamps_found = {}, set()

        timestamps = self.detector.timestamp_sequence
        requested_timestamps = set(timestamps)

        # If all timestamps found in cache, return immediately
        if len(timestamps_found) == len(requested_timestamps):
            zarr_store_path = self.detector.get_zarr_path(zarr_path)
            if not zarr_store_path.exists():
                logging.info("Consolidating existing .npy files into zarr store")
                self._detections_to_zarr(zarr_path=zarr_path)

            return self._fill_missing_detections_with_nans(detection_dict)

        # If not all found, attempt to load from API and save results
        to_download_indices = [
            idx for idx, ts in enumerate(timestamps) if ts not in timestamps_found
        ]

        logging.info(f"Downloading {len(to_download_indices)} detections from API")

        for idx in to_download_indices:
            try:
                result = self._request_single_detection(timestamps[idx])

                # Process and extract array
                detection_array = np.array(result.values, copy=True)

                # Coordinate setup and regridding
                if self.xx is None or self.yy is None:
                    self.xx, self.yy = np.meshgrid(
                        result.x.values,
                        result.y.values,
                    )
                    np.save(str(self.grid_path / "_x_coords.npy"), self.xx)
                    np.save(str(self.grid_path / "_y_coords.npy"), self.yy)

                expected_shape = (
                    len(self.detector.image_geo_coords[1]),
                    len(self.detector.image_geo_coords[0]),
                )
                if detection_array.shape != expected_shape:
                    detection_array = self._regrid_to_detector_coordinates(
                        detection_array, result.x.values, result.y.values
                    )

                # Cache the individual result to disk
                detection_dict[np.datetime64(timestamps[idx])] = detection_array
                self._save_detection_to_disk(idx, detection_array)
                result.close()

            except requests.HTTPError as e:
                if e.response.status_code == 400:
                    logging.warning(f"Frame not found in API for {timestamps[idx]}")
                    detection_dict[np.datetime64(timestamps[idx])] = None
                else:
                    raise
            except Exception as e:
                logging.error(f"Failed to download {timestamps[idx]}: {e}")
                raise

        # Consolidate and return
        detection_dict = self._fill_missing_detections_with_nans(detection_dict)
        self._detections_to_zarr(zarr_path=zarr_path)

        return detection_dict

    def run_detections(
        self: Self, force_reprocess: bool = False, zarr_path: str | Path | None = None
    ) -> dict[np.datetime64, np.ndarray]:
        """Run contrail detection by requesting from API.

        Args:
            force_reprocess: Whether to ignore existing cached files and re-download
                all detections from the API.
            zarr_path: Optional path to zarr store. If None, uses default location.

        Returns:
            Dictionary mapping timestamps to detection arrays.
        """
        # Get detections via API with caching and zarr store creation
        return self.request_detections(
            force_reprocess=force_reprocess,
            zarr_path=zarr_path,
        )

    def _format_timestamp_for_api(
        self: Self,
        timestamp: np.datetime64,
    ) -> str:
        """Format timestamp for API request string.

        Args:
            timestamp: Input timestamp to format.

        Returns:
            Formatted timestamp string in ``'YYYY-MM-DDTHH:MM:SS'`` format. The detections
            endpoint expects a full ISO 8601 datetime on a 10 minute interval
            (e.g. ``'2024-04-12T18:10:00'``), so minutes must be preserved.
        """
        return pd.to_datetime(timestamp).strftime("%Y-%m-%dT%H:%M:%S")

    def _load_cached_detections(
        self, zarr_path: str | Path | None = None
    ) -> tuple[dict[np.datetime64, np.ndarray], set[np.datetime64]]:
        """Shared logic to attempt loading detections from Zarr or individual .npy files.

        Args:
            zarr_path: Optional path to zarr store. If None, uses default location.

        Returns:
            Tuple containing:
                - Dictionary mapping timestamps to detection arrays loaded from cache.
                - Set of timestamps that were successfully loaded from cache.
        """
        detection_dict = {}
        timestamps = self.detector.timestamp_sequence
        requested_timestamps = set(timestamps)
        timestamps_found = set()

        # Attempt load from zarr store
        zarr_store_path = self.detector.get_zarr_path(zarr_path)
        if zarr_store_path.exists():
            try:
                zarr_detections = self.detector.load_detections_from_zarr(zarr_path=zarr_path)

                # Only keep timestamps that are actually in requested sequence
                zarr_timestamps = set(zarr_detections.keys()) & requested_timestamps
                detection_dict.update({ts: zarr_detections[ts] for ts in zarr_timestamps})
                timestamps_found.update(zarr_timestamps)
                logging.info(f"Loaded {len(zarr_timestamps)}/{len(requested_timestamps)} from zarr")

                # Load coordinate grids if they exist
                if self.xx is None:
                    xx_path = self.grid_path / "_x_coords.npy"
                    if xx_path.exists():
                        self.xx = np.load(str(xx_path))
                if self.yy is None:
                    yy_path = self.grid_path / "_y_coords.npy"
                    if yy_path.exists():
                        self.yy = np.load(str(yy_path))
            except Exception as e:
                logging.warning(f"Zarr load failed: {e}")

        # Attempt .npy files for still missing
        timestamps_required = requested_timestamps - timestamps_found
        if timestamps_required:
            npy_count = 0
            for idx, ts in enumerate(timestamps):
                if ts in timestamps_required:
                    npy_path = self.detector.get_npy_path(idx, self.detection_output_dir)
                    if npy_path.exists():
                        detection_dict[ts] = np.load(npy_path)
                        timestamps_found.add(ts)
                        npy_count += 1
            if npy_count > 0:
                logging.info(f"Loaded {npy_count} from .npy files")

        return detection_dict, timestamps_found

    def _request_single_detection(
        self: Self,
        timestamp: np.datetime64,
    ) -> xr.DataArray:
        """Request the contrail detection mask for one timestamp, retrying transient errors.

        Args:
            timestamp: The input timestamp.

        Returns:
            xr.DataArray: An xarray DataArray containing contrail detections.

        Raises:
            requests.Timeout: If the request times out after all retries.
            requests.HTTPError: On a 4xx response, or a 5xx after all retries.
        """
        time_str = self._format_timestamp_for_api(timestamp)
        logging.info(f"fetching detections mask at: {time_str}")
        resp = get_with_retries(
            f"{CONTRAILS_API_URL_DETECTIONS}?time={time_str}",
            self.max_retries,
            self.timeout,
            headers=CONTRAILS_API_HEADERS,
        )
        return xr.open_dataarray(io.BytesIO(resp.content)).load()
