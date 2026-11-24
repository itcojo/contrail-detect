"""Abstract base class for geostationary satellite detectors.

``Geostationary`` holds the shared grid, timing, detection-loading, decomposition-output
and geolocation logic; each satellite subclass supplies its own data access.
"""

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
import xarray as xr
import zarr
from matplotlib.colors import Colormap, ListedColormap
from shapely.geometry import MultiPoint
from tqdm.auto import tqdm

from contrail_detect import __version__
from contrail_detect.core._plotting import (
    add_globe_inset,
    lon_range,
    max_pool,
    save_figure,
    unwrap_lons,
)
from contrail_detect.core.common import (
    Colors,
    Transforms,
)
from contrail_detect.core.config import DATA_DIR
from contrail_detect.geostationary import (
    geo_utils,
    validation_utils,
)

colors, transforms = Colors(), Transforms()

# Part of the decomposition output folder's fingerprint. Bump when a code change alters what
# decomposition writes, so results saved by older code are not reused.
_DECOMPOSITION_CACHE_VERSION = 2


class Geostationary(ABC):
    """Abstract base class for geostationary satellite-based contrail detection.

    This class provides common functionality for detecting and tracking contrails
    using data from geostationary satellites including GOES, Himawari, and Meteosat.
    It handles coordinate transformations, tile generation, and temporal sequencing.

    Attributes:
        lats: Latitude grid for the satellite image.
        lons: Longitude grid for the satellite image.
        region: Satellite scene region (e.g., ``'F'`` for full disk, ``'C'`` for CONUS).
        satellite: Name of the satellite (e.g., ``'GOES-16'``, ``'HIMAWARI-9'``).
        central_lon: Central longitude of the satellite in geostationary orbit.
        satellite_height: Height of the satellite above Earth's surface (meters).
        timestamp_sequence: List of timestamps for image acquisition times.
        timestamp_strings: String representations of timestamps.
        storage_threshold: Minimum detection score written to the archive. Scores below
            this are stored as 0.0; NaN (off-disk / not imaged) is left untouched. Set to
            0.0 to disable. See ``DEFAULT_STORAGE_THRESHOLD``.
    """

    # Archive scores below this are stored as 0.0, discarding the model's noise floor
    DEFAULT_STORAGE_THRESHOLD: float = 0.01

    # Default decomposition thresholds by detection source (detection_subdirectory), used when no
    # prediction_threshold is given. 0.30 for our local model is the recall-leaning choice of the
    # GOES-East paper; 0.40 for Google is where its own precision and recall cross.
    PREDICTION_THRESHOLDS: dict = {
        "local": 0.30,
        "contrails_api": 0.30,
        "google_api": 0.40,
    }
    # For detections with no recorded source, e.g. assigned directly
    DEFAULT_PREDICTION_THRESHOLD: float = 0.30

    # Full Disk maps: extent relative to the sub-satellite point, and longitude label spacing
    FULL_DISK_MAP_EXTENT: tuple = (-80.0, 80.0, -80.0, 80.0)
    MAP_LON_TICKS: np.ndarray = np.arange(-180, 181, 30)

    # Fixed-grid fields, identical across archives for a satellite and region.
    # Deduplicated into the shared store — see save_static_geometry.
    STATIC_GEOMETRY_VARS: tuple = (
        "latitude",
        "longitude",
        "pixel_area",
        "scantime_offset",
    )

    # Regions repositioned between observation epochs, where the region name alone does not
    # identify a place on the Earth. Geometry must not be shared across epochs for these.
    SHIFTING_DOMAIN_REGIONS: tuple = ("M1", "M2")

    # True where scantime_offset changed meaning in 1.13.0 (nominal time -> scan start), so
    # stores and archives from before then mustn't be mixed with newer ones
    SCANTIME_FROM_SCAN_START: bool = False

    # Dates each satellite can be used for.
    # - GOES: operational periods, when each satellite was at the position and image rate
    #   the code assumes. Data exists earlier, but from post-launch testing at 89.5°W (wrong
    #   position) or, for GOES-16 before 2019-04-02 16:00 UTC, with Full Disk every 15 min
    #   rather than 10. GOES-16 → GOES-19 handover as GOES-East: 2025-04-07; GOES-18 declared
    #   GOES-West: 2023-01-04 18:00 UTC.
    #   https://www.goes-r.gov/users/abiScanModeInfo.html
    #   https://vlab.noaa.gov/web/towr-s/goes-18-timeline
    #   https://www.goes-r.gov/users/transitionToOperations19.html
    # - HIMAWARI SOURCE: https://registry.opendata.aws/noaa-himawari/
    __FUTURE_BACKSTOP = np.datetime64("2100-01-01T00:00")
    SATELLITE_DATES: dict = {
        "GOES-16": [
            np.datetime64("2019-04-02T16:00"),
            np.datetime64("2025-04-07T00:00"),
        ],
        "GOES-18": [
            np.datetime64("2023-01-04T18:00"),
            __FUTURE_BACKSTOP,
        ],
        "GOES-19": [
            np.datetime64("2025-04-07T00:00"),
            __FUTURE_BACKSTOP,
        ],
        "HIMAWARI-8": [
            np.datetime64("2015-07-07T02:00"),
            np.datetime64("2022-12-13T04:50"),
        ],
        "HIMAWARI-9": [
            np.datetime64("2022-11-04T15:50"),
            __FUTURE_BACKSTOP,
        ],
        "MTG": [
            np.datetime64("2025-08-26T07:00"),
            __FUTURE_BACKSTOP,
        ],
    }

    def __init__(self: Self) -> None:
        """Initialize the Geostationary class with default attributes."""
        # Standard attributes
        self.lats: np.ndarray = None
        self.lons: np.ndarray = None
        self.region: str = None
        self.satellite: str = None
        self.LONG_NAME: str = None

        # Satellite orbital properties
        self.central_lon: float = None

        # Minimum longitudes and latitudes suitable for detections
        self.min_longitude: float = None
        self.max_longitude: float = None
        self.min_latitude: float = None
        self.max_latitude: float = None

        # Initialize empty timestamp sequence and image cache
        self.timestamp_sequence: list[np.datetime64] = []
        self.timestamp_strings: list[str] = []

        # Initialize empty tile constructs and tile extent
        self.tile_lons: np.ndarray = None
        self.tile_lats: np.ndarray = None
        self.tile_idx: np.ndarray = None
        self.tile_extent: np.ndarray = None
        self.image_offset: tuple[int, int] = (0, 0)

        # min_area and DecompositionConfig fields besides stitching, used in the decomposition
        # output folder name. None means the defaults.
        self.decomposition_settings: dict | None = None

        # Initialize empty objects related to detection, decomposition, tracking
        self.detections: dict[np.datetime64, np.ndarray] = {}
        self.contrail_summary: pd.DataFrame = pd.DataFrame()
        self.contrail_point_clouds: dict[str, np.ndarray] = {}
        self.contrail_geo_point_clouds: dict[str, MultiPoint] = {}

        # Initialize detection output subdirectory tracking
        self.detection_subdirectory: str | None = None

        # Minimum detection score persisted to the archive (see DEFAULT_STORAGE_THRESHOLD)
        self.storage_threshold: float = self.DEFAULT_STORAGE_THRESHOLD

    def __repr__(self: Self) -> str:
        """Return string representation of the detector."""
        fields = ", ".join(f"{k}={v}" for k, v in self._repr_fields().items())
        return f"{type(self).__name__}({fields})"

    def _repr_fields(self: Self) -> dict:
        """Fields shown by ``__repr__``. Subclasses extend this to add their own."""
        fields = {"satellite": self.satellite, "region": self.region}
        if self.timestamp_sequence:
            fields["start_time"] = self.timestamp_sequence[0]
            fields["end_time"] = self.timestamp_sequence[-1]
        return fields

    @property
    @abstractmethod
    def image_full_extent(self: Self) -> tuple[float, float, float, float]:
        """Return the image extent in geostationary coordinates.

        Returns:
            Tuple of left, right, bottom, top in geostationary coordinates.
        """
        pass

    @property
    @abstractmethod
    def image_geo_coords(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """Return the geostationary x and y coordinates of the image pixels.

        Returns:
            Tuple of numpy arrays (x, y)
        """
        pass

    @property
    @abstractmethod
    def sat_pyproj_crs(self: Self) -> pyproj.CRS:
        """Get the satellite's pyproj CRS object.

        Returns:
            Pyproj CRS object representing the satellite's coordinate reference system.

        """
        pass

    @abstractmethod
    def _gen_scan_time_offset(self: Self) -> np.ndarray:
        """Calculate scan time offsets for each pixel in the satellite image.

        Computes the actual scan time for each pixel relative to the nominal
        image timestamp, accounting for the satellite's scanning pattern.

        Returns:
            Array of scan time offsets for each pixel in the grid (seconds).
        """
        pass

    @abstractmethod
    def _gen_tiles(self: Self) -> None:
        """Generate spatial tiles for the satellite region.

        Abstract method to create a grid of spatial tiles covering the satellite's
        field of view. Implementation varies by satellite type and region.
        """
        pass

    @abstractmethod
    def _set_central_lon(self: Self) -> None:
        """Set the central longitude for the satellite's geostationary position.

        Abstract method to define the satellite's central longitude based on its
        orbital position. Implementation varies by satellite.
        """
        pass

    @cached_property
    def ecef_crs(self: Self) -> pyproj.CRS:
        """Get the ECEF (Earth-Centered Earth-Fixed) coordinate system.

        Returns:
            Pyproj CRS object for ECEF (EPSG:4978) coordinate system.
        """
        return pyproj.CRS("EPSG:4978")

    @cached_property
    def ecef_to_wgs84_transformer(self: Self) -> pyproj.Transformer:
        """Get transformer from ECEF to WGS84 coordinates.

        Used for parallax correction to convert Earth-centered Cartesian
        coordinates back to geographic coordinates.

        Returns:
            Pyproj Transformer object for ECEF to WGS84 conversion.
        """
        return pyproj.Transformer.from_crs(
            self.ecef_crs,
            self.wgs84_crs,
            always_xy=True,
        )

    @property
    def has_shifting_domain(self: Self) -> bool:
        """True for regions that move between epochs (M1/M2); see ``SHIFTING_DOMAIN_REGIONS``."""
        return self.region in self.SHIFTING_DOMAIN_REGIONS

    @cached_property
    def latlon_pixel_corners(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate and cache the four corner coordinates (lat, lon) for each pixel.

        Uses the satellite's native projection and transforms corner points to
        geographic coordinates. Corners are ordered counter-clockwise starting
        from bottom-left. Result is cached for efficient reuse.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - corner_lons: Array of shape (ny, nx, 4) with corner longitudes
                - corner_lats: Array of shape (ny, nx, 4) with corner latitudes
                Corner order: [bottom-left, bottom-right, top-right, top-left]
        """
        # Get geostationary coordinates
        geo_x, geo_y = self.image_geo_coords

        # Calculate half-pixel spacing in geostationary coordinates
        dx = (geo_x[1] - geo_x[0]) / 2
        dy = (geo_y[1] - geo_y[0]) / 2

        # Calculate corner offsets
        corner_offsets = np.array(
            [
                [-dx, -dy],  # bottom-left
                [dx, -dy],  # bottom-right
                [dx, dy],  # top-right
                [-dx, dy],  # top-left
            ]
        )

        # Create meshgrid for pixel centers
        xx, yy = np.meshgrid(geo_x, geo_y, indexing="xy")

        # Stack all corner coordinates for batch transformation
        # Shape: (4, ny, nx) for both x and y
        ny, nx = xx.shape
        corner_x_all = xx[None, :, :] + corner_offsets[:, 0, None, None]  # (4, ny, nx)
        corner_y_all = yy[None, :, :] + corner_offsets[:, 1, None, None]  # (4, ny, nx)

        # Flatten for batch transformation: (4*ny*nx,)
        corner_x_flat = corner_x_all.ravel()
        corner_y_flat = corner_y_all.ravel()

        # Single transformation call for all corners using cached transformer
        lons_flat, lats_flat = self.sat_to_latlon_transform.transform(corner_x_flat, corner_y_flat)

        # Handle invalid points
        lons_flat[np.isinf(lons_flat)] = np.nan
        lats_flat[np.isinf(lats_flat)] = np.nan

        # Reshape back to (4, ny, nx) then transpose to (ny, nx, 4)
        corner_lons = lons_flat.reshape(4, ny, nx).transpose(1, 2, 0)
        corner_lats = lats_flat.reshape(4, ny, nx).transpose(1, 2, 0)

        return corner_lons, corner_lats

    @cached_property
    def latlon_to_sat_transform(self: Self) -> pyproj.Transformer:
        """Get transformer from lat/lon to satellite coordinates.

        Returns:
            Pyproj Transformer object for converting geographic latitude and
            longitude to satellite coordinates.
        """
        return pyproj.Transformer.from_crs(
            crs_from=pyproj.CRS("WGS84"),
            crs_to=self.sat_pyproj_crs,
            always_xy=True,
        )

    @cached_property
    def pixel_surface_area(self: Self) -> np.ndarray:
        """Calculate and cache the surface area of each pixel in square kilometers.

        Uses the corner coordinates of each pixel to compute the area on the
        Earth's surface, accounting for pixel shape distortion due to POV.

        Returns:
            Array of pixel surface areas in square kilometers.
        """
        corner_lons, corner_lats = self.latlon_pixel_corners

        area = np.zeros(corner_lats.shape[:-1])

        # Calculate area of triangle 1 (BL, BR, TR)
        area += geo_utils.latlon_triangle_area(
            corner_lats[..., 0],
            corner_lons[..., 0],
            corner_lats[..., 1],
            corner_lons[..., 1],
            corner_lats[..., 2],
            corner_lons[..., 2],
        )

        # Calculate area of triangle 2 (BL, TR, TL)
        area += geo_utils.latlon_triangle_area(
            corner_lats[..., 0],
            corner_lons[..., 0],
            corner_lats[..., 2],
            corner_lons[..., 2],
            corner_lats[..., 3],
            corner_lons[..., 3],
        )

        return area

    @cached_property
    def sat_to_latlon_transform(self: Self) -> pyproj.Transformer:
        """Get transformer from satellite coordinates to lat/lon.

        Returns:
            Pyproj Transformer object for converting satellite coordinates to
            geographic latitude and longitude.
        """
        return pyproj.Transformer.from_crs(
            crs_from=self.sat_pyproj_crs,
            crs_to=pyproj.CRS("WGS84"),
            always_xy=True,
        )

    @cached_property
    def scan_time_offset(self: Self) -> np.ndarray:
        """Calculate and cache scan time offsets for each pixel in the image.

        Computes the actual scan time for each pixel, accounting for the satellite's
        scanning pattern. The result is cached for efficient reuse.

        Returns:
            Array of scan time offsets for each pixel in the grid (seconds).
        """
        return self._gen_scan_time_offset()

    @cached_property
    def wgs84_crs(self: Self) -> pyproj.CRS:
        """Get the WGS84 geographic coordinate system.

        Returns:
            Pyproj CRS object for WGS84 (EPSG:4326) coordinate system.
        """
        return pyproj.CRS("EPSG:4326")

    @cached_property
    def wgs84_to_ecef_transformer(self: Self) -> pyproj.Transformer:
        """Get transformer from WGS84 to ECEF coordinates.

        Used for parallax correction to convert geographic coordinates
        to Earth-centered Cartesian coordinates.

        Returns:
            Pyproj Transformer object for WGS84 to ECEF conversion.
        """
        return pyproj.Transformer.from_crs(
            self.wgs84_crs,
            self.ecef_crs,
            always_xy=True,
        )

    def create_timestamp_sequence(
        self: Self,
        start_time: np.datetime64,
        end_time: np.datetime64,
    ) -> None:
        """Generate sequence of timestamps at the satellite's image interval.

        Creates a list of the satellite's image times from start_time to end_time
        inclusive, spaced by its image interval. Also validates timestamps against
        satellite operational dates.

        Images are taken on a fixed schedule from midnight UTC (e.g. every 10 minutes at
        :00, :10, ...), so start_time must be one of those times: an image time that is
        not on the schedule does not exist, and data stored under it would carry the wrong
        time.

        Args:
            start_time: Start of the time range, on the satellite's image schedule.
            end_time: End of the time range. The last image is the latest one at or
                before end_time.

        Raises:
            TypeError: If start_time or end_time are not np.datetime64 objects.
            ValueError: If end_time is before start_time, or start_time is not on the
                satellite's image schedule.
        """
        if not isinstance(start_time, np.datetime64):
            raise TypeError("start_time must be a np.datetime64 object.")

        if not isinstance(end_time, np.datetime64):
            raise TypeError("end_time must be a np.datetime64 object.")

        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time.")

        # Source image interval from the specific satellite instance
        interval_s = int(round(self.image_interval * 60.0))
        delta = np.timedelta64(interval_s, "s")

        # The start must be an image time: a whole number of intervals after midnight UTC
        start = pd.Timestamp(start_time)
        seconds_after_midnight = (start - start.normalize()).total_seconds()
        if seconds_after_midnight % interval_s != 0:
            schedule = pd.Timedelta(seconds=interval_s)
            raise ValueError(
                f"{self.satellite} region {self.region} takes an image every "
                f"{self.image_interval:g} minutes from midnight UTC, so there is no image at "
                f"{start:%Y-%m-%dT%H:%M:%S}. Start on an image time, e.g. "
                f"{start.floor(schedule):%Y-%m-%dT%H:%M:%S} or "
                f"{start.ceil(schedule):%Y-%m-%dT%H:%M:%S}."
            )

        # Image times from start_time up to and including end_time, never past it
        self.timestamp_sequence = [
            ts for ts in np.arange(start_time, end_time + delta, delta) if ts <= end_time
        ]

        # Check the sequence of timestamps for validity
        validation_utils.validate_timestamps(
            self.timestamp_sequence,
            self.satellite,
            self.SATELLITE_DATES,
        )

        # Corresponding timestamp strings, used in file names and read back as times.
        # Seconds are included only when the image interval needs them (e.g. 2.5 minutes),
        # so existing whole-minute names are unchanged.
        time_format = "%Y-%m-%dT%H%M" if interval_s % 60 == 0 else "%Y-%m-%dT%H%M%S"
        self.timestamp_strings = [
            pd.Timestamp(ts).strftime(time_format) for ts in self.timestamp_sequence
        ]

    def gen_detector_long_name(self: Self) -> str:
        """Generate descriptive name for the detector.

        Creates a name string including satellite, region, and temporal information
        in the format: SATELLITE_REGION_STARTTIME_ENDTIME.

        Returns:
            Descriptive name string for the detector.
        """
        min_time = np.min(self.timestamp_sequence)
        max_time = np.max(self.timestamp_sequence)
        return (
            f"{self.satellite.upper()}_{self.region.upper()}_"
            f"{pd.to_datetime(min_time).strftime('%Y%m%dT%H%M')}_"
            f"{pd.to_datetime(max_time).strftime('%Y%m%dT%H%M')}"
        )

    def get_geo_point_clouds(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        force_regenerate: bool = False,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> dict[str, MultiPoint]:
        """Get geo point clouds, loading from file or generating from ``self.contrail_point_clouds``.

        If a GeoJSON file already exists and force_regenerate is False, loads from
        disk. Otherwise converts ``self.contrail_point_clouds`` to lon/lat geometries
        using ``self.contrail_summary`` for timestamps, then saves the result.

        Args:
            contrail_output_dir: Directory for geo point cloud files. If None,
                uses default location.
            method: Subdirectory corresponding to the source method (e.g., 'decomposer'),
                matching the method used in ``get_decomposed_contrails()``. If None, no
                subdirectory is added.
            force_regenerate: If True, regenerate even if a cached file exists.
                Default is False.
            contrail_stitching: For the 'decomposer' method, whether the contrails came
                from a stitched run. Defaults to True, as in ``get_decomposed_contrails``.
            prediction_threshold: For the 'decomposer' method, the threshold the contrails
                were decomposed at. If None, uses the default for the detection source.

        Returns:
            Dictionary mapping contrail IDs to MultiPoint geometries in lon/lat.

        Raises:
            ValueError: If ``self.contrail_point_clouds`` is empty and no cached file exists.
        """
        if not force_regenerate:
            try:
                return self._load_geo_point_clouds(
                    contrail_output_dir, method, contrail_stitching, prediction_threshold
                )
            except FileNotFoundError:
                logging.info("No cached geo point clouds found, generating...")

        if not self.contrail_point_clouds:
            raise ValueError(
                "self.contrail_point_clouds is empty. Load point clouds via "
                "get_decomposed_contrails() before calling get_geo_point_clouds()."
            )

        self._geo_point_clouds_from_point_clouds(self.contrail_point_clouds)
        self._save_geo_point_clouds(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )
        return self.contrail_geo_point_clouds

    def get_grid_index_from_position(
        self: Self,
        lon: float | np.ndarray,
        lat: float | np.ndarray,
        geo_x: np.ndarray | None = None,
        geo_y: np.ndarray | None = None,
    ) -> tuple[int, int] | tuple[np.ndarray, np.ndarray]:
        """Get grid indices corresponding to geographic positions.

        Converts geographic coordinates (longitude, latitude) to pixel indices
        in the satellite image grid using inverse projection transformation.

        Args:
            lon: Longitude coordinate(s) in degrees.
            lat: Latitude coordinate(s) in degrees.
            geo_x: Optional geostationary x-coordinates array.
            geo_y: Optional geostationary y-coordinates array.

        Returns:
            Tuple of (i, j) grid indices where i is row index and j is column index.
            Returns -1 for points off the Earth's disk or outside the image. Returns
            scalar tuple for scalar inputs, array tuple for array inputs.
        """
        # Convert inputs to numpy arrays for consistent processing
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        # Get projection and coordinate info
        if geo_x is None or geo_y is None:
            sat_x, sat_y = self.image_geo_coords
        else:
            sat_x, sat_y = geo_x, geo_y

        # Transform lat/lon to geostationary x/y coordinates
        x_geo, y_geo = self.latlon_to_sat_transform.transform(lon, lat)

        # The pixel containing each position. The grid is evenly spaced and its coordinates
        # are pixel centres, so a pixel's edges lie half a step either side of its centre:
        # the containing pixel is the centre's index rounded from the position. The step's
        # sign handles descending axes (y runs north to south).
        def _containing_pixel(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
            step = (coords[-1] - coords[0]) / (len(coords) - 1)
            with np.errstate(invalid="ignore"):
                index = np.rint((values - coords[0]) / step)
            index = np.nan_to_num(index, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(index, 0, len(coords) - 1).astype(int)

        j_indices = _containing_pixel(np.asarray(sat_x), np.asarray(x_geo))  # column
        i_indices = _containing_pixel(np.asarray(sat_y), np.asarray(y_geo))  # row

        # Handle points outside valid projection area (off the Earth's disk), and points
        # beyond the image's edges (for images covering part of the disk, e.g. CONUS or
        # mesoscale): more than half a pixel past the outermost pixel centres
        half_dx = np.median(np.abs(np.diff(sat_x))) / 2
        half_dy = np.median(np.abs(np.diff(sat_y))) / 2
        with np.errstate(invalid="ignore"):
            outside_image = (
                (x_geo < np.min(sat_x) - half_dx)
                | (x_geo > np.max(sat_x) + half_dx)
                | (y_geo < np.min(sat_y) - half_dy)
                | (y_geo > np.max(sat_y) + half_dy)
            )
        invalid_mask = ~np.isfinite(x_geo) | ~np.isfinite(y_geo) | outside_image
        if np.any(invalid_mask):
            j_indices = np.where(invalid_mask, -1, j_indices)
            i_indices = np.where(invalid_mask, -1, i_indices)

        # Return scalar values if input was scalar
        if i_indices.shape == ():
            return int(i_indices), int(j_indices)
        return i_indices, j_indices

    def get_npy_path(
        self: Self,
        t_idx: int,
        detection_output_dir: str | Path | None = None,
    ) -> Path:
        """Get the file path for a detection output file.

        Constructs the standardized path for storing detection results based on
        the detector's metadata and timestamp. Includes detection_subdirectory if set
        to namespace outputs by model type (e.g., 'contrails_api', 'google_api', 'local').

        Args:
            t_idx: Index of the timestamp in the detector's sequence.
            detection_output_dir: Directory containing detection outputs. If None, uses
                detector's default output path with detection_subdirectory. If provided,
                assumes it already includes the subdirectory path.

        Returns:
            Path to the detection file for the specified timestamp.
        """
        if detection_output_dir is None:
            detection_output_dir = self.DETECT_PATH
            # Include subdirectory only if using default path and detection_subdirectory is set
            if self.detection_subdirectory is not None:
                detection_output_dir = detection_output_dir / self.detection_subdirectory
        else:
            detection_output_dir = Path(detection_output_dir)

        time_str = self.timestamp_strings[t_idx]
        req_fmt = f"{self.satellite}-{self.region}-{time_str}"
        req_dir = detection_output_dir / req_fmt
        return req_dir / "_prediction.npy"

    def get_static_geometry_path(
        self: Self,
        zarr_path: str | Path | None = None,
    ) -> Path:
        """Path to the shared per-satellite static geometry store.

        The store sits at ``{root}/{satellite}/{region}/static_geometry.zarr`` — above the
        model-backend subdirectory, because the grid is a property of the satellite and
        region rather than of any detection model, and is therefore shared by all backends.

        Args:
            zarr_path: Optional archive root. Defaults to ``data/processed_data``.

        Returns:
            Path to the static geometry store (which may not exist yet).
        """
        if zarr_path is None:
            zarr_path = DATA_DIR / "processed_data"
        zarr_path = Path(zarr_path)
        return zarr_path / f"{self.satellite}" / f"{self.region}" / "static_geometry.zarr"

    def get_zarr_path(
        self: Self,
        zarr_path: str | Path | None = None,
        epoch_start: np.datetime64 | None = None,
    ) -> Path:
        """Get the full path to the zarr store.

        Constructs path including detection_subdirectory if set to namespace zarr stores
        by model type (e.g., 'contrails_api', 'google_api', 'local').

        For mesoscale regions (M1, M2) that reposition between epochs, each epoch's
        detections go in a separate subdirectory, preventing coordinate metadata from one
        epoch overwriting another.

        Args:
            zarr_path: Base path to zarr store. If None, uses default location.
            epoch_start: Start timestamp of this observation epoch, for M1 and M2: a
                subdirectory named after it (``YYYY-MM-DDTHHMM``) is inserted before the
                zarr filename. Defaults to the first timestamp, as the archive writer does,
                so readers find what was written.

        Returns:
            Full path to the zarr store.
        """
        if zarr_path is None:
            zarr_path = DATA_DIR / "processed_data"
        elif isinstance(zarr_path, str):
            zarr_path = Path(zarr_path)

        # Build path with subdirectory (only include if set)
        zarr_path_full = zarr_path / f"{self.satellite}" / f"{self.region}"
        if self.detection_subdirectory is not None:
            zarr_path_full = zarr_path_full / self.detection_subdirectory

        # Add long name (only include if set)
        if self.LONG_NAME is not None:
            zarr_path_full = zarr_path_full / self.LONG_NAME

        # For mesoscale regions, insert an epoch subdirectory so that each domain
        # position gets its own zarr store with self-consistent geo_x/geo_y metadata.
        if self.region in ["M1", "M2"]:
            if epoch_start is None:
                epoch_start = self.timestamp_sequence[0]
            epoch_str = pd.Timestamp(epoch_start).strftime("%Y-%m-%dT%H%M")
            zarr_path_full = zarr_path_full / epoch_str

        zarr_path_full = zarr_path_full / "detection_archive.zarr"

        return zarr_path_full

    def load_detections(
        self: Self,
        zarr_path: str | Path | None = None,
        detection_output_dir: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """Load detection outputs from zarr store or .npy files.

        Attempts to load from zarr store first for efficiency. If zarr doesn't exist
        or is incomplete, loads from individual .npy files. Returns all available
        detections for the detector's timestamp sequence.

        Args:
            zarr_path: Path to zarr store. If None, uses default location.
            detection_output_dir: Directory containing .npy detection outputs. If None, uses
                detector's default output path.

        Returns:
            Dictionary mapping timestamps to detection arrays for available timestamps.
        """
        requested_timestamps = set(self.timestamp_sequence)
        loaded_detections: dict[np.datetime64, np.ndarray] = {}

        # Try loading from zarr store first
        zarr_store_path = self.get_zarr_path(zarr_path)
        logging.info(f"Checking for zarr store at: {zarr_store_path}")
        if zarr_store_path.exists():
            logging.info("Loading detections from zarr store...")
            try:
                zarr_detections = self.load_detections_from_zarr(zarr_path)
                zarr_timestamps = set(zarr_detections.keys()) & requested_timestamps
                loaded_detections.update({ts: zarr_detections[ts] for ts in zarr_timestamps})
                logging.info(
                    f"Loaded {len(zarr_timestamps)}/{len(requested_timestamps)} "
                    f"timestamps from zarr store."
                )
            except ValueError:
                # Raised on purpose, e.g. the archive covers a different extent: loading
                # nothing in its place would look like "no contrails"
                raise
            except Exception as e:
                logging.warning(f"Failed to load from zarr store: {e}")
        # Load any missing timestamps from .npy files
        timestamps_found = set(loaded_detections.keys())
        timestamps_missing = requested_timestamps - timestamps_found

        if timestamps_missing:
            logging.info(f"Loading {len(timestamps_missing)} missing timestamps from .npy files...")
            npy_detections = self.load_detections_from_npy(detection_output_dir)
            npy_timestamps = set(npy_detections.keys()) & timestamps_missing
            loaded_detections.update({ts: npy_detections[ts] for ts in npy_timestamps})
            logging.info(f"Loaded {len(npy_timestamps)} timestamps from .npy files.")

        total_loaded = len(loaded_detections)
        logging.info(f"Total loaded: {total_loaded}/{len(requested_timestamps)} timestamps.")

        missing_timestamps = requested_timestamps - set(loaded_detections.keys())
        if missing_timestamps:
            logging.warning(
                f"Missing timestamps: {len(missing_timestamps)} out of {len(requested_timestamps)}, "
                f"recommend running detections for these timestamps."
            )

        return loaded_detections

    def load_detections_from_npy(
        self: Self,
        detection_output_dir: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """Load detection outputs from individual .npy files.

        Args:
            detection_output_dir: Directory containing detection outputs. If None, uses
                detector's default output path.

        Returns:
            Dictionary mapping timestamps to detection arrays for available files.

        Raises:
            ValueError: If extent metadata exists and doesn't match the detector's extent.
        """
        # Check for extent metadata file in detection directories
        metadata_found = False
        for idx, _ in enumerate(self.timestamp_strings):
            npy_path = self.get_npy_path(idx, detection_output_dir)
            if npy_path.exists():
                # Check for metadata file in the same directory
                metadata_path = npy_path.parent / "_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    if "extent" in metadata:
                        stored_extent = tuple(metadata["extent"])

                        if not geo_utils.extents_match(stored_extent, self.tile_extent):
                            raise ValueError(
                                f"Extent mismatch: detection files have extent {stored_extent}, "
                                f"but detector is configured with extent {self.tile_extent}. "
                                f"Please ensure the detector's extent matches the saved detection data."
                            )
                        logging.info(f"Extent validation passed: {self.tile_extent}")
                        metadata_found = True
                        break

        if not metadata_found:
            logging.warning(
                "No metadata file found in detection directories. "
                "Cannot validate extent consistency for .npy files."
            )

        # --- --- ---
        # Load detection arrays
        loaded_detections: dict[np.datetime64, np.ndarray] = {}
        for idx, time_str in enumerate(self.timestamp_strings):
            ts = np.datetime64(pd.to_datetime(time_str), "s")  # Normalize to second precision
            npy_path = self.get_npy_path(idx, detection_output_dir)

            if npy_path.exists():
                loaded_detections[ts] = np.load(npy_path)

        return loaded_detections

    def load_detections_from_zarr(
        self: Self,
        zarr_path: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """Load detection outputs from a Zarr dataset.

        Args:
            zarr_path: Path to the Zarr dataset containing detection outputs.

        Returns:
            Dictionary mapping timestamps to detection arrays.

        Raises:
            ValueError: If the extent in the zarr file doesn't match the detector's extent.
        """
        # Initialize empty detections dictionary
        loaded_detections: dict[np.datetime64, np.ndarray] = {}

        # Get full path to zarr store
        zarr_path_full = self.get_zarr_path(zarr_path)

        # Lazy load Zarr dataset
        detection_ds = xr.open_zarr(
            zarr_path_full,
            chunks={},
            decode_timedelta=True,
        )

        # Validate extent matches
        if "extent" in detection_ds.attrs:
            stored_extent = detection_ds.attrs["extent"]
            # Convert to tuple if it's a list or array
            if isinstance(stored_extent, (list, np.ndarray)):
                stored_extent = tuple(stored_extent)

            # Check if extents match within tolerance
            if not geo_utils.extents_match(stored_extent, self.tile_extent):
                raise ValueError(
                    f"Extent mismatch: zarr file has extent {stored_extent}, "
                    f"but detector is configured with extent {self.tile_extent}. "
                    f"Please ensure the detector's extent matches the saved detection data."
                )
        else:
            logging.warning(
                "No extent metadata found in zarr file. Cannot validate extent consistency."
            )

        # Extract detection data into dictionary
        # Zarr stores cropped data - need to re-project onto full detector grid
        for t_idx, time_val in enumerate(detection_ds["time"].values):
            detection_cropped = detection_ds["detection"].isel(time=t_idx).values
            zarr_geo_x = detection_ds["geo_x"].values
            zarr_geo_y = detection_ds["geo_y"].values

            detection_full = self._reproject_cropped_to_full_grid(
                detection_cropped, zarr_geo_x, zarr_geo_y
            )
            # Normalize timestamp to second precision for consistency with timestamp_sequence
            ts_normalized = np.datetime64(time_val, "s")
            loaded_detections[ts_normalized] = detection_full

        return loaded_detections

    def load_decomposed_contrails(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        source_method: str = "decomposer",
        read_simple: bool = False,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
        """Load precomputed contrail line segments and point clouds from file.

        Args:
            contrail_output_dir: Directory containing line segment files. If None,
                uses default location.
            source_method: Method that was used to generate the segments. Options are
                'decomposer' or 'google_api'. Default is 'decomposer'.
            read_simple: If True, only load line segments without point clouds.
            contrail_stitching: Whether stitching was enabled when the files were
                generated. Determines the stitched/unstitched subfolder (decomposer only).
            prediction_threshold: Threshold used during decomposition. Determines the
                threshold subfolder (decomposer only). If None, uses the default for the
                detection source (see ``PREDICTION_THRESHOLDS``).

        Returns:
            Tuple containing:

            - DataFrame of contrail line segments
            - Dictionary mapping contrail IDs to (N, 2) arrays of [row, col]
              global image index coordinates, or None if read_simple=True or
              point cloud file doesn't exist

        Raises:
            FileNotFoundError: If line segments file does not exist.
        """
        line_segments_path = self._get_line_segments_path(
            contrail_output_dir,
            method=source_method,
            contrail_stitching=contrail_stitching,
            prediction_threshold=prediction_threshold,
        )

        if not line_segments_path.exists():
            raise FileNotFoundError(
                f"Line segments file not found at {line_segments_path}. "
                f"Run get_decomposed_contrails(method='{source_method}') first to generate the file."
            )

        # Read the parquet file
        line_segments = pd.read_parquet(line_segments_path)

        # Convert timestamp strings back to datetime objects
        line_segments["timestamp"] = pd.to_datetime(
            line_segments["timestamp"],
            format="ISO8601",
        ).dt.floor("s")

        # Skip point clouds if read_simple is True
        if read_simple:
            return line_segments, None

        # Try to load point clouds if they exist
        point_clouds = None
        points_path = self._get_point_clouds_path(
            contrail_output_dir,
            method=source_method,
            contrail_stitching=contrail_stitching,
            prediction_threshold=prediction_threshold,
        )

        if points_path.exists():
            points_df = pd.read_parquet(points_path)
            point_clouds = {
                cid: group[["row", "col"]].values
                for cid, group in points_df.groupby("contrail_id", observed=True)
            }
            logging.info(f"Loaded {len(point_clouds)} point clouds from {points_path}.")
        else:
            logging.info(
                f"Point cloud file not found at {points_path}. Returning line segments only."
            )

        return line_segments, point_clouds

    @plt.rc_context({"font.size": 8})
    def plot_ash_rgb(
        self: Self,
        ash_rgb: np.ndarray,
        img_extent: tuple[float, float, float, float],
        img_transform: ccrs.Projection,
        title_text: str | None = None,
        fig_output_dir: str | Path | None = None,
    ) -> plt.Figure:
        """Plot an Ash-RGB composite image from satellite data.

        Visualizes satellite imagery using the Ash-RGB color scheme, which is
        useful for detecting volcanic ash and contrails. Can use outputs from
        `fetch_image` as inputs.

        Args:
            ash_rgb: Ash color scheme RGB image array with shape (height, width, 3).
            img_extent: Image extent as (left, right, bottom, top) in units of
                the given transform.
            img_transform: Cartopy projection representing the satellite field of view.
            title_text: Optional title text for the plot. Defaults to None.
            fig_output_dir: Directory in which to save the figure, as
                ``<satellite>-<region>-ash-rgb.png``. Not saved if None.

        Returns:
            Matplotlib Figure.
        """
        # Figure setup - Platecarree projection
        fig = plt.figure(figsize=(7.0, 7.0), dpi=250)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        with sns.axes_style(colors.plot_params_light):
            # Map subplot, centred on the satellite
            map_crs = transforms.platecarree(central_lon=self.central_lon)
            ax1 = fig.add_subplot(121, projection=map_crs)
            ax1.text(
                x=0.5,
                y=1.05,
                s=f"{title_text}",
                fontsize=8,
                color="k",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

            # Figure extent
            if self.region in ["M1", "M2"]:
                ax1.set_extent(img_extent, crs=img_transform)
            elif self.region == "C":
                ax1.set_extent([-90.0, 30.0, 10.0, 60.0], crs=map_crs)
            elif self.region == "F":
                ax1.set_extent([-90.0, 90.0, -80.0, 80.0], crs=map_crs)

            # Add features to the map
            ax1.add_feature(
                cfeature.BORDERS,
                color=colors.facecolor,
                linestyle="--",
                linewidth=0.3,
                zorder=3,
            )
            ax1.add_feature(
                cfeature.COASTLINE,
                color=colors.facecolor,
                linestyle="-",
                linewidth=0.3,
                zorder=3,
            )

            # Add gridlines
            gl = ax1.gridlines(
                draw_labels=True,
                linestyle=":",
                linewidth=0.5,
                color=colors.facecolor,
                zorder=1,
            )
            gl.xlabel_style = {"color": colors.facecolor}
            gl.ylabel_style = {"color": colors.facecolor}
            gl.right_labels = False
            gl.left_labels = True
            gl.top_labels = False
            gl.bottom_labels = True
            plt.draw()  # Force a redraw of the figure

            # Plot the Ash RGB image
            ax1.imshow(
                ash_rgb,
                extent=img_extent,
                transform=img_transform,
            )

        save_figure(fig, fig_output_dir, f"{self.satellite}-{self.region}-ash-rgb")
        return fig

    def plot_decomposition(
        self: Self,
        timestamp: np.datetime64,
        bbox: tuple[float, float, float, float],
        contrail_summary: pd.DataFrame | None = None,
        point_clouds: dict[str, np.ndarray] | None = None,
        prediction_threshold: float | None = None,
        fig_output_dir: str | Path | None = None,
    ) -> plt.Figure:
        """Plot a frame's detection mask beside its decomposed contrails, zoomed to a region.

        The left panel shows detection scores above the threshold. The right panel colours
        each decomposed contrail's pixels on their own and draws its centre line, so a region
        the decomposer split into several contrails shows as several colours. A globe inset
        marks the region.

        Args:
            timestamp: Frame to plot.
            bbox: Region (min_lon, max_lon, min_lat, max_lat) in degrees. A region across the
                date line may be given with max_lon < min_lon, e.g. (170, -170, ...), or with
                max_lon > 180, e.g. (170, 190, ...).
            contrail_summary: Decomposed contrails to plot, as returned by
                ``get_decomposed_contrails``. Defaults to ``self.contrail_summary``.
            point_clouds: Their pixel point clouds, as returned with them. Defaults to
                ``self.contrail_point_clouds``.
            prediction_threshold: Detection scores below this are hidden. If None, uses the
                default for the detection source.
            fig_output_dir: Directory in which to save the figure, as
                ``<satellite>-<region>-decomposition-<YYYY-MM-DDTHHMM>.png``. Not saved if None.

        Returns:
            Matplotlib Figure.

        Raises:
            ValueError: If detections or decomposed contrails have not been loaded, or the
                region holds no image pixels.
        """
        if not self.detections:
            raise ValueError("Detections have not been loaded. Call get_detections() first.")
        if contrail_summary is None:
            contrail_summary = self.contrail_summary
        if point_clouds is None:
            point_clouds = self.contrail_point_clouds
        if contrail_summary.empty or not point_clouds:
            raise ValueError(
                "Decomposed contrails have not been loaded. Call get_decomposed_contrails() first."
            )
        threshold = self._resolve_prediction_threshold(prediction_threshold)

        # Crop the image to the pixels inside the region
        lon_min, lon_max, lat_min, lat_max = bbox
        lon_min, lon_max = lon_range(lon_min, lon_max)
        crosses_date_line = lon_max > 180.0 or lon_min < -180.0
        if crosses_date_line:
            # Pixel longitudes are in [-180, 180)
            in_lon = (self.lons - lon_min) % 360.0 <= lon_max - lon_min
        else:
            in_lon = (self.lons >= lon_min) & (self.lons <= lon_max)
        inside = in_lon & (self.lats >= lat_min) & (self.lats <= lat_max)
        if not inside.any():
            raise ValueError(f"No image pixels fall inside bbox {bbox}.")
        rows, cols = np.nonzero(inside)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        geo_x, geo_y = self.image_geo_coords
        geo_x, geo_y = geo_x[c0:c1], geo_y[r0:r1]
        geo_crs = self.image_transforms[0]

        scores = np.asarray(self.detections[timestamp], dtype=float)[r0:r1, c0:c1].copy()
        scores[scores < threshold] = np.nan

        # One label per contrail, cycling through the palette
        frame = contrail_summary[contrail_summary["timestamp"] == timestamp]
        palette = colors.wong[1:]
        labels = np.full(scores.shape, np.nan)
        shown = []
        for _, contrail in frame.iterrows():
            points = point_clouds.get(contrail["contrail_id"])
            if points is None:
                continue
            r, c = points[:, 0] - r0, points[:, 1] - c0
            keep = (r >= 0) & (r < labels.shape[0]) & (c >= 0) & (c < labels.shape[1])
            if keep.any():
                labels[r[keep], c[keep]] = len(shown) % len(palette)
                shown.append(contrail)

        # Centre the maps on a region across the date line
        map_crs, extent = transforms.platecarree(), bbox
        if crosses_date_line:
            centre = (lon_min + lon_max) / 2
            map_crs = transforms.platecarree(central_lon=centre)
            extent = (lon_min - centre, lon_max - centre, lat_min, lat_max)

        fig = plt.figure(figsize=(11.0, 5.6), dpi=150)
        with sns.axes_style(colors.plot_params_light):
            axes = [fig.add_subplot(1, 2, k + 1, projection=map_crs) for k in range(2)]
            for k, ax in enumerate(axes):
                ax.set_extent(extent, crs=map_crs)
                ax.add_feature(cfeature.COASTLINE, edgecolor=colors.facecolor, linewidth=0.4)
                ax.add_feature(
                    cfeature.BORDERS, edgecolor=colors.facecolor, linestyle="--", linewidth=0.3
                )
                gl = ax.gridlines(
                    draw_labels=True,
                    linestyle=":",
                    linewidth=0.5,
                    color=colors.facecolor,
                    zorder=1,
                )
                gl.xlocator = gl.ylocator = mticker.MaxNLocator(4)
                gl.top_labels = gl.right_labels = False
                gl.left_labels = k == 0
                gl.xlabel_style = gl.ylabel_style = {"fontsize": 8}

            score_map = axes[0].pcolormesh(
                geo_x,
                geo_y,
                scores,
                shading="auto",
                cmap="Blues",
                vmin=threshold,
                vmax=1.0,
                transform=geo_crs,
            )

            axes[1].pcolormesh(
                geo_x,
                geo_y,
                labels,
                shading="auto",
                cmap=ListedColormap(palette),
                vmin=-0.5,
                vmax=len(palette) - 0.5,
                transform=geo_crs,
            )
            for contrail in shown:
                axes[1].plot(
                    unwrap_lons(
                        [
                            contrail["endpoint1_lon"],
                            contrail["centroid_lon"],
                            contrail["endpoint2_lon"],
                        ]
                    ),
                    [
                        contrail["endpoint1_lat"],
                        contrail["centroid_lat"],
                        contrail["endpoint2_lat"],
                    ],
                    color=colors.facecolor,
                    linewidth=0.6,
                    transform=transforms.platecarree(),
                    zorder=4,
                )

        # Colour bar to the left of the left panel, clear of its latitude labels
        cax = axes[0].inset_axes([-0.17, 0.0, 0.035, 1.0])
        fig.colorbar(score_map, cax=cax, extend="min", label="Detection score")
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
        ts = pd.Timestamp(timestamp)
        axes[0].set_title(
            f"{self.satellite} {self.region}, {ts:%Y-%m-%d %H:%M} UTC", loc="left", fontsize=11
        )

        # Pull the panels together, then centre the globe in the gap above their tops
        fig.subplots_adjust(wspace=0.04)
        axes[0].set_anchor("E")
        axes[1].set_anchor("W")
        for ax in axes:
            ax.apply_aspect()
        left, right = axes[0].get_position(), axes[1].get_position()
        size = 0.2
        add_globe_inset(
            fig,
            list(bbox),
            rect=[(left.x1 + right.x0 - size) / 2, left.y1 - 0.6 * size, size, size],
        )

        save_figure(
            fig, fig_output_dir, f"{self.satellite}-{self.region}-decomposition-{ts:%Y-%m-%dT%H%M}"
        )
        return fig

    def migrate_static_geometry(
        self: Self,
        zarr_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> Path | None:
        """Move a pre-1.13.0 static geometry store aside, keeping its archives working.

        Before 1.13.0 the store's ``scantime_offset`` was the linear model, measured from
        the nominal time. This renames the store to ``static_geometry_pre1.13.zarr`` and
        points every archive written before 1.13.0 (no ``contrail_detect_version``) that
        referred to it at the new name, so a 1.13.0 store can be written in its place. Safe
        to re-run.

        Args:
            zarr_path: Optional archive root. Defaults to ``data/processed_data``.
            dry_run: Log what would change without changing anything.

        Returns:
            Path of the moved store, or None if there was nothing to move.
        """
        if not self.SCANTIME_FROM_SCAN_START:
            return None
        path = self.get_static_geometry_path(zarr_path)
        old = path.with_name("static_geometry_pre1.13.zarr")
        tag = "[DRY RUN] " if dry_run else ""

        if path.exists():
            with xr.open_zarr(path) as existing:
                is_old = "contrail_detect_version" not in existing.attrs
            if is_old:
                if old.exists():
                    raise FileExistsError(f"Both {path} and {old} exist; resolve by hand")
                logging.info(f"{tag}Moving {path} to {old}")
                if not dry_run:
                    shutil.move(path, old)
        if not old.exists() and not dry_run:
            return None

        for archive in self._archives_under(path.parent):
            try:
                attrs = dict(zarr.open_group(archive, mode="r").attrs)
            except Exception as e:
                logging.warning(f"Skipping {archive}, can't open it ({e})")
                continue
            target = attrs.get("static_geometry")
            if "contrail_detect_version" in attrs or not self._is_canonical_store(target):
                continue
            logging.info(f"{tag}Repointing {archive} to {old}")
            if not dry_run:
                group = zarr.open_group(archive, mode="r+")
                group.attrs["static_geometry"] = str(old)
                zarr.consolidate_metadata(archive)
        return old

    def _is_canonical_store(self: Self, target: str | None) -> bool:
        """Whether an archive's ``static_geometry`` attr names this satellite and region's
        canonical store, whatever root it was written with."""
        if not target:
            return False
        p = Path(target)
        return (
            p.name == "static_geometry.zarr"
            and p.parent.name == f"{self.region}"
            and p.parent.parent.name == f"{self.satellite}"
        )

    @staticmethod
    def _archives_under(root: Path) -> list[Path]:
        """Detection archive stores under root, without walking into any store's chunks."""
        found = []
        for dirpath, dirnames, _ in os.walk(root):
            keep = []
            for d in dirnames:
                if d.endswith(".zarr"):
                    if d.startswith("detection_archive"):
                        found.append(Path(dirpath) / d)
                else:
                    keep.append(d)
            dirnames[:] = keep
        return sorted(found)

    def static_geometry_fields(self: Self) -> dict:
        """The fixed-grid fields (``STATIC_GEOMETRY_VARS``), as xarray data_vars.

        Written to the shared static geometry store, or into each archive for a region whose
        domain shifts between epochs.
        """
        return {
            "longitude": (
                ("geo_y", "geo_x"),
                self.lons.copy(),
                {"units": "degrees_east"},
            ),
            "latitude": (
                ("geo_y", "geo_x"),
                self.lats.copy(),
                {"units": "degrees_north"},
            ),
            "pixel_area": (
                ("geo_y", "geo_x"),
                np.round(a=self.pixel_surface_area.copy(), decimals=4),
                {
                    "long_name": "Pixel area",
                    "units": "km²",
                    "description": "Area covered by each pixel in square kilometers",
                },
            ),
            "scantime_offset": (
                ("geo_y", "geo_x"),
                self.static_scan_time_offset_s(),
                {
                    "long_name": "Scan time after the frame's scan start",
                    "units": "seconds",
                    "description": (
                        "A pixel's scan time is time + scan_start_offset (from the "
                        "detection archive) + scantime_offset. NaN off the disk."
                    ),
                },
            ),
        }

    def save_static_geometry(
        self: Self,
        zarr_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path | None:
        """Write the satellite's fixed grid geometry to a single shared store.

        ``latitude``, ``longitude``, ``pixel_area`` and ``scantime_offset`` are the same in
        every daily archive for a satellite and region, so they are stored once here rather
        than in each archive. The package does not read them back; the store keeps archives
        self-describing for other users.

        Returns None without writing for a region with a shifting domain (see
        ``SHIFTING_DOMAIN_REGIONS``), whose geometry changes between epochs.

        Args:
            zarr_path: Optional archive root. Defaults to ``data/processed_data``.
            overwrite: Rewrite the store if it already exists.

        Returns:
            Path to the static geometry store, or None if the region's domain shifts
            between epochs and no shared store was written.
        """
        if self.has_shifting_domain:
            logging.info(
                f"{self.satellite}-{self.region} domain shifts between epochs, "
                f"keeping geometry in the archive rather than a shared store"
            )
            return None

        path = self.get_static_geometry_path(zarr_path)
        if path.exists():
            try:
                with xr.open_zarr(path) as existing:
                    version = existing.attrs.get("contrail_detect_version")
            except Exception as e:
                if not overwrite:
                    # Most likely another job is part-way through writing it
                    logging.info(f"Static geometry at {path} can't be opened ({e}), leaving it")
                    return path
                version = __version__  # rewriting a broken store
            if version is None and self.SCANTIME_FROM_SCAN_START:
                raise ValueError(
                    f"Static geometry at {path} predates contrail-detect 1.13.0: its "
                    "scantime_offset is the old linear model, measured from the nominal time, "
                    "which archives from 1.13.0 can't use. Run migrate_static_geometry() to "
                    "move it aside and repoint the old archives to it."
                )
            if not overwrite:
                logging.info(f"Static geometry already present at {path}, not rewriting")
                return path

        geometry_ds = xr.Dataset(
            data_vars=self.static_geometry_fields(),
            coords={
                "geo_x": ("geo_x", self.image_geo_coords[0]),
                "geo_y": ("geo_y", self.image_geo_coords[1]),
            },
            attrs={
                "title": "Static satellite grid geometry",
                "satellite": self.satellite,
                "region": f"{self.region}",
                "extent": self.tile_extent,
                "description": (
                    "Fixed-grid fields shared by every detection archive for this "
                    "satellite and region."
                ),
                "date_created": pd.Timestamp.now().isoformat(),
                "contrail_detect_version": __version__,
            },
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            name: {"compressors": geo_utils.detection_encoding()["compressors"]}
            for name in self.STATIC_GEOMETRY_VARS
        }
        geometry_ds.to_zarr(path, mode="w", encoding=encoding)
        logging.info(
            f"Saved static geometry to {path} ({geo_utils.directory_size_mb(path):,.1f} mb)"
        )
        return path

    def static_scan_time_offset_s(self: Self) -> np.ndarray:
        """``scan_time_offset`` in seconds, NaN off the disk, as stored with the archives."""
        grid = self.scan_time_offset / np.timedelta64(1, "s")
        return np.where(np.isfinite(self.lats), grid, np.nan)

    def scan_start_offsets_s(self: Self, timestamps: np.ndarray) -> np.ndarray:
        """Each frame's scan start in seconds after its nominal time.

        Zero here, where ``scan_time_offset`` is measured from the nominal time; GOES
        overrides it. A pixel's time is nominal + this + ``scan_time_offset``.
        """
        return np.zeros(np.shape(timestamps))

    def scan_time_offset_for_position(
        self: Self,
        lat: float | np.ndarray,
        lon: float | np.ndarray,
        timestamp: np.datetime64 | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Calculate scan time offset for specific geographic positions.

        Interpolates the scan time offset grid to determine the offset at
        given latitude and longitude coordinates.

        Args:
            lat: Latitude of the position(s) in degrees.
            lon: Longitude of the position(s) in degrees.
            timestamp: Nominal time of the frame each position is in, scalar or one per
                position, so each frame's own scan start is used (see
                ``scan_start_offsets_s``). Defaults to the first frame.

        Returns:
            Scan time offset(s) for the specified position(s) in seconds.
        """
        is_scalar = np.isscalar(lat) and np.isscalar(lon)

        # interpolate offset using lat lon grids
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        i, j = self.get_grid_index_from_position(
            lat=lat,
            lon=lon,
        )
        result = self.scan_time_offset[i, j]

        # Set result to NaN where i or j are -1
        nan_mask = (i == -1) | (j == -1)

        result = np.where(nan_mask, np.timedelta64("NaT"), result)
        if timestamp is None:
            timestamp = self.timestamp_sequence[0]
        starts_s = self.scan_start_offsets_s(np.broadcast_to(timestamp, np.shape(lat)))
        starts = np.where(
            np.isnan(starts_s),
            np.timedelta64("NaT"),
            (np.nan_to_num(starts_s) * 1000).round().astype(np.int64) * np.timedelta64(1, "ms"),
        )
        result = result + starts
        # Return scalar if input was scalar
        if is_scalar:
            return result.item()

        return result

    def summarize_tracks(self: Self) -> None:
        """Print a summary of contrail tracking results.

        Reports key statistics about the tracks derived from contrail_summary,
        including track counts, length distributions, temporal span, and the
        fraction of detections that form multi-frame tracks.

        Raises:
            ValueError: If contrail_summary does not contain a 'track_id' column.
        """
        if "track_id" not in self.contrail_summary.columns:
            raise ValueError(
                "'track_id' column not found in contrail_summary. "
                "Run tracking before calling summarize_tracks."
            )

        df = self.contrail_summary.copy()
        track_groups = df.groupby("track_id")

        # Frames per track
        frames_per_track = track_groups["timestamp"].nunique()
        n_tracks = len(frames_per_track)
        n_singletons = (frames_per_track == 1).sum()
        n_multi = n_tracks - n_singletons

        # Fraction of contrail detections that belong to a multi-frame track
        multi_track_ids = frames_per_track[frames_per_track > 1].index
        n_detections_in_multi = df["track_id"].isin(multi_track_ids).sum()
        frac_in_multi = n_detections_in_multi / len(df) if len(df) > 0 else 0.0

        # Temporal span of each track (minutes)
        time_span_min = track_groups["timestamp"].apply(
            lambda t: (t.max() - t.min()) / np.timedelta64(1, "m")
        )

        total_span_min = (
            self.timestamp_sequence[-1] - self.timestamp_sequence[0]
        ) / np.timedelta64(1, "m")

        # Mean contrail length per track (km)
        if "great_circle_length_km" in df.columns:
            mean_length_per_track = track_groups["great_circle_length_km"].mean()
        else:
            mean_length_per_track = None

        lines = [
            "=" * 50,
            "Contrail Tracking Summary",
            "=" * 50,
            f"Total contrail detections : {len(df)}",
            f"Total tracks              : {n_tracks}",
            f"  Single-frame tracks     : {n_singletons} ({100 * n_singletons / n_tracks:.1f}%)",
            f"  Multi-frame tracks      : {n_multi} ({100 * n_multi / n_tracks:.1f}%)",
            f"Detections in multi-frame : {n_detections_in_multi} / {len(df)} "
            f"({100 * frac_in_multi:.1f}%)",
            "",
            "Frames per track:",
            f"  min      : {frames_per_track.min():.0f}",
            f"  median   : {frames_per_track.median():.1f}",
            f"  mean     : {frames_per_track.mean():.2f}",
            f"  max      : {frames_per_track.max():.0f}",
            f"  possible : {len(self.timestamp_sequence)} (total no. frames)",
            "",
            "Track time span (minutes):",
            f"  min      : {time_span_min.min():.1f}",
            f"  median   : {time_span_min.median():.1f}",
            f"  mean     : {time_span_min.mean():.2f}",
            f"  max      : {time_span_min.max():.1f}",
            f"  possible : {total_span_min:.1f} (total time span)",
        ]

        if mean_length_per_track is not None:
            lines += [
                "",
                "Mean contrail length per track (km):",
                f"  min      : {mean_length_per_track.min():.1f}",
                f"  median   : {mean_length_per_track.median():.1f}",
                f"  mean     : {mean_length_per_track.mean():.2f}",
                f"  max      : {mean_length_per_track.max():.1f}",
            ]

        lines.append("=" * 50)
        print("\n".join(lines))

    def _geo_point_clouds_from_point_clouds(
        self: Self,
        point_clouds: dict[str, np.ndarray],
    ) -> dict[str, MultiPoint]:
        """Convert index point clouds to geo point clouds using the detector lat/lon grids.

        Converts image index coordinates to lon/lat MultiPoint geometries and stores
        the result in self.contrail_geo_point_clouds.

        Args:
            point_clouds: Dictionary mapping contrail IDs to (N, 2) arrays of
                [row, col] global image index coordinates.

        Returns:
            Dictionary mapping contrail IDs to MultiPoint geometries in lon/lat.
        """
        geo_point_clouds = {}
        for contrail_id, points in point_clouds.items():
            rows, cols = points[:, 0], points[:, 1]
            lons = self.lons[rows, cols]
            lats = self.lats[rows, cols]
            geo_point_clouds[contrail_id] = MultiPoint(list(zip(lons, lats)))

        self.contrail_geo_point_clouds = geo_point_clouds
        return geo_point_clouds

    def _resolve_prediction_threshold(self: Self, prediction_threshold: float | None) -> float:
        """Return the given threshold, or the default for this detector's detection source.

        Args:
            prediction_threshold: Threshold to use, or None for the source default.

        Returns:
            Detection-score threshold for decomposition.
        """
        if prediction_threshold is not None:
            return prediction_threshold
        return self.PREDICTION_THRESHOLDS.get(
            self.detection_subdirectory, self.DEFAULT_PREDICTION_THRESHOLD
        )

    def _plot_frames(
        self: Self,
        plot_frame: Callable[..., plt.Figure | None],
        timestamps: list[np.datetime64],
        fig_output_dir: str | Path | None,
        desc: str,
    ) -> list[plt.Figure]:
        """Plot each frame, closing the figures as they are saved if a directory is given.

        Args:
            plot_frame: Plots one frame, given ``timestamp`` and ``fig_output_dir``, and
                returns its figure, or None if the frame could not be plotted.
            timestamps: Frames to plot.
            fig_output_dir: Directory to save the figures into, or None to keep them open.
            desc: Progress bar label.

        Returns:
            The figures if not saving, otherwise an empty list.
        """
        figures = []
        for timestamp in tqdm(timestamps, desc=desc):
            fig = plot_frame(timestamp=timestamp, fig_output_dir=fig_output_dir)
            if fig is None:
                continue
            if fig_output_dir is None:
                figures.append(fig)
            else:
                plt.close(fig)
        return figures

    @plt.rc_context({"font.size": 14})
    def _plot_detection_map(
        self: Self,
        timestamp: np.datetime64,
        detection: np.ndarray,
        fig_output_dir: str | Path | None = None,
        detection_cmap: str | Colormap = "Blues",
        prediction_threshold: float | None = None,
        show_domain: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> plt.Figure:
        """Map one frame's detection mask on the Full Disk grid.

        Args:
            timestamp: Frame time, for the title and the saved file name.
            detection: Detection scores on the Full Disk grid, NaN outside the detection
                domain.
            fig_output_dir: Directory in which to save the figure. Not saved if None.
            detection_cmap: Colormap with which to plot detection mask.
            prediction_threshold: Pixels with a detection score below this
                value are hidden. If None, uses the default for the detection source.
            show_domain: Whether or not to outline the processed imaging domain.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to 80 degrees either side of the sub-satellite point.

        Returns:
            Matplotlib Figure.
        """
        detection = np.asarray(detection)
        geo_x, geo_y = (np.asarray(c) for c in self.image_geo_coords)
        cartopy_crs, _ = self.image_transforms
        map_crs, extent, extent_crs = self._full_disk_map(bbox)

        fig = plt.figure(figsize=(12.0, 12.0), dpi=100)
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

        with sns.axes_style(colors.plot_params_light):
            ax = fig.add_subplot(111, projection=map_crs)
            ax.add_feature(
                cfeature.BORDERS,
                color=colors.facecolor,
                linestyle="--",
                linewidth=0.3,
                zorder=3,
            )
            ax.add_feature(
                cfeature.COASTLINE,
                color=colors.facecolor,
                linestyle="-",
                linewidth=0.3,
                zorder=3,
            )
            gl = ax.gridlines(
                draw_labels=True,
                linestyle=":",
                linewidth=0.5,
                color=colors.facecolor,
                zorder=1,
            )
            gl.xlabel_style = {"color": colors.facecolor}
            gl.ylabel_style = {"color": colors.facecolor}
            gl.right_labels = False
            gl.left_labels = True
            gl.top_labels = False
            gl.bottom_labels = True
            if bbox is None:
                # Fixed 30-degree spacing for the whole disk; zoomed maps use cartopy's ticks
                gl.xlocator = mticker.FixedLocator(self.MAP_LON_TICKS)
            plt.draw()

            ts_str = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
            ax.set_title(f"{self.satellite} Full Disk Contrail Detections\n{ts_str}")

        if show_domain:
            # Outline of the pixels with detection data; every 5th pixel is ample for a boundary
            step = 5
            ax.contour(
                geo_x[::step],
                geo_y[::step],
                np.isfinite(detection[::step, ::step]).astype(float),
                levels=[0.5],
                colors="grey",
                linewidths=0.8,
                transform=cartopy_crs,
            )

        # Over the whole disk a figure pixel covers several image pixels, and contrails one or
        # two pixels wide would be dropped when drawn; keep each block's highest score instead
        plot_x, plot_y, plot_scores = geo_x, geo_y, detection
        if bbox is None:
            factor = int(np.ceil(detection.shape[1] / (fig.get_figwidth() * fig.dpi)))
            plot_x, plot_y, plot_scores = max_pool(detection, geo_x, geo_y, factor)

        threshold = self._resolve_prediction_threshold(prediction_threshold)
        shown = plot_scores.copy()
        shown[shown < threshold] = np.nan
        detect_map = ax.pcolormesh(
            plot_x,
            plot_y,
            shown,
            shading="auto",
            cmap=detection_cmap,
            vmin=0.0,
            vmax=1.0,
            transform=cartopy_crs,
        )
        ax.set_extent(extent, crs=extent_crs)
        pos = ax.get_position()
        ax_cbar = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, 0.75 * pos.height])
        plt.colorbar(
            detect_map,
            cax=ax_cbar,
            orientation="vertical",
            label="Detection Score",
        )

        self._save_frame_figure(fig, fig_output_dir, timestamp, dpi=400)
        return fig

    def _save_frame_figure(
        self: Self,
        fig: plt.Figure,
        fig_output_dir: str | Path | None,
        timestamp: np.datetime64,
        dpi: float | None = None,
    ) -> None:
        """Save a frame's figure as ``<YYYY-MM-DD>/<satellite>-<region>-<YYYY-MM-DDTHHMM>.png``.

        Args:
            fig: Figure to save.
            fig_output_dir: Directory to save under. Nothing is saved if None.
            timestamp: Frame time, which names the file and its date subdirectory.
            dpi: Resolution to save at. If None, uses the figure's own.
        """
        if fig_output_dir is None:
            return
        ts = pd.Timestamp(timestamp)
        save_figure(
            fig,
            Path(fig_output_dir) / f"{ts:%Y-%m-%d}",
            f"{self.satellite}-{self.region}-{ts:%Y-%m-%dT%H%M}",
            dpi=dpi,
        )

    def _full_disk_map(
        self: Self, bbox: tuple[float, float, float, float] | None = None
    ) -> tuple[ccrs.CRS, tuple[float, float, float, float], ccrs.CRS]:
        """Projection and extent shared by the Full Disk detection maps.

        Maps are centred on the sub-satellite point. Without a bbox they are cropped to
        80 degrees either side of it, rather than showing the whole globe.

        Args:
            bbox: Extent (min_lon, max_lon, min_lat, max_lat) in plain longitudes, or None.

        Returns:
            The map projection, the extent, and the projection the extent is given in.
        """
        map_crs = transforms.platecarree(central_lon=self.central_lon)
        if bbox is not None:
            return map_crs, tuple(bbox), transforms.platecarree()
        return map_crs, self.FULL_DISK_MAP_EXTENT, map_crs

    def _processed_output_dir(
        self: Self,
        contrail_output_dir: str | Path | None,
        method: str | None,
        contrail_stitching: bool | None = None,
        prediction_threshold: float | None = None,
    ) -> Path:
        """Return the directory holding processed contrail outputs for this detector.

        For the 'decomposer' method the path ends in ``settings_<fingerprint>``; see
        ``_decomposition_fingerprint``.

        Args:
            contrail_output_dir: Root output directory. If None, uses
                ``DATA_DIR / "processed_data"``.
            method: Method that produced the outputs (e.g. 'google_api', 'decomposer').
                If provided, adds a method subdirectory.
            contrail_stitching: For the 'decomposer' method, selects the stitched or
                unstitched subfolder. None skips the stitching and threshold subfolders.
            prediction_threshold: For the 'decomposer' method, selects the threshold
                subfolder. If None, uses the default for the detection source.

        Returns:
            Path to the output directory.
        """
        root = (
            DATA_DIR / "processed_data"
            if contrail_output_dir is None
            else Path(contrail_output_dir)
        )
        output_dir = (
            root / self.satellite / self.region / self.detection_subdirectory / self.LONG_NAME
        )
        if method is not None:
            output_dir = output_dir / method
        if method == "decomposer" and contrail_stitching is not None:
            stitching_dir = "stitched" if contrail_stitching else "unstitched"
            threshold = self._resolve_prediction_threshold(prediction_threshold)
            threshold_dir = f"threshold_{int(round(threshold * 100)):03d}"
            settings_dir = (
                f"settings_{self._decomposition_fingerprint(contrail_stitching, threshold)}"
            )
            output_dir = output_dir / stitching_dir / threshold_dir / settings_dir
        return output_dir

    def _decomposition_fingerprint(
        self: Self, contrail_stitching: bool, prediction_threshold: float
    ) -> str:
        """Short hash of the settings that change the decomposition output.

        Covers stitching, threshold, ``self.decomposition_settings`` (defaults where unset),
        the tile extent and ``_DECOMPOSITION_CACHE_VERSION``.

        Args:
            contrail_stitching: Whether stitching is enabled.
            prediction_threshold: Resolved detection-score threshold.

        Returns:
            Ten hexadecimal characters.
        """
        from contrail_detect.decomposition.decompose import DecompositionConfig

        # Detectors built without __init__ lack these
        settings = dict(getattr(self, "decomposition_settings", None) or {})
        min_area = settings.pop("min_area", 10)
        settings.pop("contrail_stitching", None)
        config = DecompositionConfig(**settings, contrail_stitching=contrail_stitching)
        tile_extent = getattr(self, "tile_extent", None)
        extent = (
            None if tile_extent is None else [round(float(x), 6) for x in np.ravel(tile_extent)]
        )
        payload = {
            "version": _DECOMPOSITION_CACHE_VERSION,
            "prediction_threshold": round(float(prediction_threshold), 6),
            "min_area": int(min_area),
            "extent": extent,
            **config.to_dict(),
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:10]

    def _get_geo_point_clouds_path(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> Path:
        """Return the path to the geo point clouds GeoJSON file.

        The file sits next to the line segments it was made from, so each stitching and
        threshold setting has its own and a re-run with other settings cannot reuse it.

        Args:
            contrail_output_dir: Root output directory. If None, uses the default location.
            method: Method that produced the point clouds (e.g. 'decomposer').
            contrail_stitching: For the 'decomposer' method, whether the contrails came
                from a stitched run. Defaults to True, as in ``get_decomposed_contrails``.
            prediction_threshold: For the 'decomposer' method, the threshold the contrails
                were decomposed at. If None, uses the default for the detection source.

        Returns:
            Path to the geo point clouds GeoJSON file.
        """
        output_dir = self._processed_output_dir(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )
        return output_dir / "contrail_geopoints.geojson"

    def _get_line_segments_path(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> Path:
        """Return the path to the line segments parquet file, creating its directory.

        Args:
            contrail_output_dir: Root output directory. If None, uses the default location.
            method: Method that produced the segments (e.g. 'google_api', 'decomposer').
            contrail_stitching: Whether stitching was enabled ('decomposer' only).
            prediction_threshold: Threshold used during decomposition ('decomposer' only).

        Returns:
            Path to the line segments parquet file.
        """
        output_dir = self._processed_output_dir(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "contrail_summary.parquet"

    def _get_point_clouds_path(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> Path:
        """Return the path to the point clouds parquet file.

        Args:
            contrail_output_dir: Root output directory. If None, uses the default location.
            method: Method that produced the point clouds (e.g. 'decomposer').
            contrail_stitching: Whether stitching was enabled ('decomposer' only).
            prediction_threshold: Threshold used during decomposition ('decomposer' only).

        Returns:
            Path to the point clouds parquet file.
        """
        output_dir = self._processed_output_dir(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )
        return output_dir / "contrail_points.parquet"

    def _load_geo_point_clouds(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> dict[str, MultiPoint]:
        """Load geo point clouds from GeoJSON into self.contrail_geo_point_clouds.

        Args:
            contrail_output_dir: Directory containing geo point cloud files. If None,
                uses default location.
            method: Method subdirectory (e.g., 'decomposer'). If None, no subdirectory
                is added.
            contrail_stitching: Whether the contrails came from a stitched run ('decomposer'
                only).
            prediction_threshold: Threshold the contrails were decomposed at ('decomposer'
                only). If None, uses the default for the detection source.

        Returns:
            Dictionary mapping contrail IDs to MultiPoint geometries in lon/lat.

        Raises:
            FileNotFoundError: If the geo point clouds GeoJSON file does not exist.
        """
        geo_points_path = self._get_geo_point_clouds_path(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )

        if not geo_points_path.exists():
            raise FileNotFoundError(f"Geo point clouds file not found at {geo_points_path}.")

        gdf = gpd.read_file(geo_points_path)
        self.contrail_geo_point_clouds = {
            row["contrail_id"]: row.geometry for _, row in gdf.iterrows()
        }
        logging.info(
            f"Loaded {len(self.contrail_geo_point_clouds)} geo point clouds from {geo_points_path}."
        )
        return self.contrail_geo_point_clouds

    def _reproject_cropped_to_full_grid(
        self: Self,
        cropped_array: np.ndarray,
        cropped_geo_x: np.ndarray,
        cropped_geo_y: np.ndarray,
    ) -> np.ndarray:
        """Re-project a cropped array onto the detector's full grid.

        Takes a cropped bounding box array (e.g., from zarr store) and places it
        back into a full-size array matching the detector's coordinate grid.

        Args:
            cropped_array: Cropped 2D array to re-project.
            cropped_geo_x: X-coordinates of the cropped array (1D).
            cropped_geo_y: Y-coordinates of the cropped array (1D).

        Returns:
            Full-size array with cropped data placed at correct position, NaN elsewhere.

        Raises:
            ValueError: If the cropped coordinates are not on the detector's grid.
        """
        # Initialize full-size array with NaNs
        full_array = np.full_like(self.lats, np.nan, dtype=np.float32)

        # Get detector's full grid coordinates
        detector_geo_x = self.image_geo_coords[0]
        detector_geo_y = self.image_geo_coords[1]

        # Find matching column indices (x-direction)
        col_start = np.where(np.isclose(detector_geo_x, cropped_geo_x[0], atol=1e-3))[0]
        col_end = np.where(np.isclose(detector_geo_x, cropped_geo_x[-1], atol=1e-3))[0]

        # Find matching row indices (y-direction)
        row_start = np.where(np.isclose(detector_geo_y, cropped_geo_y[0], atol=1e-3))[0]
        row_end = np.where(np.isclose(detector_geo_y, cropped_geo_y[-1], atol=1e-3))[0]

        if len(col_start) > 0 and len(col_end) > 0 and len(row_start) > 0 and len(row_end) > 0:
            col_start_idx = col_start[0]
            col_end_idx = col_end[0] + 1
            row_start_idx = row_start[0]
            row_end_idx = row_end[0] + 1

            # Place cropped data into full array at correct position
            full_array[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = cropped_array
        else:
            raise ValueError(
                "The saved detections' coordinates do not line up with this detector's "
                "grid, so they cannot be placed on it. Were they saved for a different "
                "satellite, region or extent?"
            )

        return full_array

    def _save_geo_point_clouds(
        self: Self,
        contrail_output_dir: str | Path | None = None,
        method: str | None = None,
        contrail_stitching: bool = True,
        prediction_threshold: float | None = None,
    ) -> None:
        """Save self.contrail_geo_point_clouds to a GeoJSON file.

        Uses self.contrail_summary for timestamp metadata.

        Args:
            contrail_output_dir: Directory for geo point cloud files. If None,
                uses default location.
            method: Method subdirectory (e.g., 'decomposer'). If None, no subdirectory
                is added.
            contrail_stitching: Whether the contrails came from a stitched run ('decomposer'
                only).
            prediction_threshold: Threshold the contrails were decomposed at ('decomposer'
                only). If None, uses the default for the detection source.

        Raises:
            ValueError: If contrail_geo_point_clouds is empty.
        """
        if not self.contrail_geo_point_clouds:
            raise ValueError("No geo point clouds to save. Run get_geo_point_clouds() first.")

        geo_points_path = self._get_geo_point_clouds_path(
            contrail_output_dir, method, contrail_stitching, prediction_threshold
        )

        contrail_ids = list(self.contrail_geo_point_clouds.keys())
        geometries = list(self.contrail_geo_point_clouds.values())

        contrail_summary_indexed = self.contrail_summary.set_index("contrail_id")
        timestamp = contrail_summary_indexed.loc[contrail_ids, "timestamp"].values

        gdf = gpd.GeoDataFrame(
            pd.DataFrame({"contrail_id": contrail_ids, "timestamp": timestamp}),
            geometry=geometries,
            crs="EPSG:4326",
        )
        gdf.to_file(geo_points_path, driver="GeoJSON")
        logging.info(f"Saved geo point clouds to {geo_points_path}.")

    def _set_domain(self: Self) -> None:
        """Set the geographic domain boundaries for contrail detection.

        Computes the minimum and maximum latitude and longitude from the
        actual coordinate grids (lons/lats) derived from the satellite data.
        This ensures the domain matches the actual data coverage.
        """
        lons = self.lons
        # Shift lons to a continuous range centred on the sub-satellite point, so a
        # footprint crossing the anti-meridian (e.g. GOES-West) gets correct bounds
        if hasattr(self, "central_lon") and self.central_lon is not None:
            c = self.central_lon
            lons = (((lons - c) + 180) % 360) - 180 + c
        self.min_longitude = float(np.nanmin(lons))
        self.max_longitude = float(np.nanmax(lons))
        self.min_latitude = float(np.nanmin(self.lats))
        self.max_latitude = float(np.nanmax(self.lats))

    @staticmethod
    def display_valid_datetime_ranges() -> None:
        """Display valid datetime ranges for each satellite in a formatted table."""
        # Define column widths
        sat_w, start_w, end_w = 15, 20, 20

        print("\nValid datetime ranges for geostationary satellites:")

        # Header
        header = f"{'Satellite':<{sat_w}} | {'Start Date':<{start_w}} | {'End Date':<{end_w}}"
        print("\n" + header)
        print("-" * len(header))

        for sat, (start, end) in Geostationary.SATELLITE_DATES.items():
            start_str = pd.to_datetime(start).strftime("%Y-%m-%d %H:%M")
            if end == Geostationary.__FUTURE_BACKSTOP:
                end_str = "present"
            else:
                end_str = pd.to_datetime(end).strftime("%Y-%m-%d %H:%M")

            # Print row with alignment
            print(f"{sat:<{sat_w}} | {start_str:<{start_w}} | {end_str:<{end_w}}")
        print("-" * len(header) + "\n")
