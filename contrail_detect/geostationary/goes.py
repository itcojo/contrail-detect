"""
Contrail detection using ABI data from GOES family satellites.

"""

import json
import logging
import warnings
from functools import cached_property, partial
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
import xarray as xr
from pycontrails.datalib.goes import GOES as pycon_GOES
from pycontrails.datalib.goes import extract_goes_visualization

from contrail_detect.core.common import (
    Colors,
    Transforms,
    extract_date_time_components,
)
from contrail_detect.core.config import DATA_DIR, RESULT_DIR
from contrail_detect.geostationary import abi_time_model, validation_utils
from contrail_detect.geostationary.abi_base import ABIDetector

# Standard values
colors = Colors()
transforms = Transforms()

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="facecolor will have no effect as it has been defined",
)


class GOESDetector(ABIDetector):
    """Contrail detection using ABI data from GOES-R series satellites.

    Implements contrail detection for the Geostationary Operational Environmental
    Satellites (GOES) series, including GOES-16, GOES-18, and GOES-19. Supports
    multiple scan sectors (Full Disk, CONUS, Mesoscale) and provides methods for
    data retrieval from Google Cloud Storage.

    Attributes:
        GOES_BANDS: List of GOES ABI channel names used for detection.
        VALID_REGIONS: Supported GOES scan regions (F, C, M1, M2).
        VALID_SATELLITES: Supported GOES satellites.
        SCAN_PARAMETERS: Fallback time from nominal to scan start, per scan sector.
    """

    # Class-level constants
    GOES_BANDS: list[str] = ["C11", "C14", "C15"]
    VALID_REGIONS: list[str] = ["M1", "M2", "C", "F"]
    VALID_SATELLITES: list[str] = ["GOES-16", "GOES-18", "GOES-19"]

    # Time from nominal to scan start, used when a frame's time_bounds can't be read. Measured
    # from ABI L1b time_bounds on GOES-16 Mode 6 frames.
    SCANTIME_FROM_SCAN_START: bool = True

    SCAN_PARAMETERS: dict[str, dict[str, float]] = {
        "F": {"overhead_s": 20.48},
        "C": {"overhead_s": 77.21},
        "M1": {"overhead_s": 25.36},
        "M2": {"overhead_s": 54.80},
    }

    # Expected central longitudes for each satellite (degrees East, negative is West)
    # Used to validate cached projection attributes
    SATELLITE_LONGITUDES: dict[str, float] = {
        "GOES-16": -75.0,  # GOES-East until April 2025
        "GOES-18": -137.2,  # GOES-West
        "GOES-19": -75.0,  # GOES-East from April 2025
    }

    def __init__(
        self: Self,
        goes_satellite: str,
        goes_region: str,
        start_time: np.datetime64,
        end_time: np.datetime64,
        extent: tuple[float, float, float, float] | None = None,
        *,
        color_scheme: str = "ash",
    ) -> None:
        """Initialize the GOESDetector for specified satellite, region, and time range.

        Args:
            goes_satellite: Satellite name (``'GOES-16'``, ``'GOES-18'``, or ``'GOES-19'``).
            goes_region: GOES scan region (``'M1'``, ``'M2'``, ``'C'``, or ``'F'``).
            start_time: Start of observation period.
            end_time: End of observation period.
            extent: The geographical extent to cover with tiles. Defaults to
                region-dependent standard values.
            color_scheme: Visualization color scheme. Defaults to ``'ash'``.

        Raises:
            ValueError: If invalid region or satellite is specified.
        """
        super().__init__()

        self.color_scheme: str = color_scheme
        self.region: str = goes_region.upper()
        self.satellite: str = goes_satellite.upper()
        self.sweep_axis = "x"  # Sweep axis for GOES

        # Validate satellite and region inputs
        validation_utils.validate_satellite_inputs(
            self.region,
            self.satellite,
            self.VALID_REGIONS,
            self.VALID_SATELLITES,
        )

        # Path to detection results
        self.DETECT_PATH: Path = Path(RESULT_DIR) / self.satellite / self.region

        # Generate pycon_GOES object for the specified region
        self.pycon_goes = pycon_GOES(
            region=self.region,
            bands=set(self.GOES_BANDS),
            bucket=self.goes_gcs_bucket,
        )

        # Create timestamp sequence
        self.create_timestamp_sequence(
            start_time=start_time,
            end_time=end_time,
        )

        # Load datalib entry, but use defaults if it fails
        # (usually pycontrails datalib unavailable)
        try:
            self.da_goes = self.pycon_goes.get(self.timestamp_sequence[0])
            self._data_loaded_successfully = True
            self._save_projection_cache()

        except Exception as e:
            logging.warning(
                f"Failed to load data for {self.satellite}-{self.region} "
                f"at time {self.timestamp_sequence[0]}: {e}, "
                f"Will attempt to use cached projection attributes if available. "
            )
            self.da_goes = None
            self._data_loaded_successfully = False

        # Warn rather than raise: plotting across a repositioning is legitimate, whereas
        # writing an archive is not, and _detections_to_zarr refuses it there.
        if self._data_loaded_successfully and self.spans_single_epoch() is False:
            logging.warning(
                f"{self.satellite}-{self.region} is repositioned between "
                f"{self.timestamp_sequence[0]} and {self.timestamp_sequence[-1]}: this "
                f"range covers more than one observation epoch, so its frames do not "
                f"share a single grid. Split the range at the epoch boundaries before "
                f"writing detections."
            )

        # Long name of detector (used in file construction)
        self.LONG_NAME = self.gen_detector_long_name()

        # Set domain for selected region
        self._set_central_lon()
        self.lons, self.lats = self._get_region_coords()
        self._set_domain()

        # Validate and set extent or set default
        self.tile_extent = extent or (
            self.min_longitude,
            self.max_longitude,
            self.min_latitude,
            self.max_latitude,
        )
        if extent is not None:
            validation_utils.validate_extent(
                extent,
                self.min_longitude,
                self.max_longitude,
                self.min_latitude,
                self.max_latitude,
            )

        # The scan-time grid (scan_time_offset) is built on first use, so a range in a mode
        # with no time model can still be plotted or detected

        # Initialize projection cache for coordinate/image data
        self.projection_cache: dict[pd.Timestamp, dict] = {}

        # Generate tiles
        self._gen_tiles()

        # Image offset from global grid (col, row)
        if extent is not None:
            tile_extent_idx = self._tile_extent_as_indices()
            self.image_offset = (tile_extent_idx[0], tile_extent_idx[2])

    def _repr_fields(self: Self) -> dict:
        return {**super()._repr_fields(), "color_scheme": self.color_scheme}

    @property
    def goes_gcs_bucket(self: Self) -> str:
        """Get the Google Cloud Storage bucket name for GOES data (from NOAA).

        Returns:
            GCS bucket name string.
        """
        return f"gcp-public-data-{self.satellite.lower()}"

    @property
    def image_full_extent(self: Self) -> list[float]:
        """Return the full extent of the image in geostationary coordinates.

        Returns:
            List of extent values [x_min, x_max, y_min, y_max].
        """
        goes_x, goes_y = self.image_geo_coords
        return [goes_x.min(), goes_x.max(), goes_y.min(), goes_y.max()]

    @property
    def image_geo_coords(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """Return the geostationary x and y coordinates of the image pixels.

        Returns:
            Tuple of numpy arrays (x, y) in geostationary coordinates.
        """
        goes_attrs = self._get_projection_attrs

        if self.da_goes is not None:
            x_vals = self.da_goes["x"].values
            y_vals = self.da_goes["y"].values
        else:
            cache_dir = DATA_DIR / "geostationary_data"
            cache_x_path = cache_dir / f"{self.satellite}_{self.region}_x.npy"
            cache_y_path = cache_dir / f"{self.satellite}_{self.region}_y.npy"

            if not (cache_x_path.exists() and cache_y_path.exists()):
                raise RuntimeError(
                    f"'image_geo_coords' is not available for {self.satellite}-{self.region} "
                    f"at time {self.timestamp_sequence[0]}. This means that projection "
                    f"attributes cannot be retrieved from pycontrails datalib or cache. "
                    f"Suggest running different timestamps until successful retrieval."
                )

            if cache_x_path.stat().st_size == 0 or cache_y_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Cached geo-coord files for {self.satellite}-{self.region} are empty "
                    f"(likely a partial write from a previous run). Delete "
                    f"{cache_x_path} and {cache_y_path} and re-run."
                )

            x_vals = np.load(cache_x_path)
            y_vals = np.load(cache_y_path)

        goes_x = goes_attrs["perspective_point_height"] * x_vals
        goes_x = np.asarray(goes_x, dtype=np.float64)  # Ensure float64 for zarr compatibility
        goes_y = goes_attrs["perspective_point_height"] * y_vals
        goes_y = np.asarray(goes_y, dtype=np.float64)  # Ensure float64 for zarr compatibility

        return goes_x, goes_y

    @property
    def image_interval(self: Self) -> float:
        """
        Return the image interval in minutes.
        """
        intervals = {"M1": 1.0, "M2": 1.0, "C": 5.0, "F": 10.0}
        return intervals[self.region]

    @property
    def image_transforms(self: Self) -> tuple[ccrs.Projection, pyproj.CRS]:
        """Return the cartopy and pyproj projections for the GOES region.

        Uses default projection attributes if data was not loaded successfully.
        This allows the detector to work even if pycontrails datalib is unavailable.
        """
        # Get projection attributes from pycontrails datalib or cache
        goes_attrs = self._get_projection_attrs

        # Define cartopy projection
        cartopy_crs = ccrs.Geostationary(
            satellite_height=goes_attrs["perspective_point_height"],
            central_longitude=goes_attrs["longitude_of_projection_origin"],
            sweep_axis=goes_attrs["sweep_angle_axis"],
        )

        # Define pyproj projection
        pyproj_crs = pyproj.CRS(
            proj="geos",
            h=goes_attrs["perspective_point_height"],
            lon_0=goes_attrs["longitude_of_projection_origin"],
            sweep=goes_attrs["sweep_angle_axis"],
            a=goes_attrs["semi_major_axis"],
            rf=goes_attrs["inverse_flattening"],
        )

        return cartopy_crs, pyproj_crs

    @property
    def sat_pyproj_crs(self: Self) -> pyproj.CRS:
        """Return the pyproj CRS for the satellite projection."""
        _, pyproj_crs = self.image_transforms
        return pyproj_crs

    @property
    def satellite_height(self: Self) -> float:
        """Get the satellite height above Earth's surface.

        Returns:
            Satellite height in meters.
        """
        return self._get_projection_attrs["perspective_point_height"]

    @cached_property
    def _get_projection_attrs(self: Self) -> dict:
        """Return GOES imager projection attributes.

        Reads from live data if available, otherwise falls back to the JSON
        projection cache written by ``_save_projection_cache()``.

        Raises:
            RuntimeError: If neither live data nor a cached file is available.
            ValueError: If the data puts the satellite more than 2° from its operational
                position, e.g. data from a post-launch test period or a stale cache.
        """
        cache_path = DATA_DIR / "geostationary_data" / f"{self.satellite}_projection_attrs.json"
        if self.da_goes is not None:
            attrs = self.da_goes.attrs["goes_imager_projection"]
            source = "the downloaded data"
        else:
            source = f"the projection cache at {cache_path} (delete it to rebuild it)"
            if cache_path.exists():
                with open(cache_path) as f:
                    attrs = json.load(f)
                logging.info(f"Loaded projection attributes from cache: {cache_path}")
            else:
                raise RuntimeError(
                    f"Projection attributes unavailable for {self.satellite}: "
                    f"no live data loaded and no cache found at {cache_path}. "
                    f"Run with a valid timestamp to populate the cache."
                )

        # The image's pixel positions are angles from the satellite's own position, so a
        # satellite away from its operational position (post-launch testing at 89.5°W, or a
        # stale cache) cannot be corrected by swapping the longitude: refuse it instead
        data_lon = attrs.get("longitude_of_projection_origin")
        expected_lon = self.SATELLITE_LONGITUDES.get(self.satellite)

        if expected_lon is not None and data_lon is not None:
            if abs(data_lon - expected_lon) > 2.0:
                raise ValueError(
                    f"{source} places {self.satellite} at {data_lon}°, but its operational "
                    f"position is {expected_lon}°. Locations computed from it would be wrong."
                )

        return attrs

    def fetch_image(
        self: Self,
        image_time: np.datetime64,
    ) -> tuple[np.ndarray, ccrs.Projection, np.ndarray]:
        """Fetch and prepare GOES satellite data for visualization.

        Args:
            image_time: Timestamp for the time of interest.

        Returns:
             tuple[np.ndarray, Any, np.ndarray]:
                 - rgb: A 3D array representing the RGB visualization of the GOES data.
                 - transform: The affine transform matrix for the data, used for plotting on maps.
                 - extent: The extent of the data in the coordinate system, used for plotting.
        """
        # Using pycontrails to load GOES data
        pd_time = pd.to_datetime(image_time)
        try:
            raw_data = self.pycon_goes.get(pd_time)
            rgb, transform, extent = extract_goes_visualization(
                da=raw_data,
                color_scheme=self.color_scheme,
            )

            return rgb, transform, extent

        except (FileNotFoundError, RuntimeError) as e:
            # pycontrails raises FileNotFoundError("No data found ...") for a missing image
            if "No data found" in str(e):
                logging.warning(f"No data found at {pd_time} for region '{self.region}', skipping.")
                return None, None, None
            raise

    def plot_detections(
        self: Self,
        fig_output_dir: str | Path | None = None,
        timestamps: list[np.datetime64] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[plt.Figure]:
        """Plot the Ash-RGB image beside the detection mask for each timestamp.

        Depending on the sector type, it calls the appropriate plotting function:
            - Mesoscale sector: `plot_mesoscale`
            - CONUS sector: `plot_conus`
            - Full Disk sector: `plot_full_disk`

        Args:
            fig_output_dir: Directory in which to save the figures, as
                ``<YYYY-MM-DD>/<satellite>-<region>-<YYYY-MM-DDTHHMM>.png``. Each figure is
                closed once saved, so long time ranges do not build up open figures. If
                None, nothing is saved and the figures are returned.
            timestamps: List of timestamps to plot. If None, uses the full timestamp sequence.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                Full Disk only; if None, the map is cropped to the disk.

        Returns:
            The figures if ``fig_output_dir`` is None, otherwise an empty list.
        """
        # Use full timestamp sequence if none provided
        if timestamps is None:
            timestamps = self.timestamp_sequence

        # Ensure detections data is loaded before plotting
        if self.detections is None:
            raise ValueError("Detections data is not loaded.")

        if self.region in ["M1", "M2"]:
            plot_frame, sector = self.plot_mesoscale, "Mesoscale"
        elif self.region == "C":
            plot_frame, sector = self.plot_conus, "CONUS"
        elif self.region == "F":
            plot_frame, sector = partial(self.plot_full_disk, bbox=bbox), "Full Disk"
        else:
            raise ValueError(f"Unknown GOES region {self.region!r}.")
        return self._plot_frames(
            plot_frame, timestamps, fig_output_dir, desc=f"Plotting {sector} detections"
        )

    def plot_conus(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot GOES CONUS sector detections.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.

        Returns:
            Matplotlib Figure, or None if the image or detection could not be loaded.
        """
        return self._plot_sector(timestamp, fig_output_dir)

    def plot_full_disk(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> plt.Figure | None:
        """Plot GOES Full Disk sector detections with borders, coastlines and gridlines.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to the disk.

        Returns:
            Matplotlib Figure, or None if the image or detection could not be loaded.
        """
        return self._plot_sector(timestamp, fig_output_dir, full_disk=True, bbox=bbox)

    def plot_mesoscale(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot GOES Mesoscale sector detections.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.

        Returns:
            Matplotlib Figure, or None if the image or detection could not be loaded.
        """
        return self._plot_sector(timestamp, fig_output_dir)

    @plt.rc_context({"font.size": 10})
    def _plot_sector(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
        full_disk: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> plt.Figure | None:
        """Plot the Ash-RGB image beside the detection mask, saving it if a directory is given.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.
            full_disk: Draw borders, coastlines and gridlines and plot the Ash-RGB with
                ``imshow`` in its native projection.
            bbox: Bounding box (min_lon, max_lon, min_lat, max_lat). Full Disk only; if None,
                the map is cropped to 80 degrees either side of the sub-satellite point.

        Returns:
            Matplotlib Figure, or None if the image or detection could not be loaded.
        """
        year, month, _, dom, hour, minute = extract_date_time_components(timestamp)
        date_str = f"{year:04d}-{month:02d}-{dom:02d}"
        timestamp_str = f"{date_str}T{hour:02d}{minute:02d}"

        try:
            da_goes = self.pycon_goes.get(pd.to_datetime(timestamp))
            ash_rgb, ash_transform, ash_extent = extract_goes_visualization(
                da=da_goes,
                color_scheme=self.color_scheme,
            )
        except Exception as e:
            logging.warning(
                f"Failed to load GOES Ash-RGB data from pycontrails for {timestamp_str}: {e}"
            )
            return None

        try:
            prediction = self.detections[timestamp]
        except Exception as e:
            logging.warning(
                f"Failed to load GOES detection from zarr store for {timestamp_str}: {e}"
            )
            return None

        geo_x, geo_y = self.image_geo_coords
        geo_crs, _ = self.image_transforms

        if full_disk:
            map_crs, extent, extent_crs = self._full_disk_map(bbox)
        else:
            map_crs = transforms.platecarree()

        # Drawn at screen resolution, saved at 400 dpi
        fig = plt.figure(figsize=(10.0, 5.0), dpi=150)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        with sns.axes_style(colors.plot_params_light):
            ax1 = fig.add_subplot(121, projection=map_crs)
            ax2 = fig.add_subplot(122, projection=map_crs)
            titles = [
                f"{self.satellite} Ash-RGB",
                f"Detection ({timestamp_str}, {self.region.upper()})",
            ]
            for ax, title in zip([ax1, ax2], titles, strict=True):
                ax.text(
                    x=0.5,
                    y=1.05,
                    s=title,
                    fontsize=10,
                    color="k",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            if full_disk:
                for ax in [ax1, ax2]:
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

        if full_disk:
            ax1.imshow(ash_rgb, extent=ash_extent, transform=ash_transform)
        else:
            ax1.pcolormesh(geo_x, geo_y, ash_rgb, transform=geo_crs)
        pred_map = ax2.pcolormesh(
            geo_x,
            geo_y,
            prediction,
            cmap="Blues",
            norm=mpl_colors.Normalize(vmin=0, vmax=1),
            transform=geo_crs,
        )
        if full_disk:
            for ax in [ax1, ax2]:
                ax.set_extent(extent, crs=extent_crs)

        # [left, bottom, width, height]
        pos2 = ax2.get_position()
        cax = fig.add_axes([pos2.x1 + 0.02, pos2.y0, 0.02, pos2.height])
        plt.colorbar(pred_map, cax=cax, orientation="vertical", label="Detection Score")

        self._save_frame_figure(fig, fig_output_dir, timestamp, dpi=400)
        return fig

    def spans_single_epoch(self: Self) -> bool | None:
        """Whether this detector's time range stays within one Mesoscale sector position.

        Compares the grid at the first and last timestamp. Will not catch a sector that leaves
        a position and returns to it within the range.

        Returns:
            True if both ends share a grid, False if they differ, None for F/C or if either
            grid could not be retrieved.
        """
        if not self.has_shifting_domain:
            return None
        if len(self.timestamp_sequence) < 2:
            return True

        first = self._grid_fingerprint_at(self.timestamp_sequence[0])
        last = self._grid_fingerprint_at(self.timestamp_sequence[-1])
        if first is None or last is None:
            logging.warning(
                f"Could not determine epoch span for {self.satellite}-{self.region}: "
                f"grid unavailable at one or both ends of the time range"
            )
            return None

        return first == last

    def _frame_extent(self: Self, da: xr.DataArray) -> tuple[float, float, float, float]:
        """Extent of a GOES frame, as pycontrails reports it for visualization."""
        _, _, extent = extract_goes_visualization(da=da, color_scheme=self.color_scheme)
        return extent

    def _frame_projection(
        self: Self, da: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray, pyproj.CRS]:
        """Pixel coordinates and CRS of a GOES frame, from its imager projection attributes."""
        attrs = da.attrs["goes_imager_projection"]
        crs = pyproj.CRS(
            proj="geos",
            h=attrs["perspective_point_height"],
            lon_0=attrs["longitude_of_projection_origin"],
            sweep=attrs["sweep_angle_axis"],
            a=attrs["semi_major_axis"],
            rf=attrs["inverse_flattening"],
        )
        height = attrs["perspective_point_height"]
        return height * da["x"].values, height * da["y"].values, crs

    def _gen_scan_time_offset(self: Self) -> np.ndarray:
        """Time from a frame's scan start to the moment each pixel was scanned.

        NOAA STAR's Mode 6 time-model table for the scene (see ``abi_time_model``) plus the
        detection bands' offset from Band 2. The same for every frame; each frame's own scan
        start is added by ``scan_start_offsets_s``.

        Returns:
            Per-pixel offsets from the scan start, timedelta64 in milliseconds, shaped like
            the grid.

        Raises:
            ValueError: If the table does not match the grid.
        """
        bands = [int(b.lstrip("C")) for b in self.GOES_BANDS]
        offset_s = abi_time_model.pixel_time_offsets_s(self.satellite, self.region, 0.0, bands)
        if offset_s.shape != self.lats.shape:
            raise ValueError(
                f"{self.satellite}-{self.region}: ABI time-model table is {offset_s.shape}, "
                f"but the detector grid is {self.lats.shape}."
            )
        return np.round(offset_s * 1000).astype(np.int64) * np.timedelta64(1, "ms")

    def _gen_tiles(self: Self) -> None:
        """Generate tiles for the region."""
        # Ensure timestamp sequence is populated
        if not self.timestamp_sequence:
            raise ValueError("Timestamp sequence is empty. Must populate before tile generation.")

        # Slice four tiles (256x256) from the Mesoscale image (500x500)
        if self.region in ["M1", "M2"]:
            # Slice four tiles (256x256) from the Mesoscale image (500x500)
            tile_lons, tile_lats = [], []
            tile_idx = np.full_like(self.lons, fill_value=-1, dtype=int)
            for idx in range(4):
                start_x = 244 * (idx // 2)
                end_x = start_x + 256

                start_y = 244 * (idx % 2)
                end_y = start_y + 256

                tile_lons.append(np.nanmean(self.lons[start_y:end_y, start_x:end_x]) + 360.0)
                tile_lats.append(np.nanmean(self.lats[start_y:end_y, start_x:end_x]))
                tile_idx[start_y:end_y, start_x:end_x] = idx

            self.tile_lons = np.array(tile_lons)
            self.tile_lats = np.array(tile_lats)
            self.tile_idx = np.array(tile_idx)

        elif self.region in ["C", "F"]:
            # Generate tiles by decomposing grid
            self._decompose_abi_fixed_grid(
                extent=self.tile_extent,
                overlap=32.0,
                tile_size=512.0,
            )

    def _get_frame(self: Self, time: pd.Timestamp) -> xr.DataArray:
        """Fetch the GOES frame at one time through the pycontrails datalib."""
        return self.pycon_goes.get(time)

    def _get_region_coords(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """Get the coordinates for the GOES region.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array represents the longitude meshgrid.
                - The second array represents the latitude meshgrid.
        """
        # Get projection attributes
        goes_x, goes_y = self.image_geo_coords

        # Compute lat/lon coordinates of ABI fixed grid
        xx, yy = np.meshgrid(goes_x, goes_y, indexing="xy")

        goes_lons, goes_lats = self.sat_to_latlon_transform.transform(xx, yy)
        goes_lons[np.isinf(goes_lons)] = np.nan
        goes_lats[np.isinf(goes_lats)] = np.nan

        return goes_lons, goes_lats

    def _grid_fingerprint_at(self: Self, timestamp: np.datetime64) -> tuple | None:
        """Identify which position an M1/M2 sector was in at one timestamp.

        Used only by ``spans_single_epoch``, so never called for F or C. Frames in an epoch
        share their coordinates exactly, so the end values of x and y distinguish two
        positions. Rounding absorbs float round-tripping only; adjacent sector positions are
        hundreds of kilometres apart.

        Args:
            timestamp: Time to look up.

        Returns:
            Hashable fingerprint of the grid, or None if the frame could not be fetched.
        """
        try:
            da = self.pycon_goes.get(pd.Timestamp(timestamp))
            attrs = da.attrs["goes_imager_projection"]
            height = attrs["perspective_point_height"]
            geo_x = height * np.asarray(da["x"].values, dtype=float)
            geo_y = height * np.asarray(da["y"].values, dtype=float)
        except Exception as e:  # noqa: BLE001 - absence is a valid answer here
            logging.warning(f"No grid available at {timestamp}: {type(e).__name__}: {e}")
            return None

        return (
            round(float(geo_x[0]), 3),
            round(float(geo_x[-1]), 3),
            round(float(geo_y[0]), 3),
            round(float(geo_y[-1]), 3),
        )

    def _save_projection_cache(self: Self) -> None:
        """Save projection attrs to cache after successful load.

        Only writes if the cache is new or data has actually changed.
        """

        try:
            attrs = self.da_goes.attrs["goes_imager_projection"].copy()

            # Never cache a projection from away from the operational position: it is not
            # correctable, and _get_projection_attrs refuses it when it is used
            data_lon = attrs.get("longitude_of_projection_origin")
            expected_lon = self.SATELLITE_LONGITUDES.get(self.satellite)

            if expected_lon is not None and data_lon is not None:
                if abs(data_lon - expected_lon) > 2.0:
                    logging.warning(
                        f"Not caching the projection for {self.satellite}: the data places it "
                        f"at {data_lon}°, not its operational position {expected_lon}°."
                    )
                    return

            cache_dir = DATA_DIR / "geostationary_data"
            cache_path = cache_dir / f"{self.satellite}_projection_attrs.json"
            cache_x_path = cache_dir / f"{self.satellite}_{self.region}_x.npy"
            cache_y_path = cache_dir / f"{self.satellite}_{self.region}_y.npy"

            cache_dir.mkdir(parents=True, exist_ok=True)

            # Save x and y coords only for fixed-domain regions (C, F).
            # Mesoscale (M1, M2) domains shift every epoch, so cached x/y would be wrong
            # for any subsequent epoch and must not be written.
            # For fixed regions: save only if absent — coords are constant per satellite+region
            # and unconditional overwrite causes a race condition under parallel joblib workers
            # (np.save truncates before writing; concurrent readers see an empty file).
            # Atomic temp→rename avoids partial reads even on the first write.
            if self.region not in ["M1", "M2"] and not (
                cache_x_path.exists() and cache_y_path.exists()
            ):
                x_coords = np.float64(self.da_goes["x"].values)
                y_coords = np.float64(self.da_goes["y"].values)
                tmp_x = cache_x_path.with_suffix(".tmp.npy")
                tmp_y = cache_y_path.with_suffix(".tmp.npy")
                np.save(tmp_x, x_coords)
                np.save(tmp_y, y_coords)
                tmp_x.replace(cache_x_path)
                tmp_y.replace(cache_y_path)

            # Don't rewrite if cache already has the same data
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        existing_cache = json.load(f)
                    if existing_cache == attrs:
                        logging.debug("Cache already up-to-date, skipping write")
                        return  # Cache is already correct, no need to rewrite
                except Exception as e:
                    logging.debug(f"Couldn't read existing cache: {e}, will rewrite")

            with open(cache_path, "w") as f:
                json.dump(attrs, f, indent=2)
            logging.info(f"Saved projection attributes cache to {cache_path}")

        except Exception as e:
            logging.warning(f"Failed to save projection cache: {e}")

    def _read_frame_files(self: Self, time: pd.Timestamp) -> list[tuple[pd.Timestamp, str | None]]:
        """Scan start and ``timeline_id`` from each band file of the frame at ``time``.

        pycontrails' ``GOES.get`` drops both, so they're read from the cached band files, or
        from one remote band file if the frame isn't cached.

        Returns:
            (scan start, timeline_id) for each band file, or an empty list if they can't be read.
        """
        pycon = getattr(self, "pycon_goes", None)
        if pycon is None:
            return []
        out = []
        try:
            # _lpaths is private: the cache files get() writes, one per band
            lpaths = list(pycon._lpaths(time).values()) if pycon.cachestore is not None else []
            if lpaths and all(pycon.cachestore.exists(lp) for lp in lpaths):
                for lpath in lpaths:
                    with xr.open_dataset(lpath) as ds:
                        out.append(
                            (pd.Timestamp(ds["time_bounds"].values[0]), ds.attrs.get("timeline_id"))
                        )
            else:
                # time_bounds and timeline_id are the same in every band, so one file will do
                for rpath in pycon.gcs_goes_path(time, {self.GOES_BANDS[0]}):
                    with pycon.fs.open(rpath) as f, xr.open_dataset(f, engine="h5netcdf") as ds:
                        out.append(
                            (pd.Timestamp(ds["time_bounds"].values[0]), ds.attrs.get("timeline_id"))
                        )
        except Exception as e:
            logging.warning(
                f"{self.satellite}-{self.region}: couldn't read the files for {time} ({e})."
            )
            return []
        return out

    def _scan_start_at(self: Self, time: pd.Timestamp) -> tuple[float | None, str | None]:
        """Seconds from the frame's nominal time to its scan start, and its timeline_id.

        The scan start is ``time_bounds[0]``, which is the same in every band file. It is None
        if the files can't be read or the start is implausible (the frame was matched to the
        wrong nominal time). Cached per frame.

        Raises:
            ValueError: If the band files disagree on timeline_id. They come from one scan, so
                that means a stale or mixed-up cache, and the frame's detections are suspect.
        """
        cache = self.__dict__.setdefault("_scan_start_cache", {})
        time = pd.Timestamp(time)
        if time in cache:
            return cache[time]

        files = self._read_frame_files(time)
        if not files:
            cache[time] = (None, None)
            return cache[time]

        start_s = (min(t for t, _ in files) - time).total_seconds()
        timelines = {tl for _, tl in files if tl is not None}
        if len(timelines) > 1:
            raise ValueError(
                f"{self.satellite}-{self.region}: band files at {time} disagree on "
                f"timeline_id: {sorted(timelines)}. Check the cache for stale files."
            )
        timeline_id = timelines.pop() if timelines else None

        if not -60.0 <= start_s <= 120.0:
            logging.warning(
                f"{self.satellite}-{self.region}: frame at {time} reports a scan start "
                f"{start_s:.2f} s away; ignoring it."
            )
            start_s = None
        cache[time] = (start_s, timeline_id)
        return cache[time]

    def _fallback_scan_start_s(self: Self) -> float:
        """Start for a frame whose own can't be read: the first frame's, else SCAN_PARAMETERS."""
        if "_fallback_start_s" not in self.__dict__:
            start_s, _ = self._scan_start_at(self.timestamp_sequence[0])
            if start_s is None:
                start_s = self.SCAN_PARAMETERS[self.region]["overhead_s"]
                logging.warning(
                    f"{self.satellite}-{self.region}: no usable scan start for the first "
                    f"frame, using {start_s:.2f} s where a frame's own can't be read."
                )
            self._fallback_start_s = start_s
        return self._fallback_start_s

    def scan_start_offsets_s(self: Self, timestamps: np.ndarray) -> np.ndarray:
        """Each frame's scan start, in seconds after its nominal time.

        Read from the frame's own files (``time_bounds[0]``). A frame whose files can't be
        read gets the first frame's start. A frame not in Mode 6 gets NaN, since the time
        model doesn't describe it, so its pixels have no scan time.

        Args:
            timestamps: Nominal frame times.

        Returns:
            float64 array shaped like ``timestamps``.
        """
        times = np.asarray(timestamps, dtype="datetime64[ns]")
        unique, inverse = np.unique(times, return_inverse=True)
        starts = np.full(unique.shape, np.nan)
        for k, t in enumerate(unique):
            if np.isnat(t):
                continue
            start_s, timeline_id = self._scan_start_at(pd.Timestamp(t))
            if self._is_supported_timeline(t, timeline_id):
                starts[k] = self._fallback_scan_start_s() if start_s is None else start_s
        return starts[inverse].reshape(times.shape)

    def _is_supported_timeline(self: Self, time: np.datetime64, timeline_id: str | None) -> bool:
        """True unless the frame's files say it isn't Mode 6 (warns once per frame)."""
        try:
            if timeline_id is not None:
                abi_time_model.check_timeline(timeline_id)
            return True
        except ValueError:
            warned = self.__dict__.setdefault("_timeline_warned", set())
            if time not in warned:
                warned.add(time)
                logging.warning(
                    f"{self.satellite}-{self.region}: frame at {time} is {timeline_id!r}, "
                    "which has no time model; its pixels get no scan time."
                )
            return False

    def _set_central_lon(self: Self) -> None:
        """Set the central longitude based on the satellite."""
        self.central_lon = self._get_projection_attrs["longitude_of_projection_origin"]
