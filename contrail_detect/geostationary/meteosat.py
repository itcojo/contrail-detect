"""
Contrail detection using FCI data from Meteosat family satellites.

"""

import json
import logging
from functools import partial
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import ephem
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from contrail_detect.core.common import Colors, Transforms
from contrail_detect.core.config import DATA_DIR, PACKAGE_DIR, RESULT_DIR
from contrail_detect.geostationary import validation_utils
from contrail_detect.geostationary.fci_base import FCIDetector

# Standard values
colors = Colors()
transforms = Transforms()


class MeteosatDetector(FCIDetector):
    """Contrail detection using FCI data from Meteosat series satellites.

    Implements contrail detection for Europe's Meteosat Third Generation (MTG)
    satellites, Full Disk only. Retrieves pre-computed detection results from the Google
    Contrails API, which serves MTG Full Disk only.

    Attributes:
        VALID_REGIONS: Supported scan regions (``'F'`` for Full Disk).
        VALID_SATELLITES: Supported satellites (``'MTG'``).
    """

    VALID_REGIONS = ["F"]  # Full Disk; the Google API has no Rapid Scan detections
    VALID_SATELLITES = ["MTG"]

    def __init__(
        self: Self,
        meteosat_satellite: str,
        meteosat_region: str,
        start_time: np.datetime64,
        end_time: np.datetime64,
        *,
        color_scheme: str = "ash",
    ) -> None:
        """Initialize the MeteosatDetector for specified satellite, region, and time range.

        Args:
            meteosat_satellite: Satellite identifier (only ``'MTG'``).
            meteosat_region: Scan region (only ``'F'``, Full Disk).
            start_time: Start of observation period.
            end_time: End of observation period.
            color_scheme: Visualization color scheme. Defaults to ``'ash'``.

        Raises:
            ValueError: If invalid satellite or region is specified.
        """
        super().__init__()

        self.color_scheme: str = color_scheme
        self.region: str = meteosat_region.upper()
        self.satellite: str = meteosat_satellite.upper()
        self.sweep_axis = "y"

        if self.region == "R":
            raise ValueError(
                "MTG Rapid Scan (region 'R') is not supported: the Google Contrails API, the "
                "only detection source for MTG, serves Full Disk only. Use region 'F'."
            )
        validation_utils.validate_satellite_inputs(
            self.region,
            self.satellite,
            set(self.VALID_REGIONS),
            set(self.VALID_SATELLITES),
        )

        # Path to detection results
        self.DETECT_PATH: Path = Path(RESULT_DIR) / self.satellite / self.region

        # Create timestamp sequence
        self.create_timestamp_sequence(
            start_time=start_time,
            end_time=end_time,
        )

        # Long name of detector (used in file construction)
        self.LONG_NAME = self.gen_detector_long_name()

        # Cached projection properties (populated lazily from FCI data or local cache)
        self._cached_satellite_height: float | None = None
        self._cached_image_full_extent: list | None = None
        self._cached_image_geo_coords: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_projection_attrs: dict | None = None
        self._cached_cartopy_crs: ccrs.Projection | None = None
        self._cached_pyproj_crs: pyproj.CRS | None = None

        # Attempt to load metadata from cache to allow offline usage
        self._load_projection_cache()

    def _repr_fields(self: Self) -> dict:
        return {**super()._repr_fields(), "color_scheme": self.color_scheme}

    @property
    def image_full_extent(self: Self) -> list | None:
        """Return the full extent of the image in geostationary coordinates.

        Populated from the local projection cache or by ``populate_metadata_from_dataset``.

        Returns:
            List of [x_min, x_max, y_min, y_max] in meters, or None if not yet populated.
        """
        return self._cached_image_full_extent

    @property
    def image_geo_coords(self: Self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the geostationary x and y coordinates of the image pixels.

        Populated from the local projection cache or by ``populate_metadata_from_dataset``.

        Returns:
            Tuple of (x, y) arrays in meters, or None if not yet populated.
        """
        return self._cached_image_geo_coords

    @property
    def image_interval(self: Self) -> float:
        """Get image acquisition interval for the scan region.

        Returns:
            Image interval in minutes (10.0 for Full Disk).
        """
        return 10.0

    @property
    def image_transforms(self: Self) -> tuple[ccrs.Projection, pyproj.CRS] | None:
        """Get the image projection and CRS for the scan region.

        Populated from the local projection cache or by ``populate_metadata_from_dataset``.

        Returns:
            Tuple of (cartopy projection, pyproj CRS), or None if not yet populated.
        """
        if self._cached_cartopy_crs is None or self._cached_pyproj_crs is None:
            return None

        return self._cached_cartopy_crs, self._cached_pyproj_crs

    @property
    def sat_pyproj_crs(self: Self) -> pyproj.CRS | None:
        """Get the satellite's pyproj CRS object.

        Populated from the local projection cache or by ``populate_metadata_from_dataset``.

        Returns:
            Pyproj CRS object, or None if not yet populated.
        """
        return self._cached_pyproj_crs

    @property
    def satellite_height(self: Self) -> float | None:
        """Get the satellite height above Earth's surface.

        Populated from the local projection cache or by ``populate_metadata_from_dataset``.

        Returns:
            Satellite height in meters, or None if not yet populated.
        """
        return self._cached_satellite_height

    def fetch_image(self: Self, image_time: np.datetime64) -> NotImplementedError:
        raise NotImplementedError(
            "Direct image fetching for MeteosatDetector is not yet implemented. "
        )

    def plot_detections(
        self: Self,
        fig_output_dir: str | Path | None = None,
        timestamps: list[np.datetime64] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        detection_cmap: str | mpl_colors.Colormap = "Blues",
        prediction_threshold: float | None = None,
        show_domain: bool = False,
    ) -> list[plt.Figure]:
        """Plot the detection mask for each timestamp.

        Args:
            fig_output_dir: Directory in which to save the figures, as
                ``<YYYY-MM-DD>/<satellite>-<region>-<YYYY-MM-DDTHHMM>.png``. Each figure is
                closed once saved, so long time ranges do not build up open figures. If
                None, nothing is saved and the figures are returned.
            timestamps: List of timestamps to plot. If None, uses the full timestamp sequence.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to 80 degrees either side of the sub-satellite point.
            detection_cmap: Colormap with which to plot detection mask.
            prediction_threshold: Pixels with a detection score below this
                value are hidden. If None, uses the default for the detection source.
            show_domain: Whether or not to outline the processed imaging domain.

        Returns:
            The figures if ``fig_output_dir`` is None, otherwise an empty list.

        Raises:
            ValueError: If detections have not been loaded on the detector.
        """
        if timestamps is None:
            timestamps = self.timestamp_sequence

        if self.detections is None:
            raise ValueError("Detections data is not loaded.")

        plot_frame = partial(
            self.plot_full_disk,
            bbox=bbox,
            detection_cmap=detection_cmap,
            prediction_threshold=prediction_threshold,
            show_domain=show_domain,
        )
        return self._plot_frames(
            plot_frame, timestamps, fig_output_dir, desc="Plotting Full Disk detections"
        )

    def plot_full_disk(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        detection_cmap: str | mpl_colors.Colormap = "Blues",
        prediction_threshold: float | None = None,
        show_domain: bool = False,
    ) -> plt.Figure | None:
        """Plot one frame's MTG Full Disk detection mask.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to 80 degrees either side of the sub-satellite point.
            detection_cmap: Colormap with which to plot detection mask.
            prediction_threshold: Pixels with a detection score below this
                value are hidden. If None, uses the default for the detection source.
            show_domain: Whether or not to outline the processed imaging domain.

        Returns:
            Matplotlib Figure, or None if the detection could not be loaded.
        """
        try:
            prediction = self.detections[timestamp]
        except Exception as e:
            logging.warning(f"Failed to load MTG detection for {timestamp}: {e}")
            return None

        return self._plot_detection_map(
            timestamp,
            prediction,
            fig_output_dir=fig_output_dir,
            detection_cmap=detection_cmap,
            prediction_threshold=prediction_threshold,
            show_domain=show_domain,
            bbox=bbox,
        )

    def populate_metadata_from_dataset(self: Self, ds: xr.Dataset) -> None:
        """Populate projection metadata from an xarray dataset.

        This method extracts projection parameters from a loaded FCI dataset and
        caches them for use by various properties.

        Args:
            ds: xarray Dataset or DataArray containing FCI data with projection metadata.
        """
        # Extract x and y coordinates
        x_coords = np.asarray(
            ds["x"].values if "x" in ds else ds.coords["x"].values,
            dtype=np.float64,
        )
        y_coords = np.asarray(
            ds["y"].values if "y" in ds else ds.coords["y"].values,
            dtype=np.float64,
        )

        # Try to extract projection attributes from dataset
        attrs = ds.attrs if hasattr(ds, "attrs") else {}

        # Look for projection info in various attribute formats
        projection_info = {}

        # Common attribute names for geostationary satellites
        if "perspective_point_height" in attrs:
            projection_info["satellite_height"] = float(attrs["perspective_point_height"])
        elif "h" in attrs:
            projection_info["satellite_height"] = float(attrs["h"])
        else:
            # Default MTG satellite height
            projection_info["satellite_height"] = 35786023.0

        projection_info["central_longitude"] = float(
            attrs.get("longitude_of_projection_origin", 0.0)
        )
        projection_info["sweep_axis"] = str(attrs.get("sweep_angle_axis", "y"))
        projection_info["semi_major_axis"] = float(attrs.get("semi_major_axis", 6378137.0))
        projection_info["inverse_flattening"] = float(
            attrs.get("inverse_flattening", 298.257223563)
        )

        # Cache satellite height
        self._cached_satellite_height = projection_info["satellite_height"]

        # Cache projection attributes for image_transforms
        self._cached_projection_attrs = projection_info

        # Set central longitude
        self.central_lon = projection_info["central_longitude"]

        # Store geostationary coordinates
        self._cached_image_geo_coords = (x_coords, y_coords)
        self._cached_image_full_extent = [
            float(x_coords.min()),
            float(x_coords.max()),
            float(y_coords.min()),
            float(y_coords.max()),
        ]

        # Explicit globe to ensure exact identical Cartopy interpretation
        globe = ccrs.Globe(
            semimajor_axis=projection_info["semi_major_axis"],
            semiminor_axis=projection_info["semi_major_axis"]
            * (1.0 - 1.0 / projection_info["inverse_flattening"]),
            ellipse=None,
        )

        # Create and cache cartopy projection
        self._cached_cartopy_crs = ccrs.Geostationary(
            satellite_height=projection_info["satellite_height"],
            central_longitude=projection_info["central_longitude"],
            sweep_axis=projection_info["sweep_axis"],
            globe=globe,
        )

        # Create and cache pyproj CRS
        self._cached_pyproj_crs = pyproj.CRS(
            proj="geos",
            h=projection_info["satellite_height"],
            lon_0=projection_info["central_longitude"],
            sweep=projection_info["sweep_axis"],
            a=projection_info["semi_major_axis"],
            rf=projection_info["inverse_flattening"],
        )

        # Compute lat/lon coordinates from geostationary projection
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")

        # Create pyproj transformer from geostationary to lat/lon
        ll_crs = pyproj.CRS("WGS84")
        transformer = pyproj.Transformer.from_crs(
            crs_from=self._cached_pyproj_crs,
            crs_to=ll_crs,
            always_xy=True,
        )

        # Transform to lat/lon
        self.lons, self.lats = transformer.transform(xx, yy)
        self.lons = np.where(np.isfinite(self.lons), self.lons, np.nan)
        self.lats = np.where(np.isfinite(self.lats), self.lats, np.nan)

        # Set domain boundaries from computed lat/lon grid
        self.min_longitude = float(np.nanmin(self.lons))
        self.max_longitude = float(np.nanmax(self.lons))
        self.min_latitude = float(np.nanmin(self.lats))
        self.max_latitude = float(np.nanmax(self.lats))

        # Set tile extent to full domain
        self.tile_extent = (
            self.min_longitude,
            self.max_longitude,
            self.min_latitude,
            self.max_latitude,
        )

        self._save_projection_cache()

    def _gen_scan_time_offset(self: Self) -> np.ndarray:
        """Calculate the real scan time for the pixel corresponding to
        each longitude and latitude the MTG satellite image.

        Scanning pattern alternates beween summer and winter scan cycles.

        The switch between the two is taken to be the astronomical equinox. EUMETSAT may
        change the scan pattern on a nearby operational date instead, so scan times for
        images within a few days of an equinox may come from the wrong cycle.

        Returns:
            np.ndarray: The calculated scan time offsets for each pixel in the grid,
                as timedelta64 values.
        """
        # Get specific astronomical events for the year of the timestamp
        yr = pd.to_datetime(self.timestamp_sequence[0]).year

        # Get spring and autumn equinoxes for the current year
        spring_eq_curr_dt = ephem.next_spring_equinox(f"{yr}-01-01").datetime()
        spring_eq_curr = np.datetime64(spring_eq_curr_dt)

        autumn_eq_curr_dt = ephem.next_autumn_equinox(spring_eq_curr_dt).datetime()
        autumn_eq_curr = np.datetime64(autumn_eq_curr_dt)

        # Also get autumn equinox from previous year (for timestamps before spring equinox)
        autumn_eq_prev_dt = ephem.next_autumn_equinox(f"{yr - 1}-01-01").datetime()
        autumn_eq_prev = np.datetime64(autumn_eq_prev_dt)

        # Get spring equinox for next year (winter cycle boundary)
        spring_eq_next_dt = ephem.next_spring_equinox(autumn_eq_curr_dt).datetime()
        spring_eq_next = np.datetime64(spring_eq_next_dt)

        def _is_summer(ts: np.datetime64) -> bool:
            """Determine if a given timestamp falls within the summer scan cycle.

            Summer runs from the spring equinox to the autumn equinox of the current year.
            """
            return (ts >= spring_eq_curr) and (ts < autumn_eq_curr)

        def _is_winter(ts: np.datetime64) -> bool:
            """Determine if a given timestamp falls within the winter scan cycle.

            Winter runs from the autumn equinox (previous year) to the spring equinox (current year),
            or from autumn equinox (current year) to spring equinox (next year).
            """
            return ((ts >= autumn_eq_prev) and (ts < spring_eq_curr)) or (
                (ts >= autumn_eq_curr) and (ts < spring_eq_next)
            )

        # Test if detector is in summer or winter scantime mode
        start_in_summer = _is_summer(self.timestamp_sequence[0])
        start_in_winter = _is_winter(self.timestamp_sequence[0])
        end_in_summer = _is_summer(self.timestamp_sequence[-1])
        end_in_winter = _is_winter(self.timestamp_sequence[-1])

        if start_in_summer and end_in_summer:
            logging.info("Loaded scantime offset for MTG summer scan cycle.")
            scan_time_offset = np.load(
                PACKAGE_DIR / "data" / "geostationary_data" / "mtg_summer_scantime.npz"
            )["scantime_offset"]

        elif start_in_winter and end_in_winter:
            logging.info("Loaded scantime offset for MTG winter scan cycle.")
            scan_time_offset = np.load(
                PACKAGE_DIR / "data" / "geostationary_data" / "mtg_winter_scantime.npz"
            )["scantime_offset"]

        else:
            logging.error(
                f"\nTimestamp sequence spans both summer and winter scan cycles:"
                f"\n  - Timestamps: {self.timestamp_sequence[0]} to {self.timestamp_sequence[-1]}"
                f"\n  - Autumn equinox {yr - 1}: {autumn_eq_prev}"
                f"\n  - Spring equinox {yr}: {spring_eq_curr}"
                f"\n  - Autumn equinox {yr}: {autumn_eq_curr}"
                f"\n  - Spring equinox {yr + 1}: {spring_eq_next}"
            )
            raise ValueError(
                "Timestamp sequence spans both summer and winter scan cycles. "
                "Please ensure the time range falls entirely within one cycle."
            )

        return scan_time_offset * np.timedelta64(1, "s")

    def _gen_tiles(self: Self) -> NotImplementedError:
        raise NotImplementedError("Tile generation for MeteosatDetector is not required.")

    def _load_projection_cache(self: Self) -> None:
        """Load projection metadata from local cache to allow offline execution."""
        try:
            cache_dir = DATA_DIR / "geostationary_data"
            cache_path = cache_dir / f"{self.satellite}_{self.region}_projection_attrs.json"
            cache_x_path = cache_dir / f"{self.satellite}_{self.region}_x.npy"
            cache_y_path = cache_dir / f"{self.satellite}_{self.region}_y.npy"

            if cache_path.exists() and cache_x_path.exists() and cache_y_path.exists():
                with open(cache_path) as f:
                    projection_info = json.load(f)

                x_coords = np.load(cache_x_path)
                y_coords = np.load(cache_y_path)

                # Ensure coordinates are float64 for zarr compatibility
                x_coords = np.asarray(x_coords, dtype=np.float64)
                y_coords = np.asarray(y_coords, dtype=np.float64)
                self._cached_projection_attrs = projection_info
                self._cached_satellite_height = projection_info["satellite_height"]
                self.central_lon = projection_info["central_longitude"]

                self._cached_image_geo_coords = (x_coords, y_coords)
                self._cached_image_full_extent = [
                    float(x_coords.min()),
                    float(x_coords.max()),
                    float(y_coords.min()),
                    float(y_coords.max()),
                ]

                globe = ccrs.Globe(
                    semimajor_axis=projection_info["semi_major_axis"],
                    semiminor_axis=projection_info["semi_major_axis"]
                    * (1.0 - 1.0 / projection_info["inverse_flattening"]),
                    ellipse=None,
                )

                self._cached_cartopy_crs = ccrs.Geostationary(
                    satellite_height=projection_info["satellite_height"],
                    central_longitude=projection_info["central_longitude"],
                    sweep_axis=projection_info["sweep_axis"],
                    globe=globe,
                )

                self._cached_pyproj_crs = pyproj.CRS(
                    proj="geos",
                    h=projection_info["satellite_height"],
                    lon_0=projection_info["central_longitude"],
                    sweep=projection_info["sweep_axis"],
                    a=projection_info["semi_major_axis"],
                    rf=projection_info["inverse_flattening"],
                )

                # Reconstruct min/max lon/lat
                xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
                ll_crs = pyproj.CRS("WGS84")
                transformer = pyproj.Transformer.from_crs(
                    crs_from=self._cached_pyproj_crs,
                    crs_to=ll_crs,
                    always_xy=True,
                )
                self.lons, self.lats = transformer.transform(xx, yy)
                self.lons = np.where(np.isfinite(self.lons), self.lons, np.nan)
                self.lats = np.where(np.isfinite(self.lats), self.lats, np.nan)

                self.min_longitude = float(np.nanmin(self.lons))
                self.max_longitude = float(np.nanmax(self.lons))
                self.min_latitude = float(np.nanmin(self.lats))
                self.max_latitude = float(np.nanmax(self.lats))

                self.tile_extent = (
                    self.min_longitude,
                    self.max_longitude,
                    self.min_latitude,
                    self.max_latitude,
                )
                logging.info(f"Successfully loaded MTG projection cache from {cache_path}")

        except Exception as e:
            logging.debug(
                f"Could not load MTG projection cache (this is normal if no data "
                f"has been populated yet): {e}"
            )

    def _save_projection_cache(self: Self) -> None:
        """Save projection attributes and coordinates to local cache."""
        try:
            cache_dir = DATA_DIR / "geostationary_data"
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_path = cache_dir / f"{self.satellite}_{self.region}_projection_attrs.json"
            cache_x_path = cache_dir / f"{self.satellite}_{self.region}_x.npy"
            cache_y_path = cache_dir / f"{self.satellite}_{self.region}_y.npy"

            # Save coordinates
            if self._cached_image_geo_coords is not None:
                x_coords, y_coords = self._cached_image_geo_coords
                np.save(cache_x_path, x_coords)
                np.save(cache_y_path, y_coords)

            # Save projection attributes
            if self._cached_projection_attrs is not None:
                if cache_path.exists():
                    try:
                        with open(cache_path) as f:
                            existing_cache = json.load(f)
                        if existing_cache == self._cached_projection_attrs:
                            logging.debug("MTG Cache already up-to-date, skipping write")
                            return
                    except Exception as e:
                        logging.debug(f"Couldn't read existing cache: {e}, will rewrite")

                with open(cache_path, "w") as f:
                    json.dump(self._cached_projection_attrs, f, indent=2)
                logging.info(f"Saved MTG projection attributes cache to {cache_path}")

        except Exception as e:
            logging.warning(f"Failed to save MTG projection cache: {e}")

    def _set_central_lon(self: Self) -> float:
        """Central longitude of satellite.

        Returns:
            float: Central longitude in degrees.
        """
        return 0.0
