"""
Contrail detection based on ABI (Advanced Baseline Imager) data.

"""

import itertools
import logging
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from contrail_detect.core._plotting import save_figure
from contrail_detect.core.common import Colors, Transforms, extract_date_time_components
from contrail_detect.core.config import TEST_DIR
from contrail_detect.geostationary import geo_utils, validation_utils
from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.meteorology.ecmwf import initialize_tracking_windspeeds

if TYPE_CHECKING:
    from contrail_detect.tracking.tracking import TrackingConfig

colors, transforms = Colors(), Transforms()


def mesoscale_geo_coords_path(satellite: str, region: str, time_str: str) -> Path:
    """Staging path of a mesoscale frame's geostationary coordinates (``geo_x``, ``geo_y``).

    Args:
        satellite: Satellite name.
        region: Region code (M1 or M2).
        time_str: Frame time as ``YYYY-MM-DDTHHMM``.

    Returns:
        Path of the ``.npz`` file.
    """
    tiles_dir = Path(TEST_DIR) / f"{satellite.upper()}-{region}-tiles"
    return tiles_dir / f"{time_str}_geo_coords.npz"


def mobile_tile_slices(tile_idx: int) -> tuple[slice, slice]:
    """Row and column slices of a mesoscale tile within the 500 x 500 pixel image.

    Args:
        tile_idx: Tile index (0–3).

    Returns:
        (rows, cols) slices, 256 pixels each.
    """
    start_x = 244 * (tile_idx // 2)
    start_y = 244 * (tile_idx % 2)
    return slice(start_y, start_y + 256), slice(start_x, start_x + 256)


class ABIDetector(Geostationary):
    """Contrail detection using Advanced Baseline Imager (ABI) data.

    Base class for geostationary satellites equipped with ABI instruments,
    including GOES-R series and Himawari satellites. Provides methods for
    tile generation, image fetching, and coordinate transformations specific
    to ABI data products.

    Subclasses supply frames through ``_get_frame``, ``_frame_projection`` and
    ``_frame_extent``, which the local model staging pipeline is written against.

    Attributes:
        BAND_IDS: Band numbers of the ash RGB channels (8.6, 11.2 and 12.3 µm).
    """

    BAND_IDS: list[int] = [11, 14, 15]

    def __init__(self: Self) -> None:
        """Initialize the ABIDetector class."""
        super().__init__()

    def get_detections(
        self: Self,
        method: str,
        device_ids: list[int] | None = None,
        force_reprocess: bool = False,
        zarr_path: str | Path | None = None,
        detection_output_dir: str | Path | None = None,
        tiles_to_load: list[int] | None = None,
        n_jobs: int = 1,
        unet1024: bool = False,
    ) -> dict[np.datetime64, np.ndarray]:
        """
        Get contrail detection masks, either by loading from file, running the local
        model, or requesting them from the Contrails.org or Google Contrails API.

        Loading from file is always attempted first unless 'force_reprocess' is enabled.

        Args:
            method: How contrail detection masks should be sourced. Options:
                - 'load': Load existing detections from zarr or .npy files, from the
                  source named by ``self.detection_subdirectory``
                - 'local': Run local inference with the U-Net and vision transformer
                  ensemble from the winning Kaggle (Google Research contrail detection
                  competition) solution (Jun Koda)
                - 'contrails_api': Request detections from Contrails.org API
                - 'google_api': Request detections from the Google Contrails REST API
            device_ids: GPU device indices as list (e.g., [0, 1]) or 'CPU' string.
                Only used when method='local'.
            force_reprocess: Whether to ignore data loading and process all timestamps.
            zarr_path: Path to zarr store. If None, uses default location.
            detection_output_dir: Directory for storing/loading detection outputs.
            tiles_to_load: Subset of tile indices to stage and process. If None, all
                valid tiles are used. Only used when method='local'.
            n_jobs: Number of parallel joblib workers for tile staging. Only used when
                method='local'. Default is 1 (sequential). Each worker receives a
                full copy of the 5-frame Full Disk image (~1.8 GB), so increase only
                if RAM permits.
            unet1024: Whether to run the UNet1024 model in addition to ViT4. Only used
                when method='local'. Defaults to False.

        Returns:
            Dictionary mapping timestamps to detection arrays.

        Examples:
            >>> detector.detection_subdirectory = 'contrails_api'  # whose saved detections
            >>> detector.get_detections(method='load')
            >>> detector.get_detections(method='local', device_ids=[0, 1])
            >>> detector.get_detections(method='contrails_api')

        Raises:
            ValueError: If method is not one of the valid options.
        """
        if method not in ["load", "local", "contrails_api", "google_api"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: "
                f"'load', 'local', 'contrails_api', 'google_api'."
            )

        # Set detection_subdirectory based on detection method.
        # For 'load', preserve whatever was set by the caller — the subdirectory
        # tells load_detections where to look and must not be overwritten.
        method_to_subdir = {
            "local": "local",
            "contrails_api": "contrails_api",
            "google_api": "google_api",
        }
        if method in method_to_subdir:
            self.detection_subdirectory = method_to_subdir[method]

        if method == "load":
            if self.detection_subdirectory is None:
                raise ValueError(
                    "method='load' needs to know which source's saved detections to read: "
                    "set detector.detection_subdirectory to 'local' (local model), "
                    "'contrails_api' or 'google_api' first."
                )
            # Models write to <detection_output_dir>/<detection_subdirectory>; the default already has it
            if detection_output_dir is not None:
                detection_output_dir = Path(detection_output_dir) / self.detection_subdirectory

            # Load existing detections from file
            detections = self.load_detections(
                zarr_path=zarr_path,
                detection_output_dir=detection_output_dir,
            )

            return detections

        elif method == "local":
            if device_ids is None:
                raise ValueError(
                    "'device_ids' cannot be None if 'local' method is selected. "
                    "Please specify which GPU/CPU devices to use for inference."
                )

            from contrail_detect.models.local_model import LocalContrailModel

            # Create model instance and run detections
            local_model = LocalContrailModel(
                detector=self,
                device_ids=device_ids,
                detection_output_dir=detection_output_dir,
                unet1024=unet1024,
            )
            detections = local_model.run_detections(
                force_reprocess=force_reprocess,
                zarr_path=zarr_path,
                tiles_to_load=tiles_to_load,
                n_jobs=n_jobs,
            )

            return detections

        elif method == "contrails_api":
            from contrail_detect.models.contrails_api import ContrailsAPIContrailModel

            # Create API model instance and run detections
            api_model = ContrailsAPIContrailModel(
                detector=self,
                detection_output_dir=detection_output_dir,
            )
            detections = api_model.run_detections(
                force_reprocess=force_reprocess,
                zarr_path=zarr_path,
            )

            return detections

        elif method == "google_api":
            from contrail_detect.models.google_api import GoogleAPIContrailModel

            api_model = GoogleAPIContrailModel(
                detector=self,
                detection_output_dir=detection_output_dir,
            )
            detections = api_model.run_detections(
                force_reprocess=force_reprocess,
                zarr_path=zarr_path,
            )

            return detections

    def get_decomposed_contrails(
        self: Self,
        method: str,
        contrail_output_dir: str | Path | None = None,
        force_reprocess: bool = False,
        read_simple: bool = False,
        prediction_threshold: float | None = None,
        min_area: int = 10,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
        """Get contrail detection line segments, either by loading from cache or generating.

        Each method first attempts to load cached results (unless force_reprocess=True),
        then generates new results if needed.

        Args:
            method: How contrail line segments should be sourced. Options:

                - 'decomposer': Generate line segments using the decomposer module
                - 'google_api': Request line segments from the Google Contrails REST API

            contrail_output_dir: Directory to save/read line segment files. If None,
                uses default location.
            force_reprocess: If True, skip loading cached results and regenerate.
                Default is False (load from cache if available).
            read_simple: If True, read simplified version of decomposition output without
                point cloud objects. Only applicable when loading cached results.
            prediction_threshold: Detection-score threshold for contrail pixels. Only applicable
                when method is 'decomposer'. If None, uses the default for the detection
                source: 0.30 for the local model and Contrails.org API, 0.40 for the Google API.
            min_area: Minimum area (in pixels) for detected regions to be retained. Only
                applicable when method is 'decomposer'. Defaults to 10.
            **kwargs: Additional DecompositionConfig parameters (only applicable when
                method is 'decomposer'), including:

                - contrail_stitching: Whether to rejoin the Type I/II branches split from one
                  detected region. Defaults to True.
                - min_type_1_branch_length: Minimum branch length for type 1 branches.
                  Defaults to 5.
                - angle_threshold: Angle threshold in degrees for joining vectors.
                  Defaults to 135.0.
                - elbow_threshold: Threshold for detecting elbows in branch skeletons.
                  Defaults to 0.4.
                - dilate_radius: Radius for mask dilation. If None, uses sqrt of
                  skeletonization loss. Defaults to None.
                - area_to_distance_ratio_threshold: Maximum ratio for branch pairing.
                  Defaults to 5.0.

        Returns:
            Tuple containing:

            - DataFrame of contrail line segments.
            - Dictionary mapping contrail IDs to (N, 2) arrays of [row, col] global
              image index coordinates, or None when read_simple=True or not available.
        """
        if method not in ["decomposer", "google_api"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: 'decomposer', 'google_api'."
            )

        if method == "decomposer":
            prediction_threshold = self._resolve_prediction_threshold(prediction_threshold)

            # Part of the results folder name; see _decomposition_fingerprint
            self.decomposition_settings = {
                "min_area": min_area,
                **{k: v for k, v in kwargs.items() if k != "contrail_stitching"},
            }

            # Try to load cached results first unless force_reprocess
            if not force_reprocess:
                try:
                    line_segments, point_clouds = self.load_decomposed_contrails(
                        contrail_output_dir=contrail_output_dir,
                        source_method="decomposer",
                        read_simple=read_simple,
                        contrail_stitching=kwargs.get("contrail_stitching", True),
                        prediction_threshold=prediction_threshold,
                    )
                    logging.info("Loaded decomposed contrails from cache.")
                    return line_segments, point_clouds

                except FileNotFoundError:
                    logging.info("No cached decomposition found, generating...")

            # Generate new decomposition
            from contrail_detect.decomposition.decompose import Decomposer

            # Create decomposer model instance and generate line segments
            decomposer_model = Decomposer(
                detector=self,
                contrail_output_dir=contrail_output_dir,
                prediction_threshold=prediction_threshold,
                min_area=min_area,
                **kwargs,
            )

            # Generate line segments and point cloud objects
            segment_outputs = decomposer_model.gen_decomposed_regions(
                force_regeneration=force_reprocess,
                read_simple=read_simple,
            )

            return segment_outputs[0], segment_outputs[1]

        elif method == "google_api":
            if not force_reprocess:
                try:
                    line_segments, point_clouds = self.load_decomposed_contrails(
                        contrail_output_dir=contrail_output_dir,
                        source_method="google_api",
                        read_simple=True,
                    )
                    logging.info("Loaded Google API line segments from cache.")
                    return line_segments, point_clouds

                except FileNotFoundError:
                    logging.info("No cached Google API line segments found, fetching...")

            from contrail_detect.models.google_api import GoogleAPIContrailModel

            api_model = GoogleAPIContrailModel(detector=self)
            line_segments = api_model.run_segments()

            line_segments_path = self._get_line_segments_path(
                contrail_output_dir,
                method="google_api",
            )
            line_segments.to_parquet(line_segments_path, index=False)

            logging.info(f"Saved Google API line segments to {line_segments_path}.")

            return line_segments, None

    def get_tracking_output(
        self: Self,
        force_reprocess: bool = False,
        tracking_config: Optional["TrackingConfig"] = None,
        use_wind_initialization: bool | None = None,
        estimated_contrail_height_ft: float | np.ndarray = 3.5e4,
        path_to_key: str | Path | None = None,
    ) -> None:
        """Get contrail tracking outputs, either by loading from file or running tracking.

        Args:
            force_reprocess: Whether to ignore data loading and run tracking anew.
            tracking_config: Configuration object for contrail tracking. If None,
                uses default TrackingConfig settings.
            use_wind_initialization: Whether to start new tracks with ERA5 wind
                velocities, which downloads ERA5. If None, uses
                ``tracking_config.use_wind_initialization``.
            estimated_contrail_height_ft: Estimated contrail altitude in feet, used for
                windspeed initialization. Can be a single value or an array matching the
                number of contrail detections.
            path_to_key: Optional path to directory containing .cdsapirc file for ERA5
                download. If None, defaults to ~/.cdsapirc.

        Returns:
            New column 'track_id' added to contrail summary DataFrame.
        """

        if "track_id" in self.contrail_summary.columns and not force_reprocess:
            logging.info("'track_id' column already exists in contrail summary. Skipping tracking.")
            return

        from contrail_detect.tracking.tracking import TrackingConfig, TrackingManager

        # Set up tracking configuration, using defaults if not provided
        if tracking_config is None:
            tracking_config = TrackingConfig()
            logging.info(
                "No tracking_config provided. Using default TrackingConfig settings:\n"
                f"{tracking_config.to_dict()}"
            )

        if use_wind_initialization is None:
            use_wind_initialization = tracking_config.use_wind_initialization

        # If windspeed initialization is enabled, add windspeed data to contrail summary
        if use_wind_initialization:
            self._add_windspeed_to_summary(
                estimated_alt_ft=estimated_contrail_height_ft,
                path_to_key=path_to_key,
            )

        # Run tracking manager to get track IDs for contrail detections
        manager = TrackingManager(
            contrail_summary=self.contrail_summary,
            config=tracking_config,
            use_wind_initialization=use_wind_initialization,
        )
        contrail_to_track = manager.run_tracking()

        # Map contrail IDs to track IDs and add to contrail summary
        self.contrail_summary["track_id"] = self.contrail_summary["contrail_id"].map(
            contrail_to_track
        )

        logging.info("Finished contrail tracking. Added 'track_id' to contrail summary.")

    def load_data_for_local_model(
        self: Self,
        tiles_to_load: list[int] | None = None,
        n_jobs: int = 1,
        timestamps_to_process: list[np.datetime64] | None = None,
    ) -> None:
        """
        Process satellite data for a specific time and region, saving the
        processed tiles locally. Optimized sequential processing with smart caching.

        Strategy:
        - Use sliding window to reuse downloaded satellite data
        - Cache tile data in memory to avoid redundant disk reads
        - Batch file existence checks to reduce I/O overhead
        - Skip fully processed timestamps

        Args:
            tiles_to_load: Subset of tiles to load into the local model. Defaults to None, indicating that all suitable tiles should be loaded.
            n_jobs: Number of parallel joblib workers for tile processing. Passed through
                to ``_process_single_timestamp``. See that method for memory implications.
            timestamps_to_process: Subset of timestamps to stage. If None, all timestamps
                in the sequence are staged. Use this to avoid re-staging timestamps whose
                detections already exist.
        """
        if len(self.tile_idx) == 0:
            raise ValueError("Must generate tiles before loading data into detector.")

        # Path to tile xx and yy arrays
        tile_loc_dir = Path(TEST_DIR) / f"{self.satellite.upper()}-{self.region}-tiles"
        tile_loc_dir.mkdir(parents=True, exist_ok=True)

        if tiles_to_load is None:
            tiles_to_load = [t for t in np.unique(self.tile_idx) if t != -1]

        timestamps_needed = (
            set(timestamps_to_process)
            if timestamps_to_process is not None
            else set(self.timestamp_sequence)
        )

        # Process timestamps with sliding window cache
        data_cache = {}  # Cache for downloaded satellite data
        tile_memory_cache = (
            OrderedDict()
        )  # Cache for previously loaded tiles (FIFO with OrderedDict)

        for timestamp in tqdm(self.timestamp_sequence, desc="Timestamps"):
            if timestamp not in timestamps_needed:
                continue
            self._process_single_timestamp(
                timestamp=timestamp,
                tiles_to_load=tiles_to_load,
                data_cache=data_cache,
                tile_memory_cache=tile_memory_cache,
                n_jobs=n_jobs,
            )

    def load_single_tile(
        self: Self,
        t: int,
        time_str: str,
        times: list[pd.Timestamp],
        geo_x: list[np.ndarray],
        geo_y: list[np.ndarray],
        crs: list[pyproj.CRS],
        img: list[np.ndarray],
        tile_memory_cache: dict[str, np.ndarray],
    ) -> bool:
        """
        Processes and saves a single tile with memory caching to avoid redundant disk reads.

        Args:
            t: Tile index.
            time_str: Timestamp string.
            times: List of timestamps for the 4 frames to be processed.
            geo_x: Geospatial x-coordinates for each frame.
            geo_y: Geospatial y-coordinates for each frame.
            crs: List of coordinate reference systems for each frame.
            img: List of image data arrays for each frame.
            tile_memory_cache: Cache of previously loaded tile data.

        Returns:
            False if the tile was skipped because it reaches past the edge of the Earth's
            disk or outside the image, True otherwise.
        """
        # Path to tile bands
        tile_dir = Path(TEST_DIR) / f"{self.satellite.upper()}-{self.region}-{time_str}-{t:03d}"
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Path to tile locations
        tile_loc_dir = Path(TEST_DIR) / f"{self.satellite.upper()}-{self.region}-tiles"

        # Batch check if all bands exist (single file system call is faster)
        band_files = [tile_dir / f"band_{band}.npy" for band in self.BAND_IDS]
        if all(f.exists() for f in band_files):
            return True  # Already staged

        # Initialize structure for tile_ash_rgb : (3, 256, 256, 8)
        tile_ash_rgb = np.zeros((3, 256, 256, 8), dtype=np.float32)

        # Batch prepare all cache keys and file paths upfront (more efficient)
        cache_keys = []
        prev_band_files_list = []
        disk_cache_available = []

        for time in times:
            year, month, _, dom, hour, minute = extract_date_time_components(time)
            time_str_prev = f"{year:04d}-{month:02d}-{dom:02d}T{hour:02d}{minute:02d}"
            cache_key = f"{self.satellite.upper()}-{self.region}-{time_str_prev}-{t:03d}"
            cache_keys.append(cache_key)

            # Prepare disk cache file paths
            prev_tile_dir = Path(TEST_DIR) / cache_key
            prev_band_files = [prev_tile_dir / f"band_{band}.npy" for band in self.BAND_IDS]
            prev_band_files_list.append(prev_band_files)

            # Batch check if disk cache exists (single check per time frame)
            disk_cache_available.append(all(f.exists() for f in prev_band_files))

        for t_idx, time in enumerate(times):
            cache_key = cache_keys[t_idx]

            # Try memory cache first
            if cache_key in tile_memory_cache:
                # Reuse from memory cache (much faster than disk read)
                for b_idx in range(3):
                    tile_ash_rgb[b_idx, :, :, t_idx] = tile_memory_cache[cache_key][b_idx]
            elif disk_cache_available[t_idx]:
                # Load from disk cache (already verified to exist)
                prev_band_files = prev_band_files_list[t_idx]
                cached_tile_data = []
                for b_idx, _ in enumerate(self.BAND_IDS):
                    loaded_array = np.load(prev_band_files[b_idx])
                    prev_tile_band = loaded_array[:, :, 4]
                    tile_ash_rgb[b_idx, :, :, t_idx] = prev_tile_band
                    cached_tile_data.append(prev_tile_band.copy())

                # Store in memory cache for future reuse
                tile_memory_cache[cache_key] = cached_tile_data

                # Prune memory cache to prevent excessive memory usage
                # Keep more tiles in cache for better hit rates (each tile is ~3 x 256x256x4 bytes = ~768KB)
                max_cache_size = 50  # ~38MB total
                if len(tile_memory_cache) > max_cache_size:
                    # Remove oldest entry efficiently using OrderedDict (FIFO)
                    tile_memory_cache.popitem(last=False)
            else:
                # Generate the tile from scratch
                if self.region in ["C", "F"]:
                    tile_xx, tile_yy, tile_img = self._generate_tile_fixed_grid(
                        tile_lon=self.tile_lons[t] - 360.0,
                        tile_lat=self.tile_lats[t],
                        geo_x=geo_x[t_idx],
                        geo_y=geo_y[t_idx],
                        img=img[t_idx],
                        crs=crs[t_idx],
                        tile_size=512.0,
                    )
                elif self.region in ["M1", "M2"]:
                    _, _, tile_img = self._generate_tile_mobile_grid(
                        tile_idx=t,
                        img=img[t_idx],
                        geo_x=geo_x[t_idx],
                        geo_y=geo_y[t_idx],
                    )
                    # _join_tiles expects geostationary x/y in metres
                    rows, cols = mobile_tile_slices(t)
                    tile_xx, tile_yy = np.meshgrid(geo_x[t_idx][cols], geo_y[t_idx][rows])
                else:
                    raise ValueError(f"Unsupported region: {self.region}")

                # Skip if tile is None (outside the usable domain). Remove its empty folder
                # so the local model step doesn't find a tile with no bands
                if tile_img is None:
                    logging.debug(f"Tile {t} at time {time} is outside the usable domain")
                    if not any(tile_dir.iterdir()):
                        tile_dir.rmdir()
                    return False

                tile_ash_rgb[..., t_idx] = tile_img.transpose(2, 0, 1)

                # Save tile location information.
                # For mesoscale (M1/M2): write per-timestamp into tile_dir, because the
                # domain shifts every epoch and a single shared lonlat file would be stale.
                # For fixed regions (C/F): write once into tile_loc_dir (shared across timestamps).
                if self.region in ["M1", "M2"]:
                    np.save(tile_dir / f"{t:03d}_lonlat.npy", (tile_xx, tile_yy))
                elif not (tile_loc_dir / f"{t:03d}_lonlat.npy").exists():
                    np.save(tile_loc_dir / f"{t:03d}_lonlat.npy", (tile_xx, tile_yy))

        # Save tile (256, 256, 8) - write all bands at once for better I/O performance
        for b_idx, band in enumerate(self.BAND_IDS):
            np.save(
                tile_dir / f"band_{band}.npy",
                tile_ash_rgb[b_idx, ...],
            )

        # Store current tile's 5th frame in cache for potential reuse by next timestamp
        current_cache_key = f"{self.satellite.upper()}-{self.region}-{time_str}-{t:03d}"
        tile_memory_cache[current_cache_key] = [
            tile_ash_rgb[b_idx, :, :, 4].copy() for b_idx in range(3)
        ]
        return True

    @plt.rc_context({"font.size": 10})
    def plot_abi_test_tiles(
        self: Self,
        tiles: list[int],
        global_view: bool = False,
        numbered: bool = False,
        fig_output_dir: str | Path | None = None,
    ) -> plt.Figure:
        """Display test tiles on a map for ABI detector visualization.

        Plots specified tile locations on a geographic map, useful for verifying
        tile grid generation and spatial coverage.

        Args:
            tiles: List of tile indices to display.
            global_view: Whether to show global extent or zoom to valid domain.
                Defaults to False.
            numbered: Whether to number tiles or mark with crosses. Defaults to False.
            fig_output_dir: Directory in which to save the figure, as
                ``<satellite>-<region>-tiles.png``. Not saved if None.

        Returns:
            Matplotlib Figure.
        """
        # Plot projection
        plot_projection = transforms.platecarree(central_lon=self.central_lon)

        # Figure setup
        fig = plt.figure(figsize=(10.0, 5.0), dpi=400)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        with sns.axes_style(colors.plot_params_light):
            # Map subplot
            ax1 = fig.add_subplot(111, projection=plot_projection)
            ax1.set_title(f"Tiles for {self.satellite.upper()}/{self.region}")

            if global_view:
                ax1.set_global()
            else:
                ax1.set_extent(
                    (
                        self.min_longitude - self.central_lon,
                        self.max_longitude - self.central_lon,
                        self.min_latitude,
                        self.max_latitude,
                    )
                )
                ax1.set_xlim(
                    (
                        self.min_longitude - self.central_lon,
                        self.max_longitude - self.central_lon,
                    )
                )
                ax1.set_ylim(
                    (
                        self.min_latitude,
                        self.max_latitude,
                    )
                )

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

            gl = ax1.gridlines(
                draw_labels=True,
                linestyle=":",
                color=colors.facecolor,
                linewidth=0.5,
                zorder=1,
            )
            gl.xlabel_style = {"color": colors.facecolor}
            gl.ylabel_style = {"color": colors.facecolor}
            gl.right_labels = False
            gl.left_labels = True
            gl.top_labels = False
            gl.bottom_labels = True
            plt.draw()  # Force a redraw of the figure

        # Extract projection and coordinates
        geo_x, geo_y = self.image_geo_coords
        _, geo_proj = self.image_transforms

        pyproj_dict = geo_proj.to_dict()
        pyproj_dict["lon_0"] = self.central_lon

        cmap = plt.get_cmap("tab20", 20)
        for t_idx in [item for item in list(np.unique(self.tile_idx)) if item != -1]:
            if numbered:
                ax1.text(
                    self.tile_lons[t_idx] - self.central_lon,
                    self.tile_lats[t_idx],
                    f"${t_idx}$",
                    color=cmap(t_idx % 20),
                    ha="center",
                    va="center",
                    fontsize=4,
                    zorder=4,
                    transform=plot_projection,
                )
            else:
                ax1.scatter(
                    self.tile_lons[t_idx] - self.central_lon - 360.0,
                    self.tile_lats[t_idx],
                    marker="+",
                    s=3.0,
                    color=cmap(t_idx % 20),
                    zorder=4,
                    transform=plot_projection,
                )

        # Generate tiles for each unique tile index
        frame = self._get_frame(pd.Timestamp(self.timestamp_sequence[0]))
        img_data = np.transpose(frame.values, (1, 2, 0))

        for t in tiles:
            # Generate tiling
            if self.region in ["C", "F"]:
                tile_xx, tile_yy, tile_img = self._generate_tile_fixed_grid(
                    tile_lon=self.tile_lons[t] - 360.0,
                    tile_lat=self.tile_lats[t],
                    geo_x=geo_x,
                    geo_y=geo_y,
                    img=img_data,
                    crs=geo_proj,
                    tile_size=512.0,
                )
                if tile_img is None:
                    continue

                # Turn into Ash RGB
                tile_ash_rgb = self._gen_ash_color_base(
                    band_11=tile_img[:, :, 0],
                    band_14=tile_img[:, :, 1],
                    band_15=tile_img[:, :, 2],
                )

                # Plot tile
                ax1.pcolormesh(
                    tile_xx,
                    tile_yy,
                    tile_ash_rgb,
                    transform=ccrs.Projection(pyproj_dict),
                )

            elif self.region in ["M1", "M2"]:
                tile_xx, tile_yy, tile_img = self._generate_tile_mobile_grid(
                    tile_idx=t,
                    img=img_data,
                )
                if tile_img is None:
                    continue

                # Turn into Ash RGB
                tile_ash_rgb = self._gen_ash_color_base(
                    band_11=tile_img[:, :, 0],
                    band_14=tile_img[:, :, 1],
                    band_15=tile_img[:, :, 2],
                )

                # Plot tile
                ax1.pcolormesh(
                    tile_xx - self.central_lon,
                    tile_yy,
                    tile_ash_rgb,
                )

        save_figure(fig, fig_output_dir, f"{self.satellite}-{self.region}-tiles")
        return fig

    def _add_windspeed_to_summary(
        self: Self,
        estimated_alt_ft: float | np.ndarray,
        path_to_key: str | Path | None = None,
    ) -> None:
        """Add ERA5 windspeed data to contrail summary for use in tracking initialization.

        This method retrieves ERA5 windspeed data for the relevant time period and spatial
        extent of the contrail detections, and adds interpolated u and v wind components at the
        estimated contrail altitude to the contrail summary DataFrame.

        Args:
            estimated_alt_ft: Estimated altitude of contrails in feet, used for windspeed
                interpolation. Can be a single value or an array matching the number of
                contrail detections.
            path_to_key: Optional path to directory containing .cdsapirc file for ERA5
                download. If None, defaults to ~/.cdsapirc.
        """
        # Check for valid altitude input
        if isinstance(estimated_alt_ft, (int, float)):
            if estimated_alt_ft <= 2e4 or estimated_alt_ft >= 5e4:
                raise ValueError(
                    f"Estimated altitude must be a positive value in feet between 20,000 "
                    f"and 50,000 feet. Received: {estimated_alt_ft}"
                )

            estimated_alt_ft = np.full(len(self.contrail_summary), estimated_alt_ft)

        elif isinstance(estimated_alt_ft, np.ndarray):
            if len(estimated_alt_ft) != len(self.contrail_summary):
                raise ValueError(
                    f"Length of estimated_alt_ft array must match number of contrail detections "
                    f"in contrail summary. Expected length: {len(self.contrail_summary)}, "
                    f"Received length: {len(estimated_alt_ft)}"
                )

        # Add interpolated wind speeds to contrail summary
        self.contrail_summary = initialize_tracking_windspeeds(
            self.contrail_summary,
            estimated_alt_ft=estimated_alt_ft,
            path_to_key=path_to_key,
        )

    def _decompose_abi_fixed_grid(
        self: Self,
        extent: tuple[float, float, float, float],
        overlap: float,
        tile_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute decomposition of full ABI fixed grid into tiles.

        The goal of the decomposition is to split a region defined by a longitude/latitude
        bounding box into a set of 512 x 512 km tiles that overlap on each edge by a specified
        amount. (The size of the tiles is set by requirements of the detection model, which
        expects 256 x 256 pixel images with an resolution of 2 km.) The approach taken by this
        code is to first split the region into the required number of rows, and then split each
        row into the required number of columns. The number of columns per row is not constant
        because the distance between meridians varies with latitude.

        Args:
            extent: Bounding box (minimum longitude,
                maximum longitude, minimum latitude, maximum latitude) of the region to
                decompose into tiles.
            overlap: Minimum overlap, to include at tile edges, in units of km.
            tile_size: Dimensions of each tile (square), in units of km.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                Longitude and latitude at the center of each tile, plus an array with the same
                shape as the ABI fixed grid indicating the index (in the returned latitude and
                longitude arrays) of the tile that each point on the ABI fixed grid is assigned to.
                Points not assigned to any tile (i.e., points outside) are assigned a value of -1.

        Raises:
            ValueError: If the bounding box defined by `extent` extends outside the range of
                latitudes and longitudes on which detection models were trained, or if `overlap`
                is too large to fix inside tiles.
        """
        # Validation of inputs
        validation_utils.validate_extent(
            extent,
            self.min_longitude,
            self.max_longitude,
            self.min_latitude,
            self.max_latitude,
        )
        validation_utils.validate_overlap(overlap, tile_size)

        # Extract ABI projection attributes
        x, y = self.image_geo_coords
        _, crs = self.image_transforms

        # Compute lat/lon coordinates of ABI fixed grid
        ll_crs = pyproj.CRS("WGS84")
        xx, yy = np.meshgrid(x, y, indexing="xy")
        abi_to_ll = pyproj.Transformer.from_crs(
            crs_from=crs,
            crs_to=ll_crs,
            always_xy=True,
        )
        lon, lat = abi_to_ll.transform(xx, yy)

        # Normalize lons to a continuous range centered on the satellite's sub-satellite
        # point before applying the +360 IDL offset. GOES-West (central_lon ≈ -137°)
        # crosses the anti-meridian: without this step, pyproj returns lons spanning
        # the full ±180° range, and after +360 both endpoints become 180°/540°, which
        # are the same point — causing geod.inv to return 0 m for every row width.
        c = self.central_lon
        lon = (((lon - c) + 180) % 360) - 180 + c
        min_lon, max_lon, min_lat, max_lat = extent
        min_lon = (((min_lon - c) + 180) % 360) - 180 + c
        max_lon = (((max_lon - c) + 180) % 360) - 180 + c

        # Adjust the longitudes to be in a continuous range across the
        # International Date Line (IDL)
        lon += 360.0
        min_lon += 360.0
        max_lon += 360.0

        # Assign points on ABI fixed grid to tiles
        mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
        interior_tile_size = tile_size - (2.0 * overlap)  # [km]

        # Compute row latitudes (does not depend on choice of meridian)
        geod = pyproj.Geod(ellps="WGS84")
        _, _, domain_height = geod.inv(0.0, min_lat, 0.0, max_lat)
        num_rows = int(np.ceil(domain_height / interior_tile_size / 1e3))
        lat_edges = np.linspace(min_lat, max_lat, num_rows + 1)

        # Match ABI pixels to rows (starting top left of grid)
        row_idx = np.where(
            mask,
            np.clip(np.searchsorted(lat_edges, lat) - 1, 0, lat_edges.size - 2),
            -1,
        )

        # Compute coordinates of tiles within each row
        tile_lon, tile_lat = [], []
        tile_idx = np.full_like(row_idx, -1)
        row_offset = 0
        for irow, lat_bounds in enumerate(itertools.pairwise(lat_edges)):
            # Find the latitude with the longest parallel within the current row
            if np.prod(lat_bounds) > 0:
                longest_parallel = np.min(np.abs(lat_bounds))
            else:
                longest_parallel = 0.0

            # Calculate row width as the arc length ALONG the parallel.
            # Tiles are placed evenly in longitude (see lon_edges below), so the
            # spacing that determines overlap is the along-parallel arc, not the
            # geodesic between the endpoints. Over a wide longitude span at high
            # latitude the geodesic bows poleward and is significantly shorter
            # than the parallel arc (e.g. 18% shorter for a full-disk row at
            # 52N), which would undercount num_cols and space tiles further apart
            # than tile_size — opening gaps between tiles. The arc along a
            # parallel is dlon * N(phi) * cos(phi), where N is the radius of
            # curvature in the prime vertical.
            phi = np.radians(longest_parallel)
            prime_vertical_radius = geod.a / np.sqrt(1.0 - geod.es * np.sin(phi) ** 2)
            row_width = abs(np.radians(max_lon - min_lon)) * prime_vertical_radius * np.cos(phi)

            # Calculate longitudes for tile edges
            num_cols = int(np.ceil(row_width / interior_tile_size / 1000.0))
            lon_edges = np.linspace(min_lon, max_lon, num_cols + 1)

            # Record tile centers
            tile_lon.append(0.5 * (lon_edges[1:] + lon_edges[:-1]))
            tile_lat.append(np.full_like(tile_lon[-1], 0.5 * sum(lat_bounds)))

            # Match ABI pixels to tiles
            col_idx = np.clip(np.searchsorted(lon_edges, lon) - 1, 0, lon_edges.size - 2)

            # Select only the pixels that belong to the current row.
            row_mask = row_idx == irow
            # Assign tile index
            tile_idx[row_mask] = col_idx[row_mask] + row_offset
            # Update offset
            row_offset += lon_edges.size - 1

        self.tile_lons = np.concatenate(tile_lon)
        self.tile_lats = np.concatenate(tile_lat)
        self.tile_idx = tile_idx

    def _find_tile_from_location(
        self: Self,
        target_lon: float,
        target_lat: float,
    ) -> int:
        """Find the tile index at a given location.

        Args:
            target_lon: Longitude to find.
            target_lat: Latitude to find.

        Returns:
            int: Tile index at given location.
        """
        # Compare longitudes in the domain's convention, which is continuous around the
        # sub-satellite point (e.g. 59.5° to 221.9° for Himawari), so tiles across the
        # anti-meridian are not mistaken for points outside the image
        lon = target_lon
        if getattr(self, "central_lon", None) is not None:
            lon = (((target_lon - self.central_lon) + 180) % 360) - 180 + self.central_lon
        if not (self.min_longitude <= lon <= self.max_longitude) or not (
            self.min_latitude <= target_lat <= self.max_latitude
        ):
            raise ValueError(
                f"Requested location ({target_lon}, {target_lat}) is outside of image domain."
            )

        # Squared distance in degrees to each pixel, with longitude differences wrapped
        # into [-180, 180) so pixels either side of the anti-meridian compare correctly
        lon_diff = ((self.lons - target_lon + 180) % 360 - 180) ** 2
        lat_diff = (self.lats - target_lat) ** 2
        distances_squared = lon_diff + lat_diff

        # Find the flattened index of the minimum distance
        flat_loc = np.nanargmin(distances_squared)

        # Convert the flattened index to 2D coordinates
        row, col = np.unravel_index(flat_loc, self.lons.shape)

        # Use these 2D coordinates to get the corresponding tile index
        return self.tile_idx[row, col]

    @abstractmethod
    def _frame_extent(self: Self, da: xr.DataArray) -> tuple[float, float, float, float]:
        """Extent of a frame in geostationary coordinates, used to check frames align.

        Args:
            da: Frame returned by ``_get_frame``.

        Returns:
            Extent as (x_min, x_max, y_min, y_max) in metres.
        """

    @abstractmethod
    def _frame_projection(
        self: Self, da: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray, pyproj.CRS]:
        """Geostationary coordinates and CRS of a frame's pixels.

        Args:
            da: Frame returned by ``_get_frame``.

        Returns:
            Pixel-centre x and y coordinates (1-D, metres) and the frame's pyproj CRS.
        """

    def _gen_ash_color_base(
        self: Self,
        band_11: np.ndarray,
        band_14: np.ndarray,
        band_15: np.ndarray,
    ) -> np.ndarray:
        """Generate ash color scheme RGB image from individual wavelength bands.

        Args:
            band_11: Wavelength band 11.
            band_14: Wavelength band 14.
            band_15: Wavelength band 15.

        Returns:
            rgb_img: Stacked RGB image, with normalized channels.
        """
        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)

        # Normalise RGB based on Google's provided values for the ash color scheme
        r = geo_utils.normalize_range(band_15 - band_14, _TDIFF_BOUNDS)
        g = geo_utils.normalize_range(band_14 - band_11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = geo_utils.normalize_range(band_14, _T11_BOUNDS)

        return np.clip(np.stack([r, g, b], axis=2), 0, 1)

    def _generate_tile_fixed_grid(
        self: Self,
        tile_lon: float,
        tile_lat: float,
        geo_x: np.ndarray,
        geo_y: np.ndarray,
        img: np.ndarray,
        crs: pyproj.CRS,
        tile_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate imagery for a single tile.

        Tiles are required to by 256 x 256 images with a resolution of about 2 km. Coordinates
        of tile pixels are defined in a transverse Mercator coordinate system with a central
        meridian coincident with the center of the tile. Values are assigned to tile pixels
        by computing coordinates of tile pixels in the ABI coordinate reference system before
        interpolating ABI pixel values to the locations of tile pixels.

        Returns `None` values if the tile is invalid (i.e., index is -1), or reaches past
        the edge of the Earth's disk or the image.

        Args:
            tile_lon: Longitude at tile center.
            tile_lat: Latitude at tile center.
            geo_x: x coordinates of ABI grid, in units of km.
            geo_y: y coordinates of ABI grid, in units of km.
            img: ABI image data, with shape (ny, nx, 3).
            crs: pyproj CRS object defining the coordinate reference system.
            tile_size: Dimensions of each tile (square), in units of km

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                x and y coordinates of tile pixels in the ABI fixed grid coordinate reference
                system, plus an array of shape (256, 256, 3) containing band values for each
                pixel in the tile.
        """
        # Check tile is valid
        if self._find_tile_from_location(tile_lon, tile_lat) == -1:
            logging.info(f"Skipping invalid tile at ({tile_lon}, {tile_lat})")
            return None, None, None

        # Set up interpolator for ABI data
        interp = RegularGridInterpolator(points=(geo_y, geo_x), values=img)

        # Get tile coordinates and CRS
        tile_edges = 1e3 * np.linspace(-tile_size / 2, tile_size / 2, 257)
        tile_x = 0.5 * (tile_edges[1:] + tile_edges[:-1])
        tile_y = tile_x.copy()
        tile_xx, tile_yy = np.meshgrid(tile_x, tile_y, indexing="xy")
        tile_crs = pyproj.CRS(
            proj="tmerc",
            lon_0=tile_lon,
            lat_0=tile_lat,
            ellps="WGS84",
        )

        # Transform tile coordinates to ABI projection
        tile_to_abi = pyproj.Transformer.from_crs(
            crs_from=tile_crs,
            crs_to=crs,
            always_xy=True,
        )
        xx, yy = tile_to_abi.transform(tile_xx, tile_yy)

        # A tile reaching past the edge of the Earth's disk (non-finite positions) or the
        # image can't be sampled whole: skip it rather than fail the run
        half_dx = np.median(np.abs(np.diff(geo_x))) / 2
        half_dy = np.median(np.abs(np.diff(geo_y))) / 2
        inside = (
            np.isfinite(xx)
            & np.isfinite(yy)
            & (xx >= np.min(geo_x) - half_dx)
            & (xx <= np.max(geo_x) + half_dx)
            & (yy >= np.min(geo_y) - half_dy)
            & (yy <= np.max(geo_y) + half_dy)
        )
        if not inside.all():
            return None, None, None

        # Interpolate ABI data onto tile. Positions within half a pixel of the outermost
        # pixel centres are clamped to them rather than rejected as out of bounds
        xx = np.clip(xx, np.min(geo_x), np.max(geo_x))
        yy = np.clip(yy, np.min(geo_y), np.max(geo_y))
        tile_img = interp((yy, xx))

        return xx, yy, tile_img

    def _generate_tile_mobile_grid(
        self: Self,
        tile_idx: int,
        img: np.ndarray,
        geo_x: np.ndarray | None = None,
        geo_y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate imagery for a single tile on a mobile grid.

        The mobile grid is a 500 x 500 km square with a resolution of about 2 km, split into
        four overlapping 256 x 256 km tiles.

        Args:
            tile_idx: Tile index (0–3), identifying which quadrant of the mesoscale image
                to extract. Passed directly to avoid relying on stale instance coordinates.
            img: ABI raw image data of bands 11, 14, and 15, with shape (ny, nx, 3).
            geo_x: Per-timestamp geostationary x-coordinates (1D, metres). When provided,
                fresh lon/lat values are computed for the tile rather than using the
                instance-level ``self.lons``/``self.lats``, which are fixed at construction
                time and become wrong after a mesoscale domain shift.
            geo_y: Per-timestamp geostationary y-coordinates (1D, metres). Must be supplied
                together with ``geo_x``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                Longitude and latitude arrays of shape (256, 256) for the tile pixels,
                plus an array of shape (256, 256, 3) containing band values.

        """
        # Slice requested tile (256x256) from the Mesoscale image (500x500)
        rows, cols = mobile_tile_slices(tile_idx)

        # Resolve lon/lat arrays: use per-timestamp coords when available (mesoscale
        # domain shifts every epoch), otherwise fall back to the instance-level grid
        # (C/F sectors and plotting paths where the domain is fixed).
        if geo_x is not None and geo_y is not None:
            xx_full, yy_full = np.meshgrid(geo_x, geo_y, indexing="xy")
            lons_frame, lats_frame = self.sat_to_latlon_transform.transform(xx_full, yy_full)
            lons_frame[np.isinf(lons_frame)] = np.nan
            lats_frame[np.isinf(lats_frame)] = np.nan
        else:
            lons_frame, lats_frame = self.lons, self.lats

        xx = lons_frame[rows, cols]
        yy = lats_frame[rows, cols]
        tile_img = img[rows, cols]

        return xx, yy, tile_img

    @abstractmethod
    def _get_frame(self: Self, time: pd.Timestamp) -> xr.DataArray:
        """Fetch the frame at one time, holding the bands in ``BAND_IDS``.

        Args:
            time: Nominal frame time.

        Returns:
            Brightness temperatures with dims (band_id, y, x).
        """

    def _process_single_timestamp(
        self: Self,
        timestamp: str,
        tiles_to_load: list[int],
        data_cache: dict[pd.Timestamp, Any],
        tile_memory_cache: dict[str, np.ndarray],
        n_jobs: int = 1,
    ) -> None:
        """Process a single timestamp with caching to avoid redundant downloads and disk reads.

        Uses sliding window caching to reuse satellite data across consecutive timestamps.
        Skips timestamps that don't have all 4 required previous frames available.

        Args:
            timestamp: Timestamp to process.
            tiles_to_load: List of tile indices to load.
            data_cache: Cache of downloaded satellite data (persistent across calls).
            tile_memory_cache: Cache of previously loaded tile data (persistent across calls).
            n_jobs: Number of parallel joblib workers for tile processing. Default is 1
                (sequential). Higher values increase throughput but each worker receives
                a full copy of the 5-frame image data (~1.8 GB for Full Disk), so
                set this with available RAM in mind.
        """
        # Convert input time string to datetime
        image_time = pd.to_datetime(timestamp)

        # Calculate required timestamps (4 previous + 1 current)
        times_before = [
            image_time - pd.Timedelta(minutes=self.image_interval * x) for x in [4, 3, 2, 1]
        ]
        times = times_before + [image_time]

        # Download only missing data (reuse cached data from previous timestamps)
        missing_timestamps = []

        for time in times:
            if time not in data_cache:
                try:
                    da_frame = self._get_frame(time)
                    data_cache[time] = da_frame

                    # Compute and cache projection data immediately to avoid recomputation.
                    # The extent is cached with it so the position check below covers every
                    # frame in the window, not only the ones downloaded on this call.
                    geo_x_frame, geo_y_frame, crs_frame = self._frame_projection(da_frame)
                    self.projection_cache[time] = {
                        "geo_x": geo_x_frame,
                        "geo_y": geo_y_frame,
                        "crs": crs_frame,
                        "img": np.transpose(da_frame.values, (1, 2, 0)),
                        "extent": self._frame_extent(da_frame),
                    }

                except (RuntimeError, FileNotFoundError, KeyError, ValueError) as e:
                    # Data not available for this timestamp
                    if "No data found" in str(e) or "FileNotFoundError" in str(type(e).__name__):
                        logging.warning(
                            f"Missing data at {time} for {self.satellite}-{self.region}. "
                            f"Skipping timestamp {image_time}."
                        )
                        missing_timestamps.append(time)
                    else:
                        # Unexpected error - propagate it
                        raise

        # Skip this timestamp if any required data is missing
        if missing_timestamps:
            logging.info(
                f"Skipping timestamp {image_time} due to {len(missing_timestamps)} missing frame(s): "
                f"{missing_timestamps}"
            )
            return

        # The local model stacks all five frames, so they must cover the same area. Mesoscale boxes
        # can be moved between frames; a stack that spans a move is skipped.
        extent_current = self.projection_cache[image_time]["extent"]
        if not all(
            geo_utils.same_location(self.projection_cache[t]["extent"], extent_current)
            for t in times_before
        ):
            logging.warning(
                f"Skipping {image_time} for {self.satellite}-{self.region}: the {len(times)} "
                "frames the local model needs do not all cover the same area (the mesoscale box moved "
                "within this window)."
            )
            self._prune_cache(data_cache, image_time)
            return

        # Extract time string for data writing
        year, month, _, dom, hour, minute = extract_date_time_components(timestamp)
        time_str = f"{year:04d}-{month:02d}-{dom:02d}T{hour:02d}{minute:02d}"

        # For mesoscale regions, persist the current-frame geo coordinates so that
        # _join_tiles can reconstruct the correct per-epoch canvas after projection_cache
        # has been pruned.  Written once per timestamp to a staging dir that mirrors the
        # tile band data directory structure.
        if self.region in ["M1", "M2"]:
            current_proj = self.projection_cache[image_time]
            geo_coords_path = mesoscale_geo_coords_path(self.satellite, self.region, time_str)
            geo_coords_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(geo_coords_path, geo_x=current_proj["geo_x"], geo_y=current_proj["geo_y"])

        # Retrieve cached projection data (already computed during download)
        geo_x_times, geo_y_times, crs_times, geo_img_times = [], [], [], []
        for time in times:
            cached_projection = self.projection_cache[time]
            geo_x_times.append(cached_projection["geo_x"])
            geo_y_times.append(cached_projection["geo_y"])
            crs_times.append(cached_projection["crs"])
            geo_img_times.append(cached_projection["img"])

        # Process tiles in parallel for speed
        # Note: tile_memory_cache is shared but each tile reads/writes different keys
        # so race conditions are minimal (worst case -> redundant computation)
        try:
            staged = Parallel(n_jobs=n_jobs)(
                delayed(self.load_single_tile)(
                    t=t,
                    time_str=time_str,
                    times=times,
                    geo_x=geo_x_times,
                    geo_y=geo_y_times,
                    crs=crs_times,
                    img=geo_img_times,
                    tile_memory_cache=tile_memory_cache,
                )
                for t in tiles_to_load
            )
        except Exception as e:
            logging.error(f"Error processing timestamp {timestamp}: {str(e)}")
            raise
        n_skipped = sum(result is False for result in staged)
        if n_skipped:
            logging.info(
                f"{timestamp}: skipped {n_skipped} of {len(staged)} tiles that reach past the "
                "edge of the Earth's disk or outside the image."
            )

        # Prune caches after processing to keep only data needed for next timestamp
        self._prune_cache(data_cache, image_time)

    def _prune_cache(
        self: Self,
        data_cache: dict[pd.Timestamp, Any],
        image_time: pd.Timestamp,
    ) -> None:
        """Prune data and projection caches to keep only data needed for next timestamp.

        Args:
            data_cache: Cache of downloaded satellite data.
            image_time: Current timestamp being processed.
        """
        next_image_time = image_time + pd.Timedelta(minutes=self.image_interval)
        times_to_keep = set(
            [
                next_image_time - pd.Timedelta(minutes=self.image_interval * x)
                for x in [4, 3, 2, 1, 0]
            ]
        )
        keys_to_remove = [k for k in data_cache.keys() if k not in times_to_keep]
        for k in keys_to_remove:
            del data_cache[k]
            if k in self.projection_cache:
                del self.projection_cache[k]

    def _tile_extent_as_indices(self: Self) -> tuple[int, int, int, int]:
        # All valid tile indices are non-negative
        non_negative_mask = self.tile_idx != -1

        # Find rows with non -1 values
        rows_with_data = np.where(non_negative_mask.any(axis=1))[0]
        first_row = int(rows_with_data[0]) if len(rows_with_data) > 0 else None
        last_row = int(rows_with_data[-1]) if len(rows_with_data) > 0 else None

        # Find columns with non -1 values
        cols_with_data = np.where(non_negative_mask.any(axis=0))[0]
        first_col = int(cols_with_data[0]) if len(cols_with_data) > 0 else None
        last_col = int(cols_with_data[-1]) if len(cols_with_data) > 0 else None

        return first_col, last_col, first_row, last_row
