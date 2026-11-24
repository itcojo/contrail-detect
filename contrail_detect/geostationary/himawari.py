"""
Contrail detection using AHI data from Himawari family satellites.

Pixel data come from the pycontrails ``Himawari`` datalib (AWS open data, no credentials).
The fixed grid and projection are defined here instead of taken from pycontrails, which as
of 0.63 places pixel centres one pixel north-west of their true position and labels the
projection with ``+sweep=x``, the GOES convention, where AHI uses ``y``. Both errors are
corrected below; the grid here agrees with satpy's ``ahi_hsd`` reader.

The Google Contrails API serves Himawari masks on the same 5500 x 5500 pixel array, but
labels its coordinates as a linear spacing of ±5501 km rather than ±5499 km. The masks are
used index-for-index on this grid, so the difference (zero at nadir, about one pixel at the
limb) does not carry into detections.

"""

import bz2
import datetime
import logging
import warnings
from functools import cached_property, partial
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from pycontrails.datalib.geo_utils import to_ash
from pycontrails.datalib.himawari import HIMAWARI_8_BUCKET, HIMAWARI_9_BUCKET, parse_himawari_header
from pycontrails.datalib.himawari import Himawari as pycon_Himawari

from contrail_detect.core.common import Colors, Transforms
from contrail_detect.core.config import RESULT_DIR
from contrail_detect.geostationary import validation_utils
from contrail_detect.geostationary.abi_base import ABIDetector

colors = Colors()
transforms = Transforms()

warnings.simplefilter("ignore", category=RuntimeWarning)

# AHI 2 km Full Disk navigation, as stated in every HSD header (projection_information).
# Column and line numbers count from 1 (CGMS LRIT/HRIT Global Specification, section 4.4.4).
_SUB_LON: float = 140.7
_SATELLITE_HEIGHT: float = 35785863.0  # above the equator: 42164 km - equatorial radius
_SEMI_MAJOR_AXIS: float = 6378137.0
_SEMI_MINOR_AXIS: float = 6356752.3
_CFAC: int = 20466275
_LFAC: int = 20466275
_COFF: float = 2750.5
_LOFF: float = 2750.5
_GRID_SIZE: int = 5500

# Bytes of decompressed segment file that cover its header (1523 bytes for 2 km bands)
_HEADER_READ_BYTES: int = 8192

_MJD_EPOCH = datetime.datetime(1858, 11, 17)


class HimawariDetector(ABIDetector):
    """Contrail detection using AHI data from Himawari-8 and Himawari-9.

    Supports the Full Disk scan. The grid is fixed by the AHI navigation constants, so a
    detector is fully defined without downloading any imagery; frames are fetched only
    for local inference, plotting and scan timing.

    Attributes:
        HIMAWARI_BANDS: AHI band names used for detection.
        VALID_REGIONS: Supported scan regions.
        VALID_SATELLITES: Supported satellites.
        FALLBACK_SEGMENT_TIMES_S: Start and end of each of the ten Full Disk segments, in
            seconds after nominal time, used when no frame headers can be read.
    """

    HIMAWARI_BANDS: list[str] = ["B11", "B14", "B15"]
    VALID_REGIONS: list[str] = ["F"]
    VALID_SATELLITES: list[str] = ["HIMAWARI-8", "HIMAWARI-9"]

    # Measured from Himawari-9 FLDK HSD headers (B14, 2025-06-01 03:00 UTC). Segments are
    # 550 lines each, scanned north to south.
    FALLBACK_SEGMENT_TIMES_S: list[tuple[float, float]] = [
        (21.6, 54.9),
        (54.9, 101.6),
        (101.6, 158.9),
        (158.9, 245.2),
        (245.2, 305.9),
        (305.9, 364.4),
        (364.4, 452.9),
        (452.9, 508.1),
        (508.1, 547.6),
        (547.6, 581.0),
    ]

    def __init__(
        self: Self,
        himawari_satellite: str,
        himawari_region: str,
        start_time: np.datetime64,
        end_time: np.datetime64,
        extent: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Initialise the HimawariDetector for a satellite, region and time range.

        Args:
            himawari_satellite: Satellite name (``'HIMAWARI-8'`` or ``'HIMAWARI-9'``).
            himawari_region: Scan region; only ``'F'`` (Full Disk) is supported.
            start_time: Start of observation period.
            end_time: End of observation period.
            extent: Geographic extent to cover with tiles, as
                ``(lon_min, lon_max, lat_min, lat_max)``. Defaults to the full disk.

        Raises:
            ValueError: If an unsupported region or satellite is specified.
        """
        super().__init__()

        self.region: str = himawari_region.upper()
        self.satellite: str = himawari_satellite.upper()
        self.sweep_axis = "y"  # Sweep axis for AHI (CGMS convention)

        validation_utils.validate_satellite_inputs(
            self.region,
            self.satellite,
            self.VALID_REGIONS,
            self.VALID_SATELLITES,
        )

        # Path to detection results
        self.DETECT_PATH: Path = Path(RESULT_DIR) / self.satellite / self.region

        # Data source for frames; the grid and CRS are defined in this module
        self.pycon_himawari = pycon_Himawari(
            region="FLDK",
            bands=self.HIMAWARI_BANDS,
            bucket=self.himawari_s3_bucket,
        )

        self.create_timestamp_sequence(start_time=start_time, end_time=end_time)

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

        # Initialize projection cache for coordinate/image data
        self.projection_cache: dict[pd.Timestamp, dict] = {}

        self._gen_tiles()

        # Image offset from global grid (col, row)
        if extent is not None:
            tile_extent_idx = self._tile_extent_as_indices()
            self.image_offset = (tile_extent_idx[0], tile_extent_idx[2])

    @property
    def himawari_s3_bucket(self: Self) -> str:
        """AWS open-data bucket holding this satellite's AHI L1b data."""
        return {"HIMAWARI-8": HIMAWARI_8_BUCKET, "HIMAWARI-9": HIMAWARI_9_BUCKET}[self.satellite]

    @property
    def image_full_extent(self: Self) -> list[float]:
        """Full extent of the image in geostationary coordinates [x_min, x_max, y_min, y_max]."""
        geo_x, geo_y = self.image_geo_coords
        return [geo_x.min(), geo_x.max(), geo_y.min(), geo_y.max()]

    @cached_property
    def image_geo_coords(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """Geostationary x and y coordinates of pixel centres (metres, float64).

        x runs west to east and y north to south, matching the HSD line order.
        """
        n = np.arange(1, _GRID_SIZE + 1, dtype=np.float64)
        x_rad = np.deg2rad((n - _COFF) * 2.0**16 / _CFAC)
        y_rad = -np.deg2rad((n - _LOFF) * 2.0**16 / _LFAC)
        return x_rad * _SATELLITE_HEIGHT, y_rad * _SATELLITE_HEIGHT

    @property
    def image_interval(self: Self) -> float:
        """Image cadence in minutes (10 min for Full Disk)."""
        return 10.0

    @property
    def image_transforms(self: Self) -> tuple[ccrs.Projection, pyproj.CRS]:
        """Cartopy and pyproj projections for the AHI fixed grid."""
        cartopy_crs = ccrs.Geostationary(
            central_longitude=self.central_lon,
            satellite_height=_SATELLITE_HEIGHT,
            sweep_axis=self.sweep_axis,
            globe=ccrs.Globe(semimajor_axis=_SEMI_MAJOR_AXIS, semiminor_axis=_SEMI_MINOR_AXIS),
        )
        pyproj_crs = pyproj.CRS(
            proj="geos",
            h=_SATELLITE_HEIGHT,
            lon_0=self.central_lon,
            sweep=self.sweep_axis,
            a=_SEMI_MAJOR_AXIS,
            b=_SEMI_MINOR_AXIS,
        )
        return cartopy_crs, pyproj_crs

    @property
    def sat_pyproj_crs(self: Self) -> pyproj.CRS:
        """Pyproj CRS for the satellite projection."""
        _, pyproj_crs = self.image_transforms
        return pyproj_crs

    @property
    def satellite_height(self: Self) -> float:
        """Satellite height above the equator in metres."""
        return _SATELLITE_HEIGHT

    def fetch_image(
        self: Self,
        image_time: np.datetime64,
    ) -> tuple[np.ndarray, ccrs.Projection, tuple[float, float, float, float]]:
        """Fetch a frame and prepare its ash RGB for visualization.

        Args:
            image_time: Timestamp for the time of interest.

        Returns:
            Ash RGB array of shape (height, width, 3), the cartopy projection of the grid,
            and the image extent in that projection. All three are None if no data exist.
        """
        pd_time = pd.to_datetime(image_time)
        try:
            da = self._get_frame(pd_time)
        except ValueError as e:
            if "No data found" in str(e):
                logging.warning(f"No data found at {pd_time} for {self.satellite}, skipping.")
                return None, None, None
            raise

        rgb = to_ash(da)
        np.nan_to_num(rgb, copy=False)
        cartopy_crs, _ = self.image_transforms
        return rgb, cartopy_crs, tuple(self.image_full_extent)

    def plot_detections(
        self: Self,
        timestamps: list[np.datetime64] | None = None,
        detection_cmap: str | mcolors.Colormap = "Blues",
        prediction_threshold: float | None = None,
        show_domain: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        fig_output_dir: str | Path | None = None,
    ) -> list[plt.Figure]:
        """Plot the detection mask for each timestamp.

        Args:
            timestamps: Timestamps to plot. Defaults to the full sequence.
            detection_cmap: Colormap with which to plot detection mask.
            prediction_threshold: Pixels with a detection score below this
                value are hidden. If None, uses the default for the detection source.
            show_domain: Whether or not to outline the processed imaging domain.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to 80 degrees either side of the sub-satellite point.
            fig_output_dir: Directory in which to save the figures, as
                ``<YYYY-MM-DD>/<satellite>-<region>-<YYYY-MM-DDTHHMM>.png``. Each figure is
                closed once saved, so long time ranges do not build up open figures. If
                None, nothing is saved and the figures are returned.

        Returns:
            The figures if ``fig_output_dir`` is None, otherwise an empty list.

        Raises:
            ValueError: If detections have not been loaded on the detector.
        """
        if not self.detections:
            raise ValueError("Detections have not been loaded. Call get_detections() first.")

        if timestamps is None:
            timestamps = self.timestamp_sequence

        plot_frame = partial(
            self.plot_full_disk,
            detection_cmap=detection_cmap,
            prediction_threshold=prediction_threshold,
            show_domain=show_domain,
            bbox=bbox,
        )
        return self._plot_frames(plot_frame, timestamps, fig_output_dir, desc="Plotting detections")

    def plot_full_disk(
        self: Self,
        timestamp: np.datetime64,
        fig_output_dir: str | Path | None = None,
        detection_cmap: str | mcolors.Colormap = "Blues",
        prediction_threshold: float | None = None,
        show_domain: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> plt.Figure | None:
        """Plot one frame's Full Disk detection mask.

        Args:
            timestamp: Timestamp for the image to plot.
            fig_output_dir: Directory in which to save the figure. Not saved if None.
            detection_cmap: Colormap with which to plot detection mask.
            prediction_threshold: Pixels with a detection score below this
                value are hidden. If None, uses the default for the detection source.
            show_domain: Whether or not to outline the processed imaging domain.
            bbox: Bounding box for plotting, specified as (min_lon, max_lon, min_lat, max_lat).
                If None, the map is cropped to 80 degrees either side of the sub-satellite point.

        Returns:
            Matplotlib Figure, or None if the frame has no detection data.
        """
        detection = self.detections.get(timestamp)
        if detection is None:
            logging.warning(f"No detection data for {timestamp}, skipping.")
            return None

        return self._plot_detection_map(
            timestamp,
            detection,
            fig_output_dir=fig_output_dir,
            detection_cmap=detection_cmap,
            prediction_threshold=prediction_threshold,
            show_domain=show_domain,
            bbox=bbox,
        )

    def _frame_extent(self: Self, da: xr.DataArray) -> tuple[float, float, float, float]:
        """Extent of a frame; every Full Disk frame shares the fixed grid."""
        return tuple(self.image_full_extent)

    def _frame_projection(
        self: Self, da: xr.DataArray
    ) -> tuple[np.ndarray, np.ndarray, pyproj.CRS]:
        """Pixel coordinates and CRS of a frame, from the fixed grid.

        The frame's own x/y labels are ignored (see module docstring); only its shape is
        checked against the grid.

        Raises:
            ValueError: If the frame is not on the 2 km Full Disk grid.
        """
        shape = (da.sizes["y"], da.sizes["x"])
        if shape != (_GRID_SIZE, _GRID_SIZE):
            raise ValueError(
                f"Himawari frame has shape {shape}, expected the 2 km Full Disk grid "
                f"({_GRID_SIZE}, {_GRID_SIZE})."
            )
        geo_x, geo_y = self.image_geo_coords
        return geo_x, geo_y, self.sat_pyproj_crs

    def _gen_scan_time_offset(self: Self) -> np.ndarray:
        """Time from a frame's nominal timestamp to the moment each pixel was scanned.

        The Full Disk is scanned north to south in ten segments of 550 lines whose start
        and end times are recorded in the HSD headers. Time is interpolated linearly in
        line within each segment; segment durations are uneven (about 33-86 s), as AHI
        interleaves its regional scans, so a single linear ramp would be off by tens of
        seconds mid-disk. As for GOES, there is an unmodelled column term bounded by one
        swath time.

        Returns:
            Per-pixel offsets from nominal frame time, timedelta64 in milliseconds, shaped
            like the grid (a read-only view, constant along each row).
        """
        segments = self._segment_times_s()
        n_rows = self.lats.shape[0]

        first_rows = np.array([s[0] for s in segments]) - 1
        starts = np.array([s[1] for s in segments])
        ends = np.array([s[2] for s in segments])
        last_rows = np.append(first_rows[1:], n_rows)

        rows = np.arange(n_rows)
        seg = np.searchsorted(first_rows, rows, side="right") - 1
        frac = (rows - first_rows[seg] + 0.5) / (last_rows[seg] - first_rows[seg])
        offset_s = starts[seg] + frac * (ends[seg] - starts[seg])

        per_row = np.round(offset_s * 1000).astype(np.int64) * np.timedelta64(1, "ms")
        return np.broadcast_to(per_row[:, np.newaxis], self.lats.shape)

    def _gen_tiles(self: Self) -> None:
        """Generate tiles by decomposing the fixed grid."""
        if not self.timestamp_sequence:
            raise ValueError("Timestamp sequence is empty. Must populate before tile generation.")

        self._decompose_abi_fixed_grid(
            extent=self.tile_extent,
            overlap=32.0,
            tile_size=512.0,
        )

    def _get_frame(self: Self, time: pd.Timestamp) -> xr.DataArray:
        """Fetch the Himawari frame at one time through the pycontrails datalib."""
        return self.pycon_himawari.get(time)

    def _get_region_coords(self: Self) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and latitude of each pixel, NaN off the Earth disk."""
        geo_x, geo_y = self.image_geo_coords
        xx, yy = np.meshgrid(geo_x, geo_y, indexing="xy")

        lons, lats = self.sat_to_latlon_transform.transform(xx, yy)
        lons[np.isinf(lons)] = np.nan
        lats[np.isinf(lats)] = np.nan

        return lons, lats

    def _read_segment_times_s(self: Self, time: pd.Timestamp) -> list[tuple[int, float, float]]:
        """Read each Full Disk segment's first line and scan window from its HSD header.

        Only the start of each compressed file is downloaded: enough for one bz2 block,
        which holds the header.

        Args:
            time: Nominal frame time.

        Returns:
            ``(first_line, start_s, end_s)`` per segment, ordered north to south; lines
            count from 1 and times are seconds after ``time``.

        Raises:
            ValueError: If no B14 files exist for ``time``.
        """
        rpaths = self.pycon_himawari.s3_rpaths(time.to_pydatetime())["B14"]
        if not rpaths:
            raise ValueError(f"No data found for band B14 at time {time}")

        def offset_s(mjd: float) -> float:
            scan_time = _MJD_EPOCH + datetime.timedelta(days=mjd)
            return (scan_time - time.to_pydatetime()).total_seconds()

        segments = []
        for rpath in rpaths:
            decompressor = bz2.BZ2Decompressor()
            content = b""
            with self.pycon_himawari.fs.open(rpath, "rb", block_size=2**16) as f:
                while len(content) < _HEADER_READ_BYTES and not decompressor.eof:
                    chunk = f.read(2**16)
                    if not chunk:
                        break
                    content += decompressor.decompress(chunk)

            header = parse_himawari_header(content)
            basic = header["basic_information"]
            segments.append(
                (
                    int(header["segment_information"]["first_line_number"]),
                    offset_s(basic["obs_start_time"]),
                    offset_s(basic["obs_end_time"]),
                )
            )

        return sorted(segments)

    def _segment_times_s(self: Self) -> list[tuple[int, float, float]]:
        """Segment scan windows for this range, anchored on its first frame.

        Only the first frame is consulted, as for GOES: the AHI observation timeline
        repeats every 10 minutes. Falls back to ``FALLBACK_SEGMENT_TIMES_S`` when the
        headers cannot be read, a segment is missing, or they describe an implausible scan.

        Returns:
            ``(first_line, start_s, end_s)`` per segment, as ``_read_segment_times_s``.
        """
        lines_per_segment = _GRID_SIZE // len(self.FALLBACK_SEGMENT_TIMES_S)
        fallback = [
            (k * lines_per_segment + 1, start, end)
            for k, (start, end) in enumerate(self.FALLBACK_SEGMENT_TIMES_S)
        ]
        nominal = pd.Timestamp(self.timestamp_sequence[0])

        try:
            segments = self._read_segment_times_s(nominal)
        except Exception as e:  # noqa: BLE001 - any failure here has the same remedy
            logging.warning(
                f"{self.satellite}: could not read scan times at {nominal} "
                f"({type(e).__name__}: {e}); using tabulated segment times."
            )
            return fallback

        # Every segment must be present and start on its expected line: a missing segment
        # would otherwise leave its rows with a neighbouring segment's scan times.
        expected_first_lines = [first_line for first_line, _, _ in fallback]
        times = np.array([t for _, start, end in segments for t in (start, end)])
        plausible = (
            [first_line for first_line, _, _ in segments] == expected_first_lines
            and np.all(np.diff(times) >= 0.0)
            and times[0] >= 0.0
            and times[-1] <= 60.0 * self.image_interval + 120.0
        )
        if not plausible:
            logging.warning(
                f"{self.satellite}: scan times at {nominal} are implausible ({segments}); "
                f"using tabulated segment times."
            )
            return fallback

        return segments

    def _set_central_lon(self: Self) -> None:
        """Set the central longitude of the AHI projection."""
        self.central_lon = _SUB_LON
