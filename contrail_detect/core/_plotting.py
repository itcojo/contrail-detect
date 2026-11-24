"""
Plotting helpers shared by the detector, tracking and flight figures.

Internal: not part of the public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from contrail_detect.core.common import Colors, Transforms

if TYPE_CHECKING:
    from contrail_detect.geostationary.abc import Geostationary

colors = Colors()
transforms = Transforms()


def unwrap_lons(lons) -> np.ndarray:
    """Shift longitudes by whole turns so consecutive points differ by less than 180 degrees.

    A line through the result, drawn with a PlateCarree transform, crosses the date line
    the short way. NaNs are kept.

    Args:
        lons: Sequence of longitudes in degrees.

    Returns:
        The longitudes as a float array, each shifted by a multiple of 360 degrees.
    """
    lons = np.asarray(lons, dtype=float)
    if lons.ndim != 1 or lons.size < 2:
        return lons
    finite = np.flatnonzero(np.isfinite(lons))
    jumps = np.diff(lons[finite])
    # Undo each jump of more than 180 degrees from that point on
    turns = np.zeros(lons.size)
    turns[finite[1:]] = np.cumsum(np.where(np.abs(jumps) > 180.0, -np.round(jumps / 360.0), 0.0))
    return lons + 360.0 * turns


def lon_span(lons) -> tuple[float, float]:
    """Return the shortest longitude range that holds every given longitude.

    Args:
        lons: Longitudes in degrees, in [-180, 180). NaNs are ignored.

    Returns:
        (lon_min, lon_max). If the range crosses the date line, lon_max exceeds 180;
        otherwise these are the minimum and maximum of ``lons``.
    """
    lons = np.sort(np.asarray(lons, dtype=float).ravel())
    lons = lons[np.isfinite(lons)]
    gaps = np.diff(lons)
    # The range is the circle less its largest empty gap; prefer not crossing on a tie
    if gaps.size == 0 or gaps.max() <= lons[0] + 360.0 - lons[-1]:
        return float(lons[0]), float(lons[-1])
    i = int(np.argmax(gaps))
    return float(lons[i + 1]), float(lons[i] + 360.0)


def lon_range(lon_min: float, lon_max: float) -> tuple[float, float]:
    """Return a longitude range with lon_max >= lon_min, e.g. (170, -170) as (170, 190).

    Args:
        lon_min: Western edge in degrees.
        lon_max: Eastern edge in degrees.

    Returns:
        (lon_min, lon_max), with 360 added to lon_max if it was below lon_min.
    """
    if lon_max < lon_min:
        lon_max += 360.0
    return lon_min, lon_max


def add_globe_inset(
    fig: plt.Figure,
    zoom_extent: list,
    rect: list | None = None,
) -> None:
    """Add a small globe inset marking the zoom extent with a red box.

    The globe is centered on the middle of zoom_extent using an Orthographic projection
    so the region of interest is always facing forward.

    Args:
        fig: Figure to add the inset to.
        zoom_extent: [lon_min, lon_max, lat_min, lat_max] bounding box to highlight. A box
            across the date line may have lon_max < lon_min or lon_max > 180.
        rect: [left, bottom, width, height] of the inset in figure coordinates. If None,
            the inset sits in the figure's top-left corner.
    """
    lon_min, lon_max, lat_min, lat_max = zoom_extent
    lon_min, lon_max = lon_range(lon_min, lon_max)
    central_lon = (lon_min + lon_max) / 2
    if central_lon >= 180.0:
        central_lon -= 360.0
    central_lat = (lat_min + lat_max) / 2

    projection = ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat)
    if rect is None:
        rect = [0.02, 0.83, 0.15, 0.15]
    ax_inset = fig.add_axes(rect, projection=projection)
    ax_inset.set_global()
    ax_inset.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax_inset.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax_inset.coastlines(linewidth=0.3, color="dimgray")

    box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax_inset.plot(
        box_lons,
        box_lats,
        color="red",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


def disk_map_figure(
    detector: Geostationary, figsize: tuple[float, float]
) -> tuple[plt.Figure, plt.Axes]:
    """Figure with one map of the detector's disk, styled as the Full Disk detection maps.

    The map is centred on the sub-satellite point and cropped to 80 degrees either side
    of it.

    Args:
        detector: Detector whose satellite the map is centred on.
        figsize: Figure size in inches.

    Returns:
        The figure and its map axes.
    """
    map_crs = transforms.platecarree(central_lon=detector.central_lon)
    fig = plt.figure(figsize=figsize, dpi=150)
    with sns.axes_style(colors.plot_params_light):
        ax = fig.add_subplot(111, projection=map_crs)
        ax.set_extent(detector.FULL_DISK_MAP_EXTENT, crs=map_crs)
        ax.add_feature(cfeature.COASTLINE, edgecolor=colors.facecolor, linewidth=0.3, zorder=3)
        ax.add_feature(
            cfeature.BORDERS,
            edgecolor=colors.facecolor,
            linestyle="--",
            linewidth=0.3,
            zorder=3,
        )
        gl = ax.gridlines(
            draw_labels=True, linestyle=":", linewidth=0.5, color=colors.facecolor, zorder=1
        )
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(detector.MAP_LON_TICKS)
        gl.xlabel_style = gl.ylabel_style = {"fontsize": 7}
    return fig, ax


def save_figure(
    fig: plt.Figure, fig_output_dir: str | Path | None, name: str, dpi: float | None = None
) -> None:
    """Save a figure as ``<name>.png``, unless no directory is given.

    Args:
        fig: Figure to save.
        fig_output_dir: Directory to save into, created if needed. Nothing is saved if None.
        name: File name, without the extension.
        dpi: Resolution to save at. If None, uses the figure's own.
    """
    if fig_output_dir is None:
        return
    fig_output_dir = Path(fig_output_dir)
    fig_output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_output_dir / f"{name}.png", dpi=dpi or "figure", bbox_inches="tight")


def max_pool(
    scores: np.ndarray, x: np.ndarray, y: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a score grid by ``factor`` in each direction, keeping each block's maximum.

    Used for display only, so features narrower than a figure pixel are not lost. NaNs are
    ignored; a block with no finite scores stays NaN.

    Args:
        scores: 2D score grid, indexed (y, x).
        x: X coordinates of the grid columns.
        y: Y coordinates of the grid rows.
        factor: Block size in pixels. 1 returns the inputs unchanged.

    Returns:
        The x and y coordinates of the block centres (the mean of each block's coordinates)
        and the reduced score grid.
    """
    if factor <= 1:
        return x, y, scores

    def pad(a: np.ndarray, axis: int) -> np.ndarray:
        widths = [(0, 0)] * a.ndim
        widths[axis] = (0, -a.shape[axis] % factor)
        return np.pad(np.asarray(a, dtype=float), widths, constant_values=np.nan)

    padded = pad(pad(scores, 0), 1)
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    # fmax ignores NaN without warning on all-NaN blocks
    pooled = np.fmax.reduce(np.fmax.reduce(blocks, axis=3), axis=1)
    x_centres = np.nanmean(pad(x, 0).reshape(-1, factor), axis=1)
    y_centres = np.nanmean(pad(y, 0).reshape(-1, factor), axis=1)
    return x_centres, y_centres, pooled
