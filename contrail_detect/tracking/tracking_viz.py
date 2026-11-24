"""
Visualisation utilities for contrail tracking results.

Provides a map of track centroid paths, and animated GIFs of contrail tracks over
a zoomed geographic region, with ghost-trail fading effects.
"""

from __future__ import annotations

import io
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from contrail_detect.core._plotting import (
    add_globe_inset,
    disk_map_figure,
    save_figure,
    unwrap_lons,
)
from contrail_detect.core.common import Colors, Transforms
from contrail_detect.geostationary.abc import Geostationary

colors = Colors()
transforms = Transforms()


@plt.rc_context({"font.size": 6})
def generate_tracking_GIF(
    detector: Geostationary,
    zoom_extent: list,
    n_ghost_frames: int = 4,
    frame_duration_ms: int = 500,
    dpi: int = 150,
    display_gif: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Render an animated GIF of contrail tracks over a zoomed region.

    For each frame in the detector's timestamp sequence, draws contrail line segments
    at full opacity with ghost trails from the preceding frames fading linearly to
    transparent. A globe inset in the top-right corner shows the location of the zoom
    region. The GIF can be displayed inline, saved to disk, or both.

    Args:
        detector: Geostationary detector instance with contrail_summary and
            timestamp_sequence set.
        zoom_extent: [lon_min, lon_max, lat_min, lat_max] bounding box for the plot.
        n_ghost_frames: Number of preceding frames to show as faded ghosts. Defaults to 4.
        frame_duration_ms: Duration of each GIF frame in milliseconds. Defaults to 500.
        dpi: Resolution of each rendered frame. Defaults to 150.
        display_gif: Whether to display the GIF inline in the notebook. Defaults to True.
        save_path: Optional file path to save the GIF to (e.g. 'output/tracks.gif').
            Parent directories are created if they don't exist. If None, no file is saved.

    Raises:
        ValueError: If the detector's timestamp_sequence is empty.
    """
    if len(detector.timestamp_sequence) == 0:
        raise ValueError("detector.timestamp_sequence is empty")
    contrails_to_plot = _filter_contrails_by_zoom_extent(detector.contrail_summary, zoom_extent)

    n_tracks = contrails_to_plot["track_id"].nunique()
    track_colors = plt.cm.tab20(np.linspace(0, 1, n_tracks))
    track_color_map = {
        track_id: track_colors[i % n_tracks]
        for i, track_id in enumerate(contrails_to_plot["track_id"].unique())
    }

    frames = []
    for t_idx, timestamp in enumerate(detector.timestamp_sequence):
        fig = plt.figure(figsize=(5.0, 5.0), dpi=dpi)
        plt.subplots_adjust(left=0.125, right=0.95, top=0.95, bottom=0.1)
        with sns.axes_style(colors.plot_params_light):
            ax1 = fig.add_subplot(111, projection=transforms.platecarree())
            dt_str = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M UTC")
            ax1.set_title(f"{detector.satellite}, Contrail Tracks - {dt_str}", fontsize=7)
            ax1.add_feature(
                cfeature.BORDERS, color=colors.facecolor, linestyle="--", linewidth=0.3, zorder=3
            )
            ax1.add_feature(
                cfeature.COASTLINE, color=colors.facecolor, linestyle="-", linewidth=0.3, zorder=3
            )
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
            plt.draw()

        for ghost_age in range(n_ghost_frames, 0, -1):
            ghost_idx = t_idx - ghost_age
            if ghost_idx < 0:
                continue
            # Linear fade: oldest ghost ~ 1/(n+1), most recent ghost ~ n/(n+1)
            alpha = (n_ghost_frames - ghost_age + 1) / (n_ghost_frames + 1)
            _plot_frame_contrails(
                ax1,
                contrails_to_plot,
                detector.timestamp_sequence[ghost_idx],
                track_color_map,
                zoom_extent,
                alpha=alpha,
            )

        _plot_frame_contrails(
            ax1,
            contrails_to_plot,
            timestamp,
            track_color_map,
            zoom_extent,
            alpha=1.0,
        )

        ax1.set_extent(zoom_extent, crs=transforms.platecarree())

        # Globe inset added after tight_layout to avoid being repositioned by the layout engine
        fig.tight_layout()
        add_globe_inset(fig, zoom_extent)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())  # .copy() required as PIL is lazy

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    gif_buf.seek(0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(gif_buf.read())
        gif_buf.seek(0)

    if display_gif:
        # IPython comes with the [examples] extra, so only needed for inline display
        from IPython.display import Image as IPImage
        from IPython.display import display

        display(IPImage(data=gif_buf.read()))


def plot_tracks(
    detector: Geostationary,
    contrail_summary: pd.DataFrame | None = None,
    fig_output_dir: str | Path | None = None,
) -> plt.Figure:
    """Plot the centroid path of each contrail track on a map of the detector's disk.

    Args:
        detector: Detector whose satellite the map is centred on.
        contrail_summary: Tracked contrails, with a ``track_id`` column as added by
            ``TrackingManager``. Defaults to ``detector.contrail_summary``. Untracked
            contrails are skipped.
        fig_output_dir: Directory in which to save the figure, as
            ``<satellite>-<region>-tracks.png``. Not saved if None.

    Returns:
        Matplotlib Figure.

    Raises:
        ValueError: If the contrails have not been tracked.
    """
    if contrail_summary is None:
        contrail_summary = detector.contrail_summary
    if "track_id" not in contrail_summary or contrail_summary["track_id"].isna().all():
        raise ValueError("Contrails have not been tracked. Run TrackingManager first.")
    tracks = contrail_summary.dropna(subset=["track_id"]).sort_values("timestamp")

    fig, ax = disk_map_figure(detector, figsize=(6.0, 5.0))
    for _, track in tracks.groupby("track_id"):
        ax.plot(
            unwrap_lons(track["centroid_lon"]),
            track["centroid_lat"],
            linewidth=0.5,
            marker=".",
            markersize=0.5,
            transform=transforms.platecarree(),
            zorder=4,
        )
    n_tracks = tracks["track_id"].nunique()
    plural = "" if n_tracks == 1 else "s"
    ax.set_title(
        f"{detector.satellite} {detector.region}, {n_tracks} contrail track{plural}",
        loc="left",
        fontsize=10,
    )

    save_figure(fig, fig_output_dir, f"{detector.satellite}-{detector.region}-tracks")
    return fig


def _filter_contrails_by_zoom_extent(
    contrail_summary: pd.DataFrame,
    zoom_extent: list,
) -> pd.DataFrame:
    """Filter contrail detections to those whose centroid falls within the zoom extent.

    Args:
        contrail_summary: DataFrame of contrail detections with centroid_lon/lat columns.
        zoom_extent: [lon_min, lon_max, lat_min, lat_max] bounding box.

    Returns:
        Filtered copy of contrail_summary.
    """
    return contrail_summary[
        (contrail_summary["centroid_lon"] >= zoom_extent[0])
        & (contrail_summary["centroid_lon"] <= zoom_extent[1])
        & (contrail_summary["centroid_lat"] >= zoom_extent[2])
        & (contrail_summary["centroid_lat"] <= zoom_extent[3])
    ].copy()


def _plot_frame_contrails(
    ax: plt.Axes,
    contrails: pd.DataFrame,
    timestamp: np.datetime64,
    track_color_map: dict[str, np.ndarray],
    zoom_extent: list,
    alpha: float = 1.0,
) -> None:
    """Plot contrail line segments for a single timestamp onto an existing axes.

    Filters to contrails whose final centroid falls within zoom_extent, then
    draws centroid paths and endpoint segments. Track ID labels are only drawn
    at full opacity to avoid ghost-frame clutter.

    Args:
        ax: Cartopy GeoAxes on which to draw.
        contrails: DataFrame of contrail detections with track_id and geometry columns.
        timestamp: Frame timestamp to plot detections for.
        track_color_map: Mapping from track_id to RGBA color array.
        zoom_extent: [lon_min, lon_max, lat_min, lat_max] bounding box filter.
        alpha: Opacity for all plotted elements. Defaults to 1.0.
    """
    contrails_for_timestamp = contrails[contrails["timestamp"] == timestamp]

    for track_id, track in contrails_for_timestamp.groupby("track_id"):
        if not (
            zoom_extent[0] <= track["centroid_lon"].iloc[-1] <= zoom_extent[1]
            and zoom_extent[2] <= track["centroid_lat"].iloc[-1] <= zoom_extent[3]
        ):
            continue

        c = track_color_map[track_id]

        ax.plot(
            unwrap_lons(track["centroid_lon"]),
            track["centroid_lat"],
            linestyle="-",
            linewidth=1,
            marker=".",
            markersize=2,
            color=c,
            alpha=alpha,
            transform=transforms.platecarree(),
        )

        for _, r in track.iterrows():
            ax.plot(
                unwrap_lons([r["endpoint1_lon"], r["endpoint2_lon"]]),
                [r["endpoint1_lat"], r["endpoint2_lat"]],
                linestyle="-",
                linewidth=0.3,
                color=c,
                alpha=alpha,
                transform=transforms.platecarree(),
            )

        if alpha == 1.0:
            ax.text(
                track.iloc[-1]["centroid_lon"],
                track.iloc[-1]["centroid_lat"],
                str(track_id),
                fontsize=4,
                ha="right",
                va="bottom",
                color=colors.facecolor,
                transform=transforms.platecarree(),
            )
