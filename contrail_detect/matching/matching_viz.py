"""Visualisation utilities for contrail-flight matching results."""

from __future__ import annotations

from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colormaps
from matplotlib.lines import Line2D

from contrail_detect.core._plotting import lon_span, save_figure, unwrap_lons
from contrail_detect.core.common import Colors, Transforms

colors, transforms = Colors(), Transforms()


def _rank1_prefix(matching_results: pd.DataFrame) -> str:
    """Return the rank-1 column prefix, ``'s1'`` if only ``s1_match`` is present, else ``'t1'``.

    Same rule as ``Matcher.describe``.
    """
    if "t1_match" not in matching_results and "s1_match" in matching_results:
        return "s1"
    return "t1"


def display_match(
    track_id: str,
    flight_id: str,
    matching_results: pd.DataFrame,
    flight_data: pd.DataFrame,
    adv_flight_data: pd.DataFrame,
    geo_point_clouds: dict,
    adv_timestep_minutes: int = 10,
    fov_pad: float = 2.5,
) -> go.Figure:
    """Plot an interactive Plotly figure for a single contrail-flight match.

    Renders a time-slider figure showing the raw flight path, advected flight
    waypoints, contrail point clouds, and contrail line segments, all filtered
    progressively up to the selected timestamp.

    Args:
        track_id: Track ID to visualize.
        flight_id: Flight ID to visualize.
        matching_results: Output DataFrame from Matcher.run_matching().
        flight_data: Raw flight waypoints DataFrame (from FlightData.flight_data).
        adv_flight_data: Advected flight waypoints DataFrame (from
            FlightData.adv_flight_data).
        geo_point_clouds: Dict mapping contrail_id to shapely MultiPoint in lon/lat
            (from Geostationary.get_geo_point_clouds()).
        adv_timestep_minutes: Advection timestep in minutes, used to set the slider
            range. Defaults to 10.
        fov_pad: Degrees of padding around the track centroid for the map extent.
            Defaults to 2.5.

    Returns:
        Plotly Figure with a time slider.

    Raises:
        ValueError: If the track is not in ``matching_results``.
    """
    track_df = (
        matching_results[matching_results["track_id"] == track_id]
        .reset_index(drop=True)
        .sort_values("timestamp")
    )
    if track_df.empty:
        raise ValueError(f"Track {track_id!r} is not in matching_results.")
    df_flight = (
        flight_data[flight_data["flight_id"].to_numpy() == flight_id]
        .reset_index(drop=True)
        .sort_values("time")
    )
    df_adv = adv_flight_data[adv_flight_data["flight_id"].to_numpy() == flight_id].reset_index(
        drop=True
    )

    cx = track_df.iloc[-1]["centroid_lon"]
    cy = track_df.iloc[-1]["centroid_lat"]

    start_ts = track_df[f"{_rank1_prefix(matching_results)}_first_formation_point_time"].min()
    if pd.isna(start_ts):
        start_ts = track_df["timestamp"].min()
    start_ts = start_ts - np.timedelta64(adv_timestep_minutes, "m")
    timestamps = list(
        np.arange(
            start_ts,
            track_df["timestamp"].max() + np.timedelta64(adv_timestep_minutes, "m"),
            np.timedelta64(adv_timestep_minutes, "m"),
            dtype="datetime64[m]",
        )
    )

    cmap = colormaps["tab20"]
    n_frames = len(track_df)
    frame_colors = cmap(np.linspace(0, 1, n_frames))
    frame_color_map = {
        row["timestamp"]: frame_colors[i] for i, (_, row) in enumerate(track_df.iterrows())
    }

    # Color map for advected frame_times: match contrail color where possible,
    # otherwise assign a gray from a sequential palette for pre-obs frames
    all_adv_frame_times = sorted(df_adv["frame_time"].unique())
    pre_obs_times = [ft for ft in all_adv_frame_times if ft not in frame_color_map]
    gray_cmap = colormaps["Greys"]
    pre_obs_colors = gray_cmap(np.linspace(0.3, 0.7, max(len(pre_obs_times), 1)))
    adv_color_map = {ft: pre_obs_colors[i] for i, ft in enumerate(pre_obs_times)}
    adv_color_map.update(frame_color_map)

    def rgba(c: np.ndarray, alpha: float = 1.0) -> str:
        return f"rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},{alpha})"

    # Permanent dummy traces — always visible, carry legend entries so they survive
    # slider changes (real data traces all have showlegend=False).
    legend_traces = [
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines",
            line=dict(color="black", width=1.5),
            name="Flight path",
            legendgroup="flight",
            showlegend=True,
            visible=True,
        ),
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            name="Advected path",
            legendgroup="adv",
            showlegend=True,
            visible=True,
        ),
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="markers",
            marker=dict(size=4, color="gray", opacity=0.5),
            name="Contrail point cloud",
            legendgroup="cloud",
            showlegend=True,
            visible=True,
        ),
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines+markers",
            line=dict(color="gray", width=2),
            marker=dict(size=5, color="gray"),
            name="Contrail center line",
            legendgroup="segment",
            showlegend=True,
            visible=True,
        ),
    ]
    n_legend = len(legend_traces)

    all_traces = legend_traces[:]
    step_trace_indices = []

    for ts_idx, ts in enumerate(timestamps):
        ts_pd = pd.Timestamp(ts)
        past_track = track_df[track_df["timestamp"] <= ts_pd]
        indices = []

        # Raw flight path filtered to current timestamp
        df_flight_ts = df_flight[df_flight["time"] <= ts_pd]
        all_traces.append(
            go.Scattergeo(
                lon=df_flight_ts["longitude"].tolist(),
                lat=df_flight_ts["latitude"].tolist(),
                mode="lines",
                line=dict(color="black", width=1.5),
                legendgroup="flight",
                showlegend=False,
                visible=(ts_idx == 0),
            )
        )
        indices.append(len(all_traces) - 1)

        # Advected positions: one dashed line per frame_time up to ts_pd
        adv_frame_times = sorted(df_adv[df_adv["frame_time"] <= ts_pd]["frame_time"].unique())
        for frame_time in adv_frame_times:
            adv_frame = df_adv[df_adv["frame_time"] == frame_time].sort_values("original_time")
            c = adv_color_map[frame_time]
            all_traces.append(
                go.Scattergeo(
                    lon=adv_frame["lon_pc"].tolist(),
                    lat=adv_frame["lat_pc"].tolist(),
                    mode="lines",
                    line=dict(color=rgba(c, 0.6), width=1),
                    legendgroup="adv",
                    showlegend=False,
                    visible=(ts_idx == 0),
                )
            )
            indices.append(len(all_traces) - 1)

        for _, row in past_track.iterrows():
            c = frame_color_map[row["timestamp"]]
            cid = row["contrail_id"]

            # Point cloud
            if cid in geo_point_clouds:
                pts = geo_point_clouds[cid]
                pc_lons = [pt.x for pt in pts.geoms]
                pc_lats = [pt.y for pt in pts.geoms]
            else:
                pc_lons, pc_lats = [], []

            all_traces.append(
                go.Scattergeo(
                    lon=pc_lons,
                    lat=pc_lats,
                    mode="markers",
                    marker=dict(size=3, color=rgba(c, 0.4)),
                    legendgroup="cloud",
                    showlegend=False,
                    visible=(ts_idx == 0),
                )
            )
            indices.append(len(all_traces) - 1)

            # Centre line
            all_traces.append(
                go.Scattergeo(
                    lon=[row["endpoint1_lon"], row["centroid_lon"], row["endpoint2_lon"]],
                    lat=[row["endpoint1_lat"], row["centroid_lat"], row["endpoint2_lat"]],
                    mode="lines+markers",
                    line=dict(color=rgba(c), width=2),
                    marker=dict(size=5, color=rgba(c)),
                    legendgroup="segment",
                    showlegend=False,
                    visible=(ts_idx == 0),
                )
            )
            indices.append(len(all_traces) - 1)

        step_trace_indices.append(indices)

    steps = []
    for ts_idx, ts in enumerate(timestamps):
        # Legend dummy traces are always visible; only data traces toggle.
        visibility = [True] * n_legend + [False] * (len(all_traces) - n_legend)
        for idx in step_trace_indices[ts_idx]:
            visibility[idx] = True
        steps.append(
            dict(
                method="update",
                args=[{"visible": visibility}],
                label=pd.Timestamp(ts).strftime("%H:%M"),
            )
        )

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(text=f"Track {track_id}  |  Flight {flight_id}", font=dict(size=11)),
        geo=dict(
            lonaxis=dict(range=[cx - fov_pad, cx + fov_pad]),
            lataxis=dict(range=[cy - fov_pad, cy + fov_pad]),
            showland=True,
            landcolor="#f0f0f0",
            showcoastlines=True,
            coastlinecolor="#aaa",
            showcountries=True,
            countrycolor="#aaa",
            showframe=False,
            projection_type="mercator",
            domain=dict(x=[0, 1], y=[0, 1]),
        ),
        sliders=[
            dict(
                active=0,
                steps=steps,
                currentvalue=dict(prefix="Time: ", font=dict(size=11)),
                pad=dict(t=50, b=10, l=50, r=50),
            )
        ],
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=40, b=100),
        height=650,
    )
    return fig


def plot_match(
    track_id: str,
    flight_id: str,
    matching_results: pd.DataFrame,
    adv_flight_data: pd.DataFrame,
    geo_point_clouds: dict,
    flight_data: pd.DataFrame | None = None,
    pad_deg: float = 0.3,
    ncols: int = 3,
    fig_output_dir: str | Path | None = None,
    score_col: str = "rms_distance",
) -> plt.Figure:
    """Plot a contrail-flight match as a static figure, one panel per frame of the track.

    Each panel shows the contrail's point cloud and centre line at that frame, with the
    flight's parallax-corrected waypoints advected to the frame's scan times and, if given,
    its original ADS-B path up to that frame. Every panel
    shares one extent, with the centre lines of earlier frames drawn faintly, so the
    contrail's drift reads across panels. Frames that enter the track score are marked
    "scored". The legend fills the first empty panel, or sits below a full grid. The rank-1
    columns read are ``t1_*``, or ``s1_*`` if the results only have ``s1_match``.

    Args:
        track_id: Track ID to plot.
        flight_id: Flight ID to plot, usually the track's rank-1 flight (``t1_match``).
        matching_results: Output DataFrame from Matcher.run_matching().
        adv_flight_data: Advected flight waypoints DataFrame (from
            FlightData.adv_flight_data), with ``frame_time``, ``lon_pc`` and ``lat_pc``.
        geo_point_clouds: Dict mapping contrail_id to shapely MultiPoint in lon/lat
            (from Geostationary.get_geo_point_clouds()).
        flight_data: Original flight waypoints (from FlightData.flight_data), with
            ``flight_id``, ``time``, ``longitude`` and ``latitude``. Optional.
        pad_deg: Padding around the contrails for the shared extent, in degrees of
            latitude. Defaults to 0.3.
        ncols: Maximum number of panels per row. Defaults to 3.
        fig_output_dir: Directory in which to save the figure, as
            ``match-<track_id>-<flight_id>.png``. Not saved if None.
        score_col: Primary score column without its rank prefix, as in
            ``Scorer.primary_score_col``. Marks the scored frames and gives the track score
            in the title. Defaults to ``'rms_distance'``.

    Returns:
        Matplotlib Figure.

    Raises:
        ValueError: If the track is not in ``matching_results``, or it has no
            ``<prefix>_<score_col>`` column.
    """
    track_df = matching_results[matching_results["track_id"] == track_id].sort_values("timestamp")
    if track_df.empty:
        raise ValueError(f"Track {track_id!r} is not in matching_results.")
    prefix = _rank1_prefix(matching_results)
    match_col, score_name = f"{prefix}_match", f"{prefix}_{score_col}"
    if score_name not in track_df:
        raise ValueError(f"matching_results has no {score_name!r} column; check score_col.")
    df_adv = adv_flight_data[adv_flight_data["flight_id"].to_numpy() == flight_id]
    df_flight = None
    if flight_data is not None:
        df_flight = flight_data[flight_data["flight_id"].to_numpy() == flight_id]
        df_flight = df_flight.sort_values("time")

    def cloud_lonlat(contrail_id: str) -> tuple[np.ndarray, np.ndarray]:
        points = geo_point_clouds.get(contrail_id)
        if points is None:
            return np.array([]), np.array([])
        return np.array([p.x for p in points.geoms]), np.array([p.y for p in points.geoms])

    # Shared extent around every frame's contrail, square on the ground, with longitudes
    # taken the short way across the date line
    lons, lats = [], []
    for _, row in track_df.iterrows():
        cloud_lon, cloud_lat = cloud_lonlat(row["contrail_id"])
        lons += [row["endpoint1_lon"], row["endpoint2_lon"], *cloud_lon]
        lats += [row["endpoint1_lat"], row["endpoint2_lat"], *cloud_lat]
    lon_min, lon_max = lon_span(lons)
    centre_lon, centre_lat = np.mean([lon_min, lon_max]), np.mean([min(lats), max(lats)])
    cos_lat = np.cos(np.radians(centre_lat))
    half = max((max(lats) - min(lats)) / 2, (lon_max - lon_min) * cos_lat / 2) + pad_deg
    extent = [
        centre_lon - half / cos_lat,
        centre_lon + half / cos_lat,
        centre_lat - half,
        centre_lat + half,
    ]
    # Maps reaching past the date line are centred on the contrails
    map_crs, map_extent = transforms.platecarree(), extent
    if extent[0] < -180.0 or extent[1] > 180.0:
        map_crs = transforms.platecarree(central_lon=centre_lon)
        map_extent = [extent[0] - centre_lon, extent[1] - centre_lon, *extent[2:]]

    contrail_color, flight_color = colors.wong[5], colors.wong[6]
    # Rows as evenly filled as the column limit allows
    n_frames = len(track_df)
    nrows = int(np.ceil(n_frames / ncols))
    ncols = int(np.ceil(n_frames / nrows))
    fig = plt.figure(figsize=(2.6 * ncols, 2.7 * nrows + 0.4), dpi=150)
    with sns.axes_style(colors.plot_params_light):
        for i, (_, row) in enumerate(track_df.iterrows()):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection=map_crs)
            ax.set_extent(map_extent, crs=map_crs)
            ax.add_feature(cfeature.COASTLINE, edgecolor=colors.facecolor, linewidth=0.3)
            ax.add_feature(
                cfeature.BORDERS, edgecolor=colors.facecolor, linestyle="--", linewidth=0.3
            )
            gl = ax.gridlines(
                draw_labels=True, linestyle=":", linewidth=0.5, color=colors.facecolor, zorder=1
            )
            gl.xlocator = gl.ylocator = mticker.MaxNLocator(3)
            gl.top_labels = gl.right_labels = False
            gl.left_labels = i % ncols == 0
            gl.bottom_labels = i >= n_frames - ncols
            gl.xlabel_style = gl.ylabel_style = {"fontsize": 6}

            for _, earlier in track_df.iloc[:i].iterrows():
                ax.plot(
                    unwrap_lons(
                        [
                            earlier["endpoint1_lon"],
                            earlier["centroid_lon"],
                            earlier["endpoint2_lon"],
                        ]
                    ),
                    [earlier["endpoint1_lat"], earlier["centroid_lat"], earlier["endpoint2_lat"]],
                    color="0.6",
                    linewidth=1.0,
                    alpha=0.6,
                    transform=transforms.platecarree(),
                    zorder=2,
                )

            cloud_lon, cloud_lat = cloud_lonlat(row["contrail_id"])
            ax.scatter(
                cloud_lon,
                cloud_lat,
                s=2,
                color=contrail_color,
                alpha=0.35,
                linewidths=0,
                transform=transforms.platecarree(),
                zorder=2,
            )
            ax.plot(
                unwrap_lons([row["endpoint1_lon"], row["centroid_lon"], row["endpoint2_lon"]]),
                [row["endpoint1_lat"], row["centroid_lat"], row["endpoint2_lat"]],
                color=contrail_color,
                linewidth=1.5,
                transform=transforms.platecarree(),
                zorder=3,
            )

            if df_flight is not None:
                flown = df_flight[df_flight["time"] <= row["timestamp"]]
                ax.plot(
                    unwrap_lons(flown["longitude"]),
                    flown["latitude"],
                    color=colors.facecolor,
                    linewidth=1.0,
                    transform=transforms.platecarree(),
                    zorder=4,
                )

            frame_adv = df_adv[df_adv["frame_time"] == row["timestamp"]]
            frame_adv = frame_adv.sort_values("original_time")
            ax.plot(
                unwrap_lons(frame_adv["lon_pc"]),
                frame_adv["lat_pc"],
                color=flight_color,
                linewidth=1.0,
                linestyle="--",
                marker=".",
                markersize=2,
                transform=transforms.platecarree(),
                zorder=4,
            )

            scored = row[match_col] == flight_id and pd.notna(row[score_name])
            label = pd.Timestamp(row["timestamp"]).strftime("%H:%M UTC")
            ax.set_title(f"{label}{' (scored)' if scored else ''}", fontsize=8)

    first = track_df.iloc[0]
    score = ""
    if first[match_col] == flight_id and pd.notna(first[score_name]):
        if score_col == "rms_distance":
            score = f", track RMS {first[score_name]:.2f} km"
        else:
            score = f", track {score_col} {first[score_name]:.2f}"
    fig.suptitle(f"Track {track_id}, flight {flight_id}{score}", fontsize=9)
    handles = [
        Line2D([], [], color=contrail_color, linewidth=1.5, label="Contrail centre line"),
        Line2D([], [], color="0.6", alpha=0.6, label="Earlier centre lines"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=contrail_color,
            alpha=0.35,
            label="Contrail pixels",
        ),
        Line2D(
            [],
            [],
            color=flight_color,
            linestyle="--",
            label="Flight, advected and\nparallax-corrected",
        ),
    ]
    if df_flight is not None:
        handles.append(
            Line2D([], [], color=colors.facecolor, linewidth=1.0, label="Flight, original path")
        )
    if n_frames < nrows * ncols:
        ax_legend = fig.add_subplot(nrows, ncols, n_frames + 1)
        ax_legend.axis("off")
        ax_legend.legend(handles=handles, loc="center", fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    save_figure(fig, fig_output_dir, f"match-{track_id}-{flight_id}")
    return fig
