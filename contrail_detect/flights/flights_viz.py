"""
Visualisation utilities for flight data.

Provides a map of flight waypoint density over a detector's disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from contrail_detect.core._plotting import disk_map_figure, save_figure
from contrail_detect.core.common import Transforms

if TYPE_CHECKING:
    from contrail_detect.geostationary.abc import Geostationary

transforms = Transforms()


def plot_flight_coverage(
    detector: Geostationary,
    flight_data: pd.DataFrame,
    gridsize: int = 120,
    fig_output_dir: str | Path | None = None,
) -> plt.Figure:
    """Plot the density of flight waypoints on a map of the detector's disk.

    Shows where flight data are available to match contrails against.

    Args:
        detector: Detector whose satellite the map is centred on.
        flight_data: Flight waypoints with ``longitude`` and ``latitude`` columns.
        gridsize: Number of hexagons across the map.
        fig_output_dir: Directory in which to save the figure, as
            ``<satellite>-<region>-flight-coverage.png``. Not saved if None.

    Returns:
        Matplotlib Figure.

    Raises:
        ValueError: If flight_data has no waypoints with a longitude and latitude.
    """
    if flight_data[["longitude", "latitude"]].dropna().empty:
        raise ValueError("flight_data has no waypoints to plot.")
    fig, ax = disk_map_figure(detector, figsize=(6.5, 5.0))
    hexes = ax.hexbin(
        flight_data["longitude"],
        flight_data["latitude"],
        gridsize=gridsize,
        bins="log",
        cmap="YlOrRd",
        mincnt=1,
        transform=transforms.platecarree(),
        zorder=4,
    )
    cax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
    fig.colorbar(hexes, cax=cax, label="Waypoints per cell")
    ax.set_title(
        f"{detector.satellite} {detector.region}, flight waypoint coverage", loc="left", fontsize=10
    )

    save_figure(fig, fig_output_dir, f"{detector.satellite}-{detector.region}-flight-coverage")
    return fig
