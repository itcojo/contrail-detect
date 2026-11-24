"""
This module contains common utility functions.

"""

import logging
import shutil
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import numpy as np
from pyproj import Geod

from contrail_detect.core.config import TEST_DIR


class Colors:
    """Color schemes and plot parameters for data visualization.

    This class provides color palettes and Seaborn plot parameters for creating
    accessible, visually appealing plots with both dark and light themes.

    Attributes:
        edgecolor: Light gray color for edges and backgrounds.
        facecolor: Dark gray color for faces and backgrounds.
        tol: Paul Tol's colorblind-friendly color palette.
        wong: Wong's colorblind-friendly color palette.
    """

    def __init__(self: Self) -> None:
        """Initialize the Colors class with default color schemes."""
        # Colors for low eye strain plots
        self.edgecolor: str = "#d9d9d9"
        self.facecolor: str = "#272727"

        # Colorblind-friendly color palettes
        self.tol: list[str] = [
            "#332288",
            "#117733",
            "#44aa99",
            "#88ccee",
            "#ddcc77",
            "#cc6677",
            "#aa4499",
            "#882255",
        ]
        self.wong: list[str] = [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]

    @property
    def plot_params_light(self: Self) -> dict:
        """Get Seaborn plot parameters for light theme.

        Returns:
            dict: Dictionary of plot parameters compatible with Seaborn's axes_style.

        Example:
            >>> with sns.axes_style(colors.plot_params_light):
            ...     axes.set_xticklabels(labels)
        """
        return {
            "axes.facecolor": "w",
            "axes.edgecolor": self.facecolor,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.labelcolor": self.facecolor,
            "figure.facecolor": self.edgecolor,
            "grid.color": self.facecolor,
            "grid.linestyle": "-",
            "text.color": self.facecolor,
            "xtick.color": self.facecolor,
            "ytick.color": self.facecolor,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.solid_capstyle": "round",
            "patch.edgecolor": "w",
            "image.cmap": "rocket",
            "font.family": ["sans-serif"],
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "Bitstream Vera Sans",
                "sans-serif",
            ],
            "patch.force_edgecolor": True,
            "xtick.bottom": True,
            "xtick.top": True,
            "ytick.left": True,
            "ytick.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": True,
            "axes.spines.top": True,
        }


class Transforms:
    """Coordinate transformation utilities for geostationary satellite data.

    This class provides cartographic projections and coordinate transformations
    for various geostationary satellites including GOES, Himawari, and Meteosat.

    Attributes:
        GOES_east_lon: Central longitude for GOES-East satellites.
        GOES_west_lon: Central longitude for GOES-West satellites.
        HIMAWARI_lon: Central longitude for Himawari satellites.
        MTG_lon: Central longitude for Meteosat Third Generation.
        globe: WGS84 globe definition for cartographic projections.
        geod: Geodetic calculations object for Earth geometry.
    """

    def __init__(self: Self) -> None:
        """Initialize the Transforms class with satellite parameters and projections."""
        # Geostationary satellite central longitudes
        self.GOES_east_lon: float = -75.2
        self.GOES_west_lon: float = -137.0
        self.HIMAWARI_lon: float = 140.7
        self.MTG_lon: float = 0.0

        # Define a globe for projections
        self.globe: ccrs.Globe = ccrs.Globe(
            semimajor_axis=6378137.0,
            semiminor_axis=6356752.31414,
        )

        # Define Geod object for Earth calculations
        self.geod: Geod = Geod(ellps="WGS84")

        # Valid satellite names
        self.__valid_satellites: list[str] = [
            "GOES-16",
            "GOES-17",
            "GOES-18",
            "GOES-19",
            "HIMAWARI-8",
            "HIMAWARI-9",
            "MTG",
        ]

    def platecarree(self: Self, central_lon: float = 0.0) -> ccrs.PlateCarree:
        """Get PlateCarree projection (equirectangular lon-lat grid).

        Args:
            central_lon: Central longitude for the projection (degrees). Defaults to 0.

        Returns:
            PlateCarree projection configured with the specified central longitude.
        """
        return ccrs.PlateCarree(
            central_longitude=central_lon,
            globe=self.globe,
        )


def clean_pycontrails_cache(cache_dir: str | Path) -> None:
    """Empty a pycontrails cache directory.

    Args:
        cache_dir: Cache directory to empty, e.g. ``~/.cache/pycontrails/goes``.
    """
    cache_dir = Path(cache_dir)

    if cache_dir.exists():
        # Remove all files/subfolders inside the directory
        for item in cache_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        logging.warning(f"Cache directory {cache_dir} does not exist.")


def clean_temp_dir() -> None:
    """Clean the temporary directory used for storing data during inference.

    Removes all files and subdirectories from the temporary directory used
    during contrail detection inference. This helps free up disk space after
    processing.
    """
    test_dir_path = Path(TEST_DIR)

    if test_dir_path.exists():
        # Remove all files/subfolders inside the directory
        for item in test_dir_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        logging.warning(f"Temporary directory {test_dir_path} does not exist.")


def extract_date_time_components(
    dt64: np.datetime64,
) -> tuple[int, int, int, int, int, int]:
    """Extract date and time components from a numpy datetime64 object.

    Args:
        dt64: Datetime from which components are to be extracted.

    Returns:
        Tuple containing (year, month, day_of_year, day_of_month, hour, minute).
    """
    ex_year = np.datetime64(dt64, "Y").astype(int) + 1970
    ex_month = (np.datetime64(dt64, "M").astype(int) % 12) + 1
    ex_doy = (
        np.datetime64(dt64, "D") - np.datetime64(f"{ex_year}-01-01") + np.timedelta64(1, "D")
    ).astype(int)
    ex_dom = (
        np.datetime64(dt64, "D").astype(int)
        % np.datetime64(f"{ex_year}-{ex_month:02d}-01").astype(int)
        + 1
    )
    ex_hour = np.datetime64(dt64, "h").astype(int) % 24
    ex_minute = np.datetime64(dt64, "m").astype(int) % 60

    return ex_year, ex_month, ex_doy, ex_dom, ex_hour, ex_minute
