"""
Meteorological data retrieval and processing.

This module provides interfaces for accessing meteorological data from
various sources, primarily ECMWF products including ERA5 reanalysis data.
Supports interpolation and integration with flight and satellite data.
"""

from contrail_detect.meteorology.ecmwf import get_era5_from_local, interpolate_era5_to_points

__all__ = [
    "get_era5_from_local",
    "interpolate_era5_to_points",
]
