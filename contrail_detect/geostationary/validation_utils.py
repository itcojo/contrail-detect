"""
Validation utility functions for geostationary satellite processing.

This module contains standalone validation functions for timestamps, spatial extents,
locations, and satellite configuration parameters used in contrail detection.
"""

import numpy as np


def validate_timestamps(
    timestamp_sequence: list[np.datetime64],
    satellite: str,
    satellite_dates: dict[str, list[np.datetime64]],
) -> None:
    """Validate that timestamps fall within satellite operational dates.

    Args:
        timestamp_sequence: List of timestamps to validate.
        satellite: Name of the satellite (e.g., 'GOES-16').
        satellite_dates: Dictionary mapping satellite names to [start, end] dates.

    Raises:
        ValueError: If timestamps fall outside satellite operational dates.
    """
    # Check if requested first timestamp is before satellite went online
    if timestamp_sequence[0] < satellite_dates[satellite.upper()][0]:
        raise ValueError(
            f"Requested first timestamp ({timestamp_sequence[0]}) is before "
            f"the valid first available timestamp for the requested satellite "
            f"({satellite_dates[satellite.upper()][0]})."
        )

    # Check if requested last timestamp is after satellite went offline
    if timestamp_sequence[-1] > satellite_dates[satellite.upper()][1]:
        raise ValueError(
            f"Requested last timestamp ({timestamp_sequence[-1]}) is after "
            f"the valid last available timestamp for the requested satellite "
            f"({satellite_dates[satellite.upper()][1]})."
        )


def validate_extent(
    extent: tuple[float, float, float, float],
    min_longitude: float,
    max_longitude: float,
    min_latitude: float,
    max_latitude: float,
) -> None:
    """Validate spatial extent for tile generation or data processing.

    Ensures that the specified bounding box falls within the valid detection
    domain defined by the satellite's coverage area and model training constraints.

    Args:
        extent: Bounding box as (min_lon, max_lon, min_lat, max_lat).
        min_longitude: Minimum valid longitude for the satellite domain.
        max_longitude: Maximum valid longitude for the satellite domain.
        min_latitude: Minimum valid latitude for the satellite domain.
        max_latitude: Maximum valid latitude for the satellite domain.

    Raises:
        ValueError: If any extent boundary exceeds the valid domain limits or
            if max values are less than min values.
    """
    if extent[0] < min_longitude:
        raise ValueError(f"Minimum longitude ({extent[0]}) is less than limit of {min_longitude}.")

    if extent[1] > max_longitude:
        raise ValueError(f"Maximum longitude ({extent[1]}) is greater than limit {max_longitude}.")

    if extent[2] < min_latitude:
        raise ValueError(f"Minimum latitude ({extent[2]}) is less than limit of {min_latitude}.")

    if extent[3] < extent[2]:
        raise ValueError(
            f"Maximum latitude ({extent[3]}) is less than minimum latitude ({extent[2]})."
        )

    if extent[3] > max_latitude:
        raise ValueError(f"Maximum latitude ({extent[3]}) is greater than limit of {max_latitude}.")


def validate_overlap(overlap: float, tile_size: float) -> None:
    """Validate specified tile overlap.

    Args:
        overlap: Specified tile overlap, in units of km.
        tile_size: Size of tiles, in units of km.

    Raises:
        ValueError: If overlap is negative or too large for the tile size.
    """
    if overlap < 0.0:
        raise ValueError(f"{overlap} km of overlap is invalid, please choose a positive value.")

    if 2.0 * overlap > tile_size:
        raise ValueError(f"{overlap} km of overlap does not fit inside {tile_size} km tiles.")


def validate_satellite_inputs(
    region: str,
    satellite: str,
    valid_regions: set[str],
    valid_satellites: set[str],
) -> None:
    """Validate region and satellite inputs against valid sets.

    Args:
        region: Satellite region identifier.
        satellite: Satellite name.
        valid_regions: Set of valid region identifiers.
        valid_satellites: Set of valid satellite names.

    Raises:
        ValueError: If region or satellite are not in the valid sets.
    """
    if region not in valid_regions:
        raise ValueError(f"Invalid region '{region}'. Must be one of {sorted(valid_regions)}.")

    if satellite not in valid_satellites:
        raise ValueError(
            f"Invalid satellite '{satellite}'. Must be one of {sorted(valid_satellites)}."
        )
