"""
Flight data processing and integration.

This module provides interfaces for loading and processing flight trajectory
data from the Contrails.org ADS-B API or your own data.
Supports matching flight data with satellite observations for contrail analysis.
"""

from contrail_detect.flights.contrails_api import ContrailsAPIFlightData
from contrail_detect.flights.flight_utils import (
    intersection_with_ellipsoid,
    merge_dataframes_with_index,
    resolve_adv_timestep,
    validate_and_parse_frame_timestamps,
    validate_flight_ecmwf_domains,
)
from contrail_detect.flights.flights_base import FlightData
from contrail_detect.flights.flights_viz import plot_flight_coverage
from contrail_detect.flights.generic import GenericFlightData

__all__ = [
    "ContrailsAPIFlightData",
    "FlightData",
    "GenericFlightData",
    "intersection_with_ellipsoid",
    "merge_dataframes_with_index",
    "plot_flight_coverage",
    "resolve_adv_timestep",
    "validate_and_parse_frame_timestamps",
    "validate_flight_ecmwf_domains",
]
