"""Geometry helpers shared by the tracking gates and costs."""

from contrail_detect.geostationary.geo_utils import (
    azimuth,
    convert_cartesian_latlon,
    convert_latlon_cartesian,
    find_gc_pole,
    haversine,
    hemisphere,
    perpendicular_distance_to_gc,
    project_points_on_gc,
    projected_overlap_fraction,
    unit_cross_product,
)


def dataclass_to_dict(obj: object) -> dict:
    """Recursively convert a dataclass instance to a dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return obj


__all__ = [
    "azimuth",
    "convert_cartesian_latlon",
    "convert_latlon_cartesian",
    "dataclass_to_dict",
    "find_gc_pole",
    "haversine",
    "hemisphere",
    "perpendicular_distance_to_gc",
    "projected_overlap_fraction",
    "project_points_on_gc",
    "unit_cross_product",
]
