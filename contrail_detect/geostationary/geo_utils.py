"""
Geometric and mathematical utility functions for geostationary satellite processing.

Standalone utility functions for geometric calculations, coordinate transformations,
and data normalization used across contrail detection and tracking.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pycontrails.physics.constants import radius_earth
from zarr.codecs import BloscCodec

# One whole frame per chunk. This is chosen for ALLOCATED disk usage, not apparent size:
# the RDS filesystem has an 8 mb block size, so every chunk file rounds up to a multiple of
# 8 mb and small chunks waste most of a block each. Measured over 12 real frames, allocated
# (du) against apparent (st_size):
#   (1,  512,  512)  12.7 mb apparent -> 25.2 mb allocated, 1082 files
#   (1, 1356, 1356)  13.0 mb apparent -> 15.9 mb allocated,  194 files
#   (1, 2712, 2712)  13.3 mb apparent -> 12.8 mb allocated,   50 files
#   (1, 5424, 5424)  13.1 mb apparent -> 11.1 mb allocated,   14 files
# Whole-frame chunks are 2.3x smaller on disk than (1, 512, 512) despite being marginally
# larger in apparent bytes. They also suit the access pattern: every consumer reads whole
# timestamps via isel(time=t), never spatial sub-blocks.
#
# Do not raise the time dimension above 1: blosc refuses buffers over 2 gb, and two
# float64 frames already exceed that.
#
# Spatial extent is resolved per array by detection_chunks(); this constant is the GOES
# full-disk case, kept for reference and for callers that already know the shape.
DETECTION_CHUNKS = (1, 5424, 5424)


def azimuth(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the forward azimuth in degrees from point 1 to point 2.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        Forward azimuth in degrees, in the range [0, 360).
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth_rad = np.arctan2(x, y)
    return (np.degrees(azimuth_rad) + 360) % 360


def convert_cartesian_latlon(
    x: float | NDArray,
    y: float | NDArray,
    z: float | NDArray,
    R: float = 1,
) -> tuple[NDArray, NDArray]:
    """Convert cartesian coordinates to latitude and longitude.

    Args:
        x: x coordinate.
        y: y coordinate.
        z: z coordinate.
        R: Radius, by default 1.

    Returns:
        Tuple of (latitude, longitude) in radians.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    return np.arcsin(z / R), np.arctan2(y, x)


def convert_latlon_cartesian(
    lat: float | NDArray,
    lon: float | NDArray,
    R: float = 1,
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert latitude and longitude to cartesian coordinates.

    Args:
        lat: Latitude in radians.
        lon: Longitude in radians.
        R: Radius, by default 1.

    Returns:
        Tuple of (x, y, z) cartesian coordinates in the same units as R.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    return R * np.cos(lat) * np.cos(lon), R * np.cos(lat) * np.sin(lon), R * np.sin(lat)


def detection_chunks(shape: tuple) -> tuple:
    """Chunk shape for a detection array: one whole frame per chunk.

    Derived from the array rather than hardcoded, so non-GOES grids (Himawari, MTG) and
    cropped regions get whole-frame chunks too.

    Args:
        shape: The ``(time, geo_y, geo_x)`` shape of the detection array.

    Returns:
        Chunk shape ``(1, n_row, n_col)``.
    """
    _, n_row, n_col = shape
    return (1, n_row, n_col)


def detection_encoding(shape: tuple | None = None) -> dict:
    """Zarr encoding for the ``detection`` variable of a detection archive.

    Without an explicit encoding (no compressor, no chunk choice), archives reach about
    2.1 GB per day.

    Compression is lossless, so this is free: only the storage threshold discards anything.

    Args:
        shape: Optional ``(time, geo_y, geo_x)`` shape, used to derive whole-frame chunks.
            Falls back to the GOES full-disk ``DETECTION_CHUNKS`` when omitted.

    Returns:
        Encoding dict suitable for ``to_zarr(encoding={"detection": ...})``.
    """
    chunks = detection_chunks(shape) if shape is not None else DETECTION_CHUNKS
    return {
        "chunks": chunks,
        "compressors": [BloscCodec(cname="zstd", clevel=5)],
    }


def directory_size_mb(path: Path) -> float:
    """Apparent size of a directory tree in megabytes.

    Used to report the before/after size of a converted detection archive. Bounded to a
    single zarr store, not a walk of a wider data tree.

    This sums ``st_size``, which is the logical byte count and NOT what the filesystem
    allocates. On a store of many small chunk files the two diverge sharply — the RDS
    filesystem uses 8 mb blocks, so an archive measuring 146 mb here occupied 289 mb of
    disk. ``DETECTION_CHUNKS`` keeps the file count low enough that the gap is negligible,
    but use ``du`` if you need the allocated figure.

    Args:
        path: Directory (or file) to measure.

    Returns:
        Apparent size in megabytes.
    """
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def extents_match(
    extent1: tuple[float, float, float, float],
    extent2: tuple[float, float, float, float],
    tolerance: float = 1e-4,
) -> bool:
    """Check if two extents match within a tolerance.

    Args:
        extent1: First extent as (min_lon, max_lon, min_lat, max_lat).
        extent2: Second extent as (min_lon, max_lon, min_lat, max_lat).
        tolerance: Maximum allowed difference for each coordinate. Defaults to 1e-4.

    Returns:
        True if extents match within tolerance, False otherwise.
    """
    if extent1 is None or extent2 is None:
        return False

    for e1, e2 in zip(extent1, extent2):
        # A missing value matches nothing (abs(nan - x) >= tolerance would be False)
        if np.isnan(e1) or np.isnan(e2):
            return False
        # Handle infinity values
        if np.isinf(e1) or np.isinf(e2):
            # Both must be inf with the same sign to match
            if not (np.isinf(e1) and np.isinf(e2) and np.sign(e1) == np.sign(e2)):
                return False
        # Regular comparison for finite values
        elif abs(e1 - e2) >= tolerance:
            return False

    return True


def find_gc_pole(
    latA: float,
    lonA: float,
    latB: float,
    lonB: float,
) -> tuple[float, float]:
    """Compute the pole of a great circle defined by two points.

    Args:
        latA: Latitude of first great circle point in radians.
        lonA: Longitude of first great circle point in radians.
        latB: Latitude of second great circle point in radians.
        lonB: Longitude of second great circle point in radians.

    Returns:
        Tuple of (latitude, longitude) of the pole in radians.
    """
    unit_cross = unit_cross_product(
        np.array(convert_latlon_cartesian(latA, lonA, 1)),
        np.array(convert_latlon_cartesian(latB, lonB, 1)),
    )

    return convert_cartesian_latlon(unit_cross[0], unit_cross[1], unit_cross[2], 1)


def haversine(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    input_unit: Literal["degrees", "radians"] = "radians",
) -> float:
    """Calculate the great circle distance in meters between two points on Earth.

    Args:
        lon1: Longitude of the first point.
        lat1: Latitude of the first point.
        lon2: Longitude of the second point.
        lat2: Latitude of the second point.
        input_unit: Unit of the input coordinates, either "degrees" or "radians".
            Defaults to "radians".

    Returns:
        Great circle distance between the two points in meters.
    """
    if input_unit == "degrees":
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_earth * c  # in meters


def hemisphere(
    p_lat: float,
    p_lon: float,
    gc1_lat: float,
    gc1_lon: float,
    gc2_lat: float,
    gc2_lon: float,
) -> float:
    """Return the signed hemisphere (+1 or -1) of a point relative to a great circle.

    Args:
        p_lat: Latitude of the point in radians.
        p_lon: Longitude of the point in radians.
        gc1_lat: Latitude of first great circle endpoint in radians.
        gc1_lon: Longitude of first great circle endpoint in radians.
        gc2_lat: Latitude of second great circle endpoint in radians.
        gc2_lon: Longitude of second great circle endpoint in radians.

    Returns:
        +1.0 or -1.0 indicating which side of the great circle the point lies on.
    """
    p_cart = convert_latlon_cartesian(p_lat, p_lon)
    gc1_cart = convert_latlon_cartesian(gc1_lat, gc1_lon)
    gc2_cart = convert_latlon_cartesian(gc2_lat, gc2_lon)

    return np.sign(unit_cross_product(np.array(gc1_cart), gc2_cart) @ p_cart)


def latlon_triangle_area(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate area of triangle defined by three lat/lon points on Earth's surface.

    Uses spherical excess formula to compute area on a sphere.

    Args:
        lat0: Latitude of first vertex in degrees.
        lon0: Longitude of first vertex in degrees.
        lat1: Latitude of second vertex in degrees.
        lon1: Longitude of second vertex in degrees.
        lat2: Latitude of third vertex in degrees.
        lon2: Longitude of third vertex in degrees.

    Returns:
        Area of the triangle in square kilometers.
    """
    # Convert degrees to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Spherical excess formula components
    a = np.arccos(
        np.sin(lat1_rad) * np.sin(lat2_rad)
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    )
    b = np.arccos(
        np.sin(lat0_rad) * np.sin(lat2_rad)
        + np.cos(lat0_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon0_rad)
    )
    c = np.arccos(
        np.sin(lat0_rad) * np.sin(lat1_rad)
        + np.cos(lat0_rad) * np.cos(lat1_rad) * np.cos(lon1_rad - lon0_rad)
    )

    s = (a + b + c) / 2.0

    # Spherical excess
    tan_e_over_4 = np.sqrt(
        np.tan(s / 2.0) * np.tan((s - a) / 2.0) * np.tan((s - b) / 2.0) * np.tan((s - c) / 2.0)
    )
    spherical_excess = 4.0 * np.arctan(tan_e_over_4)

    # Area calculation
    area_km2 = spherical_excess * (radius_earth**2) / 1e6  # Convert m^2 to km^2

    return area_km2


def normalize_range(
    data: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    """Map data to the range [0, 1].

    Args:
        data: Input data array.
        bounds: Tuple of (min_val, max_val) defining the input range.

    Returns:
        Normalized data array with values in [0, 1].
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def perpendicular_distance_to_gc(
    lat: NDArray,
    lon: NDArray,
    gc_start_lat: float,
    gc_start_lon: float,
    gc_end_lat: float,
    gc_end_lon: float,
) -> NDArray:
    """Compute the perpendicular distance from points to a great circle.

    Args:
        lat: Latitude(s) of point(s) in radians.
        lon: Longitude(s) of point(s) in radians.
        gc_start_lat: Latitude of the great circle beginning point in radians.
        gc_start_lon: Longitude of the great circle beginning point in radians.
        gc_end_lat: Latitude of the great circle end point in radians.
        gc_end_lon: Longitude of the great circle end point in radians.

    Returns:
        Perpendicular distance(s) from the point(s) to the great circle in meters.
    """
    projections, _ = project_points_on_gc(
        lat, lon, gc_start_lat, gc_start_lon, gc_end_lat, gc_end_lon
    )
    proj_lats, proj_lons = projections
    return haversine(lon, lat, proj_lons, proj_lats, input_unit="radians")


def project_points_on_gc(
    lat: NDArray,
    lon: NDArray,
    gc_start_lat: float,
    gc_start_lon: float,
    gc_end_lat: float,
    gc_end_lon: float,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Project lat/lon points onto a great circle defined by two endpoints.

    Args:
        lat: Latitude(s) of point(s) to project in radians.
        lon: Longitude(s) of point(s) to project in radians.
        gc_start_lat: Latitude of the great circle beginning point in radians.
        gc_start_lon: Longitude of the great circle beginning point in radians.
        gc_end_lat: Latitude of the great circle end point in radians.
        gc_end_lon: Longitude of the great circle end point in radians.

    Returns:
        A tuple of two (lats, lons) tuples. The first contains the closer
        projections onto the great circle; the second contains the antipodal
        (further) projections.
    """
    # GC pole computed once from endpoint Cartesians (avoids find_gc_pole lat/lon
    # roundtrip and the duplicate unit_cross_product call below)
    ep1_cart = np.array(convert_latlon_cartesian(gc_start_lat, gc_start_lon, 1))
    ep2_cart = np.array(convert_latlon_cartesian(gc_end_lat, gc_end_lon, 1))
    pole = np.cross(ep1_cart, ep2_cart)
    pole /= np.linalg.norm(pole)

    xs, ys, zs = convert_latlon_cartesian(lat, lon, 1)
    points = np.transpose([xs, ys, zs])

    # Broadcasting instead of np.tile — avoids two N×3 temporary allocations
    pole_crosses = np.cross(points, pole)

    pole_norms = np.linalg.norm(pole_crosses, axis=1)

    unit_pole_crosses = pole_crosses / pole_norms[:, None]

    intersections_guides = np.cross(pole, unit_pole_crosses)
    guide_norms = np.linalg.norm(intersections_guides, axis=1)

    unit_intersection_guides = intersections_guides / guide_norms[:, None]

    intersection_lats, intersection_lons = convert_cartesian_latlon(
        unit_intersection_guides[:, 0],
        unit_intersection_guides[:, 1],
        unit_intersection_guides[:, 2],
        1,
    )
    reverse_intersection_lats, reverse_intersection_lons = convert_cartesian_latlon(
        -unit_intersection_guides[:, 0],
        -unit_intersection_guides[:, 1],
        -unit_intersection_guides[:, 2],
        1,
    )

    return (
        (intersection_lats, intersection_lons),
        (reverse_intersection_lats, reverse_intersection_lons),
    )


def projected_overlap_fraction(
    projectee_lat1: float,
    projectee_lon1: float,
    projectee_lat2: float,
    projectee_lon2: float,
    ref_lat1: float,
    ref_lon1: float,
    ref_lat2: float,
    ref_lon2: float,
    tol: float = 1e-3,
) -> float:
    """Compute the fractional overlap between two segments projected onto a great circle.

    All inputs are in radians. The reference segment's endpoints are projected onto
    the great circle defined by the projectee segment, and the overlap fraction
    relative to the projectee segment length is returned.

    Args:
        projectee_lat1: Latitude of the first endpoint of the projectee segment in radians.
        projectee_lon1: Longitude of the first endpoint of the projectee segment in radians.
        projectee_lat2: Latitude of the second endpoint of the projectee segment in radians.
        projectee_lon2: Longitude of the second endpoint of the projectee segment in radians.
        ref_lat1: Latitude of the first endpoint of the reference segment in radians.
        ref_lon1: Longitude of the first endpoint of the reference segment in radians.
        ref_lat2: Latitude of the second endpoint of the reference segment in radians.
        ref_lon2: Longitude of the second endpoint of the reference segment in radians.
        tol: Tolerance in meters for determining whether a projected point lies within
            the projectee segment. Defaults to 1e-3.

    Returns:
        Fraction of the projectee segment covered by the projection of the reference
        segment, in the range [0, 1].
    """
    # Project seg2's endpoints onto seg1's great circle
    proj2_on_1, _ = project_points_on_gc(
        np.array([ref_lat1, ref_lat2]),
        np.array([ref_lon1, ref_lon2]),
        projectee_lat1,
        projectee_lon1,
        projectee_lat2,
        projectee_lon2,
    )
    # Positions along seg1
    seg1_length = haversine(projectee_lon1, projectee_lat1, projectee_lon2, projectee_lat2)

    def pos_on_seg1(lat: float, lon: float) -> float:
        d_start = haversine(projectee_lon1, projectee_lat1, lon, lat)
        d_end = haversine(projectee_lon2, projectee_lat2, lon, lat)
        if abs((d_start + d_end) - seg1_length) < tol:
            return d_start
        elif d_start < d_end:
            return 0
        else:
            return seg1_length

    pos2a = pos_on_seg1(proj2_on_1[0][0], proj2_on_1[1][0])
    pos2b = pos_on_seg1(proj2_on_1[0][1], proj2_on_1[1][1])
    seg2_on_1_start = min(pos2a, pos2b)
    seg2_on_1_end = max(pos2a, pos2b)

    # The interval of seg1 is [0, seg1_length]
    overlap_start = max(0, seg2_on_1_start)
    overlap_end = min(seg1_length, seg2_on_1_end)
    overlap_length = max(0, overlap_end - overlap_start)
    return overlap_length / seg1_length if seg1_length > 0 else 0


def same_location(
    ext1: np.ndarray,
    ext2: np.ndarray,
    tol: float = 1e-3,
) -> bool:
    """Check if two image extents match within tolerance.

    Args:
        ext1: First image extent array.
        ext2: Second image extent array.
        tol: Absolute tolerance for comparison. Defaults to 1e-3.

    Returns:
        True if all extent values match within tolerance, False otherwise.
    """
    return all(abs(a - b) < tol for a, b in zip(ext1, ext2))


def unit_cross_product(
    vec1: NDArray,
    vec2: NDArray,
) -> NDArray:
    """Calculate the unit cross product of two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Unit cross product of vec1 and vec2.
    """
    cross_product = np.cross(vec1, vec2)
    return cross_product / np.linalg.norm(cross_product)
