"""
Geometry utility functions for contrail decomposition.

This module contains geometric calculations used in contrail skeleton analysis,
including distance calculations, angle computations, and coordinate operations.

"""

import numpy as np


def calculate_angle_matrix(
    loose_coords: np.ndarray,
    stem_coords: np.ndarray,
) -> np.ndarray:
    """Calculate angles between branch directions.

    Args:
        loose_coords: Array of loose end coordinates.
        stem_coords: Array of stem end coordinates.

    Returns:
        np.ndarray: Matrix of angles between all branch pairs.
    """
    n = len(loose_coords)
    angle_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                v1 = loose_coords[i] - stem_coords[i]
                v2 = loose_coords[j] - stem_coords[j]
                angle_matrix[i, j] = calculate_vector_angle(v1, v2)

    return angle_matrix


def calculate_convex_hull_areas(
    loose_coords: np.ndarray,
    stem_coords: np.ndarray,
) -> np.ndarray:
    """Calculate convex hull areas for all pairs of branches.

    Args:
        loose_coords: Array of loose end coordinates.
        stem_coords: Array of stem end coordinates.

    Returns:
        np.ndarray: Matrix of convex hull areas for all branch pairs.
    """
    from shapely import geometry

    n = len(loose_coords)
    area_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            points = np.vstack(
                [
                    loose_coords[i],
                    stem_coords[i],
                    loose_coords[j],
                    stem_coords[j],
                ]
            )
            hull_area = geometry.MultiPoint(points).convex_hull.area
            area_matrix[i, j] = area_matrix[j, i] = hull_area

    return area_matrix


def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances between coordinates.

    Args:
        coords: Array of coordinate pairs.

    Returns:
        np.ndarray: Matrix of pairwise distances.
    """
    return np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))


def calculate_vector_angle(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Calculate angle in degrees between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        float: Angle in degrees between the vectors.
    """
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def create_skeleton_segments(
    coords: np.ndarray,
    elbow_idx: int,
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two skeleton segments split at the elbow point.

    Args:
        coords: Array of skeleton coordinates.
        elbow_idx: Index of the elbow point.
        skeleton: Original skeleton array.

    Returns:
        Tuple of (skeleton_1, skeleton_2) split at elbow.
    """
    skeleton_1 = np.zeros_like(skeleton)
    skeleton_2 = np.zeros_like(skeleton)

    # First segment: start to elbow (inclusive)
    for i in range(elbow_idx + 1):
        c = coords[i]
        skeleton_1[c[0], c[1]] = 1

    # Second segment: elbow to end (inclusive)
    for i in range(elbow_idx, len(coords)):
        c = coords[i]
        skeleton_2[c[0], c[1]] = 1

    return skeleton_1, skeleton_2


def find_elbow_point(
    src: np.ndarray,
    dst: np.ndarray,
    coords: np.ndarray,
) -> tuple[int, float]:
    """Find the elbow point and return both index and normalized deviation.

    Args:
        src: Source endpoint coordinates as (x, y) array.
        dst: Destination endpoint coordinates as (x, y) array.
        coords: Array of skeleton coordinates along the branch.

    Returns:
        Tuple of (elbow_index, normalized_max_deviation).
    """
    # Convert coordinates to (x, y) format
    coords_xy = np.array([[c[1], c[0]] for c in coords])

    # Calculate distances from each point to the direct line
    line_vec = dst - src
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return len(coords) // 2, 0.0

    line_vec_norm = line_vec / line_length
    proj_lengths = np.dot(coords_xy - src, line_vec_norm)
    proj_points = np.outer(proj_lengths, line_vec_norm) + src
    dists_to_line = np.linalg.norm(coords_xy - proj_points, axis=1)

    # Find maximum deviation and normalize by line length
    max_deviation_idx = np.argmax(dists_to_line)
    max_deviation = dists_to_line[max_deviation_idx]
    normalized_deviation = max_deviation / line_length if line_length > 0 else 0

    return max_deviation_idx, normalized_deviation
