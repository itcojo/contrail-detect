"""
Contrail decomposition and geometric analysis.

This module provides tools for decomposing detected contrails into meaningful
geometric structures, including branch operations, geometry utilities, and
decomposition algorithms for analyzing contrail morphology.
"""

from contrail_detect.decomposition.branch_operations import (
    are_branches_compatible,
    connect_branches_via_path,
    extract_branch_endpoints,
    find_optimal_pairs,
    get_node_id_from_branch,
    pair_branches,
)
from contrail_detect.decomposition.decompose import (
    ContrailDecomposer,
    Decomposer,
    DecompositionConfig,
)
from contrail_detect.decomposition.geometry_utils import (
    calculate_angle_matrix,
    calculate_convex_hull_areas,
    calculate_distance_matrix,
    calculate_vector_angle,
)

__all__ = [
    # Branch operations
    "are_branches_compatible",
    "connect_branches_via_path",
    "extract_branch_endpoints",
    "find_optimal_pairs",
    "get_node_id_from_branch",
    "pair_branches",
    # Geometry utilities
    "calculate_angle_matrix",
    "calculate_convex_hull_areas",
    "calculate_distance_matrix",
    "calculate_vector_angle",
    # Decomposition classes
    "ContrailDecomposer",
    "Decomposer",
    "DecompositionConfig",
]
