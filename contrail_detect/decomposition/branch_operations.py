"""
Branch operations for contrail decomposition.

This module contains operations for analyzing, pairing, and connecting branches
in contrail skeleton structures, including geometric compatibility checks and
optimal pairing algorithms.

"""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely import geometry
from skan import Skeleton, summarize

from contrail_detect.decomposition.geometry_utils import (
    calculate_angle_matrix,
    calculate_convex_hull_areas,
    calculate_distance_matrix,
    calculate_vector_angle,
    create_skeleton_segments,
    find_elbow_point,
)


def add_branch_with_path_connection(
    contrail_branches: set[int],
    branch: pd.Series,
    reference_branch: pd.Series,
    branch_graph: nx.Graph,
    graph_nodes: set[int],
    prefer_stem_start: bool = True,
    prefer_stem_end: bool = True,
) -> None:
    """Add a branch to contrail and connect via shortest path if possible.

    Attempts to find and include intermediate branches along the shortest path
    between the reference branch and the new branch in the skeleton graph.

    Args:
        contrail_branches: Set of branch IDs in the contrail, modified in-place.
        branch: Branch to add to the contrail.
        reference_branch: Branch to connect from.
        branch_graph: NetworkX graph of branch connectivity.
        graph_nodes: Set of valid node IDs in the graph.
        prefer_stem_start: Whether to prefer stem node for reference branch.
        prefer_stem_end: Whether to prefer stem node for branch to add.
    """
    contrail_branches.add(branch["branch_id"])

    # Try to connect via shortest path
    start_node = get_node_id_from_branch(reference_branch, prefer_stem=prefer_stem_start)
    end_node = get_node_id_from_branch(branch, prefer_stem=prefer_stem_end)

    if start_node in graph_nodes and end_node in graph_nodes:
        edge_ids = connect_branches_via_path(branch_graph, start_node, end_node)
        contrail_branches.update(edge_ids)


def add_unassigned_branches(
    contrails: dict[int, list[int]],
    all_branch_ids: set[int],
) -> dict[int, list[int]]:
    """Add any remaining unassigned branches as individual contrails.

    Identifies branches not yet assigned to any contrail and adds each as a
    separate single-branch contrail.

    Args:
        contrails: Current contrail assignments mapping contrail IDs to branch IDs.
        all_branch_ids: Set of all branch IDs in the skeleton.

    Returns:
        Updated contrail assignments with unassigned branches added.
    """
    used_branches = set()
    for branch_ids in contrails.values():
        used_branches.update(branch_ids)

    unused_branch_ids = all_branch_ids - used_branches

    # Add each unused branch as a separate contrail
    new_contrail_index = len(contrails)
    for branch_id in unused_branch_ids:
        contrails[new_contrail_index] = [branch_id]
        new_contrail_index += 1

    return contrails


def are_branches_compatible(
    branch1: pd.Series,
    branch2: pd.Series,
    angle_threshold: float,
    area_to_distance_ratio_threshold: float,
) -> bool:
    """Check if two branches are geometrically compatible.

    Uses geometric criteria to determine if branches belong to the same contrail,
    including angular alignment and spatial proximity. branch2 is tried in both
    directions unless it is Type I, which is taken from loose end to stem only.

    Args:
        branch1: First branch data.
        branch2: Second branch data.
        angle_threshold: Minimum angle in degrees for branch compatibility.
        area_to_distance_ratio_threshold: Maximum ratio for compatibility.

    Returns:
        True if branches are compatible, False otherwise.
    """
    # Get coordinates for branch1
    coord1_loose, coord1_stem = extract_branch_endpoints(
        branch1,
        use_loose_end=True,
    )

    # A reversed Type I branch would let the arms of a narrow fork pass as one line
    if "image_coord_loose_0" in branch2 and pd.notna(branch2["image_coord_loose_0"]):
        coord2_loose, coord2_stem = extract_branch_endpoints(branch2, use_loose_end=True)
        orientations = [(coord2_loose, coord2_stem)]
    else:
        coord2_src, coord2_dst = extract_branch_endpoints(branch2, use_loose_end=False)
        orientations = [(coord2_src, coord2_dst), (coord2_dst, coord2_src)]

    best_ratio = np.inf

    for coord2_loose, coord2_stem in orientations:
        # Calculate vectors
        v1 = coord1_loose - coord1_stem
        v2 = coord2_loose - coord2_stem

        # Calculate convex hull area
        points = np.vstack([coord1_loose, coord1_stem, coord2_loose, coord2_stem])
        hull = geometry.MultiPoint(points).convex_hull
        area = hull.area

        # Calculate distance between loose ends
        dist = np.linalg.norm(coord1_loose - coord2_loose)

        # Calculate angle between vectors
        angle = calculate_vector_angle(v1, v2)

        # Calculate area to distance ratio
        ratio = area / dist if dist != 0 else np.inf

        # Apply angle threshold
        if angle < angle_threshold:
            ratio = np.inf

        best_ratio = min(best_ratio, ratio)

    return best_ratio <= area_to_distance_ratio_threshold


def build_branch_graph(summary: pd.DataFrame) -> nx.MultiGraph:
    """Build a NetworkX graph representing branch connectivity.

    Creates a graph where nodes represent skeleton junctions and edges represent
    branches, with edge attributes for branch ID and distance. Both halves of a loop
    between two junctions are kept as parallel edges.

    Args:
        summary: Branch summary DataFrame with node and distance information.

    Returns:
        NetworkX multigraph with branch connectivity information.
    """
    graph = nx.MultiGraph()

    for _, branch in summary.iterrows():
        graph.add_edge(
            branch["node_id_src"],
            branch["node_id_dst"],
            branch_id=branch["branch_id"],
            distance=branch["branch_distance"],
        )

    return graph


def _path_branch_ids(branch_graph: nx.MultiGraph, path: list[int]) -> list[int]:
    """Get the branch IDs along a node path.

    Of parallel branches, the shorter is used, with ties going to the lower branch ID.

    Args:
        branch_graph: Multigraph from ``build_branch_graph``.
        path: Node IDs from ``nx.shortest_path``.

    Returns:
        Branch IDs of the edges along the path.
    """
    branch_ids = []
    for u, v in zip(path[:-1], path[1:]):
        edges = branch_graph.get_edge_data(u, v).values()
        shortest = min(edges, key=lambda d: (d["distance"], d["branch_id"]))
        branch_ids.append(shortest["branch_id"])
    return branch_ids


def build_contrail_assignments(
    pairs: list[tuple[int, int]],
    type_1_branches: pd.DataFrame,
    other_branches: pd.DataFrame,
    branch_graph: nx.Graph,
) -> dict[int, list[int]]:
    """Build final contrail assignments from paired branches.

    Constructs contrail objects by combining paired Type I branches with Type II
    branches, connecting them via shortest paths in the skeleton graph.

    Args:
        pairs: List of paired branch indices (from type_1_branches DataFrame).
        type_1_branches: DataFrame of Type I branches (one free endpoint).
        other_branches: DataFrame of Type II branches (two connected endpoints).
        branch_graph: NetworkX graph of branch connectivity.

    Returns:
        Dictionary mapping contrail indices to lists of branch IDs.
    """
    contrails = {}

    # Assign paired branches
    for pair_idx, (i, j) in enumerate(pairs):
        branch_i = type_1_branches.iloc[i]
        branch_j = type_1_branches.iloc[j]

        # Find shortest path between stems
        path = nx.shortest_path(
            branch_graph,
            source=branch_i["stem_node_id"],
            target=branch_j["stem_node_id"],
        )

        # Collect all branch IDs in the path
        edge_ids = _path_branch_ids(branch_graph, path)

        contrails[pair_idx] = [
            branch_i["branch_id"],
            branch_j["branch_id"],
        ] + edge_ids

    return contrails


def build_node_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Analyze node degrees and coordinates.

    Computes the degree (number of connected branches) for each node in the skeleton
    and extracts node coordinates from branch endpoints.

    Args:
        summary: Branch summary DataFrame with node and coordinate information.

    Returns:
        DataFrame with node IDs as index and columns for degree and coordinates.
    """
    # Calculate node degrees
    all_nodes = pd.concat([summary["node_id_src"], summary["node_id_dst"]])
    node_degree = all_nodes.value_counts().reset_index()
    node_degree.columns = ["node_id", "degree"]

    # Extract node coordinates
    src_coords = summary[["node_id_src", "image_coord_src_0", "image_coord_src_1"]].rename(
        columns={
            "node_id_src": "node_id",
            "image_coord_src_0": "coord_0",
            "image_coord_src_1": "coord_1",
        }
    )

    dst_coords = summary[["node_id_dst", "image_coord_dst_0", "image_coord_dst_1"]].rename(
        columns={
            "node_id_dst": "node_id",
            "image_coord_dst_0": "coord_0",
            "image_coord_dst_1": "coord_1",
        }
    )

    node_coords = (
        pd.concat([src_coords, dst_coords]).drop_duplicates("node_id").set_index("node_id")
    )

    return node_degree.set_index("node_id").join(node_coords)


def classify_branches(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify branches into type 1 (loose ends) and others.

    Args:
        summary: Branch summary DataFrame with type and degree information.

    Returns:
        Tuple of (type_1_branches, other_branches) DataFrames.
    """
    # Separate branch types
    type_1 = summary[summary["branch_type"] == 1].copy()
    other = summary[summary["branch_type"] != 1].copy()

    type_1 = identify_loose_ends(type_1)

    return type_1, other


def connect_branches_via_path(
    branch_graph: nx.Graph,
    start_node: int,
    end_node: int,
) -> list[int]:
    """Find shortest path between nodes and return branch IDs along the path.

    Args:
        branch_graph: NetworkX graph of branch connectivity.
        start_node: Starting node ID.
        end_node: Ending node ID.

    Returns:
        List of branch IDs along the shortest path.
    """
    path = nx.shortest_path(
        branch_graph,
        source=start_node,
        target=end_node,
    )

    return _path_branch_ids(branch_graph, path)


def connect_unused_branches(
    contrails: dict[int, list[int]],
    type_1_branches: pd.DataFrame,
    other_branches: pd.DataFrame,
    branch_graph: nx.Graph,
    angle_threshold: float,
    area_to_distance_ratio_threshold: float,
) -> dict[int, list[int]]:
    """Connect unused branches to contrails using geometric criteria.

    Builds complete contrail chains from unused branches.

    Args:
        contrails: Current contrail assignments.
        type_1_branches: DataFrame of type 1 branches.
        other_branches: DataFrame of other branch types.
        branch_graph: NetworkX graph of branch connectivity.
        angle_threshold: Minimum angle in degrees for branch compatibility.
        area_to_distance_ratio_threshold: Maximum ratio for compatibility.

    Returns:
        Updated contrail assignments with unused branches connected.
    """
    used_branches = set()
    for branch_ids in contrails.values():
        used_branches.update(branch_ids)

    unused_type_1 = type_1_branches[~type_1_branches["branch_id"].isin(used_branches)].copy()
    unused_other = other_branches[~other_branches["branch_id"].isin(used_branches)].copy()

    graph_nodes = set(branch_graph.nodes)
    new_contrail_index = len(contrails)
    processed_type_1 = set()

    for type_1_idx, type_1_branch in unused_type_1.iterrows():
        if type_1_idx in processed_type_1:
            continue

        # Start building a contrail from this type 1 branch
        contrail_branches = {type_1_branch["branch_id"]}
        branch_queue = [type_1_branch]
        processed_other = set()

        # Iteratively expand the contrail by finding compatible branches
        while branch_queue:
            current_branch = branch_queue.pop(0)

            # Find all compatible "other" branches for the current branch
            compatible_branches = []

            for other_idx, other_branch in unused_other.iterrows():
                if other_idx in processed_other:
                    continue

                # Check if this other branch is geometrically compatible
                if are_branches_compatible(
                    current_branch,
                    other_branch,
                    angle_threshold,
                    area_to_distance_ratio_threshold,
                ):
                    compatible_branches.append((other_idx, other_branch))
                    processed_other.add(other_idx)

            # Add compatible branches to the contrail
            for _, other_branch in compatible_branches:
                add_branch_with_path_connection(
                    contrail_branches=contrail_branches,
                    branch=other_branch,
                    reference_branch=current_branch,
                    branch_graph=branch_graph,
                    graph_nodes=graph_nodes,
                    prefer_stem_start=True,
                    prefer_stem_end=False,
                )

                # Add this branch to the queue to potentially find more connections
                branch_queue.append(other_branch)

        # Also check for other unused type 1 branches that might connect to this contrail
        for other_type_1_idx, other_type_1_branch in unused_type_1.iterrows():
            if other_type_1_idx in processed_type_1 or other_type_1_idx == type_1_idx:
                continue

            # Check if this type 1 branch is compatible with any branch in our current contrail
            if are_branches_compatible(
                type_1_branch,
                other_type_1_branch,
                angle_threshold,
                area_to_distance_ratio_threshold,
            ):
                processed_type_1.add(other_type_1_idx)
                add_branch_with_path_connection(
                    contrail_branches=contrail_branches,
                    branch=other_type_1_branch,
                    reference_branch=type_1_branch,
                    branch_graph=branch_graph,
                    graph_nodes=graph_nodes,
                    prefer_stem_start=True,
                    prefer_stem_end=True,
                )

        # Create the contrail if we have branches
        if len(contrail_branches) >= 1:
            contrails[new_contrail_index] = list(contrail_branches)
            new_contrail_index += 1

        processed_type_1.add(type_1_idx)

    return contrails


def extract_branch_endpoints(
    branch: dict[str, Any],
    use_loose_end: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract endpoint coordinates from a branch.

    Args:
        branch: Branch data containing endpoint coordinates (pd.Series or dict-like).
        use_loose_end: If True and branch has loose end info, use loose/stem coordinates.
            Otherwise use src/dst coordinates.

    Returns:
        Tuple of (first_endpoint, second_endpoint) as (y, x) coordinate arrays.
    """
    if use_loose_end and "image_coord_loose_0" in branch:
        # Type 1 branch with loose end identified
        first = np.array([branch["image_coord_loose_1"], branch["image_coord_loose_0"]])
        second = np.array([branch["image_coord_stem_1"], branch["image_coord_stem_0"]])
    else:
        # Standard src/dst coordinates
        first = np.array([branch["image_coord_src_1"], branch["image_coord_src_0"]])
        second = np.array([branch["image_coord_dst_1"], branch["image_coord_dst_0"]])

    return first, second


def find_optimal_pairs(
    area_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    angle_matrix: np.ndarray,
    angle_threshold: float,
    area_to_distance_ratio_threshold: float,
) -> list[tuple[int, int]]:
    """Find optimal branch pairs using geometric criteria.

    Uses greedy algorithm to pair branches based on area-to-distance ratio,
    filtered by angular alignment threshold.

    Args:
        area_matrix: Matrix of convex hull areas.
        distance_matrix: Matrix of pairwise distances.
        angle_matrix: Matrix of angles between branch directions.
        angle_threshold: Minimum angle threshold for pairing.
        area_to_distance_ratio_threshold: Maximum ratio for pairing.

    Returns:
        List of optimal branch pairs as index tuples.
    """
    n = len(area_matrix)

    # Calculate ratio matrix with angle filtering
    ratio_matrix = np.divide(
        area_matrix,
        distance_matrix,
        out=np.full_like(area_matrix, np.inf),
        where=distance_matrix != 0,
    )

    # Filter by angle threshold
    ratio_matrix[angle_matrix < angle_threshold] = np.inf

    # Only consider lower triangle to avoid duplicates
    ratio_matrix[np.triu_indices(n)] = np.inf

    # Greedy pairing
    pairs = []
    used = np.zeros(n, dtype=bool)

    while np.min(ratio_matrix) <= area_to_distance_ratio_threshold and not np.all(used):
        i, j = np.unravel_index(np.argmin(ratio_matrix), ratio_matrix.shape)

        if not used[i] and not used[j]:
            pairs.append((i, j))
            used[i] = used[j] = True

        # Remove this pair from consideration
        ratio_matrix[i, :] = ratio_matrix[:, i] = np.inf
        ratio_matrix[j, :] = ratio_matrix[:, j] = np.inf

    return pairs


def get_node_id_from_branch(
    branch: pd.Series,
    prefer_stem: bool = True,
) -> int:
    """Extract the appropriate node ID from a branch.

    Args:
        branch: Branch data.
        prefer_stem: If True, prefer stem node for Type I branches, otherwise use src node.

    Returns:
        The node ID to use for connections.
    """
    if prefer_stem and "stem_node_id" in branch:
        return branch["stem_node_id"]
    else:
        return branch["node_id_src"]


def identify_loose_ends(type_1_branches: pd.DataFrame) -> pd.DataFrame:
    """Identify which end of each type 1 branch is the loose end.

    Args:
        type_1_branches: DataFrame of type 1 branches.

    Returns:
        Updated DataFrame with loose end information.
    """
    type_1_branches["src_loose"] = (
        type_1_branches["src_degree"] < type_1_branches["dst_degree"]
    ).astype(int)

    # Set loose end coordinates
    type_1_branches["image_coord_loose_0"] = np.where(
        type_1_branches["src_loose"],
        type_1_branches["image_coord_src_0"],
        type_1_branches["image_coord_dst_0"],
    )
    type_1_branches["image_coord_loose_1"] = np.where(
        type_1_branches["src_loose"],
        type_1_branches["image_coord_src_1"],
        type_1_branches["image_coord_dst_1"],
    )

    # Set stem (connected) end coordinates
    type_1_branches["image_coord_stem_0"] = np.where(
        type_1_branches["src_loose"],
        type_1_branches["image_coord_dst_0"],
        type_1_branches["image_coord_src_0"],
    )
    type_1_branches["image_coord_stem_1"] = np.where(
        type_1_branches["src_loose"],
        type_1_branches["image_coord_dst_1"],
        type_1_branches["image_coord_src_1"],
    )

    type_1_branches["stem_node_id"] = np.where(
        type_1_branches["src_loose"],
        type_1_branches["node_id_dst"],
        type_1_branches["node_id_src"],
    )

    return type_1_branches


def pair_branches(
    type_1_branches: pd.DataFrame,
    angle_threshold: float,
    area_to_distance_ratio_threshold: float,
) -> list[tuple[int, int]]:
    """Find optimal pairs of Type I branches that likely belong to the same contrail.

    Computes geometric criteria (distances, angles, convex hull areas) between all
    pairs of Type I branches and uses greedy algorithm to find optimal pairings.

    Args:
        type_1_branches: DataFrame of Type I branches with loose end information.
        angle_threshold: Minimum angle threshold for pairing.
        area_to_distance_ratio_threshold: Maximum ratio for pairing.

    Returns:
        List of optimal branch pairs as index tuples.
    """
    if len(type_1_branches) < 2:
        return []

    # Extract coordinates
    loose_coords = type_1_branches[["image_coord_loose_1", "image_coord_loose_0"]].values
    stem_coords = type_1_branches[["image_coord_stem_1", "image_coord_stem_0"]].values

    # Calculate pairing criteria
    distance_matrix = calculate_distance_matrix(loose_coords)
    area_matrix = calculate_convex_hull_areas(loose_coords, stem_coords)
    angle_matrix = calculate_angle_matrix(loose_coords, stem_coords)

    # Find optimal pairs
    return find_optimal_pairs(
        area_matrix,
        distance_matrix,
        angle_matrix,
        angle_threshold,
        area_to_distance_ratio_threshold,
    )


def prune_short_type_I_branches(
    skan_skeleton: Skeleton,
    summary: pd.DataFrame,
    min_length: float,
) -> tuple[Skeleton, pd.DataFrame]:
    """Remove type I branches shorter than minimum length.

    Args:
        skan_skeleton: Skan skeleton object.
        summary: Summary DataFrame of skeleton branches.
        min_length: Minimum branch length threshold.

    Returns:
        Tuple of (pruned_skeleton, updated_summary).

    Raises:
        ValueError: If every branch is a short type I branch, which would leave no
            skeleton. The caller keeps such a region whole.
    """
    prune_ids = summary[
        (summary["branch_distance"] <= min_length) & (summary["branch_type"] == 1)
    ].index.tolist()

    # skan cannot build a skeleton with no paths, and fails with an opaque scipy error
    if len(prune_ids) == len(summary):
        raise ValueError(
            f"All {len(summary)} skeleton branches are type I branches no longer than "
            f"min_type_1_branch_length ({min_length}); pruning would leave no skeleton."
        )

    if prune_ids:
        skan_skeleton = skan_skeleton.prune_paths(prune_ids)
        summary = summarize(skan_skeleton, separator="_")

    return skan_skeleton, summary


def split_single_branch(
    branch_coordinates: dict[int, np.ndarray],
    summary: pd.DataFrame,
    skeleton: np.ndarray,
    elbow_threshold: float,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Split a single branch at the point of maximum deviation from direct line.

    Args:
        branch_coordinates: Dictionary of branch coordinates.
        summary: Summary DataFrame with branch information.
        skeleton: Original skeleton array.
        elbow_threshold: Threshold for elbow detection (normalized deviation).

    Returns:
        Tuple of (skeletons, types) where:
            skeletons: Dictionary with split skeleton segments.
            types: Dictionary mapping segment ID to branch type.
    """
    branch = summary.iloc[0]
    branch_id = summary.index[0]

    # Get coordinates and find elbow point
    coords = branch_coordinates[branch_id]
    if len(coords) < 3:  # Need at least 3 points to find an elbow
        return {0: skeleton}, {0: 0}

    # Extract endpoints
    src, dst = extract_branch_endpoints(branch, use_loose_end=False)
    elbow_idx, normalized_deviation = find_elbow_point(src, dst, coords)

    if normalized_deviation < elbow_threshold:
        return {0: skeleton}, {0: 0}

    # Create two skeleton segments
    skeleton_1, skeleton_2 = create_skeleton_segments(coords, elbow_idx, skeleton)

    return {0: skeleton_1, 1: skeleton_2}, {0: 1, 1: 1}


def split_type_1_branches_at_elbows(
    summary: pd.DataFrame,
    branch_coordinates: dict[int, np.ndarray],
    elbow_threshold: float,
    min_type_1_branch_length: float,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Split type 1 branches at elbow points if significant elbows are detected.

    Args:
        summary: Branch summary DataFrame.
        branch_coordinates: Dictionary of branch coordinates.
        elbow_threshold: Threshold for elbow detection (normalized deviation).
        min_type_1_branch_length: Minimum length for type 1 branches after split.

    Returns:
        Tuple of (updated_summary, updated_coordinates).
    """
    # Get type 1 branches
    type_1_branches = summary[summary["branch_type"] == 1].copy()

    if len(type_1_branches) == 0:
        return summary, branch_coordinates

    new_summary_rows = []
    new_branch_coordinates = branch_coordinates.copy()
    branches_to_remove = []

    # Get the next available node ID and branch ID
    if len(summary) == 0:
        return summary, branch_coordinates

    max_node_id = max(summary["node_id_src"].max(), summary["node_id_dst"].max())
    max_branch_id = summary.index.max() if len(summary) > 0 else 0
    next_node_id = max_node_id + 1
    next_branch_id = max_branch_id + 1

    for branch_id, branch in type_1_branches.iterrows():
        if branch_id not in branch_coordinates:
            continue

        coords = branch_coordinates[branch_id]

        if len(coords) < 3:  # Need at least 3 points to find an elbow
            continue

        # Extract endpoints and find elbow
        src, dst = extract_branch_endpoints(branch, use_loose_end=False)
        elbow_idx, normalized_deviation = find_elbow_point(src, dst, coords)

        # Check if elbow is significant
        if normalized_deviation < elbow_threshold:
            continue

        # Create elbow coordinates
        elbow_coord = coords[elbow_idx]

        # Determine which end is the loose end (lower degree)
        src_degree = branch["src_degree"]
        dst_degree = branch["dst_degree"]

        # Calculate the potential segment lengths
        if src_degree < dst_degree:
            # Source is loose end
            type_1_coords = coords[: elbow_idx + 1]  # From start to elbow
            type_2_coords = coords[elbow_idx:]  # From elbow to end
        else:
            # Destination is loose end
            type_2_coords = coords[: elbow_idx + 1]  # From start to elbow
            type_1_coords = coords[elbow_idx:]  # From elbow to end

        # Check if the new type 1 segment would be too short
        type_1_length = len(type_1_coords) - 1
        if type_1_length < min_type_1_branch_length:
            continue

        # Create the segments
        if src_degree < dst_degree:
            # Create type 1 segment (loose end to elbow)
            type_1_segment = branch.copy()
            type_1_segment["branch_type"] = 1
            # Calculate Euclidean distance from start to elbow
            start_coord = type_1_coords[0]
            end_coord = type_1_coords[-1]
            type_1_segment["branch_distance"] = np.sqrt(
                (end_coord[0] - start_coord[0]) ** 2 + (end_coord[1] - start_coord[1]) ** 2
            )
            type_1_segment["image_coord_dst_0"] = elbow_coord[0]
            type_1_segment["image_coord_dst_1"] = elbow_coord[1]
            type_1_segment["node_id_dst"] = next_node_id
            type_1_segment["dst_degree"] = 2  # Will connect to type 2 segment

            # Create type 2 segment (elbow to connected end)
            type_2_segment = branch.copy()
            type_2_segment["branch_type"] = 2
            # Calculate Euclidean distance from elbow to end
            start_coord = type_2_coords[0]
            end_coord = type_2_coords[-1]
            type_2_segment["branch_distance"] = np.sqrt(
                (end_coord[0] - start_coord[0]) ** 2 + (end_coord[1] - start_coord[1]) ** 2
            )
            type_2_segment["image_coord_src_0"] = elbow_coord[0]
            type_2_segment["image_coord_src_1"] = elbow_coord[1]
            type_2_segment["node_id_src"] = next_node_id
            type_2_segment["src_degree"] = 2  # Will connect to type 1 segment

        else:
            # Create type 2 segment (connected end to elbow)
            type_2_segment = branch.copy()
            type_2_segment["branch_type"] = 2
            # Calculate Euclidean distance from start to elbow
            start_coord = type_2_coords[0]
            end_coord = type_2_coords[-1]
            type_2_segment["branch_distance"] = np.sqrt(
                (end_coord[0] - start_coord[0]) ** 2 + (end_coord[1] - start_coord[1]) ** 2
            )
            type_2_segment["image_coord_dst_0"] = elbow_coord[0]
            type_2_segment["image_coord_dst_1"] = elbow_coord[1]
            type_2_segment["node_id_dst"] = next_node_id
            type_2_segment["dst_degree"] = 2  # Will connect to type 1 segment

            # Create type 1 segment (elbow to loose end)
            type_1_segment = branch.copy()
            type_1_segment["branch_type"] = 1
            # Calculate Euclidean distance from elbow to end
            start_coord = type_1_coords[0]
            end_coord = type_1_coords[-1]
            type_1_segment["branch_distance"] = np.sqrt(
                (end_coord[0] - start_coord[0]) ** 2 + (end_coord[1] - start_coord[1]) ** 2
            )
            type_1_segment["image_coord_src_0"] = elbow_coord[0]
            type_1_segment["image_coord_src_1"] = elbow_coord[1]
            type_1_segment["node_id_src"] = next_node_id
            type_1_segment["src_degree"] = 2  # Will connect to type 2 segment

        # Add new segments to summary with new branch IDs
        type_1_segment.name = next_branch_id
        type_2_segment.name = next_branch_id + 1

        new_summary_rows.append(type_1_segment)
        new_summary_rows.append(type_2_segment)

        # Add new coordinates
        new_branch_coordinates[next_branch_id] = type_1_coords
        new_branch_coordinates[next_branch_id + 1] = type_2_coords

        # Mark original branch for removal
        branches_to_remove.append(branch_id)

        # Update counters
        next_node_id += 1
        next_branch_id += 2

    # Remove original branches that were split
    if branches_to_remove:
        summary_filtered = summary.drop(branches_to_remove)
        for branch_id in branches_to_remove:
            if branch_id in new_branch_coordinates:
                del new_branch_coordinates[branch_id]

        # Add new segments
        if new_summary_rows:
            new_summary_df = pd.DataFrame(new_summary_rows)
            updated_summary = pd.concat([summary_filtered, new_summary_df], ignore_index=False)
        else:
            updated_summary = summary_filtered
    else:
        updated_summary = summary

    updated_summary["node_id_src"] = updated_summary["node_id_src"].astype(int)
    updated_summary["node_id_dst"] = updated_summary["node_id_dst"].astype(int)
    return updated_summary, new_branch_coordinates
