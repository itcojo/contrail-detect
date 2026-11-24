"""
Contrail decomposition from detection masks in geostationary
satellite images.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from skan import Skeleton, summarize
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from contrail_detect.core.common import Colors, Transforms
from contrail_detect.decomposition.branch_operations import (
    add_unassigned_branches,
    build_branch_graph,
    build_contrail_assignments,
    build_node_summary,
    classify_branches,
    connect_unused_branches,
    pair_branches,
    prune_short_type_I_branches,
    split_single_branch,
    split_type_1_branches_at_elbows,
)
from contrail_detect.geostationary.abc import Geostationary

# Suppress excessive pyogrio logging
logging.getLogger("pyogrio").setLevel(logging.WARNING)

# Import standard values
colors, transforms = Colors(), Transforms()


@dataclass(frozen=True)
class DecompositionConfig:
    """Configuration for contrail decomposition parameters.

    Attributes:
        min_type_1_branch_length: Minimum Type I branch length in pixels. Shorter
            branches are generally washed out by dilation.
        angle_threshold: Minimum angle in degrees between two branch vectors for them
            to be joined. Closer to 180 is straighter.
        elbow_threshold: Minimum normalized deviation for an elbow to be significant.
        dilate_radius: Dilation radius in pixels. If None, uses the square root of the
            skeletonization loss.
        area_to_distance_ratio_threshold: Maximum ratio of convex hull area to distance
            for pairing branches.
        contrail_stitching: Whether to stitch Type I/II branches back together.
    """

    min_type_1_branch_length: int = 5
    angle_threshold: float = 135.0
    elbow_threshold: float = 0.4
    dilate_radius: float | None = None
    area_to_distance_ratio_threshold: float = 5.0
    contrail_stitching: bool = True

    def to_dict(self: Self) -> dict:
        """Return the config as a dictionary."""
        return asdict(self)


class ContrailDecomposer:
    """Decompose an array of region detection scores into individual contrails.

    Uses morphological skeletonization to analyze the structure of contrail regions and
    separates them into individual contrails based on branch topology and geometric
    criteria.

    Attributes:
        region_detection_array: 2D array of detection scores for one region, as float.
        config: Decomposition parameters.
    """

    def __init__(
        self: Self,
        region_detection_array: np.ndarray,
        config: DecompositionConfig | None = None,
    ) -> None:
        """Initialize the contrail decomposer with input validation.

        Args:
            region_detection_array: 2D array representing the region detection scores to decompose.
            config: Configuration parameters for decomposition behavior. If None, uses default
                DecompositionConfig().
        """
        # Input validation
        if not isinstance(region_detection_array, np.ndarray):
            raise ValueError("region_detection_array must be a numpy ndarray.")
        if region_detection_array.ndim != 2:
            raise ValueError("region_detection_array must be a 2D array.")
        self.region_detection_array = region_detection_array.astype(float)

        # Configuration setup
        self.config = config or DecompositionConfig()

        # Lazy initialization - computed only when needed
        self._skeleton_data = None

    def __repr__(self: Self) -> str:
        """Return string representation of the contrail decomposer.

        Returns:
            String showing the decomposer's region mask shape and configuration.
        """
        return (
            f"ContrailDecomposer(region_detection_array_shape={self.region_detection_array.shape}, "
            f"config={self.config})"
        )

    @cached_property
    def contrail_info(
        self: Self,
    ) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        """Get contrail skeletons and types with caching.

        Returns:
            Tuple of (skeletons, types), both empty if the skeleton has no branches.
        """
        if self.skeleton_data["summary"].empty:
            return {}, {}
        if self.is_single_branch:
            return split_single_branch(
                self.skeleton_data["branch_coordinates"],
                self.skeleton_data["summary"],
                self.skeleton_data["skeleton"],
                elbow_threshold=self.config.elbow_threshold,
            )
        else:
            return self._get_multi_branch_info()

    @property
    def contrail_skeletons(self: Self) -> dict[int, np.ndarray]:
        """Get contrail skeletons.

        Returns:
            Dictionary mapping contrail indices to skeleton arrays.
        """
        return self.contrail_info[0]

    @property
    def contrail_types(self: Self) -> dict[int, int]:
        """Get contrail types.

        Returns:
            Dictionary mapping contrail indices to contrail type.
        """
        return self.contrail_info[1]

    @cached_property
    def is_single_branch(self: Self) -> bool:
        """Check if the region contains only a single branch.

        Returns:
            True if the skeleton has only one branch, False otherwise.
        """
        return len(self.skeleton_data["summary"]) == 1

    @cached_property
    def skeleton_data(self: Self) -> dict[str, Any]:
        """Get skeleton analysis results with caching.

        Performs morphological skeletonization on the region mask and analyzes the
        resulting skeleton structure using the skan library.

        Returns:
            Dictionary containing skeleton analysis results including skeleton array,
            skan_skeleton object, branch summary DataFrame, and skeletonization_loss.
        """
        if self._skeleton_data is None:
            self._skeleton_data = self._analyze_skeleton()
        return self._skeleton_data

    def get_contrail_detection_arrays(
        self: Self,
        dilate_radius: float | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Generate binary masks and types for each contrail.

        Reconstructs full contrail masks from skeleton representations by applying
        morphological dilation. The dilation radius can be specified or automatically
        determined from skeletonization loss.

        Args:
            dilate_radius: Radius for mask dilation. If None, uses config value or
                sqrt of skeletonization loss.

        Returns:
            List of tuples (contrail_mask, contrail_type).
        """
        if dilate_radius is None:
            if self.config.dilate_radius is not None:
                dilate_radius = self.config.dilate_radius
            else:
                dilate_radius = np.sqrt(self.skeleton_data["skeletonization_loss"])

        # Extract contrail skeletons and types (cached properties)
        contrail_skeletons = self.contrail_skeletons
        contrail_types = self.contrail_types

        if not contrail_skeletons:
            logging.debug("No contrail skeletons found.")
            return []

        # Build aligned list of (mask, type) tuples using shared keys
        results = []
        for contrail_idx in sorted(contrail_skeletons.keys()):
            skeleton = contrail_skeletons[contrail_idx]
            contrail_type = contrail_types[contrail_idx]

            # Generate mask for this contrail
            skeleton_coords = np.column_stack(np.where(skeleton > 0))
            nonzero_mask = self.region_detection_array > 0
            region_coords = np.column_stack(np.where(nonzero_mask))
            region_values = self.region_detection_array[nonzero_mask]

            contrail_mask = np.zeros_like(
                self.region_detection_array, dtype=self.region_detection_array.dtype
            )

            if len(skeleton_coords) > 0 and len(region_coords) > 0:
                distances = cdist(
                    region_coords,
                    skeleton_coords,
                    metric="euclidean",
                )
                min_distances = np.min(distances, axis=1)
                close_indices = min_distances <= dilate_radius

                if np.any(close_indices):
                    close_coords = region_coords[close_indices]
                    close_vals = region_values[close_indices]
                    contrail_mask[close_coords[:, 0], close_coords[:, 1]] = close_vals

            results.append((contrail_mask, contrail_type))

        return results

    def _analyze_skeleton(self: Self) -> dict[str, Any]:
        """Perform initial skeleton analysis.

        Skeletonizes the region mask using morphological thinning, extracts branch
        information using the skan library, and prunes short Type I branches.

        Returns:
            Dictionary containing skeleton array, skan_skeleton object, branch summary
            DataFrame, branch coordinates, and skeletonization_loss metric.
        """
        skeleton = skeletonize(self.region_detection_array)
        # skan cannot build a skeleton graph from a single pixel, which small or
        # diagonal regions can thin to, so treat it like an empty skeleton
        if np.sum(skeleton) < 2:
            skeletonization_loss = float("inf")
            return {
                "skeleton": skeleton,
                "summary": pd.DataFrame(),
                "branch_coordinates": {},
                "skeletonization_loss": skeletonization_loss,
            }
        else:
            skeletonization_loss = np.sum(self.region_detection_array > 0) / np.sum(skeleton)

        skan_skeleton = Skeleton(skeleton)
        summary = summarize(skan_skeleton, separator="_")

        # Extract branch coordinates before pruning
        branch_coordinates = {}
        for branch_id in range(len(summary)):
            coords = skan_skeleton.path_coordinates(branch_id)
            branch_coordinates[branch_id] = coords

        # Prune short branches and update both summary and coordinates
        skan_skeleton, summary = prune_short_type_I_branches(
            skan_skeleton, summary, self.config.min_type_1_branch_length
        )

        # Update branch coordinates after pruning
        branch_coordinates = {}
        for branch_id in summary.index:
            coords = skan_skeleton.path_coordinates(branch_id)
            branch_coordinates[branch_id] = coords

        # Add node degree information for type 1 branch analysis
        if len(summary) > 0:
            node_summary = build_node_summary(summary)
            summary["src_degree"] = summary["node_id_src"].map(node_summary["degree"])
            summary["dst_degree"] = summary["node_id_dst"].map(node_summary["degree"])

            # Split type 1 branches at elbows
            summary, branch_coordinates = split_type_1_branches_at_elbows(
                summary,
                branch_coordinates,
                self.config.elbow_threshold,
                self.config.min_type_1_branch_length,
            )

            summary["branch_id"] = summary.index.astype(int)

        return {
            "skeleton": skeleton,
            "summary": summary,
            "branch_coordinates": branch_coordinates,
            "skeletonization_loss": skeletonization_loss,
        }

    def _build_contrail_assignments(
        self: Self,
        pairs: list[tuple[int, int]],
        type_1_branches: pd.DataFrame,
        other_branches: pd.DataFrame,
        branch_graph: nx.Graph,
    ) -> dict[int, list[int]]:
        """Build final contrail assignments from paired branches."""
        contrails = build_contrail_assignments(pairs, type_1_branches, other_branches, branch_graph)

        # Connect unused branches
        contrails = connect_unused_branches(
            contrails,
            type_1_branches,
            other_branches,
            branch_graph,
            self.config.angle_threshold,
            self.config.area_to_distance_ratio_threshold,
        )

        # Add remaining unassigned branches
        all_branch_ids = set(self.skeleton_data["summary"]["branch_id"])
        contrails = add_unassigned_branches(contrails, all_branch_ids)

        return contrails

    def _build_skeleton_arrays(
        self: Self, contrails: dict[int, list[int]]
    ) -> dict[int, np.ndarray]:
        """Build skeleton arrays for each contrail using stored coordinates.

        Combines branch coordinates for all branches assigned to each contrail into
        a single binary skeleton array.

        Args:
            contrails: Dictionary mapping contrail indices to branch ID lists.

        Returns:
            Dictionary mapping contrail indices to skeleton arrays.
        """
        skeleton_data = self.skeleton_data
        branch_coordinates = skeleton_data["branch_coordinates"]
        contrail_skeletons = {}

        for contrail_idx, branch_ids in contrails.items():
            contrail_skeleton = np.zeros_like(skeleton_data["skeleton"])

            for branch_id in branch_ids:
                if branch_id in branch_coordinates:
                    coords = branch_coordinates[branch_id]
                    if len(coords) > 0:
                        contrail_skeleton[coords[:, 0], coords[:, 1]] = 1

            contrail_skeletons[contrail_idx] = contrail_skeleton

        return contrail_skeletons

    def _determine_contrail_types(
        self: Self,
        contrails: dict[int, list[int]],
    ) -> dict[int, int]:
        """Determine the type of each contrail based on its constituent branches.

        Returns:
            Dictionary mapping contrail indices to type (0=Simple, 1=Type I, 2=Type II).
        """
        types = {}
        summary = self.skeleton_data["summary"]

        for contrail_idx, branch_ids in contrails.items():
            if len(branch_ids) > 1:
                # Composite / Stitched -> Type 2 (acting as a path)
                types[contrail_idx] = 2
            elif len(branch_ids) == 1:
                branch_id = branch_ids[0]
                branch_type = summary.loc[branch_id, "branch_type"]
                # Map skan branch_type to our types
                # skan: 0=isolated, 1=terminal, 2=path, 3=cycle
                if branch_type == 0:
                    types[contrail_idx] = 0
                elif branch_type == 1:
                    types[contrail_idx] = 1
                elif branch_type == 2:
                    types[contrail_idx] = 2
                else:
                    types[contrail_idx] = 2
            else:
                types[contrail_idx] = 0
        return types

    def _get_multi_branch_info(
        self: Self,
    ) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        """Process multi-branch case using graph-based approach.

        Returns:
            Tuple of (skeletons, types).
        """
        if not self.config.contrail_stitching:
            all_branch_ids = set(self.skeleton_data["summary"]["branch_id"])
            contrails = add_unassigned_branches({}, all_branch_ids)
            skeletons = self._build_skeleton_arrays(contrails)
            types = self._determine_contrail_types(contrails)
            return skeletons, types

        type_1_branches, other_branches = classify_branches(self.skeleton_data["summary"])
        pairs = pair_branches(
            type_1_branches,
            self.config.angle_threshold,
            self.config.area_to_distance_ratio_threshold,
        )
        branch_graph = build_branch_graph(self.skeleton_data["summary"])

        # Build contrail assignments
        contrails = self._build_contrail_assignments(
            pairs,
            type_1_branches,
            other_branches,
            branch_graph,
        )

        # Generate skeleton arrays
        skeletons = self._build_skeleton_arrays(contrails)
        types = self._determine_contrail_types(contrails)

        return skeletons, types


class Decomposer:
    """Contrail decomposition.

    Steps:
        1. Carries out simple region proposal for initial object identification.
        2. Characterises each proposed region for a complexity score.
        3. Decomposes complex regions into simpler sub-regions.

    All contrails will be output as one of three types:
        - Type N: Simple, isolated, no tethered ends.
        - Type I: Branch of decomposed complex contrail, one tethered end.
        - Type II: Branch of decomposed complex contrail, two tethered ends.

    Stitching decomposed Type I/II branches back together is optional, set by
    ``DecompositionConfig.contrail_stitching``. When stitching:

        - A branch can belong to more than one contrail, e.g. where two contrails cross.
        - Paired ends are joined by the path with the fewest branches, which is not
          always the shortest where the skeleton has loops.

    Attributes:
        detector: Geostationary detector whose detections are decomposed.
        prediction_threshold: Threshold for binarizing detection scores.
        min_area: Minimum region area in pixels.
        decomposition_config_kwargs: Keyword arguments for each region's
            ``DecompositionConfig``.
        all_regions: Summary table with one row per decomposed contrail.
        contrail_index_points: Image row/column point clouds keyed by contrail ID.
        output_dir: Directory for this run's outputs, by stitching and threshold.
        points_path: Parquet file of contrail point clouds.
        summary_path: Parquet file of the contrail summary table.
        zarr_path: Zarr archive of the detections used.
    """

    def __init__(
        self: Self,
        detector: Geostationary,
        contrail_output_dir: str | Path | None = None,
        prediction_threshold: float | None = None,
        min_area: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the decomposer.

        Args:
            detector: Instance of the geostationary satellite detector.
            contrail_output_dir: Directory to save/read contrail files.
            prediction_threshold: Threshold for generating binary masks from detection scores.
                If None, uses the detector's default for its detection source.
            min_area: Minimum area in pixels for a region to be considered.
            **kwargs: DecompositionConfig parameters including:

                - contrail_stitching: Whether to rejoin the Type I/II branches split from
                  one detected region. Defaults to True.
                - min_type_1_branch_length: Minimum branch length to avoid short type 1
                  branches. Defaults to 5.
                - angle_threshold: Angle threshold in degrees for joining vectors.
                  Defaults to 135.0.
                - elbow_threshold: Threshold for detecting elbows in branch skeletons.
                  Defaults to 0.4.
                - dilate_radius: Radius for mask dilation. If None, uses sqrt of
                  skeletonization loss. Defaults to None.
                - area_to_distance_ratio_threshold: Maximum ratio for branch pairing.
                  Defaults to 5.0.
        """
        # Store inputs and parameters
        self.detector: Geostationary = detector
        self.prediction_threshold: float = detector._resolve_prediction_threshold(
            prediction_threshold
        )
        self.min_area: int = min_area

        # Store DecompositionConfig kwargs for later use
        self.decomposition_config_kwargs: dict[str, Any] = kwargs

        # Used with the stitching and threshold to name the output folder
        self.detector.decomposition_settings = {
            "min_area": min_area,
            **{k: v for k, v in kwargs.items() if k != "contrail_stitching"},
        }

        # Contrail region summary dataframe
        self.all_regions: pd.DataFrame = pd.DataFrame()

        # Contrail index point clouds (image row/col coordinates)
        self.contrail_index_points: dict[str, np.ndarray] = {}

        self.output_dir = self.detector._processed_output_dir(
            contrail_output_dir,
            "decomposer",
            contrail_stitching=kwargs.get("contrail_stitching", True),
            prediction_threshold=self.prediction_threshold,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.points_path = self.output_dir / "contrail_points.parquet"
        self.summary_path = self.output_dir / "contrail_summary.parquet"
        self.zarr_path = self.output_dir / "detection_archive.zarr"

    def __repr__(self: Self) -> str:
        """Return string representation of the decomposer.

        Returns:
            String showing the decomposer's configuration and parameters.
        """
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.decomposition_config_kwargs.items())
        return (
            f"Decomposer(detector={self.detector.LONG_NAME}, "
            f"prediction_threshold={self.prediction_threshold}, "
            f"min_area={self.min_area}, "
            f"{kwargs_str})"
        )

    def gen_decomposed_regions(
        self: Self,
        force_regeneration: bool = False,
        read_simple: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Generate decomposed contrail regions from detection masks.

        Checks for existing region files; if not found, performs decomposition. Can
        optionally force regeneration of region files.

        Carries out the full decomposition process including region proposal,
        complexity scoring, and region refinement/decomposition.

        Args:
            force_regeneration: If True, regenerate files even if they already exist.
            read_simple: If True, read existing summary table data without reading
                point cloud data or data from the zarr store.

        Returns:
            Tuple containing:
                - contrail_summary: DataFrame with summary information for all contrails
                - contrail_index_points: Dictionary mapping contrail IDs to (N, 2) arrays of [row, col] global image index coordinates

            If no regions are proposed in any frame, both are returned empty
            (with a warning logged) rather than raising.

        """
        # Test to see if files already exist
        if not force_regeneration:
            if self.points_path.exists() and self.summary_path.exists():
                logging.info(f"Region files already exist at {self.output_dir}, reading...")
                self._read_contrail_files(read_simple=read_simple)

                logging.info(f"Read {len(self.all_regions)} regions from files.")

                # Prepare return values
                contrail_summary = self.all_regions.copy()
                columns_to_exclude = [
                    "region_detection_array",
                    "bbox",
                    "contrail_type",
                ]
                contrail_summary = contrail_summary.drop(
                    columns=[col for col in columns_to_exclude if col in contrail_summary.columns]
                )

                return contrail_summary, self.contrail_index_points

        # 1. Simple region proposal
        self._region_proposal()
        logging.info(f"Proposed {len(self.all_regions)} initial regions.")

        # No regions proposed in any frame: nothing to refine or summarise. Return
        # empty results rather than crashing downstream. This is valid when a scene
        # genuinely contains no contrails, but can also indicate that no detections
        # passed prediction_threshold/min_area, or that detection timestamps do not
        # match the detector's timestamp_sequence.
        if self.all_regions.empty:
            logging.warning(
                "No regions proposed across any frame; returning empty decomposition "
                "results. Verify this is expected (check prediction_threshold/min_area "
                "and that detection timestamps match the detector's timestamp_sequence)."
            )
            self.contrail_index_points = {}
            empty_summary = self.all_regions.drop(
                columns=["region_detection_array", "bbox"],
                errors="ignore",
            )
            return empty_summary, self.contrail_index_points

        # 2. Region refinement/decomposition
        self._region_refinement()
        logging.info(f"Regions decomposed into {len(self.all_regions)} regions.")

        # 3. Contrail properties and point clouds extraction
        self._contrails_from_regions()

        # 4. Generate summary table, point cloud
        contrail_summary, contrail_point_clouds = self._gen_contrail_files()
        logging.info(f"Generated summary table and point cloud files at {self.output_dir}.")

        return contrail_summary, contrail_point_clouds

    def _contrails_from_regions(self: Self) -> None:
        """Extract contrail properties and point clouds from all regions."""
        new_all_regions = []
        contrail_index_points = {}
        for _, region in self.all_regions.iterrows():
            try:
                phys_props, index_points = self._get_contrail_from_region(region=region)
            except Exception as e:
                logging.error(
                    f"Error processing region ID {region.get('region_id', 'unknown')}: {e}"
                )
                raise e

            # Number repeated IDs (identical pixels or a hash collision)
            base_id = phys_props["contrail_id"]
            repeat = 1
            while phys_props["contrail_id"] in contrail_index_points:
                repeat += 1
                phys_props["contrail_id"] = f"{base_id}-{repeat}"

            # Full metadata for Parquet
            pixel_props = region.to_dict()
            new_region = {**pixel_props, **phys_props}
            new_all_regions.append(new_region)

            # Store index point cloud (N, 2) array of [row, col] global image coordinates
            contrail_index_points[phys_props["contrail_id"]] = index_points

        # Assign updated regions dataframe
        self.all_regions = pd.DataFrame(new_all_regions)

        # Assign contrail index point clouds
        self.contrail_index_points = contrail_index_points

    def _gen_contrail_files(self: Self) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Generate region summary and point cloud files.

        Generates index point cloud parquet and summary table files for all contrail
        regions after decomposition.

        Returns:
            Tuple containing:
                - contrail_summary: DataFrame with summary information for all contrails
                - contrail_index_points: Dictionary mapping contrail IDs to (N, 2) arrays
                    of [row, col] global image index coordinates

        """

        # Store contrail_summary
        contrail_summary = self.all_regions.copy()
        columns_to_exclude = [
            "region_detection_array",
            "bbox",
            "contrail_type",  # internal to decomposition
        ]
        contrail_summary = contrail_summary.drop(columns=columns_to_exclude)

        logging.info(
            f"Contrails in summary table: {len(contrail_summary)}, "
            f"Across {contrail_summary['timestamp'].nunique()} unique timestamps."
        )
        contrail_summary.to_parquet(self.summary_path, index=False)

        # Build long-format points DataFrame and save as parquet
        # contrail_id stored as categorical to avoid repeating the string per pixel
        rows_list = []
        for contrail_id, points in self.contrail_index_points.items():
            n = len(points)
            rows_list.append(
                pd.DataFrame(
                    {
                        "contrail_id": np.full(n, contrail_id),
                        "row": points[:, 0].astype(np.int32),
                        "col": points[:, 1].astype(np.int32),
                    }
                )
            )
        points_df = pd.concat(rows_list, ignore_index=True)
        points_df["contrail_id"] = pd.Categorical(points_df["contrail_id"])
        points_df.to_parquet(self.points_path, index=False)

        return contrail_summary, self.contrail_index_points

    def _generate_contrail_id(
        self: Self,
        timestamp: pd.Timestamp,
        centroid_lat: float,
        centroid_lon: float,
        points: np.ndarray,
    ) -> str:
        """Generate a reproducible contrail ID.

        The last six characters hash the satellite, region, timestamp and pixel positions.

        Args:
            timestamp: Timestamp of the contrail detection.
            centroid_lat: Latitude of the contrail centroid.
            centroid_lon: Longitude of the contrail centroid.
            points: (N, 2) [row, col] image indices of the contrail's pixels.

        Returns:
            Contrail ID string.
        """
        # Format timestamp as YYYYMMDD-HHMM
        time_str = timestamp.strftime("%Y%m%d-%H%M")

        # Format lat/lon with hemisphere indicators
        lat_str = f"N{abs(centroid_lat):.1f}" if centroid_lat >= 0 else f"S{abs(centroid_lat):.1f}"
        lon_str = f"E{abs(centroid_lon):.1f}" if centroid_lon >= 0 else f"W{abs(centroid_lon):.1f}"

        digest = hashlib.sha256(
            f"{self.detector.satellite}|{self.detector.region}|{timestamp.isoformat()}|".encode()
        )
        digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
        hash_code = digest.hexdigest()[:6]

        # Create unique ID
        return f"{self.detector.satellite}-{self.detector.region}-{time_str}-{lat_str}-{lon_str}-{hash_code}"

    def _get_contrail_from_region(
        self: Self,
        region: pd.Series,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Get contrail property summary and index point cloud from a region.

        Args:
            region: Series containing region data including detection array and bbox.

        Returns:
            Tuple of (contrail properties dictionary, (N, 2) array of [row, col]
            global image index coordinates).

        """
        # Extract binary region mask and bounding box
        region_mask = np.where(region["region_detection_array"] > 0, 1, 0)
        bbox = region["bbox"]  # (min_row, min_col, max_row, max_col)

        # Detection scores in region
        # Only calculate statistics on non-zero pixels (actual contrail pixels)
        nonzero_pixels = region["region_detection_array"][region_mask == 1]
        mean_score = np.mean(nonzero_pixels) if len(nonzero_pixels) > 0 else 0
        max_score = np.max(region["region_detection_array"])
        min_score = np.min(nonzero_pixels) if len(nonzero_pixels) > 0 else 0
        std_score = np.std(nonzero_pixels) if len(nonzero_pixels) > 0 else 0

        # Get pixel coordinates of contrail in region
        contrail_points = np.argwhere(region_mask == 1)

        # Add bbox offsets to get indices into the detector's lat/lon grid
        contrails_points_global = contrail_points + np.array([[bbox[0], bbox[1]]])

        # Convert to lat/lon
        contrail_lats = self.detector.lats[
            contrails_points_global[:, 0], contrails_points_global[:, 1]
        ]
        contrail_lons = self.detector.lons[
            contrails_points_global[:, 0], contrails_points_global[:, 1]
        ]

        if any(np.isnan(contrail_lats)) or any(np.isnan(contrail_lons)):
            raise ValueError(
                f"NaN values found in contrail lat/lon coordinates: \n"
                f" - region bbox: {bbox}\n"
                f"This should not happen - regions should not be created in NaN mask areas."
            )

        # Unwrap longitudes across the date line relative to the first pixel
        crosses_date_line = np.ptp(contrail_lons) > 180.0
        if crosses_date_line:
            contrail_lons = contrail_lons[0] + (
                (contrail_lons - contrail_lons[0] + 180.0) % 360.0 - 180.0
            )

        # Calculate centroid
        centroid_lat = np.mean(contrail_lats)
        centroid_lon = np.mean(contrail_lons)

        # Number of pixels in contrail
        num_points = len(contrails_points_global)

        # Physical geometric properties
        endpoint1, endpoint2 = self._get_contrail_endpoints(
            contrail_lons,
            contrail_lats,
        )
        if crosses_date_line:
            centroid_lon = (centroid_lon + 180.0) % 360.0 - 180.0
            endpoint1 = ((endpoint1[0] + 180.0) % 360.0 - 180.0, endpoint1[1])
            endpoint2 = ((endpoint2[0] + 180.0) % 360.0 - 180.0, endpoint2[1])
        forward_azimuth, _, great_circle_distance = transforms.geod.inv(
            lons1=endpoint1[0],
            lats1=endpoint1[1],
            lons2=endpoint2[0],
            lats2=endpoint2[1],
        )
        great_circle_distance /= 1000.0  # Convert to km

        # Area
        areas = self.detector.pixel_surface_area[
            contrails_points_global[:, 0],
            contrails_points_global[:, 1],
        ]
        total_area = np.sum(areas)

        # Get timestamp for this region
        timestamp = pd.Timestamp(region["timestamp"])

        # Calculate scantime offsets
        scan_offset_centroid = self.detector.scan_time_offset_for_position(
            centroid_lat,
            centroid_lon,
            timestamp,
        )
        scantime_centroid = timestamp + pd.Timedelta(scan_offset_centroid)

        scan_offset_end = self.detector.scan_time_offset_for_position(
            endpoint2[1],
            endpoint2[0],
            timestamp,
        )
        scantime_end = timestamp + pd.Timedelta(scan_offset_end)

        scan_offset_start = self.detector.scan_time_offset_for_position(
            endpoint1[1],
            endpoint1[0],
            timestamp,
        )
        scantime_start = timestamp + pd.Timedelta(scan_offset_start)

        # Create unique ID
        contrail_id = self._generate_contrail_id(
            timestamp, centroid_lat, centroid_lon, contrails_points_global
        )

        return {
            "contrail_id": contrail_id,
            "timestamp": timestamp,
            "n_pixels": num_points,
            "centroid_lat": np.round(centroid_lat, 4),
            "centroid_lon": np.round(centroid_lon, 4),
            "centroid_scantime": scantime_centroid,
            "endpoint1_lat": np.round(endpoint1[1], 4),
            "endpoint1_lon": np.round(endpoint1[0], 4),
            "endpoint1_scantime": scantime_start,
            "endpoint2_lat": np.round(endpoint2[1], 4),
            "endpoint2_lon": np.round(endpoint2[0], 4),
            "endpoint2_scantime": scantime_end,
            "orientation": np.round(forward_azimuth, 2),
            "great_circle_length_km": np.round(great_circle_distance, 2),
            "area_km2": np.round(total_area, 2),
            "width_km": (
                np.round(total_area / great_circle_distance, 2) if great_circle_distance > 0 else 0
            ),
            "mean_score": mean_score,
            "min_score": min_score,
            "max_score": max_score,
            "std_score": std_score,
        }, contrails_points_global

    def _read_contrail_files(
        self: Self,
        read_simple: bool = False,
    ) -> None:
        """Read region summary and point cloud files.

        Reads the Parquet point cloud and summary table files for all contrail regions
        after decomposition.

        Args:
            read_simple: If True, read existing summary table data without reading
                point cloud data or data from the zarr store.

        """
        # Use detections from the detector instead of reloading from zarr
        # (avoids coupling between detection storage and decomposition paths)
        if not self.detector.detections:
            raise ValueError(
                "No detections found in detector. Assign them first with "
                "`detector.detections = detector.get_detections(...)`."
            )

        # Read the summary file
        all_regions = pd.read_parquet(self.summary_path)

        # Convert timestamp strings back to datetime objects
        for col in [
            "timestamp",
            "centroid_scantime",
            "endpoint1_scantime",
            "endpoint2_scantime",
        ]:
            all_regions[col] = pd.to_datetime(
                all_regions[col],
                format="ISO8601",
            ).dt.floor("s")

        # If only simple read is requested, skip point cloud and zarr reading
        if read_simple:
            logging.warning(
                "Performing simple read of region summary table only. "
                "Regions will not have region_detection_array or bbox data."
            )
            self.all_regions = all_regions
            return

        # Read the index point cloud parquet file
        points_df = pd.read_parquet(self.points_path)

        self.contrail_index_points = {
            cid: group[["row", "col"]].values
            for cid, group in points_df.groupby("contrail_id", observed=True)
        }

        # Reconstruct region_detection_array and bbox from detector's detections
        self.all_regions = self._reconstruct_regions(
            summary_data=all_regions,
            index_points=self.contrail_index_points,
        )

    def _reconstruct_regions(
        self: Self,
        summary_data: pd.DataFrame,
        index_points: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Reconstruct region data using detections from the detector.

        This method uses the detections already loaded in self.detector.detections
        instead of reloading from zarr, avoiding path coupling issues.

        Args:
            summary_data: DataFrame with summary information for all contrails.
            index_points: Dictionary mapping contrail IDs to (N, 2) arrays of
                [row, col] global image index coordinates.

        Returns:
            Updated summary_data with bbox and region_detection_array columns.
        """
        # Initialize columns for bbox, region_mask
        if "bbox" not in summary_data.columns:
            summary_data["bbox"] = None
        if "region_detection_array" not in summary_data.columns:
            summary_data["region_detection_array"] = None

        # Initialize lists for batch assignment
        bbox_list = [None] * len(summary_data)
        region_detection_array_list = [None] * len(summary_data)

        # Group by timestamp to process frames
        for timestamp, df_summary_frame in summary_data.groupby("timestamp"):
            # Get detection mask for this timestamp from detector
            timestamp_np = np.datetime64(timestamp)
            if timestamp_np not in self.detector.detections:
                logging.warning(f"No detection found for timestamp {timestamp} in detector.")
                continue

            detection_mask = self.detector.detections[timestamp_np]

            for r_idx, row in df_summary_frame.iterrows():
                contrail_id = row["contrail_id"]
                points = index_points.get(contrail_id)
                if points is None:
                    logging.warning(f"No index points found for contrail_id {contrail_id}.")
                    continue

                pixel_rows = points[:, 0]
                pixel_cols = points[:, 1]

                # Extract bounding box directly from stored index coordinates
                minr, maxr = int(np.min(pixel_rows)), int(np.max(pixel_rows)) + 1
                minc, maxc = int(np.min(pixel_cols)), int(np.max(pixel_cols)) + 1
                bbox = (minr, minc, maxr, maxc)

                region_detection_array = detection_mask[minr:maxr, minc:maxc].copy()

                bbox_list[r_idx] = bbox
                region_detection_array_list[r_idx] = region_detection_array

        # Batch assignment
        summary_data["bbox"] = bbox_list
        summary_data["region_detection_array"] = region_detection_array_list

        return summary_data.reset_index(drop=True)

    def _region_proposal(self: Self) -> None:
        """Generate region proposals from the detection frames.

        Uses skimage's regionprops to identify and characterize regions in binary masks
        derived from detection scores. Region properties are extracted using the
        _region_features_extract method. Results are stored in the all_regions DataFrame.
        """
        if not self.detector.detections:
            raise ValueError(
                "No detections found in detector. Assign them first with "
                "`detector.detections = detector.get_detections(...)`."
            )

        # Initialize with the expected schema so an empty result (no regions in any
        # frame) still carries a "timestamp" column for downstream refinement.
        frame_regions = pd.DataFrame(
            columns=[
                "timestamp",
                "region_detection_array",
                "bbox",
                "solidity",
                "eccentricity",
            ]
        )
        for _, timestamp in enumerate(self.detector.timestamp_sequence):
            # Fetch contrail detection image for the timestamp
            if np.datetime64(timestamp) not in list(self.detector.detections.keys()):
                logging.warning(
                    f"No detection scores found for timestamp {timestamp}, skipping frame."
                )
                continue
            detection_scores = self.detector.detections[np.datetime64(timestamp)]

            # Extract regions from the prediction, excluding NaN values
            # NaN values are set for pixels outside the requested extent/satellite disk
            valid_mask = ~np.isnan(detection_scores)
            binary_mask = (detection_scores >= self.prediction_threshold) & valid_mask
            labeled = label(binary_mask)
            regions = regionprops(labeled)
            regions = [r for r in regions if r.area >= self.min_area]

            # Extract region features
            region_features = []
            for r in regions:
                r_features = {}
                r_features["timestamp"] = np.datetime64(timestamp)

                # Extract region mask and bounding box
                minr, minc, maxr, maxc = r.bbox
                region_mask = r.image  # Boolean mask of the region within the bbox
                region_detection_crop = detection_scores[minr:maxr, minc:maxc].copy()
                region_detection_crop[~region_mask] = 0  # Zero out non-region pixels

                # Descriptors of the region
                r_features["region_detection_array"] = region_detection_crop
                r_features["bbox"] = (minr, minc, maxr, maxc)

                # Pixel-space shape descriptors
                r_features["solidity"] = float(r.solidity)
                r_features["eccentricity"] = float(r.eccentricity)

                region_features.append(r_features)

            # Store frame-level region information
            frame_regions = pd.concat(
                [frame_regions, pd.DataFrame(region_features)],
                ignore_index=True,
            )

        # Store all region information
        frame_regions = frame_regions.reset_index(drop=True)
        self.all_regions = frame_regions

    def _region_refinement(self: Self) -> None:
        """Decompose complex regions into individual contrail objects.

        Uses ContrailDecomposer to separate regions into individual contrail objects
        of Type N/I/II. Results replace the corresponding frame entries in all_regions.

        """

        for _, timestamp in enumerate(self.detector.timestamp_sequence):
            # Extract regions for the current frame
            frame_regions = self.all_regions[
                self.all_regions["timestamp"] == np.datetime64(timestamp)
            ]

            new_regions = []
            for _, frame_region in frame_regions.iterrows():
                # Extract binary region mask
                region_detection_array = frame_region["region_detection_array"]

                # Get the original global bbox for coordinate transformation
                global_minr, global_minc, _, _ = frame_region["bbox"]

                try:
                    # Extract contrails from complex region
                    decomposition_config = DecompositionConfig(**self.decomposition_config_kwargs)
                    region_decompose = ContrailDecomposer(
                        region_detection_array,
                        config=decomposition_config,
                    )
                    contrail_results = region_decompose.get_contrail_detection_arrays()

                    # No valid contrails found - retain original (shouldn't really happen)
                    if len(contrail_results) == 0:
                        logging.debug(
                            f"No decomposition found for region at {timestamp}, bbox "
                            f"{frame_region['bbox']}, retaining original."
                        )
                        frame_region_dict = frame_region.to_dict()
                        frame_region_dict["originally_complex"] = False
                        frame_region_dict["contrail_type"] = np.nan
                        frame_region_dict["is_processed"] = False
                        new_regions.append(frame_region_dict)
                        continue

                    # One or more contrails found
                    for contrail_mask, contrail_type in contrail_results:
                        # Take region binary for decomposed contrail and propose new region
                        labeled_mask = label(contrail_mask > 0)
                        props = regionprops(labeled_mask, intensity_image=contrail_mask)
                        if len(props) == 0:
                            logging.warning(
                                f"Empty decomposed contrail found in region at {timestamp}, "
                                f"bbox {frame_region['bbox']}, skipping."
                            )
                            continue

                        # Extract features using pixel coordinates
                        r_features = {}
                        r_features["timestamp"] = np.datetime64(timestamp)

                        # Extract region mask and bounding box
                        minr, minc, maxr, maxc = props[0].bbox
                        region_mask = props[0].image  # Boolean mask of the region within the bbox
                        region_detection_crop = contrail_mask[minr:maxr, minc:maxc].copy()
                        region_detection_crop[~region_mask] = 0  # Zero out non-region pixels

                        # Descriptors of the region
                        r_features["region_detection_array"] = region_detection_crop
                        r_features["bbox"] = (
                            minr + global_minr,
                            minc + global_minc,
                            maxr + global_minr,
                            maxc + global_minc,
                        )

                        # Pixel-space shape descriptors
                        r_features["solidity"] = float(props[0].solidity)
                        r_features["eccentricity"] = float(props[0].eccentricity)

                        # Mark complexity based on whether split occurred
                        r_features["originally_complex"] = len(contrail_results) > 1
                        r_features["contrail_type"] = contrail_type
                        frame_region["is_processed"] = True

                        # Append new region features
                        new_regions.append(r_features)

                except Exception as e:
                    logging.warning(
                        f"Decomposition failed for region at {timestamp}, bbox "
                        f"{frame_region['bbox']}: {e}"
                    )
                    frame_region_dict = frame_region.to_dict()
                    frame_region_dict["originally_complex"] = False
                    frame_region_dict["contrail_type"] = np.nan
                    frame_region_dict["is_processed"] = False
                    new_regions.append(frame_region_dict)

            # Replace current frame in all_regions with new regions
            non_frame_regions = self.all_regions[
                self.all_regions["timestamp"] != np.datetime64(timestamp)
            ]
            if len(new_regions) > 0:
                self.all_regions = pd.concat(
                    [non_frame_regions, pd.DataFrame(new_regions)],
                    ignore_index=True,
                ).reset_index(drop=True)
            else:
                # No regions found in this frame, just keep the non-frame regions
                self.all_regions = non_frame_regions.reset_index(drop=True)
                logging.warning(f"No regions generated for timestamp {timestamp}.")

    def _validate_detector(self: Self) -> None:
        """Validate that the detector has all required attributes for decomposition operations.

        Raises:
            AttributeError: If detector is missing required attributes.
            ValueError: If detector attributes are not properly configured.
        """
        if not isinstance(self.detector, Geostationary):
            raise TypeError(
                f"Detector must be an instance of Geostationary, got {type(self.detector)} instead."
            )

        # Check required class attributes
        required_attrs = [
            "satellite",
            "region",
            "lons",
            "lats",
            "timestamp_sequence",
            "pixel_surface_area",
            "image_geo_coords",
            "detections",
        ]

        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(self.detector, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            raise AttributeError(
                f"Detector {self.detector.__class__.__name__} is missing required "
                f"attributes: {', '.join(missing_attrs)}"
            )

        # Check required methods
        required_methods = [
            "get_grid_index_from_position",
            "scan_time_offset_for_position",
        ]
        missing_methods = []
        for method in required_methods:
            if not hasattr(self.detector, method) or not callable(getattr(self.detector, method)):
                missing_methods.append(method)

        if missing_methods:
            raise AttributeError(
                f"Detector {self.detector.__class__.__name__} is missing required "
                f"methods: {', '.join(missing_methods)}"
            )

        # Check that critical attributes are not None
        if self.detector.timestamp_sequence is None or len(self.detector.timestamp_sequence) == 0:
            raise ValueError(
                "Detector timestamp_sequence is None or empty. "
                "Please ensure detector has loaded timestamp data."
            )

        if self.detector.lons is None or self.detector.lats is None:
            raise ValueError(
                "Detector geographic coordinates (lons/lats) are None. "
                "Please ensure detector has loaded coordinate data."
            )

        if self.detector.image_geo_coords is None:
            raise ValueError(
                "Detector image_geo_coords is None. "
                "Please ensure detector has loaded projection coordinates."
            )

        # Check that detections dictionary is not empty
        if self.detector.detections is None or len(self.detector.detections) == 0:
            raise ValueError(
                "Detector detections dictionary is None or empty. "
                "Please ensure detector has run detection/loaded detection data before decomposition."
            )

    @staticmethod
    def _get_contrail_endpoints(
        lons: np.ndarray,
        lats: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the endpoints of a contrail given its lon/lat point cloud.

        Args:
            lons: Array of longitudes of contrail points.
            lats: Array of latitudes of contrail points.
        Returns:
            Tuple containing two tuples of (lon, lat) for each endpoint.
        """
        coords = np.column_stack((lons, lats))
        distances = squareform(pdist(coords))
        max_dist_idx = np.unravel_index(distances.argmax(), distances.shape)
        start_point = coords[max_dist_idx[0]]
        end_point = coords[max_dist_idx[1]]
        return start_point, end_point
