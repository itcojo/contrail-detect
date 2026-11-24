"""
Contrail detection based on FCI (Flexible Combined Imager) data.

"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self

import numpy as np
import pandas as pd

from contrail_detect.geostationary.abc import Geostationary

if TYPE_CHECKING:
    from contrail_detect.tracking.tracking import TrackingConfig


class FCIDetector(Geostationary):
    """Contrail detection using Flexible Combined Imager (FCI) data.

    Base class for satellites equipped with FCI instruments, primarily the
    Meteosat Third Generation (MTG) series. Extends the Geostationary class
    with FCI-specific functionality.

    Attributes:
        df_contrail: DataFrame containing detected contrail objects.
    """

    def __init__(self: Self) -> None:
        """Initialize the FCIDetector class."""
        super().__init__()

    def get_detections(
        self: Self,
        method: str,
        force_reprocess: bool = False,
        zarr_path: str | Path | None = None,
        detection_output_dir: str | Path | None = None,
    ) -> dict[np.datetime64, np.ndarray]:
        """
        Get contrail detection masks, either by loading from file or requesting them
        from the Google Contrails API.

        Loading from file is always attempted first unless 'force_reprocess' is enabled.

        Args:
            method: How contrail detection masks should be sourced. Options:
                - 'load': Load existing detections from zarr or .npy files
                - 'google_api': Request detections from the Google Contrails REST API
            force_reprocess: Whether to ignore data loading and process all timestamps.
            zarr_path: Path to zarr store. If None, uses default location.
            detection_output_dir: Directory for storing/loading detection outputs.

        Returns:
            Dictionary mapping timestamps to detection arrays.

        Examples:
            >>> detector.get_detections(method='load')
            >>> detector.get_detections(method='google_api')

        Raises:
            ValueError: If method is not one of the valid options.
        """
        if method not in ["load", "google_api"]:
            raise ValueError(f"Invalid method '{method}'. Must be one of: 'load', 'google_api'.")

        # Detections are saved under their source's subdirectory, and the Google API is the
        # only source for MTG, so 'load' reads from it too
        self.detection_subdirectory = "google_api"

        if method == "load":
            # Models write to <detection_output_dir>/<detection_subdirectory>; the default already has it
            if detection_output_dir is not None:
                detection_output_dir = Path(detection_output_dir) / self.detection_subdirectory

            # Load existing detections from file
            detections = self.load_detections(
                zarr_path=zarr_path,
                detection_output_dir=detection_output_dir,
            )

            return detections

        elif method == "google_api":
            from contrail_detect.models.google_api import GoogleAPIContrailModel

            api_model = GoogleAPIContrailModel(
                detector=self,
                detection_output_dir=detection_output_dir,
            )
            detections = api_model.run_detections(
                force_reprocess=force_reprocess,
                zarr_path=zarr_path,
            )

            return detections

    def get_decomposed_contrails(
        self: Self,
        method: str,
        contrail_output_dir: str | Path | None = None,
        force_reprocess: bool = False,
        read_simple: bool = False,
        prediction_threshold: float | None = None,
        min_area: int = 10,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
        """Get contrail detection line segments, either by loading from cache or generating.

        Each method first attempts to load cached results (unless force_reprocess=True),
        then generates new results if needed.

        Args:
            method: How contrail line segments should be sourced. Options:

                - 'decomposer': Generate line segments using the decomposer module
                - 'google_api': Request line segments from the Google Contrails REST API

            contrail_output_dir: Directory to save/read line segment files. If None,
                uses default location.
            force_reprocess: If True, skip loading cached results and regenerate.
                Default is False (load from cache if available).
            read_simple: If True, read simplified version of decomposition output without
                point cloud objects. Only applicable when loading cached results.
            prediction_threshold: Detection-score threshold for contrail pixels. Only applicable
                when method is 'decomposer'. If None, uses the default for the detection
                source: 0.30 for the local model and Contrails.org API, 0.40 for the Google API.
            min_area: Minimum area (in pixels) for detected regions to be retained. Only
                applicable when method is 'decomposer'. Defaults to 10.
            **kwargs: Additional DecompositionConfig parameters (only applicable when
                method is 'decomposer'), including:

                - contrail_stitching: Whether to rejoin the Type I/II branches split from one
                  detected region. Defaults to True.
                - min_type_1_branch_length: Minimum branch length for type 1 branches.
                  Defaults to 5.
                - angle_threshold: Angle threshold in degrees for joining vectors.
                  Defaults to 135.0.
                - elbow_threshold: Threshold for detecting elbows in branch skeletons.
                  Defaults to 0.4.
                - dilate_radius: Radius for mask dilation. If None, uses sqrt of
                  skeletonization loss. Defaults to None.
                - area_to_distance_ratio_threshold: Maximum ratio for branch pairing.
                  Defaults to 5.0.

        Returns:
            Tuple containing:

            - DataFrame of contrail line segments.
            - Dictionary mapping contrail IDs to (N, 2) arrays of [row, col] global
              image index coordinates, or None when read_simple=True or not available.
        """
        if method not in ["decomposer", "google_api"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: 'decomposer', 'google_api'."
            )

        if method == "decomposer":
            prediction_threshold = self._resolve_prediction_threshold(prediction_threshold)

            # Part of the results folder name; see _decomposition_fingerprint
            self.decomposition_settings = {
                "min_area": min_area,
                **{k: v for k, v in kwargs.items() if k != "contrail_stitching"},
            }

            # Try to load cached results first unless force_reprocess
            if not force_reprocess:
                try:
                    segment_outputs = self.load_decomposed_contrails(
                        contrail_output_dir=contrail_output_dir,
                        source_method="decomposer",
                        read_simple=read_simple,
                        contrail_stitching=kwargs.get("contrail_stitching", True),
                        prediction_threshold=prediction_threshold,
                    )
                    logging.info("Loaded decomposed contrails from cache.")
                    return segment_outputs[0], segment_outputs[1]
                except FileNotFoundError:
                    logging.info("No cached decomposition found, generating...")

            # Generate new decomposition
            from contrail_detect.decomposition.decompose import Decomposer

            # Create decomposer model instance and generate line segments
            decomposer_model = Decomposer(
                detector=self,
                contrail_output_dir=contrail_output_dir,
                prediction_threshold=prediction_threshold,
                min_area=min_area,
                **kwargs,
            )

            # Generate line segments and point cloud objects
            segment_outputs = decomposer_model.gen_decomposed_regions(
                force_regeneration=force_reprocess,
                read_simple=read_simple,
            )

            return segment_outputs[0], segment_outputs[1]

        elif method == "google_api":
            if not force_reprocess:
                try:
                    line_segments, point_clouds = self.load_decomposed_contrails(
                        contrail_output_dir=contrail_output_dir,
                        source_method="google_api",
                        read_simple=True,
                    )
                    logging.info("Loaded Google API line segments from cache.")
                    return line_segments, point_clouds

                except FileNotFoundError:
                    logging.info("No cached Google API line segments found, fetching...")

            from contrail_detect.models.google_api import GoogleAPIContrailModel

            api_model = GoogleAPIContrailModel(detector=self)
            line_segments = api_model.run_segments()

            line_segments_path = self._get_line_segments_path(
                contrail_output_dir,
                method="google_api",
            )
            line_segments.to_parquet(line_segments_path, index=False)

            logging.info(f"Saved Google API line segments to {line_segments_path}.")

            return line_segments, None

    def get_tracking_output(
        self: Self,
        force_reprocess: bool = False,
        tracking_config: Optional["TrackingConfig"] = None,
        use_wind_initialization: bool | None = None,
        estimated_contrail_height_ft: float | np.ndarray = 3.5e4,
    ) -> NotImplementedError:
        """Get contrail tracking outputs, either by loading from file or running tracking.

        Args:
            force_reprocess: Whether to ignore data loading and run tracking anew.
            tracking_config: Configuration object for contrail tracking. If None,
                uses default TrackingConfig settings.
            use_wind_initialization: Whether to start new tracks with ERA5 wind
                velocities. If None, the tracking configuration's setting is used.
            estimated_contrail_height_ft: Estimated contrail altitude in feet, used for
                windspeed initialization. Can be a single value or an array matching the
                number of contrail detections.

        Returns:
            New column 'track_id' added to contrail summary DataFrame.
        """
        raise NotImplementedError("Tracking functionality not yet implemented for FCIDetector.")
