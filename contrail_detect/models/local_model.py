"""
Contrail detection using ABI data and the local model from kaggle_contrails_solution.

"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import contrail_detect.models.kaggle_contrails_solution.input.models.unet1024 as unet1024
import contrail_detect.models.kaggle_contrails_solution.input.models.vit4 as vit4
from contrail_detect.core.common import (
    clean_pycontrails_cache,
    clean_temp_dir,
)
from contrail_detect.core.config import TEST_DIR, WEIGHTS_DIR
from contrail_detect.geostationary import geo_utils
from contrail_detect.geostationary.abi_base import mesoscale_geo_coords_path
from contrail_detect.geostationary.goes import ABIDetector
from contrail_detect.models.contrail_model_base import ContrailModelBase


class LocalContrailModel(ContrailModelBase):
    """Contrail detection on the local machine.

    Uses the U-Net and vision transformer ensemble from the winning Kaggle (Google
    Research contrail detection competition) solution.

    Implements the winning solution from the Google Research - Identify Contrails
    competition on Kaggle. Supports both ViT4 and UNet1024 architectures for
    detecting contrails in ABI satellite imagery.

    Attributes:
        WEIGHTS_PATH: Path to pre-trained model weights directory.
        detector: ABIDetector instance providing satellite data.
        device_ids: GPU device IDs or ``'CPU'`` for inference.
        unet1024: Whether to use UNet1024 architecture instead of default ViT4.
        timestamps_processed: List of timestamps with existing detections on disk.
    """

    WEIGHTS_PATH = WEIGHTS_DIR

    def __init__(
        self: Self,
        detector: ABIDetector,
        device_ids: list[int] | str,
        detection_output_dir: str | Path | None = None,
        *,
        unet1024: bool = False,
    ) -> None:
        """Initialize the LocalContrailModel for local detection.

        Args:
            detector: Satellite detector instance using ABI data (e.g., GOESDetector).
            device_ids: GPU device indices as list (e.g., [0, 1]) or ``'CPU'`` string.
            detection_output_dir: Directory for storing local model outputs. Defaults to detector's
                output path with 'local' subdirectory.
            unet1024: Whether to use UNet1024 model instead of default ViT4.
        """
        super().__init__(
            detector=detector,
            detection_output_dir=detection_output_dir,
            subdirectory="local",
        )

        self.contrail_model_name = "Local Contrail Model"

        # Store model-specific parameters
        self.device_ids: list[int] | str = device_ids
        self.unet1024: bool = unet1024

        # Validate GPU devices
        self._validate_device_ids()

    def __repr__(self: Self) -> str:
        """Return string representation of the LocalContrailModel instance.

        Returns:
            String representation including detector, devices, model type, and output directory.
        """
        return (
            f"{self.__class__.__name__}("
            f"detector={self.detector.__class__.__name__}, "
            f"device_ids={self.device_ids}, "
            f"unet1024={self.unet1024}, "
            f"detection_output_dir={self.detection_output_dir})"
        )

    def run_detections(
        self: Self,
        force_reprocess: bool = False,
        zarr_path: str | Path | None = None,
        dataset_attrs: dict | None = None,
        tiles_to_load: list[int] | None = None,
        n_jobs: int = 1,
        clean_caches: bool = False,
    ) -> dict[np.datetime64, np.ndarray]:
        """
        Executes the inference pipeline for all loaded timestamps, processing all
        the timestamps specified in 'kaggle_contrails_solution/data/test'.

        First checks for existing detections in zarr store. If zarr doesn't exist
        or is incomplete, checks for .npy files and automatically builds zarr store.
        If .npy files don't exist, runs the local model and builds zarr store.

        The test directory is expected to have folders named like:
        "<satellite>-<region>-<YYYY>-<MM>-<DD>T<HH><MM>-<tile_number>"
        (e.g., "GOES-16-F-2024-11-10T0130-020")

        Tile predictions are saved in the output directory under subdirectories named
        by detector metadata and timestamp:
        "<satellite>-<region>-<YYYY>-<MM>-<DD>T<HH><MM>/tiles_data.npz"

        Args:
            force_reprocess: Skip all checks and reprocess all timestamps.
            zarr_path: Optional path to zarr store. If None, uses default location.
            dataset_attrs: Optional dictionary of global attributes for zarr dataset.
            tiles_to_load: Subset of tile indices to stage and process. If None, all
                valid tiles are used. Passed to ``detector.load_data_for_local_model`` and
                ``_join_tiles``.
            n_jobs: Number of parallel joblib workers for tile staging. Passed through
                to ``detector.load_data_for_local_model``. See that method for memory implications.
            clean_caches: Also empty the detector's pycontrails satellite cache
                (e.g. ``~/.cache/pycontrails/goes``) before inference, to save disk
                space. The imagery is downloaded again when next needed.

        Returns:
            Dictionary mapping timestamps to detection arrays.
        """
        # Start populating detections dictionary with requested timestamps
        requested_timestamps = set(self.detector.timestamp_sequence)
        image_coords = self.detector.image_geo_coords
        output_detections = {}
        for ts in requested_timestamps:
            output_detections[ts] = np.full(
                (len(image_coords[1]), len(image_coords[0])),
                np.nan,
            )

        # Track which timestamps still need processing
        timestamps_found = set()

        if not force_reprocess:
            # Check which requested timestamps exist in zarr store
            logging.info("Checking for existing detections in zarr store...")

            # Initial test for zarr store existence
            zarr_store_path = self.detector.get_zarr_path(zarr_path)
            if not zarr_store_path.exists():
                logging.info(f"Zarr store not found at: {zarr_store_path}")

            else:
                # Load existing detections from zarr store
                loaded_detections = self.detector.load_detections_from_zarr(zarr_path=zarr_path)
                zarr_timestamps = set(loaded_detections.keys()) & requested_timestamps

                # Populate output detections with existing ones from zarr
                for ts in zarr_timestamps:
                    output_detections[ts] = loaded_detections[ts]
                    timestamps_found.add(ts)

                logging.info(
                    f"Found {len(zarr_timestamps)}/{len(requested_timestamps)} "
                    f"requested timestamps in zarr store."
                )

                # Return data if all requested timestamps exist in zarr
                if len(zarr_timestamps) == len(requested_timestamps):
                    logging.info("All requested timestamps found in zarr store. Returning.")
                    self.timestamps_processed = list(timestamps_found)
                    self.timestamps_to_process = []
                    return output_detections

            # Check for individual .npy detection files (for timestamps not yet found)
            timestamps_required = requested_timestamps - timestamps_found
            if timestamps_required:
                logging.info(
                    f"Checking for {len(timestamps_required)} existing detections as .npy files..."
                )

                # Check which .npy files exist and load them
                for time_str in self.detector.timestamp_strings:
                    ts = np.datetime64(pd.to_datetime(time_str))
                    if ts not in timestamps_required:
                        continue

                    req_fmt = f"{self.detector.satellite}-{self.detector.region}-{time_str}"
                    req_dir = self.detection_output_dir / req_fmt
                    prediction_file = req_dir / "_prediction.npy"

                    # The cache is keyed by time only, so check its extent before loading
                    if prediction_file.exists() and self._cached_extent_matches(req_dir):
                        detection_array = np.load(prediction_file)
                        output_detections[ts] = detection_array
                        timestamps_found.add(ts)

                if zarr_store_path.exists():
                    npy_found = len(timestamps_found) - len(zarr_timestamps)
                else:
                    npy_found = len(timestamps_found)

                logging.info(f"Found {npy_found} requested timestamps as .npy files.")
                logging.info(
                    f"Total requested timestamps found: {len(timestamps_found)}/{len(requested_timestamps)}"
                )

            # Determine what still needs to be processed
            self.timestamps_processed = list(timestamps_found)
            self.timestamps_to_process = list(requested_timestamps - timestamps_found)

            # If all timestamps found, build zarr store and return
            if not self.timestamps_to_process:
                logging.info(
                    f"All {len(requested_timestamps)} timestamps found. "
                    "Building zarr store from existing detections then returning."
                )
                self._detections_to_zarr(
                    dataset_attrs=dataset_attrs,
                    zarr_path=zarr_path,
                )
                return output_detections
            else:
                logging.info(
                    f"Still need to process {len(self.timestamps_to_process)} timestamps with the local model."
                )

        if force_reprocess:
            self.timestamps_processed = []
            self.timestamps_to_process = sorted(requested_timestamps)

        # Otherwise a timestamp that fails below would keep its old prediction
        self._remove_stale_outputs(self.timestamps_to_process)

        logging.info("Preparing environment for local model inference...")
        self._local_model_prepare(clean_caches=clean_caches)

        if hasattr(self.detector, "load_data_for_local_model"):
            logging.info("Loading tile data for local inference...")
            self.detector.load_data_for_local_model(
                tiles_to_load=tiles_to_load,
                n_jobs=n_jobs,
                timestamps_to_process=self.timestamps_to_process if not force_reprocess else None,
            )

        logging.info("Running local contrail detection model...")
        self._local_model_run()

        logging.info("Stitching tiles together into full-resolution detection maps...")
        self._join_tiles(tiles_to_join=tiles_to_load)

        logging.info("Loading newly generated detections from .npy files...")
        for ts in self.timestamps_to_process:
            time_str = pd.Timestamp(ts).strftime("%Y-%m-%dT%H%M")
            req_fmt = f"{self.detector.satellite}-{self.detector.region}-{time_str}"
            req_dir = self.detection_output_dir / req_fmt
            prediction_file = req_dir / "_prediction.npy"

            if prediction_file.exists():
                detection_array = np.load(prediction_file)
                output_detections[ts] = detection_array
                timestamps_found.add(ts)
            else:
                logging.warning(f"Expected detection file not found: {prediction_file}")

        # Update tracking attributes
        self.timestamps_processed = list(timestamps_found)
        self.timestamps_to_process = []

        # Build zarr store from all detections (existing + newly generated)
        logging.info("Building zarr store from all detections...")
        self._detections_to_zarr(
            dataset_attrs=dataset_attrs,
            zarr_path=zarr_path,
        )

        # Return the complete output_detections dictionary
        return output_detections

    def _cached_extent_matches(self: Self, req_dir: Path) -> bool:
        """Whether a cached prediction in ``req_dir`` was made for the detector's extent.

        Args:
            req_dir: Per-timestamp output directory holding ``_metadata.json``.

        Returns:
            False if the metadata records a different extent, True otherwise.
        """
        metadata_path = req_dir / "_metadata.json"
        if not metadata_path.exists():
            logging.warning(
                f"No _metadata.json in {req_dir}; cannot check the cached prediction's extent."
            )
            return True
        with open(metadata_path) as f:
            stored_extent = json.load(f).get("extent")
        if stored_extent is None:
            logging.warning(f"No extent in {metadata_path}; cannot check the cached extent.")
            return True
        if geo_utils.extents_match(tuple(stored_extent), tuple(self.detector.tile_extent)):
            return True
        logging.info(
            f"Cached prediction in {req_dir} is for extent {tuple(stored_extent)}, not "
            f"{tuple(self.detector.tile_extent)}; reprocessing it."
        )
        return False

    def _remove_stale_outputs(self: Self, timestamps: list[np.datetime64]) -> None:
        """Delete existing prediction, metadata and tile files for the given timestamps.

        Args:
            timestamps: Timestamps about to be processed afresh.
        """
        for ts in timestamps:
            time_str = pd.Timestamp(ts).strftime("%Y-%m-%dT%H%M")
            req_dir = (
                self.detection_output_dir
                / f"{self.detector.satellite}-{self.detector.region}-{time_str}"
            )
            for name in ("_prediction.npy", "_metadata.json", "tiles_data.npz"):
                stale = req_dir / name
                if stale.exists():
                    logging.debug(f"Removing stale output {stale}")
                    stale.unlink()

    def _local_model_prepare(self: Self, clean_caches: bool = False) -> None:
        """Clear the tile staging directory before inference.

        Args:
            clean_caches: Also empty the detector's pycontrails satellite cache.
        """
        clean_temp_dir()
        logging.info(f"Cleared test directory: {TEST_DIR}")

        if clean_caches:
            datalib = getattr(self.detector, "pycon_goes", None) or getattr(
                self.detector, "pycon_himawari", None
            )
            if datalib is not None and datalib.cachestore is not None:
                clean_pycontrails_cache(datalib.cachestore.cache_dir)
                logging.info(f"Cleared pycontrails cache: {datalib.cachestore.cache_dir}")

    def _local_model_run(self: Self) -> None:
        """Run the local contrail detection model.

        Executes the inference pipeline using the specified model architecture
        (ViT4 or UNet1024) on the ABI satellite data provided by the detector.

        The model configuration is loaded from a YAML string, and predictions
        are generated for all tiles in the test directory. The results are saved
        to the output directory specified during initialization.

        """
        # Batch size for the model
        if isinstance(self.device_ids, list):
            batch_size = 2 * len(self.device_ids)  # GPU mode
        else:
            batch_size = 1  # CPU mode - conservative setting for memory safety

        # Load model configuration from YAML string
        # full fold: [5,6,7,8,9]
        cfg = yaml.safe_load(
            f"""
            input:
              weight: "{str(self.WEIGHTS_PATH.resolve())}"

            vit4:
              batch_size: {int(batch_size)}
              models:
                - name: vit4_1024
                  encoder: maxvit_tiny_tf_512.in1k
                  decoder_channels: [256, 128, 64, 32, 32]
                  w: 0.6669411076255138
                  resize: 512
                  tta: d4prob
                  folds: [5]

            unet1024:
              batch_size: {int(batch_size)}
              models:
                - name: unet1024
                  encoder: maxvit_tiny_tf_512.in1k
                  decoder_channels: [256, 128, 64, 32, 16]
                  w: 0.42019871465275055
                  resize: 1024
                  tta: d4prob
                  folds: [0,2,3,4,5]
            """
        )
        logging.info(
            "Model configuration loaded\n"
            f"  - Using device IDs: {self.device_ids}\n"
            f"  - Using output directory: {self.detection_output_dir}\n"
            f"  - Using batch size: {batch_size}\n"
            f"  - Folds for vit4: {cfg['vit4']['models'][0]['folds']}\n"
            f"  - Using UNet1024 model: {self.unet1024}"
        )

        # Define data type and debug mode (Kaggle params)
        data_type = "test"
        debug = False

        # Set up the vit4 model
        preds = vit4.run(
            data_type=data_type,
            cfg=cfg,
            device_ids=self.device_ids,
            preds=None,
            debug=debug,
        )
        logging.info(f"Predictions obtained after Vit4: {len(preds)}")

        # # Check that predictions are finite

        # Optionally set up the unet1024 model
        if self.unet1024:
            unet1024.run(
                data_type=data_type,
                cfg=cfg,
                device_ids=self.device_ids,
                preds=preds,
                debug=debug,
            )
            logging.info(f"Predictions obtained after UNet1024: {len(preds)}")

        # List folders in the test directory (each folder is a tile)
        test_folders = [f for f in sorted(os.listdir(TEST_DIR)) if "tiles" not in f]

        # Create a lookup dictionary for predictions using 'file_id' as key
        pred_lookup = {p["file_id"]: p for p in preds}

        # Group tiles by timestamp for batch JSON saving
        self._process_tiles_by_timestamp(test_folders, pred_lookup)

        # Copy tile locations to the relevant output directory (if necessary)
        tile_dir = (
            self.detection_output_dir / f"{self.detector.satellite}-{self.detector.region}-tiles"
        )
        tile_dir.mkdir(parents=True, exist_ok=True)

        tile_loc_dir = Path(TEST_DIR) / f"{self.detector.satellite}-{self.detector.region}-tiles"
        for item in tqdm(tile_loc_dir.glob("*_lonlat.npy"), desc="Tile"):
            # Destination path
            dest_file = Path(tile_dir) / item.name

            # Overwrite, as tile numbering depends on the extent
            shutil.copy(item, dest_file)

            # Delete the original
            item.unlink()

    def _join_tiles(
        self: Self,
        tiles_to_join: list[int] | None = None,
    ) -> None:
        """Combine individual tile predictions into full-resolution detection maps.

        Reconstructs the complete satellite image by stitching together predictions
        from individual tiles. Uses weighted averaging in overlap regions to ensure
        smooth transitions between tiles. Only timestamps in
        ``self.timestamps_to_process`` are joined.

        Args:
            tiles_to_join: Specific tile indices to process. If None, processes all
                valid tiles in the detector.
        """
        # Set base paths for retrieving data
        tile_dir = (
            self.detection_output_dir / f"{self.detector.satellite}-{self.detector.region}-tiles"
        )

        # For fixed-domain regions (C, F) the geostationary coordinates are constant across
        # all timestamps and can be loaded once here.  For mesoscale (M1, M2) the domain
        # shifts every epoch, so geo_x/geo_y are loaded per-timestamp inside the loop below.
        is_mesoscale = self.detector.region in ["M1", "M2"]
        if not is_mesoscale:
            geo_x_global, geo_y_global = self.detector.image_geo_coords

        # Determine which tiles to join together
        if tiles_to_join is None:
            tiles_to_join = [item for item in np.unique(self.detector.tile_idx) if item != -1]

        # Convert timestamps_to_process to string format for comparison
        timestamps_to_process_str = [
            pd.Timestamp(ts).strftime("%Y-%m-%dT%H%M") for ts in self.timestamps_to_process
        ]

        # Loop over timestamps and tiles - only process timestamps that need it
        for time_str in tqdm(self.detector.timestamp_strings, desc="Timestamps"):
            # Skip if not in timestamps_to_process
            if time_str not in timestamps_to_process_str:
                logging.debug(f"Skipping already processed timestamp: {time_str}")
                continue

            # Directory for this timestamp
            req_fmt = f"{self.detector.satellite}-{self.detector.region}-{time_str}"
            req_dir = self.detection_output_dir / req_fmt

            # Resolve geostationary coordinates and lonlat file location for this timestamp.
            # Mesoscale: load per-timestamp coords (domain may have shifted since construction).
            # Fixed regions: reuse the global coords loaded above.
            if is_mesoscale:
                geo_coords_path = req_dir / "_geo_coords.npz"
                if not geo_coords_path.exists():
                    logging.warning(f"Missing _geo_coords.npz for {time_str}, skipping timestamp.")
                    continue
                _gc = np.load(geo_coords_path)
                geo_x, geo_y = _gc["geo_x"], _gc["geo_y"]
                lonlat_dir = req_dir
            else:
                geo_x, geo_y = geo_x_global, geo_y_global
                lonlat_dir = tile_dir

            # Grid parameters for indexing (recomputed per-timestamp for mesoscale)
            dx, dy = geo_x[1] - geo_x[0], geo_y[1] - geo_y[0]
            x_min, y_max = geo_x.min(), geo_y.max()

            # Initialize Canvas
            pred_sum = np.zeros((len(geo_y), len(geo_x)), dtype=float)
            weight_sum = np.zeros((len(geo_y), len(geo_x)), dtype=float)

            # Load NPZ data for all tiles at once
            tiles_data_file = req_dir / "tiles_data.npz"

            # Load compressed NPZ file containing the tile predictions
            if tiles_data_file.exists():
                tiles_data = np.load(tiles_data_file)
            else:
                logging.warning(f"Tiles data file not found: {tiles_data_file}")
                continue

            for tile in tiles_to_join:
                pred_key = f"pred_{tile:03d}"

                # Load prediction data from NPZ (256,256)
                if pred_key in tiles_data:
                    processed_img = tiles_data[pred_key].astype(np.float32)
                else:
                    logging.warning(f"Tile {tile:03d} predictions not found")
                    continue

                # Load tile coordinates (256,256)
                xx_tile, yy_tile = np.load(lonlat_dir / f"{tile:03d}_lonlat.npy")

                # Mask for NaN values - check predictions and coordinates
                valid_pixel_mask = (
                    ~np.isnan(processed_img) & ~np.isnan(xx_tile) & ~np.isnan(yy_tile)
                )
                weight_mask = valid_pixel_mask.astype(float)

                # Replace NaNs in input data with 0 (summation issues)
                processed_img = np.nan_to_num(processed_img, nan=0.0)

                # Convert geo coords (xx, yy) to integer indices (i, j) on the full grid
                row_idx = np.round((xx_tile - x_min) / dx).astype(int)
                col_idx = np.round((yy_tile - y_max) / dy).astype(int)
                valid_mask_2d = (
                    (row_idx >= 0)
                    & (row_idx < len(geo_x))
                    & (col_idx >= 0)
                    & (col_idx < len(geo_y))
                )

                # Get only valid indices and corresponding data
                i_valid = row_idx[valid_mask_2d]
                j_valid = col_idx[valid_mask_2d]
                pred_valid = processed_img[valid_mask_2d]
                mask_valid = weight_mask[valid_mask_2d]

                # Perform weighted accumulation using numpy's scatter-like function
                np.add.at(pred_sum, (j_valid, i_valid), pred_valid * mask_valid)
                np.add.at(weight_sum, (j_valid, i_valid), mask_valid)

            # Stitch together and save detections
            stitched_pred = np.divide(
                pred_sum,
                weight_sum,
                out=np.full_like(pred_sum, np.nan),
                where=weight_sum != 0,
            )
            np.save(req_dir / "_prediction.npy", stitched_pred)

            # Save metadata with extent information
            metadata = {
                "extent": list(self.detector.tile_extent),
                "satellite": self.detector.satellite,
                "region": self.detector.region,
                "timestamp": time_str,
                "date_created": pd.Timestamp.now().isoformat(),
            }
            with open(req_dir / "_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    def _process_tiles_by_timestamp(
        self: Self,
        test_folders: list[str],
        pred_lookup: dict,
    ) -> None:
        """
        Group tile predictions by timestamp and save each group to ``tiles_data.npz``.

        Args:
            test_folders: List of tile folder names from test directory.
            pred_lookup: Dictionary mapping tile_id to prediction results.
        """
        # Group tiles by timestamp
        tiles_by_timestamp = {}
        for tile_id in sorted(test_folders):
            if tile_id not in pred_lookup:
                continue

            # Extract timestamp from tile_id (e.g., 'GOES-16-F-2024-11-10T0130-020')
            timestamp_parts = tile_id.split("-")[-4:-1]  # Get YYYY-MM-DDTHHMMM parts
            timestamp_key = "-".join(timestamp_parts)

            if timestamp_key not in tiles_by_timestamp:
                tiles_by_timestamp[timestamp_key] = []
            tiles_by_timestamp[timestamp_key].append(tile_id)

        # Process each timestamp
        for timestamp_key, tile_ids in tqdm(
            tiles_by_timestamp.items(), desc="Processing Timestamps"
        ):
            # Create timestamp directory
            timestamp_fmt = f"{self.detector.satellite}-{self.detector.region}-{timestamp_key}"
            timestamp_dir = self.detection_output_dir / timestamp_fmt
            timestamp_dir.mkdir(parents=True, exist_ok=True)

            # Containers for all tiles in this timestamp
            predictions_data = {}

            # Process all tiles for this timestamp
            for tile_id in tile_ids:
                tile_num = int(tile_id.split("-")[-1])

                if tile_id in pred_lookup:
                    pred = pred_lookup[tile_id]["y_pred"][0]  # (256, 256)
                    # Store in dictionary (keep as numpy arrays)
                    predictions_data[f"pred_{tile_num:03d}"] = pred
                else:
                    logging.warning(
                        f"No prediction found for {tile_id}.\n"
                        f"Available keys: {list(pred_lookup.keys())}"
                    )

            # Save all tiles for this timestamp to compressed NPZ file
            if predictions_data:
                np.savez_compressed(timestamp_dir / "tiles_data.npz", **predictions_data)

            # For mesoscale regions, copy per-timestamp coordinate files from the staging
            # directory into the output directory so that _join_tiles can access them
            # after TEST_DIR has been cleared. Existing copies are overwritten, as the
            # mesoscale domain can move.
            if self.detector.region in ["M1", "M2"]:
                # Tile coordinate files, geostationary metres, one per tile
                for tile_id in tile_ids:
                    tile_num = int(tile_id.split("-")[-1])
                    src_lonlat = Path(TEST_DIR) / tile_id / f"{tile_num:03d}_lonlat.npy"
                    dst_lonlat = timestamp_dir / f"{tile_num:03d}_lonlat.npy"
                    if src_lonlat.exists():
                        shutil.copy(src_lonlat, dst_lonlat)

                # Copy the per-timestamp geo coordinate file (written by _process_single_timestamp)
                src_geo = mesoscale_geo_coords_path(
                    self.detector.satellite, self.detector.region, timestamp_key
                )
                dst_geo = timestamp_dir / "_geo_coords.npz"
                if src_geo.exists():
                    shutil.copy(src_geo, dst_geo)

    def _validate_device_ids(self: Self) -> None:
        """
        Validate whether the requested GPU device IDs are available.

        Raises:
            ValueError: If any of the requested GPUs are not available.
        """
        available_gpus = torch.cuda.device_count()

        # Check if device_ids is a non-empty list or the string 'CPU'
        if not isinstance(self.device_ids, (list, str)):
            raise ValueError(
                "'device_ids' must be a non-empty list of GPU indices or the string 'CPU'."
            )

        # Check for CPU call
        if isinstance(self.device_ids, str):
            if self.device_ids.upper() != "CPU":
                raise ValueError(
                    "If 'device_ids' is a string, it must be 'CPU' (case-insensitive)."
                )
        else:
            # Check for GPU call - only iterate if device_ids is a list
            for device_id in self.device_ids:
                # Check that each item is an integer
                if not isinstance(device_id, int):
                    raise ValueError(
                        f"Each item in 'device_ids' must be an integer if list provided. "
                        f"Found invalid item: {device_id} of type {type(device_id)}"
                    )
                # Check that the integer is within the valid range
                if device_id < 0 or device_id >= available_gpus:
                    raise ValueError(
                        f"Requested GPU device ID {device_id} is not available. "
                        f"Available GPU indices: [0 to {available_gpus - 1}]"
                    )

        logging.info(
            f"Validated 'device_ids': {self.device_ids} (available GPUs: {available_gpus})"
        )
