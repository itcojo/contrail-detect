import logging

import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from contrail_detect.core.common import Colors, Transforms
from contrail_detect.core.config import FIGURE_DIR
from contrail_detect.geostationary.goes import GOESDetector

colors, transforms = Colors(), Transforms()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


def check_gpu_availability() -> list[int] | str:
    """Return the available GPU device IDs, or ``"CPU"`` if there are none.

    Returns:
        List of available GPU device IDs, or ``"CPU"``.
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        device_ids = list(range(n_gpus))
        logging.info(
            f"Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in device_ids]}"
        )
        return device_ids
    logging.warning("No GPUs found; running on the CPU, which is much slower.")
    return "CPU"


def plot_inference_outputs(
    detector: GOESDetector,
    extent: tuple[float, float, float, float],
    satellite: str = "GOES-19",
    timestamp: str = "2026-01-09T15:00:00",
    score_threshold: float = 0.30,
) -> None:
    """Plot contrail detection scores over the eastern US and western Atlantic.

    Args:
        detector: Detector providing the per-pixel ``lons``/``lats`` grids and
            the ``detections`` produced by :meth:`get_detections`.
        extent: ``(lon_min, lon_max, lat_min, lat_max)`` corridor bounds.
        satellite: Satellite name, used in the plot title.
        timestamp: UTC timestamp, used in the plot title.
        score_threshold: Scores below this are hidden from the plot. Defaults to the
            local model's decomposition threshold.
    """
    detections = detector.detections[detector.timestamp_sequence[0]]
    lon_min, lon_max, lat_min, lat_max = extent

    # Mask detection array to the corridor extent
    extent_mask_2d = (
        (detector.lons >= lon_min)
        & (detector.lons <= lon_max)
        & (detector.lats >= lat_min)
        & (detector.lats <= lat_max)
    )

    # Crop to bounding box
    rows = np.where(np.any(extent_mask_2d, axis=1))[0]
    cols = np.where(np.any(extent_mask_2d, axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    lons_crop = detector.lons[r0:r1, c0:c1]
    lats_crop = detector.lats[r0:r1, c0:c1]

    # Mask data where coordinates are invalid or outside extent
    invalid_coords = ~np.isfinite(lons_crop) | ~np.isfinite(lats_crop)
    det_crop = np.where(
        extent_mask_2d[r0:r1, c0:c1] & ~invalid_coords,
        detections[r0:r1, c0:c1],
        np.nan,
    )

    # pcolormesh requires finite coordinate arrays — fill invalid coord positions
    # with a dummy finite value (those cells are NaN in data so won't be rendered)
    lons_plot = np.where(np.isfinite(lons_crop), lons_crop, 0.0)
    lats_plot = np.where(np.isfinite(lats_crop), lats_crop, 0.0)

    fig, ax = plt.subplots(
        figsize=(12, 7),
        dpi=120,
        subplot_kw={"projection": transforms.platecarree()},
    )
    lon_pad, lat_pad = 5, 3
    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=transforms.platecarree(),
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="k")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="k", linestyle=":")
    ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.4, color="lightgrey", zorder=1)

    det_crop_plot = det_crop.copy()
    det_crop_plot[det_crop_plot < score_threshold] = np.nan
    im = ax.pcolormesh(
        lons_plot,
        lats_plot,
        det_crop_plot,
        cmap="afmhot_r",
        vmin=0,
        vmax=1,
        transform=transforms.platecarree(),
        zorder=2,
    )
    plt.colorbar(im, ax=ax, label="Detection Score", shrink=0.9)

    rect = mpatches.Rectangle(
        xy=(lon_min, lat_min),
        width=lon_max - lon_min,
        height=lat_max - lat_min,
        linewidth=1.5,
        edgecolor="steelblue",
        facecolor="none",
        transform=transforms.platecarree(),
        zorder=3,
    )
    ax.add_patch(rect)
    ax.set_title(
        f"GOES-East ({satellite}) contrail detections — eastern US and western Atlantic"
        f"\n{timestamp}",
        fontsize=11,
    )
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / f"inference_demo_{satellite}_{timestamp.replace(':', '')}.png"
    fig.savefig(out_path, bbox_inches="tight")
    logging.info(f"Figure saved to {out_path}")
    plt.show()


def run_inference(
    timestamp: str = "2026-01-09T15:00:00",
    extent: tuple[float, float, float, float] = (-100.0, -50.0, 20.0, 55.0),
    satellite: str = "GOES-19",
    region: str = "F",
) -> GOESDetector:
    # Check GPU availability
    device_ids = check_gpu_availability()

    # Configure satellite, region, and extent
    start_time = np.datetime64(timestamp)
    end_time = start_time  # a single Full Disk frame

    logging.info(f"Satellite : {satellite} ({region})")
    logging.info(f"Period    : {start_time} → {end_time}")
    logging.info(f"Extent    : lon [{extent[0]}, {extent[1]}]  lat [{extent[2]}, {extent[3]}]")

    # Initialize GOESDetector
    detector = GOESDetector(
        goes_satellite=satellite,
        goes_region=region,
        start_time=start_time,
        end_time=end_time,
    )

    n_tiles = int(np.sum(np.unique(detector.tile_idx) != -1))
    logging.info(f"Full Disk tiles generated: {n_tiles}")

    # Downselect tiles whose centres fall within EXTENT
    lon_min, lon_max, lat_min, lat_max = extent
    extent_mask = (
        (detector.tile_lons - 360.0 >= lon_min)
        & (detector.tile_lons - 360.0 <= lon_max)
        & (detector.tile_lats >= lat_min)
        & (detector.tile_lats <= lat_max)
    )
    selected_tiles = list(np.where(extent_mask)[0])
    logging.info(f"Tiles in EXTENT: {len(selected_tiles)} / {n_tiles}")

    # Run inference on selected tiles
    detections_dict = detector.get_detections(
        method="local",
        device_ids=device_ids,
        tiles_to_load=selected_tiles,
        n_jobs=1,
    )
    logging.info("Local inference complete.")

    detector.detections = detections_dict

    detections = detector.detections[detector.timestamp_sequence[0]]
    logging.info(f"Shape: {detections.shape}   Non-NaN pixels: {np.isfinite(detections).sum():,}")

    return detector


if __name__ == "__main__":
    # Configuration
    timestamp = "2026-01-09T15:00:00"
    extent = (-100.0, -50.0, 20.0, 55.0)
    satellite = "GOES-19"
    region = "F"

    # Run inference
    detector = run_inference(
        timestamp=timestamp,
        extent=extent,
        satellite=satellite,
        region=region,
    )

    # Plot results
    plot_inference_outputs(
        detector=detector,
        extent=extent,
        satellite=satellite,
        timestamp=timestamp,
    )
