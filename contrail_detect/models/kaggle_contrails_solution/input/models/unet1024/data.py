"""Data loading and preprocessing for UNet1024 model inference.

This module provides dataset and dataloader classes for loading satellite imagery
with single timestep, applying false-color transformations, and preparing data
for model inference.
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from contrail_detect.core.config import TEST_DIR

# Module-level constants
# Parent of the tile folder ABIDetector writes to, which is outside the package once installed
DATA_DIR = TEST_DIR.parent


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def ash_color(x: torch.Tensor) -> torch.Tensor:
    """Apply false-color transformation for contrail visualization.

    Converts satellite bands 11, 14, and 15 into an RGB false-color image
    optimized for contrail detection.

    Args:
        x: Tensor of shape (3, H, W) where channels correspond to:
            - Channel 0: band 11
            - Channel 1: band 14
            - Channel 2: band 15

    Returns:
        False-color RGB tensor of shape (3, H, W) with values in [0, 1].

    Note:
        Channel mapping:
        - Red: band15 - band14
        - Green: band14 - band11
        - Blue: band14
    """
    r = rescale_range(x[2] - x[1], -4, 2)
    g = rescale_range(x[1] - x[0], -4, 5)
    b = rescale_range(x[1], 243, 303)

    out = torch.stack([r, g, b], axis=0)
    out = 1 - out

    return out


def get_optimal_num_workers() -> int:
    """Determine optimal number of DataLoader workers based on system resources.

    The function dynamically calculates the number of workers to use for data
    loading, balancing performance and system resource constraints.

    Returns:
        Optimal number of workers in range [0, 8]. Returns 0 if CPU count
        cannot be determined.

    Note:
        Logic:
        - Leaves at least 1 core free for the main process
        - Capped at 8 workers to avoid excessive overhead
        - Ensures minimum 2 workers on systems with 4+ cores
        - Falls back to 2 workers on exception
    """
    try:
        cpu_count = os.cpu_count() or 1
        num_workers = min(8, max(0, cpu_count - 1))

        if cpu_count >= 4:
            num_workers = max(2, num_workers)

        return num_workers
    except Exception:
        return 2


def load_data(path: str, t: int = 4) -> dict:
    """Load satellite band data from disk for a single timestep.

    Args:
        path: Directory containing band_{11,14,15}.npy files.
        t: Timestep index to extract (default: 4).

    Returns:
        Dictionary with keys:
            - 'file_id': Directory name used as identifier
            - 'x': Array of shape (3, 256, 256) containing bands 11, 14, 15
                   at timestep t

    Raises:
        FileNotFoundError: If band files are missing.
    """
    bands_to_load = ['11', '14', '15']
    loaded_bands = []

    for band in bands_to_load:
        file_path = Path(path) / f"band_{band}.npy"
        a = np.load(file_path)  # Shape: (256, 256, T)
        loaded_bands.append(a)

    # Stack: (3, 256, 256, T) -> Select timestep t: (3, 256, 256)
    x = np.stack(loaded_bands, axis=0)

    return {
        'file_id': Path(path).name,
        'x': x[:, :, :, t],
    }


def load_dirs(data_type: str) -> np.ndarray:
    """Load and return sorted list of data directories.

    Args:
        data_type: One of 'train', 'validation', or 'test'.

    Returns:
        Array of directory paths, excluding tile directories.

    Raises:
        AssertionError: If data_type is invalid.
    """
    assert data_type in ['train', 'validation', 'test']

    dirs = glob.glob(str(DATA_DIR / data_type / '*'))
    dirs.sort()
    dirs = [item for item in dirs if '-tiles' not in item]

    return np.array(dirs)


def rescale_range(x: torch.Tensor, f_min: float, f_max: float) -> torch.Tensor:
    """Rescale tensor values from [f_min, f_max] to [0, 1].

    Args:
        x: Input tensor.
        f_min: Minimum value of input range.
        f_max: Maximum value of input range.

    Returns:
        Rescaled tensor with values in [0, 1].
    """
    return (x - f_min) / (f_max - f_min)


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class Data:
    """Data manager for loading and organizing satellite imagery directories.

    This class handles directory management and DataLoader creation for the
    UNet1024 model inference pipeline.

    Attributes:
        dirs: Array of directory paths containing satellite data.
    """

    def __init__(self, data_type: str, *, debug: bool = False) -> None:
        """Initialize data manager.

        Args:
            data_type: Type of data split ('train', 'validation', or 'test').
            debug: If True, limit to first 100 samples for testing.

        Raises:
            AssertionError: If data_type is not valid.
        """
        assert data_type in ['train', 'validation', 'test']

        dirs = load_dirs(data_type)
        if debug:
            dirs = dirs[:100]

        self.dirs = dirs

    def __len__(self) -> int:
        """Return number of data samples."""
        return len(self.dirs)

    def loader(self, cfg: dict, m: nn.Module) -> torch.utils.data.DataLoader:
        """Create DataLoader for model inference.

        Args:
            cfg: Configuration dictionary with 'batch_size' and 'resize' keys.
            m: Model configuration passed to Dataset.

        Returns:
            Configured DataLoader with optimal worker settings.
        """
        batch_size = cfg['batch_size']
        num_workers = get_optimal_num_workers()

        return torch.utils.data.DataLoader(
            Dataset(self.dirs, m),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )


class Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading single-timestep satellite imagery.

    Loads satellite bands 11, 14, and 15 for a specific timestep, applies
    false-color transformation, and prepares data for model input.

    Attributes:
        dirs: List of directory paths.
        resize: Transform to resize images to target size.
    """

    def __init__(self, dirs: list[str], cfg: dict) -> None:
        """Initialize dataset.

        Args:
            dirs: List of directory paths containing satellite data.
            cfg: Configuration dictionary with 'resize' key specifying target size.
        """
        self.dirs = dirs
        nc = cfg['resize']
        self.resize = nn.Identity() if nc == 256 else T.Resize(nc, antialias=False)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dirs)

    def __getitem__(self, i: int) -> dict:
        """Load and preprocess a single sample.

        Args:
            i: Sample index.

        Returns:
            Dictionary with keys:
                - 'file_id': Sample identifier
                - 'x': Tensor of shape (3, H, W) containing false-color RGB image
        """
        path = self.dirs[i]
        d = load_data(path)

        x = torch.from_numpy(d['x'])
        x = ash_color(x)
        x = self.resize(x)

        return {
            'file_id': d['file_id'],
            'x': x,
        }
