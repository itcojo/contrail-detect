from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Data
from .model import Model


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def init(dirs: list[str]) -> list[dict[str, Any]]:
    """
    Initialize prediction dictionaries for a list of directories.

    Args:
        dirs: List of directory paths corresponding to data samples.

    Returns:
        list[dict[str, Any]]: A list of prediction dictionaries, each containing:
            - 'file_id' (str): The base name of the directory.
            - 'y_pred' (np.ndarray): Zero-initialized prediction array of shape (1, 256, 256).
    """
    preds = []
    for d in dirs:
        pred = {
            'file_id': str(Path(d).name),
            'y_pred': np.zeros((1, 256, 256), dtype=np.float32)
        }
        preds.append(pred)

    return preds


def predict1(
    model: torch.nn.Module,
    loader: DataLoader,
    w: float,
    device: torch.device,
    preds: list[dict[str, Any]]
) -> None:
    """
    Perform inference with a single model and update predictions.

    Args:
        model: The PyTorch model to use for prediction.
        loader: DataLoader providing input batches.
        w: Weight to scale predictions before accumulation.
        device: Target device (e.g., 'cuda:0') for inference.
        preds: List of prediction dictionaries to be updated in-place.
            Each dict must contain:
                - 'y_pred': np.ndarray to accumulate predictions.
    """
    i = 0
    for d in tqdm(loader, desc="Predicting"):
        x = d['x'].to(device)  # input image tensor

        with torch.no_grad():
            _, y_pred = model(x)  # Forward pass (expecting a tuple output)

        y_pred = y_pred.sigmoid().cpu().numpy()  # Apply sigmoid activation & move to CPU

        for y_pred1 in y_pred:
            preds[i]['y_pred'] += w * y_pred1  # Weighted accumulation
            i += 1


def run(
    data_type: str,
    cfg: dict[str, Any],
    device_ids: list[int] | str,
    preds: list[Any] | None,
    *,
    debug: bool = False
) -> list[Any]:
    """
    Run model inference using specified configuration and device setup.

    Args:
        data_type: Type of data to load (e.g., 'train', 'test').
        cfg: Configuration dictionary containing model parameters.
        device_ids: List of GPU devices to use (zero-based) or 'CPU'.
        preds: Existing list of predictions to update. If None, initializes a new list.
        debug: Flag to enable debug mode for data loading. Defaults to False.

    Returns:
        list[Any]: Updated list of predictions after model inference.
    """
    # Specify the main device
    if isinstance(device_ids, list):
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = torch.device('cpu')

    # Data
    data = Data(data_type, debug=debug)

    if preds is None:
        preds = init(data.dirs)

    dataset_dir = cfg['input']['weight']

    for m in cfg['vit4']['models']:
        # m (dict)
        #   name: resnest26
        #   encoder: timm-resnest26d
        #   folds: [0, 1, 2, 3, 4]
        folds = m['folds']
        w = m['w'] / len(folds)

        loader = data.loader(cfg['vit4'], m)

        for ifold in folds:
            # Load Model
            model_filename = Path(dataset_dir) / 'vit4' / m['name'] / f"model{ifold}.pytorch"

            model = Model(m)
            model.load_state_dict(torch.load(
                model_filename,
                map_location=device,
                weights_only=True,
            ))
            model.to(device)
            
            # Parallelize if >1 GPU
            if len(device_ids) > 1 and isinstance(device_ids, list):
                model = nn.DataParallel(model, device_ids=device_ids)

            model.eval()

            # Predict using your inference function
            predict1(model, loader, w, device, preds)
            
            # Clean up
            del model
            if isinstance(device_ids, list):
                torch.cuda.empty_cache()  # GPU only

    return preds