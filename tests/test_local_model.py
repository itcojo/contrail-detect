"""Tests for the local model backend (``LocalContrailModel``) without torch or downloads.

torch and the bundled model packages are stubbed for the duration of each test, and the
model itself is replaced by a function that writes tile predictions as the real one does.
"""

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import contrail_detect.geostationary.abi_base as abi_base

_KAGGLE_PKG = "contrail_detect.models.kaggle_contrails_solution"
_STUBBED = [
    _KAGGLE_PKG,
    f"{_KAGGLE_PKG}.input",
    f"{_KAGGLE_PKG}.input.models",
    f"{_KAGGLE_PKG}.input.models.unet1024",
    f"{_KAGGLE_PKG}.input.models.vit4",
]


@pytest.fixture
def local_model():
    """The ``local_model`` module, imported against stub torch and model packages."""
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.cuda = MagicMock(device_count=MagicMock(return_value=0))
    stubs = {"torch": torch_stub, **{name: MagicMock() for name in _STUBBED}}
    # patch.dict restores sys.modules afterwards, dropping the module imported here
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("contrail_detect.models.local_model", None)
        import contrail_detect.models.local_model as module

        yield module


# ── Full Disk run with a stub detector and a fake model ─────────────────────────

_TS = [np.datetime64("2024-11-10T01:30:00"), np.datetime64("2024-11-10T01:40:00")]
_TS_STR = ["2024-11-10T0130", "2024-11-10T0140"]
_GEO_X = np.arange(4.0) * 2000.0
_GEO_Y = (np.arange(4.0) * 2000.0)[::-1].copy()
_EXTENT = (0.0, 1.0, 0.0, 1.0)


@pytest.fixture
def make_model(local_model, tmp_path, geo_detector):
    """Build a LocalContrailModel whose fake model writes ``local_value`` for ``produce``.

    If ``produce`` is None the fake model writes every timestamp in ``timestamps_to_process``.
    """

    def _model(n_times: int = 1, local_value: float = 0.8, produce=None):
        model = object.__new__(local_model.LocalContrailModel)
        detector = geo_detector(tmp_path, _TS[:n_times], "local", _GEO_X, _GEO_Y, "GOES-16")
        detector.tile_idx = np.zeros((4, 4), int)
        detector.load_data_for_local_model = MagicMock()
        model.detector = detector
        model.detection_output_dir = tmp_path / "local"
        model.contrail_model_name = "k"
        model.timestamps_processed, model.timestamps_to_process = [], []
        model.device_ids, model.unet1024 = "CPU", False
        model._local_model_prepare = MagicMock()
        model.local_runs = 0

        def fake_local_run():
            model.local_runs += 1
            times = model.timestamps_to_process if produce is None else produce
            for ts in times:
                time_str = pd.Timestamp(ts).strftime("%Y-%m-%dT%H%M")
                req_dir = model.detection_output_dir / f"GOES-16-F-{time_str}"
                req_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    req_dir / "tiles_data.npz", pred_000=np.full((4, 4), local_value, np.float32)
                )
            tiles = model.detection_output_dir / "GOES-16-F-tiles"
            tiles.mkdir(parents=True, exist_ok=True)
            np.save(tiles / "000_lonlat.npy", np.meshgrid(_GEO_X, _GEO_Y))

        model._local_model_run = fake_local_run
        return model

    return _model


def _cache(model, t_idx: int, value: float, extent=_EXTENT) -> None:
    """Write a cached prediction and its metadata."""
    req_dir = model.detection_output_dir / f"GOES-16-F-{_TS_STR[t_idx]}"
    req_dir.mkdir(parents=True, exist_ok=True)
    np.save(req_dir / "_prediction.npy", np.full((4, 4), value))
    with open(req_dir / "_metadata.json", "w") as f:
        json.dump({"extent": list(extent)}, f)


def _archive(model, tmp_path: Path) -> xr.Dataset:
    return xr.open_zarr(model.detector.get_zarr_path(tmp_path / "z"))


class TestForceReprocess:
    def test_replaces_a_stale_cached_prediction(self, make_model, tmp_path):
        model = make_model()
        _cache(model, 0, 0.1)
        out = model.run_detections(force_reprocess=True, zarr_path=tmp_path / "z")
        assert model.local_runs == 1
        np.testing.assert_allclose(out[_TS[0]], 0.8)
        archive = _archive(model, tmp_path)
        assert float(archive.detection.max()) == pytest.approx(0.8)

    def test_missed_timestamp_not_filled_from_stale_cache(self, make_model, tmp_path):
        model = make_model(n_times=2, produce=[_TS[0]])
        _cache(model, 0, 0.1)
        _cache(model, 1, 0.1)
        out = model.run_detections(force_reprocess=True, zarr_path=tmp_path / "z")
        kwargs = model.detector.load_data_for_local_model.call_args.kwargs
        assert kwargs["timestamps_to_process"] is None
        np.testing.assert_allclose(out[_TS[0]], 0.8)
        assert np.isnan(out[_TS[1]]).all()
        archive = _archive(model, tmp_path)
        assert np.isnan(archive.detection.isel(time=1).values).all()


class TestCachedExtent:
    def test_cache_for_another_extent_is_reprocessed(self, make_model, tmp_path):
        model = make_model()
        _cache(model, 0, 0.5, extent=(50.0, 60.0, 10.0, 20.0))
        out = model.run_detections(zarr_path=tmp_path / "z")
        assert model.local_runs == 1
        np.testing.assert_allclose(out[_TS[0]], 0.8)
        archive = _archive(model, tmp_path)
        assert tuple(archive.attrs["extent"]) == _EXTENT
        assert float(archive.detection.max()) == pytest.approx(0.8)
        req_dir = model.detection_output_dir / f"GOES-16-F-{_TS_STR[0]}"
        assert json.loads((req_dir / "_metadata.json").read_text())["extent"] == list(_EXTENT)

    def test_cache_for_the_same_extent_is_reused(self, make_model, tmp_path):
        model = make_model()
        _cache(model, 0, 0.5)
        out = model.run_detections(zarr_path=tmp_path / "z")
        assert model.local_runs == 0
        np.testing.assert_allclose(out[_TS[0]], 0.5)

    def test_cache_without_metadata_is_reused(self, make_model, tmp_path, caplog):
        model = make_model()
        _cache(model, 0, 0.5)
        (model.detection_output_dir / f"GOES-16-F-{_TS_STR[0]}" / "_metadata.json").unlink()
        with caplog.at_level(logging.WARNING):
            out = model.run_detections(zarr_path=tmp_path / "z")
        assert model.local_runs == 0
        np.testing.assert_allclose(out[_TS[0]], 0.5)
        assert "cannot check the cached prediction's extent" in caplog.text


# ── _local_model_run ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("unet1024", [False, True])
def test_unet1024_log_only_when_it_ran(
    local_model, make_model, tmp_path, monkeypatch, caplog, unet1024
):
    monkeypatch.setattr(local_model, "TEST_DIR", tmp_path)
    local_model.vit4.run.return_value = []
    model = make_model()
    model.unet1024 = unet1024
    model.WEIGHTS_PATH = tmp_path
    del model._local_model_run  # use the real method
    (tmp_path / "GOES-16-F-tiles").mkdir()
    with caplog.at_level(logging.INFO):
        model._local_model_run()
    assert ("Predictions obtained after UNet1024" in caplog.text) is unet1024
    assert local_model.unet1024.run.called is unet1024


def test_earlier_tile_positions_are_replaced(local_model, make_model, tmp_path, monkeypatch):
    monkeypatch.setattr(local_model, "TEST_DIR", tmp_path / "test")
    local_model.vit4.run.return_value = []
    model = make_model()
    model.WEIGHTS_PATH = tmp_path
    del model._local_model_run
    staged = tmp_path / "test" / "GOES-16-F-tiles"
    staged.mkdir(parents=True)
    np.save(staged / "000_lonlat.npy", np.full((2, 4, 4), 7.0))
    kept = model.detection_output_dir / "GOES-16-F-tiles"
    kept.mkdir(parents=True)
    np.save(kept / "000_lonlat.npy", np.full((2, 4, 4), -1.0))
    model._local_model_run()
    np.testing.assert_array_equal(np.load(kept / "000_lonlat.npy"), 7.0)


# ── Mesoscale: staging through joining, on a synthetic 500 x 500 frame ────────


def test_mesoscale_staging_and_join(local_model, stub_abi, tmp_path, monkeypatch):
    """Joined tiles cover every pixel of a synthetic 500 x 500 M1 frame."""
    monkeypatch.setattr(abi_base, "TEST_DIR", tmp_path / "test")
    monkeypatch.setattr(local_model, "TEST_DIR", tmp_path / "test")
    (tmp_path / "test").mkdir()

    # 2 km pixels, y descending, far from the origin
    geo_x = 1.0e6 + 2004.0 * np.arange(500)
    geo_y = 3.0e6 - 2004.0 * np.arange(500)
    image_time = pd.Timestamp("2024-11-10T01:30")
    times = [image_time - pd.Timedelta(minutes=k) for k in [4, 3, 2, 1, 0]]

    det = object.__new__(stub_abi)
    det.satellite, det.region, det.image_interval = "GOES-16", "M1", 1
    det.sat_to_latlon_transform = MagicMock()
    det.sat_to_latlon_transform.transform.side_effect = lambda xx, yy: (
        np.full(xx.shape, -90.0),
        np.full(yy.shape, 40.0),
    )
    img = np.ones((500, 500, 3), np.float32)
    det.projection_cache = {
        t: {"geo_x": geo_x, "geo_y": geo_y, "crs": None, "img": img, "extent": (0, 1, 2, 3)}
        for t in times
    }
    data_cache = dict.fromkeys(times, object())
    det._process_single_timestamp(
        timestamp=image_time,
        tiles_to_load=[0, 1, 2, 3],
        data_cache=data_cache,
        tile_memory_cache={},
    )

    # Frame coordinates go in the '-tiles' directory, not a tile directory
    staged = sorted(p.name for p in (tmp_path / "test").iterdir())
    tile_dirs = [d for d in staged if "-tiles" not in d]
    assert tile_dirs == [f"GOES-16-M1-2024-11-10T0130-{t:03d}" for t in range(4)]
    for d in tile_dirs:
        assert all((tmp_path / "test" / d / f"band_{b}.npy").exists() for b in (11, 14, 15))
    geo_path = abi_base.mesoscale_geo_coords_path("GOES-16", "M1", "2024-11-10T0130")
    assert geo_path.exists() and geo_path.parent.name == "GOES-16-M1-tiles"

    # Each tile records its pixels' geostationary coordinates in metres
    tile_xx, tile_yy = np.load(tmp_path / "test" / tile_dirs[3] / "003_lonlat.npy")
    np.testing.assert_array_equal(tile_xx[0], geo_x[244:500])
    np.testing.assert_array_equal(tile_yy[:, 0], geo_y[244:500])

    # Predictions sampled from a known field, which the join should rebuild
    truth = np.add.outer(np.arange(500.0), 1000.0 * np.arange(500.0))
    pred_lookup = {}
    for t, d in enumerate(tile_dirs):
        rows, cols = abi_base.mobile_tile_slices(t)
        pred_lookup[d] = {"y_pred": [truth[rows, cols]]}

    model = object.__new__(local_model.LocalContrailModel)
    model.detector = types.SimpleNamespace(
        satellite="GOES-16",
        region="M1",
        timestamp_strings=["2024-11-10T0130"],
        tile_extent=_EXTENT,
    )
    model.detection_output_dir = tmp_path / "local"
    model.timestamps_to_process = [np.datetime64("2024-11-10T01:30")]
    model._process_tiles_by_timestamp(tile_dirs, pred_lookup)
    model._join_tiles(tiles_to_join=[0, 1, 2, 3])

    joined = np.load(tmp_path / "local" / "GOES-16-M1-2024-11-10T0130" / "_prediction.npy")
    assert joined.shape == (500, 500)
    np.testing.assert_allclose(joined, truth)
