"""Unit tests for the detection models in contrail_detect.models.

Covers the shared base (retries, disk cache, the M1/M2 epoch guard), the Google API
model and the Contrails.org API model. The real APIs are only called by the tests marked
``integration``; elsewhere requests.get is patched.
"""

import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from contrail_detect.core.config import CONTRAILS_API_HEADERS, CONTRAILS_API_URL_DETECTIONS
from contrail_detect.models.contrail_model_base import (
    ContrailModelBase,
    get_with_retries,
    require_api_frame_times,
)
from contrail_detect.models.contrails_api import ContrailsAPIContrailModel
from contrail_detect.models.google_api import (
    _SATELLITE_ORIGIN_MAP,
    GoogleAPIContrailModel,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _bare(satellite: str = "GOES-16") -> GoogleAPIContrailModel:
    """Bypass __init__ and return a minimal GoogleAPIContrailModel."""
    obj = object.__new__(GoogleAPIContrailModel)
    detector = MagicMock()
    detector.satellite = satellite
    detector.region = "F"
    detector.timestamp_strings = ["20260115_1200"]
    detector.timestamp_sequence = [np.datetime64("2026-01-15T12:00")]
    detector.tile_extent = [0, 0, 100, 100]
    detector.lons = None
    detector.lats = None
    detector.image_geo_coords = (np.linspace(-100, 100, 10), np.linspace(-60, 60, 8))
    obj.detector = detector
    obj.satellite_origin = _SATELLITE_ORIGIN_MAP[satellite]
    obj.timeout = 120.0
    obj.max_retries = 3
    obj.xx = None
    obj.yy = None
    obj.detection_output_dir = MagicMock(spec=Path)
    obj.grid_path = MagicMock(spec=Path)
    return obj


def _segments_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-15T12:00:00Z")] * n,
            "lon0": np.linspace(-5.0, 5.0, n),
            "lat0": np.full(n, 45.0),
            "lon1": np.linspace(-4.0, 6.0, n),
            "lat1": np.full(n, 46.0),
        }
    )


# ── _group_into_windows ───────────────────────────────────────────────────────


class TestGroupIntoWindows:
    def test_single_timestamp_window(self):
        ts = np.datetime64("2026-01-15T12:00")
        windows = _bare()._group_into_windows([ts])
        assert len(windows) == 1
        start, end, ts_list = windows[0]
        assert ts in ts_list
        assert start < end == "2026-01-15T12:01:00Z"

    def test_window_boundaries(self):
        def windows(*times):
            return _bare()._group_into_windows([np.datetime64(f"2026-01-15T{t}") for t in times])

        assert windows() == []

        grouped = windows("12:00", "12:30")
        assert len(grouped) == 1
        assert len(grouped[0][2]) == 2

        # The window is open-ended: ts2 - ts1 == 1h is not < 1h
        assert len(windows("12:00", "13:00")) == 2
        assert len(windows("12:00", "13:01")) == 2
        assert len(windows("10:00", "12:00", "14:00")) == 3
        assert len(windows("12:30", "12:00")) == 1


# ── _validate_request_timestamp ───────────────────────────────────────────────


class TestValidateRequestTimestamp:
    @pytest.mark.parametrize(
        "satellite, timestamp",
        [
            ("GOES-16", "2023-12-31"),
            ("GOES-18", "2023-12-31"),
            ("HIMAWARI-9", "2023-06-01"),
            ("MTG", "2025-01-01"),
        ],
    )
    def test_before_api_start_raises(self, satellite, timestamp):
        obj = _bare(satellite)
        obj.detector.timestamp_sequence = [np.datetime64(timestamp)]
        with pytest.raises(ValueError, match="data availability"):
            obj._validate_request_timestamp()

    @pytest.mark.parametrize(
        "satellite, timestamp",
        [("GOES-16", "2024-02-01"), ("GOES-18", "2024-02-01"), ("MTG", "2025-02-01")],
    )
    def test_after_api_start_passes(self, satellite, timestamp):
        obj = _bare(satellite)
        obj.detector.timestamp_sequence = [np.datetime64(timestamp)]
        obj._validate_request_timestamp()

    def test_goes18_warns_about_partial_coverage(self, caplog):
        obj = _bare("GOES-18")
        obj.detector.timestamp_sequence = [np.datetime64("2026-02-01T00:00")]
        with caplog.at_level(logging.WARNING):
            obj._validate_request_timestamp()
        assert "partial coverage only" in caplog.text.lower()


# ── _validate_detector ────────────────────────────────────────────────────────


class TestValidateDetector:
    def test_guard_clauses_raise(self):
        from contrail_detect.geostationary.abc import Geostationary

        obj = _bare()
        obj.detector = MagicMock(spec=[])  # not a Geostationary instance
        with pytest.raises(TypeError, match="Geostationary"):
            obj._validate_detector()

        # Use a Geostationary instance but strip a required attribute
        detector = MagicMock(spec=Geostationary)
        del detector.timestamp_strings
        obj.detector = detector
        with pytest.raises(AttributeError, match="timestamp_strings"):
            obj._validate_detector()

        detector = MagicMock(spec=Geostationary)
        for attr in [
            "DETECT_PATH",
            "satellite",
            "region",
            "tile_extent",
            "lons",
            "lats",
            "timestamp_strings",
            "image_geo_coords",
        ]:
            setattr(detector, attr, MagicMock())
        detector.timestamp_strings = []
        obj.detector = detector
        with pytest.raises(ValueError, match="timestamp_strings"):
            obj._validate_detector()


# ── _fill_missing_detections_with_nans ────────────────────────────────────────


class TestFillMissingDetectionsWithNans:
    def test_none_entries_become_nan_detector_grids(self):
        obj = _bare()
        obj.xx = obj.yy = np.zeros((4, 6))  # API grid; the detector's is 8 x 10
        ts1, ts2 = np.datetime64("2026-01-15T12:00"), np.datetime64("2026-01-15T12:10")
        result = obj._fill_missing_detections_with_nans({ts1: None, ts2: None})
        for ts in (ts1, ts2):
            assert result[ts].shape == (8, 10)
            assert np.all(np.isnan(result[ts]))

    def test_lazy_detector_without_grid_uses_api_grid(self):
        obj = _bare()
        obj.detector.image_geo_coords = None
        obj.xx = obj.yy = np.zeros((4, 6))
        ts = np.datetime64("2026-01-15T12:00")
        assert obj._fill_missing_detections_with_nans({ts: None})[ts].shape == (4, 6)


# ── run_segments (mocked HTTP) ────────────────────────────────────────────────


def _geojson_response(n_features: int = 2) -> dict:
    return {
        "features": [
            {
                "geometry": {"coordinates": [[-5.0 + i, 45.0], [-4.0 + i, 46.0]]},
                "properties": {"time": "2026-01-15T12:00:00Z"},
            }
            for i in range(n_features)
        ]
    }


class TestRunSegments:
    def _obj(self) -> GoogleAPIContrailModel:
        obj = _bare()
        obj.detector.timestamp_sequence = [np.datetime64("2026-01-15T12:00")]
        obj.detector.scan_time_offset_for_position = MagicMock(return_value=pd.Timedelta(seconds=0))
        return obj

    @patch("requests.get")
    def test_empty_features_returns_empty_dataframe(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"features": []}
        mock_get.return_value.raise_for_status = MagicMock()

        result = self._obj().run_segments()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("requests.get")
    def test_request_and_parse(self, mock_get):
        from contrail_detect.models.google_api import _GOOGLE_SEGMENTS_URL

        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = _geojson_response(3)
        mock_get.return_value.raise_for_status = MagicMock()

        obj = self._obj()
        bounds = [(45.0, -5.0), (46.0, -4.0), (45.5, -3.0)]
        result = obj.run_segments(bounds=bounds)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert mock_get.call_args[0][0] == _GOOGLE_SEGMENTS_URL
        params = mock_get.call_args[1]["params"]
        assert params["satellite_origins"] == obj.satellite_origin
        assert "bounds" in params

    def test_bounds_fewer_than_3_vertices_raises(self):
        with pytest.raises(ValueError, match="3 vertices"):
            self._obj().run_segments(bounds=[(45.0, -5.0), (46.0, -4.0)])


# ── Satellite origin map ──────────────────────────────────────────────────────


def test_satellite_origin_map():
    # GOES-East and GOES-West are the API's only GOES origins
    assert _SATELLITE_ORIGIN_MAP == {
        "GOES-16": "GOES-EAST-FULL-DISK",
        "GOES-19": "GOES-EAST-FULL-DISK",
        "GOES-18": "GOES-WEST-FULL-DISK",
        "HIMAWARI-9": "HIMAWARI-FULL-DISK",
        "MTG": "MTG-000-FULL-DISK",
    }


# ── _populate_lazy_grid ───────────────────────────────────────────────────────


@pytest.fixture()
def mtg_detector(monkeypatch):
    """MeteosatDetector with an empty grid, as on a clean install (no projection cache)."""
    from contrail_detect.geostationary.meteosat import MeteosatDetector

    monkeypatch.setattr(MeteosatDetector, "_load_projection_cache", lambda self: None)
    monkeypatch.setattr(MeteosatDetector, "_save_projection_cache", lambda self: None)
    t = np.datetime64("2025-09-01T12:00")
    return MeteosatDetector("MTG", "F", t, t)


class TestPopulateLazyGrid:
    def test_mtg_grid_filled_from_coordinates(self, mtg_detector):
        model = _bare("MTG")
        model.detector = mtg_detector
        assert mtg_detector.image_geo_coords is None

        x = np.linspace(-5.0e6, 5.0e6, 51)
        model._populate_lazy_grid(x, x[::-1])

        assert mtg_detector.lons.shape == (51, 51)
        # MTG sits over 0°E: the centre pixel is on the equator at the prime meridian
        assert abs(mtg_detector.lons[25, 25]) < 1e-6
        assert abs(mtg_detector.lats[25, 25]) < 1e-6
        assert mtg_detector.tile_extent is not None

        first = mtg_detector.image_geo_coords

        model._populate_lazy_grid(x[:10], x[:10])
        assert mtg_detector.image_geo_coords is first


# ── get_with_retries ──────────────────────────────────────────────────────────


def _http_error(status: int) -> requests.HTTPError:
    return requests.HTTPError(response=MagicMock(status_code=status))


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("contrail_detect.models.contrail_model_base.time.sleep", sleeps.append)
    return sleeps


class TestGetWithRetries:
    @patch("requests.get")
    def test_retries_5xx_then_succeeds(self, mock_get, no_sleep):
        ok = MagicMock()
        mock_get.side_effect = [_http_error(503), requests.Timeout(), ok]
        assert get_with_retries("u", max_retries=3, timeout=1.0) is ok
        assert no_sleep == [1, 2]  # exponential backoff

    @patch("requests.get")
    def test_gives_up_after_max_retries(self, mock_get, no_sleep):
        mock_get.side_effect = _http_error(500)
        with pytest.raises(requests.HTTPError):
            get_with_retries("u", max_retries=2, timeout=1.0)
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_4xx_raises_without_retry(self, mock_get, no_sleep):
        mock_get.side_effect = _http_error(400)
        with pytest.raises(requests.HTTPError):
            get_with_retries("u", max_retries=3, timeout=1.0)
        assert mock_get.call_count == 1


# ── _save_detection_to_disk ───────────────────────────────────────────────────


def test_save_detection_to_disk(tmp_path) -> None:
    model = object.__new__(ContrailsAPIContrailModel)
    model.detection_output_dir = tmp_path
    model.detector = SimpleNamespace(
        timestamp_strings=["20250601T1200"], satellite="G16", region="F", tile_extent=(0, 1, 2, 3)
    )
    mask = np.eye(3)

    model._save_detection_to_disk(0, mask, source="test")

    frame_dir = tmp_path / "G16-F-20250601T1200"
    np.testing.assert_array_equal(np.load(frame_dir / "_prediction.npy"), mask)
    metadata = json.loads((frame_dir / "_metadata.json").read_text())
    assert metadata["extent"] == [0, 1, 2, 3]
    assert (metadata["satellite"], metadata["region"]) == ("G16", "F")
    assert metadata["timestamp"] == "20250601T1200"
    assert metadata["source"] == "test"


# ── _detections_to_zarr epoch guard ───────────────────────────────────────────


class _ProbeModel(ContrailModelBase):
    """Minimal concrete model, only to reach the epoch guard."""

    def __init__(self, detector):
        self.detector = detector
        self.detection_output_dir = None

    def run_detections(self):
        """Unused."""


def _guard_detector(spans_single_epoch, has_shifting_domain=True):
    """Stub with only what the guard reads, so the write fails just after it, on tile_extent."""
    return SimpleNamespace(
        satellite="GOES-16",
        region="M1" if has_shifting_domain else "F",
        timestamp_sequence=[np.datetime64("2024-08-28T02:30")],
        has_shifting_domain=has_shifting_domain,
        spans_single_epoch=spans_single_epoch,
    )


class TestDetectionsToZarrEpochGuard:
    """Writing a range that spans a repositioning must fail rather than mislabel."""

    def test_refuses_range_spanning_epochs(self):
        detector = _guard_detector(lambda: False)
        with pytest.raises(ValueError, match="repositioned.*one archive per epoch"):
            _ProbeModel(detector)._detections_to_zarr()

    def test_single_epoch_range_passes_the_guard(self):
        """Fails later, on the stub's missing zarr path, rather than on the epoch check."""
        detector = _guard_detector(lambda: True)
        with pytest.raises(AttributeError):
            _ProbeModel(detector)._detections_to_zarr()

    def test_fixed_region_skips_the_guard(self):
        def _explode():
            raise AssertionError("fixed region must not run the epoch check")

        detector = _guard_detector(_explode, has_shifting_domain=False)
        with pytest.raises(AttributeError, match="tile_extent"):
            _ProbeModel(detector)._detections_to_zarr()


# ── ContrailsAPIContrailModel ─────────────────────────────────────────────────


class TestContrailsAPIContrailModel:
    @patch("requests.get")
    def test_cached_frames_are_not_requested(self, mock_get, contrails_org_run):
        model, detector, zarr_path = contrails_org_run
        _cache_frame(model, detector, 0, np.zeros((8, 10)))
        mock_get.return_value = _co_response(np.ones((8, 10)))

        detections = model.run_detections(zarr_path=zarr_path)

        assert mock_get.call_count == 2
        assert [c.args[0][-8:] for c in mock_get.call_args_list] == ["12:10:00", "12:20:00"]
        assert np.all(detections[np.datetime64(_CO_TIMES[0])] == 0)

    @patch("requests.get")
    def test_force_reprocess_ignores_cache(self, mock_get, contrails_org_run):
        model, detector, zarr_path = contrails_org_run
        _cache_frame(model, detector, 0, np.zeros((8, 10)))
        mock_get.return_value = _co_response(np.ones((8, 10)))
        model.run_detections(force_reprocess=True, zarr_path=zarr_path)
        assert mock_get.call_count == 3

    @patch("requests.get")
    def test_server_error_propagates(self, mock_get, contrails_org_run):
        model, _, zarr_path = contrails_org_run
        mock_get.side_effect = _http_error(500)
        with pytest.raises(requests.HTTPError):
            model.run_detections(zarr_path=zarr_path)

    @pytest.mark.parametrize("satellite, region", [("GOES-18", "F"), ("GOES-16", "C")])
    def test_rejects_unsupported_detector(self, satellite, region, tmp_path):
        detector = SimpleNamespace(satellite=satellite, region=region)
        with pytest.raises(ValueError, match="GOES-16 or GOES-19|Full Disk"):
            ContrailsAPIContrailModel(detector, detection_output_dir=tmp_path)

    @patch("requests.get")
    def test_off_schedule_times_refused_before_any_request(self, mock_get, contrails_org_run):
        """The API answers any time without saying which frame it sent, so refuse first."""
        model, detector, zarr_path = contrails_org_run
        detector.timestamp_sequence = [
            np.datetime64("2026-01-09T15:05"),
            np.datetime64("2026-01-09T15:15"),
        ]
        with pytest.raises(ValueError, match="2026-01-09T15:05:00, 2026-01-09T15:15:00"):
            model.run_detections(zarr_path=zarr_path)
        mock_get.assert_not_called()


# ── API frame schedule ────────────────────────────────────────────────────────


class TestRequireApiFrameTimes:
    @pytest.mark.parametrize(
        "timestamp", ["2026-01-09T15:05", "2026-01-09T15:13", "2026-01-09T15:10:30"]
    )
    def test_off_schedule_time_raises(self, timestamp):
        with pytest.raises(ValueError, match="no frame of their own"):
            require_api_frame_times([np.datetime64(timestamp)], "test API")

    def test_error_names_the_times_and_an_aligned_start(self):
        with pytest.raises(ValueError) as err:
            require_api_frame_times([np.datetime64("2026-01-09T15:05")], "test API")
        message = str(err.value)
        assert "test API" in message
        assert "2026-01-09T15:05:00" in message
        assert "e.g. 2026-01-09T15:00 or 2026-01-09T15:10" in message

        times = [np.datetime64("2026-01-09T15:05") + np.timedelta64(10 * i, "m") for i in range(8)]
        with pytest.raises(ValueError, match="and 3 more"):
            require_api_frame_times(times, "test API")


def test_google_off_schedule_times_refused_before_any_request():
    model = _bare()
    model.detector.timestamp_sequence = [np.datetime64("2026-01-15T12:05")]
    with patch.object(model, "_load_cached_detections") as load:
        with pytest.raises(ValueError, match="Google Contrails API"):
            model.run_detections()
    load.assert_not_called()


class TestGoogleFrameMatching:
    """A stored Google mask must be the frame at its own timestamp, never a neighbour."""

    def _extract(self, response_times, requested, tmp_path):
        frames = np.stack([np.full((8, 10), float(i + 1)) for i in range(len(response_times))])
        model = _google_model(tmp_path)
        model._save_detection_to_disk = MagicMock()
        detections: dict = {}
        ds = _mask_ds(frames, _GEO_X, _GEO_Y, times=response_times)
        model._extract_and_cache(ds, requested, requested, detections)
        return detections, model

    def test_exact_frame_is_stored(self, tmp_path):
        requested = [np.datetime64("2026-01-15T12:10")]
        out, _ = self._extract(["2026-01-15T12:00", "2026-01-15T12:10"], requested, tmp_path)
        assert np.all(out[requested[0]] == 2.0)

    def test_frame_seconds_are_ignored(self, tmp_path):
        """A frame stamped 12:10:20 is the 12:10 frame."""
        requested = [np.datetime64("2026-01-15T12:10")]
        out, _ = self._extract(["2026-01-15T12:10:20"], requested, tmp_path)
        assert np.all(out[requested[0]] == 1.0)

    def test_neighbouring_frame_is_not_substituted(self, tmp_path, caplog):
        requested = [np.datetime64("2026-01-15T12:10")]
        with caplog.at_level(logging.WARNING):
            out, model = self._extract(["2026-01-15T12:00"], requested, tmp_path)
        assert out[requested[0]] is None
        assert "closest is 2026-01-15T12:00" in caplog.text
        model._save_detection_to_disk.assert_not_called()


@pytest.mark.integration
def test_contrails_api_detections_endpoint() -> None:
    resp = requests.get(
        CONTRAILS_API_URL_DETECTIONS,
        headers=CONTRAILS_API_HEADERS,
        params={"time": "2026-01-01T00:10:00"},
        timeout=120,
    )
    resp.raise_for_status()
    da = xr.open_dataarray(io.BytesIO(resp.content)).load()
    assert {"x", "y"} <= set(da.dims)


@pytest.mark.parametrize(
    "satellite, region, message",
    [
        ("MTG", "R", "Rapid Scan"),
        ("MTG-I1", "F", "Invalid satellite"),
        ("MTG", "X", "Invalid region"),
    ],
)
def test_meteosat_rejects_unsupported_names(satellite, region, message):
    t = np.datetime64("2025-09-01T12:00")
    with pytest.raises(ValueError, match=message):
        from contrail_detect.geostationary.meteosat import MeteosatDetector

        MeteosatDetector(satellite, region, t, t)


# ── Detector grid ─────────────────────────────────────────────────────────────

_GEO_X = np.arange(10.0) * 2000.0
_GEO_Y = (np.arange(8.0) * 2000.0)[::-1].copy()  # descending, as GOES


# ── Google API: masks stored on the detector grid ─────────────────────────────


def _google_model(tmp_path, detector=None) -> GoogleAPIContrailModel:
    model = object.__new__(GoogleAPIContrailModel)
    model.detector = detector or SimpleNamespace(
        image_geo_coords=(_GEO_X, _GEO_Y), satellite="GOES-19", region="F"
    )
    model.grid_path = tmp_path
    model.detection_output_dir = tmp_path / "google_api"
    model.xx = model.yy = None
    model.timeout, model.max_retries = 1.0, 1
    model.satellite_origin = "GOES-EAST-FULL-DISK"
    model.contrail_model_name = "Google Contrails API"
    return model


def _mask_ds(values, x, y, times=("2025-06-01T12:00",), dims=("time", "y", "x")):
    coords = {"time": pd.to_datetime(list(times)), "x": x, "y": y}
    return xr.Dataset({"mask": (dims, values)}, coords=coords)


_REQ = [np.datetime64("2025-06-01T12:00")]


class TestGoogleDetectorGrid:
    def _extract(self, tmp_path, ds):
        model = _google_model(tmp_path)
        model._save_detection_to_disk = MagicMock()
        out: dict = {}
        model._extract_and_cache(ds, _REQ, _REQ, out)
        return out[_REQ[0]], model

    def test_same_grid_stored_unchanged(self, tmp_path):
        values = np.random.default_rng(0).random((1, 8, 10)).astype(np.float32)
        stored, model = self._extract(tmp_path, _mask_ds(values, _GEO_X, _GEO_Y))
        assert stored.dtype == values.dtype
        np.testing.assert_array_equal(stored, values[0])
        np.testing.assert_array_equal(model._save_detection_to_disk.call_args.args[1], values[0])

        height = 35786023.0
        values = np.random.default_rng(1).random((1, 8, 10))
        ds = _mask_ds(values, _GEO_X / height, _GEO_Y / height)
        stored, _ = self._extract(tmp_path, ds)
        np.testing.assert_array_equal(stored, values[0])

        values = np.random.default_rng(2).random((1, 8, 10))
        ds = _mask_ds(values.transpose(0, 2, 1), _GEO_X, _GEO_Y, dims=("time", "x", "y"))
        stored, _ = self._extract(tmp_path, ds)
        np.testing.assert_array_equal(stored, values[0])

    def test_ascending_y_is_flipped(self, tmp_path):
        rows = np.add.outer(np.arange(8.0), np.zeros(10))  # value = row index
        stored, _ = self._extract(tmp_path, _mask_ds(rows[None], _GEO_X, _GEO_Y[::-1]))
        # Response row 7 is at the northmost y, which is the detector's row 0
        np.testing.assert_array_equal(stored[:, 0], np.arange(8.0)[::-1])

    def test_response_on_another_grid_is_regridded(self, tmp_path):
        x, y = _GEO_X[2:7], _GEO_Y[1:5]
        ds = _mask_ds(np.ones((1, 4, 5)), x, y)
        stored, _ = self._extract(tmp_path, ds)
        assert stored.shape == (8, 10)
        np.testing.assert_array_equal(stored[1:5, 2:7], 1.0)
        assert np.isnan(stored[0]).all() and np.isnan(stored[:, 0]).all()

    def test_regridded_response_keeps_north_up(self, tmp_path):
        # Half-pixel shifted grid, y ascending: values increase northwards
        x, y = _GEO_X + 1000.0, (_GEO_Y + 1000.0)[::-1]
        north = np.add.outer(y, np.zeros(10))
        stored, _ = self._extract(tmp_path, _mask_ds(north[None], x, y))
        column = stored[:, 5]
        finite = column[np.isfinite(column)]
        assert len(finite) > 2 and np.all(np.diff(finite) < 0)  # north (row 0) largest

    def test_response_without_time_is_refused(self, tmp_path):
        ds = xr.Dataset({"mask": (("y", "x"), np.ones((8, 10)))}, coords={"x": _GEO_X, "y": _GEO_Y})
        with pytest.raises(ValueError, match="no time"):
            self._extract(tmp_path, ds)

    def test_scalar_time_coordinate(self, tmp_path):
        ds = xr.Dataset(
            {"mask": (("y", "x"), np.full((8, 10), 0.5))},
            coords={"x": _GEO_X, "y": _GEO_Y, "time": pd.Timestamp("2025-06-01T12:00")},
        )
        stored, _ = self._extract(tmp_path, ds)
        np.testing.assert_array_equal(stored, 0.5)


# ── Google API: failed requests are not persisted ─────────────────────────────

_G_TIMES = ["2025-06-01T12:00", "2025-06-01T13:00"]


def _nc_response(ds: xr.Dataset) -> MagicMock:
    return MagicMock(content=bytes(ds.to_netcdf(engine="h5netcdf")))


def _frame_response(time: str, value: float = 0.9) -> MagicMock:
    return _nc_response(_mask_ds(np.full((1, 8, 10), value), _GEO_X, _GEO_Y, times=(time,)))


@pytest.fixture
def google_run(tmp_path, geo_detector):
    detector = geo_detector(tmp_path, _G_TIMES, "google_api", _GEO_X, _GEO_Y)
    return _google_model(tmp_path, detector), detector, tmp_path / "zarr"


def _archived_times(detector, zarr_path):
    ds = xr.open_zarr(detector.get_zarr_path(zarr_path))
    return [str(t)[:16] for t in ds.time.values]


@patch("requests.get")
def test_archive_records_scan_starts_and_version(mock_get, google_run):
    """Each frame's scan start goes into the archive, so times can be rebuilt from it alone."""
    import contrail_detect

    model, detector, zarr_path = google_run
    detector.scan_start_offsets_s = lambda ts: np.full(len(ts), 21.22)
    mock_get.side_effect = [_frame_response(t) for t in _G_TIMES]
    model.run_detections(zarr_path=zarr_path)
    ds = xr.open_zarr(detector.get_zarr_path(zarr_path))
    np.testing.assert_allclose(ds["scan_start_offset"].values, [21.22, 21.22])
    assert ds["scan_start_offset"].dims == ("time",)
    assert ds.attrs["contrail_detect_version"] == contrail_detect.__version__


@patch("requests.get")
def test_mesoscale_archive_holds_its_grid_fields(mock_get, google_run):
    """No shared store for a shifting domain, so the archive must carry scantime_offset."""
    model, detector, zarr_path = google_run
    detector.has_shifting_domain = True
    detector.spans_single_epoch = lambda: True
    grid = np.full((len(_GEO_Y), len(_GEO_X)), 3.5)
    detector.static_geometry_fields = lambda: {"scantime_offset": (("geo_y", "geo_x"), grid)}
    detector.scan_start_offsets_s = lambda ts: np.full(len(ts), 26.02)
    mock_get.side_effect = [_frame_response(t) for t in _G_TIMES]
    model.run_detections(zarr_path=zarr_path)
    ds = xr.open_zarr(detector.get_zarr_path(zarr_path))
    np.testing.assert_array_equal(ds["scantime_offset"].values, grid)
    np.testing.assert_allclose(ds["scan_start_offset"].values, 26.02)


class TestGoogleFailedRequestsNotPersisted:
    @pytest.mark.parametrize(
        "failure", [requests.Timeout("simulated"), _http_error(503), requests.ConnectionError()]
    )
    @patch("requests.get")
    def test_failed_window_is_requested_again(self, mock_get, google_run, failure):
        model, detector, zarr_path = google_run
        mock_get.side_effect = [_frame_response(_G_TIMES[0]), failure]
        first = model.run_detections(zarr_path=zarr_path)

        assert np.all(first[np.datetime64(_G_TIMES[0])] == 0.9)
        assert np.isnan(first[np.datetime64(_G_TIMES[1])]).all()
        assert first[np.datetime64(_G_TIMES[1])].shape == (8, 10)
        assert _archived_times(detector, zarr_path) == ["2025-06-01T12:00"]

        model.xx = model.yy = None
        mock_get.reset_mock(side_effect=True)
        mock_get.side_effect = [_frame_response(_G_TIMES[1], 0.7)]
        second = model.run_detections(zarr_path=zarr_path)

        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["start_time"] == "2025-06-01T13:00:00Z"
        assert np.all(second[np.datetime64(_G_TIMES[1])] == pytest.approx(0.7))
        assert _archived_times(detector, zarr_path) == ["2025-06-01T12:00", "2025-06-01T13:00"]

    @patch("requests.get")
    def test_unreadable_response_is_requested_again(self, mock_get, google_run):
        model, detector, zarr_path = google_run
        mock_get.side_effect = [_frame_response(_G_TIMES[0]), MagicMock(content=b"not netcdf")]
        model.run_detections(zarr_path=zarr_path)
        assert _archived_times(detector, zarr_path) == ["2025-06-01T12:00"]

    @patch("requests.get")
    def test_no_data_answer_is_archived(self, mock_get, google_run):
        model, detector, zarr_path = google_run
        mock_get.side_effect = [
            _frame_response(_G_TIMES[0]),
            _frame_response("2025-06-01T13:10"),  # no 13:00 frame
        ]
        first = model.run_detections(zarr_path=zarr_path)
        assert np.isnan(first[np.datetime64(_G_TIMES[1])]).all()
        assert _archived_times(detector, zarr_path) == ["2025-06-01T12:00", "2025-06-01T13:00"]

        model.xx = model.yy = None
        mock_get.reset_mock(side_effect=True)
        second = model.run_detections(zarr_path=zarr_path)
        mock_get.assert_not_called()
        assert np.isnan(second[np.datetime64(_G_TIMES[1])]).all()


# ── ContrailsAPIContrailModel.run_detections ──────────────────────────────────

_CO_TIMES = ["2025-06-01T12:00", "2025-06-01T12:10", "2025-06-01T12:20"]


@pytest.fixture
def contrails_org_run(tmp_path, geo_detector):
    detector = geo_detector(tmp_path, _CO_TIMES, "contrails_api", _GEO_X, _GEO_Y)
    model = object.__new__(ContrailsAPIContrailModel)
    model.detector = detector
    model.detection_output_dir = tmp_path / "contrails_api"
    model.grid_path = tmp_path / "contrails_api" / "GRID"
    model.grid_path.mkdir(parents=True)
    model.xx = model.yy = None
    model.timeout, model.max_retries = 1.0, 1
    model.contrail_model_name = "Contrails.org API"
    return model, detector, tmp_path / "zarr"


def _co_response(values, x=_GEO_X, y=_GEO_Y) -> MagicMock:
    da = xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name="mask")
    return MagicMock(content=da.to_netcdf())


def _cache_frame(model, detector, idx, values) -> None:
    path = detector.get_npy_path(idx, model.detection_output_dir)
    path.parent.mkdir(parents=True)
    np.save(path, values)


class TestContrailsOrgRun:
    @patch("requests.get")
    def test_full_disk_run_output(self, mock_get, contrails_org_run):
        model, detector, zarr_path = contrails_org_run
        frame0 = np.linspace(0.0, 1.0, 80).reshape(8, 10)
        mock_get.side_effect = [
            _co_response(frame0),
            _http_error(400),
            _co_response(frame0[:4], y=_GEO_Y[:4]),
        ]
        out = model.run_detections(zarr_path=zarr_path)

        assert mock_get.call_count == 3
        url = mock_get.call_args_list[1].args[0]
        assert url == f"{CONTRAILS_API_URL_DETECTIONS}?time=2025-06-01T12:10:00"
        assert mock_get.call_args.kwargs["headers"] == CONTRAILS_API_HEADERS
        t0, t1, t2 = (np.datetime64(t) for t in _CO_TIMES)
        np.testing.assert_array_equal(out[t0], frame0)
        assert out[t1].shape == (8, 10) and np.isnan(out[t1]).all()
        # The third response covers only the top four rows
        assert out[t2].shape == (8, 10) and out[t2].dtype == np.float32
        np.testing.assert_allclose(out[t2][:4], frame0[:4])
        assert np.isnan(out[t2][4:]).all()
        assert model.xx.shape == (8, 10)

        ds = xr.open_zarr(detector.get_zarr_path(zarr_path))
        assert [str(t)[:16] for t in ds.time.values] == _CO_TIMES
        stored0 = np.where(frame0 < 0.01, 0.0, frame0)
        np.testing.assert_allclose(ds.detection.isel(time=0).values, stored0, atol=1e-3)
        assert np.isnan(ds.detection.isel(time=1).values).all()
        assert tuple(ds.attrs["extent"]) == (0.0, 1.0, 0.0, 1.0)
        assert ds.attrs["source"] == "Contrails.org API model"
        np.testing.assert_array_equal(ds.geo_y.values, _GEO_Y)

        meta = json.loads(
            (
                model.detection_output_dir / "GOES-19-F-2025-06-01T1200" / "_metadata.json"
            ).read_text()
        )
        assert meta["extent"] == [0.0, 1.0, 0.0, 1.0] and "source" not in meta

    @patch("requests.get")
    def test_second_run_reads_the_archive(self, mock_get, contrails_org_run):
        model, _, zarr_path = contrails_org_run
        frame0 = np.linspace(0.0, 1.0, 80).reshape(8, 10)
        mock_get.side_effect = [_co_response(frame0), _http_error(400), _co_response(frame0)]
        model.run_detections(zarr_path=zarr_path)
        mock_get.reset_mock(side_effect=True)
        out = model.run_detections(zarr_path=zarr_path)
        mock_get.assert_not_called()
        assert np.isnan(out[np.datetime64(_CO_TIMES[1])]).all()


# ── _refactor_segments: WGS84 geodesic ────────────────────────────────────────


class TestRefactorSegmentsGeodesic:
    def _refactor(self, df):
        model = object.__new__(_ProbeModel)
        model.detector = SimpleNamespace(
            scan_time_offset_for_position=lambda lat, lon, timestamp=None: pd.Timedelta(0)
        )
        return model._refactor_segments(df)

    def test_length_and_orientation_match_wgs84_geodesic(self):
        from pyproj import Geod

        df = _segments_df(3)
        out = self._refactor(df)
        az, _, dist = Geod(ellps="WGS84").inv(df.lon0, df.lat0, df.lon1, df.lat1)
        np.testing.assert_allclose(out.great_circle_length_km, np.asarray(dist) / 1000.0)
        np.testing.assert_allclose(out.orientation, az)

        geod = Geod(ellps="WGS84")
        out = self._refactor(_segments_df(1))
        _, _, d1 = geod.inv(-5.0, 45.0, out.centroid_lon[0], out.centroid_lat[0])
        _, _, d2 = geod.inv(out.centroid_lon[0], out.centroid_lat[0], -4.0, 46.0)
        assert d1 == pytest.approx(d2, abs=1e-3)

    def test_centroid_longitude_wraps_at_dateline(self):
        df = pd.DataFrame(
            {
                "time": [pd.Timestamp("2026-01-15T12:00")],
                "lon0": [179.0],
                "lat0": [10.0],
                "lon1": [-178.0],
                "lat1": [10.0],
            }
        )
        out = self._refactor(df)
        assert -180.0 <= out.centroid_lon[0] < 180.0
        assert out.centroid_lon[0] == pytest.approx(-179.5, abs=0.01)
        assert out.great_circle_length_km[0] == pytest.approx(329.0, abs=1.0)


# ── get_detections('load', detection_output_dir=...) ────────────────────────────────


class TestLoadReadsWriterLayout:
    @pytest.fixture
    def det(self, tmp_path, geo_detector):
        from contrail_detect.geostationary.goes import GOESDetector

        det = object.__new__(GOESDetector)
        stub = geo_detector(tmp_path, _CO_TIMES[:1], "contrails_api", _GEO_X, _GEO_Y)
        for name in [
            "satellite",
            "region",
            "timestamp_sequence",
            "timestamp_strings",
            "tile_extent",
            "detection_subdirectory",
            "LONG_NAME",
        ]:
            setattr(det, name, getattr(stub, name))
        return det

    def test_custom_dir_reads_source_subdirectory(self, tmp_path, det):
        written = tmp_path / "out" / "contrails_api" / "GOES-19-F-2025-06-01T1200"
        written.mkdir(parents=True)
        np.save(written / "_prediction.npy", np.full((8, 10), 0.4))
        out = det.get_detections(
            "load", zarr_path=tmp_path / "z", detection_output_dir=tmp_path / "out"
        )
        np.testing.assert_array_equal(out[np.datetime64(_CO_TIMES[0])], 0.4)

    def test_default_dir_is_passed_through_unchanged(self, tmp_path, det):
        with patch.object(type(det), "load_detections", return_value={}) as load:
            det.get_detections("load", zarr_path=tmp_path / "z")
        load.assert_called_once_with(zarr_path=tmp_path / "z", detection_output_dir=None)

    def test_meteosat_custom_dir_reads_google_api(self, tmp_path):
        from contrail_detect.geostationary.meteosat import MeteosatDetector

        det = object.__new__(MeteosatDetector)
        with patch.object(MeteosatDetector, "load_detections", return_value={}) as load:
            det.get_detections("load", detection_output_dir=tmp_path / "out")
        assert load.call_args.kwargs["detection_output_dir"] == tmp_path / "out" / "google_api"
