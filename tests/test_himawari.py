"""Unit tests for the Himawari AHI detector and the shared ABI local model staging.

Tests cover:
  - The AHI fixed grid and CRS, pinned against satpy's ahi_hsd navigation (pycontrails
    0.63 is one pixel off and uses the wrong sweep axis, so it is not the reference)
  - _gen_scan_time_offset: per-segment scan timing, and its fallbacks
  - _frame_projection: frame shape checks
  - ABIDetector._process_single_timestamp: skipping frames with no data
"""

import logging

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from contrail_detect.geostationary.himawari import HimawariDetector

_NOMINAL = np.datetime64("2025-06-01T03:00")

# satpy ahi_hsd area for the same data: pixel centres at ±5498999.9 m, sweep y (PROJ default)
_SATPY_CRS = "+proj=geos +lon_0=140.7 +h=35785863 +a=6378137 +rf=298.257024882273 +units=m"
_SATPY_X_END = 5498999.901192693


def _bare_himawari(n_cols: int = 2) -> HimawariDetector:
    """HimawariDetector with no network access and a grid of 5500 rows by n_cols columns."""
    obj = object.__new__(HimawariDetector)
    obj.satellite = "HIMAWARI-9"
    obj.region = "F"
    obj.sweep_axis = "y"
    obj.central_lon = 140.7
    obj.timestamp_sequence = [_NOMINAL]
    obj.lats = np.zeros((5500, n_cols))
    return obj


def _no_headers(self, time):
    raise ConnectionError("offline")


# ── Fixed grid and CRS ────────────────────────────────────────────────────────


class TestGrid:
    def test_pixel_centres_match_satpy(self):
        x, y = _bare_himawari().image_geo_coords
        assert x.shape == y.shape == (5500,)
        assert x.dtype == y.dtype == np.float64
        np.testing.assert_allclose([x[0], x[-1]], [-_SATPY_X_END, _SATPY_X_END], atol=0.01)
        np.testing.assert_allclose([y[0], y[-1]], [_SATPY_X_END, -_SATPY_X_END], atol=0.01)
        np.testing.assert_allclose(x[2749], -x[2750], atol=1e-6)
        np.testing.assert_allclose(x[2750], 1000.0, atol=0.01)
        np.testing.assert_allclose(y, -x, atol=1e-6)

    def test_crs_uses_sweep_y(self):
        params = _bare_himawari().sat_pyproj_crs.to_dict()
        assert params.get("sweep", "y") == "y"  # PROJ omits its default, y
        assert params["lon_0"] == 140.7
        assert params["h"] == 35785863.0
        cartopy_crs, _ = _bare_himawari().image_transforms
        assert cartopy_crs.proj4_params["sweep"] == "y"

    @pytest.mark.parametrize("i, j", [(1200, 1200), (4300, 1200), (1000, 4500), (2750, 600)])
    def test_navigation_matches_satpy(self, i, j):
        obj = _bare_himawari()
        x, y = obj.image_geo_coords
        lon, lat = obj.sat_to_latlon_transform.transform(x[i], y[j])
        ref = pyproj.Transformer.from_crs(pyproj.CRS(_SATPY_CRS), "EPSG:4326", always_xy=True)
        ref_lon, ref_lat = ref.transform(x[i], y[j])
        distance_m = pyproj.Geod(ellps="WGS84").inv(lon, lat, ref_lon, ref_lat)[2]
        assert distance_m < 5.0

    def test_bucket_follows_satellite(self):
        obj = _bare_himawari()
        obj.satellite = "HIMAWARI-8"
        assert obj.himawari_s3_bucket == "noaa-himawari8"
        obj.satellite = "HIMAWARI-9"
        assert obj.himawari_s3_bucket == "noaa-himawari9"


# ── Scan timing ───────────────────────────────────────────────────────────────


def _offsets_s(obj: HimawariDetector) -> np.ndarray:
    return obj._gen_scan_time_offset()[:, 0] / np.timedelta64(1, "s")


class TestScanTimeOffset:
    def test_fallback_when_headers_unreadable(self, monkeypatch, caplog):
        monkeypatch.setattr(HimawariDetector, "_read_segment_times_s", _no_headers)
        with caplog.at_level(logging.WARNING):
            full = _bare_himawari(n_cols=7)._gen_scan_time_offset()
        assert "tabulated segment times" in caplog.text
        assert full.shape == (5500, 7)
        assert full.dtype == np.dtype("timedelta64[ms]")
        offsets = full[:, 0] / np.timedelta64(1, "s")
        assert np.all(np.diff(offsets) > 0)
        first_start, first_end = HimawariDetector.FALLBACK_SEGMENT_TIMES_S[0]
        assert first_start < offsets[0] < first_start + 0.1
        assert offsets[549] == pytest.approx(first_end, abs=0.1)
        assert offsets[550] == pytest.approx(
            HimawariDetector.FALLBACK_SEGMENT_TIMES_S[1][0], abs=0.1
        )
        assert offsets[-1] == pytest.approx(
            HimawariDetector.FALLBACK_SEGMENT_TIMES_S[-1][1], abs=0.1
        )

    def test_uses_header_times(self, monkeypatch):
        segments = [(k * 550 + 1, 30.0 + 50.0 * k, 80.0 + 50.0 * k) for k in range(10)]
        monkeypatch.setattr(HimawariDetector, "_read_segment_times_s", lambda self, t: segments)
        offsets = _offsets_s(_bare_himawari())
        assert offsets[0] == pytest.approx(30.0 + 50.0 * 0.5 / 550, abs=0.001)
        assert offsets[275] == pytest.approx(55.0, abs=0.1)
        assert offsets[-1] == pytest.approx(530.0, abs=0.1)

    def test_anchors_on_first_timestamp(self, monkeypatch):
        seen = []

        def record(self, time):
            seen.append(time)
            raise ConnectionError("offline")

        monkeypatch.setattr(HimawariDetector, "_read_segment_times_s", record)
        obj = _bare_himawari()
        obj.timestamp_sequence = [_NOMINAL, _NOMINAL + np.timedelta64(10, "m")]
        obj._gen_scan_time_offset()
        assert seen == [pd.Timestamp(_NOMINAL)]

    @pytest.mark.parametrize(
        "segments",
        [
            [(k * 550 + 1, 500.0 - 40.0 * k, 510.0 - 40.0 * k) for k in range(10)],  # backwards
            [(k * 550 + 1, 60.0 * k, 60.0 * k + 900.0) for k in range(10)],  # too long
            [(k * 550 + 551, 30.0 * k, 30.0 * k + 30.0) for k in range(10)],  # missing top
            [(k * 550 + 1, 50.0 * k, 50.0 * k + 50.0) for k in range(10) if k != 4],  # gap
            [(k * 550 + 1, 50.0 * k, 50.0 * k + 50.0) for k in range(9)],  # missing bottom
        ],
    )
    def test_implausible_header_times_fall_back(self, monkeypatch, caplog, segments):
        monkeypatch.setattr(HimawariDetector, "_read_segment_times_s", lambda self, t: segments)
        with caplog.at_level(logging.WARNING):
            offsets = _offsets_s(_bare_himawari())
        assert "implausible" in caplog.text
        assert offsets[-1] == pytest.approx(
            HimawariDetector.FALLBACK_SEGMENT_TIMES_S[-1][1], abs=0.1
        )


# ── Frame hooks ───────────────────────────────────────────────────────────────


class TestFrameProjection:
    def test_himawari_returns_fixed_grid(self):
        obj = _bare_himawari()
        da = xr.DataArray(np.zeros((3, 5500, 5500), dtype=np.float32), dims=("band_id", "y", "x"))
        geo_x, geo_y, crs = obj._frame_projection(da)
        np.testing.assert_array_equal(geo_x, obj.image_geo_coords[0])
        np.testing.assert_array_equal(geo_y, obj.image_geo_coords[1])
        assert crs.to_dict().get("sweep", "y") == "y"

    def test_himawari_rejects_other_grids(self):
        da = xr.DataArray(np.zeros((3, 11000, 11000), dtype=np.float32), dims=("band_id", "y", "x"))
        with pytest.raises(ValueError, match="2 km Full Disk"):
            _bare_himawari()._frame_projection(da)


# ── Shared local model staging ────────────────────────────────────────────────────────


class TestProcessSingleTimestamp:
    def _staging_himawari(self, error: Exception) -> HimawariDetector:
        obj = _bare_himawari()
        obj.projection_cache = {}

        def fail(time):
            raise error

        obj._get_frame = fail
        return obj

    def test_missing_frame_skips_timestamp(self, caplog):
        obj = self._staging_himawari(ValueError("No data found for band B11 at time ..."))
        with caplog.at_level(logging.INFO):
            obj._process_single_timestamp(
                timestamp=_NOMINAL, tiles_to_load=[0], data_cache={}, tile_memory_cache={}
            )
        assert "Skipping timestamp" in caplog.text

    def test_other_errors_propagate(self):
        obj = self._staging_himawari(ValueError("Himawari frame has shape (1, 1)"))
        with pytest.raises(ValueError, match="shape"):
            obj._process_single_timestamp(
                timestamp=_NOMINAL, tiles_to_load=[0], data_cache={}, tile_memory_cache={}
            )
