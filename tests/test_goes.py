"""Unit tests for the GOES ABI detector and the ABIDetector base.

Tests cover:
  - _scan_start_at: a frame's scan start and timeline_id from its band files
  - scan_start_offsets_s: each frame's own start, the fallbacks, and non-Mode 6 frames
  - _gen_scan_time_offset: the table + band correction grid
  - scan_time_offset_for_position: grid + the frame's own start
  - _frame_projection: scan angles to metres
  - _generate_tile_mobile_grid: per-timestamp coordinates for M1/M2, and tile slicing
  - spans_single_epoch: detecting a range that crosses a sector repositioning
"""

import logging
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrail_detect.geostationary import abi_time_model
from contrail_detect.geostationary.goes import GOESDetector

_NOMINAL = np.datetime64("2024-08-28T02:30")
_TABLE_START_S = {"F": 20.89, "C": 77.43, "M1": 28.21, "M2": 55.31}


def _bare_goes(region: str = "F", satellite: str = "GOES-16") -> GOESDetector:
    """GOESDetector with no frame loaded and no network access."""
    obj = object.__new__(GOESDetector)
    obj.satellite = satellite
    obj.region = region
    obj.timestamp_sequence = [_NOMINAL]
    obj.da_goes = None
    return obj


def _with_band_files(tmp_path, region="F", starts_s=None, timelines=None, satellite="GOES-16"):
    """GOESDetector whose cached band files have the given scan starts and timelines."""
    starts_s = starts_s or {"C11": 20.89, "C14": 20.89, "C15": 20.89}
    timelines = timelines or dict.fromkeys(starts_s, "ABI Mode 6")
    lpaths = OrderedDict()
    for band, start_s in starts_s.items():
        path = tmp_path / f"{band}.nc"
        t0 = _NOMINAL + np.timedelta64(round(start_s * 1000), "ms")
        xr.Dataset(
            {"time_bounds": (("number_of_time_bounds",), [t0, t0 + np.timedelta64(570, "s")])},
            attrs={"timeline_id": timelines[band]},
        ).to_netcdf(path)
        lpaths[band] = str(path)
    obj = _bare_goes(region, satellite)
    obj.da_goes = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    obj.pycon_goes = MagicMock()
    obj.pycon_goes._lpaths.return_value = lpaths
    return obj


def _frames(tmp_path, region="M1", frames=None):
    """GOESDetector over several frames, each with cached band files.

    ``frames`` maps minutes after _NOMINAL to (scan start in s, timeline_id); a start of None
    means that frame's files are missing.
    """
    frames = frames or {0: (28.21, "ABI Mode 6"), 1: (25.31, "ABI Mode 6")}
    lpaths = {}
    for minute, (start_s, timeline) in frames.items():
        nominal = _NOMINAL + np.timedelta64(minute, "m")
        paths = {}
        for band in ("C11", "C14", "C15"):
            path = tmp_path / f"{minute}_{band}.nc"
            if start_s is not None:
                t0 = nominal + np.timedelta64(round(start_s * 1000), "ms")
                xr.Dataset(
                    {"time_bounds": (("n",), [t0, t0 + np.timedelta64(5, "s")])},
                    attrs={"timeline_id": timeline},
                ).to_netcdf(path)
            paths[band] = str(path)
        lpaths[pd.Timestamp(nominal)] = paths
    obj = _bare_goes(region)
    obj.timestamp_sequence = [_NOMINAL + np.timedelta64(m, "m") for m in sorted(frames)]
    obj.da_goes = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    obj.pycon_goes = MagicMock()
    obj.pycon_goes._lpaths.side_effect = lambda t: lpaths[pd.Timestamp(t)]
    obj.pycon_goes.cachestore.exists.side_effect = lambda p: Path(p).exists()
    obj.pycon_goes.gcs_goes_path.side_effect = FileNotFoundError("no network in tests")
    table = abi_time_model.load_table(abi_time_model.SCENE_BY_REGION[region])
    obj.lats = np.zeros(table.shape)
    return obj


def _minute(m: int) -> np.datetime64:
    return np.datetime64(_NOMINAL + np.timedelta64(m, "m"), "ns")


# ── _scan_start_at ────────────────────────────────────────────────────────────


class TestScanStartAt:
    def test_reads_earliest_start_and_timeline(self, tmp_path):
        obj = _with_band_files(tmp_path, starts_s={"C11": 20.95, "C14": 20.89, "C15": 20.91})
        start_s, timeline = obj._scan_start_at(_NOMINAL)
        assert start_s == pytest.approx(20.89, abs=1e-3)
        assert timeline == "ABI Mode 6"

    def test_cached_per_frame(self, tmp_path):
        obj = _with_band_files(tmp_path)
        obj._scan_start_at(_NOMINAL)
        obj._scan_start_at(_NOMINAL)
        assert obj.pycon_goes._lpaths.call_count == 1

    def test_unreadable_files(self, tmp_path, caplog):
        obj = _bare_goes("F")
        obj.pycon_goes = MagicMock()
        obj.pycon_goes._lpaths.return_value = {"C11": "/nonexistent/C11.nc"}
        with caplog.at_level(logging.WARNING):
            assert obj._scan_start_at(_NOMINAL) == (None, None)
        assert "couldn't read the files" in caplog.text

    @pytest.mark.parametrize("start_s", [-61.0, 121.0, 600.0])
    def test_implausible_start_is_ignored(self, tmp_path, start_s, caplog):
        obj = _with_band_files(tmp_path, region="M1", starts_s={"C14": start_s})
        with caplog.at_level(logging.WARNING):
            got, timeline = obj._scan_start_at(_NOMINAL)
        assert got is None and timeline == "ABI Mode 6"
        assert "ignoring it" in caplog.text

    def test_disagreeing_timelines_raise(self, tmp_path):
        obj = _with_band_files(
            tmp_path,
            starts_s={"C11": 20.9, "C14": 20.9},
            timelines={"C11": "ABI Mode 6", "C14": "ABI Mode 4"},
        )
        with pytest.raises(ValueError, match="stale files"):
            obj._scan_start_at(_NOMINAL)


def test_uncached_frame_reads_one_remote_band(tmp_path):
    path = tmp_path / "C11.nc"
    t0 = _NOMINAL + np.timedelta64(21220, "ms")
    xr.Dataset(
        {"time_bounds": (("n",), [t0, t0 + np.timedelta64(570, "s")])},
        attrs={"timeline_id": "ABI Mode 6"},
    ).to_netcdf(path, engine="h5netcdf")
    obj = _bare_goes("F")
    obj.pycon_goes = MagicMock()
    obj.pycon_goes.cachestore = None
    obj.pycon_goes.gcs_goes_path.return_value = [str(path)]
    obj.pycon_goes.fs.open.side_effect = lambda p: open(p, "rb")
    assert obj._scan_start_at(pd.Timestamp(_NOMINAL)) == (pytest.approx(21.22), "ABI Mode 6")
    obj.pycon_goes.gcs_goes_path.assert_called_once_with(pd.Timestamp(_NOMINAL), {"C11"})


# ── scan_start_offsets_s ──────────────────────────────────────────────────────


class TestScanStartOffsets:
    def test_each_frame_uses_its_own_start(self, tmp_path):
        # M1: the first frame of each 10-min cycle starts 2.9 s later than the rest
        obj = _frames(tmp_path)
        starts = obj.scan_start_offsets_s(np.array([_minute(0), _minute(1)]))
        np.testing.assert_allclose(starts, [28.21, 25.31], atol=1e-6)

    def test_shape_follows_input(self, tmp_path):
        obj = _frames(tmp_path)
        assert obj.scan_start_offsets_s(np.full((2, 3), _minute(1))).shape == (2, 3)

    def test_unreadable_frame_gets_the_first_frames_start(self, tmp_path, caplog):
        obj = _frames(tmp_path, frames={0: (28.21, "ABI Mode 6"), 1: (None, None)})
        with caplog.at_level(logging.WARNING):
            start = obj.scan_start_offsets_s(np.array([_minute(1)]))[0]
        assert start == pytest.approx(28.21)
        assert "couldn't read the files" in caplog.text

    def test_unreadable_first_frame_falls_back_to_table(self, tmp_path, caplog):
        obj = _frames(tmp_path, frames={0: (None, None), 1: (None, None)})
        with caplog.at_level(logging.WARNING):
            start = obj.scan_start_offsets_s(np.array([_minute(1)]))[0]
        assert start == GOESDetector.SCAN_PARAMETERS["M1"]["overhead_s"]
        assert "no usable scan start for the first frame" in caplog.text

    def test_non_mode_6_frame_is_nan(self, tmp_path, caplog):
        obj = _frames(tmp_path, frames={0: (28.21, "ABI Mode 6"), 1: (25.3, "ABI Mode 4")})
        times = np.array([_minute(0), _minute(1), _minute(1)])
        with caplog.at_level(logging.WARNING):
            starts = obj.scan_start_offsets_s(times)
        assert starts[0] == pytest.approx(28.21)
        assert np.isnan(starts[1:]).all()
        assert caplog.text.count("has no time model") == 1


# ── _gen_scan_time_offset ─────────────────────────────────────────────────────


class TestGenScanTimeOffset:
    @pytest.mark.parametrize("region", ["F", "C", "M1", "M2"])
    def test_table_plus_band_correction(self, region):
        obj = _bare_goes(region)
        table = abi_time_model.load_table(abi_time_model.SCENE_BY_REGION[region])
        obj.lats = np.zeros(table.shape)
        offsets = obj._gen_scan_time_offset()
        assert offsets.dtype == np.dtype("timedelta64[ms]")
        expected = table + abi_time_model.band_correction_s([11, 14, 15])
        np.testing.assert_allclose(offsets.astype(np.int64) / 1000, expected, atol=1e-3)

    def test_needs_no_frame_files(self):
        obj = _bare_goes("F")  # a Mode 3 or offline range can still be built
        obj.lats = np.zeros((5424, 5424))
        obj._gen_scan_time_offset()

    def test_grid_mismatch_raises(self):
        obj = _bare_goes("F")
        obj.lats = np.zeros((5, 3))
        with pytest.raises(ValueError, match="detector grid"):
            obj._gen_scan_time_offset()


# ── scan_time_offset_for_position ─────────────────────────────────────────────


class TestPositionLookup:
    @pytest.fixture
    def obj(self, tmp_path, monkeypatch):
        def make(frames=None):
            o = _frames(tmp_path, frames=frames)
            monkeypatch.setattr(
                o,
                "get_grid_index_from_position",
                lambda lat, lon: (np.zeros(np.shape(lat), int), np.zeros(np.shape(lat), int)),
            )
            return o

        return make

    def test_grid_plus_the_frames_own_start(self, obj):
        o = obj()
        lat = np.array([40.0, 40.0])
        grid_s = float(abi_time_model.load_table("MESO")[0, 0]) - 0.10233
        for minute, start_s in ((0, 28.21), (1, 25.31)):
            got = o.scan_time_offset_for_position(lat, lat, _minute(minute)) / np.timedelta64(
                1, "s"
            )
            np.testing.assert_allclose(got, grid_s + start_s, atol=2e-3)

    def test_one_timestamp_per_position(self, obj):
        o = obj()
        lat = np.array([40.0, 40.0])
        got = o.scan_time_offset_for_position(lat, lat, np.array([_minute(0), _minute(1)]))
        assert (got[0] - got[1]) == np.timedelta64(2900, "ms")

    def test_scalar_in_scalar_out(self, obj):
        o = obj()
        scalar = o.scan_time_offset_for_position(40.0, 40.0, _minute(0))
        array = o.scan_time_offset_for_position(np.array([40.0]), np.array([40.0]), _minute(0))
        assert scalar == array[0]

    def test_no_timestamp_means_the_first_frame(self, obj):
        o = obj()
        lat = np.array([40.0])
        assert o.scan_time_offset_for_position(lat, lat) == o.scan_time_offset_for_position(
            lat, lat, _minute(0)
        )

    def test_non_mode_6_frame_has_no_time(self, obj):
        o = obj(frames={0: (25.3, "ABI Mode 4"), 1: (25.31, "ABI Mode 6")})
        lat = np.array([40.0])
        assert np.isnat(o.scan_time_offset_for_position(lat, lat)[0])
        assert not np.isnat(o.scan_time_offset_for_position(lat, lat, _minute(1))[0])


# ── stored scan times ─────────────────────────────────────────────────────────


class TestStoredScanTimes:
    def test_stored_grid_is_the_table_from_scan_start(self, tmp_path):
        obj = _frames(tmp_path)
        obj.lats[0, 0] = np.nan  # off the disk
        grid = obj.static_scan_time_offset_s()
        table = abi_time_model.load_table("MESO") + abi_time_model.band_correction_s([11, 14, 15])
        assert np.isnan(grid[0, 0])
        np.testing.assert_allclose(grid[1:, 1:], table[1:, 1:], atol=1e-3)

    def test_rebuilt_times_match_the_lookup(self, tmp_path, monkeypatch):
        obj = _frames(tmp_path)
        monkeypatch.setattr(
            obj,
            "get_grid_index_from_position",
            lambda lat, lon: (np.full(np.shape(lat), 7), np.full(np.shape(lat), 9)),
        )
        t = _minute(1)
        rebuilt = obj.scan_start_offsets_s(np.array([t]))[0] + obj.static_scan_time_offset_s()[7, 9]
        live = obj.scan_time_offset_for_position(40.0, 40.0, t) / np.timedelta64(1, "s")
        assert rebuilt == pytest.approx(live, abs=1e-3)


# ── _frame_projection ─────────────────────────────────────────────────────────


def test_frame_projection_scales_scan_angles_by_height():
    obj = object.__new__(GOESDetector)
    attrs = {
        "perspective_point_height": 35786023.0,
        "longitude_of_projection_origin": -75.0,
        "sweep_angle_axis": "x",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.2572221,
    }
    da = xr.DataArray(
        np.zeros((3, 2, 2)),
        dims=("band_id", "y", "x"),
        coords={"x": [-0.1, 0.1], "y": [0.1, -0.1]},
        attrs={"goes_imager_projection": attrs},
    )
    geo_x, geo_y, crs = obj._frame_projection(da)
    np.testing.assert_allclose(geo_x, [-3578602.3, 3578602.3])
    np.testing.assert_allclose(geo_y, [3578602.3, -3578602.3])
    assert crs.to_dict()["sweep"] == "x"


# ── _generate_tile_mobile_grid ────────────────────────────────────────────────


@pytest.fixture
def bare_abi(stub_abi):
    """Build a minimal ABIDetector instance for _generate_tile_mobile_grid."""

    def _bare_abi(region: str = "M1"):
        obj = object.__new__(stub_abi)
        obj.region = region
        rng = np.random.default_rng(0)
        obj.lons = rng.uniform(-120, -100, (500, 500))
        obj.lats = rng.uniform(30, 50, (500, 500))
        return obj

    return _bare_abi


def _dummy_img() -> np.ndarray:
    """500x500x3 image filled with recognisable values."""
    img = np.zeros((500, 500, 3), dtype=np.float32)
    for band in range(3):
        img[..., band] = band + 1
    return img


class TestGenerateTileMobileGrid:
    """Tile slicing and coordinate-source logic."""

    @pytest.mark.parametrize(
        "tile_idx,expected_start_x,expected_start_y",
        [
            (0, 0, 0),
            (1, 0, 244),
            (2, 244, 0),
            (3, 244, 244),
        ],
    )
    def test_tile_slicing_indices(self, bare_abi, tile_idx, expected_start_x, expected_start_y):
        """Verify the correct quadrant of the image is extracted."""
        obj = bare_abi()
        img = _dummy_img()
        xx, yy, tile_img = obj._generate_tile_mobile_grid(tile_idx=tile_idx, img=img)
        assert xx.shape == (256, 256)
        assert yy.shape == (256, 256)
        assert tile_img.shape == (256, 256, 3)
        end_x = expected_start_x + 256
        end_y = expected_start_y + 256
        expected = img[expected_start_y:end_y, expected_start_x:end_x]
        np.testing.assert_array_equal(tile_img, expected)

    def test_fallback_to_instance_lons_lats_when_no_geo(self, bare_abi):
        """Without geo_x/geo_y the method uses self.lons/self.lats."""
        obj = bare_abi()
        img = _dummy_img()
        xx, yy, _ = obj._generate_tile_mobile_grid(tile_idx=0, img=img)
        np.testing.assert_array_equal(xx, obj.lons[0:256, 0:256])
        np.testing.assert_array_equal(yy, obj.lats[0:256, 0:256])

    def test_geo_coords_path_calls_transform(self, bare_abi):
        """When geo_x/geo_y are provided the satellite transform is called."""
        obj = bare_abi()

        # Build small 1-D coord arrays and a matching transform mock
        geo_x = np.linspace(-0.05, 0.05, 500)
        geo_y = np.linspace(0.05, -0.05, 500)

        fresh_lons = np.full((500, 500), -105.0)
        fresh_lons[0, 0] = np.inf
        fresh_lats = np.full((500, 500), 40.0)
        mock_transform = MagicMock()
        mock_transform.transform.return_value = (fresh_lons.copy(), fresh_lats.copy())
        obj.sat_to_latlon_transform = mock_transform

        img = _dummy_img()
        xx, yy, _ = obj._generate_tile_mobile_grid(tile_idx=0, img=img, geo_x=geo_x, geo_y=geo_y)

        mock_transform.transform.assert_called_once()
        assert np.isnan(xx[0, 0])
        # Result is a slice of the transformed arrays, not the instance grid
        expected_lons = np.where(np.isinf(fresh_lons), np.nan, fresh_lons)
        np.testing.assert_array_equal(xx, expected_lons[0:256, 0:256])
        np.testing.assert_array_equal(yy, fresh_lats[0:256, 0:256])


# ── spans_single_epoch ────────────────────────────────────────────────────────


def _goes_range(region: str = "M1", n_times: int = 3) -> GOESDetector:
    """Minimal GOESDetector for spans_single_epoch, with no network access."""
    obj = object.__new__(GOESDetector)
    obj.satellite = "GOES-16"
    obj.region = region
    obj.timestamp_sequence = [
        np.datetime64("2024-08-28T02:30") + np.timedelta64(i, "m") for i in range(n_times)
    ]
    return obj


def _grids(*fingerprints):
    """Stand in for _grid_fingerprint_at, returning first/last values in order."""
    seq = list(fingerprints)
    calls = {"i": 0}

    def _fake(_timestamp):
        value = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        return value

    return _fake


class TestSpansSingleEpoch:
    """Detecting a time range that crosses a sector repositioning."""

    _A = (1.0, 2.0, 3.0, 4.0)
    _B = (9.0, 8.0, 7.0, 6.0)

    def test_compares_grid_at_both_ends(self):
        obj = _goes_range(region="M1")
        obj._grid_fingerprint_at = _grids(self._A, self._A)
        assert obj.spans_single_epoch() is True

        obj = _goes_range(region="M1")
        obj._grid_fingerprint_at = _grids(self._A, self._B)
        assert obj.spans_single_epoch() is False

    @pytest.mark.parametrize("region", ["C", "F"])
    def test_fixed_region_returns_none(self, region):
        """Inert for fixed domains, and must not fetch anything to decide that."""
        obj = _goes_range(region=region)

        def _explode(_timestamp):
            raise AssertionError("fixed region must not fetch geometry")

        obj._grid_fingerprint_at = _explode
        assert obj.spans_single_epoch() is None

    def test_single_timestamp_is_trivially_one_epoch(self):
        obj = _goes_range(region="M1", n_times=1)
        assert obj.spans_single_epoch() is True

    def test_unavailable_grid_returns_none(self):
        """Unknown is not the same as safe — it must not report True."""
        obj = _goes_range(region="M1")
        obj._grid_fingerprint_at = _grids(self._A, None)
        assert obj.spans_single_epoch() is None

    def test_only_ends_are_fetched(self):
        """Cost is one extra fetch, not one per frame."""
        obj = _goes_range(region="M1", n_times=120)
        seen = []

        def _record(timestamp):
            seen.append(timestamp)
            return self._A

        obj._grid_fingerprint_at = _record
        obj.spans_single_epoch()
        assert seen == [obj.timestamp_sequence[0], obj.timestamp_sequence[-1]]


# ── _process_single_timestamp: mesoscale box moves ───────────────────────────


@pytest.fixture
def moving_box_abi(stub_abi):
    """Mesoscale stub whose box moves to a new position from ``move_time`` onwards."""

    class _MovingBoxABI(stub_abi):
        def _get_frame(self, time):
            x0 = 1000.0 if time >= self.move_time else 0.0
            return xr.DataArray(np.zeros((3, 2, 2)), dims=("band_id", "y", "x"), attrs={"x0": x0})

        def _frame_extent(self, da):
            x0 = da.attrs["x0"]
            return (x0, x0 + 500.0, 0.0, 500.0)

        def _frame_projection(self, da):
            return np.zeros(2), np.zeros(2), None

    return _MovingBoxABI


def test_window_spanning_a_box_move_is_skipped(monkeypatch, tmp_path, moving_box_abi):
    """Every stack of five frames that includes the move is skipped, not just the first."""
    import contrail_detect.geostationary.abi_base as abi_base

    monkeypatch.setattr(abi_base, "TEST_DIR", tmp_path)
    obj = object.__new__(moving_box_abi)
    obj.satellite, obj.region, obj.image_interval = "goes-19", "M1", 1
    obj.projection_cache = {}
    start = pd.Timestamp("2025-06-01T12:00")
    obj.move_time = start + pd.Timedelta(minutes=6)
    staged_times = []
    monkeypatch.setattr(
        obj, "load_single_tile", lambda **kw: staged_times.append(kw["time_str"]) or True
    )

    data_cache, tile_cache = {}, OrderedDict()
    for minute in range(4, 14):
        timestamp = str(np.datetime64(start + pd.Timedelta(minutes=minute)))
        obj._process_single_timestamp(timestamp, [0], data_cache, tile_cache)

    # 12:04 and 12:05 are entirely before the move, 12:10 onwards entirely after it.
    # 12:06 to 12:09 each mix the two positions.
    assert staged_times == [
        "2025-06-01T1204",
        "2025-06-01T1205",
        "2025-06-01T1210",
        "2025-06-01T1211",
        "2025-06-01T1212",
        "2025-06-01T1213",
    ]


# ── fetch_image ───────────────────────────────────────────────────────────────


def test_fetch_image_skips_only_missing_image(caplog):
    """pycontrails reports a missing image as FileNotFoundError; plotting skips that time."""
    obj = object.__new__(GOESDetector)
    obj.region = "C"
    obj.pycon_goes = MagicMock()
    obj.pycon_goes.get.side_effect = FileNotFoundError("No data found for 2025-06-01 12:00")
    with caplog.at_level(logging.WARNING):
        assert obj.fetch_image(np.datetime64("2025-06-01T12:00")) == (None, None, None)
    assert "No data found" in caplog.text

    obj = object.__new__(GOESDetector)
    obj.region = "C"
    obj.pycon_goes = MagicMock()
    obj.pycon_goes.get.side_effect = FileNotFoundError("cache directory is gone")
    with pytest.raises(FileNotFoundError):
        obj.fetch_image(np.datetime64("2025-06-01T12:00"))


class TestWindInitialisationSetting:
    """get_tracking_output follows TrackingConfig unless the argument is given."""

    @pytest.fixture
    def run(self, monkeypatch):
        import contrail_detect.tracking.tracking as tracking
        from contrail_detect.tracking.tracking import TrackingConfig

        seen = {}

        class FakeManager:
            def __init__(self, contrail_summary, config, use_wind_initialization):
                seen["manager"] = use_wind_initialization

            def run_tracking(self):
                return {"c1": "c1"}

        monkeypatch.setattr(tracking, "TrackingManager", FakeManager)

        def _run(config_value, argument):
            det = _bare_goes()
            det.contrail_summary = pd.DataFrame({"contrail_id": ["c1"]})
            seen["era5"] = False
            det._add_windspeed_to_summary = lambda **kw: seen.update(era5=True)
            kwargs = {} if argument == "omit" else {"use_wind_initialization": argument}
            det.get_tracking_output(
                tracking_config=TrackingConfig(use_wind_initialization=config_value), **kwargs
            )
            return seen["era5"], seen["manager"]

        return _run

    @pytest.mark.parametrize(
        "config_value, argument, expected",
        [
            (True, "omit", True),
            (False, "omit", False),
            (False, True, True),
            (True, False, False),
        ],
    )
    def test_config_unless_overridden(self, run, config_value, argument, expected):
        assert run(config_value, argument) == (expected, expected)
