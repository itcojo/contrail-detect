"""Unit tests for contrail_detect.meteorology.ecmwf.

Tests cover input validation in _download_era5 (no network required,
credentials come from environment variables or a temporary file) and interpolate_era5_to_points (pure function, tested
with a synthetic xarray Dataset).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrail_detect.meteorology import ecmwf
from contrail_detect.meteorology.ecmwf import (
    _download_era5,
    get_era5_from_local,
    initialize_tracking_windspeeds,
    interpolate_era5_to_points,
)

# ── Synthetic ERA5 dataset ────────────────────────────────────────────────────


def _synthetic_era5(
    times: pd.DatetimeIndex | None = None,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    levels: np.ndarray | None = None,
) -> xr.Dataset:
    """Build a minimal ERA5-like Dataset for testing."""
    if times is None:
        times = pd.date_range("2026-01-01", periods=4, freq="h")
    if lats is None:
        lats = np.array([40.0, 45.0, 50.0, 55.0])
    if lons is None:
        lons = np.array([-10.0, 0.0, 10.0])
    if levels is None:
        levels = np.array([200, 300, 500])

    shape = (len(times), len(levels), len(lats), len(lons))
    dims = ["time", "level", "latitude", "longitude"]
    coords = {"time": times, "level": levels, "latitude": lats, "longitude": lons}

    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "u": (dims, rng.uniform(-30, 30, shape).astype(np.float32)),
            "v": (dims, rng.uniform(-20, 20, shape).astype(np.float32)),
        },
        coords=coords,
    )


# ── _download_era5: input validation ─────────────────────────────


@pytest.fixture
def cds_env(monkeypatch):
    """CDS credentials from environment variables, so no ~/.cdsapirc is needed."""
    monkeypatch.setenv("CDSAPI_URL", "https://cds.example/api")
    monkeypatch.setenv("CDSAPI_KEY", "env-key")


@pytest.fixture
def no_cds_credentials(monkeypatch, tmp_path):
    """No environment variables and an empty home directory."""
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setattr("contrail_detect.meteorology.ecmwf.Path.home", lambda: tmp_path)


def _download(**kwargs):
    return _download_era5(
        start_time="2026-01-01T00:00",
        end_time="2026-01-01T06:00",
        variables=["temperature"],
        **kwargs,
    )


class TestDownloadERA5Validation:
    def test_missing_credentials_raise(self, no_cds_credentials):
        with pytest.raises(ValueError, match="API key"):
            _download(path_to_key="/nonexistent/path")
        with pytest.raises(ValueError, match="API key"):
            _download()

    @pytest.mark.usefixtures("no_cds_credentials")
    @patch("contrail_detect.meteorology.ecmwf.ERA5")
    def test_path_to_key_credentials_passed_to_era5(self, era5, tmp_path):
        key_dir = tmp_path / "keys"
        key_dir.mkdir()
        (key_dir / ".cdsapirc").write_text("url: https://cds.example/api\nkey: file-key\n")
        _download(path_to_key=key_dir)
        assert era5.call_args.kwargs["url"] == "https://cds.example/api"
        assert era5.call_args.kwargs["key"] == "file-key"

    @pytest.mark.usefixtures("cds_env")
    @patch("contrail_detect.meteorology.ecmwf.ERA5")
    def test_environment_credentials_need_no_file(self, era5, monkeypatch, tmp_path):
        monkeypatch.setattr("contrail_detect.meteorology.ecmwf.Path.home", lambda: tmp_path)
        _download()
        # Left to pycontrails, which reads the environment variables itself
        assert era5.call_args.kwargs["url"] is None
        assert era5.call_args.kwargs["key"] is None


# ── interpolate_era5_to_points ────────────────────────────────────────────────


class TestInterpolateERA5ToPoints:
    def setup_method(self):
        self.ds = _synthetic_era5()
        self.times = pd.date_range("2026-01-01", periods=4, freq="h")

    def test_output_shape_and_columns(self):
        n = 5
        result = interpolate_era5_to_points(
            ds=self.ds,
            lats=np.linspace(40, 55, n),
            lons=np.zeros(n),
            pressure_hPa=np.full(n, 300),
            times=pd.DatetimeIndex([self.times[0]] * n),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n
        assert set(result.columns) == {"u", "v"}

    def test_point_on_grid_node_returns_exact_value(self):
        # Place a point exactly on a grid node and verify the interpolated value matches
        ds = self.ds
        lat, lon, level = 45.0, 0.0, 300
        t = self.times[1]
        expected_u = float(ds["u"].sel(latitude=lat, longitude=lon, level=level, time=t))

        result = interpolate_era5_to_points(
            ds=ds,
            lats=np.array([lat]),
            lons=np.array([lon]),
            pressure_hPa=np.array([level]),
            times=pd.DatetimeIndex([t]),
        )
        assert result["u"].iloc[0] == pytest.approx(expected_u, rel=1e-4)

    def test_off_hour_point_is_time_interpolated(self):
        # A point halfway between two ERA5 hours is the mean of the bracketing values
        # (linear in time), not snapped to the nearest hour.
        lat, lon, level = 45.0, 0.0, 300
        t_half = pd.DatetimeIndex([self.times[0] + pd.Timedelta(minutes=30)])
        u0 = float(self.ds["u"].sel(latitude=lat, longitude=lon, level=level, time=self.times[0]))
        u1 = float(self.ds["u"].sel(latitude=lat, longitude=lon, level=level, time=self.times[1]))

        result = interpolate_era5_to_points(
            ds=self.ds,
            lats=np.array([lat]),
            lons=np.array([lon]),
            pressure_hPa=np.array([level]),
            times=t_half,
            max_time_gap_hours=1,
        )
        assert result["u"].iloc[0] == pytest.approx(0.5 * (u0 + u1), rel=1e-4)

    def test_mixed_valid_and_invalid_times(self):
        valid_t = self.times[0]
        invalid_t = pd.Timestamp("2030-01-01")
        times = pd.DatetimeIndex([valid_t, invalid_t, valid_t])

        result = interpolate_era5_to_points(
            ds=self.ds,
            lats=np.array([45.0, 45.0, 45.0]),
            lons=np.array([0.0, 0.0, 0.0]),
            pressure_hPa=np.array([300, 300, 300]),
            times=times,
            max_time_gap_hours=1,
        )
        assert not np.isnan(result["u"].iloc[0])
        assert np.isnan(result["u"].iloc[1])
        assert np.isnan(result["v"].iloc[1])
        assert not np.isnan(result["u"].iloc[2])

    def test_custom_level_dim_name(self):
        ds = self.ds.rename({"level": "pressure_level"})
        result = interpolate_era5_to_points(
            ds=ds,
            lats=np.array([45.0]),
            lons=np.array([0.0]),
            pressure_hPa=np.array([300]),
            times=pd.DatetimeIndex([self.times[0]]),
            level_dim="pressure_level",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_point_at_last_hour_is_bracketed_by_last_two_hours(self):
        t_last = pd.DatetimeIndex([self.times[-1]])
        result = interpolate_era5_to_points(
            ds=self.ds,
            lats=np.array([45.0]),
            lons=np.array([0.0]),
            pressure_hPa=np.array([300]),
            times=t_last,
        )
        expected = self.ds["u"].sel(time=self.times[-1], level=300, latitude=45.0, longitude=0.0)
        assert result["u"].iloc[0] == pytest.approx(float(expected))

    def test_point_in_time_gap_returns_nan(self):
        # When the two bracketing hours are further apart than max_time_gap_hours
        # (here a missing 02:00), the point sits in a data gap and is NaN.
        ds = _synthetic_era5(
            times=pd.DatetimeIndex(["2026-01-01T00", "2026-01-01T01", "2026-01-01T03"])
        )
        t_gap = pd.DatetimeIndex([pd.Timestamp("2026-01-01T02:00")])
        result = interpolate_era5_to_points(
            ds=ds,
            lats=np.array([45.0]),
            lons=np.array([0.0]),
            pressure_hPa=np.array([300]),
            times=t_gap,
            max_time_gap_hours=1,
        )
        assert np.isnan(result["u"].iloc[0])

    def test_points_at_block_boundaries_are_not_nan(self):
        # Points are loaded in blocks of time_block_hours, and each block's slice must
        # include the upper bracketing hour of its latest point. Since that slice is
        # end-exclusive, an off-by-one there does not raise -- bounds_error=False turns
        # the missing bracket into a silent NaN at the last point of every block. This
        # spans 30 h so the default 24 h blocking is actually crossed; the default 4 h
        # fixture fits in one block and never exercises the boundary.
        times = pd.date_range("2026-01-01", periods=30, freq="h")
        ds = _synthetic_era5(times=times)

        # Half-past every hour: each point needs a bracket on both sides, and the
        # points at 23:30 and 28:30 sit exactly on the block edges.
        pts = times[:-1] + pd.Timedelta(minutes=30)
        n = len(pts)
        result = interpolate_era5_to_points(
            ds=ds,
            lats=np.full(n, 45.0),
            lons=np.zeros(n),
            pressure_hPa=np.full(n, 300),
            times=pts,
            time_block_hours=24,
        )

        assert len(result) == n
        assert not result.isna().to_numpy().any()

    def test_result_is_independent_of_time_block_size(self):
        # Blocking is purely an I/O batching strategy, so the interpolated values must
        # not depend on time_block_hours. Comparing a small block size against one
        # larger than the dataset catches boundary errors regardless of where they fall.
        times = pd.date_range("2026-01-01", periods=30, freq="h")
        ds = _synthetic_era5(times=times)

        pts = times[:-1] + pd.Timedelta(minutes=30)
        n = len(pts)
        kwargs = {
            "ds": ds,
            "lats": np.full(n, 45.0),
            "lons": np.zeros(n),
            "pressure_hPa": np.full(n, 300),
            "times": pts,
        }

        blocked = interpolate_era5_to_points(**kwargs, time_block_hours=6)
        single_block = interpolate_era5_to_points(**kwargs, time_block_hours=1000)

        pd.testing.assert_frame_equal(blocked, single_block)

    @pytest.mark.parametrize("descending_lat", [False, True])
    def test_matches_interpolation_on_the_whole_grid(self, descending_lat):
        lats = np.arange(30.0, 61.0, 2.5)
        ds = _synthetic_era5(
            lats=lats[::-1] if descending_lat else lats,
            lons=np.arange(-20.0, 21.0, 2.5),
            levels=np.array([150, 200, 250, 300, 400, 500, 700]),
        )
        # Inside the grid, on grid lines, on the edges and outside it
        lat = np.array([41.3, 45.0, 30.0, 60.0, 29.0, 41.3, np.nan])
        lon = np.array([3.7, 5.0, -20.0, 20.0, 3.7, 25.0, 3.7])
        lev = np.array([260.0, 300.0, 150.0, 700.0, 260.0, 260.0, 260.0])
        times = pd.DatetimeIndex(["2026-01-01 01:20"] * len(lat))

        result = interpolate_era5_to_points(ds, lat, lon, lev, times)

        expected = ds.interp(
            latitude=xr.DataArray(lat, dims="p"),
            longitude=xr.DataArray(lon, dims="p"),
            level=xr.DataArray(lev, dims="p"),
            time=xr.DataArray(times, dims="p"),
            method="linear",
            kwargs={"fill_value": np.nan, "bounds_error": False},
        )
        for var in ("u", "v"):
            np.testing.assert_allclose(result[var].to_numpy(), expected[var].values, rtol=1e-12)


# ── get_era5_from_local: which days, and cached files ────────────────────────


class _FakeMet:
    def __init__(self, ds):
        self.data = ds


@pytest.fixture
def fake_download(monkeypatch):
    """Record each _download_era5 call and return a one-day dataset with those variables."""
    calls = []

    def fake(start_time, end_time, variables, pressure_levels, path_to_key=None):
        calls.append((pd.Timestamp(start_time), list(variables), list(pressure_levels)))
        times = pd.date_range(start_time, end_time, freq="h")
        levels = np.array(pressure_levels, dtype=float)
        ds = _synthetic_era5(times=times, levels=levels).drop_vars(["u", "v"])
        for var in variables:
            ds[var] = (
                ("time", "level", "latitude", "longitude"),
                np.full((len(times), len(levels), 4, 3), 7.0),
            )
        return _FakeMet(ds)

    monkeypatch.setattr(ecmwf, "_download_era5", fake)
    return calls


class TestGetERA5FromLocal:
    def test_time_after_23_also_loads_next_day(self, fake_download, tmp_path):
        met = get_era5_from_local(
            times=[np.datetime64("2026-01-01T21:30"), np.datetime64("2026-01-01T23:10")],
            met_variables=["eastward_wind"],
            convert_pressure_levels=False,
            file_path_met=tmp_path,
        )
        assert [c[0] for c in fake_download] == [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
        ]
        assert pd.Timestamp(met["time"].max().values) == pd.Timestamp("2026-01-02T23:00")

    def test_time_on_the_hour_does_not_load_next_day(self, fake_download, tmp_path):
        get_era5_from_local(
            times=[np.datetime64("2026-01-01T22:00"), np.datetime64("2026-01-01T23:00")],
            met_variables=["eastward_wind"],
            convert_pressure_levels=False,
            file_path_met=tmp_path,
        )
        assert [c[0] for c in fake_download] == [pd.Timestamp("2026-01-01")]

    def test_days_between_first_and_last_are_all_loaded(self, fake_download, tmp_path):
        get_era5_from_local(
            times=[np.datetime64("2026-01-01T12:00"), np.datetime64("2026-01-03T12:00")],
            met_variables=["eastward_wind"],
            convert_pressure_levels=False,
            file_path_met=tmp_path,
        )
        assert len(fake_download) == 3

    def test_cached_day_missing_a_variable_is_downloaded_again(self, fake_download, tmp_path):
        kwargs = dict(
            times=[np.datetime64("2026-01-01T12:00")],
            convert_pressure_levels=False,
            file_path_met=tmp_path,
        )
        get_era5_from_local(met_variables=["eastward_wind"], **kwargs)
        met = get_era5_from_local(met_variables=["eastward_wind", "temperature"], **kwargs)
        assert fake_download[-1][1] == ["eastward_wind", "temperature"]
        assert {"eastward_wind", "temperature"} <= set(met.data_vars)
        get_era5_from_local(met_variables=["temperature"], **kwargs)
        assert len(fake_download) == 2  # now complete, so not downloaded again

    def test_pressure_levels(self, fake_download, tmp_path):
        kwargs = dict(
            times=[np.datetime64("2026-01-01T12:00")],
            met_variables=["eastward_wind"],
            convert_pressure_levels=False,
            file_path_met=tmp_path,
        )
        met = get_era5_from_local(**kwargs)
        assert fake_download[-1][2] == ecmwf.DEFAULT_PRESSURE_LEVELS
        assert list(met["level"].values) == ecmwf.DEFAULT_PRESSURE_LEVELS

        # A level the saved day lacks: downloaded again with the saved levels too
        met = get_era5_from_local(pressure_levels=[100, 250], **kwargs)
        assert fake_download[-1][2] == [100] + ecmwf.DEFAULT_PRESSURE_LEVELS
        assert list(met["level"].values) == [100, 250]

        get_era5_from_local(pressure_levels=[250, 100], **kwargs)
        met = get_era5_from_local(pressure_levels=[1000], **kwargs)
        assert list(met["level"].values) == [1000]
        assert len(fake_download) == 2

    @pytest.mark.parametrize(
        "levels, match", [([250.5], "integers"), ([50], "between"), ([1100], "between")]
    )
    def test_invalid_pressure_levels_raise(self, fake_download, tmp_path, levels, match):
        with pytest.raises(ValueError, match=match):
            get_era5_from_local(
                times=[np.datetime64("2026-01-01T12:00")],
                met_variables=["eastward_wind"],
                file_path_met=tmp_path,
                pressure_levels=levels,
            )
        assert not fake_download

    def test_interrupted_download_leaves_no_day_file(self, monkeypatch, tmp_path):
        class _HalfWritten:
            def to_netcdf(self, path):
                Path(path).write_text("partial")
                raise OSError("disk full")

        monkeypatch.setattr(ecmwf, "_download_era5", lambda **kwargs: _FakeMet(_HalfWritten()))
        with pytest.raises(OSError):
            get_era5_from_local(
                times=[np.datetime64("2026-01-01T12:00")],
                met_variables=["eastward_wind"],
                file_path_met=tmp_path,
            )
        assert not (tmp_path / "ERA5-2026-01-01.netcdf").exists()


# ── initialize_tracking_windspeeds ───────────────────────────────────────────


def _detections(scan_times):
    return pd.DataFrame(
        {
            "centroid_lon": 0.0,
            "centroid_lat": 45.0,
            "centroid_scantime": pd.to_datetime(scan_times),
            "timestamp": pd.to_datetime(scan_times).floor("10min"),
        }
    )


class TestInitializeTrackingWindspeeds:
    def test_scan_after_23_gets_real_wind(self, fake_download, monkeypatch, tmp_path):
        monkeypatch.setattr(ecmwf, "DATA_DIR", tmp_path)
        df = initialize_tracking_windspeeds(_detections(["2026-01-01T22:50", "2026-01-01T23:40"]))
        assert [c[0] for c in fake_download] == [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
        ]
        assert (df["u_init"] == 7.0).all() and (df["v_init"] == 7.0).all()

    def test_detection_outside_wind_data_is_nan_with_warning(
        self, fake_download, monkeypatch, tmp_path, caplog
    ):
        monkeypatch.setattr(ecmwf, "DATA_DIR", tmp_path)
        df = _detections(["2026-01-01T12:00", "2026-01-01T12:10"])
        df.loc[1, "centroid_lat"] = 80.0  # north of the synthetic grid
        with caplog.at_level("WARNING"):
            out = initialize_tracking_windspeeds(df)
        assert out["u_init"].iloc[0] == 7.0
        assert np.isnan(out["u_init"].iloc[1])
        assert "No ERA5 wind for 1 of 2" in caplog.text
