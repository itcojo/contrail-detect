"""End-to-end tests of FlightData.advect_flights_to_scan_times on synthetic data.

A fake detector supplies frame times and a constant scan-time offset, and a synthetic ERA5
field supplies uniform winds, so the exact advected position of every waypoint is known.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails.physics import units

from contrail_detect.flights.flights_base import FlightData

T0 = np.datetime64("2024-06-01T12:00", "ns")
TEN_MIN = np.timedelta64(10, "m")


class _Flights(FlightData):
    def _normalize_columns(self):
        pass


class _FakeDetector:
    """Frame times and a constant scan-time offset; positions outside ``lat_max`` are unseen."""

    def __init__(self, n_frames=4, offset_s=0.0, lat_max=80.0, satellite="GOES-19", region="F"):
        self.timestamp_sequence = [T0 + i * TEN_MIN for i in range(n_frames)]
        self.offset = np.timedelta64(int(offset_s * 1e9), "ns")
        self.lat_max = lat_max
        self.satellite = satellite
        self.region = region

    def scan_time_offset_for_position(self, lat, lon, timestamp=None):
        lat = np.asarray(lat, dtype=float)
        out = np.full(lat.shape, self.offset, dtype="timedelta64[ns]")
        out[np.abs(lat) > self.lat_max] = np.timedelta64("NaT")
        return out


def _era5(
    u=20.0,
    v=0.0,
    lon=(-100.0, -50.0, 1.0),
    start=T0 - np.timedelta64(3, "h"),
    hours=6,
) -> xr.Dataset:
    """Uniform-wind field on a small grid, in the layout ``get_era5_from_local`` returns."""
    longitude = np.arange(lon[0], lon[1] + lon[2] / 2, lon[2])
    latitude = np.arange(20.0, 61.0, 1.0)
    altitude_ft = np.array([20000.0, 30000.0, 40000.0])
    time = start + np.arange(hours + 1) * np.timedelta64(1, "h")
    shape = (len(longitude), len(latitude), len(altitude_ft), len(time))
    coords = {
        "longitude": longitude,
        "latitude": latitude,
        "altitude_ft": altitude_ft,
        "time": time,
    }
    dims = list(coords)
    # u may be a function of time, giving the same wind everywhere at each hour
    u_values = np.broadcast_to(np.asarray(u(time) if callable(u) else u, dtype=float), shape)
    return xr.Dataset(
        {"u": (dims, u_values.copy()), "v": (dims, np.full(shape, v))},
        coords=coords,
    )


def _flights(rows) -> _Flights:
    """FlightData holding waypoints given as (flight_id, time, lon, lat, alt_ft)."""
    f = _Flights()
    f.flight_data = pd.DataFrame(
        rows, columns=["flight_id", "time", "longitude", "latitude", "altitude_ft"]
    )
    f.flight_data["time"] = pd.to_datetime(f.flight_data["time"])
    f.ecmwf_data = _era5()
    return f


def _straight_flight(flight_id="F1", lon0=-80.0, lat=40.0, start=-30, end=35, step=5):
    """Waypoints every ``step`` min from ``start`` to ``end`` min after T0, moving east."""
    return [
        (flight_id, T0 + np.timedelta64(m, "m"), lon0 + 0.01 * m, lat, 35000.0)
        for m in range(start, end, step)
    ]


def _expected_lon(row, u=20.0):
    dt_s = (row["scan_time"] - row["original_time"]) / np.timedelta64(1, "s")
    lon0 = -80.0 + 0.01 * ((row["original_time"] - T0) / np.timedelta64(1, "m"))
    return lon0 + units.m_to_longitude_distance(u * dt_s, row["latitude"])


@pytest.mark.parametrize("offset_s", [0.0, 170.0])
def test_uniform_wind_moves_each_point_by_wind_times_age(offset_s):
    flights = _flights(_straight_flight())
    detector = _FakeDetector(offset_s=offset_s)
    flights.advect_flights_to_scan_times(detector, adv_max_past=np.timedelta64(30, "m"))
    out = flights.adv_flight_data
    assert len(out) > 0
    np.testing.assert_allclose(out["longitude"], out.apply(_expected_lon, axis=1), atol=1e-9)
    np.testing.assert_allclose(out["latitude"], 40.0)
    assert (out["scan_time"] == out["frame_time"] + detector.offset).all()


# ── Setting checks ────────────────────────────────────────────────────────────


def test_bad_lookback_settings_raise():
    flights = _flights(_straight_flight())
    with pytest.raises(ValueError, match="whole number of advection steps"):
        flights.advect_flights_to_scan_times(
            _FakeDetector(), adv_max_past=np.timedelta64(63, "m"), adv_timestep=5
        )
    with pytest.raises(TypeError, match="numpy.timedelta64"):
        flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=60)

    flights = _Flights(req_timestamps=[T0], adv_max=1)
    flights.flight_data = _flights(_straight_flight()).flight_data
    flights.ecmwf_data = _era5()
    with pytest.raises(ValueError, match="larger adv_max"):
        flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(2, "h"))


# ── Missing wind data ─────────────────────────────────────────────────────────


def test_missing_winds_raise():
    flights = _flights(_straight_flight())
    flights.ecmwf_data = _era5(hours=3)  # ends 12:00, the run needs winds to 12:40
    with pytest.raises(ValueError, match="does not cover the advection time range"):
        flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))

    flights.ecmwf_data = None
    with pytest.raises(ValueError, match="not loaded"):
        flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))


# ── Date line and wind-data edges ─────────────────────────────────────────────


def _wrapped(lon):
    return (lon + 180.0) % 360.0 - 180.0


def test_flight_crossing_the_date_line_keeps_moving():
    flights = _flights(_straight_flight(lon0=179.9))
    flights.ecmwf_data = _era5(lon=(-180.0, 175.0, 5.0))  # global, like pycontrails ERA5
    flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))
    out = flights.adv_flight_data
    dt_s = (out["scan_time"] - out["original_time"]) / np.timedelta64(1, "s")
    lon0 = 179.9 + 0.01 * ((out["original_time"] - T0) / np.timedelta64(1, "m"))
    expected = _wrapped(lon0 + units.m_to_longitude_distance(20.0 * dt_s, 40.0))
    np.testing.assert_allclose(out["longitude"], expected, atol=1e-9)
    assert (out["longitude"] < 0).any(), "some points should have crossed to the west side"
    assert ((out["longitude"] >= -180) & (out["longitude"] < 180)).all()


def test_points_leaving_the_wind_area_are_dropped_not_frozen(caplog):
    # Regional winds end at -50; the flight starts just inside and is blown out of it
    flights = _flights(_straight_flight(lon0=-50.4))
    with caplog.at_level("WARNING"):
        flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))
    out = flights.adv_flight_data
    assert (out["longitude"] <= -50.0).all()
    np.testing.assert_allclose(
        out["longitude"],
        -50.4
        + 0.01 * ((out["original_time"] - T0) / np.timedelta64(1, "m"))
        + units.m_to_longitude_distance(
            20.0 * (out["scan_time"] - out["original_time"]) / np.timedelta64(1, "s"), 40.0
        ),
        atol=1e-9,
    )
    assert "outside the wind data" in caplog.text


# ── Wind sampling ─────────────────────────────────────────────────────────────


def _rising_wind(t):
    """20 m/s at T0, rising by 10 m/s per hour (linear, so hourly data interpolates it exactly)."""
    return 20.0 + 10.0 * (np.asarray(t, dtype="datetime64[ns]") - T0) / np.timedelta64(1, "h")


def _start_of_move_position(lon0, t0, scan_time, lat=40.0, step=TEN_MIN):
    """Position after moving from t0 to scan_time in pieces split at each 10-minute mark,
    each piece using the wind at its own start."""
    marks = [t0]
    k = (t0 - T0) // step + 1
    while T0 + k * step < scan_time:
        marks.append(T0 + k * step)
        k += 1
    marks.append(scan_time)
    lon = lon0
    for a, b in zip(marks[:-1], marks[1:]):
        dist_m = _rising_wind(a) * ((b - a) / np.timedelta64(1, "s"))
        lon += units.m_to_longitude_distance(dist_m, lat)
    return lon


def test_each_move_uses_the_wind_at_its_start():
    """Before and after the first frame, a point moves with the wind at the time it starts
    each move, including points that join part-way through a step."""
    rows = [("F1", T0 + np.timedelta64(m, "m"), -80.0, 40.0, 35000.0) for m in (-27, -13, 4, 17)]
    flights = _flights(rows)
    flights.ecmwf_data = _era5(u=_rising_wind)
    flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))
    out = flights.adv_flight_data
    assert set(out["original_time"]) == {r[1] for r in rows}
    expected = [
        _start_of_move_position(
            -80.0, np.datetime64(r.original_time, "ns"), np.datetime64(r.scan_time, "ns")
        )
        for r in out.itertuples()
    ]
    np.testing.assert_allclose(out["longitude"], expected, atol=1e-9)


# ── Altitude used for winds ───────────────────────────────────────────────────


def _sheared_era5():
    """u = 10 m/s at 30,000 ft rising to 30 m/s at 40,000 ft (2 m/s per 1,000 ft)."""
    ds = _era5()
    u_by_alt = xr.DataArray(
        [0.0, 10.0, 30.0], dims="altitude_ft", coords={"altitude_ft": ds.altitude_ft}
    )
    ds["u"] = (ds["u"] * 0 + u_by_alt).transpose(*ds["u"].dims)
    return ds


def _one_point_flight(pressure_altitude_ft=None):
    flights = _flights([("F1", T0 - np.timedelta64(10, "m"), -80.0, 40.0, 35000.0)])
    if pressure_altitude_ft is not None:
        flights.flight_data["pressure_altitude_ft"] = pressure_altitude_ft
    flights.ecmwf_data = _sheared_era5()
    return flights


def _implied_wind(out):
    row = out.iloc[0]
    dt_s = (row["scan_time"] - row["original_time"]) / np.timedelta64(1, "s")
    return units.longitude_distance_to_m(row["longitude"] + 80.0, 40.0) / dt_s


def test_winds_come_from_pressure_altitude_but_altitude_stays_geometric():
    flights = _one_point_flight(pressure_altitude_ft=34000.0)
    flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))
    out = flights.adv_flight_data
    assert _implied_wind(out) == pytest.approx(18.0)  # wind at 34,000 ft, not 35,000 ft
    assert (out["altitude_ft"] == 35000.0).all()  # GNSS altitude kept for parallax
    assert (out["pressure_altitude_ft"] == 34000.0).all()


def test_without_pressure_altitude_winds_use_altitude_ft():
    flights = _one_point_flight()
    flights.advect_flights_to_scan_times(_FakeDetector(), adv_max_past=np.timedelta64(30, "m"))
    assert _implied_wind(flights.adv_flight_data) == pytest.approx(20.0)


def test_resampling_carries_pressure_altitude():
    rows = [("F1", T0 + np.timedelta64(m, "m"), -80.0 + 0.01 * m, 40.0, 35000.0) for m in (0, 20)]
    flights = _flights(rows)
    flights.flight_data["pressure_altitude_ft"] = [34000.0, 34200.0]
    flights.flight_data["original_time"] = flights.flight_data["time"]
    out = flights._resample_flight_data(flights.flight_data, resample_timestep=5)
    np.testing.assert_allclose(out["pressure_altitude_ft"], [34000, 34050, 34100, 34150, 34200])


# ── Saved advected flights ────────────────────────────────────────────────────


def _run(tmp_path, rows=None, detector=None, **kwargs):
    flights = _flights(rows or _straight_flight())
    flights.save_path = tmp_path
    settings = {"adv_max_past": np.timedelta64(30, "m"), "save_output": True, **kwargs}
    flights.advect_flights_to_scan_times(detector or _FakeDetector(), **settings)
    return flights


def test_identical_rerun_reuses_saved_file(tmp_path, caplog):
    first = _run(tmp_path)
    with caplog.at_level("INFO"):
        second = _run(tmp_path)
    assert "Loaded existing advected flight data" in caplog.text
    pd.testing.assert_frame_equal(first.adv_flight_data, second.adv_flight_data)


@pytest.mark.parametrize(
    "change",
    [
        {"rows": _straight_flight(flight_id="OTHER", lon0=-70.0)},
        {"adv_timestep": 5},
        {"adv_max_past": np.timedelta64(20, "m")},
        {"flight_resample_timestep": 1},
        {"detector": _FakeDetector(satellite="GOES-18")},
        {"detector": _FakeDetector(n_frames=3)},
    ],
)
def test_changed_input_is_recomputed(tmp_path, caplog, change):
    _run(tmp_path)
    with caplog.at_level("INFO"):
        changed = _run(tmp_path, **change)
    assert "Loaded existing advected flight data" not in caplog.text
    assert len(list((tmp_path / "advected").glob("*.pq"))) == 2
    if "rows" in change:
        assert set(changed.adv_flight_data["flight_id"]) == {"OTHER"}


def test_no_point_is_older_than_the_lookback():
    # Scan offset 170 s, so points carried a full lookback are scanned past it mid-step
    flights = _flights(_straight_flight(start=-30, end=35, step=1))
    lookback = np.timedelta64(30, "m")
    flights.advect_flights_to_scan_times(_FakeDetector(offset_s=170.0), adv_max_past=lookback)
    ages = flights.adv_flight_data["total_advected_time"]
    assert len(ages) > 0
    assert (ages <= lookback).all()
