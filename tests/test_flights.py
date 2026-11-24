"""Unit tests for FlightData and the public flight sources.

Covers shared init logic, bbox handling, column normalisation and the
ContrailsAPIFlightData and GenericFlightData sources, without network access.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from contrail_detect.flights.flights_base import FlightData

# ── Fixtures / helpers ────────────────────────────────────────────────────────

TS_DT64 = [np.datetime64("2024-06-01"), np.datetime64("2024-06-02")]
BBOX = [-10.0, 10.0, 40.0, 60.0]


# ── Concrete dummy subclass for testing the ABC ───────────────────────────────


class _Dummy(FlightData):
    def _normalize_columns(self):
        pass


# ── GenericFlightData ─────────────────────────────────────────────────────────


class TestGenericFlightData:
    def test_generic_flight_data_save_without_path(self, tmp_path):
        """Regression: save_to_file() crashed when no path was given."""
        from contrail_detect.flights.generic import GenericFlightData

        flights = GenericFlightData(save_path=tmp_path)
        flights.load_from_dataframe(
            pd.DataFrame(
                {
                    "flight_id": ["a", "a"],
                    "time": ["2026-01-01T10:00", "2026-01-01T11:30"],
                    "latitude": [50.0, 51.0],
                    "longitude": [0.0, 1.0],
                    "altitude_ft": [35000, 35000],
                }
            )
        )
        flights.save_to_file()
        assert (tmp_path / "generic_20260101T1000_20260101T1130.parquet").exists()


# ── FlightData.__init__ ───────────────────────────────────────────────────────


class TestFlightDataInit:
    def test_adv_max_accepted_types(self):
        for adv_max in [2, np.int64(2), np.int32(2), np.timedelta64(120, "m")]:
            d = _Dummy(req_timestamps=TS_DT64, adv_max=adv_max)
            assert d.adv_max == np.timedelta64(2, "h")
            assert d.req_timestamps[0] == np.datetime64("2024-05-31T22:00")
            assert len(d.req_timestamps) == 3
            assert all(t.dtype == np.dtype("datetime64[D]") for t in d.req_dates)
            assert len(d.req_dates) == len(d.req_timestamps)

    def test_adv_max_rejected_types(self):
        for adv_max in [2.0, "2h", True]:
            with pytest.raises(TypeError, match="adv_max"):
                _Dummy(req_timestamps=TS_DT64, adv_max=adv_max)

    def test_bbox_buffer(self, caplog):
        d = _Dummy(req_timestamps=TS_DT64, adv_max=1, req_bbox=BBOX)
        assert d.req_bbox == BBOX
        d.add_buffer_to_bbox(1.0)
        assert d.req_bbox == [-11.0, 11.0, 39.0, 61.0]

        d = _Dummy(req_timestamps=TS_DT64, adv_max=1, req_bbox=BBOX, add_bbox_buffer=2.0)
        assert d.req_bbox == [-12.0, 12.0, 38.0, 62.0]

        with caplog.at_level(logging.WARNING):
            _Dummy().add_buffer_to_bbox(1.0)
        assert "bounding box" in caplog.text.lower()

    def test_bad_req_timestamps_raise(self, tmp_path):
        from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

        with pytest.raises(ValueError, match="empty"):
            _Dummy(req_timestamps=[])
        with pytest.raises(ValueError, match="datetime64"):
            _Dummy(req_timestamps=[12345])
        # Non-consecutive dates raise instead of silently downloading the gap
        with pytest.raises(ValueError, match="consecutive"):
            ContrailsAPIFlightData(
                req_timestamps=["2026-01-01T00", "2026-01-03T00"],
                adv_max=1,
                save_path=tmp_path,
            )


# ── Column normalisation ──────────────────────────────────────────────────────

STANDARD_COLUMNS = {"flight_id", "time", "latitude", "longitude", "altitude_ft"}


def _raw(id_col: str, altitude_col: str) -> pd.DataFrame:
    """Two waypoints of one flight under a provider's own column names, plus a duplicate."""
    return pd.DataFrame(
        {
            id_col: ["abc123", "abc123", "abc123"],
            "timestamp": pd.to_datetime(
                ["2024-06-01T00:00", "2024-06-01T00:00", "2024-06-01T00:01"]
            ),
            "latitude": [50.0, 50.0, 50.1],
            "longitude": [0.0, 0.0, 0.1],
            altitude_col: [35000.0, 35000.0, 35000.0],
        }
    )


class TestNormalizeColumns:
    def test_existing_flight_id_is_kept(self):
        from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

        flights = object.__new__(ContrailsAPIFlightData)
        flights.flight_data = _raw("icao_address", "altitude_baro").assign(flight_id="API1")
        flights._normalize_columns()
        assert set(flights.flight_data["flight_id"]) == {"API1"}
        assert "icao_address" in flights.flight_data.columns

    def test_api_prefers_gnss_altitude_and_fills_from_baro(self):
        from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

        flights = object.__new__(ContrailsAPIFlightData)
        flights.flight_data = _raw("icao_address", "altitude_gnss").assign(
            altitude_baro=[1.0, 2.0, 3.0], airline_iata="BA"
        )
        flights.flight_data.loc[2, "altitude_gnss"] = np.nan
        flights._normalize_columns()
        assert STANDARD_COLUMNS <= set(flights.flight_data.columns)
        assert "airline_iata" in flights.flight_data.columns  # not relabelled
        assert flights.flight_data["altitude_ft"].tolist() == [35000.0, 35000.0, 3.0]
        # Barometric altitude for the wind level, filled from GNSS where missing
        assert flights.flight_data["pressure_altitude_ft"].tolist() == [1.0, 2.0, 3.0]
        assert "altitude_gnss" not in flights.flight_data.columns

    def test_api_baro_only_serves_both_altitudes(self):
        from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

        flights = object.__new__(ContrailsAPIFlightData)
        flights.flight_data = _raw("icao_address", "altitude_baro")
        flights._normalize_columns()
        assert flights.flight_data["altitude_ft"].tolist() == [35000.0] * 3
        assert flights.flight_data["pressure_altitude_ft"].tolist() == [35000.0] * 3


# ── ContrailsAPIFlightData constructor ────────────────────────────────────────


def test_api_flight_data_sorts_and_prepends_lookback(tmp_path) -> None:
    """Timestamps are sorted before the lookback is prepended, and req_dates is set."""
    from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

    flight_data = ContrailsAPIFlightData(
        req_timestamps=["2026-01-02T03", "2026-01-01T12"],
        adv_max=2,
        save_path=tmp_path,
    )
    expected = np.array(["2026-01-01T10", "2026-01-01T12", "2026-01-02T03"], dtype="datetime64[h]")
    np.testing.assert_array_equal(flight_data.req_timestamps, expected)
    assert flight_data.adv_max == np.timedelta64(2, "h")
    assert flight_data.req_dates[0] == np.datetime64("2026-01-01")


# ── Live Contrails API (needs credentials) ────────────────────────────────────


@pytest.mark.integration
def test_api_flight_data_download() -> None:
    from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

    flights = ContrailsAPIFlightData(req_timestamps=["2026-01-01T00"], adv_max=1)
    flights.download_from_api(save_data=False)
    assert not flights.flight_data.empty
    assert STANDARD_COLUMNS <= set(flights.flight_data.columns)


# ── ContrailsAPIFlightData.download_from_api ─────────────────────────────────


class _FakeTelemetryAPI:
    """Stands in for requests.get: serves one waypoint per hour, failing chosen hours."""

    def __init__(self, fail=None):
        self.fail = fail or {}  # date_str -> exception to raise, or list of them in turn
        self.calls = []

    def __call__(self, url, timeout=None, headers=None, params=None):
        import io

        import requests

        date_str = params["date"]
        self.calls.append(date_str)
        failure = self.fail.get(date_str)
        if isinstance(failure, list):
            failure = failure.pop(0) if failure else None
        if failure is not None:
            raise failure
        df = pd.DataFrame(
            {
                "icao_address": ["abc123"],
                "timestamp": [pd.Timestamp(date_str)],
                "latitude": [50.0],
                "longitude": [0.0],
                "altitude_baro": [35000.0],
            }
        )
        buf = io.BytesIO()
        df.to_parquet(buf)
        resp = requests.Response()
        resp.status_code = 200
        resp._content = buf.getvalue()
        return resp


def _http(status):
    import requests

    resp = requests.Response()
    resp.status_code = status
    resp._content = b"refused"
    return requests.HTTPError(response=resp)


@pytest.fixture
def api_flights(tmp_path, monkeypatch):
    from contrail_detect.flights.contrails_api import ContrailsAPIFlightData

    monkeypatch.setattr("contrail_detect.models.contrail_model_base.time.sleep", lambda s: None)
    return ContrailsAPIFlightData(
        req_timestamps=["2026-01-01T12:00", "2026-01-01T13:30"], adv_max=1, save_path=tmp_path
    )


class TestDownloadFromAPI:
    def test_downloads_every_hour_and_replaces_on_repeat(self, api_flights, monkeypatch):
        api = _FakeTelemetryAPI()
        monkeypatch.setattr("requests.get", api)
        api_flights.download_from_api(save_data=False)
        assert api.calls == ["2026-01-01T11", "2026-01-01T12", "2026-01-01T13"]
        assert len(api_flights.flight_data) == 3
        api_flights.download_from_api(save_data=False)
        assert len(api_flights.flight_data) == 3

    def test_transient_failures_are_retried(self, api_flights, monkeypatch):
        import requests

        api = _FakeTelemetryAPI(
            fail={"2026-01-01T12": [_http(503), requests.ConnectionError(), _http(429)]}
        )
        monkeypatch.setattr("requests.get", api)
        api_flights.download_from_api(save_data=False)
        assert len(api_flights.flight_data) == 3

    def test_missing_hour_stops_the_run_unless_allowed(self, api_flights, monkeypatch, caplog):
        monkeypatch.setattr("requests.get", _FakeTelemetryAPI(fail={"2026-01-01T12": _http(403)}))
        with pytest.raises(RuntimeError, match="2026-01-01T12"):
            api_flights.download_from_api(save_data=False)
        with caplog.at_level(logging.WARNING):
            api_flights.download_from_api(save_data=False, allow_missing_hours=True)
        assert len(api_flights.flight_data) == 2
        assert "Continuing without them" in caplog.text


@pytest.mark.parametrize("peak, warns", [(12500.0, True), (39000.0, False)])
def test_geometric_height_in_metres_warns(peak, warns, caplog):
    d = _Dummy()
    d.flight_data = pd.DataFrame(
        {
            "flight_id": ["A", "A"],
            "time": pd.to_datetime(["2024-06-01T00:00", "2024-06-01T00:01"]),
            "latitude": [50.0, 50.1],
            "longitude": [0.0, 0.1],
            "geometric_height": [peak - 100.0, peak],
        }
    )
    with caplog.at_level(logging.WARNING):
        d._validate_flight_data_columns()
    assert ("looks like metres" in caplog.text) is warns
    assert d.flight_data["altitude_ft"].max() == peak
