"""Unit tests for contrail_detect.tracking: geometry, costs, gates and whole-scene tracking."""

from types import SimpleNamespace

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from contrail_detect.tracking.tracking import (
    Contrail,
    KalmanFilterConfig,
    TrackedContrail,
    TrackingConfig,
    TrackingManager,
)
from contrail_detect.tracking.tracking_costs import LengthCost, OrientationCost
from contrail_detect.tracking.tracking_gates import (
    CentroidTravelDirectionGate,
    GateContext,
    OrientationChangeGate,
    ProportionalPositionChangeGate,
)
from contrail_detect.tracking.tracking_viz import (
    _filter_contrails_by_zoom_extent,
    _plot_frame_contrails,
    generate_tracking_GIF,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _contrail(cid, lon1, lat1, lon2, lat2, u=None, v=None):
    return Contrail(
        contrail_id=cid,
        endpoint1_lon=lon1,
        endpoint1_lat=lat1,
        endpoint2_lon=lon2,
        endpoint2_lat=lat2,
        u_init=u,
        v_init=v,
    )


def _track(contrail):
    return TrackedContrail(contrail, KalmanFilterConfig())


def _ctx(contrail, track):
    return GateContext(contrail=contrail, potential_tracked_contrail=track)


# ── Contrail dataclass ────────────────────────────────────────────────────────


class TestContrailProperties:
    def test_centroid(self):
        assert _contrail("a", 0.0, 40.0, 0.0, 60.0).centroid_lat == pytest.approx(50.0)
        assert _contrail("a", -20.0, 50.0, 20.0, 50.0).centroid_lon == pytest.approx(0.0)

    def test_length(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", 10.0, 50.0, -10.0, 50.0)
        assert c1.length > 0
        assert c1.length == pytest.approx(c2.length)

    def test_azimuth(self):
        # On the equator, forward azimuth of an E-W contrail is exactly 90°
        assert _contrail("a", -10.0, 0.0, 10.0, 0.0).azimuth == pytest.approx(90.0, abs=0.01)
        assert _contrail("a", 0.0, 40.0, 0.0, 50.0).azimuth == pytest.approx(0.0, abs=0.1)

    def test_reversed_wind(self):
        r = _contrail("a", 0.0, 50.0, 1.0, 51.0, u=5.0, v=-3.0).reversed_wind()
        assert r.u_init == pytest.approx(-5.0)
        assert r.v_init == pytest.approx(3.0)
        r = _contrail("a", 0.0, 50.0, 1.0, 51.0).reversed_wind()
        assert r.u_init is None
        assert r.v_init is None


# ── LengthCost ────────────────────────────────────────────────────────────────


class TestLengthCost:
    def test_length_cost_values(self):
        # Use N-S contrails shifted in longitude — identical lengths on a sphere
        c1 = _contrail("a", 0.0, 50.0, 0.0, 55.0)
        c2 = _contrail("b", 5.0, 50.0, 5.0, 55.0)
        assert LengthCost().compute(_ctx(c2, _track(c1))) == pytest.approx(0.0, abs=0.001)
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", -5.0, 50.0, 5.0, 50.0)
        assert LengthCost().compute(_ctx(c2, _track(c1))) == pytest.approx(0.5, abs=0.02)

    def test_length_cost_bounds(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", -3.0, 52.0, 3.0, 52.0)
        assert LengthCost().compute(_ctx(c2, _track(c1))) >= 0.0
        c2 = _contrail("b", -0.1, 50.0, 0.1, 50.0)  # much shorter
        assert LengthCost().compute(_ctx(c2, _track(c1))) <= 1.0


# ── OrientationCost ───────────────────────────────────────────────────────────


class TestOrientationCost:
    def test_same_or_reversed_orientation_returns_zero(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", -10.0, 51.0, 10.0, 51.0)
        assert OrientationCost().compute(_ctx(c2, _track(c1))) == pytest.approx(0.0, abs=0.01)
        # Use N-S contrails where reversal gives exactly 180° azimuth difference
        c1 = _contrail("a", 0.0, 50.0, 0.0, 55.0)  # azimuth ~0° (northward)
        c2 = _contrail("b", 5.0, 55.0, 5.0, 50.0)  # azimuth ~180° (southward)
        assert OrientationCost().compute(_ctx(c2, _track(c1))) == pytest.approx(0.0, abs=0.01)

    def test_orientation_cost_bounds(self):
        c1 = _contrail("a", 0.0, 40.0, 0.0, 50.0)  # N-S
        c2 = _contrail("b", -5.0, 50.0, 5.0, 50.0)  # E-W
        assert OrientationCost().compute(_ctx(c2, _track(c1))) >= 0.0
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)  # E-W ~90°
        c2 = _contrail("b", 0.0, 49.0, 0.0, 51.0)  # N-S ~0° — 90° apart
        cost = OrientationCost(normalizing_angle=30.0).compute(_ctx(c2, _track(c1)))
        assert cost == 1.0


# ── OrientationChangeGate ─────────────────────────────────────────────────────


class TestOrientationChangeGate:
    def test_same_or_reversed_orientation_allowed(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", -10.0, 51.0, 10.0, 51.0)
        assert OrientationChangeGate().allow(_ctx(c2, _track(c1)))
        # Contrails at 0° and 180° are the same orientation due to ambiguity
        c2 = _contrail("b", 10.0, 51.0, -10.0, 51.0)  # reversed endpoints
        assert OrientationChangeGate().allow(_ctx(c2, _track(c1)))

    def test_threshold_controls_orthogonal(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)  # E-W ~90°
        c2 = _contrail("b", 0.0, 49.0, 0.0, 51.0)  # N-S ~0°
        assert not OrientationChangeGate(max_orientation_change=30.0).allow(_ctx(c2, _track(c1)))
        assert OrientationChangeGate(max_orientation_change=91.0).allow(_ctx(c2, _track(c1)))


# ── ProportionalPositionChangeGate ────────────────────────────────────────────


class TestProportionalPositionChangeGate:
    def test_consistent_movement_allowed(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        far = _contrail("b", 0.0, 60.0, 1.0, 61.0)  # wildly different position
        assert ProportionalPositionChangeGate().allow(_ctx(far, _track(c1)))
        c2 = _contrail("b", -10.0, 50.1, 10.0, 50.1)
        c3 = _contrail("c", -10.0, 50.2, 10.0, 50.2)  # same step as c1→c2
        track = _track(c1)
        track.contrails.append(c2)
        assert ProportionalPositionChangeGate().allow(_ctx(c3, track))

    def test_large_position_jump_rejected(self):
        c1 = _contrail("a", -10.0, 50.0, 10.0, 50.0)
        c2 = _contrail("b", -10.0, 50.01, 10.0, 50.01)  # tiny step
        c3 = _contrail("c", -10.0, 55.0, 10.0, 55.0)  # massive jump
        track = _track(c1)
        track.contrails.append(c2)
        assert not ProportionalPositionChangeGate(max_position_change=0.5).allow(_ctx(c3, track))


# ── CentroidTravelDirectionGate ───────────────────────────────────────────────


class TestCentroidTravelDirectionGate:
    def test_consistent_direction_allowed(self):
        single = _track(_contrail("a", -10.0, 50.0, 10.0, 50.0))
        far = _contrail("b", 0.0, 60.0, 1.0, 70.0)
        assert CentroidTravelDirectionGate().allow(_ctx(far, single))
        # Track moving northward; new contrail also to the north
        c1 = _contrail("a", 0.0, 50.0, 1.0, 50.0)
        c2 = _contrail("b", 0.0, 51.0, 1.0, 51.0)
        c3 = _contrail("c", 0.0, 52.0, 1.0, 52.0)
        track = _track(c1)
        track.contrails.append(c2)
        assert CentroidTravelDirectionGate().allow(_ctx(c3, track))

    def test_reversed_direction_rejected(self):
        # Track moving northward; new contrail jumps far south
        c1 = _contrail("a", 0.0, 50.0, 1.0, 50.0)
        c2 = _contrail("b", 0.0, 51.0, 1.0, 51.0)  # moved north
        c3 = _contrail("c", 0.0, 40.0, 1.0, 40.0)  # jumps far south — reversed
        track = _track(c1)
        track.contrails.append(c2)
        assert not CentroidTravelDirectionGate(max_direction_change=30.0).allow(_ctx(c3, track))


# ── TrackingManager ───────────────────────────────────────────────────────────

T0 = pd.Timestamp("2026-01-09T15:00")


def _deg_lon(metres: float, lat: float = 40.0) -> float:
    return metres / (111_320 * np.cos(np.radians(lat)))


def _obs(cid, frame, lon, lat=40.0, dlon=0.5, dlat=0.0, u=10.0, v=0.0, minutes=10) -> dict:
    """One contrail observation: a line of extent (dlon, dlat) centred on (lon, lat)."""
    return {
        "contrail_id": cid,
        "timestamp": T0 + pd.Timedelta(minutes=minutes * frame),
        "endpoint1_lon": lon - dlon / 2,
        "endpoint1_lat": lat - dlat / 2,
        "endpoint2_lon": lon + dlon / 2,
        "endpoint2_lat": lat + dlat / 2,
        "u_init": u,
        "v_init": v,
    }


STEP = _deg_lon(6000)  # 10 m/s eastward over a 10 minute frame


def _drifting(prefix: str, lon0: float, frames=range(4), **kwargs) -> list[dict]:
    return [_obs(f"{prefix}{i}", i, lon0 + i * STEP, **kwargs) for i in frames]


def _run_tracking(rows: list[dict], **config) -> dict[str, str]:
    return TrackingManager(pd.DataFrame(rows), TrackingConfig(**config)).run_tracking()


def _tracks(mapping: dict[str, str]) -> set[frozenset]:
    """Group contrail_ids by track, ignoring track_id naming."""
    groups: dict[str, set] = {}
    for cid, tid in mapping.items():
        groups.setdefault(tid, set()).add(cid)
    return {frozenset(g) for g in groups.values()}


class TestTrackingManager:
    @pytest.mark.parametrize("method", ["forward", "backward", "bidirectional"])
    def test_separate_contrails_give_separate_tracks(self, method):
        rows = (
            _drifting("a", -80.0)
            + _drifting("b", -78.0, lat=41.0, dlon=0.0, dlat=0.4)
            + [_obs("lone", 2, -75.0, lat=38.0)]
        )
        assert _tracks(_run_tracking(rows, method=method)) == {
            frozenset({"a0", "a1", "a2", "a3"}),
            frozenset({"b0", "b1", "b2", "b3"}),
        }  # the single-frame contrail is in no track

    def test_jump_starts_a_new_track(self):
        rows = _drifting("a", -80.0, frames=[0, 1]) + _drifting("b", -78.0, frames=[2, 3])
        assert _tracks(_run_tracking(rows)) == {frozenset({"a0", "a1"}), frozenset({"b2", "b3"})}

    def test_one_missed_frame_is_bridged(self):
        rows = _drifting("a", -80.0, frames=[0, 2, 3])
        assert _tracks(_run_tracking(rows)) == {frozenset({"a0", "a2", "a3"})}

    def test_longer_gap_closes_the_track(self):
        rows = _drifting("a", -80.0, frames=[0, 3, 4])
        assert _tracks(_run_tracking(rows)) == {frozenset({"a3", "a4"})}

    def test_wind_carries_fast_perpendicular_drift(self):
        # A north-south contrail blown 15 km east each frame, beyond the 10 km gate
        step = _deg_lon(15_000)
        rows = [_obs(f"f{i}", i, -80 + i * step, dlon=0.0, dlat=0.4, u=25.0) for i in range(4)]
        assert _tracks(_run_tracking(rows)) == {frozenset({"f0", "f1", "f2", "f3"})}
        assert _run_tracking([dict(r, u_init=0.0) for r in rows]) == {}

    def test_missing_wind_raises_unless_disabled(self):
        rows = pd.DataFrame(_drifting("a", -80.0)).drop(columns=["u_init", "v_init"])
        with pytest.raises(ValueError, match="u_init"):
            TrackingManager(rows)
        mapping = TrackingManager(rows, use_wind_initialization=False).run_tracking()
        assert len(mapping) == 4

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="fewer than 2 timestamps"):
            TrackingManager(pd.DataFrame([_obs("a0", 0, -80.0)]))
        manager = TrackingManager(pd.DataFrame(_drifting("a", -80.0)))
        manager.config = TrackingConfig(method="sideways")
        with pytest.raises(ValueError, match="Unknown tracking method"):
            manager.run_tracking()


def _wrap(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


class TestDateLine:
    def test_centroid_of_contrail_across_date_line(self):
        contrail = Contrail("x", 179.7, 40.0, -179.9, 40.0)
        assert contrail.centroid_lon == pytest.approx(179.9)

    @pytest.mark.parametrize("method", ["forward", "backward", "bidirectional"])
    def test_track_drifting_across_date_line(self, method):
        rows = [
            {
                **_obs(f"a{i}", i, 179.0 + i * STEP, dlon=0.4),
                "endpoint1_lon": _wrap(179.0 + i * STEP - 0.2),
                "endpoint2_lon": _wrap(179.0 + i * STEP + 0.2),
            }
            for i in range(20)
        ]
        assert any(r["endpoint1_lon"] > 0 > r["endpoint2_lon"] for r in rows)  # one spans it
        assert _tracks(_run_tracking(rows, method=method)) == {
            frozenset(f"a{i}" for i in range(20))
        }


class TestFrameTimes:
    def test_frame_off_the_grid_raises_unless_allowed(self):
        rows = _drifting("a", -80.0, frames=range(3)) + [_obs("b", 3.5, -70.0)]
        with pytest.raises(ValueError, match="not a whole number"):
            TrackingManager(pd.DataFrame(rows), TrackingConfig())
        rows = _drifting("a", -80.0, frames=range(3)) + [_obs("a3", 3.5, -80.0 + 3.5 * STEP)]
        manager = TrackingManager(
            pd.DataFrame(rows), TrackingConfig(), enforce_uniform_intervals=False
        )
        assert T0 + pd.Timedelta(minutes=35) in manager.frame_order
        assert _tracks(manager.run_tracking()) == {frozenset({"a0", "a1", "a2", "a3"})}


class TestTrackIds:
    @pytest.mark.parametrize("method", ["forward", "backward", "bidirectional"])
    def test_track_named_by_earliest_contrail(self, method):
        mapping = _run_tracking(_drifting("a", -80.0), method=method)
        assert set(mapping.values()) == {"a0"}

    @pytest.mark.parametrize("method", ["forward", "backward", "bidirectional"])
    def test_tracks_with_same_id_tails_stay_separate(self, method):
        rows = [
            dict(r, contrail_id=f"{p}-x{i}-abc123" if i == 0 else f"{p}-x{i}-def456")
            for p, lon in (("A", -80.0), ("B", -60.0))
            for i, r in enumerate(_drifting(p, lon, frames=range(2)))
        ]
        mapping = _run_tracking(rows, method=method)
        assert len(mapping) == 4
        assert len(set(mapping.values())) == 2


# ── Tracking figures ──────────────────────────────────────────────────────────


def _frame(lon0: float) -> pd.DataFrame:
    """One tracked contrail at one time, centred at ``lon0``."""
    return pd.DataFrame(
        {
            "track_id": ["a"],
            "timestamp": [pd.Timestamp("2024-05-01 12:00")],
            "centroid_lon": [_wrap(lon0)],
            "centroid_lat": [40.0],
            "endpoint1_lon": [_wrap(lon0 - 0.3)],
            "endpoint1_lat": [40.0],
            "endpoint2_lon": [_wrap(lon0 + 0.3)],
            "endpoint2_lat": [40.0],
        }
    )


class TestTrackingFigures:
    @staticmethod
    def _draw(contrails, extent):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        _plot_frame_contrails(
            ax, contrails, contrails["timestamp"].iloc[0], {"a": np.ones(4)}, extent
        )
        return fig, ax

    def test_segment_across_date_line(self):
        fig, ax = self._draw(_frame(179.9), [170.0, 180.0, 30.0, 50.0])
        segment = ax.lines[1]
        np.testing.assert_allclose(segment.get_xdata(), [179.6, 180.2])
        # Cut at the map edge into two short pieces
        path = segment.get_transform().transform_path(segment.get_path())
        x = ax.transData.inverted().transform(path.vertices)[:, 0]
        pieces = np.split(x, np.flatnonzero(path.codes == path.MOVETO)[1:])
        assert len(pieces) == 2 and all(np.ptp(p) < 0.5 for p in pieces)
        plt.close(fig)

    def test_segment_away_from_date_line(self):
        contrails = _frame(-80.0)
        fig, ax = self._draw(contrails, [-90.0, -70.0, 30.0, 50.0])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), contrails["centroid_lon"])
        np.testing.assert_array_equal(
            ax.lines[1].get_xdata(), contrails[["endpoint1_lon", "endpoint2_lon"]].iloc[0]
        )
        plt.close(fig)

    def test_zoom_filter_tests_every_detection(self):
        contrails = pd.concat([_frame(-80.0), _frame(-60.0)])
        kept = _filter_contrails_by_zoom_extent(contrails, [-90.0, -70.0, 30.0, 50.0])
        assert kept["centroid_lon"].tolist() == [-80.0]

    def test_generate_tracking_gif(self, tmp_path):
        contrails = _frame(-80.0)
        empty = SimpleNamespace(contrail_summary=contrails, timestamp_sequence=[])
        with pytest.raises(ValueError, match="timestamp_sequence is empty"):
            generate_tracking_GIF(empty, [-90.0, -70.0, 30.0, 50.0], display_gif=False)
        detector = SimpleNamespace(
            contrail_summary=contrails,
            timestamp_sequence=contrails["timestamp"].to_numpy(),
            satellite="GOES-16",
        )
        with plt.rc_context({"font.size": 11.0}):
            generate_tracking_GIF(
                detector,
                [-90.0, -70.0, 30.0, 50.0],
                dpi=40,
                display_gif=False,
                save_path=tmp_path / "tracks.gif",
            )
            assert plt.rcParams["font.size"] == 11.0
        assert (tmp_path / "tracks.gif").exists()
