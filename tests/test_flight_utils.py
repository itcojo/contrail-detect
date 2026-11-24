"""Unit tests for contrail_detect.flights.flight_utils.

Covers the pure functions that require no satellite detector or external data:
intersection_with_ellipsoid, merge_dataframes_with_index,
validate_and_parse_frame_timestamps, and resolve_adv_timestep.
"""

import numpy as np
import pandas as pd
import pytest

from contrail_detect.flights.flight_utils import (
    intersection_with_ellipsoid,
    merge_dataframes_with_index,
    resolve_adv_timestep,
    validate_and_parse_frame_timestamps,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _col(*xyz) -> np.ndarray:
    """Build a (3, 1) column vector from three scalars."""
    return np.array([[xyz[0]], [xyz[1]], [xyz[2]]], dtype=float)


def _cols(*points) -> np.ndarray:
    """Build a (3, N) array from N (x, y, z) tuples."""
    return np.array(points, dtype=float).T


# ── intersection_with_ellipsoid ───────────────────────────────────────────────


class TestIntersectionWithEllipsoid:
    """Unit sphere (a = b = 1) keeps the maths tractable."""

    a = b = 1.0

    def test_ray_misses_sphere_returns_nan(self):
        # Ray at x=5, parallel to y-axis: never intersects unit sphere
        result = intersection_with_ellipsoid(_col(5, 0, 0), _col(5, 1, 0), self.a, self.b)
        assert np.all(np.isnan(result))

    def test_near_side_point_found_far_side_point_nan(self):
        # Satellite above north pole at (0, 0, 3)
        # Column 0: aircraft inside sphere at (0, 0, 0.8), between satellite and surface;
        #   expected intersection: north pole (0, 0, 1)
        # Column 1: line from (0,0,3) to (0,0,-2) exits the sphere at (0,0,-1); t0 < 1
        p0 = _cols((0, 0, 3), (0, 0, 3))
        p1 = _cols((0, 0, 0.8), (0, 0, -2))
        result = intersection_with_ellipsoid(p0, p1, self.a, self.b)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert result[1, 0] == pytest.approx(0.0, abs=1e-9)
        assert result[2, 0] == pytest.approx(1.0, rel=1e-6)
        assert np.all(np.isnan(result[:, 1]))

    def test_oblate_spheroid_equatorial_intersection(self):
        # Use WGS84 semi-axes in km (approximate)
        a, b = 6378.137, 6356.752
        # Satellite at (42_164, 0, 0) km (GEO orbit), aircraft at (6_800, 0, 0)
        # Expected intersection close to equatorial radius (6378 km)
        result = intersection_with_ellipsoid(_col(42_164, 0, 0), _col(6_800, 0, 0), a, b)
        assert not np.any(np.isnan(result))
        # Intersection x-component should be ≈ a (equatorial radius)
        assert result[0, 0] == pytest.approx(a, rel=1e-4)


# ── merge_dataframes_with_index ───────────────────────────────────────────────


class TestMergeDataframesWithIndex:
    def test_merge(self):
        dfs = [
            pd.DataFrame({"lat": [50.0, 51.0], "lon": [-1.0, -2.0]}),
            pd.DataFrame(),
            pd.DataFrame({"lat": [52.0], "lon": [-3.0]}),
        ]
        result = merge_dataframes_with_index(dfs, advection_step=5)
        assert len(result) == 3
        assert list(result["adv_frame"]) == [0, 0, 2]
        assert all(result["advection_step"] == 5)
        assert list(result.index) == [0, 1, 2]
        assert list(result["lat"]) == [50.0, 51.0, 52.0]
        assert list(result["lon"]) == [-1.0, -2.0, -3.0]

        # Every frame empty: an empty frame with the metadata columns, not a crash
        result = merge_dataframes_with_index([pd.DataFrame({"lat": []}), pd.DataFrame()])
        assert result.empty
        assert {"lat", "adv_frame", "advection_step"} <= set(result.columns)


# ── validate_and_parse_frame_timestamps ──────────────────────────────────────


class TestValidateAndParseFrameTimestamps:
    def _ts(self, *hours) -> list[np.datetime64]:
        return [np.datetime64(f"2026-01-01T{h:02d}:00") for h in hours]

    def test_parse(self):
        ts, interval = validate_and_parse_frame_timestamps(self._ts(3, 1, 1, 0, 2, 0))
        assert list(ts) == self._ts(0, 1, 2, 3)
        assert interval == np.timedelta64(1, "h")

        ts, interval = validate_and_parse_frame_timestamps(self._ts(0))
        assert interval is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="uniformly"):
            validate_and_parse_frame_timestamps(self._ts(0, 1, 3))
        with pytest.raises(ValueError, match="two"):
            validate_and_parse_frame_timestamps(self._ts(0), require_multiple=True)


# ── resolve_adv_timestep ──────────────────────────────────────────────────────


class TestResolveAdvTimestep:
    _interval = np.timedelta64(60, "m")

    def test_resolve(self):
        assert resolve_adv_timestep(None, self._interval) == self._interval
        assert resolve_adv_timestep(15, self._interval) == np.timedelta64(15, "m")
        assert resolve_adv_timestep(20, self._interval) == np.timedelta64(20, "m")
        td = np.timedelta64(30, "m")
        assert resolve_adv_timestep(td, self._interval) == td

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="single frame"):
            resolve_adv_timestep(None, frame_interval=None)
        # 7 minutes does not divide 60-minute interval
        with pytest.raises(ValueError, match="divide"):
            resolve_adv_timestep(7, self._interval)
