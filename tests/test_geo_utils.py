import numpy as np
import pytest
from pycontrails.physics.constants import radius_earth

from contrail_detect.geostationary.geo_utils import (
    azimuth,
    convert_cartesian_latlon,
    convert_latlon_cartesian,
    extents_match,
    haversine,
    normalize_range,
    same_location,
    unit_cross_product,
)


class TestConvertLatLonCartesian:
    def test_conversion(self):
        x, y, z = convert_latlon_cartesian(0.0, 0.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

        x, y, z = convert_latlon_cartesian(np.pi / 2, 0.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(1.0)

        lats, lons = np.radians([0.0, 45.0, 30.0]), np.radians([0.0, 90.0, 60.0])
        x, y, z = convert_latlon_cartesian(lats, lons)
        assert x.shape == (3,)
        np.testing.assert_allclose(np.sqrt(x**2 + y**2 + z**2), 1.0)
        lats2, lons2 = convert_cartesian_latlon(x, y, z)
        np.testing.assert_allclose(lats2, lats)
        np.testing.assert_allclose(lons2, lons)


class TestHaversine:
    def test_distance(self):
        assert haversine(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

        d = haversine(0.0, 0.0, np.pi, 0.0)
        assert d == pytest.approx(np.pi * radius_earth, rel=1e-5)

        # London to Paris: roughly 340 km
        d = haversine(-0.1278, 51.5074, 2.3522, 48.8566, input_unit="degrees")
        assert 330_000 < d < 350_000

        d_deg = haversine(10.0, 20.0, 30.0, 40.0, input_unit="degrees")
        d_rad = haversine(
            np.radians(10.0),
            np.radians(20.0),
            np.radians(30.0),
            np.radians(40.0),
            input_unit="radians",
        )
        assert d_deg == pytest.approx(d_rad)


class TestAzimuth:
    def test_azimuth(self):
        assert azimuth(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=0.1)
        assert azimuth(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=0.1)
        assert azimuth(0.0, 0.0, -1.0, 0.0) == pytest.approx(180.0, abs=0.1)
        assert 0 <= azimuth(45.0, 10.0, 30.0, 20.0) < 360


class TestNormalizeRange:
    def test_normalize(self):
        data = np.array([[0.0, 5.0, 10.0], [10.0, 5.0, 0.0]])
        result = normalize_range(data, (0.0, 10.0))
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])


class TestExtentsMatch:
    def test_extents_match(self):
        ext = (0.0, 1.0, 0.0, 1.0)
        assert extents_match(ext, ext)
        assert extents_match(ext, (0.0, 1.0 + 1e-5, 0.0, 1.0))
        e_inf = (float("-inf"), float("inf"), 0.0, 1.0)
        assert extents_match(e_inf, (float("-inf"), float("inf"), 0.0, 1.0))

    def test_extents_do_not_match(self):
        ext = (0.0, 1.0, 0.0, 1.0)
        assert not extents_match(None, ext)
        assert not extents_match(ext, None)
        assert not extents_match(ext, (0.0, 2.0, 0.0, 1.0))
        assert not extents_match((float("-inf"), 1.0, 0.0, 1.0), (float("inf"), 1.0, 0.0, 1.0))
        assert not extents_match((0.0, 1.0, np.nan, 3.0), (0.0, 1.0, 2.0, 3.0))


class TestSameLocation:
    def test_same_location(self):
        ext = np.array([0.0, 1.0, 0.0, 1.0])
        assert same_location(ext, ext)
        assert same_location(np.array([0.0, 1.0]), np.array([0.0, 1.0 + 1e-4]))
        assert not same_location(np.array([0.0, 1.0]), np.array([0.0, 2.0]))


class TestUnitCrossProduct:
    def test_known_result(self):
        v1 = np.array([2.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        result = unit_cross_product(v1, v2)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])
        assert np.linalg.norm(result) == pytest.approx(1.0)
        assert np.dot(result, v1) == pytest.approx(0.0, abs=1e-10)
        assert np.dot(result, v2) == pytest.approx(0.0, abs=1e-10)
