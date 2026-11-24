"""Unit tests for contrail_detect.geostationary.validation_utils.

All functions are pure input-validation routines with no I/O or credentials
required — no mocking needed.
"""

import numpy as np
import pytest

from contrail_detect.geostationary.validation_utils import (
    validate_extent,
    validate_overlap,
    validate_satellite_inputs,
    validate_timestamps,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_SATELLITE_DATES = {
    "GOES-16": [np.datetime64("2017-01-01"), np.datetime64("2030-12-31")],
}

_DOMAIN = dict(
    min_longitude=-135.0,
    max_longitude=-15.0,
    min_latitude=15.0,
    max_latitude=65.0,
)


# ── validate_timestamps ───────────────────────────────────────────────────────


class TestValidateTimestamps:
    def test_valid_timestamps_pass(self):
        ts = [np.datetime64("2024-06-01"), np.datetime64("2024-06-02")]
        validate_timestamps(ts, "GOES-16", _SATELLITE_DATES)
        validate_timestamps([np.datetime64("2017-01-01")], "GOES-16", _SATELLITE_DATES)
        validate_timestamps(ts, "goes-16", _SATELLITE_DATES)


# ── validate_extent ───────────────────────────────────────────────────────────


class TestValidateExtent:
    def test_valid_extent_passes(self):
        validate_extent((-130.0, -20.0, 20.0, 60.0), **_DOMAIN)
        validate_extent((-135.0, -15.0, 15.0, 65.0), **_DOMAIN)

    def test_extent_outside_domain_raises(self):
        with pytest.raises(ValueError, match="lon"):
            validate_extent((-140.0, -20.0, 20.0, 60.0), **_DOMAIN)
        with pytest.raises(ValueError, match="-136"):
            validate_extent((-136.0, -20.0, 20.0, 60.0), **_DOMAIN)
        with pytest.raises(ValueError, match="lon"):
            validate_extent((-130.0, -10.0, 20.0, 60.0), **_DOMAIN)
        with pytest.raises(ValueError, match="lat"):
            validate_extent((-130.0, -20.0, 10.0, 60.0), **_DOMAIN)
        with pytest.raises(ValueError, match="lat"):
            validate_extent((-130.0, -20.0, 50.0, 30.0), **_DOMAIN)
        with pytest.raises(ValueError, match="lat"):
            validate_extent((-130.0, -20.0, 20.0, 70.0), **_DOMAIN)


# ── validate_overlap ──────────────────────────────────────────────────────────


class TestValidateOverlap:
    def test_valid_overlap_passes(self):
        validate_overlap(0.0, tile_size=50.0)
        validate_overlap(10.0, tile_size=50.0)
        # 2 * overlap == tile_size is on the boundary — strictly > check, so allowed
        validate_overlap(25.0, tile_size=50.0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_overlap(-1.0, tile_size=50.0)
        with pytest.raises(ValueError):
            validate_overlap(30.0, tile_size=50.0)


# ── validate_satellite_inputs ─────────────────────────────────────────────────


class TestValidateSatelliteInputs:
    _REGIONS = {"F", "C", "M1", "M2"}
    _SATELLITES = {"GOES-16", "GOES-19"}

    def test_satellite_inputs(self):
        validate_satellite_inputs("F", "GOES-16", self._REGIONS, self._SATELLITES)
        with pytest.raises(ValueError, match="region"):
            validate_satellite_inputs("X", "GOES-16", self._REGIONS, self._SATELLITES)
        with pytest.raises(ValueError, match="satellite"):
            validate_satellite_inputs("F", "GOES-18", self._REGIONS, self._SATELLITES)
        with pytest.raises(ValueError, match="GOES-16"):
            validate_satellite_inputs("F", "UNKNOWN", self._REGIONS, self._SATELLITES)
