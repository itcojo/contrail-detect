"""Unit tests for abi_time_model.

Tests cover:
  - the bundled tables: source, shapes, orientation and swath rows
  - band_correction_s
  - check_timeline
  - pixel_time_offsets_s
"""

import numpy as np
import pytest

from contrail_detect.geostationary import abi_time_model as atm

# First row of swaths 2-22 in the 4 Sep 2020 FD table (the 31 Jul 2020 release has 174, 428, ...)
_FD_SWATH_STARTS = [162, 416, 669, 923, 1177, 1431, 1685, 1939, 2192, 2446, 2700,
                    2954, 3208, 3461, 3715, 3969, 4223, 4477, 4731, 4984, 5238]  # fmt: skip


def _swath_starts(table: np.ndarray) -> list[int]:
    """Rows where the median time jumps by more than 1 s."""
    med = np.median(table, axis=1)
    return (np.where(np.abs(np.diff(med)) > 1)[0] + 1).tolist()


# ── bundled tables ────────────────────────────────────────────────────────────


class TestBundledTables:
    def test_source_recorded(self):
        with np.load(atm.TABLE_PATH) as z:
            assert str(z["source_file"]) == "ABI-Timeline05B_Mode 6A_20190612-183017.nc"
            assert str(z["source_sha256"]) == (
                "be2dcd3dc7f4efbe0c309f63a9166d4d77a00468ea68fd3106e08126d77d9501"
            )
            assert str(z["source_creation_date"]) == "29-Aug-2020 12:22:16"

    @pytest.mark.parametrize(
        "scene, shape", [("FD", (5424, 5424)), ("CONUS", (1500, 2500)), ("MESO", (500, 500))]
    )
    def test_shapes_and_read_only(self, scene, shape):
        table = atm.load_table(scene)
        assert table.shape == shape
        assert table.dtype == np.float32
        assert not table.flags.writeable
        assert np.isfinite(table).all()

    def test_fd_swaths_are_the_4_september_release(self):
        assert _swath_starts(atm.load_table("FD")) == _FD_SWATH_STARTS

    @pytest.mark.parametrize("scene, n_swaths", [("FD", 22), ("CONUS", 6), ("MESO", 2)])
    def test_swaths_run_along_rows(self, scene, n_swaths):
        table = atm.load_table(scene)
        assert len(_swath_starts(table)) == n_swaths - 1
        assert (np.diff(np.median(table, axis=1)) >= -1e-3).all()

    def test_within_a_swath_time_runs_west_to_east(self):
        row = atm.load_table("FD")[2712]
        assert row[-1] - row[0] > 5.0  # equator, swept west to east

    def test_conus_and_meso_start_at_zero(self):
        for scene in ("CONUS", "MESO"):
            assert float(atm.load_table(scene).min()) == 0.0


# ── band_correction_s ─────────────────────────────────────────────────────────


class TestBandCorrection:
    def test_band_2_is_zero(self):
        assert atm.band_correction_s([2]) == 0.0

    def test_detection_bands(self):
        expected = (-0.535 + 0.319 - 0.256) / 3 + 0.055
        assert atm.band_correction_s([11, 14, 15]) == pytest.approx(expected)
        assert atm.band_correction_s([11, 14, 15]) == pytest.approx(-0.10233, abs=1e-5)

    @pytest.mark.parametrize("bands", [[], [0], [17]])
    def test_rejects_bad_bands(self, bands):
        with pytest.raises(ValueError):
            atm.band_correction_s(bands)


# ── check_timeline ────────────────────────────────────────────────────────────


class TestCheckTimeline:
    def test_mode_6_accepted(self):
        atm.check_timeline("ABI Mode 6")

    @pytest.mark.parametrize("timeline", ["ABI Mode 3", "ABI Mode 4", "ABI Mode 6A", ""])
    def test_other_modes_raise(self, timeline):
        with pytest.raises(ValueError, match="only 'ABI Mode 6'"):
            atm.check_timeline(timeline)


# ── pixel_time_offsets_s ──────────────────────────────────────────────────────


class TestPixelTimeOffsets:
    @pytest.mark.parametrize("satellite", ["GOES-16", "GOES-18", "GOES-19"])
    def test_table_plus_start_plus_correction(self, satellite):
        out = atm.pixel_time_offsets_s(satellite, "C", 77.43, [11, 14, 15], "ABI Mode 6")
        expected = atm.load_table("CONUS").astype(np.float64) + 77.43 - 0.10233
        np.testing.assert_allclose(out, expected, atol=1e-4)
        assert out.dtype == np.float64

    def test_regions_share_meso_table(self):
        m1 = atm.pixel_time_offsets_s("GOES-16", "M1", 0.0, [2])
        m2 = atm.pixel_time_offsets_s("GOES-16", "M2", 0.0, [2])
        np.testing.assert_array_equal(m1, m2)

    def test_full_disk_spans_the_scan(self):
        # G16 2025-04-07 15:00: scan start 20.89 s, C14 end 591.69 s
        out = atm.pixel_time_offsets_s("GOES-16", "F", 20.89, [14])
        assert out.min() == pytest.approx(20.89 - 4.14 + 0.374, abs=0.01)
        assert 585.0 < out.max() < 600.0

    def test_unsupported_satellite_raises(self):
        with pytest.raises(ValueError, match="GOES-17"):
            atm.pixel_time_offsets_s("GOES-17", "F", 30.0, [14])

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="region"):
            atm.pixel_time_offsets_s("GOES-16", "X", 20.0, [14])

    def test_non_mode_6_raises(self):
        with pytest.raises(ValueError, match="ABI Mode 4"):
            atm.pixel_time_offsets_s("GOES-16", "F", 20.0, [14], "ABI Mode 4")
