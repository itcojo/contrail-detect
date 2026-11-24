"""Unit tests for contrail_detect.decomposition.decompose.

Covers ContrailDecomposer on synthetic shapes (one contrail per line, crossings and sharp
elbows split, short spurs pruned), regions too small to skeletonize, and the Decomposer
pipeline from a detection mask to a contrail summary and point clouds.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation
from skimage.draw import line

from contrail_detect.decomposition.branch_operations import _path_branch_ids, build_branch_graph
from contrail_detect.decomposition.decompose import (
    ContrailDecomposer,
    Decomposer,
    DecompositionConfig,
)
from contrail_detect.geostationary.goes import GOESDetector


def _draw(shape: tuple[int, int], *segments: tuple[int, int, int, int]) -> np.ndarray:
    """Mask of 3-pixel-wide lines, each given as (row0, col0, row1, col1)."""
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in segments:
        rr, cc = line(r0, c0, r1, c1)
        mask[rr, cc] = True
    return binary_dilation(mask).astype(float)


STRAIGHT = _draw((20, 60), (10, 5, 10, 54))
CROSS_A, CROSS_B = (5, 5, 54, 54), (5, 54, 54, 5)
CROSS = _draw((60, 60), CROSS_A, CROSS_B)


# ── ContrailDecomposer ────────────────────────────────────────────────────────


class TestContrailDecomposer:
    @pytest.mark.parametrize(
        "mask",
        [
            STRAIGHT,
            _draw((50, 50), (5, 5, 44, 44)),  # diagonal
            _draw((40, 80), (20, 5, 20, 40), (20, 40, 24, 75)),  # gentle bend
            _draw((40, 80), (20, 5, 20, 74), (20, 40, 17, 42)),  # short spur
        ],
        ids=["straight", "diagonal", "gentle_bend", "short_spur"],
    )
    def test_single_contrail_kept_whole(self, mask):
        results = ContrailDecomposer(mask).get_contrail_detection_arrays()
        assert len(results) == 1
        contrail, _ = results[0]
        assert (contrail > 0).sum() >= 0.98 * (mask > 0).sum()

    def test_crossing_contrails_are_separated(self):
        results = ContrailDecomposer(CROSS).get_contrail_detection_arrays()
        assert len(results) == 2
        lines = [_draw((60, 60), CROSS_A) > 0, _draw((60, 60), CROSS_B) > 0]
        for contrail, _ in results:
            pixels = contrail > 0
            # Each output follows one diagonal; only the crossing is shared
            overlaps = sorted((pixels & line_px).sum() / pixels.sum() for line_px in lines)
            assert overlaps[1] > 0.95
            assert overlaps[0] < 0.2

    def test_t_junction_gives_two_contrails(self):
        mask = _draw((60, 80), (10, 5, 10, 74), (10, 40, 55, 40))
        assert len(ContrailDecomposer(mask).get_contrail_detection_arrays()) == 2

    def test_sharp_elbow_is_split(self):
        mask = _draw((60, 60), (5, 5, 30, 30), (30, 30, 5, 55))
        results = ContrailDecomposer(mask).get_contrail_detection_arrays()
        assert len(results) == 2
        cols = sorted(np.nonzero(contrail)[1].mean() for contrail, _ in results)
        assert cols[0] < 30 < cols[1]  # one arm each side of the elbow
        config = DecompositionConfig(elbow_threshold=1e6)
        assert len(ContrailDecomposer(mask, config).get_contrail_detection_arrays()) == 1

    def test_detection_scores_are_carried_over(self):
        scores = STRAIGHT * np.linspace(0.3, 0.9, STRAIGHT.shape[1])
        contrail, _ = ContrailDecomposer(scores).get_contrail_detection_arrays()[0]
        on = contrail > 0
        np.testing.assert_array_equal(contrail[on], scores[on])

    def test_rejects_non_2d_input(self):
        for bad in ([[1.0]], np.ones(3), np.ones((2, 2, 2))):
            with pytest.raises(ValueError):
                ContrailDecomposer(bad)


# ── Regions too small to skeletonize ──────────────────────────────────────────


# Real GOES-East regions that skeletonize to one pixel
SMALL_BLOB = np.array(
    [[0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]],
    dtype=float,
)
DIAGONAL_BAND = np.array(
    [[1.0 if 0 <= c - r + 1 <= 3 else 0.0 for c in range(13)] for r in range(13)]
)
DIAGONAL_BAND[0, 0] = 0.0


@pytest.mark.parametrize("mask", [SMALL_BLOB, DIAGONAL_BAND, np.zeros((3, 3))])
def test_degenerate_skeleton_has_no_decomposition(mask, caplog):
    with caplog.at_level(logging.WARNING):
        assert ContrailDecomposer(mask).get_contrail_detection_arrays() == []
    assert not caplog.records


# Real GOES-East region whose skeleton is only short type I branches
ALL_SHORT_BRANCHES = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ],
    dtype=float,
)


def test_pruning_every_branch_raises_a_clear_error():
    with pytest.raises(ValueError, match="pruning would leave no skeleton"):
        ContrailDecomposer(ALL_SHORT_BRANCHES).get_contrail_detection_arrays()


def test_refinement_retains_degenerate_region(caplog):
    timestamp = np.datetime64("2026-01-09T15:00")
    decomposer = object.__new__(Decomposer)
    decomposer.detector = SimpleNamespace(timestamp_sequence=[timestamp])
    decomposer.decomposition_config_kwargs = {}
    decomposer.all_regions = pd.DataFrame(
        [{"timestamp": timestamp, "region_detection_array": SMALL_BLOB, "bbox": (10, 20, 15, 23)}]
    )

    with caplog.at_level(logging.WARNING):
        decomposer._region_refinement()

    assert not caplog.records
    assert len(decomposer.all_regions) == 1
    row = decomposer.all_regions.iloc[0]
    np.testing.assert_array_equal(row["region_detection_array"], SMALL_BLOB)
    assert not row["originally_complex"]


# ── Decomposer pipeline ───────────────────────────────────────────────────────

_T0, _T1 = np.datetime64("2026-01-09T15:00"), np.datetime64("2026-01-09T15:10")


def _scene() -> np.ndarray:
    """A crossing pair, a lone contrail, a blob below min_area and sub-threshold noise."""
    scene = np.zeros((100, 120))
    scene[:60, :60] = CROSS * 0.8
    scene[70:90, 50:110] = STRAIGHT * 0.6
    scene[5:7, 100:102] = 0.9  # 4 pixels, below min_area
    scene[95:, :] = 0.1  # below prediction_threshold
    return scene


@pytest.fixture
def scene_detector(bare):
    """GOESDetector stand-in on a 0.01 degree lon/lat grid, with one detected frame."""
    lons, lats = np.meshgrid(np.linspace(-80, -78.81, 120), np.linspace(40, 39.01, 100))
    detector = bare(
        GOESDetector,
        satellite="GOES-16",
        region="F",
        detection_subdirectory="contrails_api",
        LONG_NAME="TEST",
        timestamp_sequence=[_T0, _T1],
        detections={_T0: _scene()},  # second frame has no prediction
        lons=lons,
        lats=lats,
        image_offset=(0, 0),
    )
    detector.__dict__["pixel_surface_area"] = np.ones_like(lons)
    detector.scan_time_offset_for_position = lambda lat, lon, timestamp=None: np.timedelta64(
        30, "s"
    )
    return detector


def _decompose(detector, out_dir, **kwargs):
    return Decomposer(detector, out_dir, **kwargs).gen_decomposed_regions()


class TestDecomposer:
    def test_scene_gives_one_row_per_contrail(self, scene_detector, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            summary, points = _decompose(scene_detector, tmp_path)

        assert "No detection scores found for timestamp 2026-01-09T15:10" in caplog.text
        assert len(summary) == 3
        assert summary["contrail_id"].is_unique
        assert set(points) == set(summary["contrail_id"])
        assert summary["originally_complex"].sum() == 2  # the two halves of the cross
        assert (summary["great_circle_length_km"] > 0).all()
        assert (summary["centroid_scantime"] == pd.Timestamp(_T0) + pd.Timedelta(30, "s")).all()
        scene = _scene()
        for rows_cols in points.values():
            assert (scene[rows_cols[:, 0], rows_cols[:, 1]] >= 0.25).all()

    def test_prediction_threshold_and_min_area(self, scene_detector, tmp_path):
        summary, _ = _decompose(scene_detector, tmp_path, prediction_threshold=0.7, min_area=500)
        assert summary.empty  # 0.7 drops the lone contrail; min_area drops the cross

    def test_empty_scene_returns_empty(self, scene_detector, tmp_path, caplog):
        scene_detector.detections = {_T0: np.zeros((100, 120))}
        with caplog.at_level(logging.WARNING):
            summary, points = _decompose(scene_detector, tmp_path)
        assert summary.empty and points == {}
        assert "No regions proposed" in caplog.text

    def test_no_detections_raises_with_remedy(self, scene_detector, tmp_path):
        scene_detector.detections = {}
        with pytest.raises(ValueError, match="detector.detections = detector.get_detections"):
            _decompose(scene_detector, tmp_path)

    def test_output_paths(self, scene_detector, tmp_path):
        decomposer = Decomposer(scene_detector, str(tmp_path), prediction_threshold=0.5)
        assert decomposer.points_path == scene_detector._get_point_clouds_path(
            tmp_path, method="decomposer", prediction_threshold=0.5
        )
        assert decomposer.summary_path == scene_detector._get_line_segments_path(
            tmp_path, method="decomposer", prediction_threshold=0.5
        )
        assert isinstance(decomposer.output_dir, Path)
        default = Decomposer(scene_detector, tmp_path).output_dir
        assert Decomposer(scene_detector, tmp_path, elbow_threshold=0.5).output_dir != default
        scene_detector.tile_extent = np.array([-80.0, -79.0, 39.0, 40.0])
        assert Decomposer(scene_detector, tmp_path).output_dir != default


class TestDecomposerCacheAndPositions:
    def test_other_settings_are_not_served_from_cache(self, scene_detector, tmp_path):
        first, _ = _decompose(scene_detector, tmp_path)
        scene_detector.detections = {_T0: np.zeros((100, 120))}
        other, _ = _decompose(scene_detector, tmp_path, min_area=11)
        assert other.empty  # recomputed on the new detections, not read back
        again, points = _decompose(scene_detector, tmp_path)
        assert sorted(again["contrail_id"]) == sorted(first["contrail_id"])
        assert len(points) == 3

    def test_unknown_setting_raises(self, scene_detector, tmp_path):
        with pytest.raises(TypeError):
            Decomposer(scene_detector, tmp_path, not_a_setting=1)

    def test_repeated_ids_are_numbered(self, scene_detector, tmp_path):
        decomposer = Decomposer(scene_detector, tmp_path)
        region = {
            "timestamp": _T0,
            "region_detection_array": STRAIGHT * 0.6,
            "bbox": (70, 50, 90, 110),
        }
        decomposer.all_regions = pd.DataFrame([region, region])
        decomposer._contrails_from_regions()
        ids = list(decomposer.all_regions["contrail_id"])
        assert ids[1] == ids[0] + "-2"
        assert set(decomposer.contrail_index_points) == set(ids)

    def test_ids_reproducible_and_image_offset_not_added(self, scene_detector, tmp_path):
        reference, ref_points = _decompose(scene_detector, tmp_path / "a")
        scene_detector.image_offset = (30, 10)
        shifted, points = _decompose(scene_detector, tmp_path / "b")
        cols = ["contrail_id", "centroid_lat", "centroid_lon", "great_circle_length_km"]
        pd.testing.assert_frame_equal(
            reference[cols].sort_values("contrail_id").reset_index(drop=True),
            shifted[cols].sort_values("contrail_id").reset_index(drop=True),
        )
        for cid, rows_cols in points.items():
            np.testing.assert_array_equal(rows_cols, ref_points[cid])

    def test_contrail_across_date_line(self, scene_detector, tmp_path):
        reference, _ = _decompose(scene_detector, tmp_path / "a")
        # Shift columns 0-119 to run 179.40 to 180.59, wrapped into [-180, 180)
        lons = scene_detector.lons + 259.40
        scene_detector.lons = (lons + 180.0) % 360.0 - 180.0
        crossing, _ = _decompose(scene_detector, tmp_path / "b")
        assert (crossing["centroid_lon"] >= -180).all() and (crossing["centroid_lon"] < 180).all()
        # The lone contrail (cols 50-110) spans 179.9 to 180.5
        lone = crossing.loc[crossing["centroid_lat"] < 39.4].iloc[0]
        lone_ref = reference.loc[reference["centroid_lat"] < 39.4].iloc[0]
        assert lone["centroid_lon"] == pytest.approx(180.2 - 360.0, abs=0.02)
        assert lone["endpoint1_lon"] * lone["endpoint2_lon"] < 0  # one end either side
        assert lone["great_circle_length_km"] == pytest.approx(
            lone_ref["great_circle_length_km"], abs=0.05
        )


# ── Branch joining ────────────────────────────────────────────────────────────


def _fork_mask() -> np.ndarray:
    """A straight line with two short arms leaving one junction about 28 degrees apart."""
    img = np.zeros((100, 140))
    junction = (50, 40)

    def draw(end):
        rr, cc = line(*junction, int(round(end[0])), int(round(end[1])))
        img[rr, cc] = 1

    draw((50, 2))
    draw((50, 100))
    for deg in (-14, 14):
        rad = np.deg2rad(deg)
        draw((50 - 18 * np.sin(rad), 40 + 18 * np.cos(rad)))
    return img


class TestBranchJoining:
    def test_narrow_fork_arms_stay_separate(self):
        decomposer = ContrailDecomposer(_fork_mask())
        sizes = sorted(int(s.sum()) for s in decomposer.contrail_skeletons.values())
        types = sorted(decomposer.contrail_types.values())
        # the straight line and each arm separately
        assert len(sizes) == 3
        assert types == [1, 1, 2]

    def test_parallel_branches_use_shorter(self):
        summary = pd.DataFrame(
            {
                "branch_id": [0, 1, 2, 3],
                "node_id_src": [10, 11, 11, 12],
                "node_id_dst": [11, 12, 12, 13],
                "branch_distance": [40.0, 45.0, 30.0, 40.0],
            }
        )
        graph = build_branch_graph(summary)
        assert graph.number_of_edges() == 4  # both halves of the loop kept
        assert _path_branch_ids(graph, [10, 11, 12, 13]) == [0, 2, 3]  # shorter half
        # a tie goes to the lower branch ID, regardless of row order
        tied = summary.assign(branch_distance=40.0).iloc[::-1]
        assert _path_branch_ids(build_branch_graph(tied), [11, 12]) == [1]
