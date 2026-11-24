import numpy as np
import pytest

from contrail_detect.decomposition.geometry_utils import (
    calculate_angle_matrix,
    calculate_distance_matrix,
    calculate_vector_angle,
    create_skeleton_segments,
    find_elbow_point,
)


class TestCalculateVectorAngle:
    def test_angles(self):
        assert calculate_vector_angle(np.array([1, 0]), np.array([1, 0])) == pytest.approx(0.0)
        assert calculate_vector_angle(np.array([1, 0]), np.array([0, 1])) == pytest.approx(90.0)
        assert calculate_vector_angle(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(180.0)
        assert calculate_vector_angle(np.array([1, 0]), np.array([1, 1])) == pytest.approx(
            45.0, abs=1e-5
        )
        assert calculate_vector_angle(np.array([0, 0]), np.array([1, 0])) == 0.0


class TestCalculateDistanceMatrix:
    def test_distances(self):
        coords = np.array([[0, 0], [3, 4], [1, 1]])
        matrix = calculate_distance_matrix(coords)
        assert matrix[0, 1] == pytest.approx(5.0)
        np.testing.assert_array_almost_equal(matrix, matrix.T)
        np.testing.assert_array_almost_equal(np.diag(matrix), [0, 0, 0])

        matrix = calculate_distance_matrix(np.array([[5, 7]]))
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(0.0)


class TestCalculateAngleMatrix:
    def test_angles(self):
        # Branch directions (1, 0), (0, 1) and (-1, 0)
        loose = np.array([[1, 0], [0, 1], [-1, 0]])
        stem = np.zeros((3, 2), dtype=int)
        matrix = calculate_angle_matrix(loose, stem)
        assert matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(matrix), [0, 0, 0])
        assert matrix[0, 1] == pytest.approx(90.0)
        assert matrix[0, 2] == pytest.approx(180.0)
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 180)


class TestFindElbowPoint:
    def test_elbow(self):
        coords = np.array([[i, i] for i in range(10)])
        _, deviation = find_elbow_point(np.array([0.0, 0.0]), np.array([9.0, 9.0]), coords)
        assert deviation == pytest.approx(0.0, abs=1e-10)

        # coords are (row, col); src and dst are (x, y)
        coords = np.array([[0, 0], [5, 1], [0, 2]])
        idx, deviation = find_elbow_point(np.array([0.0, 0.0]), np.array([2.0, 0.0]), coords)
        assert isinstance(idx, (int, np.integer))
        assert isinstance(deviation, float)
        assert idx == 1
        assert deviation == pytest.approx(2.5)

        coords = np.array([[0, 0], [1, 1]])
        _, deviation = find_elbow_point(np.array([0.0, 0.0]), np.array([0.0, 0.0]), coords)
        assert deviation == 0.0


class TestCreateSkeletonSegments:
    def test_split(self):
        skeleton = np.zeros((5, 5))
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        s1, s2 = create_skeleton_segments(coords, elbow_idx=2, skeleton=skeleton)
        assert s1.shape == skeleton.shape
        assert s2.shape == skeleton.shape
        elbow = coords[2]
        assert s1[elbow[0], elbow[1]] == 1
        assert s2[elbow[0], elbow[1]] == 1
        assert set(np.unique(s1)).issubset({0, 1})
        assert set(np.unique(s2)).issubset({0, 1})
