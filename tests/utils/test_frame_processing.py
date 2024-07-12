"""Test functionalities of src/utils/frame_processing.py"""

from unittest import TestCase

import numpy as np

from src.utils.frame_processing import nearest_point


class TestNearestPoint(TestCase):
    """Test the nearest_point function."""

    def test_nearest_point_none(self):
        """Test that no point is found when no contours are present."""
        point = np.array([1, 1])
        contours = []
        p_nearest = nearest_point(point, contours)
        self.assertEqual(p_nearest, None)

    def test_nearest_point_simple(self):
        """Test that the nearest point is found correctly."""
        point = np.array([1, 1])
        contours = [np.array([[2, 1], [2, 2], [2, 3]]), np.array([[3, 1], [3, 2], [3, 3]])]
        p_nearest = nearest_point(point, contours)
        self.assertTrue(np.array_equal(p_nearest, np.array([2, 1])))

    def test_nearest_point_last_contour(self):
        """Test that the nearest point is found correctly when it is in the last contour."""
        point = np.array([5, 1])
        contours = [
            np.array([[2, 1], [2, 2], [2, 3]]),
            np.array([[3, 1], [3, 2], [3, 3]]),
            np.array([[4, 1], [4, 2], [4, 3]]),
        ]
        p_nearest = nearest_point(point, contours)
        self.assertTrue(np.array_equal(p_nearest, np.array([4, 1])))
