"""Test functionalities of src/utils/frame_processing.py"""

from unittest import TestCase

import numpy as np

from src.utils.frame_processing import find_outliers, handle_outliers


class TestFindOutliers(TestCase):
    """Test find_outliers function."""

    def test_find_outliers_no_outlier(self):
        """Test that no outliers are found."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [9, 5]])
        outlier_indices = find_outliers(contour)
        self.assertTrue(len(outlier_indices) == 0)

    def test_find_outliers_end(self):
        """Test that an outlier at the end of the contour is found."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [14, 8]])
        outlier_indices = find_outliers(contour)
        self.assertTrue(len(outlier_indices) == 1)
        self.assertTrue(outlier_indices[0] == 5)

    def test_find_outliers_middle(self):
        """Test that an outlier in the middle of the contour is found."""
        contour = np.array([[8, 1], [8, 2], [13, 7], [9, 3], [9, 4], [9, 5]])
        outlier_indices = find_outliers(contour)
        self.assertTrue(len(outlier_indices) == 1)
        self.assertTrue(outlier_indices[0] == 2)


class TestHandleOutliers(TestCase):
    """Test function handle_outliers."""

    def test_handle_outliers_none(self):
        """Test that no point is removed when no outlier is present."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [9, 5]])
        contour_filtered = handle_outliers(contour)
        self.assertTrue(np.array_equal(contour, contour_filtered))

    def test_handle_outliers_end(self):
        """Test that the last point is removed."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [14, 8]])
        contour_filtered = handle_outliers(contour)
        self.assertTrue(np.array_equal(contour[:-1], contour_filtered))

    def test_handle_outliers_end_prev(self):
        """Test that the last two points are removed."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [14, 7], [14, 8]])
        contour_filtered = handle_outliers(contour)
        self.assertTrue(np.array_equal(contour[:-2], contour_filtered))

    def test_handle_outliers_middle(self):
        """Test that the part of the contour after the outlier is reversed and prepended."""
        contour = np.array(
            [
                [609, 441],
                [592, 448],
                [577, 461],
                [567, 477],
                [564, 494],
                [564, 511],
                [567, 528],
                [571, 545],
                [574, 562],
                [580, 579],
                [584, 596],
                [590, 613],
                [594, 630],
                [600, 647],
                [607, 664],  # Outlier part starts.
                [610, 441],
                [627, 443],
                [644, 450],
                [649, 452],
            ]
        )
        contour_filtered = handle_outliers(contour)
        contour_expected = np.array(
            [
                [649, 452],
                [644, 450],
                [627, 443],
                [610, 441],  # Reversed and concatenated to front.
                [609, 441],
                [592, 448],
                [577, 461],
                [567, 477],
                [564, 494],
                [564, 511],
                [567, 528],
                [571, 545],
                [574, 562],
                [580, 579],
                [584, 596],
                [590, 613],
                [594, 630],
                [600, 647],
                [607, 664],
            ]
        )
        self.assertTrue(np.array_equal(contour_filtered, contour_expected))
