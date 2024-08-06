"""Test functionalities of src/utils/frame_processing.py"""

from unittest import TestCase

import numpy as np

from src.utils.frame_processing import _find_outliers, _handle_outliers


class TestFindOutliers(TestCase):
    """Test _find_outliers function."""

    def test_find_outliers_no_outlier(self):
        """Test that no outliers are found."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [9, 5]])
        outlier_indices, mean_dist = _find_outliers(contour)
        self.assertEqual(outlier_indices.size, 0)
        self.assertEqual(mean_dist, 1.0)

    def test_find_outliers_end(self):
        """Test that an outlier at the end of the contour is found."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [14, 8]])
        outlier_indices, mean_dist = _find_outliers(contour)
        self.assertEqual(outlier_indices.size, 1)
        self.assertEqual(outlier_indices[0], 5.0)
        self.assertEqual(mean_dist, 2.1)

    def test_find_outliers_middle(self):
        """Test that an outlier in the middle of the contour is found."""
        contour = np.array([[8, 1], [8, 2], [13, 7], [9, 3], [9, 4], [9, 5]])
        outlier_indices, mean_dist = _find_outliers(contour)
        self.assertEqual(outlier_indices.size, 1)
        self.assertEqual(outlier_indices[0], 2.0)
        self.assertEqual(mean_dist, 3.1)


class TestHandleOutliers(TestCase):
    """Test function _handle_outliers."""

    def test_handle_outliers_none(self):
        """Test that no point is removed when no outlier is present."""
        contour = np.array([[8, 1], [8, 2], [8, 3], [9, 3], [9, 4], [9, 5]])
        contour_filtered = _handle_outliers(contour)
        self.assertTrue(np.array_equal(contour, contour_filtered))

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
        contour_filtered = _handle_outliers(contour)
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

    def test_handle_outliers_collision(self):
        """Test that the outliers created when two reaction fronts collide are removed."""
        contour = np.array(
            [
                [83, 4],
                [82, 25],
                [82, 46],
                [82, 67],
                [84, 88],
                [88, 109],
                [93, 130],
                [99, 151],
                [107, 172],
                [117, 193],
                [130, 214],
                [144, 234],
                [161, 254],
                [179, 272],
                [200, 289],
                [221, 303],
                [242, 315],
                [263, 324],
                [284, 332],
                [305, 341],
                [287, 352],
                [266, 360],
                [245, 371],
                [224, 385],
                [205, 400],
                [186, 421],
                [170, 442],
                [158, 463],
                [148, 484],
                [141, 505],
                [136, 526],
                [132, 547],
                [131, 568],
                [129, 589],
                [130, 610],
                [131, 631],
                [132, 652],
                [135, 673],
                [318, 341],
                [339, 343],
                [359, 341],
            ]
        )
        contour_expected = np.array(
            [
                [83, 4],
                [82, 25],
                [82, 46],
                [82, 67],
                [84, 88],
                [88, 109],
                [93, 130],
                [99, 151],
                [107, 172],
                [117, 193],
                [130, 214],
                [144, 234],
                [161, 254],
                [179, 272],
                [200, 289],
                [221, 303],
                [242, 315],
                [263, 324],
                [284, 332],
                [305, 341],
                [287, 352],
                [266, 360],
                [245, 371],
                [224, 385],
                [205, 400],
                [186, 421],
                [170, 442],
                [158, 463],
                [148, 484],
                [141, 505],
                [136, 526],
                [132, 547],
                [131, 568],
                [129, 589],
                [130, 610],
                [131, 631],
                [132, 652],
                [135, 673],
            ]
        )
        contour_filtered = _handle_outliers(contour)
        self.assertTrue(np.array_equal(contour_filtered, contour_expected))
