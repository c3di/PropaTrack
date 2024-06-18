"""Test functionalities of src/utils/frame_processing.py"""

from unittest import TestCase

import numpy as np

from src.utils.frame_processing import binarize


class TestBinarize(TestCase):
    """Test the binarize function."""

    def test_binarize(self):
        """
        Test that the binarize function sets pixel values below a threshold to 0
        and pixel values above to 255.
        """
        frame0 = np.array([[0, 10], [20, 30]])
        frame1 = np.array([[40, 50], [60, 70]])
        binarize(frame0, frame1, 25)
        self.assertTrue(np.all(frame0 == np.array([[0, 0], [0, 255]])))
        self.assertTrue(np.all(frame1 == np.array([[255, 255], [255, 255]])))
