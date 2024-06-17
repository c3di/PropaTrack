"""Test functionalities of src/utils/visualization.py"""

from unittest import TestCase

from src.utils.visualization import clamp


class TestClamp(TestCase):
    """Test the clamp function."""

    def test_smaller_than_min(self):
        """Test that a value smaller than the minimum is clamped to the minimum."""
        self.assertEqual(clamp(0, 1, 10), 1)

    def test_between_min_and_max(self):
        """Test that a value between the minimum and maximum is not changed."""
        self.assertEqual(clamp(5, 1, 10), 5)

    def test_greater_than_max(self):
        """Test that a value greater than the maximum is clamped to the maximum."""
        self.assertEqual(clamp(11, 1, 10), 10)
