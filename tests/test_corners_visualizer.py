# tests/test_corners_visualizer.py
"""
Tests for corner visualizer.
"""
import numpy as np
import pytest

from app.features.corners.visualizer import (
    visualize_corners,
    _all_corners_detected,
    _get_corner_directions
)


def test_visualize_corners_with_none():
    """Test visualization with no corners."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = visualize_corners(image, None)

    assert result is not None
    assert result.shape == image.shape


def test_visualize_corners_with_data():
    """Test visualization with corner data."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90, 10)},
        'bottom_left': None,
        'bottom_right': None
    }

    result = visualize_corners(image, corners)
    assert result is not None


def test_all_corners_detected():
    """Test corner detection validation."""
    corners_complete = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90, 10)},
        'bottom_left': {'center': (10, 90)},
        'bottom_right': {'center': (90, 90)}
    }

    assert _all_corners_detected(corners_complete) == True

    corners_incomplete = {
        'top_left': {'center': (10, 10)},
        'top_right': None
    }

    assert _all_corners_detected(corners_incomplete) == False


def test_get_corner_directions():
    """Test corner direction calculation."""
    assert _get_corner_directions('top_left') == [(1, 0), (0, 1)]
    assert _get_corner_directions('bottom_right') == [(-1, 0), (0, -1)]
    assert _get_corner_directions('invalid') == []
