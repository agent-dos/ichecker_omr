# tests/test_corners_visualizer.py
"""
Tests for corner visualizer.
"""
import numpy as np
import pytest

from app.features.corners.visualizer import (
    visualize_corners,
    _all_corners_detected_and_valid,  # Fixed: correct function name
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


def test_all_corners_detected_and_valid():
    """Test corner detection validation."""
    corners_complete = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90, 10)},
        'bottom_left': {'center': (10, 90)},
        'bottom_right': {'center': (90, 90)}
    }

    assert _all_corners_detected_and_valid(corners_complete) == True

    corners_incomplete = {
        'top_left': {'center': (10, 10)},
        'top_right': None
    }

    assert _all_corners_detected_and_valid(corners_incomplete) == False


def test_all_corners_detected_with_invalid_structure():
    """Test corner validation with invalid data structures."""
    # Test with None
    assert _all_corners_detected_and_valid(None) == False

    # Test with non-dict
    assert _all_corners_detected_and_valid("not a dict") == False

    # Test with missing required keys
    corners_missing_keys = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90, 10)},
        # Missing bottom_left and bottom_right
    }
    assert _all_corners_detected_and_valid(corners_missing_keys) == False

    # Test with invalid center data
    corners_invalid_center = {
        'top_left': {'center': (10, 10)},
        'top_right': {'wrong_key': (90, 10)},  # Wrong key
        'bottom_left': {'center': (10, 90)},
        'bottom_right': {'center': (90, 90)}
    }
    assert _all_corners_detected_and_valid(corners_invalid_center) == False

    # Test with invalid coordinate format
    corners_invalid_coords = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90,)},  # Only one coordinate
        'bottom_left': {'center': (10, 90)},
        'bottom_right': {'center': (90, 90)}
    }
    assert _all_corners_detected_and_valid(corners_invalid_coords) == False


def test_get_corner_directions():
    """Test corner direction calculation."""
    assert _get_corner_directions('top_left') == [(1, 0), (0, 1)]
    assert _get_corner_directions('top_right') == [(-1, 0), (0, 1)]
    assert _get_corner_directions('bottom_left') == [(1, 0), (0, -1)]
    assert _get_corner_directions('bottom_right') == [(-1, 0), (0, -1)]
    assert _get_corner_directions('invalid') == []


def test_visualize_corners_with_message():
    """Test visualization with custom message."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = {
        'top_left': {'center': (10, 10)},
        'top_right': {'center': (90, 10)},
        'bottom_left': {'center': (10, 90)},
        'bottom_right': {'center': (90, 90)}
    }

    message = "Test Message"
    result = visualize_corners(image, corners, message)

    assert result is not None
    assert result.shape == image.shape
    # We can't easily test if the message was drawn without analyzing pixels,
    # but we can ensure the function completes without error


def test_visualize_corners_edge_cases():
    """Test edge cases for corner visualization."""
    # Test with grayscale image (should be converted to BGR)
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    result = visualize_corners(gray_image, None)
    assert result.shape == (100, 100, 3)  # Should be converted to BGR

    # Test with BGRA image (should be converted to BGR)
    bgra_image = np.zeros((100, 100, 4), dtype=np.uint8)
    result = visualize_corners(bgra_image, None)
    assert result.shape == (100, 100, 3)  # Should be converted to BGR

    # Test with None image (should create placeholder)
    result = visualize_corners(None, None)
    assert result is not None
    assert len(result.shape) == 3  # Should be a valid BGR image
