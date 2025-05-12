# tests/test_l_shape_detection.py
"""
Tests for L-shape corner pattern detection.
"""
import pytest
import numpy as np
import cv2

from app.features.corners.strategies.pattern_based import PatternBasedStrategy


class TestLShapeDetection:
    """Test L-shape pattern detection functionality."""

    def setup_method(self):
        """Setup test parameters."""
        self.params = {
            'enabled': True,
            'pattern_type': 'l_shape',
            'debug': False
        }
        self.strategy = PatternBasedStrategy(self.params)

    def test_create_l_shape_pattern(self):
        """Test creation of L-shape test pattern."""
        # Create test image with L-shape
        img = np.ones((100, 100), dtype=np.uint8) * 255

        # Draw L-shape (top-left corner style)
        thickness = 10
        cv2.rectangle(img, (10, 10), (40, 10 + thickness), 0, -1)
        cv2.rectangle(img, (10, 10), (10 + thickness, 40), 0, -1)

        candidates, _ = self.strategy.detect(
            img, min_area=100, max_area=2000,
            qr_polygon=None, visualize_steps=False
        )

        assert len(candidates) > 0
        assert candidates[0]['corner_type'] == 'top_left'

    def test_l_shape_orientation(self):
        """Test L-shape orientation verification."""
        # Create L-shape contour
        points = np.array([
            [10, 10], [40, 10], [40, 20],
            [20, 20], [20, 40], [10, 40]
        ], dtype=np.int32)

        contour = points.reshape((-1, 1, 2))

        # Test orientation verification
        assert self.strategy._verify_l_orientation(contour, 'top_left')
        assert not self.strategy._verify_l_orientation(contour, 'bottom_right')

    def test_corner_region_determination(self):
        """Test corner region determination."""
        corner_regions = {
            'top_left': (0, 0, 100, 100),
            'top_right': (200, 0, 300, 100),
            'bottom_left': (0, 200, 100, 300),
            'bottom_right': (200, 200, 300, 300)
        }

        assert self.strategy._determine_corner_type(
            50, 50, corner_regions) == 'top_left'
        assert self.strategy._determine_corner_type(
            250, 50, corner_regions) == 'top_right'
        assert self.strategy._determine_corner_type(
            50, 250, corner_regions) == 'bottom_left'
        assert self.strategy._determine_corner_type(
            250, 250, corner_regions) == 'bottom_right'
        assert self.strategy._determine_corner_type(
            150, 150, corner_regions) is None

    def test_detect_multiple_l_shapes(self):
        """Test detection of multiple L-shapes in one image."""
        # Create image with L-shapes in all corners
        img = np.ones((300, 300), dtype=np.uint8) * 255

        # Top-left L
        cv2.rectangle(img, (10, 10), (40, 20), 0, -1)
        cv2.rectangle(img, (10, 10), (20, 40), 0, -1)

        # Top-right L
        cv2.rectangle(img, (260, 10), (290, 20), 0, -1)
        cv2.rectangle(img, (280, 10), (290, 40), 0, -1)

        # Bottom-left L
        cv2.rectangle(img, (10, 280), (40, 290), 0, -1)
        cv2.rectangle(img, (10, 260), (20, 290), 0, -1)

        # Bottom-right L
        cv2.rectangle(img, (260, 280), (290, 290), 0, -1)
        cv2.rectangle(img, (280, 260), (290, 290), 0, -1)

        candidates, viz_steps = self.strategy.detect(
            img, min_area=100, max_area=2000,
            qr_polygon=None, visualize_steps=True
        )

        assert len(candidates) == 4
        corner_types = {c['corner_type'] for c in candidates}
        assert corner_types == {'top_left',
                                'top_right', 'bottom_left', 'bottom_right'}
