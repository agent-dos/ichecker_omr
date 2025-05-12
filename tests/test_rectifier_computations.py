# tests/test_rectification_computations.py
"""
Computational tests for rectification system without external images.
"""
import pytest
import numpy as np
import cv2

from app.features.rectification.enhanced_rectifier import EnhancedRectifier
from app.features.corners.detector import CornerDetector


class TestRectifierCalculations:
    """Test mathematical computations in rectifier."""

    def setup_method(self):
        """Setup test instance."""
        self.rectifier = EnhancedRectifier(params={'dst_margin': 10})

    def test_angle_calculation_horizontal(self):
        """Test angle calculation for horizontal edge."""
        corners = {
            'top_left': {'center': (0, 0)},
            'top_right': {'center': (100, 0)},
            'bottom_left': {'center': (0, 100)},
            'bottom_right': {'center': (100, 100)}
        }
        angle = self.rectifier.calculate_angle(corners)
        assert abs(angle) < 0.001  # Should be exactly 0

    def test_angle_calculation_positive_rotation(self):
        """Test angle calculation for positive rotation."""
        corners = {
            'top_left': {'center': (0, 0)},
            'top_right': {'center': (100, 10)},  # 10 pixels higher
            'bottom_left': {'center': (0, 100)},
            'bottom_right': {'center': (100, 110)}
        }
        angle = self.rectifier.calculate_angle(corners)
        # arctan(10/100) = 5.71 degrees
        assert 5.5 < angle < 6.0

    def test_angle_calculation_negative_rotation(self):
        """Test angle calculation for negative rotation."""
        corners = {
            'top_left': {'center': (0, 10)},
            'top_right': {'center': (100, 0)},  # 10 pixels lower
            'bottom_left': {'center': (0, 110)},
            'bottom_right': {'center': (100, 100)}
        }
        angle = self.rectifier.calculate_angle(corners)
        assert -6.0 < angle < -5.5

    def test_angle_calculation_with_vertical_edge(self):
        """Test angle calculation for vertical edge."""
        corners = {
            'top_left': {'center': (0, 0)},
            'top_right': {'center': (0, 100)},  # Vertical
            'bottom_left': {'center': (100, 0)},
            'bottom_right': {'center': (100, 100)}
        }
        angle = self.rectifier.calculate_angle(corners)
        assert abs(angle - 90) < 0.1  # Should be near 90 degrees

    def test_extract_corner_points_valid(self):
        """Test corner point extraction with valid data."""
        corners = {
            'top_left': {'center': (10, 20)},
            'top_right': {'center': (90, 25)},
            'bottom_right': {'center': (85, 95)},
            'bottom_left': {'center': (15, 90)}
        }
        points = self.rectifier._extract_corner_points(corners)

        assert points is not None
        assert points.shape == (4, 2)
        assert points.dtype == np.float32
        np.testing.assert_array_equal(points[0], [10, 20])
        np.testing.assert_array_equal(points[1], [90, 25])

    def test_extract_corner_points_invalid(self):
        """Test corner point extraction with invalid data."""
        # Missing corner
        corners = {
            'top_left': {'center': (10, 20)},
            'top_right': {'center': (90, 25)},
            'bottom_right': {'center': (85, 95)}
            # Missing bottom_left
        }
        points = self.rectifier._extract_corner_points(corners)
        assert points is None

        # Invalid center format
        corners = {
            'top_left': {'center': (10, 20)},
            'top_right': {'center': (90,)},  # Only one coordinate
            'bottom_right': {'center': (85, 95)},
            'bottom_left': {'center': (15, 90)}
        }
        points = self.rectifier._extract_corner_points(corners)
        assert points is None

    def test_calculate_dst_points(self):
        """Test destination points calculation."""
        margin = 10
        src_points = np.array([
            [10, 10], [90, 10],
            [90, 90], [10, 90]
        ], dtype=np.float32)
        
        # Test the actual method that exists
        dst_points, width, height = self.rectifier._calculate_optimal_dst_points(
            src_points, (100, 100, 3)  # Mock image shape
        )
        
        assert dst_points.shape == (4, 2)
        assert dst_points.dtype == np.float32
        
        # Check dimensions are reasonable
        assert width > 0
        assert height > 0


class TestTransformCalculations:
    """Test perspective transform calculations."""

    def test_perspective_transform_matrix(self):
        """Test perspective transform matrix calculation."""
        src_points = np.array([
            [10, 10], [90, 15],   # Top corners
            [85, 95], [15, 90]    # Bottom corners
        ], dtype=np.float32)
        
        dst_points = np.array([
            [0, 0], [100, 0],     # Top corners
            [100, 100], [0, 100]  # Bottom corners
        ], dtype=np.float32)
        
        # Calculate transform
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
        
        assert transform is not None
        assert transform.shape == (3, 3)
        
        # Verify transform works
        test_point = np.array([[[10, 10]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(test_point, transform)
        # Use more lenient tolerance for floating-point comparison
        np.testing.assert_allclose(transformed[0, 0], [0, 0], rtol=1e-4, atol=1e-6)

    def test_transform_corner_coordinates(self):
        """Test corner coordinate transformation."""
        # Create a simple rotation transform
        angle_rad = np.radians(10)
        center = (50, 50)
        M = cv2.getRotationMatrix2D(center, 10, 1.0)

        # Convert to 3x3 perspective matrix
        transform = np.vstack([M, [0, 0, 1]])

        # Test corner transformation
        corners = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]],
                           dtype=np.float32)

        transformed = cv2.perspectiveTransform(corners, transform)

        assert transformed.shape == corners.shape
        # Verify rotation occurred
        assert not np.allclose(transformed, corners)


class TestCornerAnalysis:
    """Test corner detection computations."""

    def test_corner_scoring(self):
        """Test corner scoring algorithm."""
        detector = CornerDetector(params={
            'scoring': {
                'distance_weight': 0.5,
                'area_weight': 0.25,
                'solidity_weight': 0.25,
                'area_norm_factor': 1000.0
            }
        })

        candidate = {
            'center': (50, 50),
            'area': 500,
            'solidity': 0.9
        }

        ideal_pos = (45, 45)  # Close to candidate
        score = detector._calculate_corner_score(candidate, ideal_pos)

        assert score > 0  # Valid score
        assert score < 1  # Normalized score

        # Test with far position
        far_pos = (200, 200)
        far_score = detector._calculate_corner_score(candidate, far_pos)

        assert far_score < score  # Should score worse when farther

    def test_duplicate_removal(self):
        """Test duplicate corner removal algorithm."""
        detector = CornerDetector(params={'duplicate_threshold': 30})

        candidates = [
            {'center': (10, 10), 'area': 500},
            {'center': (15, 15), 'area': 450},  # Duplicate (within threshold)
            {'center': (100, 100), 'area': 500},
            {'center': (90, 90), 'area': 600}   # Duplicate but larger
        ]

        unique = detector._remove_duplicates(candidates, 30)

        assert len(unique) == 2  # Should remove duplicates
        # Should keep the larger area duplicate
        assert any(c['area'] == 600 for c in unique)


class TestGeometricCalculations:
    """Test geometric utility functions."""

    def test_distance_calculation(self):
        """Test Euclidean distance calculation."""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle

    def test_angle_between_vectors(self):
        """Test angle calculation between vectors."""
        # Horizontal vector
        v1 = np.array([1, 0])
        # 45-degree vector
        v2 = np.array([1, 1])

        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / magnitude
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        assert abs(angle_deg - 45.0) < 0.001

    def test_polygon_area(self):
        """Test polygon area calculation."""
        # Square: (0,0), (10,0), (10,10), (0,10)
        square = np.array([[0, 0], [10, 0], [10, 10],
                          [0, 10]], dtype=np.float32)
        area = cv2.contourArea(square)
        assert abs(area - 100.0) < 0.001

    def test_bounding_box_calculation(self):
        """Test bounding box calculation."""
        points = np.array([[5, 5], [15, 8], [12, 20], [3, 15]], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        assert x == 3   # Leftmost point
        assert y == 5   # Topmost point
        assert w == 13  # Width (15-3+1) - bounding box is inclusive
        assert h == 16  # Height (20-5+1) - bounding box is inclusive
