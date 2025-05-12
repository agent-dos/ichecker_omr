# tests/test_enhanced_rectification.py
"""
Tests for enhanced rectification system.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from app.features.rectification.pipeline import RectificationPipeline
from app.features.rectification.enhanced_detector import EnhancedCornerDetector
from app.features.rectification.enhanced_rectifier import EnhancedRectifier


class TestEnhancedRectification:
    """Test enhanced rectification components."""

    def setup_method(self):
        """Setup test parameters."""
        self.params = {
            'corner_detection': {
                'min_area': 200,
                'max_area': 8000,
                'duplicate_threshold': 30
            },
            'rectification': {
                'dst_margin': 10
            },
            'analyzer': {
                'rectification_threshold': 3.0
            }
        }

    def test_enhanced_corner_detection(self):
        """Test enhanced corner detection on synthetic image."""
        # Create test image with corner markers
        image = np.ones((600, 400, 3), dtype=np.uint8) * 255

        # Add corner markers
        marker_size = 30
        cv2.rectangle(image, (20, 20), (20+marker_size,
                      20+marker_size), (0, 0, 0), -1)
        cv2.rectangle(image, (350, 20), (350+marker_size,
                      20+marker_size), (0, 0, 0), -1)
        cv2.rectangle(image, (20, 550), (20+marker_size,
                      550+marker_size), (0, 0, 0), -1)
        cv2.rectangle(image, (350, 550), (350+marker_size,
                      550+marker_size), (0, 0, 0), -1)

        detector = EnhancedCornerDetector(self.params['corner_detection'])
        corners, viz = detector.detect(image, visualize_steps=True)

        assert corners is not None
        assert len(corners) == 4
        assert 'top_left' in corners
        assert len(viz) > 0

    def test_angle_calculation(self):
        """Test angle calculation with various orientations."""
        rectifier = EnhancedRectifier(self.params['rectification'])

        # Test horizontal edge
        corners = {
            'top_left': {'center': (100, 100)},
            'top_right': {'center': (300, 100)},
            'bottom_left': {'center': (100, 300)},
            'bottom_right': {'center': (300, 300)}
        }
        angle = rectifier.calculate_angle(corners)
        assert abs(angle) < 1  # Should be close to 0

        # Test rotated edge
        corners_rotated = {
            'top_left': {'center': (100, 100)},
            'top_right': {'center': (300, 120)},  # 20 pixels higher
            'bottom_left': {'center': (100, 300)},
            'bottom_right': {'center': (300, 320)}
        }
        angle_rotated = rectifier.calculate_angle(corners_rotated)
        assert angle_rotated > 0  # Should detect positive rotation

    def test_rectification_pipeline(self, tmp_path):
        """Test complete rectification pipeline."""
        # Create rotated test image
        image = np.ones((600, 400, 3), dtype=np.uint8) * 255

        # Add content
        cv2.putText(image, "TEST", (150, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Add corner markers at rotated positions
        angle_rad = np.radians(10)  # 10 degree rotation
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        center_x, center_y = 200, 300
        corners = [
            (-180, -280), (180, -280), (180, 280), (-180, 280)
        ]

        for i, (dx, dy) in enumerate(corners):
            x = int(center_x + dx * cos_a - dy * sin_a)
            y = int(center_y + dx * sin_a + dy * cos_a)
            cv2.rectangle(image, (x-15, y-15), (x+15, y+15), (0, 0, 0), -1)

        # Process through pipeline
        pipeline = RectificationPipeline(self.params)
        rectified, results = pipeline.process(image, visualize=True)

        assert rectified is not None
        assert results['corners'] is not None
        assert abs(results['angle']) > 5  # Should detect rotation
        assert results['method_used'] in ['enhanced', 'basic']

        # Save for manual inspection
        cv2.imwrite(str(tmp_path / "original.jpg"), image)
        cv2.imwrite(str(tmp_path / "rectified.jpg"), rectified)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        pipeline = RectificationPipeline(self.params)

        # Empty image
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rectified, results = pipeline.process(empty_image)
        assert rectified is None
        assert results['corners'] is None

        # Image with no corners
        no_corners = np.ones((400, 400, 3), dtype=np.uint8) * 128
        rectified, results = pipeline.process(no_corners)
        assert rectified is None

        # Image with partial corners
        partial = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(partial, (10, 10), (40, 40), (0, 0, 0), -1)
        cv2.rectangle(partial, (360, 10), (390, 40), (0, 0, 0), -1)
        rectified, results = pipeline.process(partial)
        # Should fail gracefully
        assert results['corners'] is None or len(results['corners']) < 4


class TestRectificationIntegration:
    """Integration tests for rectification system."""

    def test_service_integration(self):
        """Test integration with AnalyzerService."""
        from app.features.analyzer.service import AnalyzerService

        # Create test image
        image = np.ones((1200, 850, 3), dtype=np.uint8) * 255

        # Test with analyzer service
        params = {
            'analyzer': {
                'enable_rectification': True,
                'rectification_threshold': 3.0
            }
        }

        service = AnalyzerService(params)
        results = service.analyze(image)

        assert 'steps' in results
        rect_step = results['steps'][0]  # Rectification is first step
        assert 'Rectification' in rect_step['name']
