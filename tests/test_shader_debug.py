# tests/test_shader_debug.py
"""
Debug tests for bubble shader to identify issues.
"""
import pytest
import numpy as np
import cv2
import logging
from pathlib import Path

from app.features.generator.service import GeneratorService
from app.features.generator.shaders.bubble_shader import BubbleShader

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


class TestShaderDebug:
    """Debug tests for bubble shading issues."""

    def test_bubble_position_calculation(self):
        """Test that bubble positions are calculated correctly."""
        shader = BubbleShader()
        positions = shader._calculate_bubble_positions()

        # Should have 60 questions * 5 choices = 300 bubbles
        assert len(
            positions) == 300, f"Expected 300 positions, got {len(positions)}"

        # Check first row positions
        first_row = positions[:5]
        for i, (x, y, r) in enumerate(first_row):
            print(f"Bubble {i}: x={x}, y={y}, r={r}")
            assert r == 13, "Radius should be 13"
            assert x > 0 and y > 0, "Coordinates should be positive"

    def test_direct_coordinate_shading(self, tmp_path):
        """Test shading using direct coordinates."""
        generator = GeneratorService()
        sheet = generator.generate_blank()

        # Create shader and manually shade specific coordinates
        shader = BubbleShader()

        # Define specific answers for testing
        test_answers = {
            'row_0': 0,  # First bubble in first row
            'row_1': 2,  # Third bubble in second row
            'row_5': 4,  # Last bubble in sixth row
        }

        # Apply shading with high contrast
        shaded = shader.shade(
            sheet,
            answers=test_answers,
            intensity=(30, 60),  # Very dark
            offset_range=0,      # No offset
            size_percent=0       # No size variation
        )

        # Save for inspection
        output_path = tmp_path / "debug_shaded.jpg"
        cv2.imwrite(str(output_path), shaded)

        # Verify changes
        assert not np.array_equal(sheet, shaded), "No changes detected"

        # Calculate difference
        diff = cv2.absdiff(sheet, shaded)
        changed_pixels = np.count_nonzero(diff > 0)

        print(f"Changed pixels: {changed_pixels}")
        print(f"Output saved to: {output_path}")

        assert changed_pixels > 0, "No pixels were changed"

    def test_verify_bubble_detection(self, tmp_path):
        """Verify that bubbles exist where we expect them."""
        generator = GeneratorService()
        sheet = generator.generate_blank()

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)

        # Check a known bubble position
        from app.core.constants import (
            LEFT_COL_X, QUESTION_START_Y, OPTION_START_OFFSET_X,
            DEFAULT_FONT_SIZE, CIRCLE_RADIUS
        )

        # Calculate first bubble position
        x = LEFT_COL_X + OPTION_START_OFFSET_X
        y = QUESTION_START_Y + DEFAULT_FONT_SIZE // 2

        # Create a mask for this bubble
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(x), int(y)), CIRCLE_RADIUS, 255, -1)

        # Extract bubble area
        bubble_area = cv2.bitwise_and(gray, mask)

        # Count non-white pixels (should be the circle outline)
        non_white = np.count_nonzero(bubble_area < 250)

        print(f"First bubble at ({x}, {y})")
        print(f"Non-white pixels in bubble area: {non_white}")

        # Save debug images
        cv2.imwrite(str(tmp_path / "bubble_mask.jpg"), mask)
        cv2.imwrite(str(tmp_path / "bubble_area.jpg"), bubble_area)

        assert non_white > 0, "No bubble found at expected position"
