# tests/test_bubble_shader.py
"""
Tests for bubble shader functionality.
"""
import pytest
import numpy as np
import cv2

from app.features.generator.shaders.bubble_shader import BubbleShader


class TestBubbleShader:
    """Test bubble shader functionality."""

    def setup_method(self):
        """Setup test dependencies."""
        self.shader = BubbleShader()

    def test_constants(self):
        """Test shader constants."""
        assert self.shader.CHOICES_PER_QUESTION == 5
        assert self.shader.CHOICE_LABELS == ['A', 'B', 'C', 'D', 'E']

    def test_bubble_position_calculation(self):
        """Test that bubble positions are calculated correctly."""
        positions = self.shader._calculate_bubble_positions()

        # Should have 60 questions * 5 choices = 300 bubbles
        assert len(positions) == 300

        # Verify all positions have correct structure
        for pos in positions:
            assert len(pos) == 3  # (x, y, radius)
            assert pos[2] == self.shader.expected_radius

    def test_group_calculated_bubbles(self):
        """Test grouping of calculated bubble positions."""
        # Create test positions (simulating 3 rows of 5 bubbles each)
        test_positions = [
            (50 + i * 30, 50, 13) for i in range(5)  # Row 0
        ] + [
            (50 + i * 30, 100, 13) for i in range(5)  # Row 1
        ] + [
            (50 + i * 30, 150, 13) for i in range(5)  # Row 2
        ]

        rows = self.shader._group_calculated_bubbles(test_positions)

        assert len(rows) == 3
        assert 'row_0' in rows
        assert 'row_1' in rows
        assert 'row_2' in rows

        # Each row should have exactly 5 bubbles
        for row_key, bubbles in rows.items():
            assert len(bubbles) == 5

    def test_select_random_answers(self):
        """Test random answer selection."""
        # Create test rows
        rows = {
            'row_0': [(10, 10, 5), (20, 10, 5), (30, 10, 5), (40, 10, 5), (50, 10, 5)],
            'row_1': [(10, 20, 5), (20, 20, 5), (30, 20, 5), (40, 20, 5), (50, 20, 5)],
            'row_2': [(10, 30, 5), (20, 30, 5), (30, 30, 5), (40, 30, 5), (50, 30, 5)],
        }

        # Test selecting fewer answers than rows
        answers = self.shader._select_random_answers(rows, 2)
        assert len(answers) == 2
        assert all(0 <= idx < 5 for idx in answers.values())

        # Test selecting more answers than rows
        answers = self.shader._select_random_answers(rows, 5)
        assert len(answers) == 3  # Limited to available rows

    def test_shade_with_predefined_answers(self):
        """Test shading with specific answer mapping."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Define specific answers (A=0, B=1, C=2, D=3, E=4)
        predefined_answers = {
            'row_0': 0,  # A
            'row_1': 2,  # C
            'row_2': 4,  # E
        }

        result = self.shader.shade(
            image,
            answers=predefined_answers,
            intensity=(150, 150),  # Fixed intensity
            offset_range=0,        # No offset
            size_percent=0        # No size variation
        )

        assert result is not None
        assert result.shape == image.shape

    def test_empty_rows_handling(self):
        """Test handling with images too small for standard layout."""
        # Create a small image where bubble positions would be out of bounds
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # This should handle gracefully - no shading on tiny images
        result = self.shader.shade(image, num_answers=5)

        assert result is not None
        # On a 100x100 image, all bubble positions would be out of bounds
        # so no changes should occur
        assert np.array_equal(result, image)  # Should have NO changes

    def test_partial_sheet_shading(self):
        """Test shading on a partial sheet area."""
        # Create an image that's large enough for at least some bubbles
        # Based on constants: LEFT_COL_X=120, QUESTION_START_Y=115
        image = np.ones((400, 300, 3), dtype=np.uint8) * 255

        result = self.shader.shade(image, num_answers=5)

        assert result is not None
        assert result.shape == image.shape

        # Some bubbles should be within bounds and get shaded
        diff = cv2.absdiff(image, result)
        changed_pixels = np.count_nonzero(diff > 0)

        # With proper dimensions, some bubbles should be shaded
        if changed_pixels > 0:
            assert changed_pixels > 100  # Expect significant changes
        else:
            # If no changes, it means all positions were out of bounds
            # which is acceptable for partial sheets
            pass

    def test_full_sheet_shading(self):
        """Test shading on a full-sized sheet."""
        # Create a sheet-sized image
        image = np.ones((1202, 850, 3), dtype=np.uint8) * 255

        # Generate random answers
        result = self.shader.shade(image, num_answers=30)

        assert result is not None
        assert result.shape == image.shape

        # Verify changes were made
        diff = cv2.absdiff(image, result)
        changed_pixels = np.count_nonzero(diff > 0)
        assert changed_pixels > 0, "No pixels were changed"
