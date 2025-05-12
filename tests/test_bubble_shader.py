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

    def test_group_bubbles_standard_5_per_row(self):
        """Test grouping with standard 5 bubbles per row."""
        # Create exactly 5 bubbles per row
        bubbles = np.array([
            # Row 1 (A-E)
            [30, 50, 10],
            [60, 51, 10],
            [90, 49, 10],
            [120, 50, 10],
            [150, 52, 10],
            # Row 2 (A-E)
            [30, 100, 10],
            [60, 101, 10],
            [90, 99, 10],
            [120, 100, 10],
            [150, 98, 10],
        ])

        rows = self.shader._group_bubbles_by_rows(bubbles, 200)

        assert len(rows) == 2
        assert all(len(row) == 5 for row in rows.values())

        # Verify ordering (should be sorted by X)
        for row_bubbles in rows.values():
            x_coords = [b[0] for b in row_bubbles]
            assert x_coords == sorted(x_coords)

    def test_group_bubbles_filters_incorrect_counts(self):
        """Test that rows with != 5 bubbles are filtered out."""
        bubbles = np.array([
            # Row 1: Only 3 bubbles (should be filtered)
            [30, 50, 10],
            [60, 51, 10],
            [90, 49, 10],
            # Row 2: 5 bubbles (should be kept)
            [30, 100, 10],
            [60, 101, 10],
            [90, 99, 10],
            [120, 100, 10],
            [150, 98, 10],
            # Row 3: 6 bubbles (should be filtered)
            [30, 150, 10],
            [60, 151, 10],
            [90, 149, 10],
            [120, 150, 10],
            [150, 148, 10],
            [180, 150, 10],
        ])

        rows = self.shader._group_bubbles_by_rows(bubbles, 200)

        assert len(rows) == 1  # Only row 2 should remain
        assert list(rows.keys()) == ['row_0']
        assert len(rows['row_0']) == 5

    def test_select_random_answers(self):
        """Test random answer selection."""
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
        """Test handling of empty bubble detection."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # This should handle empty detection gracefully
        result = self.shader.shade(image, num_answers=5)

        assert result is not None
        assert np.array_equal(result, image)  # Should return unchanged
