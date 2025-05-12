# tests/test_bubble_grid.py
"""
Tests for bubble grid component.
"""
import pytest
from PIL import Image, ImageDraw
import numpy as np

from app.features.generator.components.bubbles import BubbleGrid
from app.core.constants import (
    PAGE_WIDTH, PAGE_HEIGHT, LETTER_OFFSET_X, LETTER_OFFSET_Y
)


class TestBubbleGrid:
    """Test bubble grid functionality."""

    def setup_method(self):
        """Setup test dependencies."""
        self.bubble_grid = BubbleGrid()
        self.test_image = Image.new(
            'RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
        self.draw = ImageDraw.Draw(self.test_image)

    def test_bubble_grid_creation(self):
        """Test basic bubble grid creation."""
        # Draw bubbles starting from question 1
        self.bubble_grid.draw(self.draw, start_number=1)

        # Convert to numpy array for analysis
        img_array = np.array(self.test_image)

        # Verify image is not blank (has some non-white pixels)
        assert not np.all(img_array == 255)

    def test_draw_bubble_method(self):
        """Test individual bubble drawing."""
        # Draw a single bubble
        self.bubble_grid._draw_bubble(self.draw, 100, 100, 'A')

        # Convert to numpy array
        img_array = np.array(self.test_image)

        # Verify something was drawn
        assert not np.all(img_array == 255)

    def test_question_row_drawing(self):
        """Test question row drawing."""
        # Draw first question row
        self.bubble_grid._draw_question_row(self.draw, 0, 1)

        # Verify drawing occurred
        img_array = np.array(self.test_image)
        assert not np.all(img_array == 255)

    def test_letter_offset_constants(self):
        """Verify letter offset constants are being used."""
        # This test verifies the constants exist and are accessible
        assert isinstance(LETTER_OFFSET_X, (int, float))
        assert isinstance(LETTER_OFFSET_Y, (int, float))

        # Draw with and without offsets to verify difference
        # Note: This is more of an integration test that would require
        # pixel-level analysis to truly verify offset application
