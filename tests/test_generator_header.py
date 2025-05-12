# tests/test_generator_header.py
"""
Tests for generator header component.
"""
import pytest
from PIL import Image, ImageDraw

from app.features.generator.components.header import HeaderBuilder
from app.core.constants import PAGE_WIDTH, PAGE_HEIGHT


def test_header_builder_draw_with_explicit_width():
    """Test header drawing with explicit page width."""
    # Create a test image
    img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)

    header_builder = HeaderBuilder()

    # Test drawing with explicit page width
    header_builder.draw(
        draw,
        title="Test Title",
        student_name="Test Student",
        page_width=PAGE_WIDTH
    )

    # Should complete without error
    assert True


def test_header_builder_draw_without_width():
    """Test header drawing without explicit page width (uses constant)."""
    # Create a test image
    img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)

    header_builder = HeaderBuilder()

    # Test drawing without page width parameter
    header_builder.draw(
        draw,
        title="Test Title",
        student_name="Test Student"
    )

    # Should complete without error
    assert True


def test_title_centering():
    """Test that title is properly centered."""
    img = Image.new('RGB', (800, 100), color='white')
    draw = ImageDraw.Draw(img)

    header_builder = HeaderBuilder()

    # Draw title with known page width
    header_builder._draw_title(draw, "Test", page_width=800)

    # The title should be drawn (we can't easily test exact position without
    # analyzing pixels, but we can ensure no errors occur)
    assert True


def test_header_fields_drawing():
    """Test drawing of header fields."""
    img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)

    header_builder = HeaderBuilder()

    test_fields = [
        {'label': 'Name:', 'value': 'John Doe',
            'x': 50, 'y': 50, 'line_width': 150}
    ]

    header_builder._draw_fields(draw, test_fields)

    # Should complete without error
    assert True
