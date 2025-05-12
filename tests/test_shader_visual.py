# tests/test_shader_visual.py
"""
Visual tests for bubble shader to verify shading is visible.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from app.features.generator.service import GeneratorService
from app.features.generator.shaders.bubble_shader import BubbleShader


class TestShaderVisual:
    """Visual tests for bubble shading."""

    def test_generate_and_shade(self, tmp_path):
        """Generate a sheet and verify shading is visible."""
        generator = GeneratorService()

        # Generate a blank sheet
        blank_sheet = generator.generate_blank(
            title="Test Sheet",
            student_id="TEST001",
            student_name="Test Student"
        )

        # Generate a shaded sheet
        shaded_sheet = generator.generate_shaded(
            title="Test Sheet",
            student_id="TEST001",
            student_name="Test Student",
            num_answers=10,
            shade_intensity=(50, 150),  # Darker shading
            offset_range=1,
            size_percent=0.1
        )

        # Save images for manual inspection
        blank_path = tmp_path / "blank_sheet.jpg"
        shaded_path = tmp_path / "shaded_sheet.jpg"

        cv2.imwrite(str(blank_path), blank_sheet)
        cv2.imwrite(str(shaded_path), shaded_sheet)

        # Compare pixel values to verify shading occurred
        blank_gray = cv2.cvtColor(blank_sheet, cv2.COLOR_BGR2GRAY)
        shaded_gray = cv2.cvtColor(shaded_sheet, cv2.COLOR_BGR2GRAY)

        # Calculate difference
        diff = cv2.absdiff(blank_gray, shaded_gray)
        non_zero_count = np.count_nonzero(diff > 10)

        # Verify significant differences exist (shading present)
        assert non_zero_count > 100, "No significant shading detected"

        print(f"Test images saved to: {tmp_path}")
        print(f"Pixels with significant difference: {non_zero_count}")

    def test_direct_shader(self, tmp_path):
        """Test the shader directly on a generated sheet."""
        generator = GeneratorService()
        shader = BubbleShader()

        # Generate blank sheet
        sheet = generator.generate_blank()

        # Apply shading directly
        predefined_answers = {
            f'row_{i}': i % 5 for i in range(10)  # A, B, C, D, E pattern
        }

        shaded = shader.shade(
            sheet,
            answers=predefined_answers,
            intensity=(40, 120),  # Very dark for testing
            offset_range=0,
            size_percent=0
        )

        # Save for inspection
        output_path = tmp_path / "direct_shaded.jpg"
        cv2.imwrite(str(output_path), shaded)

        # Verify shading
        assert shaded is not None
        assert shaded.shape == sheet.shape
        assert not np.array_equal(sheet, shaded), "No changes detected"
