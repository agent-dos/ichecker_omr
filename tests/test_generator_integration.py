# tests/test_generator_integration.py
"""
Integration tests for sheet generator.
"""
import pytest
import numpy as np

from app.features.generator.service import GeneratorService


class TestGeneratorIntegration:
    """Test generator service integration."""

    def setup_method(self):
        """Setup test dependencies."""
        self.generator = GeneratorService()

    def test_generate_blank_sheet(self):
        """Test blank sheet generation."""
        sheet = self.generator.generate_blank(
            title="Test Sheet",
            student_id="TEST001",
            student_name="Test Student",
            exam_part=1,
            items_per_part=60
        )

        assert sheet is not None
        assert isinstance(sheet, np.ndarray)
        assert len(sheet.shape) == 3  # Should be a color image
        assert sheet.shape[2] == 3    # BGR format

    def test_generate_shaded_sheet(self):
        """Test shaded sheet generation."""
        sheet = self.generator.generate_shaded(
            title="Test Sheet",
            student_id="TEST002",
            student_name="Test Student",
            exam_part=1,
            items_per_part=60,
            num_answers=10
        )

        assert sheet is not None
        assert isinstance(sheet, np.ndarray)
        assert len(sheet.shape) == 3
        assert sheet.shape[2] == 3

    def test_generate_with_custom_parameters(self):
        """Test generation with custom parameters."""
        sheet = self.generator.generate_shaded(
            title="Custom Test",
            student_id="CUSTOM001",
            student_name="Custom Student",
            exam_part=2,
            items_per_part=30,
            num_answers=5,
            shade_intensity=(150, 150),
            offset_range=1,
            size_percent=0.1
        )

        assert sheet is not None
        # Verify the sheet has the expected dimensions
        assert sheet.shape[0] > 0
        assert sheet.shape[1] > 0

    def test_create_answer_key(self):
        """Test answer key creation."""
        answers = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
            5: 'E'
        }

        sheet = self.generator.create_answer_key(
            answers=answers,
            title="Answer Key",
            student_id="KEY001"
        )

        assert sheet is not None
        assert isinstance(sheet, np.ndarray)
