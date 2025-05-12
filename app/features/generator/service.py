# app/features/generator/service.py
"""
Main generator service for creating answer sheets.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from app.features.generator.sheet_builder import SheetBuilder
from app.features.generator.shaders.bubble_shader import BubbleShader
from app.features.generator.components.qr_code import QRCodeGenerator


class GeneratorService:
    """
    Orchestrates answer sheet generation.
    """

    def __init__(self):
        self.sheet_builder = SheetBuilder()
        self.bubble_shader = BubbleShader()
        self.qr_generator = QRCodeGenerator()

    def generate_blank(
        self,
        title: str = "Answer Sheet",
        student_id: str = "000000",
        student_name: str = "",
        exam_part: int = 1,
        items_per_part: int = 60,
        show_corner_markers: bool = True
    ) -> np.ndarray:
        """
        Generate a blank answer sheet.
        """
        sheet = self.sheet_builder.build(
            title=title,
            student_id=student_id,
            student_name=student_name,
            exam_part=exam_part,
            items_per_part=items_per_part,
            show_corner_markers=show_corner_markers
        )

        # Convert PIL to OpenCV format
        sheet_np = np.array(sheet)
        return cv2.cvtColor(sheet_np, cv2.COLOR_RGB2BGR)

    def generate_shaded(
        self,
        **kwargs
    ) -> np.ndarray:
        """
        Generate answer sheet with shaded bubbles.
        """
        # Extract shading parameters
        num_answers = kwargs.pop('num_answers', 20)
        shade_params = {
            'intensity': kwargs.pop('shade_intensity', (100, 200)),
            'offset_range': kwargs.pop('offset_range', 2),
            'size_percent': kwargs.pop('size_percent', 0.2)
        }
        answers = kwargs.pop('predefined_answers', None)

        # Generate blank sheet
        sheet = self.generate_blank(**kwargs)

        # Apply shading
        shaded = self.bubble_shader.shade(
            sheet,
            num_answers=num_answers,
            answers=answers,
            **shade_params
        )

        return shaded

    def generate_batch(
        self,
        num_sheets: int,
        **kwargs
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Generate multiple sheets.
        """
        sheets = []

        for i in range(num_sheets):
            sheet = self.generate_shaded(**kwargs)
            filename = f"sheet_{i+1:04d}.jpg"
            sheets.append((filename, sheet))

        return sheets

    def create_answer_key(
        self,
        answers: Dict[int, str],
        **kwargs
    ) -> np.ndarray:
        """
        Create an answer key sheet.
        """
        # Convert answers to format expected by shader
        answer_map = {}
        for q_num, answer in answers.items():
            row_key = f"row_{q_num - 1}"
            choice_idx = ord(answer) - ord('A')
            answer_map[row_key] = choice_idx

        # Generate with consistent shading
        return self.generate_shaded(
            predefined_answers=answer_map,
            shade_intensity=(120, 120),
            offset_range=0,
            size_percent=0,
            **kwargs
        )
