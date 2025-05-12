# app/features/generator/sheet_builder.py
"""
Builds the basic structure of answer sheets.
"""
from PIL import Image, ImageDraw
from typing import Dict, Optional

from app.features.generator.components.header import HeaderBuilder
from app.features.generator.components.bubbles import BubbleGrid
from app.features.generator.components.corner_patterns import AdvancedCornerMarkers
from app.features.generator.components.qr_code import QRCodeGenerator
from app.core.constants import PAGE_WIDTH, PAGE_HEIGHT, BG_COLOR


class SheetBuilder:
    """
    Builds answer sheet structure.
    """

    def __init__(self, corner_pattern: str = "concentric"):
        """
        Initialize sheet builder with specified corner pattern.

        Args:
            corner_pattern: Type of corner pattern to use
        """
        self.header_builder = HeaderBuilder()
        self.bubble_grid = BubbleGrid()
        self.corner_markers = AdvancedCornerMarkers(corner_pattern)
        self.qr_generator = QRCodeGenerator()

    def build(
        self,
        title: str,
        student_id: str,
        student_name: str,
        exam_part: int,
        items_per_part: int,
        show_corner_markers: bool,
        header_fields: Optional[Dict] = None
    ) -> Image.Image:
        """
        Build complete answer sheet.
        """
        # Create blank image
        img = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color=BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Add components
        self.header_builder.draw(
            draw, title, student_name, header_fields,
            page_width=PAGE_WIDTH
        )

        if show_corner_markers:
            self.corner_markers.draw(draw, PAGE_WIDTH, PAGE_HEIGHT)

        self.qr_generator.draw(img, student_id, PAGE_WIDTH)

        start_number = (exam_part - 1) * items_per_part + 1
        self.bubble_grid.draw(draw, start_number)

        return img
