# app/features/generator/components/header.py
"""
Header component for answer sheets.
"""
from PIL import ImageDraw
from typing import Dict, Optional

from app.features.generator.components.fonts import load_font
from app.core.constants import (
    TITLE_FONT_SIZE, DEFAULT_FONT_SIZE,
    TITLE_COLOR, HEADER_TEXT_COLOR, HEADER_LINE_COLOR,
    PAGE_WIDTH  # Import PAGE_WIDTH
)


class HeaderBuilder:
    """
    Builds header section of answer sheet.
    """

    def __init__(self):
        self.title_font = load_font(TITLE_FONT_SIZE)
        self.normal_font = load_font(DEFAULT_FONT_SIZE)

    def draw(
        self,
        draw: ImageDraw.Draw,
        title: str,
        student_name: str,
        header_fields: Optional[Dict] = None,
        page_width: Optional[int] = None  # Add optional page_width parameter
    ) -> None:
        """
        Draw header with title and fields.
        """
        # Use provided page_width or fall back to constant
        if page_width is None:
            page_width = PAGE_WIDTH

        # Draw title
        self._draw_title(draw, title, page_width)

        # Draw header fields
        if header_fields is None:
            header_fields = self._get_default_fields(student_name)

        self._draw_fields(draw, header_fields)

    def _draw_title(self, draw: ImageDraw.Draw, title: str, page_width: int) -> None:
        """
        Draw centered title.

        Args:
            draw: ImageDraw object
            title: Title text to draw
            page_width: Width of the page
        """
        bbox = draw.textbbox((0, 0), title, font=self.title_font)
        title_width = bbox[2] - bbox[0]

        x = (page_width - title_width) // 2
        draw.text((x, 14), title, font=self.title_font, fill=TITLE_COLOR)

    def _draw_fields(
        self,
        draw: ImageDraw.Draw,
        fields: Dict
    ) -> None:
        """
        Draw header fields with underlines.
        """
        for field_info in fields:
            label = field_info['label']
            value = field_info.get('value', '')
            x = field_info['x']
            y = field_info['y']
            line_width = field_info.get('line_width', 100)

            # Draw label
            draw.text((x, y), label, font=self.normal_font,
                      fill=HEADER_TEXT_COLOR)

            # Calculate label width
            bbox = draw.textbbox((0, 0), label, font=self.normal_font)
            label_width = bbox[2] - bbox[0]

            # Draw underline
            line_y = y + DEFAULT_FONT_SIZE
            draw.line(
                [(x + label_width + 5, line_y),
                 (x + label_width + line_width, line_y)],
                fill=HEADER_LINE_COLOR,
                width=1
            )

            # Draw value if provided
            if value:
                draw.text(
                    (x + label_width + 10, y),
                    value,
                    font=self.normal_font,
                    fill=HEADER_TEXT_COLOR
                )

    def _get_default_fields(self, student_name: str) -> list:
        """
        Get default header fields.
        """
        return [
            {'label': 'Name:', 'value': student_name,
                'x': 80, 'y': 40, 'line_width': 180},
            {'label': 'Year:', 'value': '', 'x': 310, 'y': 40, 'line_width': 100},
            {'label': 'Strand:', 'value': '', 'x': 460, 'y': 40, 'line_width': 100},
            {'label': 'Subject:', 'value': '', 'x': 620, 'y': 40, 'line_width': 100}
        ]
