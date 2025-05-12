# app/features/generator/components/corners.py
"""
Corner markers component for answer sheets.
"""
from PIL import ImageDraw

from app.core.constants import (
    CORNER_MARGIN, CORNER_MARKER_SIZE, CORNER_MARKER_COLOR
)


class CornerMarkers:
    """
    Draws corner markers for sheet alignment.
    """

    def draw(
        self,
        draw: ImageDraw.Draw,
        page_width: int,
        page_height: int
    ) -> None:
        """
        Draw corner markers at all four corners.
        """
        positions = [
            (CORNER_MARGIN, CORNER_MARGIN),  # Top-left
            (page_width - CORNER_MARGIN -
             CORNER_MARKER_SIZE, CORNER_MARGIN),  # Top-right
            (CORNER_MARGIN, page_height - CORNER_MARGIN -
             CORNER_MARKER_SIZE),  # Bottom-left
            (page_width - CORNER_MARGIN - CORNER_MARKER_SIZE,  # Bottom-right
             page_height - CORNER_MARGIN - CORNER_MARKER_SIZE)
        ]

        for x, y in positions:
            draw.rectangle(
                (x, y, x + CORNER_MARKER_SIZE, y + CORNER_MARKER_SIZE),
                fill=CORNER_MARKER_COLOR
            )
