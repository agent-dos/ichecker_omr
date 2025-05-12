# app/features/generator/components/corner_patterns.py
"""
Advanced corner marker patterns for better detection.
"""
from PIL import ImageDraw
import numpy as np
from typing import Tuple, Optional

from app.core.constants import CORNER_MARKER_SIZE, CORNER_MARKER_COLOR


class AdvancedCornerMarkers:
    """
    Creates advanced corner markers with distinctive patterns
    for improved detection accuracy.
    """

    def __init__(self, pattern_type: str = "concentric"):
        """
        Initialize with pattern type.

        Args:
            pattern_type: "concentric", "l_shape", "cross", or "checkerboard"
        """
        self.pattern_type = pattern_type
        self.size = CORNER_MARKER_SIZE
        self.color = CORNER_MARKER_COLOR

    def draw(
        self,
        draw: ImageDraw.Draw,
        page_width: int,
        page_height: int,
        margin: int = 80
    ) -> None:
        """
        Draw corner markers at all four corners.
        """
        positions = [
            (margin, margin, "top_left"),
            (page_width - margin - self.size, margin, "top_right"),
            (margin, page_height - margin - self.size, "bottom_left"),
            (page_width - margin - self.size,
             page_height - margin - self.size, "bottom_right")
        ]

        for x, y, corner_type in positions:
            if self.pattern_type == "concentric":
                self._draw_concentric_squares(draw, x, y)
            elif self.pattern_type == "l_shape":
                self._draw_l_shape(draw, x, y, corner_type)
            elif self.pattern_type == "cross":
                self._draw_cross_pattern(draw, x, y)
            elif self.pattern_type == "checkerboard":
                self._draw_checkerboard(draw, x, y)
            else:
                # Fallback to filled square
                draw.rectangle(
                    (x, y, x + self.size, y + self.size),
                    fill=self.color
                )

    def _draw_concentric_squares(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int
    ) -> None:
        """
        Draw concentric squares pattern.
        """
        # Outer square (filled)
        draw.rectangle(
            (x, y, x + self.size, y + self.size),
            fill=self.color
        )

        # Inner white square
        inner_margin = self.size // 4
        draw.rectangle(
            (x + inner_margin, y + inner_margin,
             x + self.size - inner_margin, y + self.size - inner_margin),
            fill="white"
        )

        # Center black square
        center_margin = self.size // 3
        draw.rectangle(
            (x + center_margin, y + center_margin,
             x + self.size - center_margin, y + self.size - center_margin),
            fill=self.color
        )

    def _draw_l_shape(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        corner_type: str
    ) -> None:
        """
        Draw L-shaped pattern oriented to corner.
        """
        thickness = self.size // 3

        if corner_type == "top_left":
            # Horizontal bar
            draw.rectangle(
                (x, y, x + self.size, y + thickness),
                fill=self.color
            )
            # Vertical bar
            draw.rectangle(
                (x, y, x + thickness, y + self.size),
                fill=self.color
            )
        elif corner_type == "top_right":
            # Horizontal bar
            draw.rectangle(
                (x, y, x + self.size, y + thickness),
                fill=self.color
            )
            # Vertical bar
            draw.rectangle(
                (x + self.size - thickness, y, x + self.size, y + self.size),
                fill=self.color
            )
        elif corner_type == "bottom_left":
            # Horizontal bar
            draw.rectangle(
                (x, y + self.size - thickness, x + self.size, y + self.size),
                fill=self.color
            )
            # Vertical bar
            draw.rectangle(
                (x, y, x + thickness, y + self.size),
                fill=self.color
            )
        else:  # bottom_right
            # Horizontal bar
            draw.rectangle(
                (x, y + self.size - thickness, x + self.size, y + self.size),
                fill=self.color
            )
            # Vertical bar
            draw.rectangle(
                (x + self.size - thickness, y, x + self.size, y + self.size),
                fill=self.color
            )

    def _draw_cross_pattern(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int
    ) -> None:
        """
        Draw cross/plus pattern.
        """
        thickness = self.size // 3
        center = self.size // 2

        # Horizontal bar
        draw.rectangle(
            (x, y + center - thickness // 2,
             x + self.size, y + center + thickness // 2),
            fill=self.color
        )

        # Vertical bar
        draw.rectangle(
            (x + center - thickness // 2, y,
             x + center + thickness // 2, y + self.size),
            fill=self.color
        )

    def _draw_checkerboard(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int
    ) -> None:
        """
        Draw checkerboard pattern.
        """
        cell_size = self.size // 3

        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:
                    draw.rectangle(
                        (x + i * cell_size, y + j * cell_size,
                         x + (i + 1) * cell_size, y + (j + 1) * cell_size),
                        fill=self.color
                    )
