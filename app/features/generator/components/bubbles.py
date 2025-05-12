# app/features/generator/components/bubbles.py
"""
Bubble grid component for answer sheets.
"""
from PIL import ImageDraw
from typing import List

from app.features.generator.components.fonts import load_font
from app.core.constants import (
    DEFAULT_FONT_SIZE, CIRCLE_RADIUS, CIRCLE_THICKNESS,
    BUBBLE_COLOR, LETTER_COLOR, QUESTION_NUMBER_COLOR,
    LEFT_COL_X, RIGHT_COL_X, QUESTION_START_Y,
    QUESTION_SPACING_Y, OPTION_SPACING, OPTION_START_OFFSET_X,
    ITEMS_PER_COLUMN, ITEMS_PER_SHEET
)


class BubbleGrid:
    """
    Creates grid of answer bubbles.
    """

    def __init__(self):
        self.normal_font = load_font(DEFAULT_FONT_SIZE)
        self.choice_labels = ['A', 'B', 'C', 'D', 'E']

    def draw(
        self,
        draw: ImageDraw.Draw,
        start_number: int
    ) -> None:
        """
        Draw complete bubble grid.
        """
        for item_idx in range(ITEMS_PER_SHEET):
            self._draw_question_row(draw, item_idx, start_number)

    def _draw_question_row(
        self,
        draw: ImageDraw.Draw,
        item_idx: int,
        start_number: int
    ) -> None:
        """
        Draw a single question row.
        """
        # Calculate position
        if item_idx < ITEMS_PER_COLUMN:
            x = LEFT_COL_X
            y = QUESTION_START_Y + item_idx * QUESTION_SPACING_Y
            question_num = start_number + item_idx
        else:
            x = RIGHT_COL_X
            rel_idx = item_idx - ITEMS_PER_COLUMN
            y = QUESTION_START_Y + rel_idx * QUESTION_SPACING_Y
            question_num = start_number + item_idx

        # Draw question number
        draw.text(
            (x, y),
            f"{question_num}.",
            font=self.normal_font,
            fill=QUESTION_NUMBER_COLOR
        )

        # Draw bubbles
        for j, letter in enumerate(self.choice_labels):
            center_x = x + OPTION_START_OFFSET_X + j * OPTION_SPACING
            center_y = y + DEFAULT_FONT_SIZE // 2

            self._draw_bubble(draw, center_x, center_y, letter)

    def _draw_bubble(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        letter: str
    ) -> None:
        """
        Draw a single bubble with letter.
        """
        # Draw circle
        draw.ellipse(
            (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
             x + CIRCLE_RADIUS, y + CIRCLE_RADIUS),
            outline=BUBBLE_COLOR,
            fill=None,
            width=CIRCLE_THICKNESS
        )

        # Draw letter centered
        bbox = draw.textbbox((0, 0), letter, font=self.normal_font)
        letter_width = bbox[2] - bbox[0]
        letter_height = bbox[3] - bbox[1]

        text_x = x - letter_width // 2
        text_y = y - letter_height // 2

        draw.text(
            (text_x, text_y),
            letter,
            font=self.normal_font,
            fill=LETTER_COLOR
        )
