# app/features/generator/shaders/bubble_shader.py
"""
Enhanced bubble shader with reliable detection for generated sheets.
"""
from app.core.constants import DEFAULT_FONT_SIZE
import cv2
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import logging

from app.core.constants import (
    CIRCLE_RADIUS, LEFT_COL_X, RIGHT_COL_X,
    QUESTION_START_Y, QUESTION_SPACING_Y, OPTION_SPACING,
    OPTION_START_OFFSET_X, ITEMS_PER_COLUMN
)

logger = logging.getLogger(__name__)


class BubbleShader:
    """
    Adds shading to answer sheet bubbles with precise coordinate calculation.
    """

    CHOICES_PER_QUESTION = 5  # A, B, C, D, E
    CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']

    def __init__(self):
        """Initialize the bubble shader."""
        self.expected_radius = CIRCLE_RADIUS

    def shade(
        self,
        image: np.ndarray,
        num_answers: int = 20,
        answers: Optional[Dict] = None,
        intensity: Tuple[int, int] = (80, 180),
        offset_range: int = 1,
        size_percent: float = 0.15
    ) -> np.ndarray:
        """
        Shade bubbles on answer sheet using known coordinates.

        Since we're working with generated sheets, we can calculate exact bubble
        positions rather than detecting them.
        """
        logger.info(f"Starting bubble shading with {num_answers} answers")

        # Calculate bubble positions based on known layout
        bubble_positions = self._calculate_bubble_positions()
        logger.info(f"Calculated {len(bubble_positions)} bubble positions")

        # Verify bubbles exist at calculated positions (optional debug step)
        if logger.isEnabledFor(logging.DEBUG):
            self._verify_bubble_positions(image, bubble_positions)

        # Group bubbles into rows
        rows = self._group_calculated_bubbles(bubble_positions)
        logger.info(f"Grouped into {len(rows)} rows")

        # Select bubbles to shade
        if answers is None:
            answers = self._select_random_answers(rows, num_answers)

        logger.info(f"Shading {len(answers)} answers")

        # Apply shading
        result = image.copy()
        self._shade_bubbles_precise(result, rows, answers, intensity,
                                    offset_range, size_percent)

        return result

    def _calculate_bubble_positions(self) -> List[Tuple[float, float, float]]:
        """
        Calculate exact bubble positions based on sheet layout constants.
        """
        positions = []

        # Process left column (questions 1-30)
        for i in range(ITEMS_PER_COLUMN):
            base_x = LEFT_COL_X + OPTION_START_OFFSET_X
            base_y = QUESTION_START_Y + i * QUESTION_SPACING_Y + DEFAULT_FONT_SIZE // 2

            for j in range(self.CHOICES_PER_QUESTION):
                x = base_x + j * OPTION_SPACING
                positions.append((x, base_y, self.expected_radius))

        # Process right column (questions 31-60)
        for i in range(ITEMS_PER_COLUMN):
            base_x = RIGHT_COL_X + OPTION_START_OFFSET_X
            base_y = QUESTION_START_Y + i * QUESTION_SPACING_Y + DEFAULT_FONT_SIZE // 2

            for j in range(self.CHOICES_PER_QUESTION):
                x = base_x + j * OPTION_SPACING
                positions.append((x, base_y, self.expected_radius))

        return positions

    def _verify_bubble_positions(self, image: np.ndarray, positions: List[Tuple[float, float, float]]) -> None:
        """
        Debug method to verify bubbles exist at calculated positions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image

        # Check first 5 positions
        for i, (x, y, r) in enumerate(positions[:5]):
            # Create a mask for the bubble area
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

            # Check if there's a circle outline in this area
            bubble_area = cv2.bitwise_and(gray, mask)
            non_white_pixels = np.count_nonzero(bubble_area < 250)

            logger.debug(
                f"Position {i}: ({x:.1f}, {y:.1f}) has {non_white_pixels} non-white pixels")

    def _group_calculated_bubbles(self, positions: List[Tuple[float, float, float]]) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Group calculated bubble positions into rows.
        """
        rows = {}
        bubbles_per_row = self.CHOICES_PER_QUESTION
        total_rows = len(positions) // bubbles_per_row

        for row_num in range(total_rows):
            start_idx = row_num * bubbles_per_row
            end_idx = start_idx + bubbles_per_row
            row_bubbles = positions[start_idx:end_idx]
            rows[f"row_{row_num}"] = row_bubbles

        return rows

    def _select_random_answers(
        self,
        rows: Dict[str, List[Tuple[float, float, float]]],
        num_answers: int
    ) -> Dict[str, int]:
        """
        Select random answers for specified number of questions.
        """
        if not rows:
            return {}

        actual_answers = min(num_answers, len(rows))
        selected_rows = random.sample(list(rows.keys()), actual_answers)

        answers = {}
        for row_key in selected_rows:
            answers[row_key] = random.randint(0, self.CHOICES_PER_QUESTION - 1)

        return answers

    def _shade_bubbles_precise(
        self,
        image: np.ndarray,
        rows: Dict[str, List[Tuple[float, float, float]]],
        answers: Dict[str, int],
        intensity: Tuple[int, int],
        offset_range: int,
        size_percent: float
    ) -> None:
        """
        Apply shading to selected bubbles with precise positioning.
        """
        for row_key, choice_idx in answers.items():
            if row_key not in rows:
                logger.warning(f"Row {row_key} not found")
                continue

            bubbles = rows[row_key]
            if choice_idx >= len(bubbles):
                logger.warning(
                    f"Invalid choice index {choice_idx} for row {row_key}")
                continue

            x, y, r = bubbles[choice_idx]

            # Apply slight random variations
            dx = random.randint(-offset_range,
                                offset_range) if offset_range > 0 else 0
            dy = random.randint(-offset_range,
                                offset_range) if offset_range > 0 else 0

            # Vary the size slightly
            size_variation = random.uniform(-size_percent,
                                            size_percent) if size_percent > 0 else 0
            # Fill most of the bubble
            fill_radius = int(r * (0.8 - size_variation))

            # Random intensity within range
            shade = random.randint(intensity[0], intensity[1])
            color = (shade, shade, shade)

            # Draw filled circle
            center = (int(x + dx), int(y + dy))
            cv2.circle(image, center, fill_radius, color, -1, cv2.LINE_AA)

            logger.debug(
                f"Shaded bubble at {center} with radius {fill_radius} and color {color}")

            # Add gradient effect for realism
            if fill_radius > 5:
                for i in range(1, 4):
                    gradient_radius = fill_radius - i * 2
                    if gradient_radius > 0:
                        gradient_shade = min(255, shade + i * 15)
                        gradient_color = (
                            gradient_shade, gradient_shade, gradient_shade)
                        cv2.circle(image, center, gradient_radius,
                                   gradient_color, 1, cv2.LINE_AA)

    def _group_bubbles_by_rows(self, bubbles: np.ndarray, image_width: int) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Legacy method for backward compatibility with tests.
        Groups detected bubbles into rows (for testing purposes).
        """
        # Convert numpy array to list of tuples
        if isinstance(bubbles, np.ndarray):
            bubble_list = bubbles.tolist()
        else:
            bubble_list = list(bubbles)

        # Sort by Y coordinate
        y_sorted = sorted(bubble_list, key=lambda b: b[1])

        rows = {}
        row_threshold = 15
        current_row = []
        current_y = y_sorted[0][1] if y_sorted else 0
        row_count = 0

        for bubble in y_sorted:
            x, y, r = bubble

            if abs(y - current_y) > row_threshold:
                if len(current_row) == self.CHOICES_PER_QUESTION:
                    current_row = sorted(current_row, key=lambda b: b[0])
                    rows[f"row_{row_count}"] = current_row
                    row_count += 1
                current_row = []
                current_y = y

            current_row.append((x, y, r))

        # Add the last row
        if len(current_row) == self.CHOICES_PER_QUESTION:
            current_row = sorted(current_row, key=lambda b: b[0])
            rows[f"row_{row_count}"] = current_row

        return rows
