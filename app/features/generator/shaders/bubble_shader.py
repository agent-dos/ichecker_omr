# app/features/generator/shaders/bubble_shader.py
"""
Shades bubbles on answer sheets.
"""
import cv2
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import logging

from app.features.analyzer.bubble_detector import BubbleDetector

logger = logging.getLogger(__name__)


class BubbleShader:
    """
    Adds shading to answer sheet bubbles.
    """

    # Standard answer sheet configuration
    CHOICES_PER_QUESTION = 5  # A, B, C, D, E
    CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E']

    def __init__(self):
        # Initialize detector with params for generated sheets
        detector_params = {
            'gaussian_blur_ksize': 5,
            'hough_dp': 1.0,
            'hough_minDist': 20,
            'hough_param1': 50,
            'hough_param2': 18,
            'hough_minRadius': 10,
            'hough_maxRadius': 14,
            'filter_by_corners': False,
            'filter_by_qr': False
        }
        self.detector = BubbleDetector(detector_params)

    def shade(
        self,
        image: np.ndarray,
        num_answers: int = 20,
        answers: Optional[Dict] = None,
        intensity: Tuple[int, int] = (100, 200),
        offset_range: int = 2,
        size_percent: float = 0.2
    ) -> np.ndarray:
        """
        Shade bubbles on answer sheet.

        Args:
            image: Input image (BGR format)
            num_answers: Number of random answers to shade
            answers: Predefined answers as {row_key: choice_index}
            intensity: Range of shading intensity
            offset_range: Random position offset range
            size_percent: Random size variation percentage

        Returns:
            Image with shaded bubbles
        """
        # Detect bubbles
        bubbles, _ = self.detector.detect(
            image,
            qr_polygon=None,
            corners=None,
            visualize_steps=False
        )

        if bubbles is None or bubbles.size == 0:
            logger.warning("No bubbles detected for shading")
            return image

        # Group bubbles into rows (questions)
        rows = self._group_bubbles_by_rows(bubbles, image.shape[1])

        # Select bubbles to shade
        if answers is None:
            answers = self._select_random_answers(rows, num_answers)

        # Apply shading
        result = image.copy()
        self._shade_bubbles(result, rows, answers, intensity,
                            offset_range, size_percent)

        return result

    def _group_bubbles_by_rows(
        self,
        bubbles: np.ndarray,
        image_width: int
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Group bubbles into rows (questions).
        Only keeps rows with exactly 5 bubbles (A-E).
        """
        row_threshold = 10  # Y-coordinate threshold for same row
        rows = {}

        if bubbles is None or bubbles.size == 0:
            return rows

        # Convert to list for sorting
        bubble_list = bubbles.tolist()
        y_sorted = sorted(bubble_list, key=lambda b: b[1])

        current_row = []
        current_y = y_sorted[0][1]
        row_count = 0

        for bubble in y_sorted:
            x, y, r = bubble

            # Check if this bubble belongs to a new row
            if abs(y - current_y) > row_threshold:
                if len(current_row) == self.CHOICES_PER_QUESTION:
                    # Sort by X to ensure A-B-C-D-E order
                    current_row = sorted(current_row, key=lambda b: b[0])
                    rows[f"row_{row_count}"] = current_row
                    row_count += 1
                current_row = []
                current_y = y

            current_row.append((x, y, r))

        # Don't forget the last row
        if len(current_row) == self.CHOICES_PER_QUESTION:
            current_row = sorted(current_row, key=lambda b: b[0])
            rows[f"row_{row_count}"] = current_row

        logger.debug(f"Found {len(rows)} valid rows with 5 bubbles each")
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

        # Limit to available rows
        actual_answers = min(num_answers, len(rows))

        # Randomly select rows
        selected_rows = random.sample(list(rows.keys()), actual_answers)

        # For each selected row, randomly choose a bubble (0-4 for A-E)
        answers = {}
        for row_key in selected_rows:
            answers[row_key] = random.randint(0, self.CHOICES_PER_QUESTION - 1)

        return answers

    def _shade_bubbles(
        self,
        image: np.ndarray,
        rows: Dict[str, List[Tuple[float, float, float]]],
        answers: Dict[str, int],
        intensity: Tuple[int, int],
        offset_range: int,
        size_percent: float
    ) -> None:
        """
        Apply shading to selected bubbles.
        """
        for row_key, choice_idx in answers.items():
            if row_key not in rows:
                continue

            bubbles = rows[row_key]
            if choice_idx >= len(bubbles):
                logger.warning(
                    f"Invalid choice index {choice_idx} for row {row_key}")
                continue

            x, y, r = bubbles[choice_idx]

            # Apply random variations
            dx = random.randint(-offset_range, offset_range)
            dy = random.randint(-offset_range, offset_range)

            size_var = random.uniform(-size_percent, size_percent)
            radius = int(r * (1 + size_var))

            # Random intensity within range
            shade = random.randint(intensity[0], intensity[1])
            color = (shade, shade, shade)

            # Draw filled circle
            cv2.circle(image, (int(x + dx), int(y + dy)), radius, color, -1)
