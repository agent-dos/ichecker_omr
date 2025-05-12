# app/features/generator/shaders/bubble_shader.py
"""
Shades bubbles on answer sheets.
"""
import cv2
import numpy as np
import random
from typing import Dict, List, Optional, Tuple

from app.features.analyzer.bubble_detector import BubbleDetector


class BubbleShader:
    """
    Adds shading to answer sheet bubbles.
    """

    def __init__(self):
        self.detector = BubbleDetector({'min_radius': 10, 'max_radius': 14})

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
        """
        # Detect bubbles
        bubbles = self.detector.detect(image)

        if bubbles is None:
            return image

        # Group bubbles by rows
        rows = self._group_bubbles_by_rows(bubbles, image.shape[1])

        # Select bubbles to shade
        if answers is None:
            answers = self._select_random_answers(rows, num_answers)

        # Shade selected bubbles
        result = image.copy()
        self._shade_bubbles(result, rows, answers, intensity,
                            offset_range, size_percent)

        return result

    def _group_bubbles_by_rows(
        self,
        bubbles: np.ndarray,
        image_width: int
    ) -> Dict[str, List[np.ndarray]]:
        """
        Group bubbles into rows.
        """
        midpoint_x = image_width // 2
        row_threshold = 10
        rows = {}

        # Sort by Y coordinate
        y_sorted = sorted(bubbles, key=lambda b: b[1])

        current_row = []
        current_y = y_sorted[0][1]
        row_count = 0

        for bubble in y_sorted:
            x, y, r = bubble

            if abs(y - current_y) > row_threshold:
                if current_row:
                    row_key = f"row_{row_count}"
                    rows[row_key] = sorted(current_row, key=lambda b: b[0])
                    row_count += 1
                current_row = []
                current_y = y

            current_row.append(bubble)

        # Add last row
        if current_row:
            row_key = f"row_{row_count}"
            rows[row_key] = sorted(current_row, key=lambda b: b[0])

        # Filter valid rows (5+ bubbles)
        valid_rows = {k: v for k, v in rows.items() if len(v) >= 5}

        return valid_rows

    def _select_random_answers(
        self,
        rows: Dict[str, List[np.ndarray]],
        num_answers: int
    ) -> Dict[str, int]:
        """
        Select random bubbles to shade.
        """
        answers = {}
        row_keys = list(rows.keys())

        if not row_keys:
            return answers

        # Select random rows
        selected_rows = random.sample(
            row_keys,
            min(num_answers, len(row_keys))
        )

        # Select random bubble in each row
        for row_key in selected_rows:
            max_choice = min(5, len(rows[row_key]))
            choice_idx = random.randint(0, max_choice - 1)
            answers[row_key] = choice_idx

        return answers

    def _shade_bubbles(
        self,
        image: np.ndarray,
        rows: Dict[str, List[np.ndarray]],
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

            if choice_idx >= len(rows[row_key]):
                continue

            bubble = rows[row_key][choice_idx]
            x, y, r = bubble

            # Apply random variations
            dx = random.randint(-offset_range, offset_range)
            dy = random.randint(-offset_range, offset_range)

            size_var = random.uniform(-size_percent, size_percent)
            radius = int(r * (1 + size_var))

            shade = random.randint(intensity[0], intensity[1])
            color = (shade, shade, shade)

            # Draw filled circle
            cv2.circle(image, (x + dx, y + dy), radius, color, -1)
