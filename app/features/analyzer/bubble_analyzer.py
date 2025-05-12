# app/features/analyzer/bubble_analyzer.py
"""
Analyzes bubble fill scores.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional

from app.core.constants import DEFAULT_BLOCK_SIZE, DEFAULT_C_VALUE


class BubbleAnalyzer:
    """
    Analyzes bubbles to determine fill scores.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.bubble_threshold = params.get('bubble_threshold', 100.0)
        self.score_multiplier = params.get('score_multiplier', 1.5)
        self.block_size = params.get('block_size', DEFAULT_BLOCK_SIZE)
        self.c_value = params.get('c_value', DEFAULT_C_VALUE)

    def analyze(
        self,
        image: np.ndarray,
        bubbles: np.ndarray
    ) -> List[Dict]:
        """
        Analyze bubble fill scores.
        """
        if bubbles is None:
            return []

        # Create threshold image
        thresh = self._create_threshold_image(image)

        # Group bubbles by position
        grouped = self._group_bubbles(bubbles, image.shape[1])

        # Calculate scores for each group
        scores = []
        for q_num, row_bubbles in grouped:
            row_scores = self._analyze_row(thresh, row_bubbles, q_num)
            scores.append(row_scores)

        return scores

    def _create_threshold_image(self, image: np.ndarray) -> np.ndarray:
        """
        Create adaptive threshold image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure odd block size
        block_size = self.block_size
        if block_size % 2 == 0:
            block_size += 1

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, self.c_value
        )

        return thresh

    def _group_bubbles(
        self,
        bubbles: np.ndarray,
        image_width: int
    ) -> List[tuple]:
        """
        Group bubbles by question rows.
        """
        midpoint_x = image_width // 2
        left_bubbles = [b for b in bubbles if b[0] < midpoint_x]
        right_bubbles = [b for b in bubbles if b[0] >= midpoint_x]

        # Process columns
        items_per_column = 30
        columns = [
            (left_bubbles, 1),
            (right_bubbles, items_per_column + 1)
        ]

        grouped = []
        for column_bubbles, start_q in columns:
            if not column_bubbles:
                continue

            rows = self._group_into_rows(column_bubbles)
            for row_idx, row_bubbles in enumerate(rows):
                q_num = start_q + row_idx
                grouped.append((q_num, row_bubbles))

        return grouped

    def _group_into_rows(
        self,
        bubbles: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Group bubbles into rows based on Y position.
        """
        row_threshold = self.params.get('row_threshold', 8)
        y_sorted = sorted(bubbles, key=lambda b: b[1])

        rows = []
        current_row = [y_sorted[0]]

        for i in range(1, len(y_sorted)):
            if abs(y_sorted[i][1] - current_row[0][1]) < row_threshold:
                current_row.append(y_sorted[i])
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [y_sorted[i]]

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        rows.sort(key=lambda r: r[0][1] if r else 0)
        return rows

    def _analyze_row(
        self,
        thresh: np.ndarray,
        row_bubbles: List[np.ndarray],
        q_num: int
    ) -> Dict:
        """
        Analyze a single row of bubbles.
        """
        choice_labels = ['A', 'B', 'C', 'D', 'E']
        bubble_details = []

        for idx, bubble in enumerate(row_bubbles[:len(choice_labels)]):
            x, y, r = bubble
            label = choice_labels[idx] if idx < len(choice_labels) else ''

            # Calculate fill score
            score = self._calculate_fill_score(thresh, x, y, r)
            final_score = score * self.score_multiplier

            bubble_details.append({
                'label': label,
                'x': int(x),
                'y': int(y),
                'r': int(r),
                'score': final_score,
                'selected': False
            })

        # Find highest scoring bubble
        max_score = 0
        selected_idx = -1

        for i, detail in enumerate(bubble_details):
            if detail['score'] > self.bubble_threshold and detail['score'] > max_score:
                max_score = detail['score']
                selected_idx = i

        if selected_idx >= 0:
            bubble_details[selected_idx]['selected'] = True

        return {
            'question_number': q_num,
            'choices': bubble_details
        }

    def _calculate_fill_score(
        self,
        thresh: np.ndarray,
        x: int,
        y: int,
        r: int
    ) -> float:
        """
        Calculate bubble fill score.
        """
        if r <= 0:
            return 0.0

        # Create mask for inner bubble area
        bubble_mask = np.zeros_like(thresh)
        inner_radius = int(r * 0.8)

        if inner_radius <= 0:
            return 0.0

        cv2.circle(bubble_mask, (x, y), inner_radius, 255, -1)

        # Count filled pixels
        fill_count = cv2.countNonZero(cv2.bitwise_and(thresh, bubble_mask))

        # Normalize by area
        area = np.pi * inner_radius * inner_radius
        normalized = fill_count / area * 100 if area > 0 else 0.0

        return normalized
