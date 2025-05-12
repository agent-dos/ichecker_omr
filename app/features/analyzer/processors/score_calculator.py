# app/features/analyzer/processors/score_calculator.py (new file)
"""
Score calculation utilities.
"""
import cv2
import numpy as np
from typing import List, Dict


class ScoreCalculator:
    """Calculates bubble fill scores."""

    def __init__(self, params: Dict):
        self.params = params
        self.bubble_threshold = params.get('bubble_threshold', 100.0)
        self.score_multiplier = params.get('score_multiplier', 1.5)

    def analyze_row(
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
