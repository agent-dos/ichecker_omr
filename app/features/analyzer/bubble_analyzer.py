# app/features/analyzer/bubble_analyzer.py (refactored)
"""
Analyzes bubble fill scores.
"""
import cv2
import numpy as np
from typing import List, Dict

from app.features.analyzer.processors.bubble_processor import BubbleProcessor
from app.features.analyzer.processors.score_calculator import ScoreCalculator
from app.core.constants import DEFAULT_BLOCK_SIZE, DEFAULT_C_VALUE


class BubbleAnalyzer:
    """
    Analyzes bubbles to determine fill scores.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.bubble_processor = BubbleProcessor(params)
        self.score_calculator = ScoreCalculator(params)

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
        grouped = self.bubble_processor.group_bubbles(
            bubbles, image.shape[1]
        )

        # Calculate scores for each group
        scores = []
        for q_num, row_bubbles in grouped:
            row_scores = self.score_calculator.analyze_row(
                thresh, row_bubbles, q_num
            )
            scores.append(row_scores)

        return scores

    def _create_threshold_image(self, image: np.ndarray) -> np.ndarray:
        """
        Create adaptive threshold image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure odd block size
        block_size = self.params.get('block_size', DEFAULT_BLOCK_SIZE)
        if block_size % 2 == 0:
            block_size += 1

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size,
            self.params.get('c_value', DEFAULT_C_VALUE)
        )

        return thresh
