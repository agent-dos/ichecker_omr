# app/features/corners/validators.py
"""
Corner validation utilities.
"""
import cv2
import numpy as np
from typing import List, Dict


class CornerValidator:
    """
    Validates corner candidates to filter out false positives.
    """

    def filter_qr_patterns(
        self,
        candidates: List[Dict],
        gray: np.ndarray
    ) -> List[Dict]:
        """
        Filter out QR code-like patterns from candidates.
        """
        filtered = []

        for candidate in candidates:
            x, y, w, h = candidate['bbox']
            roi = gray[y:y+h, x:x+w]

            if not self._is_qr_pattern(roi):
                filtered.append(candidate)

        return filtered

    def _is_qr_pattern(self, roi: np.ndarray) -> bool:
        """
        Check if ROI contains QR-like patterns.
        """
        if roi.size == 0:
            return False

        # Check edge density
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / edges.size

        if edge_ratio > 0.15:
            return True

        # Check pattern complexity
        complexity = self._calculate_complexity(roi)
        if complexity > 0.3:
            return True

        return False

    def _calculate_complexity(self, roi: np.ndarray) -> float:
        """
        Calculate internal complexity of ROI.
        """
        if roi.size == 0:
            return 0

        variance = np.var(roi)
        normalized_variance = variance / (roi.size * 255)
        return normalized_variance
