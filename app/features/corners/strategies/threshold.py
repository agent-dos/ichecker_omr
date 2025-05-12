# app/features/corners/strategies/threshold.py
"""
Threshold-based corner detection strategy.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from app.core.constants import CORNER_THRESHOLD_LEVELS


class ThresholdStrategy:
    """
    Detects corners using binary threshold with morphology.
    """

    def detect(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Detect candidates using multiple threshold levels.

        Args:
            gray: Grayscale image
            min_area: Minimum contour area
            max_area: Maximum contour area
            qr_polygon: QR polygon to exclude

        Returns:
            List of corner candidates
        """
        all_candidates = []

        for threshold_value in CORNER_THRESHOLD_LEVELS:
            candidates = self._detect_at_threshold(
                gray, threshold_value, min_area, max_area, qr_polygon
            )
            all_candidates.extend(candidates)

        return all_candidates

    def _detect_at_threshold(
        self,
        gray: np.ndarray,
        threshold_value: int,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]]
    ) -> List[Dict]:
        """Detect candidates at a specific threshold."""
        _, thresh = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY_INV
        )

        # Apply morphology to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            candidate = self._analyze_contour(
                contour, thresh, min_area, max_area, qr_polygon
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    def _analyze_contour(
        self,
        contour: np.ndarray,
        thresh: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]]
    ) -> Optional[Dict]:
        """Analyze a contour for corner candidacy."""
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            return None

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)

        # Skip if in QR area
        if qr_polygon and self._point_in_polygon(center, qr_polygon):
            return None

        # Check properties
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        aspect_ratio = float(w) / h

        if not (0.7 < aspect_ratio < 1.3) or solidity < 0.8:
            return None

        # Check fill uniformity
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        masked = cv2.bitwise_and(thresh, mask)
        fill_ratio = cv2.countNonZero(masked) / cv2.countNonZero(mask)

        if fill_ratio < 0.85:
            return None

        return {
            'contour': contour,
            'center': center,
            'area': area,
            'bbox': (x, y, w, h),
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'fill_ratio': fill_ratio
        }

    def _point_in_polygon(
        self,
        point: Tuple[int, int],
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if point is inside polygon."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / \
                                (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
