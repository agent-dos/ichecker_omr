# app/features/corners/strategies/edge.py
"""
Edge detection-based corner detection strategy.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class EdgeStrategy:
    """
    Detects corners using edge detection.
    """

    def detect(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Detect candidates using edge detection.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            candidate = self._analyze_contour(
                contour, min_area, max_area, qr_polygon
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    def _analyze_contour(
        self,
        contour: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]]
    ) -> Optional[Dict]:
        """Analyze contour for corner candidacy."""
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            return None

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)

        if qr_polygon and self._point_in_polygon(center, qr_polygon):
            return None

        return {
            'contour': contour,
            'center': center,
            'area': area,
            'bbox': (x, y, w, h)
        }

    def _point_in_polygon(
        self,
        point: Tuple[int, int],
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if point is inside polygon."""
        import cv2
        return cv2.pointPolygonTest(
            np.array(polygon), point, False
        ) >= 0
