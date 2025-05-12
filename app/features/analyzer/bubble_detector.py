# app/features/analyzer/bubble_detector.py
"""
Bubble detection using Hough Circle Transform.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from app.common.geometry.filters import filter_by_polygon, filter_by_quadrilateral
from app.core.constants import (
    DEFAULT_BUBBLE_RADIUS_MIN,
    DEFAULT_BUBBLE_RADIUS_MAX,
    DEFAULT_BUBBLE_SPACING
)


class BubbleDetector:
    """
    Detects circular bubbles in answer sheets.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.min_radius = params.get('min_radius', DEFAULT_BUBBLE_RADIUS_MIN)
        self.max_radius = params.get('max_radius', DEFAULT_BUBBLE_RADIUS_MAX)
        self.param1 = params.get('param1', 50)
        self.param2 = params.get('param2', 18)

    def detect(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]] = None,
        corners: Optional[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        Detect bubbles with optional filtering.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=DEFAULT_BUBBLE_SPACING,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)

        # Apply filters
        if corners:
            boundary = self._get_boundary_from_corners(corners)
            circles = filter_by_quadrilateral(circles, boundary, margin=5)

        if qr_polygon:
            circles = filter_by_polygon(
                circles, qr_polygon, margin=self.max_radius + 5
            )

        return circles

    def _get_boundary_from_corners(self, corners: Dict) -> Optional[np.ndarray]:
        """
        Create boundary quadrilateral from corners.
        """
        if not all(corners.values()):
            return None

        points = []
        for corner_name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            corner = corners[corner_name]
            points.append(corner['center'])

        return np.array(points, dtype=np.float32)
