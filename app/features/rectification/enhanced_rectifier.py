# app/features/rectification/enhanced_rectifier.py
"""
Enhanced image rectification with better angle calculation and transform.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class EnhancedRectifier:
    """Enhanced rectification with robust angle detection and transform."""

    def __init__(self, params: Dict):
        """Initialize enhanced rectifier."""
        self.params = params
        self.margin = params.get('dst_margin', 10)

    def rectify(
        self,
        image: np.ndarray,
        corners: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Perform enhanced rectification."""
        logger.info("Starting enhanced rectification")

        # Extract corner points
        src_points = self._extract_corner_points(corners)
        if src_points is None:
            logger.error("Failed to extract corner points")
            return None, None

        # Calculate optimal destination points
        dst_points, dst_width, dst_height = self._calculate_optimal_dst_points(
            src_points, image.shape
        )

        # Compute perspective transform
        try:
            transform = cv2.getPerspectiveTransform(
                src_points.astype(np.float32),
                dst_points.astype(np.float32)
            )

            # Apply transform with high-quality interpolation
            rectified = cv2.warpPerspective(
                image, transform, (dst_width, dst_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            logger.info("Rectification successful")
            return rectified, transform

        except cv2.error as e:
            logger.error(f"OpenCV error during rectification: {e}")
            return None, None

    def calculate_angle(self, corners: Dict) -> float:
        """Calculate rotation angle with fallback options."""
        if not corners:
            return 0.0

        # Try top edge first
        tl = corners.get('top_left', {}).get('center')
        tr = corners.get('top_right', {}).get('center')

        if tl and tr:
            return self._calculate_edge_angle(tl, tr)

        # Fallback to bottom edge
        bl = corners.get('bottom_left', {}).get('center')
        br = corners.get('bottom_right', {}).get('center')

        if bl and br:
            return self._calculate_edge_angle(bl, br)

        # Try diagonal
        if tl and br:
            expected_angle = np.degrees(
                np.arctan2(br[1] - tl[1], br[0] - tl[0]))
            actual_angle = 45.0  # Expected diagonal angle
            return expected_angle - actual_angle

        return 0.0

    def _calculate_edge_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate angle of an edge."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        if abs(dx) < 1:  # Vertical edge
            return 90.0 if dy > 0 else -90.0

        return np.degrees(np.arctan2(dy, dx))

    def _extract_corner_points(self, corners: Dict) -> Optional[np.ndarray]:
        """Extract and validate corner points."""
        order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        points = []

        for name in order:
            corner_data = corners.get(name)
            if not corner_data or 'center' not in corner_data:
                logger.error(f"Missing corner: {name}")
                return None

            points.append(corner_data['center'])

        return np.array(points, dtype=np.float32)

    def _calculate_optimal_dst_points(
        self,
        src_points: np.ndarray,
        src_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, int, int]:
        """Calculate optimal destination points maintaining aspect ratio."""
        # Calculate dimensions of source quadrilateral
        width = np.linalg.norm(src_points[1] - src_points[0])
        height = np.linalg.norm(src_points[3] - src_points[0])

        # Use detected dimensions with margin
        dst_width = int(width) + 2 * self.margin
        dst_height = int(height) + 2 * self.margin

        # Define destination points
        dst_points = np.array([
            [self.margin, self.margin],
            [dst_width - self.margin, self.margin],
            [dst_width - self.margin, dst_height - self.margin],
            [self.margin, dst_height - self.margin]
        ], dtype=np.float32)

        return dst_points, dst_width, dst_height
