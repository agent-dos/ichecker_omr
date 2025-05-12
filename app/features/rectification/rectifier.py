# app/features/rectification/rectifier.py
"""
Image rectification module.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class ImageRectifier:
    """
    Handles image rectification for tilted sheets.
    """

    def rectify(
        self,
        image: np.ndarray,
        corners: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify tilted image using corner markers.

        Returns:
            tuple: (rectified_image, transform_matrix)
        """
        if image is None or corners is None:
            return image, None

        # Extract corner points
        src_points = self._extract_corner_points(corners)
        if src_points is None:
            return image, None

        # Calculate destination points
        dst_points = self._calculate_dst_points(src_points, image.shape)

        # Compute transformation matrix
        transform = cv2.getPerspectiveTransform(
            src_points.astype(np.float32),
            dst_points.astype(np.float32)
        )

        # Apply transformation
        h, w = self._calculate_output_size(dst_points)
        rectified = cv2.warpPerspective(
            image, transform, (w, h),
            flags=cv2.INTER_LINEAR
        )

        return rectified, transform

    def calculate_angle(self, corners: Dict) -> float:
        """
        Calculate rotation angle from corners.
        """
        if not corners:
            return 0.0

        tl = corners.get('top_left', {}).get('center')
        tr = corners.get('top_right', {}).get('center')

        if not tl or not tr:
            return 0.0

        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _extract_corner_points(self, corners: Dict) -> Optional[np.ndarray]:
        """
        Extract corner points in order.
        """
        order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        points = []

        for name in order:
            if name not in corners or corners[name] is None:
                return None
            points.append(corners[name]['center'])

        return np.array(points, dtype=np.float32)

    def _calculate_dst_points(
        self,
        src_points: np.ndarray,
        image_shape: tuple
    ) -> np.ndarray:
        """
        Calculate destination points for rectified image.
        """
        x_coords = src_points[:, 0]
        y_coords = src_points[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        width = int(max_x - min_x)
        height = int(max_y - min_y)

        margin = 20
        dst_points = np.array([
            [margin, margin],
            [width - margin, margin],
            [width - margin, height - margin],
            [margin, height - margin]
        ], dtype=np.float32)

        return dst_points

    def _calculate_output_size(
        self,
        dst_points: np.ndarray
    ) -> Tuple[int, int]:
        """
        Calculate output dimensions.
        """
        width = int(np.max(dst_points[:, 0]) - np.min(dst_points[:, 0]))
        height = int(np.max(dst_points[:, 1]) - np.min(dst_points[:, 1]))

        # Ensure minimum size
        width = max(width, 400)
        height = max(height, 600)

        return height, width
