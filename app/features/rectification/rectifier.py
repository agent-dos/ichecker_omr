# filename: app/features/rectification/rectifier.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Import constants for standard page size
# Make sure constants.py defines PAGE_WIDTH and PAGE_HEIGHT appropriately
try:
    from app.core.constants import PAGE_WIDTH, PAGE_HEIGHT
except ImportError:
    # Fallback values if constants cannot be imported (adjust as needed)
    logger.warning(
        "Could not import PAGE_WIDTH, PAGE_HEIGHT from constants. Using default 850x1202.")
    PAGE_WIDTH, PAGE_HEIGHT = 850, 1202

class ImageRectifier:
    """
    Handles image rectification for tilted sheets.
    """

    def rectify(
        self,
        image: np.ndarray,
        corners: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:  # Return Optional np.ndarray
        """
        Rectify tilted image using corner markers.

        Returns:
            tuple: (rectified_image, transform_matrix) or (None, None) if failed
        """
        if image is None or corners is None:
            logger.warning("Rectify called with None image or corners.")
            return None, None

        # Extract corner points from detected corners dict
        src_points = self._extract_corner_points(corners)
        if src_points is None:
            logger.warning(
                "Failed to extract all 4 source corner points for rectification.")
            # If rectification is crucial, maybe raise error or return original image?
            # Returning None signals failure to rectify based on corners.
            return None, None  # Return None if not all corners are valid

        # Calculate destination points based on standard page size
        # The output size will be determined by these destination points
        dst_points, output_width, output_height = self._calculate_standard_dst_points()

        # Compute transformation matrix
        try:
            # Ensure points have the correct shape (N, 1, 2) if required by some OpenCV versions,
            # but getPerspectiveTransform usually takes (N, 2)
            transform = cv2.getPerspectiveTransform(
                src_points.astype(np.float32),  # Shape (4, 2)
                dst_points.astype(np.float32)  # Shape (4, 2)
            )
            if transform is None:
                raise cv2.error("getPerspectiveTransform returned None.")

        except cv2.error as e:
            logger.error(f"cv2.getPerspectiveTransform failed: {e}")
            logger.error(
                f"Source Points (shape {src_points.shape}):\n{src_points}")
            logger.error(
                f"Destination Points (shape {dst_points.shape}):\n{dst_points}")
            return None, None  # Return None on transform calculation failure

        # Apply transformation using the calculated standard output size
        try:
            rectified = cv2.warpPerspective(
                image,
                transform,
                (output_width, output_height),  # Target size (width, height)
                flags=cv2.INTER_LINEAR  # Use INTER_LINEAR for better quality
            )
        except cv2.error as e:
            logger.error(f"cv2.warpPerspective failed: {e}")
            return None, None  # Return None on warping failure

        return rectified, transform

    def calculate_angle(self, corners: Dict) -> float:
        """
        Calculate rotation angle from corners (using top edge preferably).
        """
        if not corners:
            return 0.0

        tl = corners.get('top_left', {}).get('center')
        tr = corners.get('top_right', {}).get('center')

        if tl and tr:
            # Use top edge if available
            pass
        else:
            # Fallback to bottom edge if top edge is missing
            bl = corners.get('bottom_left', {}).get('center')
            br = corners.get('bottom_right', {}).get('center')
            if bl and br:
                logger.debug("Calculating angle using bottom edge.")
                tl = bl
                tr = br  # Use bottom corners for angle calculation
            else:
                logger.warning(
                    "Cannot calculate angle, missing top_left/top_right or bottom_left/bottom_right corners.")
                return 0.0

        dx = float(tr[0]) - float(tl[0])
        dy = float(tr[1]) - float(tl[1])

        # Avoid division by zero if points are identical
        if dx == 0 and dy == 0:
            return 0.0

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _extract_corner_points(self, corners: Dict) -> Optional[np.ndarray]:
        """
        Extract corner points in order: top-left, top-right, bottom-right, bottom-left.
        Returns None if any corner is missing or invalid.
        """
        # Standard order for cv2.getPerspectiveTransform
        order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        points = []

        for name in order:
            corner_data = corners.get(name)
            # Check if corner_data exists and contains the 'center' key
            if corner_data is None or 'center' not in corner_data:
                logger.warning(
                    f"Corner '{name}' is missing or invalid in provided dict for rectification.")
                return None  # Crucial: ensure all 4 are present and valid
            point_coords = corner_data['center']
            # Validate the coordinates themselves
            if not (isinstance(point_coords, (tuple, list)) and len(point_coords) == 2 and all(isinstance(coord, (int, float, np.number)) for coord in point_coords)):
                logger.warning(
                    f"Invalid coordinates format for corner {name}: {point_coords}")
                return None
            points.append(point_coords)

        # Return as float32 numpy array, shape (4, 2)
        return np.array(points, dtype=np.float32)

    # NEW METHOD to define destination points based on standard size
    def _calculate_standard_dst_points(
        self,
        # Margin allows the corner markers to be fully visible after rectification
        # Adjust if the markers themselves are large or if you want less border
        margin: int = 0
    ) -> Tuple[np.ndarray, int, int]:
        """
        Calculate destination points based on standard page dimensions (PAGE_WIDTH, PAGE_HEIGHT).

        Returns:
            tuple: (dst_points_array, output_width, output_height)
        """
        # Use standard page dimensions from constants (or fallback)
        output_width = PAGE_WIDTH
        output_height = PAGE_HEIGHT

        # Define destination points as the exact corners of the standard page (0-based index)
        # The source corners will be mapped to these exact locations.
        dst_points = np.array([
            [margin, margin],                                       # Top-left
            [output_width - 1 - margin, margin],                    # Top-right
            [output_width - 1 - margin, output_height - 1 - margin],  # Bottom-right
            [margin, output_height - 1 - margin]                    # Bottom-left
        ], dtype=np.float32)  # Must be float32

        logger.debug(
            f"Calculated standard destination points for output size {output_width}x{output_height}")

        return dst_points, output_width, output_height
