# filename: app/features/rectification/rectifier.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any  # Added Any

# Import config helper and constants
from app.core.config import get_cv2_flag, CV2_INTERPOLATION_FLAGS
try:
    # Ensure these constants define the target output size correctly
    from app.core.constants import PAGE_WIDTH, PAGE_HEIGHT
except ImportError:
    # Provide fallback values if constants cannot be imported
    logging.warning(
        "Could not import PAGE_WIDTH, PAGE_HEIGHT from constants. Using default 850x1202.")
    PAGE_WIDTH, PAGE_HEIGHT = 850, 1202

logger = logging.getLogger(__name__)


class ImageRectifier:
    """
    Handles image rectification for tilted sheets using provided parameters.
    """
    # --- ADDED __init__ method ---

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the rectifier with configuration parameters.

        Args:
            params (Dict[str, Any]): Dictionary containing rectification parameters, e.g.,
                    {'warp_interpolation': 'INTER_LINEAR', 'dst_margin': 0}
        """
        self.params = params
        # Pre-fetch specific params for clarity and use within methods
        self.warp_interpolation_key = self.params.get(
            'warp_interpolation', 'INTER_LINEAR')
        self.dst_margin = self.params.get(
            'dst_margin', 0)  # Get margin from params
        # Fail-safe parameter (check if needed based on config structure)
        # self.fail_safe = self.params.get('fail_safe_return_original', True)
        logger.debug(
            f"ImageRectifier initialized with params: warp_interpolation='{self.warp_interpolation_key}', dst_margin={self.dst_margin}")
        # Validate margin immediately
        if self.dst_margin < 0 or self.dst_margin * 2 >= min(PAGE_WIDTH, PAGE_HEIGHT):
            logger.warning(
                f"Invalid dst_margin ({self.dst_margin}) corrected to 0.")
            self.dst_margin = 0

    def rectify(
        self,
        image: np.ndarray,
        corners: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """ Rectify tilted image using corner markers. """
        logger.info("--- Entering ImageRectifier.rectify ---")
        if image is None or corners is None:
            logger.warning("Rectify called with None image or corners.")
            return None, None

        logger.info("Attempting to extract source points...")
        src_points = self._extract_corner_points(corners)
        if src_points is None:
            # Log already happens in _extract_corner_points
            logger.error(
                "Failed to extract 4 valid source points. Cannot rectify.")
            return None, None
        logger.info(f"Source points extracted (Shape: {src_points.shape})")

        logger.info("Calculating standard destination points...")
        dst_points, output_width, output_height = self._calculate_standard_dst_points(
            margin=self.dst_margin
        )
        logger.info(
            f"Destination points calculated for size {output_width}x{output_height} with margin {self.dst_margin}")

        logger.info("Computing perspective transform...")
        try:
            transform = cv2.getPerspectiveTransform(
                src_points.astype(np.float32),
                dst_points.astype(np.float32)
            )
            if transform is None:
                logger.error(
                    "getPerspectiveTransform returned None (points might be collinear).")
                return None, None  # Explicitly return None if transform is None
            logger.info(f"Transform matrix calculated.")
            # Log matrix at debug level
            logger.debug(f"Transform Matrix:\n{transform}")
        except cv2.error as e:
            logger.error(f"cv2.getPerspectiveTransform failed: {e}")
            logger.error(
                f"Source Points (type {type(src_points)}):\n{src_points}")
            logger.error(
                f"Destination Points (type {type(dst_points)}):\n{dst_points}")
            return None, None

        interpolation_flag = get_cv2_flag(
            self.warp_interpolation_key, CV2_INTERPOLATION_FLAGS, cv2.INTER_LINEAR
        )
        logger.info(
            f"Applying warpPerspective (Flag: {interpolation_flag}, Size: {output_width}x{output_height})...")

        try:
            rectified = cv2.warpPerspective(
                image, transform, (output_width,
                                   output_height), flags=interpolation_flag
            )
            logger.info("warpPerspective successful.")
        except cv2.error as e:
            logger.error(f"cv2.warpPerspective failed: {e}")
            return None, None  # Return None on warping failure

        logger.info("--- Exiting ImageRectifier.rectify ---")
        return rectified, transform

    def calculate_angle(self, corners: Dict) -> float:
        """
        Calculate rotation angle from corners (using top edge preferably).
        """
        if not corners:
            return 0.0
        # Ensure corner data is valid before accessing 'center'
        tl_data = corners.get('top_left', {})
        tr_data = corners.get('top_right', {})
        tl = tl_data.get('center') if isinstance(tl_data, dict) else None
        tr = tr_data.get('center') if isinstance(tr_data, dict) else None

        if tl and tr:
            pass  # Use top edge if available
        else:
            # Fallback to bottom edge if top edge is missing
            bl_data = corners.get('bottom_left', {})
            br_data = corners.get('bottom_right', {})
            bl = bl_data.get('center') if isinstance(bl_data, dict) else None
            br = br_data.get('center') if isinstance(br_data, dict) else None

            if bl and br:
                logger.debug("Calculating angle using bottom edge.")
                tl, tr = bl, br  # Use bottom corners for angle calculation
            else:
                logger.warning(
                    "Cannot calculate angle, missing required corner pairs.")
                return 0.0

        # Ensure coordinates are numbers before calculation
        try:
            dx = float(tr[0]) - float(tl[0])
            dy = float(tr[1]) - float(tl[1])
        except (TypeError, IndexError) as e:
            logger.error(
                f"Invalid coordinate format for angle calculation: tl={tl}, tr={tr}. Error: {e}")
            return 0.0

        if dx == 0 and dy == 0:
            return 0.0  # Avoid atan2(0,0)
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
            if corner_data is None or not isinstance(corner_data, dict) or 'center' not in corner_data:
                logger.warning(
                    f"Corner '{name}' missing or invalid dict structure for rectification.")
                return None  # Ensure all 4 are present and are dicts with 'center'
            point_coords = corner_data['center']
            # Validate the coordinates themselves
            if not (isinstance(point_coords, (tuple, list)) and len(point_coords) == 2 and all(isinstance(coord, (int, float, np.number)) for coord in point_coords)):
                logger.warning(
                    f"Invalid coordinates format for corner {name}: {point_coords}")
                return None
            points.append(point_coords)
        # Return as float32 numpy array, shape (4, 2)
        return np.array(points, dtype=np.float32)

    # Accepts margin argument now

    def _calculate_standard_dst_points(
        self,
        margin: int  # Get margin from __init__ via method call
    ) -> Tuple[np.ndarray, int, int]:
        """
        Calculate destination points based on standard page dimensions (PAGE_WIDTH, PAGE_HEIGHT).
        """
        output_width = PAGE_WIDTH
        output_height = PAGE_HEIGHT

        # Margin validation is now done in __init__

        # Define destination points using the validated margin
        dst_points = np.array([
            [margin, margin],                                       # Top-left
            [output_width - 1 - margin, margin],                    # Top-right
            [output_width - 1 - margin, output_height - 1 - margin],  # Bottom-right
            [margin, output_height - 1 - margin]                    # Bottom-left
        ], dtype=np.float32)  # Must be float32

        # logger.debug(f"Calculated standard destination points with margin {margin} for output size {output_width}x{output_height}")

        return dst_points, output_width, output_height
