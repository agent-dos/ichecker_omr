# filename: app/features/corners/strategies/threshold.py
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any  # Added Any
import logging

# Import config helper
from app.core.config import get_cv2_flag, CV2_THRESH_TYPES, CV2_MORPH_OPS

logger = logging.getLogger(__name__)


class ThresholdStrategy:
    """ Detects corners using binary threshold with morphology. """

    def __init__(self, params: Dict[str, Any]):
        """ Initialize with threshold strategy parameters. """
        self.params = params
        self.enabled = self.params.get('enabled', True)
        self.levels = self.params.get('levels', [30, 50, 70, 90])
        self.thresh_type_key = self.params.get(
            'threshold_type', 'THRESH_BINARY_INV')
        self.morph_op1_key = self.params.get('morph_op1', 'MORPH_CLOSE')
        self.morph_op2_key = self.params.get('morph_op2', 'MORPH_OPEN')
        self.morph_ksize = self.params.get('morph_ksize', 5)
        self.solidity_min = self.params.get('solidity_min', 0.8)
        self.ar_min = self.params.get('aspect_ratio_min', 0.7)
        self.ar_max = self.params.get('aspect_ratio_max', 1.3)
        self.fill_min = self.params.get('fill_ratio_min', 0.85)
        logger.debug(
            f"ThresholdStrategy initialized (Enabled: {self.enabled})")

    def detect(
        self,
        gray: np.ndarray,
        min_area: int,  # Passed from CornerDetector
        max_area: int,  # Passed from CornerDetector
        # Passed from CornerDetector
        qr_polygon: Optional[List[Tuple[int, int]]],
        visualize_steps: bool = False
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Detect candidates using multiple threshold levels.
        Returns: (list_of_candidates, visualization_steps_dict)
        """
        viz_steps = {}
        all_candidates = []

        if not self.enabled:
            logger.debug("Threshold Strategy disabled.")
            return [], viz_steps

        # Get CV2 flags from keys
        thresh_type = get_cv2_flag(
            self.thresh_type_key, CV2_THRESH_TYPES, cv2.THRESH_BINARY_INV)
        morph_op1 = get_cv2_flag(
            self.morph_op1_key, CV2_MORPH_OPS, cv2.MORPH_CLOSE)
        morph_op2 = get_cv2_flag(
            self.morph_op2_key, CV2_MORPH_OPS, cv2.MORPH_OPEN)
        # Ensure kernel size is odd
        ksize = self.morph_ksize if self.morph_ksize % 2 != 0 else self.morph_ksize + 1
        kernel = np.ones((ksize, ksize), np.uint8) if ksize > 0 else None

        logger.info(f"Running Threshold Strategy with levels: {self.levels}")
        base_viz_img = cv2.cvtColor(
            gray, cv2.COLOR_GRAY2BGR) if visualize_steps else None

        for level in self.levels:
            level_key = f"level_{level:03d}"
            try:
                # --- Step 1: Threshold ---
                _, thresh = cv2.threshold(gray, level, 255, thresh_type)
                if visualize_steps:
                    viz_steps[f"{level_key}_01_Threshold"] = cv2.cvtColor(
                        thresh, cv2.COLOR_GRAY2BGR)

                processed_thresh = thresh.copy()  # Image to apply morphology on

                # --- Step 2: Morphology Op 1 ---
                if kernel is not None and self.morph_op1_key != 'NONE':  # Add NONE option?
                    processed_thresh = cv2.morphologyEx(
                        processed_thresh, morph_op1, kernel)
                    if visualize_steps:
                        viz_steps[f"{level_key}_02_Morph_{self.morph_op1_key}"] = cv2.cvtColor(
                            processed_thresh, cv2.COLOR_GRAY2BGR)

                # --- Step 3: Morphology Op 2 ---
                if kernel is not None and self.morph_op2_key != 'NONE':
                    processed_thresh = cv2.morphologyEx(
                        processed_thresh, morph_op2, kernel)
                    if visualize_steps:
                        viz_steps[f"{level_key}_03_Morph_{self.morph_op2_key}"] = cv2.cvtColor(
                            processed_thresh, cv2.COLOR_GRAY2BGR)

                # --- Step 4: Find Contours ---
                contours, hierarchy = cv2.findContours(
                    processed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                logger.debug(
                    f"{level_key}: Found {len(contours)} raw contours.")
                if visualize_steps:
                    viz_contours = base_viz_img.copy()
                    # Draw all raw contours
                    cv2.drawContours(
                        viz_contours, contours, -1, (0, 255, 0), 1)
                    viz_steps[f"{level_key}_04_RawContours"] = viz_contours

                # --- Step 5: Analyze Contours ---
                level_candidates = []
                viz_candidates = base_viz_img.copy() if visualize_steps else None
                for contour in contours:
                    candidate = self._analyze_contour(
                        contour, processed_thresh, min_area, max_area, qr_polygon
                    )
                    if candidate:
                        level_candidates.append(candidate)
                        if visualize_steps:
                            # Draw accepted contour and center
                            cv2.drawContours(viz_candidates, [
                                             # Yellow
                                             contour], -1, (0, 255, 255), 1)
                            cv2.circle(viz_candidates, tuple(
                                # Blue center
                                map(int, candidate['center'])), 3, (255, 0, 0), -1)

                logger.debug(
                    f"{level_key}: Found {len(level_candidates)} valid candidates.")
                if visualize_steps:
                    viz_steps[f"{level_key}_05_ValidCandidates"] = viz_candidates

                all_candidates.extend(level_candidates)

            except cv2.error as e:
                logger.error(
                    f"OpenCV error during Threshold Strategy for level {level}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error during Threshold Strategy for level {level}: {e}", exc_info=True)

        logger.info(
            f"Threshold Strategy produced {len(all_candidates)} total candidates.")
        return all_candidates, viz_steps

    def _analyze_contour(
        self,
        contour: np.ndarray,
        thresh_img_for_fill_check: np.ndarray,  # Pass specific image for fill check
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]]
    ) -> Optional[Dict]:
        """Analyze a contour for corner candidacy based on strategy params."""
        # Check area
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            return None

        # Check bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None  # Invalid bounding box
        aspect_ratio = float(w) / h
        if not (self.ar_min < aspect_ratio < self.ar_max):
            return None

        center = (x + w//2, y + h//2)

        # Skip if in QR area (use helper function)
        if qr_polygon and self._point_in_polygon(center, qr_polygon):
            return None

        # Check solidity
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                return None  # Avoid division by zero
            solidity = area / hull_area
        except cv2.error:  # Can happen for degenerate contours
            solidity = 0
        if solidity < self.solidity_min:
            return None

        # Check fill uniformity using the right thresholded image
        try:
            mask = np.zeros_like(thresh_img_for_fill_check)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask_pixels = cv2.countNonZero(mask)
            if mask_pixels == 0:
                return None  # Avoid division by zero

            masked_fill = cv2.bitwise_and(thresh_img_for_fill_check, mask)
            fill_count = cv2.countNonZero(masked_fill)
            fill_ratio = fill_count / float(mask_pixels)  # Use float division
        except cv2.error as e:
            logger.warning(f"Error during fill ratio check: {e}")
            fill_ratio = 0  # Treat as invalid if check fails

        if fill_ratio < self.fill_min:
            return None

        # If all checks pass, return candidate info
        return {
            'contour': contour, 'center': center, 'area': area,
            'bbox': (x, y, w, h), 'solidity': solidity,
            'aspect_ratio': aspect_ratio, 'fill_ratio': fill_ratio
        }

    def _point_in_polygon(self, point: Tuple, polygon: List[Tuple]) -> bool:
        """ Basic point-in-polygon test (robustness depends on library). """
        try:
            # Use cv2.pointPolygonTest if available and reliable
            # Needs polygon as Nx1x2 or Nx2 int/float array, point as tuple (float, float)
            poly_np = np.array(polygon, dtype=np.float32)
            point_float = (float(point[0]), float(point[1]))
            return cv2.pointPolygonTest(poly_np, point_float, False) >= 0
        except Exception as e:
            logger.error(
                f"Error in _point_in_polygon check: {e}. Point: {point}, Polygon: {polygon}")
            return False  # Assume outside if error occurs
