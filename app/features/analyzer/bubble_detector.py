# filename: app/features/analyzer/bubble_detector.py
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any  # Added Any
import logging

# Assuming these are available or define them if not already
from app.common.geometry.filters import filter_by_polygon, filter_by_quadrilateral
# Constants like DEFAULT_BUBBLE_RADIUS_MIN etc., are now expected to come from self.params

logger = logging.getLogger(__name__)


class BubbleDetector:
    """ Detects circular bubbles in answer sheets using Hough Circle Transform. """

    def __init__(self, params: Dict[str, Any]):
        """ Initialize with bubble_detection parameters. """
        self.params = params
        # Fetch parameters specific to this detector for use in detect()
        self.blur_ksize = self.params.get('gaussian_blur_ksize', 5)
        self.hough_dp = float(self.params.get('hough_dp', 1.0))  # Ensure float
        self.hough_minDist = int(self.params.get('hough_minDist', 20))
        self.hough_param1 = int(self.params.get('hough_param1', 50))
        self.hough_param2 = int(self.params.get('hough_param2', 18))
        self.hough_minRadius = int(self.params.get('hough_minRadius', 10))
        self.hough_maxRadius = int(self.params.get('hough_maxRadius', 20))

        self.filter_by_corners_enabled = self.params.get(
            'filter_by_corners', True)
        self.boundary_margin = int(
            self.params.get('boundary_filter_margin', 5))
        self.filter_by_qr_enabled = self.params.get('filter_by_qr', True)
        # Ensure qr_margin_factor is float for multiplication
        self.qr_margin_factor = float(
            self.params.get('qr_filter_margin_factor', 1.0))

        logger.debug(f"BubbleDetector initialized with params: {self.params}")

    # --- UPDATED SIGNATURE AND RETURN TYPE ---

    def detect(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]] = None,
        corners: Optional[Dict] = None,
        visualize_steps: bool = False  # NEW PARAMETER
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Detect bubbles with optional filtering and intermediate visualizations.
        Returns: (detected_bubbles_array_or_None, visualization_steps_dict)
        """
        viz_steps: Dict[str, np.ndarray] = {}
        logger.info("--- Starting Bubble Detection ---")

        if image is None:
            logger.error("Bubble detection received None image.")
            return None, viz_steps

        # --- Step 1: Grayscale Conversion ---
        current_image_for_viz = image.copy()  # For drawing on BGR
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:  # BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # BGRA
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                current_image_for_viz = cv2.cvtColor(
                    image, cv2.COLOR_BGRA2BGR)  # Update BGR copy
            elif len(image.shape) == 2:  # Already Grayscale
                gray = image.copy()
                current_image_for_viz = cv2.cvtColor(
                    image, cv2.COLOR_GRAY2BGR)  # Create BGR copy
            else:
                logger.error(
                    f"Unsupported image format for bubble detection: {image.shape}")
                return None, viz_steps

            if visualize_steps:
                viz_steps["01_Grayscale"] = cv2.cvtColor(
                    gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(
                f"Error during grayscale conversion in BubbleDetector: {e}", exc_info=True)
            return None, viz_steps

        # --- Step 2: Gaussian Blur ---
        blurred_gray = gray  # Initialize with gray in case blurring fails
        try:
            ksize = self.blur_ksize if self.blur_ksize % 2 != 0 else self.blur_ksize + 1
            ksize = max(1, ksize)  # ksize must be positive
            if ksize > 0:
                blurred_gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
                if visualize_steps:
                    viz_steps["02_GaussianBlur"] = cv2.cvtColor(
                        blurred_gray, cv2.COLOR_GRAY2BGR)
            else:
                logger.warning("Blur kernel size is 0, skipping blur.")
        except Exception as e:
            logger.error(
                f"Error during GaussianBlur in BubbleDetector: {e}", exc_info=True)
            # Continue with unblurred gray image if blur fails, or return. For now, continue.
            blurred_gray = gray  # Fallback to unblurred

        # --- Step 3: Hough Circle Transform ---
        circles: Optional[np.ndarray] = None
        try:
            logger.debug(f"HoughCircles params: dp={self.hough_dp}, minDist={self.hough_minDist}, "
                         f"p1={self.hough_param1}, p2={self.hough_param2}, "
                         f"minR={self.hough_minRadius}, maxR={self.hough_maxRadius}")
            circles_raw = cv2.HoughCircles(
                blurred_gray, cv2.HOUGH_GRADIENT,
                dp=self.hough_dp, minDist=self.hough_minDist,
                param1=self.hough_param1, param2=self.hough_param2,
                minRadius=self.hough_minRadius, maxRadius=self.hough_maxRadius
            )

            if circles_raw is not None:
                circles = np.round(circles_raw[0, :]).astype(
                    int)  # x, y, radius
                logger.info(f"HoughCircles found {len(circles)} raw circles.")
                if visualize_steps:
                    viz_hough = current_image_for_viz.copy()  # Draw on original BGR for context
                    for (x_c, y_c, r_c) in circles:
                        cv2.circle(viz_hough, (x_c, y_c), r_c,
                                   (0, 255, 0), 1)  # Green outline
                        cv2.circle(viz_hough, (x_c, y_c), 2,
                                   (0, 0, 255), -1)  # Red center
                    viz_steps["03_HoughCircles_Raw"] = viz_hough
            else:
                logger.warning("HoughCircles found no circles.")
                # Return empty array, not None, if no circles
                return np.array([]), viz_steps

        except Exception as e:
            logger.error(
                f"Error during HoughCircles in BubbleDetector: {e}", exc_info=True)
            return None, viz_steps  # Return None if Hough fails catastrophically

        # Initialize with current set of circles for sequential filtering
        current_circles_to_filter = circles if circles is not None else np.array([
        ])

        # --- Step 4: Filter by Corner Boundary ---
        if self.filter_by_corners_enabled and current_circles_to_filter.size > 0:
            if corners and isinstance(corners, dict) and len(corners) == 4 and all(c is not None for c in corners.values()):
                logger.info("Filtering bubbles by corner boundary...")
                boundary_quad = self._get_boundary_from_corners(corners)
                if boundary_quad is not None:
                    filtered_by_boundary = filter_by_quadrilateral(
                        current_circles_to_filter, boundary_quad, margin=self.boundary_margin
                    )
                    current_circles_to_filter = filtered_by_boundary if filtered_by_boundary is not None else np.array([
                    ])
                    logger.info(
                        f"After corner boundary filter: {len(current_circles_to_filter)} circles.")
                    if visualize_steps:
                        viz_boundary = current_image_for_viz.copy()
                        cv2.polylines(viz_boundary, [boundary_quad.astype(
                            np.int32)], True, (255, 0, 0), 1)  # Blue boundary
                        if current_circles_to_filter.size > 0:
                            for (x_f, y_f, r_f) in current_circles_to_filter:
                                cv2.circle(viz_boundary, (x_f, y_f),
                                           r_f, (0, 255, 0), 1)
                        viz_steps["04_Filter_CornerBoundary"] = viz_boundary
                else:
                    logger.warning(
                        "Could not form boundary from corners for bubble filtering.")
            else:
                logger.warning(
                    "Corner filtering for bubbles enabled, but no valid/complete corners provided.")
        elif self.filter_by_corners_enabled:
            logger.info(
                "Corner filtering for bubbles enabled, but no circles to filter or no corners provided.")

        # --- Step 5: Filter by QR Polygon ---
        if self.filter_by_qr_enabled and current_circles_to_filter.size > 0:
            if qr_polygon and len(qr_polygon) > 0:
                logger.info("Filtering bubbles by QR polygon...")
                # filter_by_polygon keeps points OUTSIDE the polygon (dist < -margin)
                # If margin is positive, it means further outside.
                # For QR, we want to exclude points INSIDE the QR.
                # So, we want points whose distance to polygon is < 0.
                # If filter_by_polygon has margin as "distance from edge", a negative margin could work
                # or we invert the logic.
                # Let's assume filter_by_polygon is correctly implemented to keep points OUTSIDE.
                qr_filter_margin_px = 0  # For QR, we usually want to exclude if center is inside
                # Or add a small buffer around QR.
                if self.hough_maxRadius > 0:  # Protective margin
                    qr_filter_margin_px = int(
                        self.qr_margin_factor * self.hough_maxRadius)

                # We need points *not* in QR. filter_by_polygon as is should keep points OUTSIDE.
                # A positive margin here means "keep if at least margin away from polygon boundary"
                # For QR exclusion, we want to *remove* points *inside*.
                # A simpler approach: iterate and use cv2.pointPolygonTest.
                circles_after_qr = []
                for circ in current_circles_to_filter:
                    pt_center = (int(circ[0]), int(circ[1]))
                    # If point is outside QR polygon (dist < 0), keep it. Add margin.
                    # pointPolygonTest returns positive if inside.
                    # We want to exclude if test_dist >= -qr_filter_margin_px
                    test_dist = cv2.pointPolygonTest(
                        np.array(qr_polygon, dtype=np.int32), pt_center, True)
                    if test_dist < qr_filter_margin_px:  # Keep if outside or on edge or slightly inside margin
                        circles_after_qr.append(circ)
                    # else: logger.debug(f"Circle at {pt_center} excluded by QR filter (dist: {test_dist})")

                current_circles_to_filter = np.array(
                    circles_after_qr) if circles_after_qr else np.array([])

                logger.info(
                    f"After QR polygon filter: {len(current_circles_to_filter)} circles.")
                if visualize_steps:
                    viz_qr_filt = current_image_for_viz.copy()
                    if qr_polygon:  # Draw QR polygon
                        cv2.polylines(viz_qr_filt, [np.array(
                            # Red QR area
                            qr_polygon, dtype=np.int32)], True, (0, 0, 255), 1)
                    if current_circles_to_filter.size > 0:
                        for (x_q, y_q, r_q) in current_circles_to_filter:
                            cv2.circle(viz_qr_filt, (x_q, y_q),
                                       r_q, (0, 255, 0), 1)
                    viz_steps["05_Filter_QRBoundary"] = viz_qr_filt
            else:
                logger.info(
                    "No circles to filter by QR or QR polygon not available.")
        elif self.filter_by_qr_enabled:
            logger.info(
                "QR filtering for bubbles enabled, but no circles or QR polygon.")

        final_bubbles = current_circles_to_filter

        # --- Step 6: Final Visualization ---
        if visualize_steps:
            from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles  # Standard viz
            if final_bubbles is not None and final_bubbles.size > 0:
                viz_steps["99_FinalDetection"] = visualize_bubbles(
                    current_image_for_viz.copy(), final_bubbles)
            else:  # Case where no bubbles are left or found
                final_viz_img = current_image_for_viz.copy()
                cv2.putText(final_viz_img, "No Bubbles Detected/Left", (50, int(final_viz_img.shape[0]/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                viz_steps["99_FinalDetection"] = final_viz_img

        logger.info(
            f"--- Bubble Detection Finished. Found {len(final_bubbles) if final_bubbles is not None else 0} bubbles. ---")
        return final_bubbles, viz_steps

    def _get_boundary_from_corners(self, corners: Dict) -> Optional[np.ndarray]:
        """ Create boundary quadrilateral from a valid corners dictionary. """
        # Assumes corners dict is valid (e.g., {'top_left': {'center': (x,y)}, ...})
        required_corners = ['top_left', 'top_right',
                            'bottom_right', 'bottom_left']
        if not all(name in corners and isinstance(corners[name], dict) and 'center' in corners[name] for name in required_corners):
            logger.warning(
                "Cannot form boundary: one or more corners missing or malformed.")
            return None
        try:
            points = [corners[name]['center'] for name in required_corners]
            for i, p in enumerate(points):  # Validate points
                if not (isinstance(p, (tuple, list)) and len(p) == 2 and all(isinstance(coord, (int, float, np.number)) for coord in p)):
                    logger.error(
                        f"Invalid point format in _get_boundary_from_corners for '{required_corners[i]}': {p}")
                    return None
            return np.array(points, dtype=np.float32)
        except Exception as e:
            logger.error(
                f"Error creating boundary from corners: {e}", exc_info=True)
            return None
