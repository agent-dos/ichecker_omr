# filename: app/features/analyzer/bubble_detector.py
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any  # Added Any
import logging

# Import filtering functions
from app.common.geometry.filters import filter_by_polygon, filter_by_quadrilateral
# Import visualization function for fallback/final viz step
from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles


logger = logging.getLogger(__name__)


class BubbleDetector:
    """ Detects circular bubbles in answer sheets using Hough Circle Transform. """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the BubbleDetector with configuration parameters.

        Args:
            params (Dict[str, Any]): Configuration dictionary specifically for
                bubble detection (e.g., from config['bubble_detection']). Expected
                keys include Hough transform parameters and filtering options.
        """
        self.params = params
        # Fetch parameters specific to this detector for use in detect()
        self.blur_ksize = self.params.get('gaussian_blur_ksize', 5)
        self.hough_dp = float(self.params.get('hough_dp', 1.0))
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
        self.qr_margin_factor = float(
            self.params.get('qr_filter_margin_factor', 1.0))

        logger.debug(f"BubbleDetector initialized with params: {self.params}")

    def detect(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]] = None,
        corners: Optional[Dict] = None,
        visualize_steps: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Detects bubbles in the image using Hough Circle Transform and applies filtering.

        Args:
            image (np.ndarray): The input image (expected BGR or Grayscale) where
                                bubbles need to be detected.
            qr_polygon (Optional[List[Tuple[int, int]]]): A list of (x, y) tuples
                representing the vertices of the detected QR code polygon. Used for
                filtering bubbles that fall within the QR code area. Defaults to None.
            corners (Optional[Dict]): A dictionary containing the coordinates of the
                detected sheet corners (e.g., {'top_left': {'center':(x,y)}, ...}).
                Used for filtering bubbles outside the main answer area. Defaults to None.
            visualize_steps (bool): If True, intermediate visualization images
                for each processing step (grayscale, blur, Hough, filters) will
                be generated and returned in the dictionary. Defaults to False.

        Returns:
            Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]: A tuple containing:
                - Detected Bubbles (Optional[np.ndarray]): A NumPy array where each
                  row represents a detected bubble as (x_center, y_center, radius).
                  Returns None if a critical error occurs, or an empty array if no
                  circles are found or all are filtered out.
                - Visualization Steps (Dict[str, np.ndarray]): A dictionary where keys
                  are step names (e.g., "01_Grayscale", "03_HoughCircles_Raw",
                  "99_FinalDetection") and values are the corresponding BGR visualization
                  images, if visualize_steps was True.
        """
        viz_steps: Dict[str, np.ndarray] = {}
        logger.info("--- Starting Bubble Detection ---")

        if image is None:
            logger.error("Bubble detection received None image.")
            return None, viz_steps

        # --- Step 1: Ensure Grayscale and prepare BGR copy for visualization ---
        current_image_for_viz = None  # Will hold BGR version
        gray = None  # Will hold Grayscale version
        try:
            if len(image.shape) == 2:  # Already Grayscale
                gray = image.copy()
                current_image_for_viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 3:  # BGR
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    current_image_for_viz = image.copy()
                elif image.shape[2] == 4:  # BGRA
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                    current_image_for_viz = cv2.cvtColor(
                        image, cv2.COLOR_BGRA2BGR)
                else:
                    raise ValueError(
                        f"Unsupported channel count: {image.shape[2]}")
            else:
                raise ValueError(
                    f"Unsupported image dimensions: {len(image.shape)}")

            if visualize_steps:
                viz_steps["01_Grayscale"] = cv2.cvtColor(
                    gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(
                f"Error during grayscale conversion in BubbleDetector: {e}", exc_info=True)
            return None, viz_steps  # Cannot proceed without grayscale

        # --- Step 2: Gaussian Blur ---
        blurred_gray = gray  # Initialize with gray in case blur is skipped or fails
        try:
            # Ensure ksize is odd and positive
            ksize = self.blur_ksize if isinstance(
                self.blur_ksize, int) and self.blur_ksize > 0 else 0
            if ksize > 0:
                if ksize % 2 == 0:
                    ksize += 1
                blurred_gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
                if visualize_steps:
                    viz_steps["02_GaussianBlur"] = cv2.cvtColor(
                        blurred_gray, cv2.COLOR_GRAY2BGR)
            else:
                logger.debug("Blur kernel size is <= 0, skipping blur.")
        except Exception as e:
            logger.warning(
                f"Error during GaussianBlur in BubbleDetector: {e}. Using unblurred image.")
            blurred_gray = gray  # Fallback to unblurred

        # --- Step 3: Hough Circle Transform ---
        # Holds detected circles (x, y, r)
        circles: Optional[np.ndarray] = None
        try:
            logger.debug(f"HoughCircles params: dp={self.hough_dp}, minDist={self.hough_minDist}, "
                         f"p1={self.hough_param1}, p2={self.hough_param2}, "
                         f"minR={self.hough_minRadius}, maxR={self.hough_maxRadius}")
            # Validate radii to prevent errors
            min_radius = max(0, self.hough_minRadius)
            # Max must be > min
            max_radius = max(min_radius + 1, self.hough_maxRadius)

            circles_raw = cv2.HoughCircles(
                blurred_gray, cv2.HOUGH_GRADIENT,
                dp=self.hough_dp, minDist=self.hough_minDist,
                param1=self.hough_param1, param2=self.hough_param2,
                minRadius=min_radius, maxRadius=max_radius
            )

            if circles_raw is not None:
                # Convert to integer coordinates and radius: [N, 3] array
                circles = np.round(circles_raw[0, :]).astype(int)
                logger.info(f"HoughCircles found {len(circles)} raw circles.")
                if visualize_steps:
                    # Draw raw detections on the BGR image copy
                    viz_hough = current_image_for_viz.copy()
                    for (x_c, y_c, r_c) in circles:
                        cv2.circle(viz_hough, (x_c, y_c), r_c,
                                   (0, 255, 0), 1)  # Green outline
                        cv2.circle(viz_hough, (x_c, y_c), 2,
                                   (0, 0, 255), -1)   # Red center dot
                    viz_steps["03_HoughCircles_Raw"] = viz_hough
            else:
                logger.warning("HoughCircles found no circles.")
                # Return an empty array if no circles are found
                return np.array([]), viz_steps

        except Exception as e:
            logger.error(
                f"Error during HoughCircles in BubbleDetector: {e}", exc_info=True)
            # Return None to indicate a critical failure in detection
            return None, viz_steps

        # --- Filtering Stages ---
        # Start with the raw circles found
        current_circles_to_filter = circles if circles is not None else np.array([
        ])
        num_before_filter = len(current_circles_to_filter)

        # --- Step 4: Filter by Corner Boundary ---
        num_after_boundary_filter = num_before_filter
        if self.filter_by_corners_enabled and current_circles_to_filter.size > 0:
            # Check if valid corners are provided
            if corners and isinstance(corners, dict) and len(corners) == 4 and all(c and isinstance(c, dict) and 'center' in c for c in corners.values()):
                logger.info("Filtering bubbles by corner boundary...")
                boundary_quad = self._get_boundary_from_corners(corners)
                if boundary_quad is not None:
                    # Keep points INSIDE the quad + margin
                    filtered_by_boundary = filter_by_quadrilateral(
                        current_circles_to_filter, boundary_quad, margin=self.boundary_margin
                    )
                    # Update the list of circles to carry forward
                    current_circles_to_filter = filtered_by_boundary if filtered_by_boundary is not None else np.array([
                    ])
                    num_after_boundary_filter = len(current_circles_to_filter)
                    logger.info(
                        f"After corner boundary filter: {num_after_boundary_filter} circles remain.")

                    if visualize_steps:
                        viz_boundary = current_image_for_viz.copy()
                        # Draw the boundary used for filtering
                        cv2.polylines(viz_boundary, [boundary_quad.astype(
                            np.int32)], True, (255, 0, 0), 1)  # Blue boundary
                        # Draw the circles that remain *after* filtering
                        if current_circles_to_filter.size > 0:
                            for (x_f, y_f, r_f) in current_circles_to_filter:
                                cv2.circle(viz_boundary, (x_f, y_f),
                                           # Green outline
                                           r_f, (0, 255, 0), 1)
                        viz_steps["04_Filter_CornerBoundary"] = viz_boundary
                else:
                    logger.warning(
                        "Could not form boundary from corners for bubble filtering.")
            else:
                logger.warning(
                    "Corner filtering for bubbles enabled, but invalid or incomplete corners provided.")
        elif self.filter_by_corners_enabled:
            logger.debug("Corner filtering enabled, but no circles to filter.")

        # --- Step 5: Filter by QR Polygon ---
        num_after_qr_filter = num_after_boundary_filter
        if self.filter_by_qr_enabled and current_circles_to_filter.size > 0:
            # Check if a valid polygon exists
            if qr_polygon and len(qr_polygon) >= 3:
                logger.info("Filtering bubbles overlapping with QR polygon...")
                # Keep points OUTSIDE the polygon
                filtered_by_qr = filter_by_polygon(
                    # Keep if outside or exactly on edge
                    current_circles_to_filter, qr_polygon, margin=0
                    # Consider adding a small negative margin if needed: e.g., margin=-(self.qr_margin_factor * self.hough_maxRadius)
                )
                # Update the list of circles
                current_circles_to_filter = filtered_by_qr if filtered_by_qr is not None else np.array([
                ])
                num_after_qr_filter = len(current_circles_to_filter)
                logger.info(
                    f"After QR polygon filter: {num_after_qr_filter} circles remain.")

                if visualize_steps:
                    viz_qr_filt = current_image_for_viz.copy()
                    # Draw the QR polygon used for filtering
                    cv2.polylines(viz_qr_filt, [np.array(
                        # Red QR area
                        qr_polygon, dtype=np.int32)], True, (0, 0, 255), 1)
                    # Draw the circles that remain *after* filtering
                    if current_circles_to_filter.size > 0:
                        for (x_q, y_q, r_q) in current_circles_to_filter:
                            cv2.circle(viz_qr_filt, (x_q, y_q), r_q,
                                       (0, 255, 0), 1)  # Green outline
                    viz_steps["05_Filter_QRBoundary"] = viz_qr_filt
            else:
                logger.debug(
                    "QR filtering enabled, but no QR polygon provided or polygon is invalid.")
        elif self.filter_by_qr_enabled:
            logger.debug("QR filtering enabled, but no circles to filter.")

        # The final set of bubbles after all filtering stages
        final_bubbles = current_circles_to_filter

        # --- Step 6: Final Visualization ---
        if visualize_steps:
            # Use the standard bubble visualizer for the final output
            if final_bubbles is not None and final_bubbles.size > 0:
                # Generate visualization using the final bubbles
                viz_steps["99_FinalDetection"] = visualize_bubbles(
                    current_image_for_viz.copy(), final_bubbles)
            else:
                # Create an image indicating no bubbles were detected/left
                final_viz_img = current_image_for_viz.copy()
                cv2.putText(final_viz_img, "No Bubbles Detected / Filtered",
                            # Position text
                            (30, int(final_viz_img.shape[0] / 2)),
                            # Red text
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                viz_steps["99_FinalDetection"] = final_viz_img

        logger.info(
            f"--- Bubble Detection Finished. Found {len(final_bubbles) if final_bubbles is not None else 0} final bubbles. ---")
        # Return the final numpy array of bubbles and the visualization dict
        return final_bubbles, viz_steps

    def _get_boundary_from_corners(self, corners: Dict) -> Optional[np.ndarray]:
        """
        Creates a quadrilateral boundary polygon from a dictionary of corner data.

        Args:
            corners (Dict): A dictionary where keys are corner names ('top_left',
                'top_right', 'bottom_right', 'bottom_left') and values are
                dictionaries containing at least a 'center' key with (x, y) coordinates.

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (4, 2) containing the
                corner coordinates as float32, in the order TL, TR, BR, BL.
                Returns None if the input dictionary is invalid or incomplete.
        """
        # Define the required order for the quadrilateral
        required_order = ['top_left', 'top_right',
                          'bottom_right', 'bottom_left']

        # Check if all required keys exist and contain valid center data
        if not all(name in corners and isinstance(corners[name], dict) and 'center' in corners[name] for name in required_order):
            logger.warning(
                "Cannot form boundary: one or more required corners are missing or malformed in the input dict.")
            return None

        points = []
        try:
            for name in required_order:
                point_coords = corners[name]['center']
                # Validate the coordinates themselves
                if not (isinstance(point_coords, (tuple, list)) and len(point_coords) == 2 and all(isinstance(coord, (int, float, np.number)) for coord in point_coords)):
                    raise ValueError(
                        f"Invalid coordinate format for corner '{name}': {point_coords}")
                points.append(point_coords)
            # Return as a float32 numpy array, shape (4, 2)
            return np.array(points, dtype=np.float32)
        except (KeyError, ValueError, TypeError) as e:
            logger.error(
                f"Error creating boundary polygon from corners: {e}", exc_info=True)
            return None
