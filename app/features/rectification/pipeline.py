# In app/features/rectification/pipeline.py

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict

from app.features.rectification.enhanced_detector import EnhancedCornerDetector
from app.features.rectification.enhanced_rectifier import EnhancedRectifier
from app.features.corners.detector import CornerDetector
from app.features.corners.visualizer import visualize_corners

logger = logging.getLogger(__name__)


class RectificationPipeline:
    """Complete rectification pipeline with multiple strategies."""

    def __init__(self, params: Dict):
        """Initialize rectification pipeline."""
        self.params = params
        corner_params = params.get('corner_detection', {})
        corner_params.setdefault('min_area', 100)
        corner_params.setdefault('max_area', 10000)
        corner_params.setdefault('duplicate_threshold', 30)

        self.enhanced_detector = EnhancedCornerDetector(corner_params)
        self.basic_detector = CornerDetector(corner_params)
        self.rectifier = EnhancedRectifier(
            params.get('rectification', {})
        )

    def process(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Process image through rectification pipeline."""
        results = {
            'rectified': None,
            'transform': None,
            'corners': None,
            'angle': 0.0,
            'method_used': None,
            'visualizations': {}
        }

        # Check for edge cases
        if image is None or image.size == 0:
            logger.error("Invalid image provided")
            return None, results

        # Check if image is effectively empty
        if np.std(image) < 1.0:
            logger.error("Image appears to be empty or uniform")
            return None, results

        # Try enhanced detection first
        logger.info("Trying enhanced corner detection")
        corners = None
        detection_viz = None

        try:
            corners, viz = self.enhanced_detector.detect(image, visualize)

            if corners and len(corners) == 4:
                results['method_used'] = 'enhanced'
                results['corners'] = corners
                if visualize:
                    results['visualizations']['enhanced'] = viz
                    # Create corner visualization on original image
                    detection_viz = visualize_corners(image.copy(), corners,
                                                      "Enhanced Detection: Found 4 Corners")
                logger.info(f"Enhanced detection found {len(corners)} corners")
            else:
                logger.warning(
                    f"Enhanced detection found {len(corners) if corners else 0} corners")
        except Exception as e:
            logger.error(f"Enhanced detection failed: {e}")
            corners = None

        # If enhanced detection failed, try basic detection
        if not corners or len(corners) != 4:
            logger.info("Falling back to basic corner detection")
            try:
                corners, viz = self.basic_detector.detect(
                    image, visualize_steps=visualize)

                if corners and len(corners) == 4:
                    results['method_used'] = 'basic'
                    results['corners'] = corners
                    if visualize:
                        results['visualizations']['basic'] = viz
                        # Create corner visualization
                        detection_viz = visualize_corners(image.copy(), corners,
                                                          "Basic Detection: Found 4 Corners")
                    logger.info(
                        f"Basic detection found {len(corners)} corners")
                else:
                    logger.warning(
                        f"Basic detection found {len(corners) if corners else 0} corners")
            except Exception as e:
                logger.error(f"Basic detection failed: {e}")
                corners = None

        # If both detection methods failed, return None
        if not corners or len(corners) != 4:
            logger.error("Failed to detect corners with any method")
            if visualize:
                # Show failure visualization
                no_corners_viz = image.copy()
                cv2.putText(no_corners_viz, "Corner Detection Failed",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                results['visualizations']['corner_detection'] = no_corners_viz
            return None, results

        # Store corner detection visualization
        if visualize and detection_viz is not None:
            results['visualizations']['corner_detection'] = detection_viz

        # Calculate angle
        try:
            angle = self.rectifier.calculate_angle(corners)
            results['angle'] = angle
            logger.info(f"Calculated angle: {angle:.2f}°")

            if visualize:
                # Create angle visualization
                angle_viz = detection_viz.copy() if detection_viz is not None else image.copy()
                cv2.putText(angle_viz, f"Angle: {angle:.2f} degrees",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                results['visualizations']['angle_detection'] = angle_viz

        except Exception as e:
            logger.error(f"Angle calculation failed: {e}")
            angle = 0.0

        # Check if rectification is needed
        threshold = self.params.get('analyzer', {}).get(
            'rectification_threshold', 3.0)

        if abs(angle) > threshold:
            logger.info(
                f"Angle {angle:.2f}° exceeds threshold {threshold}°, rectifying")
            try:
                rectified, transform = self.rectifier.rectify(image, corners)

                if rectified is not None:
                    results['rectified'] = rectified
                    results['transform'] = transform
                    logger.info("Rectification successful")

                    if visualize:
                        # Create rectification visualization
                        # Detect corners in rectified image to show result
                        rect_corners, _ = self.corner_detector.detect(
                            rectified, False)
                        if rect_corners and len(rect_corners) == 4:
                            rect_viz = visualize_corners(rectified.copy(), rect_corners,
                                                         "After Rectification")
                        else:
                            rect_viz = rectified.copy()
                            cv2.putText(rect_viz, "Rectified Image",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        results['visualizations']['rectified'] = rect_viz
                else:
                    logger.error("Rectification returned None")
                    results['rectified'] = image
            except Exception as e:
                logger.error(f"Rectification failed: {e}")
                results['rectified'] = image
        else:
            logger.info(
                f"Angle {angle:.2f}° within threshold {threshold}°, no rectification needed")
            results['rectified'] = image

            if visualize:
                # Show that no rectification was needed
                no_rect_viz = detection_viz.copy() if detection_viz is not None else image.copy()
                cv2.putText(no_rect_viz, "No Rectification Needed",
                            (10, image.shape[0] -
                             20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                results['visualizations']['no_rectification'] = no_rect_viz

        return results['rectified'], results
