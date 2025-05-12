# app/features/rectification/pipeline.py
"""
Complete rectification pipeline with fallback strategies.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict

from app.features.rectification.enhanced_detector import EnhancedCornerDetector
from app.features.rectification.enhanced_rectifier import EnhancedRectifier
from app.features.corners.detector import CornerDetector

logger = logging.getLogger(__name__)


class RectificationPipeline:
    """Complete rectification pipeline with multiple strategies."""

    def __init__(self, params: Dict):
        """Initialize rectification pipeline."""
        self.params = params
        self.enhanced_detector = EnhancedCornerDetector(
            params.get('corner_detection', {})
        )
        self.basic_detector = CornerDetector(
            params.get('corner_detection', {})
        )
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

        # Try enhanced detection first
        logger.info("Trying enhanced corner detection")
        corners, viz = self.enhanced_detector.detect(image, visualize)

        if corners and len(corners) == 4:
            results['method_used'] = 'enhanced'
            results['corners'] = corners
            if visualize:
                results['visualizations']['enhanced'] = viz
        else:
            # Fallback to basic detection
            logger.info("Falling back to basic corner detection")
            corners, viz = self.basic_detector.detect(
                image, visualize_steps=visualize)

            if corners and len(corners) == 4:
                results['method_used'] = 'basic'
                results['corners'] = corners
                if visualize:
                    results['visualizations']['basic'] = viz
            else:
                logger.error("Failed to detect corners with any method")
                return None, results

        # Calculate angle
        angle = self.rectifier.calculate_angle(corners)
        results['angle'] = angle

        # Check if rectification is needed
        threshold = self.params.get('analyzer', {}).get(
            'rectification_threshold', 3.0)

        if abs(angle) > threshold:
            logger.info(f"Angle {angle:.2f}° exceeds threshold, rectifying")
            rectified, transform = self.rectifier.rectify(image, corners)

            if rectified is not None:
                results['rectified'] = rectified
                results['transform'] = transform
                logger.info("Rectification successful")
            else:
                logger.error("Rectification failed")
        else:
            logger.info(
                f"Angle {angle:.2f}° within threshold, no rectification needed")
            results['rectified'] = image

        return results['rectified'], results
