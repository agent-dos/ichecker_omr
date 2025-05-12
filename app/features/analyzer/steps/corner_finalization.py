# app/features/analyzer/steps/corner_finalization.py
"""Corner finalization step."""
import logging
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.corners.visualizer import visualize_corners

logger = logging.getLogger(__name__)


class CornerFinalizationStep(AnalysisStep):
    """Handles corner finalization with transform and QR filtering."""

    def process(self, context: Dict) -> Dict:
        """Process corner finalization."""
        logger.info("=== STEP 3: Corner Finalization ===")

        processing_img = context['processing_img']
        visual_chain = context['visual_chain']
        corners = context.get('corners')
        transform = context.get('transform')
        qr_polygon = context.get('qr_polygon')

        # Finalize corners
        finalized_corners = self._finalize_corners(
            corners, transform, qr_polygon)

        # Create visualization
        output_viz, description, success = self._create_visualization(
            processing_img, visual_chain, finalized_corners
        )

        # Create step info
        step_info = self.create_step_info(
            description=description,
            success=success,
            input_image=visual_chain,
            output_image=output_viz
        )

        return {
            'step_info': step_info,
            'context_update': {
                'visual_chain': output_viz,
                'corners_for_bubbles': finalized_corners
            }
        }

    def _finalize_corners(self, corners: Optional[Dict],
                          transform: Optional[np.ndarray],
                          qr_polygon: Optional[List]) -> Optional[Dict]:
        """Finalize corners with transform and filtering."""
        if not corners or len(corners) != 4:
            return None

        finalized = corners.copy()

        # Apply transform if available
        if transform is not None:
            finalized = self._transform_corners(corners, transform)

        # Apply QR filtering if enabled
        if qr_polygon and self._should_filter_qr():
            finalized = self._filter_by_qr(finalized, qr_polygon)

        return finalized

    def _should_filter_qr(self) -> bool:
        """Check if QR filtering is enabled."""
        return self.params.get('corner_detection', {}).get(
            'validator', {}
        ).get('qr_filter_enabled', True)

    def _transform_corners(self, corners: Dict,
                           transform: np.ndarray) -> Optional[Dict]:
        """Transform corners to rectified space."""
        try:
            pts = []
            for key in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                pts.append(corners[key]['center'])

            src_pts = np.array([pts], dtype=np.float32)
            dst_pts = cv2.perspectiveTransform(src_pts, transform)

            transformed = {}
            for i, key in enumerate(['top_left', 'top_right', 'bottom_right', 'bottom_left']):
                transformed[key] = {'center': tuple(map(float, dst_pts[0, i]))}

            return transformed
        except Exception as e:
            logger.error(f"Failed to transform corners: {e}")
            return corners

    def _filter_by_qr(self, corners: Dict, qr_polygon: List) -> Dict:
        """Filter corners that may overlap with QR code."""
        # Implementation would filter corners inside QR polygon
        return corners

    def _create_visualization(self, processing_img: np.ndarray,
                              visual_chain: np.ndarray,
                              corners: Optional[Dict]) -> Tuple[np.ndarray, str, bool]:
        """Create visualization and determine status."""
        if corners and all(corners.values()):
            viz = visualize_corners(
                processing_img.copy(), corners,
                "Finalized Corners for Bubbles"
            )
            return viz, "Corners finalized successfully", True
        else:
            viz = visual_chain.copy()
            cv2.putText(viz, "Corner finalization failed", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return viz, "Failed to finalize corners", False
