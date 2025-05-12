# app/features/analyzer/steps/rectification.py
"""Rectification and initial corner detection step."""
import logging
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.rectification.pipeline import RectificationPipeline
from app.features.corners.visualizer import visualize_corners

logger = logging.getLogger(__name__)


class RectificationStep(AnalysisStep):
    """Handles rectification and initial corner detection."""

    def __init__(self, params: Dict):
        """Initialize with rectification parameters."""
        super().__init__(params)
        self.pipeline = RectificationPipeline(params)

    def process(self, context: Dict) -> Dict:
        """Process rectification and corner detection."""
        logger.info("=== STEP 1: Rectification & Initial Corner Detection ===")

        processing_img = context['processing_img']
        visual_chain = context['visual_chain']
        visualize = context['visualize']

        # Perform rectification
        rectified_img, results = self.pipeline.process(
            processing_img, visualize)

        # Extract results
        corners = results.get('corners')
        transform = results.get('transform')
        viz_steps = results.get('visualizations', {})

        # Update processing image if rectified
        if rectified_img is not None:
            processing_img = rectified_img
            success_msg = "Rectification applied successfully"
        else:
            rectified_img = processing_img
            success_msg = "No rectification needed/possible"

        # Create corner visualization for visual chain
        output_viz = self._create_visualization(processing_img, corners)

        # Create step info
        step_info = self.create_step_info(
            description=f'{success_msg}. Corners: {len(corners) if corners else 0}',
            success=True,
            input_image=visual_chain,
            output_image=output_viz,
            viz_steps=viz_steps if visualize else None
        )

        return {
            'step_info': step_info,
            'transform_matrix': transform,
            'context_update': {
                'processing_img': processing_img,
                'visual_chain': output_viz,
                'corners': corners,
                'transform': transform
            }
        }

    def _create_visualization(self, image: np.ndarray,
                              corners: Optional[Dict]) -> np.ndarray:
        """Create visualization showing detected corners."""
        if corners and len(corners) == 4:
            return visualize_corners(
                image.copy(), corners,
                f"Detected {len(corners)} Corners"
            )
        else:
            viz = image.copy()
            cv2.putText(viz, "No corners detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return viz
