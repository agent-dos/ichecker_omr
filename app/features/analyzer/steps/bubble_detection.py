# app/features/analyzer/steps/bubble_detection.py
"""Bubble detection step."""
import logging
from typing import Dict, Optional

import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.analyzer.bubble_detector import BubbleDetector
from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles

logger = logging.getLogger(__name__)


class BubbleDetectionStep(AnalysisStep):
    """Handles bubble detection in answer sheets."""

    def __init__(self, params: Dict):
        """Initialize with bubble detection parameters."""
        super().__init__(params)
        self.detector = BubbleDetector(params.get('bubble_detection', {}))

    def process(self, context: Dict) -> Dict:
        """Process bubble detection."""
        logger.info("=== STEP 4: Bubble Detection ===")

        processing_img = context['processing_img']
        visual_chain = context['visual_chain']
        visualize = context['visualize']
        corners = context.get('corners_for_bubbles')
        qr_polygon = context.get('qr_polygon')

        # Check if we have valid corners
        if not corners:
            no_corners_result = self._handle_no_corners(visual_chain)
            return no_corners_result

        # Detect bubbles
        bubbles, viz_steps = self.detector.detect(
            processing_img, qr_polygon, corners, visualize
        )

        # Create visualization
        output_viz = visualize_bubbles(processing_img.copy(), bubbles)

        # Create step info
        step_info = self.create_step_info(
            description=f'Detected {len(bubbles) if bubbles is not None else 0} bubbles',
            success=bubbles is not None and bubbles.size > 0,
            input_image=visual_chain,
            output_image=output_viz,
            viz_steps=viz_steps if visualize else None
        )

        return {
            'step_info': step_info,
            'context_update': {
                'visual_chain': output_viz,
                'bubbles': bubbles
            }
        }

    def _handle_no_corners(self, visual_chain: np.ndarray) -> Dict:
        """Handle case when no valid corners are available."""
        step_info = self.create_step_info(
            description='Skipped: No valid corners',
            success=False,
            input_image=visual_chain,
            output_image=visual_chain
        )

        return {
            'step_info': step_info,
            'context_update': {
                'visual_chain': visual_chain,
                'bubbles': None
            }
        }
