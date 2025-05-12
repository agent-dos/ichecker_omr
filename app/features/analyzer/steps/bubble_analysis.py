# app/features/analyzer/steps/bubble_analysis.py
"""Bubble analysis step."""
import logging
from typing import Dict, List

import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.analyzer.bubble_analyzer import BubbleAnalyzer
from app.features.analyzer.visualizers.answer_viz import visualize_scores

logger = logging.getLogger(__name__)


class BubbleAnalysisStep(AnalysisStep):
    """Handles bubble analysis to determine filled bubbles."""

    def __init__(self, params: Dict):
        """Initialize with bubble analysis parameters."""
        super().__init__(params)
        self.analyzer = BubbleAnalyzer(params.get('bubble_analysis', {}))

    def process(self, context: Dict) -> Dict:
        """Process bubble analysis."""
        logger.info("=== STEP 5: Bubble Analysis ===")

        processing_img = context['processing_img']
        visual_chain = context['visual_chain']
        bubbles = context.get('bubbles')

        # Check if we have bubbles to analyze
        if bubbles is None or bubbles.size == 0:
            return self._handle_no_bubbles(visual_chain)

        # Analyze bubbles
        scores = self.analyzer.analyze(processing_img, bubbles)

        # Create visualization
        output_viz = visualize_scores(processing_img.copy(), scores)

        # Create step info
        step_info = self.create_step_info(
            description=f'Analyzed {len(scores)} question rows',
            success=len(scores) > 0,
            input_image=visual_chain,
            output_image=output_viz
        )

        return {
            'step_info': step_info,
            'context_update': {
                'visual_chain': output_viz,
                'bubble_scores': scores
            }
        }

    def _handle_no_bubbles(self, visual_chain: np.ndarray) -> Dict:
        """Handle case when no bubbles are available."""
        step_info = self.create_step_info(
            description='Skipped: No bubbles detected',
            success=False,
            input_image=visual_chain,
            output_image=visual_chain
        )

        return {
            'step_info': step_info,
            'context_update': {
                'visual_chain': visual_chain,
                'bubble_scores': []
            }
        }
