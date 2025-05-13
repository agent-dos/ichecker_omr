# app/features/analyzer/steps/answer_extraction.py
"""Answer extraction step."""
import logging
from typing import Dict, List, Tuple

import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.analyzer.answer_extractor import AnswerExtractor
from app.features.analyzer.visualizers.answer_viz import visualize_answers

logger = logging.getLogger(__name__)


class AnswerExtractionStep(AnalysisStep):
    """Handles final answer extraction from analyzed bubbles."""

    def __init__(self, params: Dict = None):
        """Initialize with answer extraction parameters."""
        super().__init__(params or {})
        self.extractor = AnswerExtractor()

    def process(self, context: Dict) -> Dict:
        """Process answer extraction."""
        logger.info("=== STEP 6: Answer Extraction ===")
        self.step_name = "6. Answer Extraction"

        visual_chain = context['visual_chain']
        bubble_scores = context.get('bubble_scores', [])

        # Extract answers
        answers = self.extractor.extract(bubble_scores)

        # Create final visualization
        output_viz = visualize_answers(
            visual_chain.copy(), answers, bubble_scores
        )

        # Create step info
        step_info = self.create_step_info(
            description=f'Extracted {len(answers)} answers',
            success=len(answers) > 0,
            input_image=visual_chain,
            output_image=output_viz
        )

        return {
            'step_info': step_info,
            'final_answers': answers,
            'context_update': {
                'visual_chain': output_viz
            }
        }
