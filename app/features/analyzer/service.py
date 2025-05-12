# app/features/analyzer/service.py
"""Main analyzer service orchestrator."""
import logging
from typing import Dict, Optional

import numpy as np

from app.features.analyzer.steps.rectification import RectificationStep
from app.features.analyzer.steps.qr_detection import QRDetectionStep
from app.features.analyzer.steps.corner_finalization import CornerFinalizationStep
from app.features.analyzer.steps.bubble_detection import BubbleDetectionStep
from app.features.analyzer.steps.bubble_analysis import BubbleAnalysisStep
from app.features.analyzer.steps.answer_extraction import AnswerExtractionStep

logger = logging.getLogger(__name__)


class AnalyzerService:
    """Orchestrates answer sheet analysis pipeline."""

    def __init__(self, params: Dict):
        """Initialize analyzer with configuration parameters."""
        self.params = params
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize all pipeline steps."""
        self.rectification_step = RectificationStep(self.params)
        self.qr_detection_step = QRDetectionStep(self.params)
        self.corner_finalization_step = CornerFinalizationStep(self.params)
        self.bubble_detection_step = BubbleDetectionStep(self.params)
        self.bubble_analysis_step = BubbleAnalysisStep(self.params)
        self.answer_extraction_step = AnswerExtractionStep()

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Performs the full analysis pipeline on the input image.
        Each step's output visualization becomes the next step's input.
        """
        logger.info("Starting answer sheet analysis")

        results = self._initialize_results(image)
        visualize = self._should_visualize()

        # Track images through the pipeline
        processing_img = image.copy()
        visual_chain = image.copy()

        # Execute pipeline steps in sequence
        context = {
            'processing_img': processing_img,
            'visual_chain': visual_chain,
            'visualize': visualize
        }

        # Step 1: Rectification & Corner Detection
        context = self._execute_step(
            self.rectification_step, context, results, 'transform_matrix'
        )

        # Step 2: QR Code Detection
        context = self._execute_step(
            self.qr_detection_step, context, results, 'qr_data'
        )

        # Step 3: Corner Finalization
        context = self._execute_step(
            self.corner_finalization_step, context, results
        )

        # Step 4: Bubble Detection
        context = self._execute_step(
            self.bubble_detection_step, context, results
        )

        # Step 5: Bubble Analysis
        context = self._execute_step(
            self.bubble_analysis_step, context, results
        )

        # Step 6: Answer Extraction
        context = self._execute_step(
            self.answer_extraction_step, context, results, 'final_answers'
        )

        logger.info("Analysis completed")
        return results

    def _initialize_results(self, image: np.ndarray) -> Dict:
        """Initialize the results dictionary."""
        return {
            'original_image': image.copy(),
            'steps': [],
            'final_answers': [],
            'qr_data': None,
            'transform_matrix': None,
        }

    def _should_visualize(self) -> bool:
        """Check if intermediate visualization is enabled."""
        return self.params.get('debug_options', {}).get(
            'visualize_intermediate_steps', False
        )

    def _execute_step(self, step, context: Dict, results: Dict,
                      result_key: Optional[str] = None) -> Dict:
        """Execute a pipeline step and update results."""
        step_result = step.process(context)

        # Add step info to results
        results['steps'].append(step_result['step_info'])

        # Update specific result field if specified
        if result_key and result_key in step_result:
            results[result_key] = step_result[result_key]

        # Update context for next step
        context.update(step_result.get('context_update', {}))

        return context
