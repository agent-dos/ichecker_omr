# app/features/analyzer/bubble_analyzer.py (refactored)
"""
Analyzes bubble fill scores based on detected bubbles and image properties.
"""
import cv2
import numpy as np
from typing import List, Dict, Any  # Added Any
import logging  # Added logging

# Import processors
from app.features.analyzer.processors.bubble_processor import BubbleProcessor
from app.features.analyzer.processors.score_calculator import ScoreCalculator
# Import config helper and constants map if needed for defaults (currently defaults are literals)
from app.core.config import get_cv2_flag, CV2_ADAPTIVE_METHODS

logger = logging.getLogger(__name__)


class BubbleAnalyzer:
    """
    Analyzes detected bubbles to determine fill scores and identify selected choices.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the BubbleAnalyzer.

        Args:
            params (Dict[str, Any]): Configuration dictionary specifically for
                bubble analysis (e.g., from config['bubble_analysis']). Expected
                keys include thresholding, grouping, and scoring parameters.
        """
        self.params = params
        logger.debug(f"Initializing BubbleAnalyzer with params: {params}")
        # Initialize processors with the same parameters (or specific subsets if needed)
        self.bubble_processor = BubbleProcessor(params.get(
            'grouping', params))  # Pass grouping params or all
        self.score_calculator = ScoreCalculator(params.get(
            'scoring', params))  # Pass scoring params or all

        # Store thresholding parameters directly for _create_threshold_image
        self.thresh_adaptive_method_key = self.params.get(
            'adaptive_method', 'ADAPTIVE_THRESH_MEAN_C')
        self.thresh_blocksize = self.params.get('adaptive_blocksize', 31)
        self.thresh_c = self.params.get('adaptive_c', 10)

    def analyze(
        self,
        image: np.ndarray,
        bubbles: np.ndarray
        # visualize_steps: bool = False # Removed - visualization handled externally now
    ) -> List[Dict]:  # Return type is List[Dict] - list of score dictionaries per question
        """
        Analyze bubble fill scores for detected bubbles.

        Args:
            image (np.ndarray): The input image (preferably the original or
                                rectified image, BGR format expected) used for
                                calculating pixel intensity scores.
            bubbles (np.ndarray): A NumPy array of detected bubbles, where each
                                   row is (x_center, y_center, radius).

        Returns:
            List[Dict]: A list where each dictionary represents a question row
                        and contains 'question_number' and 'choices'. 'choices'
                        is a list of dictionaries for each bubble in the row,
                        detailing 'label', 'x', 'y', 'r', 'score', and 'selected'.
        """
        # --- Input Validation ---
        if bubbles is None or bubbles.size == 0:
            logger.warning("BubbleAnalyzer received no bubbles to analyze.")
            return []
        if image is None:
            logger.error("BubbleAnalyzer received None image for analysis.")
            return []

        # --- Step 1: Create Threshold Image for Scoring ---
        try:
            # Thresholding uses parameters from self.params
            thresh = self._create_threshold_image(image)
            # Optional: Add thresh to viz_steps if visualization were handled here
            # if visualize_steps: viz_steps["01_AnalysisThreshold"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(
                f"Failed to create threshold image in BubbleAnalyzer: {e}", exc_info=True)
            return []  # Cannot proceed without threshold image

        # --- Step 2: Group Bubbles into Question Rows ---
        try:
            # Grouping uses parameters (e.g., row_threshold) from self.params via BubbleProcessor init
            grouped_rows = self.bubble_processor.group_bubbles(
                bubbles, image.shape[1]  # Pass image width for grouping logic
            )
            logger.info(
                f"Grouped {len(bubbles)} bubbles into {len(grouped_rows)} potential question rows.")
            # Optional: Add grouping visualization here if needed
            # if visualize_steps: ...
        except Exception as e:
            logger.error(
                f"Failed to group bubbles in BubbleAnalyzer: {e}", exc_info=True)
            return []  # Cannot proceed without grouping

        # --- Step 3: Calculate Scores for Each Row ---
        all_scores = []
        for q_num, row_bubbles in grouped_rows:
            if not row_bubbles:  # Skip empty rows if they somehow occur
                continue
            try:
                # Scoring uses parameters (e.g., bubble_threshold) from self.params via ScoreCalculator init
                row_score_details = self.score_calculator.analyze_row(
                    thresh, row_bubbles, q_num
                )
                all_scores.append(row_score_details)
            except Exception as e:
                logger.error(
                    f"Failed to analyze score for row {q_num}: {e}", exc_info=True)
                # Optionally append a failure marker or skip the row
                all_scores.append(
                    {'question_number': q_num, 'choices': [], 'error': str(e)})

        # --- Step 4: Final Visualization (Handled Externally) ---
        # Visualization of scores (e.g., using visualize_scores) is now done
        # in the calling function (AnalyzerService) based on visualize_intermediate flag.
        # if visualize_steps:
        #     final_viz = visualize_scores(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape)==2 else image.copy(), all_scores)
        #     viz_steps["99_FinalScores"] = final_viz

        logger.info(f"Bubble analysis completed for {len(all_scores)} rows.")
        # Return only the scores list, visualization dict is removed
        return all_scores

    def _create_threshold_image(self, image: np.ndarray) -> np.ndarray:
        """
        Creates an adaptive threshold binary image suitable for scoring bubbles.

        Args:
            image (np.ndarray): The input image (BGR).

        Returns:
            np.ndarray: A binary (0 or 255) thresholded image (grayscale).
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image  # Already grayscale
        else:
            raise ValueError(
                f"Unsupported image shape for thresholding: {image.shape}")

        # Get adaptive method flag
        adaptive_method_flag = get_cv2_flag(
            self.thresh_adaptive_method_key,
            CV2_ADAPTIVE_METHODS,
            cv2.ADAPTIVE_THRESH_MEAN_C  # Default fallback
        )

        # Ensure block size is odd and positive
        block_size = self.thresh_blocksize
        if not isinstance(block_size, int) or block_size <= 0:
            logger.warning(
                f"Invalid block_size {block_size}, using default 31.")
            block_size = 31
        if block_size % 2 == 0:
            block_size += 1
            logger.debug(f"Adjusted block_size to odd value: {block_size}")

        # Constant C
        c_value = self.thresh_c
        if not isinstance(c_value, (int, float)):
            logger.warning(f"Invalid C value {c_value}, using default 10.")
            c_value = 10

        logger.debug(
            f"Applying adaptiveThreshold: Method={adaptive_method_flag}, BlockSize={block_size}, C={c_value}")

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,  # Max value
            adaptive_method_flag,
            # Invert threshold (dark marks become white)
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value
        )

        return thresh
