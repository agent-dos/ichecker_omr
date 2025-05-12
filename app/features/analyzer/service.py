# filename: app/features/analyzer/service.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any  # Added Any
import logging

# Import components
from app.features.analyzer.bubble_detector import BubbleDetector
from app.features.analyzer.bubble_analyzer import BubbleAnalyzer
from app.features.analyzer.answer_extractor import AnswerExtractor
from app.features.corners.detector import CornerDetector
from app.features.qr.detector import QRDetector
from app.features.rectification.rectifier import ImageRectifier
from app.features.corners.visualizer import visualize_corners

# Config helper
from app.core.config import get_cv2_flag, CV2_INTERPOLATION_FLAGS

logger = logging.getLogger(__name__)


class AnalyzerService:
    """Orchestrates answer sheet analysis."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize service with detailed configuration."""
        self.params = params
        logger.info("AnalyzerService initialized with new configuration.")
        # Initialize components with their specific parameter sub-dicts
        # Add error handling for missing keys if necessary
        try:
            self.bubble_detector = BubbleDetector(
                params.get('bubble_detection', {}))
            self.bubble_analyzer = BubbleAnalyzer(
                params.get('bubble_analysis', {}))
            self.answer_extractor = AnswerExtractor()  # Doesn't seem to have params yet
            self.corner_detector = CornerDetector(
                params.get('corner_detection', {}))
            self.qr_detector = QRDetector(params.get('qr_detection', {}))
            self.rectifier = ImageRectifier(params.get(
                'rectification', {}))  # Pass params here too
        except KeyError as e:
            logger.error(
                f"Missing configuration section: {e}. Using defaults may fail.")
            # Handle appropriately - maybe raise an error or use fallback defaults
            raise ValueError(
                f"Configuration error: Missing section {e}") from e

    def analyze(self, image: np.ndarray) -> Dict:
        """Analyze answer sheet and return results with optional intermediate steps."""
        results = {
            'original_image': image.copy(),  # Store a copy
            'steps': [],
            'final_answers': [],
            'qr_data': None,
            'transform_matrix': None,
            # Add fields for extracted corners, bubbles if needed outside steps
        }
        # Get the master debug flag
        visualize_intermediate = self.params.get('debug_options', {}).get(
            'visualize_intermediate_steps', False)
        logger.info(
            f"Starting analysis. Visualize intermediate steps: {visualize_intermediate}")

        processing_image = image.copy()  # Start with a copy
        transform = None
        initial_corners = None
        corners = None  # Final corners after QR filtering
        qr_polygon = None
        bubbles = None

        # --- Step 1: Rectification Check & Execution ---
        try:
            # Pass the visualize_steps flag if rectifier supports it internally
            rectified, transform, initial_corners, rect_viz = self._rectify_if_needed(
                processing_image, visualize_intermediate
            )
            if rectified is not None:
                processing_image = rectified  # Update image for next steps
                results['transform_matrix'] = transform  # Store matrix
                step_success = True
            else:
                # Rectification failed or not needed, continue with original/current image
                step_success = False  # Mark step as not performed or failed

            results['steps'].append({
                'name': '1. Rectification',
                'description': f'Attempted image rectification. Angle threshold: {self.params.get("analyzer", {}).get("rectification_threshold", 5.0)}°.',
                'success': step_success,
                'input_image': image.copy(),  # Show original input for this step
                # Show image after potential rectification
                'output_image': processing_image.copy(),
                'intermediate_visualizations': rect_viz if visualize_intermediate else {}
            })

        except Exception as e:
            logger.error(
                f"Error during Rectification step: {e}", exc_info=True)
            results['steps'].append({
                'name': '1. Rectification', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })
            # Decide whether to stop or continue
            # return results # Option: Stop on critical failure

        # --- Step 2: QR Detection ---
        try:
            qr_data, qr_info, qr_viz = self.qr_detector.detect(
                processing_image, visualize_intermediate
            )
            results['qr_data'] = qr_data
            results['qr_info'] = qr_info  # Store full info
            qr_polygon = qr_info.get('polygon') if qr_info else None
            step_success = qr_data is not None

            # Use final visualization from qr_viz if available, else generate standard one
            output_viz = qr_viz.get(
                '99_FinalDetection') if visualize_intermediate and qr_viz else None
            if output_viz is None:
                from app.features.qr.visualizer import visualize_qr
                output_viz = visualize_qr(
                    processing_image.copy(), qr_data, qr_info)

            results['steps'].append({
                'name': '2. QR Detection',
                'description': f'Detected QR Data: {qr_data if qr_data else "None"}',
                'success': step_success,
                'input_image': processing_image.copy(),  # Input to this step
                'output_image': output_viz,  # Output viz for this step
                'intermediate_visualizations': qr_viz if visualize_intermediate else {}
            })
        except Exception as e:
            logger.error(f"Error during QR Detection step: {e}", exc_info=True)
            results['steps'].append({
                'name': '2. QR Detection', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': processing_image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })
            # QR polygon might be None, processing can often continue

        # --- Step 3: Corner Detection (Main) ---
        try:
            # Pass qr_polygon and visualize flag
            corners, corner_viz = self.corner_detector.detect(
                processing_image, qr_polygon, visualize_intermediate
            )
            step_success = corners is not None and len(corners) == 4

            # Use final visualization from corner_viz if available
            output_viz = corner_viz.get(
                '99_FinalDetection') if visualize_intermediate and corner_viz else None
            if output_viz is None:
                output_viz = visualize_corners(
                    processing_image.copy(), corners)

            results['steps'].append({
                'name': '3. Corner Detection',
                'description': f'Detected {len(corners) if corners else 0}/4 corners (QR area excluded).',
                'success': step_success,
                'input_image': processing_image.copy(),
                'output_image': output_viz,
                'intermediate_visualizations': corner_viz if visualize_intermediate else {}
            })
            if not step_success:
                logger.warning("Corner detection failed or incomplete.")
                # Decide whether to proceed without corners for bubble filtering
        except Exception as e:
            logger.error(
                f"Error during Corner Detection step: {e}", exc_info=True)
            results['steps'].append({
                'name': '3. Corner Detection', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': processing_image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })
            corners = None  # Ensure corners is None if step failed

        # --- Step 4: Bubble Detection ---
        try:
            # Pass corners (could be None), qr_polygon, and visualize flag
            bubbles, bubble_viz = self.bubble_detector.detect(
                processing_image, qr_polygon, corners, visualize_intermediate
            )
            step_success = bubbles is not None and bubbles.size > 0

            # Use final visualization from bubble_viz if available
            output_viz = bubble_viz.get(
                '99_FinalDetection') if visualize_intermediate and bubble_viz else None
            if output_viz is None:
                from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles
                output_viz = visualize_bubbles(
                    processing_image.copy(), bubbles)

            results['steps'].append({
                'name': '4. Bubble Detection',
                'description': f'Detected {len(bubbles) if bubbles is not None else 0} bubble candidates.',
                'success': step_success,
                'input_image': processing_image.copy(),
                'output_image': output_viz,
                'intermediate_visualizations': bubble_viz if visualize_intermediate else {}
            })
            if not step_success:
                logger.warning("Bubble detection failed or found no bubbles.")
                # Stop processing if no bubbles found
                return results
        except Exception as e:
            logger.error(
                f"Error during Bubble Detection step: {e}", exc_info=True)
            results['steps'].append({
                'name': '4. Bubble Detection', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': processing_image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })
            return results  # Stop if bubble detection fails

        # --- Step 5: Bubble Analysis ---
        try:
            bubble_scores, analysis_viz = self.bubble_analyzer.analyze(
                processing_image, bubbles, visualize_intermediate
            )
            step_success = len(bubble_scores) > 0

            # Use final visualization from analysis_viz if available
            output_viz = analysis_viz.get(
                '99_FinalScores') if visualize_intermediate and analysis_viz else None
            if output_viz is None:
                from app.features.analyzer.visualizers.answer_viz import visualize_scores
                output_viz = visualize_scores(
                    processing_image.copy(), bubble_scores)

            results['steps'].append({
                'name': '5. Bubble Analysis',
                'description': f'Analyzed scores for {len(bubble_scores)} question rows.',
                'success': step_success,
                # Maybe use bubble detection output viz?
                'input_image': processing_image.copy(),
                'output_image': output_viz,
                'intermediate_visualizations': analysis_viz if visualize_intermediate else {}
            })
            if not step_success:
                logger.warning("Bubble analysis resulted in zero scores.")
                # Can still proceed to answer extraction (will yield empty answers)
        except Exception as e:
            logger.error(
                f"Error during Bubble Analysis step: {e}", exc_info=True)
            results['steps'].append({
                'name': '5. Bubble Analysis', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': processing_image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })
            bubble_scores = []  # Ensure scores list is empty

        # --- Step 6: Answer Extraction ---
        try:
            # This step is usually simple and less prone to errors / less visualization needed
            answers = self.answer_extractor.extract(bubble_scores)
            results['final_answers'] = answers  # Store final result separately
            step_success = len(answers) > 0

            # Visualize final answers overlaid on the last step's output
            last_step_output = results['steps'][-1]['output_image'] if results['steps'] else processing_image
            from app.features.analyzer.visualizers.answer_viz import visualize_answers
            final_answer_viz = visualize_answers(
                last_step_output.copy(), answers, bubble_scores)

            results['steps'].append({
                'name': '6. Answer Extraction',
                'description': f'Extracted {len(answers)} answers.',
                'success': step_success,
                'input_image': last_step_output,  # Show input for context
                'output_image': final_answer_viz,  # Show final combined visualization
                # Intermediate visualizations usually not needed here unless debugging AnswerExtractor
                'intermediate_visualizations': {}
            })
        except Exception as e:
            logger.error(
                f"Error during Answer Extraction step: {e}", exc_info=True)
            results['steps'].append({
                'name': '6. Answer Extraction', 'success': False,
                'description': f'Failed with error: {e}',
                'input_image': processing_image.copy(), 'output_image': processing_image.copy(),
                'intermediate_visualizations': {}
            })

        logger.info("Analysis finished.")
        return results

    # --- Internal helper methods ---
    # Need to modify _rectify_if_needed etc. to accept visualize_steps
    # and return intermediate visualizations dict

    def _rectify_if_needed(
        self, image: np.ndarray, visualize_steps: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Dict]:
        """
        Rectify image if tilted. Checks corners, angle, and calls rectifier.
        Returns: rectified_image, transform, corners_used, viz_steps dictionary
        """
        viz_steps = {}
        corners = None  # Corners detected in this initial step
        rectified = None
        transform = None
        logger.info("--- Entering Rectification Check ---")

        try:
            logger.info(
                "Attempting initial corner detection for rectification...")
            # Assuming self.corner_detector is initialized correctly with params
            corners, corner_viz = self.corner_detector.detect(
                image, qr_polygon=None, visualize_steps=visualize_steps
            )

            # Add visualization steps regardless of success for debugging
            if visualize_steps and isinstance(corner_viz, dict):
                viz_steps.update(
                    {f"00_Initial_{k}": v for k, v in corner_viz.items()})
                # Add basic visualization of detected corners (even if incomplete)
                from app.features.corners.visualizer import visualize_corners
                viz_steps["01_InitialCornersDetected"] = visualize_corners(
                    image.copy(), corners)

            # --- Critical Check: Were 4 corners found? ---
            if corners is None or len(corners) != 4:
                logger.warning(
                    f"Initial corner detection for rectification failed or incomplete. Found {len(corners) if corners else 'None'} corners. Cannot proceed with angle check or rectification.")
                # Return None for rectified/transform, but return detected corners/viz
                return None, None, corners, viz_steps

            logger.info(
                f"Initial corners detected ({len(corners)}). Calculating angle...")
            angle = self.rectifier.calculate_angle(corners)
            logger.info(f"Calculated initial angle: {angle:.2f} degrees.")

            # Check parameters for enabling rectification
            analyzer_params = self.params.get('analyzer', {})
            enable_rect = analyzer_params.get('enable_rectification', True)
            rect_thresh = analyzer_params.get('rectification_threshold', 5.0)
            logger.info(
                f"Rectification Params: Enabled={enable_rect}, Threshold={rect_thresh}")

            if not enable_rect or abs(angle) < rect_thresh:
                logger.info(
                    f"Rectification not required (Enabled={enable_rect}, Angle={abs(angle):.2f}°, Threshold={rect_thresh:.2f}°).")
                return None, None, corners, viz_steps  # No rectification needed

            # --- Attempt Rectification ---
            logger.info(
                f"Angle {abs(angle):.2f}° exceeds threshold {rect_thresh:.2f}°. Attempting rectification...")
            # Assuming self.rectifier is initialized correctly with params
            rectified, transform = self.rectifier.rectify(
                image, corners)  # Call rectifier

            if rectified is not None:
                logger.info("Image rectification call successful.")
                if visualize_steps:
                    # Ensure rectified is a valid image before adding
                    if isinstance(rectified, np.ndarray):
                        viz_steps["02_RectifiedImage"] = rectified.copy()
                    else:
                        logger.error("Rectifier returned non-image type.")
            else:
                # This means the self.rectifier.rectify call failed internally
                logger.error(
                    "Image rectification call failed (returned None).")
                # Return None for rectified/transform, but keep initially detected corners
                return None, None, corners, viz_steps

        except Exception as e:
            logger.error(
                f"Exception during rectification check/execution: {e}", exc_info=True)
            # Return None for rectified/transform, keep initially detected corners
            return None, None, corners, viz_steps
        finally:
            logger.info("--- Exiting Rectification Check ---")

        # Return all results
        return rectified, transform, corners, viz_steps

    # Other internal methods (_detect_qr, _detect_corners, etc.) are now called
    # directly in analyze() and don't need to be separate private methods here.
