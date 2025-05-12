# filename: app/features/analyzer/service.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import components
from app.features.analyzer.bubble_detector import BubbleDetector
from app.features.analyzer.bubble_analyzer import BubbleAnalyzer
from app.features.analyzer.answer_extractor import AnswerExtractor
from app.features.corners.detector import CornerDetector
from app.features.qr.detector import QRDetector
from app.features.rectification.rectifier import ImageRectifier

# Visualizers and helpers
from app.features.corners.visualizer import visualize_corners
from app.features.qr.visualizer import visualize_qr  # For QR step output
from app.features.analyzer.visualizers.answer_viz import visualize_scores, visualize_answers
from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles


logger = logging.getLogger(__name__)


class AnalyzerService:
    """Orchestrates answer sheet analysis with granular step visualization."""

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the AnalyzerService with configuration parameters.

        Args:
            params (Dict[str, Any]): A nested dictionary containing configuration
                parameters for all sub-components (detectors, analyzers, etc.).
                Expected keys correspond to configuration sections like
                'bubble_detection', 'corner_detection', 'qr_detection', etc.
        """
        self.params = params
        logger.info("AnalyzerService initializing with detailed configuration.")
        try:
            # Initialize components with their respective parameter sections
            self.bubble_detector = BubbleDetector(
                params.get('bubble_detection', {}))
            self.bubble_analyzer = BubbleAnalyzer(
                params.get('bubble_analysis', {}))
            self.answer_extractor = AnswerExtractor()  # Doesn't require params currently
            self.corner_detector = CornerDetector(
                params.get('corner_detection', {}))
            self.qr_detector = QRDetector(params.get('qr_detection', {}))
            self.rectifier = ImageRectifier(params.get('rectification', {}))
        except KeyError as e:
            logger.error(
                f"Missing configuration section during service init: {e}. Analysis may fail.", exc_info=True)
            raise ValueError(
                f"Configuration error: Missing section {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during service init: {e}", exc_info=True)
            raise

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Performs the full analysis pipeline on the input image.

        Args:
            image (np.ndarray): The input answer sheet image (expected in BGR format).

        Returns:
            Dict: A dictionary containing the analysis results, including:
                'original_image': The input image.
                'steps': A list of dictionaries, each describing a pipeline step
                         (name, description, success, input_image, output_image,
                          intermediate_visualizations).
                'final_answers': A list of tuples (question_number, answer_char).
                'qr_data': The decoded QR code string, or None.
                'transform_matrix': The perspective transform matrix used for
                                     rectification, or None.
        """
        results = {
            'original_image': image.copy(),
            'steps': [],
            'final_answers': [],
            'qr_data': None,
            'transform_matrix': None,
            # Consider adding 'overall_success': bool here later
        }
        # Determine if intermediate visualization steps should be generated and stored
        visualize_intermediate = self.params.get('debug_options', {}).get(
            'visualize_intermediate_steps', False)
        logger.info(
            f"Starting analysis. Visualize intermediate steps: {visualize_intermediate}")

        # --- Pipeline Execution ---
        processing_image = image.copy()  # Image used for actual processing steps
        # Image to log as input for the current step
        current_step_input_image = image.copy()

        corners_initial_raw = None  # Corners detected before rectification
        # Corners potentially transformed, used for bubble detection
        corners_for_bubbles = None

        # --- 1. Rectification & Initial Corner Detection ---
        logger.info("=== STEP 1: Rectification & Initial Corner Detection ===")
        rect_viz_steps = {}  # Stores visualizations from this phase

        # _rectify_if_needed performs initial corner detection and optional rectification
        rectified_img, transform, corners_initial_raw, initial_cd_viz = self._rectify_if_needed(
            processing_image, visualize_intermediate
        )
        rect_viz_steps.update(initial_cd_viz)  # Merge corner detection visuals

        step_1_output_img = processing_image.copy()  # Default output is the input
        rect_success = False  # Flag indicating if rectification was applied

        if rectified_img is not None:
            logger.info("Rectification applied. Updating processing image.")
            processing_image = rectified_img  # Update the image being processed
            step_1_output_img = rectified_img.copy()  # Set step output
            results['transform_matrix'] = transform  # Store the matrix
            rect_success = True
            if visualize_intermediate:
                rect_viz_steps["99_RectifiedOutput"] = rectified_img.copy()
        # Rectification not applied/failed, but initial corners found
        elif corners_initial_raw and len(corners_initial_raw) == 4:
            logger.info(
                "Rectification not applied/failed, but initial corners were detected.")
            # rect_success remains False
            # Ensure final detection viz exists if needed
            if visualize_intermediate and "InitialCD_99_FinalDetection" not in rect_viz_steps:
                rect_viz_steps["InitialCD_99_FinalDetection"] = visualize_corners(
                    image.copy(), corners_initial_raw, "Initial Corners (No Rectification)")
                # Use visualization as output
                step_1_output_img = rect_viz_steps["InitialCD_99_FinalDetection"]
        else:  # No rectification AND no initial corners found
            logger.warning(
                "Rectification not applied AND initial corner detection failed.")
            rect_success = False  # Explicitly false

        results['steps'].append({
            'name': '1. Rectification & Initial Corners',
            'description': f'Rectification attempt. Rectified: {rect_success}. Angle threshold: {self.params.get("analyzer", {}).get("rectification_threshold", 5.0):.1f}°. Initial Corners Found: {bool(corners_initial_raw and len(corners_initial_raw) == 4)}.',
            'success': True,  # Step always runs; sub-ops determine effective success
            'input_image': image.copy(),  # Original image input
            'output_image': step_1_output_img,  # Image after potential rectification
            'intermediate_visualizations': rect_viz_steps if visualize_intermediate else {}
        })
        # Input for the next logical step
        current_step_input_image = processing_image.copy()

        # --- 2. QR Code Detection ---
        logger.info("=== STEP 2: QR Code Detection ===")
        qr_viz_steps = {}
        qr_data, qr_info, qr_viz_steps_from_detector = self.qr_detector.detect(
            processing_image, visualize_intermediate  # Use potentially rectified image
        )
        results['qr_data'] = qr_data
        # Needed for filtering later
        qr_polygon = qr_info.get('polygon') if qr_info else None
        qr_success = qr_data is not None
        if visualize_intermediate:
            qr_viz_steps.update(qr_viz_steps_from_detector)

        # Determine output image for QR step visualization
        qr_step_output_img_key = "99_FinalDetection"
        qr_step_output_img = qr_viz_steps.get(qr_step_output_img_key)
        if qr_step_output_img is None:  # Generate fallback visualization if not provided by detector
            qr_step_output_img = visualize_qr(
                processing_image.copy(), qr_data, qr_info)

        results['steps'].append({
            'name': '2. QR Code Detection',
            'description': f'Detected QR Data: {str(qr_data) if qr_data else "None"}.',
            'success': qr_success,
            'input_image': current_step_input_image.copy(),  # Image before QR detection
            'output_image': qr_step_output_img,  # Visualization of QR detection result
            'intermediate_visualizations': qr_viz_steps if visualize_intermediate else {}
        })
        # current_step_input_image = processing_image.copy() # Input for next step is still processing_image

        # --- 3. Corner Finalization (Transform & QR Filter) ---
        logger.info("=== STEP 3: Corner Finalization ===")
        corner_finalize_viz_steps = {}
        corners_finalized = None  # Corners ready for bubble detection boundary
        corner_finalize_description = "Processing initial corners for final use."
        corner_finalize_success = False

        if corners_initial_raw and len(corners_initial_raw) == 4:
            corners_in_current_space = corners_initial_raw  # Start with raw corners
            visual_base_img = image.copy()  # Image space for initial corners

            if transform is not None:  # If rectification happened, transform corners
                logger.info(
                    "Transforming initial corners to rectified space...")
                visual_base_img = processing_image.copy()  # Image space is now rectified
                if visualize_intermediate:
                    corner_finalize_viz_steps["00_InitialCorners_ForTransform"] = visualize_corners(
                        image.copy(), corners_initial_raw, "Raw Corners Before Transform")
                try:
                    src_pts_list = [
                        corners_initial_raw['top_left']['center'], corners_initial_raw['top_right']['center'],
                        corners_initial_raw['bottom_right']['center'], corners_initial_raw['bottom_left']['center']
                    ]
                    src_pts_np = np.array([src_pts_list], dtype=np.float32)
                    transformed_pts_np = cv2.perspectiveTransform(
                        src_pts_np, transform)

                    if transformed_pts_np is not None:
                        corners_in_current_space = {  # Update corners to transformed coordinates
                            'top_left': {'center': tuple(map(float, transformed_pts_np[0, 0]))},
                            'top_right': {'center': tuple(map(float, transformed_pts_np[0, 1]))},
                            'bottom_right': {'center': tuple(map(float, transformed_pts_np[0, 2]))},
                            'bottom_left': {'center': tuple(map(float, transformed_pts_np[0, 3]))}
                        }
                        corner_finalize_description = "Initial corners transformed to rectified space. "
                        if visualize_intermediate:
                            viz_transformed = visualize_corners(
                                processing_image.copy(), corners_in_current_space, "Transformed Corners")
                            corner_finalize_viz_steps["01_TransformedInitialCorners"] = viz_transformed
                    else:
                        raise ValueError(
                            "cv2.perspectiveTransform returned None")
                except Exception as e:
                    logger.error(
                        f"Error transforming initial corners: {e}", exc_info=True)
                    corners_in_current_space = None  # Mark transform as failed
                    corner_finalize_description = "Failed to transform initial corners. "
            else:  # No transform needed, corners_in_current_space remains corners_initial_raw
                corner_finalize_description = "No rectification transform applied. "
                if visualize_intermediate:
                    corner_finalize_viz_steps["00_InitialCorners_NoTransform"] = visualize_corners(
                        image.copy(), corners_initial_raw, "Initial Corners (No Transform Needed)")

            # Apply QR filtering if needed and possible
            if corners_in_current_space and len(corners_in_current_space) == 4:
                qr_filter_enabled = self.params.get('corner_detection', {}).get(
                    'validator', {}).get('qr_filter_enabled', True)
                if qr_polygon and qr_filter_enabled:
                    logger.info("Filtering finalized corners by QR polygon...")
                    temp_filtered_corners = {}
                    valid_after_qr_count = 0
                    for name, corner_data in corners_in_current_space.items():
                        if corner_data and 'center' in corner_data:
                            center_pt = tuple(
                                map(float, corner_data['center']))
                            # Keep if point is OUTSIDE or ON edge of QR polygon
                            if cv2.pointPolygonTest(np.array(qr_polygon, dtype=np.int32), center_pt, False) < 0:
                                temp_filtered_corners[name] = corner_data
                                valid_after_qr_count += 1
                            else:
                                logger.warning(
                                    f"Corner '{name}' at {center_pt} excluded by QR filter.")
                                # Mark as excluded
                                temp_filtered_corners[name] = None
                        else:  # Should not happen if input was 4 valid corners
                            logger.error(
                                f"Invalid corner data encountered before QR filtering for {name}")
                            valid_after_qr_count = -1  # Indicate error
                            break

                    if valid_after_qr_count == 4:
                        corners_finalized = temp_filtered_corners
                        corner_finalize_description += "Successfully filtered by QR polygon."
                        corner_finalize_success = True
                    else:
                        logger.warning(
                            f"QR filtering resulted in {valid_after_qr_count}/4 valid corners. Falling back.")
                        corners_finalized = corners_in_current_space  # Use corners before QR filter
                        corner_finalize_description += f"QR filtering removed corners. Using pre-QR corners."
                        # Success depends on whether pre-QR corners were valid
                        corner_finalize_success = corners_finalized is not None and all(
                            corners_finalized.values())

                    if visualize_intermediate:
                        viz_qr_filtered = visualize_corners(
                            visual_base_img.copy(), corners_finalized, "Corners After QR Filter")
                        cv2.polylines(viz_qr_filtered, [np.array(
                            # Draw QR poly
                            qr_polygon, dtype=np.int32)], True, (0, 0, 255), 1)
                        corner_finalize_viz_steps["02_CornersAfterQRFilter"] = viz_qr_filtered

                else:  # No QR polygon or QR filter disabled
                    corners_finalized = corners_in_current_space
                    corner_finalize_description += "No QR filtering applied."
                    corner_finalize_success = corners_finalized is not None and all(
                        corners_finalized.values())
            else:  # corners_in_current_space is None or not 4
                corner_finalize_description += "Not enough valid corners after transform."
                corner_finalize_success = False
        else:  # initial corners not found or incomplete
            corner_finalize_description = "Initial corner detection did not yield 4 corners."
            corner_finalize_success = False

        corners_for_bubbles = corners_finalized  # Set the final corners for next step

        # Output image for this logical step
        # Default to base image for this step
        corner_finalize_output_img = visual_base_img.copy()
        if corners_finalized:  # If we have corners, visualize them as output
            corner_finalize_output_img = visualize_corners(
                visual_base_img.copy(), corners_finalized, "Finalized Corners for Bubbles")
            if visualize_intermediate:  # Prefer specific viz step if available
                # Try to get the last relevant viz step
                last_viz_key = next((k for k in reversed(
                    corner_finalize_viz_steps) if k.startswith("0")), None)
                if last_viz_key:
                    corner_finalize_output_img = corner_finalize_viz_steps[last_viz_key]

        results['steps'].append({
            'name': '3. Corner Finalization',
            'description': corner_finalize_description,
            'success': corner_finalize_success,
            'input_image': current_step_input_image.copy(),  # Image before this logic
            # Visualization of the finalized corners
            'output_image': corner_finalize_output_img,
            'intermediate_visualizations': corner_finalize_viz_steps if visualize_intermediate else {}
        })
        # current_step_input_image = processing_image.copy() # Input for next step

        # --- 4. Bubble Detection ---
        logger.info("=== STEP 4: Bubble Detection ===")
        bubble_viz_steps = {}
        bubbles = None
        bubble_detector_input_img = processing_image.copy()  # Input for this step

        if corners_for_bubbles is None:  # Check dependency
            logger.warning(
                "Skipping bubble detection: No valid corners available.")
            bubble_success = False
            bubble_description = "Skipped: No valid corners from previous steps."
            bubble_output_img = bubble_detector_input_img  # Output is same as input
        else:
            logger.info(f"Starting bubble detection using finalized corners.")
            # BubbleDetector uses corners for boundary filtering if enabled
            bubbles, bubble_viz_steps_from_detector = self.bubble_detector.detect(
                processing_image, qr_polygon, corners_for_bubbles, visualize_intermediate
            )
            bubble_success = bubbles is not None and bubbles.size > 0
            bubble_description = f'Detected {len(bubbles) if bubbles is not None else 0} bubble candidates.'
            if visualize_intermediate:
                bubble_viz_steps.update(bubble_viz_steps_from_detector)

            # Get or create the output visualization for this step
            bubble_output_img_key = "99_FinalDetection"
            bubble_output_img = bubble_viz_steps.get(bubble_output_img_key)
            if bubble_output_img is None:
                bubble_output_img = visualize_bubbles(
                    processing_image.copy(), bubbles)

        results['steps'].append({
            'name': '4. Bubble Detection',
            'description': bubble_description,
            'success': bubble_success,
            'input_image': bubble_detector_input_img,  # Image fed into bubble detector
            'output_image': bubble_output_img,  # Visualization of detected bubbles
            'intermediate_visualizations': bubble_viz_steps if visualize_intermediate else {}
        })

        if not bubble_success:
            logger.warning(
                "Bubble detection failed or found no bubbles. Subsequent steps might be affected.")
            # Option to return early: return results

        # --- 5. Bubble Analysis ---
        logger.info("=== STEP 5: Bubble Analysis ===")
        analysis_viz_steps = {}
        bubble_scores = []
        # Input for this step's log is the output of the previous step
        analysis_input_img = bubble_output_img.copy(
        ) if bubble_success else bubble_detector_input_img.copy()

        if bubbles is None or bubbles.size == 0:
            logger.warning("Skipping bubble analysis: No bubbles detected.")
            analysis_success = False
            analysis_description = "Skipped: No bubbles detected."
            analysis_output_img = analysis_input_img  # Output is same as input
        else:
            # BubbleAnalyzer calculates scores based on detected bubbles and the image
            bubble_scores = self.bubble_analyzer.analyze(
                processing_image, bubbles  # Use processing_image for pixel analysis
            )
            analysis_success = len(bubble_scores) > 0
            analysis_description = f'Analyzed scores for {len(bubble_scores)} question rows.'

            # Generate visualization of the scores
            analysis_output_img = visualize_scores(
                # Base image for viz is processing_image
                processing_image.copy(), bubble_scores
            )
            if visualize_intermediate:
                # BubbleAnalyzer itself doesn't currently produce intermediate viz,
                # but we store the final score viz if intermediates are enabled.
                analysis_viz_steps["99_FinalScores"] = analysis_output_img

        results['steps'].append({
            'name': '5. Bubble Analysis',
            'description': analysis_description,
            'success': analysis_success,
            # Visual input reference (bubble detections)
            'input_image': analysis_input_img,
            'output_image': analysis_output_img,  # Visualization of scores
            'intermediate_visualizations': analysis_viz_steps if visualize_intermediate else {}
        })

        # --- 6. Answer Extraction ---
        logger.info("=== STEP 6: Answer Extraction ===")
        # Input for this step's log is the output of the previous step
        answer_extraction_input_img = analysis_output_img.copy()

        # Extract final answers based on the calculated scores
        answers = self.answer_extractor.extract(bubble_scores)
        results['final_answers'] = answers
        extraction_success = len(answers) > 0
        extraction_description = f'Extracted {len(answers)} answers.'

        # Generate final visualization showing answers overlaid
        final_answers_viz = visualize_answers(
            # Base image is the score visualization
            answer_extraction_input_img, answers, bubble_scores
        )

        results['steps'].append({
            'name': '6. Answer Extraction',
            'description': extraction_description,
            'success': extraction_success,
            'input_image': answer_extraction_input_img,  # Score visualization input
            'output_image': final_answers_viz,  # Final answer visualization
            'intermediate_visualizations': {}  # No intermediate steps typically
        })

        logger.info("Analysis finished.")
        return results

    def _rectify_if_needed(
        self,
        image: np.ndarray,
        visualize_steps: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Dict]:
        """
        Performs initial corner detection and optional rectification based on angle.

        This function attempts to detect the four corner markers on the input image.
        If successful and the calculated angle of tilt exceeds the configured
        threshold (and rectification is enabled), it applies perspective warping
        to straighten the image.

        Args:
            image (np.ndarray): The input image (BGR) to process.
            visualize_steps (bool): If True, generate and return intermediate
                visualization steps.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Dict]: A tuple containing:
                - Rectified Image (np.ndarray): The perspective-warped image if
                  rectification was performed, otherwise None.
                - Transform Matrix (np.ndarray): The 3x3 perspective transform matrix
                  used for rectification, otherwise None.
                - Detected Corners (Optional[Dict]): A dictionary containing the initially
                  detected corner details (before potential transform), or None if
                  detection failed. Format: {'corner_name': {'center': (x,y), ...}}
                - Visualization Steps (Dict): A dictionary containing intermediate
                  visualization images if visualize_steps was True. Keys indicate
                  the step (e.g., 'InitialCD_01_GrayscaleInput', 'Rectification_99_RectifiedOutput').
        """
        viz_steps_combined = {}  # All visualizations from this phase
        corners_detected_dict: Optional[Dict] = None
        rectified_image: Optional[np.ndarray] = None
        transform_matrix: Optional[np.ndarray] = None

        logger.info(
            "--- Sub-Phase: Initial Corner Detection (for Rectification) ---")
        try:
            # Detect corners using the configured strategies and parameters
            # We don't provide a QR polygon here as QR hasn't been detected yet.
            corners_detected_dict, cd_internal_viz = self.corner_detector.detect(
                image, qr_polygon=None, visualize_steps=visualize_steps
            )

            # Merge visualization steps from corner detector, prefixing keys
            if visualize_steps and isinstance(cd_internal_viz, dict):
                for k, v_img in cd_internal_viz.items():
                    viz_steps_combined[f"InitialCD_{k}"] = v_img

            # Check if corner detection was successful
            if not (corners_detected_dict and len(corners_detected_dict) == 4 and all(corners_detected_dict.values())):
                logger.warning(
                    f"Initial corner detection failed or found incomplete corners. Found: {corners_detected_dict}")
                # Cannot rectify, return detected corners (even if incomplete) and any visualizations
                return None, None, corners_detected_dict, viz_steps_combined

            logger.info(
                f"Initial corners successfully detected: {list(corners_detected_dict.keys())}")

            # Calculate angle based on detected corners
            angle = self.rectifier.calculate_angle(corners_detected_dict)
            logger.info(
                f"Calculated initial angle for rectification: {angle:.2f} degrees.")

            # Check if rectification should be performed
            analyzer_cfg = self.params.get('analyzer', {})
            enable_rect = analyzer_cfg.get('enable_rectification', True)
            rect_thresh = analyzer_cfg.get('rectification_threshold', 5.0)

            if not enable_rect:
                logger.info("Rectification explicitly disabled in config.")
                return None, None, corners_detected_dict, viz_steps_combined
            if abs(angle) < rect_thresh:
                logger.info(
                    f"Rectification skipped: Angle {abs(angle):.2f}° is below threshold {rect_thresh:.2f}°.")
                return None, None, corners_detected_dict, viz_steps_combined

            # Perform rectification
            logger.info("--- Sub-Phase: Performing Image Rectification ---")
            # The rectifier uses parameters passed during its initialization
            rectified_image, transform_matrix = self.rectifier.rectify(
                image, corners_detected_dict
            )

            if rectified_image is not None:
                logger.info("Image rectification successful.")
                if visualize_steps:
                    # Add the final rectified image to visualizations
                    viz_steps_combined["Rectification_99_RectifiedOutput"] = rectified_image.copy(
                    )
            else:
                logger.error(
                    "Image rectification failed: rectifier.rectify returned None.")
                # Return None for rectification results, but keep the initially detected corners
                return None, None, corners_detected_dict, viz_steps_combined

        except Exception as e:
            logger.error(
                f"Exception during initial corner detection or rectification: {e}", exc_info=True)
            # Return None for rectification results, pass through any corners/viz found before the error
            return None, None, corners_detected_dict, viz_steps_combined

        # Return results including the rectified image and transform matrix
        return rectified_image, transform_matrix, corners_detected_dict, viz_steps_combined
