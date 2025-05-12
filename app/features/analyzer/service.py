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
# from app.common.geometry.filters import filter_by_polygon # Replaced with manual check

logger = logging.getLogger(__name__)


class AnalyzerService:
    """Orchestrates answer sheet analysis with granular step visualization."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.info("AnalyzerService initializing with detailed configuration.")
        try:
            self.bubble_detector = BubbleDetector(
                params.get('bubble_detection', {}))
            self.bubble_analyzer = BubbleAnalyzer(
                params.get('bubble_analysis', {}))
            self.answer_extractor = AnswerExtractor()
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
        results = {
            'original_image': image.copy(),
            'steps': [],
            'final_answers': [],
            'qr_data': None,
            'transform_matrix': None,
        }
        visualize_intermediate = self.params.get('debug_options', {}).get(
            'visualize_intermediate_steps', False)
        logger.info(
            f"Starting analysis. Visualize intermediate steps: {visualize_intermediate}")

        processing_image = image.copy()
        current_step_input_image = image.copy()

        corners_for_bubbles = None  # This will hold the final corners for bubble stage

        # --- 1. Rectification & Initial Corner Detection ---
        logger.info("=== STEP 1: Rectification & Initial Corner Detection ===")
        rect_viz_steps = {}  # All viz from this phase
        # _rectify_if_needed now encapsulates initial corner detection and actual rectification
        rectified_img, transform, corners_initial_raw, initial_cd_viz = self._rectify_if_needed(
            processing_image, visualize_intermediate
        )
        # Add viz from corner detection within _rectify
        rect_viz_steps.update(initial_cd_viz)

        # Default to current processing image
        step_1_output_img = processing_image.copy()
        rect_success = False  # Assume rectification didn't occur or failed

        if rectified_img is not None:
            logger.info("Rectification applied. Updating processing image.")
            processing_image = rectified_img
            # This is the image after rectification
            step_1_output_img = rectified_img.copy()
            results['transform_matrix'] = transform
            rect_success = True  # Rectification was successful
            if visualize_intermediate:  # Add the final rectified image to its own step's viz if not already there
                rect_viz_steps["99_RectifiedOutput"] = rectified_img.copy()
        elif corners_initial_raw:  # Rectification not needed/failed, but corners were found
            logger.info(
                "Rectification not applied or failed, but initial corners were detected.")
            # rect_success remains False if no rectification happened
            # If initial_cd_viz doesn't have a 'final' view, add one.
            if visualize_intermediate and "InitialCD_99_FinalDetection" not in rect_viz_steps and corners_initial_raw:
                rect_viz_steps["InitialCD_99_FinalDetection"] = visualize_corners(
                    image.copy(), corners_initial_raw)

        else:  # No rectification AND no initial corners
            logger.warning(
                "Rectification not applied AND initial corner detection failed.")
            rect_success = False

        results['steps'].append({
            'name': '1. Rectification & Initial Corners',
            'description': f'Rectification attempt. Rectified: {rect_success}. Angle threshold: {self.params.get("analyzer", {}).get("rectification_threshold", 5.0):.1f}°.',
            'success': True,  # Step always "runs", success depends on sub-ops
            'input_image': image.copy(),  # Original image as input to this phase
            # Image after this phase (rectified or original)
            'output_image': step_1_output_img,
            'intermediate_visualizations': rect_viz_steps if visualize_intermediate else {}
        })
        current_step_input_image = processing_image.copy()  # For the next logical step

        # --- 2. QR Code Detection ---
        logger.info("=== STEP 2: QR Code Detection ===")
        qr_viz_steps = {}
        qr_data, qr_info, qr_viz_steps_from_detector = self.qr_detector.detect(
            processing_image, visualize_intermediate
        )
        results['qr_data'] = qr_data
        qr_polygon = qr_info.get('polygon') if qr_info else None
        qr_success = qr_data is not None
        if visualize_intermediate:
            qr_viz_steps.update(qr_viz_steps_from_detector)

        # Determine output image for QR step
        qr_step_output_img_key = "99_FinalDetection"  # Standard key from detectors
        qr_step_output_img = qr_viz_steps.get(qr_step_output_img_key)
        if qr_step_output_img is None:  # Fallback if key not found
            qr_step_output_img = visualize_qr(
                processing_image.copy(), qr_data, qr_info)

        results['steps'].append({
            'name': '2. QR Code Detection',
            'description': f'Detected QR Data: {str(qr_data) if qr_data else "None"}.',
            'success': qr_success,
            'input_image': current_step_input_image.copy(),
            'output_image': qr_step_output_img,
            'intermediate_visualizations': qr_viz_steps if visualize_intermediate else {}
        })
        # processing_image usually doesn't change in QR step; current_step_input_image is processing_image

        # --- 3. Corner Finalization (Transform & QR Filter) ---
        # This is a "logical" step that processes corners_initial_raw
        logger.info("=== STEP 3: Corner Finalization ===")
        corner_finalize_viz_steps = {}
        corners_finalized = None  # This will be corners_for_bubbles
        corner_finalize_description = "Processing initial corners."
        corner_finalize_success = False

        if corners_initial_raw and len(corners_initial_raw) == 4:
            corners_in_current_space = corners_initial_raw
            if visualize_intermediate:
                corner_finalize_viz_steps["00_InitialCorners_ForFinalize"] = visualize_corners(image.copy() if transform is None else processing_image.copy(
                ), corners_initial_raw if transform is None else None, message="Input to Finalize (Raw if no transform)")

            if transform is not None:  # If rectification happened, transform corners
                logger.info(
                    "Transforming initial corners to rectified space...")
                try:
                    src_pts_list = [  # TL, TR, BR, BL order
                        corners_initial_raw['top_left']['center'], corners_initial_raw['top_right']['center'],
                        corners_initial_raw['bottom_right']['center'], corners_initial_raw['bottom_left']['center']
                    ]
                    src_pts_np = np.array(
                        [src_pts_list], dtype=np.float32)  # Shape (1, 4, 2)
                    transformed_pts_np = cv2.perspectiveTransform(
                        src_pts_np, transform)
                    if transformed_pts_np is not None:
                        corners_in_current_space = {
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
                    corners_in_current_space = None  # Mark as failed
                    corner_finalize_description = "Failed to transform initial corners. "

            if corners_in_current_space and len(corners_in_current_space) == 4:
                if qr_polygon and self.params.get('corner_detection', {}).get('validator', {}).get('qr_filter_enabled', True):
                    logger.info("Filtering finalized corners by QR polygon...")
                    temp_filtered_corners = {}
                    all_valid_after_qr = True
                    for name, corner_data in corners_in_current_space.items():
                        if corner_data and 'center' in corner_data:
                            center_pt = tuple(
                                map(float, corner_data['center']))
                            # pointPolygonTest returns >0 for inside, 0 for on edge, <0 for outside
                            if cv2.pointPolygonTest(np.array(qr_polygon, dtype=np.int32), center_pt, False) >= 0:
                                logger.warning(
                                    f"Corner '{name}' at {center_pt} is INSIDE QR polygon. Excluding.")
                                # temp_filtered_corners[name] = None # Mark as excluded
                            else:
                                temp_filtered_corners[name] = corner_data
                        else:  # Invalid corner data before QR check
                            all_valid_after_qr = False
                            break

                    # Check if we still have 4 valid corners
                    valid_after_qr_count = sum(
                        1 for c in temp_filtered_corners.values() if c is not None and 'center' in c)

                    if valid_after_qr_count == 4:
                        corners_finalized = temp_filtered_corners
                        corner_finalize_description += "Successfully filtered by QR polygon."
                        corner_finalize_success = True
                    else:
                        logger.warning(
                            f"QR filtering resulted in {valid_after_qr_count}/4 corners. Using corners before QR filter if available, or failing.")
                        corners_finalized = corners_in_current_space if all(
                            corners_in_current_space.values()) else None  # Fallback
                        corner_finalize_description += f"QR filtering left {valid_after_qr_count} corners. Fallback used."
                        corner_finalize_success = corners_finalized is not None and all(
                            corners_finalized.values())

                    if visualize_intermediate:
                        viz_qr_filtered = visualize_corners(
                            processing_image.copy(), corners_finalized, "Corners After QR Filter")
                        cv2.polylines(viz_qr_filtered, [np.array(
                            # Draw QR
                            qr_polygon, dtype=np.int32)], True, (0, 0, 255), 1)
                        corner_finalize_viz_steps["02_CornersAfterQRFilter"] = viz_qr_filtered
                else:  # No QR polygon or filter disabled
                    corners_finalized = corners_in_current_space
                    corner_finalize_description += "No QR filtering applied."
                    corner_finalize_success = corners_finalized is not None and all(
                        corners_finalized.values())
            else:  # corners_in_current_space is None or not 4
                corner_finalize_description += "Not enough valid corners before QR filtering."
                corner_finalize_success = False
        else:  # initial corners not found or incomplete
            corner_finalize_description = "Initial corner detection did not yield 4 corners. Cannot finalize."
            corner_finalize_success = False

        corners_for_bubbles = corners_finalized  # Set for bubble detection step

        # Output image for this logical step
        corner_finalize_output_img = processing_image.copy()
        if visualize_intermediate:
            # Prefer the last visualization if available
            corner_finalize_output_img = corner_finalize_viz_steps.get("02_CornersAfterQRFilter",
                                                                       corner_finalize_viz_steps.get("01_TransformedInitialCorners",
                                                                                                     processing_image.copy()))
        elif corners_finalized:  # If not visualizing intermediate, still show final corners
            corner_finalize_output_img = visualize_corners(
                processing_image.copy(), corners_finalized, "Finalized Corners for Bubbles")

        results['steps'].append({
            'name': '3. Corner Finalization',
            'description': corner_finalize_description,
            'success': corner_finalize_success,
            'input_image': current_step_input_image.copy(),  # Image before this logic
            'output_image': corner_finalize_output_img,
            'intermediate_visualizations': corner_finalize_viz_steps if visualize_intermediate else {}
        })
        # current_step_input_image = processing_image.copy() for next step

        # --- 4. Bubble Detection ---
        logger.info("=== STEP 4: Bubble Detection ===")
        bubble_viz_steps = {}
        bubbles = None  # Initialize
        bubble_detector_input_img = processing_image.copy()  # Capture input for this step

        if corners_for_bubbles is None:  # Critical dependency
            logger.warning(
                "No valid corners available for bubble detection. Skipping bubble detection.")
            bubble_success = False
            bubble_description = "Skipped: No valid corners from previous steps."
            bubble_output_img = bubble_detector_input_img
        else:
            logger.info(
                f"Starting bubble detection with {len(corners_for_bubbles)} corners.")
            bubbles, bubble_viz_steps_from_detector = self.bubble_detector.detect(
                processing_image, qr_polygon, corners_for_bubbles, visualize_intermediate
            )
            bubble_success = bubbles is not None and bubbles.size > 0
            bubble_description = f'Detected {len(bubbles) if bubbles is not None else 0} bubble candidates.'
            if visualize_intermediate:
                bubble_viz_steps.update(bubble_viz_steps_from_detector)

            bubble_output_img_key = "99_FinalDetection"
            bubble_output_img = bubble_viz_steps.get(bubble_output_img_key)
            if bubble_output_img is None:
                from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles
                bubble_output_img = visualize_bubbles(
                    processing_image.copy(), bubbles)

        results['steps'].append({
            'name': '4. Bubble Detection',
            'description': bubble_description,
            'success': bubble_success,
            'input_image': bubble_detector_input_img,
            'output_image': bubble_output_img,
            'intermediate_visualizations': bubble_viz_steps if visualize_intermediate else {}
        })
        if not bubble_success:
            logger.warning(
                "Bubble detection failed or found no bubbles. Subsequent steps might be affected.")
            # Decide if to return or continue with empty bubbles
            # return results

        current_step_input_image_for_analysis = bubble_output_img.copy(
        ) if bubble_success else processing_image.copy()

        # --- 5. Bubble Analysis ---
        logger.info("=== STEP 5: Bubble Analysis ===")
        analysis_viz_steps = {}
        bubble_scores = []
        analysis_input_img = current_step_input_image_for_analysis

        if bubbles is None or bubbles.size == 0:
            logger.warning("No bubbles to analyze. Skipping bubble analysis.")
            analysis_success = False
            analysis_description = "Skipped: No bubbles detected."
            analysis_output_img = analysis_input_img
        else:
            bubble_scores, analysis_viz_steps_from_analyzer = self.bubble_analyzer.analyze(
                # Use original processing_image for pixel analysis
                processing_image, bubbles, visualize_intermediate
            )
            analysis_success = len(bubble_scores) > 0
            analysis_description = f'Analyzed scores for {len(bubble_scores)} question rows.'
            if visualize_intermediate:
                analysis_viz_steps.update(analysis_viz_steps_from_analyzer)

            analysis_output_img_key = "99_FinalScores"
            analysis_output_img = analysis_viz_steps.get(
                analysis_output_img_key)
            if analysis_output_img is None:
                from app.features.analyzer.visualizers.answer_viz import visualize_scores
                analysis_output_img = visualize_scores(
                    processing_image.copy(), bubble_scores)

        results['steps'].append({
            'name': '5. Bubble Analysis',
            'description': analysis_description,
            'success': analysis_success,
            'input_image': analysis_input_img,
            'output_image': analysis_output_img,
            'intermediate_visualizations': analysis_viz_steps if visualize_intermediate else {}
        })

        # --- 6. Answer Extraction ---
        logger.info("=== STEP 6: Answer Extraction ===")
        # This step is mostly data processing, less complex viz usually
        # Use output of previous step as input
        answer_extraction_input_img = analysis_output_img.copy()

        answers = self.answer_extractor.extract(bubble_scores)
        results['final_answers'] = answers
        extraction_success = len(answers) > 0

        # Simple output viz for this step
        from app.features.analyzer.visualizers.answer_viz import visualize_answers
        final_answers_viz = visualize_answers(
            answer_extraction_input_img, answers, bubble_scores)  # Pass scores for coords

        results['steps'].append({
            'name': '6. Answer Extraction',
            'description': f'Extracted {len(answers)} answers.',
            'success': extraction_success,
            'input_image': answer_extraction_input_img,
            'output_image': final_answers_viz,
            'intermediate_visualizations': {}  # Typically no intermediate viz here
        })

        logger.info("Analysis finished.")
        return results

    def _rectify_if_needed(
        self, image: np.ndarray, visualize_steps: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Dict]:
        """
        Performs initial corner detection and optional rectification.
        Returns: (rectified_image_or_none, transform_matrix_or_none, detected_corners_dict_or_none, viz_steps_dict)
        """
        viz_steps_combined = {}  # All visualizations from this phase
        corners_detected_dict: Optional[Dict] = None
        rectified_image: Optional[np.ndarray] = None
        transform_matrix: Optional[np.ndarray] = None

        logger.info(
            "--- Sub-Phase: Initial Corner Detection (for Rectification) ---")
        try:
            # self.corner_detector.detect should return (corners_dict, cd_internal_viz_steps)
            corners_detected_dict, cd_internal_viz = self.corner_detector.detect(
                image, qr_polygon=None, visualize_steps=visualize_steps
            )
            if visualize_steps and isinstance(cd_internal_viz, dict):
                for k, v_img in cd_internal_viz.items():  # Prefix to avoid name clashes
                    viz_steps_combined[f"InitialCD_{k}"] = v_img

            if not (corners_detected_dict and len(corners_detected_dict) == 4 and all(corners_detected_dict.values())):
                logger.warning(
                    f"Initial corner detection did not find 4 valid corners. Found: {corners_detected_dict}")
                # Still return any viz steps collected, but no rectification possible
                # Return the partial/None corners
                return None, None, corners_detected_dict, viz_steps_combined

            logger.info(
                f"Initial corners successfully detected: {list(corners_detected_dict.keys())}")

            # Proceed with angle calculation and potential rectification
            angle = self.rectifier.calculate_angle(corners_detected_dict)
            logger.info(
                f"Calculated initial angle for rectification: {angle:.2f} degrees.")

            analyzer_cfg = self.params.get('analyzer', {})
            enable_rect = analyzer_cfg.get('enable_rectification', True)
            rect_thresh = analyzer_cfg.get('rectification_threshold', 5.0)

            if not enable_rect or abs(angle) < rect_thresh:
                logger.info(
                    f"Rectification skipped (Enabled: {enable_rect}, Angle: {abs(angle):.2f}°, Thresh: {rect_thresh:.2f}°).")
                # No rectification, return original state but with detected corners
                return None, None, corners_detected_dict, viz_steps_combined

            logger.info("--- Sub-Phase: Performing Image Rectification ---")
            # Rectifier uses params from its __init__
            rectified_image, transform_matrix = self.rectifier.rectify(
                image, corners_detected_dict)

            if rectified_image is not None:
                logger.info("Image rectification successful.")
                if visualize_steps:
                    viz_steps_combined["Rectification_99_RectifiedOutput"] = rectified_image.copy(
                    )
            else:
                logger.error(
                    "Image rectification call failed (rectifier.rectify returned None).")
                # Failed rectification, but keep original corners and existing viz
                return None, None, corners_detected_dict, viz_steps_combined

        except Exception as e:
            logger.error(
                f"Exception in _rectify_if_needed: {e}", exc_info=True)
            # Return None for rectified outputs, but pass through any corners/viz found before error
            return None, None, corners_detected_dict, viz_steps_combined

        return rectified_image, transform_matrix, corners_detected_dict, viz_steps_combined
