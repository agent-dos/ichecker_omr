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
from app.features.rectification.pipeline import RectificationPipeline
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
        Each step's output visualization becomes the next step's input visualization.
        """
        results = self._initialize_results(image)
        visualize = self.params.get('debug_options', {}).get('visualize_intermediate_steps', False)
        
        # Track images through the pipeline
        processing_img = image.copy()
        visual_chain = image.copy()  # Visual representation passed between steps
        
        # Step 1: Rectification & Corner Detection
        rect_results = self._process_rectification(processing_img, visual_chain, visualize)
        results['steps'].append(rect_results['step_info'])
        results['transform_matrix'] = rect_results.get('transform')
        
        # Update processing state
        processing_img = rect_results['processing_img']
        visual_chain = rect_results['visual_output']
        corners = rect_results['corners']
        
        # Step 2: QR Code Detection
        qr_results = self._process_qr_detection(processing_img, visual_chain, visualize)
        results['steps'].append(qr_results['step_info'])
        results['qr_data'] = qr_results['qr_data']
        
        visual_chain = qr_results['visual_output']
        qr_polygon = qr_results.get('qr_polygon')
        
        # Step 3: Corner Finalization
        final_corner_results = self._process_corner_finalization(
            processing_img, visual_chain, corners, rect_results.get('transform'), 
            qr_polygon, visualize
        )
        results['steps'].append(final_corner_results['step_info'])
        
        visual_chain = final_corner_results['visual_output']
        corners_for_bubbles = final_corner_results['corners']
        
        # Step 4: Bubble Detection
        bubble_results = self._process_bubble_detection(
            processing_img, visual_chain, corners_for_bubbles, qr_polygon, visualize
        )
        results['steps'].append(bubble_results['step_info'])
        
        visual_chain = bubble_results['visual_output']
        bubbles = bubble_results.get('bubbles')
        
        # Step 5: Bubble Analysis
        if bubbles is not None and bubbles.size > 0:
            analysis_results = self._process_bubble_analysis(
                processing_img, visual_chain, bubbles, visualize
            )
            results['steps'].append(analysis_results['step_info'])
            visual_chain = analysis_results['visual_output']
            bubble_scores = analysis_results['scores']
        else:
            # Skip analysis if no bubbles
            results['steps'].append({
                'name': '5. Bubble Analysis',
                'description': 'Skipped: No bubbles detected',
                'success': False,
                'input_image': visual_chain,
                'output_image': visual_chain,
                'intermediate_visualizations': {}
            })
            bubble_scores = []
        
        # Step 6: Answer Extraction
        answer_results = self._process_answer_extraction(visual_chain, bubble_scores)
        results['steps'].append(answer_results['step_info'])
        results['final_answers'] = answer_results['answers']
        
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

    def _process_rectification(self, processing_img: np.ndarray, visual_chain: np.ndarray, 
                            visualize: bool) -> Dict:
        """Process rectification and initial corner detection."""
        logger.info("=== STEP 1: Rectification & Initial Corner Detection ===")
        
        # Perform rectification
        rectified_img, transform, corners, viz_steps = self._rectify_if_needed(
            processing_img, visualize
        )
        
        # Determine output image and create visualization
        if rectified_img is not None:
            processing_img = rectified_img
            success_msg = "Rectification applied successfully"
        else:
            success_msg = "No rectification needed/possible"
        
        # Create corner visualization for visual chain
        if corners and len(corners) == 4:
            output_viz = visualize_corners(processing_img.copy(), corners, 
                                        f"Detected {len(corners)} Corners")
        else:
            output_viz = processing_img.copy()
            cv2.putText(output_viz, "No corners detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return {
            'step_info': {
                'name': '1. Rectification & Initial Corners',
                'description': f'{success_msg}. Corners: {len(corners) if corners else 0}',
                'success': True,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': viz_steps if visualize else {}
            },
            'processing_img': processing_img,
            'visual_output': output_viz,
            'corners': corners,
            'transform': transform
        }

    def _process_qr_detection(self, processing_img: np.ndarray, visual_chain: np.ndarray,
                            visualize: bool) -> Dict:
        """Process QR code detection."""
        logger.info("=== STEP 2: QR Code Detection ===")
        
        # Detect QR code
        qr_data, qr_info, viz_steps = self.qr_detector.detect(processing_img, visualize)
        
        # Create visualization with corners in background
        output_viz = visual_chain.copy()
        
        if qr_data and qr_info:
            # Draw QR polygon if available
            qr_polygon = qr_info.get('polygon')
            if qr_polygon:
                cv2.polylines(output_viz, [np.array(qr_polygon, dtype=np.int32)], 
                            True, (0, 0, 255), 2)
            
            # Add QR data text
            if 'rect' in qr_info:
                x, y = qr_info['rect']['left'], qr_info['rect']['top'] - 10
                cv2.putText(output_viz, f"QR: {qr_data}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(output_viz, "No QR Code Detected", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return {
            'step_info': {
                'name': '2. QR Code Detection',
                'description': f'QR Data: {qr_data if qr_data else "None"}',
                'success': qr_data is not None,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': viz_steps if visualize else {}
            },
            'visual_output': output_viz,
            'qr_data': qr_data,
            'qr_polygon': qr_info.get('polygon') if qr_info else None
        }

    def _process_corner_finalization(self, processing_img: np.ndarray, visual_chain: np.ndarray,
                                    corners: Optional[Dict], transform: Optional[np.ndarray],
                                    qr_polygon: Optional[List], visualize: bool) -> Dict:
        """Process corner finalization with transform and QR filtering."""
        logger.info("=== STEP 3: Corner Finalization ===")
        
        finalized_corners = self._finalize_corners(corners, transform, qr_polygon)
        
        # Create visualization
        if finalized_corners and all(finalized_corners.values()):
            output_viz = visualize_corners(processing_img.copy(), finalized_corners,
                                        "Finalized Corners for Bubbles")
            description = "Corners finalized successfully"
            success = True
        else:
            output_viz = visual_chain.copy()
            cv2.putText(output_viz, "Corner finalization failed", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            description = "Failed to finalize corners"
            success = False
        
        return {
            'step_info': {
                'name': '3. Corner Finalization',
                'description': description,
                'success': success,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': {}
            },
            'visual_output': output_viz,
            'corners': finalized_corners
        }

    def _process_bubble_detection(self, processing_img: np.ndarray, visual_chain: np.ndarray,
                                corners: Optional[Dict], qr_polygon: Optional[List],
                                visualize: bool) -> Dict:
        """Process bubble detection."""
        logger.info("=== STEP 4: Bubble Detection ===")
        
        if not corners:
            return {
                'step_info': {
                    'name': '4. Bubble Detection',
                    'description': 'Skipped: No valid corners',
                    'success': False,
                    'input_image': visual_chain,
                    'output_image': visual_chain,
                    'intermediate_visualizations': {}
                },
                'visual_output': visual_chain,
                'bubbles': None
            }
        
        # Detect bubbles
        bubbles, viz_steps = self.bubble_detector.detect(
            processing_img, qr_polygon, corners, visualize
        )
        
        # Create visualization
        output_viz = visualize_bubbles(processing_img.copy(), bubbles)
        
        return {
            'step_info': {
                'name': '4. Bubble Detection',
                'description': f'Detected {len(bubbles) if bubbles is not None else 0} bubbles',
                'success': bubbles is not None and bubbles.size > 0,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': viz_steps if visualize else {}
            },
            'visual_output': output_viz,
            'bubbles': bubbles
        }

    def _process_bubble_analysis(self, processing_img: np.ndarray, visual_chain: np.ndarray,
                                bubbles: np.ndarray, visualize: bool) -> Dict:
        """Process bubble analysis to determine filled bubbles."""
        logger.info("=== STEP 5: Bubble Analysis ===")
        
        # Analyze bubbles
        scores = self.bubble_analyzer.analyze(processing_img, bubbles)
        
        # Create visualization
        output_viz = visualize_scores(processing_img.copy(), scores)
        
        return {
            'step_info': {
                'name': '5. Bubble Analysis',
                'description': f'Analyzed {len(scores)} question rows',
                'success': len(scores) > 0,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': {}
            },
            'visual_output': output_viz,
            'scores': scores
        }

    def _process_answer_extraction(self, visual_chain: np.ndarray, 
                                bubble_scores: List[Dict]) -> Dict:
        """Process final answer extraction."""
        logger.info("=== STEP 6: Answer Extraction ===")
        
        # Extract answers
        answers = self.answer_extractor.extract(bubble_scores)
        
        # Create final visualization
        output_viz = visualize_answers(visual_chain.copy(), answers, bubble_scores)
        
        return {
            'step_info': {
                'name': '6. Answer Extraction',
                'description': f'Extracted {len(answers)} answers',
                'success': len(answers) > 0,
                'input_image': visual_chain,
                'output_image': output_viz,
                'intermediate_visualizations': {}
            },
            'answers': answers
        }

    def _rectify_if_needed(
        self,
        image: np.ndarray,
        visualize_steps: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Dict]:
        """
        Performs initial corner detection and optional rectification based on angle.
        """
        viz_steps_combined = {}
        corners_detected_dict: Optional[Dict] = None
        rectified_image: Optional[np.ndarray] = None
        transform_matrix: Optional[np.ndarray] = None

        logger.info("--- Sub-Phase: Initial Corner Detection (for Rectification) ---")
        
        # Use the rectification pipeline
        pipeline = RectificationPipeline(self.params)
        rectified_image, results = pipeline.process(image, visualize_steps)
        
        # Extract results
        corners_detected_dict = results.get('corners')
        transform_matrix = results.get('transform')
        
        # Merge visualization steps
        if visualize_steps and 'visualizations' in results:
            for key, viz_img in results['visualizations'].items():
                viz_steps_combined[f"Rectification_{key}"] = viz_img
        
        # Add corner visualization specifically
        if visualize_steps and corners_detected_dict:
            corner_viz = visualize_corners(
                image.copy(), 
                corners_detected_dict, 
                f"Detected Corners (Angle: {results.get('angle', 0):.2f}°)"
            )
            viz_steps_combined["Rectification_corner_detection"] = corner_viz
        
        return rectified_image, transform_matrix, corners_detected_dict, viz_steps_combined