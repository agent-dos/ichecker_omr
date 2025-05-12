# filename: app/features/analyzer/service.py

# ... (imports remain the same) ...
from typing import Dict, List, Optional, Tuple

import numpy as np
from app.features.analyzer.bubble_detector import BubbleDetector
from app.features.analyzer.bubble_analyzer import BubbleAnalyzer
from app.features.analyzer.answer_extractor import AnswerExtractor
# CornerDetector import might become unused if no other method calls it directly
from app.features.corners.detector import CornerDetector
from app.features.qr.detector import QRDetector
from app.features.rectification.rectifier import ImageRectifier
# Import the visualizer if you want to add back the visualization step
# from app.features.corners.visualizer import visualize_corners


class AnalyzerService:
    """
    Orchestrates answer sheet analysis.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.bubble_detector = BubbleDetector(params)
        self.bubble_analyzer = BubbleAnalyzer(params)
        self.answer_extractor = AnswerExtractor()
        self.corner_detector = CornerDetector()  # Keep instance for _rectify_if_needed
        self.qr_detector = QRDetector()
        self.rectifier = ImageRectifier()

    def analyze(
        self,
        image: np.ndarray
    ) -> Dict:
        """
        Analyze answer sheet and return results.
        """
        results = {
            'original_image': image,
            'steps': []
        }
        initial_corners = None  # Define variable to hold corners

        # Step 1: Rectify image if needed (modified to capture corners)
        # Now returns: rectified_image, transform_matrix, corners_found
        rectified, transform, initial_corners = self._rectify_if_needed(
            image, results)
        processing_image = rectified if rectified is not None else image

        # Step 2: Detect QR code
        qr_polygon = self._detect_qr(processing_image, results)

        # Step 3: Detect corners (REMOVED)
        # corners = self._detect_corners(processing_image, qr_polygon, results)
        # --- Visualization step for corners is also removed ---
        # If you want to visualize the initial_corners, you could add a step here:
        # if initial_corners:
        #     from app.features.corners.visualizer import visualize_corners
        #     viz = visualize_corners(processing_image, initial_corners) # Visualize on processing_image
        #     results['steps'].append({
        #         'name': 'Initial Corner Visualization',
        #         'input_image': processing_image, # Show input for context
        #         'output_image': viz,
        #         'description': 'Visualizing corners used for rectification check',
        #         'success': True
        #     })

        # Step 4: Detect bubbles (modified to use initial_corners)
        # Pass initial_corners instead of the 'corners' variable from the removed step
        bubbles = self._detect_bubbles(
            processing_image, qr_polygon, initial_corners, results)  # Use initial_corners

        # Step 5: Analyze bubbles
        bubble_scores = self._analyze_bubbles(
            processing_image, bubbles, results)

        # Step 6: Extract answers
        answers = self._extract_answers(bubble_scores, results)

        results['answers'] = answers
        # Store initial corners if needed, though they might be in original image space
        results['transform_matrix'] = transform
        # results['corners_used_for_bubbles'] = initial_corners # Optional: add for debugging

        return results

    def _rectify_if_needed(
        self,
        image: np.ndarray,
        results: Dict
        # Modified return type hint
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Rectify image if tilted. Returns rectified image, transform, and detected corners.
        """
        # Detect corners on the original image
        corners = self.corner_detector.detect(
            image)  # No QR polygon filter here
        if corners is None:
            # Return None for image, transform, and corners if detection fails
            return None, None, None

        angle = self.rectifier.calculate_angle(corners)
        # Check if rectification is enabled and angle exceeds threshold
        if not self.params.get('enable_rectification', True) or \
           abs(angle) < self.params.get('rectification_threshold', 5.0):
            # No rectification needed, return original image, no transform, but return the found corners
            return None, None, corners

        # Perform rectification
        rectified, transform = self.rectifier.rectify(image, corners)

        # Add step to results
        results['steps'].append({
            'name': 'Image Rectification',
            'input_image': image,
            'output_image': rectified,
            'description': f'Corrected {angle:.1f}° tilt',
            'success': rectified is not None and transform is not None
        })

        # Return rectified image, transform, and the corners used for this rectification
        return rectified, transform, corners

    def _detect_qr(
        self,
        image: np.ndarray,
        results: Dict
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Detect QR code and return polygon. (No changes needed here)
        """
        qr_data, qr_info = self.qr_detector.detect(image)

        # Visualize QR detection
        from app.features.qr.visualizer import visualize_qr  # Import locally if needed
        viz_qr = visualize_qr(image, qr_data, qr_info)

        results['steps'].append({
            'name': 'QR Detection',
            'input_image': image,
            'output_image': viz_qr,  # Use the visualized image
            'description': 'Detecting QR code',
            'success': qr_data is not None
        })

        results['qr_data'] = qr_data
        results['qr_info'] = qr_info

        return qr_info.get('polygon') if qr_info else None

    # REMOVED _detect_corners method and its call in analyze()
    # def _detect_corners(
    #     self,
    #     image: np.ndarray,
    #     qr_polygon: Optional[List[Tuple[int, int]]],
    #     results: Dict
    # ) -> Optional[Dict]:
    #     """
    #     Detect corner markers.
    #     """
    #     # This method is no longer called

    def _detect_bubbles(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]],
        # Parameter name changed to reflect source
        initial_corners: Optional[Dict],
        results: Dict
    ) -> Optional[np.ndarray]:
        """
        Detect answer bubbles. Uses initial_corners for boundary.
        """
        # Pass initial_corners to the detector
        bubbles = self.bubble_detector.detect(
            image, qr_polygon, initial_corners)

        from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles
        viz = visualize_bubbles(image, bubbles)

        results['steps'].append({
            'name': 'Bubble Detection',
            'input_image': image,
            'output_image': viz,
            'description': 'Detecting answer bubbles (using initial corners for boundary)',
            'success': bubbles is not None and bubbles.size > 0  # Check size too
        })

        return bubbles

    def _analyze_bubbles(
        self,
        image: np.ndarray,
        bubbles: Optional[np.ndarray],
        results: Dict
    ) -> List[Dict]:
        """
        Analyze bubble fill scores. (No changes needed here)
        """
        if bubbles is None or bubbles.size == 0:  # Check size
            results['steps'].append({
                'name': 'Bubble Analysis',
                'description': 'Skipped: No bubbles detected in previous step.',
                'success': False
            })
            return []

        scores = self.bubble_analyzer.analyze(image, bubbles)

        from app.features.analyzer.visualizers.answer_viz import visualize_scores
        viz = visualize_scores(image, scores)

        results['steps'].append({
            'name': 'Bubble Analysis',
            'input_image': image,  # Show image before score viz if needed
            'output_image': viz,
            'description': 'Analyzing fill scores',
            'success': len(scores) > 0
        })

        return scores

    def _extract_answers(
        self,
        bubble_scores: List[Dict],
        results: Dict
    ) -> List[Tuple[int, str]]:
        """
        Extract final answers from scores. (No changes needed here)
        """
        # Add input check
        if not bubble_scores:
            results['steps'].append({
                'name': 'Answer Extraction',
                'description': 'Skipped: No bubble scores from previous step.',
                'success': False
            })
            return []

        answers = self.answer_extractor.extract(bubble_scores)

        # Optionally visualize final answers on the last visualized image
        last_viz_image = results['steps'][-1]['output_image'] if results['steps'] else results['original_image']
        from app.features.analyzer.visualizers.answer_viz import visualize_answers
        final_viz = visualize_answers(last_viz_image, answers, bubble_scores)

        results['steps'].append({
            'name': 'Answer Extraction',
            'input_image': last_viz_image,  # Show image before answer viz
            'output_image': final_viz,  # Show final visualization
            'description': 'Extracting final answers',
            'success': len(answers) > 0
        })

        return answers
