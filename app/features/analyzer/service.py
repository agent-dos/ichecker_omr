# app/features/analyzer/service.py
"""
Main analyzer service orchestrating the analysis pipeline.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from app.features.analyzer.bubble_detector import BubbleDetector
from app.features.analyzer.bubble_analyzer import BubbleAnalyzer
from app.features.analyzer.answer_extractor import AnswerExtractor
from app.features.corners.detector import CornerDetector
from app.features.qr.detector import QRDetector
from app.features.rectification.rectifier import ImageRectifier


class AnalyzerService:
    """
    Orchestrates answer sheet analysis.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.bubble_detector = BubbleDetector(params)
        self.bubble_analyzer = BubbleAnalyzer(params)
        self.answer_extractor = AnswerExtractor()
        self.corner_detector = CornerDetector()
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

        # Step 1: Rectify image if needed
        rectified, transform = self._rectify_if_needed(image, results)
        processing_image = rectified if rectified is not None else image

        # Step 2: Detect QR code
        qr_polygon = self._detect_qr(processing_image, results)

        # Step 3: Detect corners
        corners = self._detect_corners(processing_image, qr_polygon, results)

        # Step 4: Detect bubbles
        bubbles = self._detect_bubbles(
            processing_image, qr_polygon, corners, results)

        # Step 5: Analyze bubbles
        bubble_scores = self._analyze_bubbles(
            processing_image, bubbles, results)

        # Step 6: Extract answers
        answers = self._extract_answers(bubble_scores, results)

        results['answers'] = answers
        results['transform_matrix'] = transform

        return results

    def _rectify_if_needed(
        self,
        image: np.ndarray,
        results: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Rectify image if tilted.
        """
        corners = self.corner_detector.detect(image)
        if corners is None:
            return None, None

        angle = self.rectifier.calculate_angle(corners)
        if abs(angle) < self.params.get('rectification_threshold', 5.0):
            return None, None

        rectified, transform = self.rectifier.rectify(image, corners)

        results['steps'].append({
            'name': 'Image Rectification',
            'input_image': image,
            'output_image': rectified,
            'description': f'Corrected {angle:.1f}° tilt',
            'success': True
        })

        return rectified, transform

    def _detect_qr(
        self,
        image: np.ndarray,
        results: Dict
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Detect QR code and return polygon.
        """
        qr_data, qr_info = self.qr_detector.detect(image)

        results['steps'].append({
            'name': 'QR Detection',
            'input_image': image,
            'output_image': self.qr_detector.visualize(image, qr_data, qr_info),
            'description': 'Detecting QR code',
            'success': qr_data is not None
        })

        results['qr_data'] = qr_data
        results['qr_info'] = qr_info

        return qr_info.get('polygon') if qr_info else None

    def _detect_corners(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]],
        results: Dict
    ) -> Optional[Dict]:
        """
        Detect corner markers.
        """
        corners = self.corner_detector.detect(image, qr_polygon)

        from app.features.corners.visualizer import visualize_corners
        viz = visualize_corners(image, corners)

        results['steps'].append({
            'name': 'Corner Detection',
            'input_image': image,
            'output_image': viz,
            'description': 'Detecting corner markers',
            'success': corners is not None
        })

        return corners

    def _detect_bubbles(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]],
        corners: Optional[Dict],
        results: Dict
    ) -> Optional[np.ndarray]:
        """
        Detect answer bubbles.
        """
        bubbles = self.bubble_detector.detect(image, qr_polygon, corners)

        from app.features.analyzer.visualizers.bubble_viz import visualize_bubbles
        viz = visualize_bubbles(image, bubbles)

        results['steps'].append({
            'name': 'Bubble Detection',
            'input_image': image,
            'output_image': viz,
            'description': 'Detecting answer bubbles',
            'success': bubbles is not None
        })

        return bubbles

    def _analyze_bubbles(
        self,
        image: np.ndarray,
        bubbles: Optional[np.ndarray],
        results: Dict
    ) -> List[Dict]:
        """
        Analyze bubble fill scores.
        """
        if bubbles is None:
            return []

        scores = self.bubble_analyzer.analyze(image, bubbles)

        from app.features.analyzer.visualizers.answer_viz import visualize_scores
        viz = visualize_scores(image, scores)

        results['steps'].append({
            'name': 'Bubble Analysis',
            'input_image': image,
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
        Extract final answers from scores.
        """
        answers = self.answer_extractor.extract(bubble_scores)

        results['steps'].append({
            'name': 'Answer Extraction',
            'description': 'Extracting final answers',
            'success': len(answers) > 0
        })

        return answers
