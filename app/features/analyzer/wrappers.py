# app/features/analyzer/wrappers.py
"""Detection wrapper functions from original utils/detect_*.py"""
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def detect_answers(
    image,
    visualize=True,
    **params
):
    """Wrapper for answer detection (from detect_answers.py)."""
    from app.features.analyzer.service import AnalyzerService

    analyzer = AnalyzerService(params)
    results = analyzer.analyze(image)

    answers = results.get('answers', [])

    if not visualize:
        return answers

    viz_image = results['steps'][-1]['output_image'] if results.get(
        'steps') else image
    return answers, viz_image


def detect_qr_code(image, visualize=True):
    """Wrapper for QR detection (from detect_qr_code.py)."""
    from app.features.qr.detector import QRDetector

    detector = QRDetector()
    qr_data, qr_info = detector.detect(image)

    if not visualize:
        return qr_data

    viz_image = detector.visualize(image, qr_data, qr_info)
    return qr_data, viz_image
