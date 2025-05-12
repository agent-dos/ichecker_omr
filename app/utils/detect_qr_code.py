# app/utils/detect_qr_code.py
import logging
from app.utils.qr_processing import detect_qr_code_data
from app.utils.qr_visualization import visualize_qr_code

logger = logging.getLogger(__name__)


def detect_qr_code_with_info(image):
    """
    Detects a QR code and returns:
    - qr_data: string content of QR
    - qr_info: dict with polygon, bounding box, and center

    Args:
            image: Input image (BGR format)

    Returns:
            tuple: (qr_data, qr_info)
    """
    return detect_qr_code_data(image)


def detect_qr_code(image, visualize=True):
    """
    Detects QR code and optionally renders visualization.

    Args:
            image: Input image
            visualize: Whether to create visualization

    Returns:
            - If visualize: (qr_data, annotated_image)
            - Else: qr_data
    """
    # Detect QR code
    qr_data, qr_info = detect_qr_code_data(image)

    if not visualize:
        return qr_data

    # Create visualization
    viz_image = visualize_qr_code(image, qr_data, qr_info)

    return qr_data, viz_image
