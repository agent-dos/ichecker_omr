# app/utils/answer_processing.py
import cv2
import numpy as np
import logging
from app.utils.circle_detection import detect_circles_with_filters
from app.utils.bubble_analysis import analyze_bubbles
from app.utils.circle_grouping import group_circles_by_position
from app.utils.corner_detection import detect_corner_markers, get_bounding_quadrilateral

logger = logging.getLogger(__name__)


def process_answer_sheet(
        image,
        bubble_threshold=100.0,
        param1=50,
        param2=18,
        min_radius=10,
        max_radius=20,
        resize_factor=1.0,
        block_size=31,
        c_value=10,
        row_threshold=8,
        score_multiplier=1.5,
        qr_polygon=None,
        use_corner_detection=True
):
    """
    Process answer sheet with QR exclusion and corner-based boundary detection.

    Args:
            image: Input image
            bubble_threshold: Threshold for bubble detection
            param1: HoughCircles parameter
            param2: HoughCircles parameter
            min_radius: Minimum bubble radius
            max_radius: Maximum bubble radius
            resize_factor: Image resize factor
            block_size: Adaptive threshold block size
            c_value: Adaptive threshold constant
            row_threshold: Row grouping threshold
            score_multiplier: Score multiplier for bubble detection
            qr_polygon: QR code polygon to exclude from detection
            use_corner_detection: Whether to use corner markers for boundary

    Returns:
            tuple: (answers, coords)
    """
    # Validate inputs
    if image is None:
        logger.warning("process_answer_sheet: Input image is None.")
        return [], []

    # Detect corner markers if enabled
    answer_boundary = None
    if use_corner_detection:
        answer_boundary = _detect_answer_boundary(image)

    # Preprocess image with resize if needed
    resized = _resize_image(image, resize_factor)

    # Scale boundary if image was resized
    if answer_boundary is not None and resize_factor != 1.0:
        answer_boundary = answer_boundary * resize_factor

    # Detect circles with all filters applied
    circles = detect_circles_with_filters(
        resized, param1, param2, min_radius, max_radius,
        resize_factor, qr_polygon, answer_boundary
    )

    if circles is None:
        logger.debug("process_answer_sheet: No circles detected.")
        return [], []

    # Group circles into columns and rows
    grouped_circles = group_circles_by_position(
        circles, resized.shape[1], row_threshold, resize_factor
    )

    # Create threshold image for bubble fill detection
    thresh = _create_threshold_image(resized, block_size, c_value)

    # Analyze each bubble for fill score
    answers, coords = analyze_bubbles(
        grouped_circles, thresh, bubble_threshold, score_multiplier
    )

    logger.debug(
        f"process_answer_sheet completed. Found {len(answers)} answers.")
    return answers, coords


def _detect_answer_boundary(image):
    """
    Detect the answer sheet boundary using corner markers.

    Returns:
            numpy.ndarray: Quadrilateral defining the answer area, or None
    """
    corners = detect_corner_markers(image)
    if corners is None:
        logger.warning(
            "Could not detect corner markers; processing entire image")
        return None

    # Get bounding quadrilateral with small inward margin
    boundary = get_bounding_quadrilateral(corners, margin=20)
    return boundary


def _resize_image(image, resize_factor):
    """Resize image if factor is not 1.0"""
    if resize_factor == 1.0:
        return image.copy()

    height, width = image.shape[:2]
    new_dimensions = (int(width * resize_factor), int(height * resize_factor))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


def _create_threshold_image(image, block_size, c_value):
    """
    Create adaptive threshold image for bubble fill detection.
    """
    gray = _convert_to_grayscale(image)

    # Ensure block_size is odd
    block_size = block_size if block_size % 2 != 0 else block_size + 1
    block_size = max(3, block_size)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )

    return thresh


def _convert_to_grayscale(image):
    """Convert image to grayscale handling different formats."""
    if len(image.shape) == 2:
        return image
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
