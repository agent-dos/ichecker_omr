# app/services/answer_service.py
import cv2
import numpy as np
from app.utils.answer_processing import process_answer_sheet as process_answers_core
from app.utils.answer_visualization import visualize_answers
from app.utils.image_processing import convert_to_grayscale, apply_adaptive_threshold
from app.utils.visualization import visualize_circles, visualize_threshold
from app.utils.circle_detection import detect_circles_with_filters  # ADD THIS IMPORT
from app.services.answer_pipeline_steps import (
    create_grayscale_step,
    create_corner_detection_step,
    create_circle_detection_step,
    create_threshold_step,
    create_answer_extraction_step
)


def process_answer_sheet_pipeline(image, params=None, include_steps=False, qr_polygon=None):
    """
    Process the answer sheet image with full pipeline visualization.

    Args:
        image: The input image
        params: Optional dictionary of parameters
        include_steps: Whether to include step-by-step results
        qr_polygon: QR code polygon to exclude from bubble detection

    Returns:
        dict: Processing results including visualizations and answers
    """
    # Use provided parameters or defaults
    if params is None:
        from app.utils.config import get_default_parameters
        params = get_default_parameters()

    # Initialize result dictionary
    results = {}
    steps = []

    # Step 1: Convert to grayscale
    gray = convert_to_grayscale(image)
    grayscale_viz = gray.copy()

    if include_steps:
        steps.append(create_grayscale_step(image, grayscale_viz))

    # Step 2: Corner detection (optional)
    answer_boundary = None
    if include_steps and params.get('use_corner_detection', True):
        corner_step, corners = create_corner_detection_step(image, qr_polygon)
        if corner_step:
            steps.append(corner_step)
            if corners is not None:
                from app.utils.corner_detection import get_bounding_quadrilateral
                answer_boundary = get_bounding_quadrilateral(
                    corners, margin=20)

    # Step 3: Detect circles with filters
    circles = _detect_circles_with_filters(
        gray, params, qr_polygon, answer_boundary
    )

    # Create visualization and statistics
    circles_viz, circles_stats = _create_circle_visualization(
        image, circles
    )

    if include_steps:
        steps.append(create_circle_detection_step(
            grayscale_viz, circles_viz, circles_stats, circles
        ))

    # Step 4: Apply thresholding
    thresh = apply_adaptive_threshold(
        gray,
        block_size=params['block_size'],
        c_value=params['c_value']
    )
    thresh_color = visualize_threshold(thresh.copy())

    if include_steps:
        steps.append(create_threshold_step(circles_viz, thresh_color))

    # Step 5: Detect answers
    answers, coords = process_answers_core(
        image,
        bubble_threshold=params['bubble_threshold'],
        param1=params['param1'],
        param2=params['param2'],
        min_radius=params['min_radius'],
        max_radius=params['max_radius'],
        resize_factor=params['resize_factor'],
        block_size=params['block_size'],
        c_value=params['c_value'],
        row_threshold=params['row_threshold'],
        score_multiplier=params['score_multiplier'],
        qr_polygon=qr_polygon,
        use_corner_detection=params.get('use_corner_detection', True)
    )

    final_viz = visualize_answers(image.copy(), answers, coords)

    if include_steps:
        steps.append(create_answer_extraction_step(
            thresh_color, final_viz, answers
        ))

    # Build results
    results = {
        'grayscale': grayscale_viz,
        'circles_visualization': circles_viz,
        'threshold_visualization': thresh_color,
        'final_visualization': final_viz,
        'answers': answers,
        'coords': coords,
        'parameters': params,
        'circles_stats': circles_stats
    }

    if include_steps:
        results['steps'] = steps

    return results


def process_answer_sheet(image, params=None):
    """
    Simple answer sheet processing without pipeline steps.
    """
    return process_answer_sheet_pipeline(image, params, include_steps=False)


def _detect_circles_with_filters(gray, params, qr_polygon, answer_boundary):
    """
    Detect circles with QR exclusion and boundary filtering.
    """
    return detect_circles_with_filters(
        gray,
        param1=params['param1'],
        param2=params['param2'],
        min_radius=params['min_radius'],
        max_radius=params['max_radius'],
        resize_factor=params.get('resize_factor', 1.0),
        qr_polygon=qr_polygon,
        answer_boundary=answer_boundary
    )


def _create_circle_visualization(image, circles):
    """
    Create circle visualization and calculate statistics.
    """
    circles_viz = visualize_circles(
        image.copy(), circles) if circles is not None else image.copy()

    circles_stats = {'total': 0, 'left': 0, 'right': 0}
    if circles is not None:
        circles_stats['total'] = len(circles)
        midpoint_x = image.shape[1] // 2
        circles_stats['left'] = sum(1 for c in circles if c[0] < midpoint_x)
        circles_stats['right'] = sum(1 for c in circles if c[0] >= midpoint_x)

    return circles_viz, circles_stats
