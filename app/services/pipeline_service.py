# app/services/pipeline_service.py (key modifications)
import cv2
import numpy as np
from app.services.qr_service import process_qr_code
from app.services.answer_service import process_answer_sheet_pipeline
from app.utils.image_display import ensure_bgr
from app.utils.image_rectification import (
    rectify_image_with_corners,
    is_rectification_needed,
    calculate_rotation_angle
)
from app.utils.rectification_visualization import visualize_rectification


def process_full_pipeline(image, params=None):
    """
    Process both QR detection and answer detection with tilt correction.
    """
    # Ensure input image is in BGR format
    image = ensure_bgr(image)

    # Initialize the pipeline result dictionary
    pipeline_results = {
        'original_image': image,
        'steps': []
    }

    # Step 1: Initial corner detection for rectification
    rectified_image, transform_matrix = _apply_rectification_if_needed(
        image, pipeline_results
    )

    # Use rectified image for processing
    processing_image = rectified_image if rectified_image is not None else image

    # Step 2: QR Code Detection
    qr_polygon = None
    try:
        qr_results = process_qr_code(processing_image)

        pipeline_results['qr_data'] = qr_results['qr_data']
        pipeline_results['qr_info'] = qr_results['qr_info']

        if qr_results['qr_info'] and 'polygon' in qr_results['qr_info']:
            qr_polygon = qr_results['qr_info']['polygon']

        # Add QR detection step
        pipeline_results['steps'].append({
            'name': 'QR Code Detection',
            'input_image': processing_image,
            'output_image': qr_results['visualization'],
            'description': 'Detects and decodes QR codes in the rectified image.',
            'functions': qr_results.get('functions', []),
            'success': qr_results['qr_data'] is not None
        })

    except Exception as e:
        _add_error_step(pipeline_results, 'QR Code Detection',
                        processing_image, str(e))

    # Step 3: Answer Sheet Analysis
    try:
        answer_results = process_answer_sheet_pipeline(
            processing_image,
            params,
            include_steps=True,
            qr_polygon=qr_polygon
        )

        # Add answer detection steps
        if 'steps' in answer_results:
            pipeline_results['steps'].extend(answer_results['steps'])

        # Store results
        pipeline_results['answers'] = answer_results['answers']
        pipeline_results['coords'] = answer_results['coords']

        # Store transformation matrix for mapping back to original
        pipeline_results['transform_matrix'] = transform_matrix

    except Exception as e:
        _add_error_step(pipeline_results, 'Answer Detection',
                        processing_image, str(e))

    return pipeline_results


def _apply_rectification_if_needed(image, pipeline_results):
    """
    Apply rectification if image is tilted.

    Returns:
        tuple: (rectified_image, transform_matrix)
    """
    from app.utils.corner_detection import detect_corner_markers

    # Detect corners for rectification check
    corners = detect_corner_markers(image)

    if corners is None:
        return None, None

    # Check if rectification is needed
    angle = calculate_rotation_angle(corners)
    needs_rectification = is_rectification_needed(corners)

    if not needs_rectification:
        pipeline_results['steps'].append({
            'name': 'Tilt Detection',
            'input_image': image,
            'output_image': image,
            'description': f'No significant tilt detected (angle: {angle:.1f}°)',
            'success': True
        })
        return None, None

    # Apply rectification
    rectified, transform_matrix = rectify_image_with_corners(image, corners)

    # Create visualization
    viz = visualize_rectification(image, rectified, corners, angle)

    pipeline_results['steps'].append({
        'name': 'Image Rectification',
        'input_image': image,
        'output_image': viz,
        'description': f'Corrected {angle:.1f}° tilt using corner detection',
        'functions': ['cv2.getPerspectiveTransform', 'cv2.warpPerspective'],
        'success': True
    })

    return rectified, transform_matrix


def _add_error_step(pipeline_results, step_name, image, error_msg):
    """Helper to add error step to pipeline."""
    pipeline_results['steps'].append({
        'name': step_name,
        'input_image': image,
        'output_image': image.copy(),
        'description': f'Error: {error_msg}',
        'functions': [],
        'success': False
    })
