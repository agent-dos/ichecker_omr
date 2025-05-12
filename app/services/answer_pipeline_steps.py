# app/services/answer_pipeline_steps.py
"""
Pipeline step creation functions for answer sheet processing.
"""


def create_grayscale_step(input_image, output_image):
    """Create grayscale conversion step."""
    return {
        'name': 'Grayscale Conversion',
        'input_image': input_image,
        'output_image': output_image,
        'description': 'The original color image is converted to grayscale to simplify processing.',
        'functions': ['cv2.cvtColor'],
        'success': True
    }


def create_corner_detection_step(image, qr_polygon=None):
    """Create corner detection step."""
    from app.utils.corner_detection import detect_corner_markers, visualize_corners

    corners = detect_corner_markers(image, qr_polygon=qr_polygon)
    corner_viz = visualize_corners(image, corners)

    step = {
        'name': 'Corner Detection',
        'input_image': image,
        'output_image': corner_viz,
        'description': 'Detects the four corner markers to define the valid answer area.',
        'functions': ['cv2.findContours', 'cv2.boundingRect', 'cv2.polylines'],
        'success': corners is not None
    }

    return step, corners


def create_circle_detection_step(input_image, output_image, stats, circles):
    """Create circle detection step."""
    return {
        'name': 'Circle Detection',
        'input_image': input_image,
        'output_image': output_image,
        'description': 'The Hough Circle Transform algorithm detects circular shapes (bubbles) in the image.',
        'functions': ['cv2.HoughCircles', 'cv2.circle', 'cv2.line'],
        'stats': stats,
        'success': circles is not None and len(circles) > 0
    }


def create_threshold_step(input_image, output_image):
    """Create threshold analysis step."""
    return {
        'name': 'Bubble Analysis',
        'input_image': input_image,
        'output_image': output_image,
        'description': 'Adaptive thresholding identifies filled bubbles by separating dark marks from the background.',
        'functions': ['cv2.adaptiveThreshold', 'cv2.applyColorMap'],
        'success': True
    }


def create_answer_extraction_step(input_image, output_image, answers):
    """Create answer extraction step."""
    return {
        'name': 'Answer Extraction',
        'input_image': input_image,
        'output_image': output_image,
        'description': 'The filled bubbles are analyzed to determine the selected answers for each question.',
        'functions': ['cv2.bitwise_and', 'cv2.countNonZero', 'cv2.circle', 'cv2.putText', 'cv2.rectangle', 'cv2.addWeighted'],
        'success': len(answers) > 0
    }
