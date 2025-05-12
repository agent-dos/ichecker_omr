# app/utils/qr_visualization.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def visualize_qr_code(image, qr_data, qr_info):
    """
    Visualize QR code detection results on the image.

    Args:
            image: Original image
            qr_data: Detected QR code data string
            qr_info: QR code information dictionary

    Returns:
            numpy.ndarray: Image with QR code visualization
    """
    if image is None:
        logger.warning("visualize_qr_code: Input image is None.")
        return None

    # Create a copy to avoid modifying the original
    viz = image.copy()

    if qr_data and qr_info:
        # Draw detected QR code
        _draw_qr_detection(viz, qr_data, qr_info)
    else:
        # Draw no detection message
        _draw_no_detection_message(viz)

    return viz


def _draw_qr_detection(viz, qr_data, qr_info):
    """
    Draw QR code detection results including polygon and text.

    Args:
            viz: Image to draw on
            qr_data: QR code data string
            qr_info: QR code information dictionary
    """
    # Draw polygon around QR code
    if qr_info.get('polygon'):
        pts = np.array(qr_info['polygon'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=3)

    # Draw rectangle if polygon not available
    elif qr_info.get('rect'):
        rect = qr_info['rect']
        cv2.rectangle(
            viz,
            (rect['left'], rect['top']),
            (rect['left'] + rect['width'], rect['top'] + rect['height']),
            (0, 255, 0),
            3
        )

    # Draw QR code data text
    if qr_info.get('rect'):
        text_position = (qr_info['rect']['left'], qr_info['rect']['top'] - 10)
        _draw_text_with_background(viz, qr_data, text_position)

    # Draw center point
    if qr_info.get('center'):
        cv2.circle(viz, qr_info['center'], 5, (0, 255, 0), -1)


def _draw_no_detection_message(viz):
    """
    Draw message indicating no QR code was detected.

    Args:
            viz: Image to draw on
    """
    text = "No QR Code Detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Calculate text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate centered position
    text_x = (viz.shape[1] - text_size[0]) // 2
    text_y = 50

    # Draw text with background
    _draw_text_with_background(
        viz, text, (text_x, text_y),
        text_color=(0, 0, 255), bg_color=(255, 255, 255)
    )


def _draw_text_with_background(image, text, position, text_color=(0, 255, 0),
                               bg_color=(255, 255, 255), font_scale=0.8):
    """
    Draw text with a background rectangle for better visibility.

    Args:
            image: Image to draw on
            text: Text to display
            position: (x, y) position for text
            text_color: Color of the text
            bg_color: Background color
            font_scale: Font size scale
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # Calculate background rectangle coordinates
    x, y = position
    padding = 5
    rect_x1 = x - padding
    rect_y1 = y - text_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + padding

    # Draw background rectangle
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
