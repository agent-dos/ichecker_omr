# app/services/qr_service.py
import cv2
import numpy as np
from app.utils.detect_qr_code import detect_qr_code_with_info


def process_qr_code(image):
    """
    Process the image to detect QR codes.

    Args:
            image: The input image

    Returns:
            dict: QR code detection results
    """
    # Detect QR code
    qr_data, qr_info = detect_qr_code_with_info(image)

    # Visualize the QR code detection
    viz = image.copy()

    if qr_data and qr_info:
        pts = np.array(qr_info['polygon'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=3)
        cv2.putText(
            viz, qr_data,
            (qr_info['rect']['left'], qr_info['rect']['top'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
    else:
        cv2.putText(viz, "No QR Code Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Return the results with function information
    return {
        'qr_data': qr_data,
        'qr_info': qr_info,
        'visualization': viz,
        'functions': [
            'cv2.cvtColor',
            'pyzbar.decode',
            'cv2.adaptiveThreshold',
            'cv2.GaussianBlur',
            'cv2.equalizeHist',
            'cv2.addWeighted',
            'cv2.polylines',
            'cv2.putText'
        ]
    }
