# app/features/qr/visualizer.py
"""
QR code visualization.
"""
import cv2
import numpy as np
from typing import Optional, Dict

from app.core.constants import COLOR_GREEN, COLOR_RED


def visualize_qr(
    image: np.ndarray,
    qr_data: Optional[str],
    qr_info: Optional[Dict]
) -> np.ndarray:
    """
    Visualize QR detection results.
    """
    viz = image.copy()

    if qr_data and qr_info:
        # Draw polygon
        if 'polygon' in qr_info:
            pts = np.array(qr_info['polygon'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [pts], True, COLOR_GREEN, 3)

        # Draw data text
        if 'rect' in qr_info:
            x = qr_info['rect']['left']
            y = qr_info['rect']['top'] - 10
            cv2.putText(viz, qr_data, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
    else:
        # No QR detected
        cv2.putText(viz, "No QR Code Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)

    return viz
