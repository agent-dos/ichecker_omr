# app/utils/rectification_visualization.py
import cv2
import numpy as np


def visualize_rectification(original, rectified, corners, angle):
    """
    Create visualization showing rectification results.

    Args:
        original: Original tilted image
        rectified: Rectified image
        corners: Detected corners
        angle: Rotation angle

    Returns:
        numpy.ndarray: Visualization image
    """
    # Resize images for side-by-side display
    h1, w1 = original.shape[:2]
    h2, w2 = rectified.shape[:2]

    # Scale to common height
    target_height = 600
    scale1 = target_height / h1
    scale2 = target_height / h2

    resized1 = cv2.resize(original, (int(w1 * scale1), target_height))
    resized2 = cv2.resize(rectified, (int(w2 * scale2), target_height))

    # Create side-by-side visualization
    total_width = resized1.shape[1] + resized2.shape[1] + 20
    viz = np.zeros((target_height, total_width, 3), dtype=np.uint8)

    # Place images
    viz[:, :resized1.shape[1]] = resized1
    viz[:, resized1.shape[1] + 20:] = resized2

    # Draw corner markers on original
    if corners:
        for corner_name, corner in corners.items():
            if corner:
                x, y = corner['center']
                x = int(x * scale1)
                y = int(y * scale1)
                cv2.circle(viz, (x, y), 5, (0, 255, 0), -1)

    # Add labels
    cv2.putText(viz, f"Original (Angle: {angle:.1f}°)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(viz, "Rectified",
                (resized1.shape[1] + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    return viz
