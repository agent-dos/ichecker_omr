# app/common/image/conversion.py
"""
Image format conversion utilities.
"""
import cv2
import numpy as np


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in BGR format.
    """
    if image is None:
        return None

    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # Already BGR
            return image
        elif image.shape[2] == 4:  # BGRA
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unexpected image shape: {image.shape}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    """
    if len(image.shape) == 2:
        return image
    elif image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unexpected image shape: {image.shape}")
