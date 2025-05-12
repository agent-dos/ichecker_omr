# app/features/analyzer/processors/image_processor.py
"""Image processing utilities for analyzer."""
import cv2
import numpy as np


class ImageProcessor:
    """Handles image preprocessing for analysis."""

    @staticmethod
    def resize_if_needed(
        image: np.ndarray,
        resize_factor: float
    ) -> np.ndarray:
        """Resize image if factor is not 1.0."""
        if resize_factor == 1.0:
            return image.copy()

        height, width = image.shape[:2]
        new_dimensions = (
            int(width * resize_factor),
            int(height * resize_factor)
        )
        return cv2.resize(
            image, new_dimensions,
            interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def apply_preprocessing(
        image: np.ndarray,
        blur_kernel: int = 5
    ) -> np.ndarray:
        """Apply preprocessing steps."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        if blur_kernel > 0:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        return gray
