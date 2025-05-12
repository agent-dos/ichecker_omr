# app/utils/image_display.py
import cv2
import numpy as np
import streamlit as st


def safe_display_image(image, use_column_width=True):
    """
    Safely display images in Streamlit regardless of channel count.

    Args:
            image: Image as numpy array
            use_column_width: Whether to expand image to column width
    """
    if image is None:
        st.info("No image available")
        return

    # Check number of channels
    if len(image.shape) == 2:  # Grayscale (1 channel)
        st.image(image, channels="GRAY", use_column_width=use_column_width)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # BGR (3 channels)
            st.image(image, channels="BGR", use_column_width=use_column_width)
        elif image.shape[2] == 4:  # BGRA (4 channels)
            # Convert BGRA to BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            st.image(bgr_image, channels="BGR",
                     use_column_width=use_column_width)
        else:
            # Unexpected number of channels, convert to RGB to be safe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(rgb_image, use_column_width=use_column_width)
    else:
        st.error(f"Unexpected image shape: {image.shape}")


def ensure_bgr(image):
    """
    Ensure image is in BGR format regardless of original channels.

    Args:
            image: Input image as numpy array

    Returns:
            numpy.ndarray: Image in BGR format
    """
    if image is None:
        return None

    # Check number of channels
    if len(image.shape) == 2:  # Grayscale (1 channel)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # Already BGR (3 channels)
            return image
        elif image.shape[2] == 4:  # BGRA (4 channels)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            # Unexpected number of channels, assume it's another color space and convert to BGR
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
