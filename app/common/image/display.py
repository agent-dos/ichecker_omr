# app/common/image/display.py
"""
Image display utilities for Streamlit.
"""
import cv2
import numpy as np
import streamlit as st


def display_image(
    image: np.ndarray,
    use_column_width: bool = True
) -> None:
    """
    Display image in Streamlit with proper format handling.
    """
    if image is None:
        st.info("No image available")
        return

    # Handle different channel configurations
    if len(image.shape) == 2:  # Grayscale
        st.image(image, channels="GRAY", use_column_width=use_column_width)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # BGR
            st.image(image, channels="BGR", use_column_width=use_column_width)
        elif image.shape[2] == 4:  # BGRA
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            st.image(bgr, channels="BGR", use_column_width=use_column_width)
    else:
        st.error(f"Unexpected image shape: {image.shape}")
