# filename: app/common/image/display.py
import cv2
import numpy as np
import streamlit as st
import logging  # Add logging

logger = logging.getLogger(__name__)


def display_image(
    image: np.ndarray,
    # use_column_width parameter is deprecated
    use_container_width: bool = True  # Use new parameter name
) -> None:
    """
    Display image in Streamlit with proper format handling.
    Uses use_container_width=True by default.
    """
    if image is None:
        st.warning("No image available to display.")  # Changed from info
        return
    if not isinstance(image, np.ndarray):
        st.error(f"Invalid image type for display: {type(image)}")
        logger.error(f"Attempted to display non-numpy array: {type(image)}")
        return

    channels = "BGR"  # Default assumption
    img_to_display = image

    try:
        # Handle different channel configurations
        if len(image.shape) == 2:  # Grayscale
            channels = "GRAY"
            img_to_display = image
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR
                channels = "BGR"
                img_to_display = image
            elif image.shape[2] == 4:  # BGRA -> Convert to BGR
                logger.debug("Converting BGRA image to BGR for display.")
                img_to_display = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                channels = "BGR"
            else:
                st.error(
                    f"Unsupported number of image channels: {image.shape[2]}")
                logger.error(f"Unsupported image channels: {image.shape}")
                return
        else:
            st.error(f"Unexpected image dimensions: {image.shape}")
            logger.error(f"Unexpected image shape: {image.shape}")
            return

        # Display the image using the CORRECT parameter name
        st.image(
            img_to_display,
            caption=None,  # Add caption if needed later
            width=None,  # Let container width control it
            output_format='auto',
            channels=channels,
            # --- Use the new parameter ---
            use_column_width='auto',  # Recommended setting for container width behavior
            clamp=False,
            use_container_width=use_container_width
        )
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        logger.exception("Failed during st.image call.")
