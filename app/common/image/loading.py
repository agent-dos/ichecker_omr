# app/common/image/loading.py
"""
Image loading utilities.
"""
import cv2
import numpy as np
from PIL import Image
import io


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Load image from Streamlit uploaded file.
    """
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    
    if image is None:
        raise ValueError("Failed to load image")
    
    # Ensure BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to OpenCV format.
    """
    np_image = np.array(pil_image)
    
    # Convert RGB to BGR
    if len(np_image.shape) == 3:
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    
    return np_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL format.
    """
    # Convert BGR to RGB
    if len(cv2_image.shape) == 3:
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    return Image.fromarray(cv2_image)