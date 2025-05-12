# app/utils/image_processing.py
import cv2
import numpy as np
from PIL import Image
import io


def load_and_preprocess_image(uploaded_file):
    """
    Load and preprocess the uploaded image.

    Args:
            uploaded_file: The uploaded file from Streamlit

    Returns:
            numpy.ndarray: The preprocessed image
    """
    # Read the uploaded image
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Check if image was loaded successfully
    if image is None:
        raise ValueError("Failed to load image. Please try a different file.")

    # Return the image
    return image


def convert_to_grayscale(image):
    """
    Convert image to grayscale if it's not already.

    Args:
            image: The input image

    Returns:
            numpy.ndarray: Grayscale image
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    elif image.shape[2] == 4:  # BGRA
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:  # BGR
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_adaptive_threshold(gray_image, block_size=31, c_value=10):
    """
    Apply adaptive thresholding to the grayscale image.

    Args:
            gray_image: Grayscale image
            block_size: Size of a pixel neighborhood for thresholding
            c_value: Constant subtracted from the mean

    Returns:
            numpy.ndarray: Binary image after thresholding
    """
    # Ensure block_size is odd and greater than 1
    block_size = block_size if block_size % 2 != 0 else block_size + 1
    block_size = max(3, block_size)

    # Apply adaptive thresholding
    return cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )


def detect_circles(gray_image, param1=50, param2=18, min_radius=10, max_radius=20):
    """
    Detect circles in the grayscale image.

    Args:
            gray_image: Grayscale image
            param1: Parameter for edge detection
            param2: Parameter for circle detection
            min_radius: Minimum radius of the circles
            max_radius: Maximum radius of the circles

    Returns:
            numpy.ndarray or None: Detected circles or None if no circles are detected
    """
    # Apply HoughCircles to detect circles
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Return detected circles or None
    if circles is not None:
        return np.round(circles[0, :]).astype(int)
    else:
        return None

def load_and_preprocess_image(uploaded_file):
	"""
	Load and preprocess the uploaded image.
	
	Args:
		uploaded_file: The uploaded file from Streamlit
		
	Returns:
		numpy.ndarray: The preprocessed image in BGR format
	"""
	# Read the uploaded image
	image_bytes = uploaded_file.read()
	image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
	
	# Check if image was loaded successfully
	if image is None:
		raise ValueError("Failed to load image. Please try a different file.")
	
	# Ensure we have a 3-channel BGR image
	if len(image.shape) == 2:  # Grayscale
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	elif image.shape[2] == 4:  # BGRA
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	
	# Return the image
	return image

def convert_to_grayscale(image):
	"""
	Convert image to grayscale if it's not already.
	
	Args:
		image: The input image
		
	Returns:
		numpy.ndarray: Grayscale image
	"""
	if len(image.shape) == 2:
		return image  # Already grayscale
	elif image.shape[2] == 4:  # BGRA
		return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
	else:  # BGR
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
