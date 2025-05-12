# app/utils/circle_detection.py
import cv2
import numpy as np
import logging
from app.utils.geometry import filter_circles_outside_polygon, filter_circles_inside_quadrilateral

logger = logging.getLogger(__name__)


def detect_circles_with_filters(image, param1, param2, min_radius, max_radius,
								resize_factor, qr_polygon, answer_boundary):
	"""
	Detect circular bubbles with QR exclusion and boundary filtering.
	
	Args:
		image: Input image
		param1: HoughCircles parameter
		param2: HoughCircles parameter
		min_radius: Minimum bubble radius
		max_radius: Maximum bubble radius
		resize_factor: Image resize factor
		qr_polygon: QR code polygon to exclude
		answer_boundary: Quadrilateral boundary for valid answers
		
	Returns:
		numpy.ndarray: Detected and filtered circles
	"""
	# Convert to grayscale
	gray = _convert_to_grayscale(image)
	
	# Apply slight Gaussian blur to improve circle detection
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	
	# Apply resize factor to circle detection parameters
	scaled_min_radius = int(min_radius * resize_factor)
	scaled_max_radius = int(max_radius * resize_factor)
	scaled_min_dist = int(20 * resize_factor)
	
	# Detect circles using Hough transform
	circles = cv2.HoughCircles(
		gray,
		cv2.HOUGH_GRADIENT,
		dp=1,
		minDist=scaled_min_dist,
		param1=param1,
		param2=param2,
		minRadius=scaled_min_radius,
		maxRadius=scaled_max_radius
	)
	
	if circles is None:
		return None
	
	# Round and convert to integer
	circles = np.round(circles[0, :]).astype(int)
	
	# Apply boundary filter first (if available)
	if answer_boundary is not None:
		circles = filter_circles_inside_quadrilateral(
			circles,
			answer_boundary,
			margin=5  # Small margin for safety
		)
		
		if circles is None:
			return None
	
	# Then filter out circles in QR region
	if qr_polygon is not None:
		# Scale QR polygon if image was resized
		if resize_factor != 1.0:
			scaled_polygon = [(int(x * resize_factor), int(y * resize_factor))
							for x, y in qr_polygon]
		else:
			scaled_polygon = qr_polygon
		
		# Filter circles outside QR region
		circles = filter_circles_outside_polygon(
			circles,
			scaled_polygon,
			margin=scaled_max_radius + 5
		)
	
	return circles


def _convert_to_grayscale(image):
	"""Convert image to grayscale handling different formats."""
	if len(image.shape) == 2:
		return image
	elif image.shape[2] == 4:
		return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
	else:
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)