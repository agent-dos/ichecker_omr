# app/utils/image_rectification.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def rectify_image_with_corners(image, corners):
    """
    Rectify a tilted answer sheet using detected corner markers.

    Args:
        image: Input image
        corners: Dictionary with corner positions

    Returns:
        tuple: (rectified_image, transformation_matrix)
    """
    if image is None or corners is None:
        logger.warning("Invalid input for rectification")
        return image, None

    # Extract corner points in clockwise order
    src_points = _extract_corner_points(corners)
    if src_points is None:
        logger.warning("Could not extract corner points")
        return image, None

    # Calculate destination points for a perfect rectangle
    dst_points = _calculate_destination_points(src_points, image.shape)

    # Compute perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(
        src_points.astype(np.float32),
        dst_points.astype(np.float32)
    )

    # Apply perspective transformation
    height, width = _calculate_output_dimensions(dst_points)
    rectified = cv2.warpPerspective(
        image, transform_matrix, (width, height),
        flags=cv2.INTER_LINEAR
    )

    return rectified, transform_matrix


def _extract_corner_points(corners):
    """
    Extract corner points in clockwise order: TL, TR, BR, BL.

    Args:
        corners: Dictionary with corner positions

    Returns:
        numpy.ndarray: Array of corner points
    """
    required_corners = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

    points = []
    for corner_name in required_corners:
        if corner_name not in corners or corners[corner_name] is None:
            return None
        points.append(corners[corner_name]['center'])

    return np.array(points, dtype=np.float32)


def _calculate_destination_points(src_points, image_shape):
    """
    Calculate destination points for a perfect rectangle.

    Args:
        src_points: Source corner points
        image_shape: Original image shape

    Returns:
        numpy.ndarray: Destination points for rectified image
    """
    # Calculate bounding box of source points
    x_coords = src_points[:, 0]
    y_coords = src_points[:, 1]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Create destination rectangle with same aspect ratio
    width = int(max_x - min_x)
    height = int(max_y - min_y)

    # Add margin to avoid edge cropping
    margin = 20
    dst_points = np.array([
        [margin, margin],  # Top-left
        [width - margin, margin],  # Top-right
        [width - margin, height - margin],  # Bottom-right
        [margin, height - margin]  # Bottom-left
    ], dtype=np.float32)

    return dst_points


def _calculate_output_dimensions(dst_points):
    """
    Calculate output dimensions for the rectified image.

    Args:
        dst_points: Destination corner points

    Returns:
        tuple: (height, width)
    """
    width = int(np.max(dst_points[:, 0]) - np.min(dst_points[:, 0]))
    height = int(np.max(dst_points[:, 1]) - np.min(dst_points[:, 1]))

    # Ensure minimum dimensions
    width = max(width, 400)
    height = max(height, 600)

    return height, width


def calculate_rotation_angle(corners):
    """
    Calculate the rotation angle of the answer sheet.

    Args:
        corners: Dictionary with corner positions

    Returns:
        float: Rotation angle in degrees
    """
    if not corners or 'top_left' not in corners or 'top_right' not in corners:
        return 0.0

    # Calculate angle from top edge
    tl = corners['top_left']['center']
    tr = corners['top_right']['center']

    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]

    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def is_rectification_needed(corners, threshold=5.0):
    """
    Determine if image rectification is needed based on rotation angle.

    Args:
        corners: Dictionary with corner positions
        threshold: Angle threshold in degrees

    Returns:
        bool: True if rectification is needed
    """
    angle = calculate_rotation_angle(corners)
    return abs(angle) > threshold


def apply_inverse_transform(points, transform_matrix):
    """
    Apply inverse transformation to map points back to original image.

    Args:
        points: Points in rectified image
        transform_matrix: Perspective transformation matrix

    Returns:
        numpy.ndarray: Points in original image coordinates
    """
    if transform_matrix is None:
        return points

    # Calculate inverse transformation
    inv_matrix = np.linalg.inv(transform_matrix)

    # Convert points to homogeneous coordinates
    num_points = len(points)
    homogeneous = np.ones((num_points, 3))
    homogeneous[:, :2] = points

    # Apply inverse transformation
    transformed = homogeneous @ inv_matrix.T

    # Convert back to 2D coordinates
    transformed_2d = transformed[:, :2] / transformed[:, 2:3]

    return transformed_2d.astype(int)
