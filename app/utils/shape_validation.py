# app/utils/shape_validation.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def ensure_compatible_shapes(array1, array2, operation_name="operation"):
    """
    Ensure two arrays have compatible shapes for operations.

    Args:
        array1: First numpy array
        array2: Second numpy array
        operation_name: Name of operation for logging

    Returns:
        tuple: (array1, array2) with compatible shapes
    """
    shape1 = array1.shape
    shape2 = array2.shape

    if shape1 == shape2:
        return array1, array2

    logger.warning(f"{operation_name}: Shape mismatch {shape1} vs {shape2}")

    # Handle 2D point arrays
    if len(shape1) == 3 and len(shape2) == 3:
        # Arrays of points (n, 1, 2) or (n, m, 2)
        if shape1[1:] == shape2[1:]:
            # Same point dimensions, different counts
            min_points = min(shape1[0], shape2[0])
            logger.info(f"Truncating to {min_points} points")
            return array1[:min_points], array2[:min_points]

    # Handle 1D arrays
    if len(shape1) == 1 and len(shape2) == 1:
        min_length = min(len(array1), len(array2))
        return array1[:min_length], array2[:min_length]

    # Try to broadcast
    try:
        result1 = np.broadcast_to(array1, shape2)
        return result1, array2
    except ValueError:
        try:
            result2 = np.broadcast_to(array2, shape1)
            return array1, result2
        except ValueError:
            raise ValueError(
                f"Cannot make shapes {shape1} and {shape2} compatible")


def validate_polygon(polygon):
    """
    Validate and normalize polygon format.

    Args:
        polygon: Polygon points (list or array)

    Returns:
        numpy.ndarray: Validated polygon as (n, 2) array
    """
    if polygon is None:
        return None

    # Convert to numpy array
    poly_array = np.array(polygon)

    # Handle different input formats
    if len(poly_array.shape) == 3:
        # Shape (n, 1, 2) -> (n, 2)
        poly_array = poly_array.squeeze(axis=1)
    elif len(poly_array.shape) == 1:
        # Flat array -> reshape
        if len(poly_array) % 2 == 0:
            poly_array = poly_array.reshape(-1, 2)
        else:
            raise ValueError("Invalid polygon: odd number of coordinates")

    # Ensure 2D array of shape (n, 2)
    if len(poly_array.shape) != 2 or poly_array.shape[1] != 2:
        raise ValueError(f"Invalid polygon shape: {poly_array.shape}")

    # Ensure at least 3 points
    if len(poly_array) < 3:
        raise ValueError("Polygon must have at least 3 points")

    return poly_array.astype(np.float32)


def validate_quadrilateral(quad):
    """
    Validate quadrilateral has exactly 4 points.

    Args:
        quad: Quadrilateral points

    Returns:
        numpy.ndarray: Validated quadrilateral as (4, 2) array
    """
    if quad is None:
        return None

    quad_array = validate_polygon(quad)

    if len(quad_array) != 4:
        raise ValueError(
            f"Quadrilateral must have exactly 4 points, got {len(quad_array)}")

    return quad_array
