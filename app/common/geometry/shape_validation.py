# app/common/geometry/validators.py
"""Shape validation from original utils/shape_validation.py"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_polygon(polygon):
    """Validate and normalize polygon format."""
    if polygon is None:
        return None

    poly_array = np.array(polygon)

    if len(poly_array.shape) == 3:
        poly_array = poly_array.squeeze(axis=1)
    elif len(poly_array.shape) == 1:
        if len(poly_array) % 2 == 0:
            poly_array = poly_array.reshape(-1, 2)
        else:
            raise ValueError("Invalid polygon: odd number of coordinates")

    if len(poly_array.shape) != 2 or poly_array.shape[1] != 2:
        raise ValueError(f"Invalid polygon shape: {poly_array.shape}")

    if len(poly_array) < 3:
        raise ValueError("Polygon must have at least 3 points")

    return poly_array.astype(np.float32)


def validate_quadrilateral(quad):
    """Validate quadrilateral has exactly 4 points."""
    if quad is None:
        return None

    quad_array = validate_polygon(quad)

    if len(quad_array) != 4:
        raise ValueError(
            f"Quadrilateral must have 4 points, got {len(quad_array)}")

    return quad_array
