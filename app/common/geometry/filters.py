# filename: app/common/geometry/filters.py
# app/common/geometry/filters.py
"""
Geometric filtering utilities.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging  # Add logging

# Initialize logger at the module level
logger = logging.getLogger(__name__)


def filter_by_polygon(
    points: np.ndarray,
    polygon: List[Tuple[int, int]],
    margin: int = 0
) -> Optional[np.ndarray]:
    """
    Filter points *outside* a polygon, keeping points whose distance
    to the polygon is less than -margin.
    """
    # Input validation
    if points is None or polygon is None:
        logger.debug("filter_by_polygon: Received None for points or polygon.")
        return points
    if not isinstance(points, np.ndarray) or points.size == 0:
        logger.debug(
            f"filter_by_polygon: Received invalid or empty points array (shape: {points.shape if isinstance(points, np.ndarray) else type(points)}).")
        return np.array([])  # Return empty array for consistency

    # Validate and convert polygon
    try:
        poly_np = np.array(polygon, dtype=np.int32)
        if len(poly_np.shape) != 2 or poly_np.shape[1] != 2:
            logger.error(
                f"Invalid polygon shape for filtering: {poly_np.shape}")
            return points  # Return original points if polygon is invalid
    except Exception as e:
        logger.error(f"Error converting polygon to numpy array: {e}")
        return points

    filtered = []
    # Ensure points array has the expected structure (at least 2 columns)
    if points.shape[1] < 2:
        logger.error(
            f"Points array has insufficient columns ({points.shape[1]}) for coordinate extraction.")
        return np.array([])

    for i, point in enumerate(points):
        try:
            # Extract and explicitly convert coordinates to float
            x = float(point[0])
            y = float(point[1])
            point_tuple = (x, y)  # Create the tuple for the function

            # Check distance: keep points strictly outside the margin
            dist = cv2.pointPolygonTest(poly_np, point_tuple, True)
            if dist < -margin:
                filtered.append(point)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Skipping point due to type error at index {i}: {point}. Error: {e}")
            continue  # Skip this point if conversion fails
        except IndexError:
            logger.warning(
                f"Skipping point due to insufficient elements at index {i}: {point}")
            continue  # Skip if point doesn't have x, y
        except Exception as e:
            logger.error(
                f"Unexpected error processing point {point} at index {i}: {e}")
            continue

    # Return an array, even if empty, maintaining original data type if possible
    if filtered:
        return np.array(filtered, dtype=points.dtype)
    else:
        # Return empty array with original number of columns and dtype
        return np.empty((0, points.shape[1]), dtype=points.dtype)


def filter_by_quadrilateral(
    points: np.ndarray,
    quad: np.ndarray,
    margin: int = 0
) -> Optional[np.ndarray]:
    """
    Filter points *inside* a quadrilateral, keeping points whose distance
    to the quadrilateral is greater than margin.
    """
    # Input validation
    if points is None or quad is None:
        logger.debug(
            "filter_by_quadrilateral: Received None for points or quad.")
        return points
    if not isinstance(points, np.ndarray) or points.size == 0:
        logger.debug(
            f"filter_by_quadrilateral: Received invalid or empty points array (shape: {points.shape if isinstance(points, np.ndarray) else type(points)}).")
        return np.array([])  # Return empty array for consistency
    if not isinstance(quad, np.ndarray):
        logger.error(
            f"filter_by_quadrilateral: Received non-numpy array for quad: {type(quad)}")
        return points

    # Validate and convert quadrilateral
    try:
        # Ensure float first for calculations, then int32 for pointPolygonTest
        if quad.dtype != np.float32:
            quad_float = quad.astype(np.float32)
        else:
            quad_float = quad

        # Convert to int32 for the function
        quad_np = quad_float.astype(np.int32)

        if len(quad_np.shape) != 2 or quad_np.shape[1] != 2 or len(quad_np) != 4:
            logger.error(
                f"Invalid quadrilateral shape for filtering: {quad_np.shape}")
            return points  # Return original points if quad is invalid
    except Exception as e:
        logger.error(f"Error processing quadrilateral array: {e}")
        return points

    filtered = []
    # Ensure points array has the expected structure (at least 2 columns)
    if points.shape[1] < 2:
        logger.error(
            f"Points array has insufficient columns ({points.shape[1]}) for coordinate extraction.")
        return np.array([])

    for i, point in enumerate(points):
        try:
            # Extract and explicitly convert coordinates to float
            x = float(point[0])
            y = float(point[1])
            point_tuple = (x, y)  # Create the tuple for the function

            # Check distance: keep points inside the margin
            dist = cv2.pointPolygonTest(quad_np, point_tuple, True)
            if dist > margin:
                filtered.append(point)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Skipping point due to type error at index {i}: {point}. Error: {e}")
            continue  # Skip this point if conversion fails
        except IndexError:
            logger.warning(
                f"Skipping point due to insufficient elements at index {i}: {point}")
            continue  # Skip if point doesn't have x, y
        except Exception as e:
            logger.error(
                f"Unexpected error processing point {point} at index {i}: {e}")
            continue

    # Return an array, even if empty, maintaining original data type if possible
    if filtered:
        return np.array(filtered, dtype=points.dtype)
    else:
        # Return empty array with original number of columns and dtype
        return np.empty((0, points.shape[1]), dtype=points.dtype)
