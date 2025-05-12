# app/common/geometry/filters.py
"""
Geometric filtering utilities.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


def filter_by_polygon(
    points: np.ndarray,
    polygon: List[Tuple[int, int]],
    margin: int = 0
) -> Optional[np.ndarray]:
    """
    Filter points outside a polygon.
    """
    if points is None or polygon is None:
        return points

    # Convert to numpy array
    poly_np = np.array(polygon, dtype=np.int32)

    filtered = []
    for point in points:
        x, y = point[:2]

        # Check if outside polygon (with margin)
        dist = cv2.pointPolygonTest(poly_np, (x, y), True)
        if dist < -margin:
            filtered.append(point)

    return np.array(filtered) if filtered else None


def filter_by_quadrilateral(
    points: np.ndarray,
    quad: np.ndarray,
    margin: int = 0
) -> Optional[np.ndarray]:
    """
    Filter points inside a quadrilateral.
    """
    if points is None or quad is None:
        return points

    filtered = []
    for point in points:
        x, y = point[:2]

        # Check if inside quadrilateral
        dist = cv2.pointPolygonTest(quad.astype(np.int32), (x, y), True)
        if dist > margin:
            filtered.append(point)

    return np.array(filtered) if filtered else None
