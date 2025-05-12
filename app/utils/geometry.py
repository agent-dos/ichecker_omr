# app/utils/geometry.py (updated sections)
import cv2
import numpy as np
from app.utils.shape_validation import validate_polygon, validate_quadrilateral


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: (x, y) tuple
        polygon: List of (x, y) vertices

    Returns:
        bool: True if point is inside polygon
    """
    # Validate inputs
    polygon = validate_polygon(polygon)
    if polygon is None:
        return False

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_in_quadrilateral(point, quad):
    """
    Check if a point is inside a quadrilateral using OpenCV.

    Args:
        point: (x, y) tuple  
        quad: Array of four vertices defining the quadrilateral

    Returns:
        bool: True if point is inside quadrilateral
    """
    # Validate and normalize quad
    quad = validate_quadrilateral(quad)
    if quad is None:
        return False

    # Ensure point is proper format
    point = (float(point[0]), float(point[1]))

    # Use OpenCV's pointPolygonTest
    result = cv2.pointPolygonTest(quad.astype(np.int32), point, False)
    return result >= 0


def expand_polygon(polygon, expansion_pixels):
    """
    Expand a polygon outward by a specified number of pixels.

    Args:
        polygon: List of (x, y) vertices
        expansion_pixels: Number of pixels to expand

    Returns:
        numpy.ndarray: Expanded polygon vertices
    """
    polygon = validate_polygon(polygon)
    if polygon is None:
        return None

    # Calculate center
    center = np.mean(polygon, axis=0)

    # Expand each vertex away from center
    expanded = []
    for vertex in polygon:
        direction = vertex - center
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            new_vertex = vertex + direction * expansion_pixels
            expanded.append(new_vertex)
        else:
            expanded.append(vertex)

    return np.array(expanded, dtype=np.float32)


def filter_circles_outside_polygon(circles, polygon, margin=10):
    """
    Filter out circles that are inside or too close to a polygon.

    Args:
        circles: Array of circles [(x, y, radius), ...]
        polygon: List of (x, y) vertices
        margin: Additional margin around polygon

    Returns:
        numpy.ndarray: Filtered circles
    """
    if circles is None or polygon is None:
        return circles

    # Validate polygon
    polygon = validate_polygon(polygon)

    # Expand polygon by margin
    expanded_polygon = expand_polygon(polygon, margin)

    # Filter circles
    filtered = []
    for circle in circles:
        x, y, r = circle
        center = (x, y)

        # Check if circle center is outside the expanded polygon
        if not point_in_polygon(center, expanded_polygon):
            filtered.append(circle)

    return np.array(filtered) if filtered else None


def filter_circles_inside_quadrilateral(circles, quad, margin=0):
    """
    Filter circles to keep only those inside a quadrilateral.

    Args:
        circles: Array of circles [(x, y, radius), ...]
        quad: Array of four vertices defining the quadrilateral
        margin: Additional margin inside the quadrilateral

    Returns:
        numpy.ndarray: Filtered circles
    """
    if circles is None or quad is None:
        return circles

    # Validate quadrilateral
    quad = validate_quadrilateral(quad)

    # Apply margin if needed
    if margin > 0:
        # Shrink quadrilateral inward
        center = np.mean(quad, axis=0)
        shrunk_quad = []
        for vertex in quad:
            direction = vertex - center
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                new_vertex = vertex - direction * margin
                shrunk_quad.append(new_vertex)
            else:
                shrunk_quad.append(vertex)
        quad = np.array(shrunk_quad, dtype=np.float32)

    # Filter circles
    filtered = []
    for circle in circles:
        x, y, r = circle
        center = (float(x), float(y))

        # Check if circle center is inside the quadrilateral
        if point_in_quadrilateral(center, quad):
            filtered.append(circle)

    return np.array(filtered) if filtered else None
