# app/features/corners/visualizer.py
"""
Corner detection visualization utilities.
"""
import cv2
import numpy as np
from typing import Dict, Optional

from app.core.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW


def visualize_corners(
    image: np.ndarray,
    corners: Optional[Dict]
) -> np.ndarray:
    """
    Visualize detected corner markers.

    Args:
        image: Input image
        corners: Dictionary of detected corners

    Returns:
        Visualization image
    """
    viz = image.copy()

    if corners is None:
        _draw_no_corners_found(viz)
        return viz

    # Draw each corner
    for corner_name, corner_data in corners.items():
        if corner_data is None:
            continue

        _draw_corner_marker(viz, corner_name, corner_data)

    # Draw quadrilateral if all corners found
    if _all_corners_detected(corners):
        _draw_quadrilateral(viz, corners)

    return viz


def _draw_corner_marker(
    viz: np.ndarray,
    corner_name: str,
    corner_data: Dict
) -> None:
    """Draw individual corner marker."""
    center = corner_data['center']
    x, y = center

    # Draw marker
    cv2.circle(viz, (x, y), 5, COLOR_GREEN, -1)
    cv2.circle(viz, (x, y), 10, COLOR_GREEN, 2)

    # Draw corner lines
    line_length = 30
    directions = _get_corner_directions(corner_name)

    for dx, dy in directions:
        end_x = x + dx * line_length
        end_y = y + dy * line_length
        cv2.line(viz, (x, y), (end_x, end_y), COLOR_GREEN, 3)

    # Add label
    label_pos = _get_label_position(corner_name, x, y)
    cv2.putText(
        viz, corner_name.replace('_', ' ').title(),
        label_pos, cv2.FONT_HERSHEY_SIMPLEX,
        0.6, COLOR_YELLOW, 2
    )


def _draw_quadrilateral(
    viz: np.ndarray,
    corners: Dict
) -> None:
    """Draw boundary quadrilateral."""
    points = []
    for name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
        if name in corners and corners[name]:
            points.append(corners[name]['center'])

    if len(points) == 4:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], True, COLOR_YELLOW, 2)


def _draw_no_corners_found(viz: np.ndarray) -> None:
    """Draw message when no corners found."""
    cv2.putText(
        viz, "No corners detected",
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, COLOR_RED, 2
    )


def _all_corners_detected(corners: Dict) -> bool:
    """Check if all corners were detected."""
    required = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    return all(
        corners.get(name) is not None
        for name in required
    )


def _get_corner_directions(corner_name: str) -> list:
    """Get line directions for corner marker."""
    if corner_name == 'top_left':
        return [(1, 0), (0, 1)]
    elif corner_name == 'top_right':
        return [(-1, 0), (0, 1)]
    elif corner_name == 'bottom_left':
        return [(1, 0), (0, -1)]
    elif corner_name == 'bottom_right':
        return [(-1, 0), (0, -1)]
    return []


def _get_label_position(corner_name: str, x: int, y: int) -> tuple:
    """Calculate label position for corner."""
    offset = 40

    if corner_name == 'top_left':
        return (x + offset, y - offset//2)
    elif corner_name == 'top_right':
        return (x - offset*2, y - offset//2)
    elif corner_name == 'bottom_left':
        return (x + offset, y + offset)
    elif corner_name == 'bottom_right':
        return (x - offset*2, y + offset)

    return (x, y - offset)
