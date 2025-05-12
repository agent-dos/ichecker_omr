# app/features/analyzer/visualizers/debug_viz.py
"""
Debug visualization for analyzer.
"""
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

from app.core.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW


def visualize_debug(
    image: np.ndarray,
    debug_data: Dict
) -> np.ndarray:
    """
    Create debug visualization.
    """
    viz = image.copy()

    # Draw different detection stages
    if 'raw_circles' in debug_data:
        _draw_circles(viz, debug_data['raw_circles'], COLOR_GREEN)

    if 'filtered_circles' in debug_data:
        _draw_circles(viz, debug_data['filtered_circles'], COLOR_YELLOW)

    if 'excluded_circles' in debug_data:
        _draw_excluded_circles(viz, debug_data['excluded_circles'])

    # Draw exclusion zones
    if 'qr_polygon' in debug_data:
        _draw_polygon(viz, debug_data['qr_polygon'], COLOR_RED)

    if 'answer_boundary' in debug_data:
        _draw_polygon(viz, debug_data['answer_boundary'], COLOR_GREEN)

    # Add statistics
    if 'stats' in debug_data:
        _draw_stats(viz, debug_data['stats'])

    return viz


def _draw_circles(
    viz: np.ndarray,
    circles: Optional[np.ndarray],
    color: tuple
) -> None:
    """
    Draw circles with given color.
    """
    if circles is None:
        return

    for x, y, r in circles:
        cv2.circle(viz, (x, y), r, color, 2)


def _draw_excluded_circles(
    viz: np.ndarray,
    circles: Optional[np.ndarray]
) -> None:
    """
    Draw excluded circles with cross marks.
    """
    if circles is None:
        return

    for x, y, r in circles:
        cv2.circle(viz, (x, y), r, COLOR_RED, 2)
        cv2.line(viz, (x-r, y), (x+r, y), COLOR_RED, 2)
        cv2.line(viz, (x, y-r), (x, y+r), COLOR_RED, 2)


def _draw_polygon(
    viz: np.ndarray,
    polygon: List[Tuple[int, int]],
    color: tuple
) -> None:
    """
    Draw polygon boundary.
    """
    if not polygon:
        return

    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(viz, [pts], True, color, 2)


def _draw_stats(
    viz: np.ndarray,
    stats: Dict
) -> None:
    """
    Draw statistics overlay.
    """
    y_pos = 30
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(viz, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
