# app/features/analyzer/visualizers/bubble_viz.py
"""
Bubble detection visualization.
"""
import cv2
import numpy as np
from typing import Optional

from app.core.constants import COLOR_GREEN, COLOR_BLUE, COLOR_RED


def visualize_bubbles(
    image: np.ndarray,
    bubbles: Optional[np.ndarray]
) -> np.ndarray:
    """
    Visualize detected bubbles on image.
    """
    viz = image.copy()

    if bubbles is None:
        return viz

    # Draw midpoint reference line
    midpoint_x = image.shape[1] // 2
    cv2.line(viz, (midpoint_x, 0), (midpoint_x, image.shape[0]), COLOR_RED, 1)

    # Draw bubbles with different colors for columns
    for x, y, r in bubbles:
        color = COLOR_GREEN if x < midpoint_x else COLOR_BLUE
        cv2.circle(viz, (x, y), r, color, 2)
        cv2.circle(viz, (x, y), 2, COLOR_RED, 3)

    return viz
