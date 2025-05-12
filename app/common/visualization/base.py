# app/common/visualization/base.py
"""
Base visualization utilities shared across features.
Currently unused - feature-specific visualizers are in their respective modules.
"""
import cv2
import numpy as np
from typing import Tuple


def get_color_by_name(color_name: str) -> Tuple[int, int, int]:
    """
    Get BGR color tuple by name.
    """
    colors = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    return colors.get(color_name, (128, 128, 128))


def add_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Add text with background for better visibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Draw background rectangle
    x, y = position
    cv2.rectangle(
        image,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        bg_color,
        -1
    )

    # Draw text
    cv2.putText(
        image, text, position,
        font, font_scale, color, thickness
    )
