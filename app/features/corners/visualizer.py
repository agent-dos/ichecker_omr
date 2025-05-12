# filename: app/features/corners/visualizer.py
import logging
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple  # Added List, Tuple

# Assuming these constants are correctly defined in your constants module
try:
    from app.core.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE, COLOR_BLACK
except ImportError:
    # Fallback colors if constants cannot be imported
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)  # For text if needed
    COLOR_BLACK = (0, 0, 0)  # For text background


def visualize_corners(
    image: np.ndarray,
    corners: Optional[Dict],
    message: Optional[str] = None  # NEW: Optional message parameter
) -> np.ndarray:
    """
    Visualize detected corner markers.

    Args:
        image: Input image (BGR or Grayscale, will be copied and converted to BGR if needed).
        corners: Dictionary of detected corners, e.g., {'top_left': {'center': (x,y)}, ...}.
                 Can be None.
        message: Optional message/title to display on the top-left of the image.

    Returns:
        Visualization image (BGR format).
    """
    # Ensure working with a BGR copy for consistent drawing
    if image is None:
        # Create a dummy image if input is None, to show the message
        logger.warning(
            "visualize_corners received a None image. Creating a placeholder.")
        viz = np.zeros((200, 300, 3), dtype=np.uint8)  # Small black image
        cv2.putText(viz, "Input Image Missing", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
    elif len(image.shape) == 2:  # Grayscale
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # BGRA
        viz = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 3:  # BGR
        viz = image.copy()
    else:
        logger.error(
            f"Unsupported image shape for visualize_corners: {image.shape}")
        # Create a dummy error image
        viz = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(viz, "Invalid Image Input", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)

    if corners is None or not isinstance(corners, dict) or not _all_corners_detected_and_valid(corners):
        _draw_no_corners_found(viz)  # This func expects BGR
    else:
        # Draw each corner
        for corner_name, corner_data in corners.items():
            # Check if corner_data is valid (dict with 'center')
            if isinstance(corner_data, dict) and 'center' in corner_data and \
               isinstance(corner_data['center'], (tuple, list)) and len(corner_data['center']) == 2:
                try:
                    # Ensure center is a tuple of ints for drawing
                    center = tuple(map(int, corner_data['center']))
                    _draw_corner_marker(viz, corner_name, center)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid center format for corner '{corner_name}': {corner_data.get('center')}. Error: {e}")
            # else:
            #      logger.debug(f"Skipping drawing for invalid/missing corner: {corner_name}")

        # Draw quadrilateral if all corners were considered valid for drawing markers
        # _all_corners_detected_and_valid already checks this
        _draw_quadrilateral(viz, corners)

    # NEW: Display the message if provided
    if message:
        text_pos = (10, 30)  # Top-left position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = COLOR_YELLOW
        # Semi-transparent black conceptually
        bg_color = (
            int(COLOR_BLACK[0]*0.5), int(COLOR_BLACK[1]*0.5), int(COLOR_BLACK[2]*0.5))
        # Actual transparency needs alpha blending

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            message, font, font_scale, font_thickness)

        # Draw background rectangle (slightly larger than text)
        # Ensure coordinates are int
        rect_pt1 = (text_pos[0] - 5, text_pos[1] + baseline + 5)
        rect_pt2 = (text_pos[0] + text_width + 5,
                    text_pos[1] - text_height - 5)
        cv2.rectangle(viz, rect_pt1, rect_pt2, bg_color, -1)  # cv2.FILLED

        # Draw text
        cv2.putText(viz, message, text_pos, font, font_scale,
                    text_color, font_thickness, cv2.LINE_AA)

    return viz


def _draw_corner_marker(
    viz: np.ndarray,
    corner_name: str,
    center: Tuple[int, int]  # Expect int tuple after map()
) -> None:
    """Draw individual corner marker."""
    x, y = center

    # Draw marker
    cv2.circle(viz, (x, y), 5, COLOR_GREEN, -1)  # Filled center
    cv2.circle(viz, (x, y), 10, COLOR_GREEN, 2)  # Outline

    # Draw corner lines/guides
    line_length = 20  # Shorter lines
    directions = _get_corner_directions(corner_name)

    for dx, dy in directions:
        end_x = int(x + dx * line_length)
        end_y = int(y + dy * line_length)
        cv2.line(viz, (x, y), (end_x, end_y), COLOR_GREEN, 2)  # Thinner lines

    # Add label for the corner name
    label_pos = _get_label_position(corner_name, x, y)
    cv2.putText(
        viz, corner_name.replace('_', ' ').title(),
        label_pos, cv2.FONT_HERSHEY_SIMPLEX,
        0.5, COLOR_YELLOW, 1, cv2.LINE_AA  # Smaller font, AA
    )


def _draw_quadrilateral(
    viz: np.ndarray,
    corners: Dict  # Assumes corners dict passed _all_corners_detected_and_valid
) -> None:
    """Draw boundary quadrilateral. Assumes valid corners are present."""
    points = []
    # Ensure standard order for drawing polylines
    ordered_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    for name in ordered_names:
        corner_data = corners.get(name)
        if isinstance(corner_data, dict) and 'center' in corner_data:
            try:
                points.append(list(map(int, corner_data['center'])))
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid center for quadrilateral point '{name}': {corner_data['center']}")
                return  # Do not draw incomplete quadrilateral
        else:
            # This case should ideally be caught by _all_corners_detected_and_valid
            # but good to have a fallback here too.
            logger.debug(
                f"Quadrilateral drawing: Missing or invalid data for corner '{name}'")
            return  # Do not draw incomplete quadrilateral

    if len(points) == 4:  # Should always be true if _all_corners_detected_and_valid passed
        pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts_np], True, COLOR_YELLOW, 1)  # Thinner line


def _draw_no_corners_found(viz: np.ndarray) -> None:
    """Draw message when no corners found or they are invalid."""
    cv2.putText(
        viz, "Corners: None or Invalid",
        # Centered vertically
        (10, int(viz.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, COLOR_RED, 2, cv2.LINE_AA
    )


def _all_corners_detected_and_valid(corners: Optional[Dict]) -> bool:
    """
    Check if the 'corners' dict is not None, is a dict,
    and all required corner keys exist with valid 'center' data.
    """
    if corners is None or not isinstance(corners, dict):
        return False
    required = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for name in required:
        corner_data = corners.get(name)
        if not (isinstance(corner_data, dict) and
                'center' in corner_data and
                isinstance(corner_data['center'], (tuple, list)) and
                len(corner_data['center']) == 2 and
                all(isinstance(coord, (int, float, np.number)) for coord in corner_data['center'])):
            return False
    return True


def _get_corner_directions(corner_name: str) -> List[Tuple[int, int]]:
    """Get line directions for corner marker visualization."""
    # (No changes needed here)
    if corner_name == 'top_left':
        return [(1, 0), (0, 1)]
    if corner_name == 'top_right':
        return [(-1, 0), (0, 1)]
    if corner_name == 'bottom_left':
        return [(1, 0), (0, -1)]
    if corner_name == 'bottom_right':
        return [(-1, 0), (0, -1)]
    return []


def _get_label_position(corner_name: str, x: int, y: int) -> Tuple[int, int]:
    """Calculate label position for corner name text."""
    # (Adjusted offsets for smaller font)
    text_offset_x = 15
    text_offset_y = 15

    if 'left' in corner_name:
        label_x = x + text_offset_x
    else:  # 'right'
        # Estimate text width to shift leftwards (very rough estimate)
        # Assuming avg char width of 7px for 0.5 scale
        approx_text_width = len(corner_name.replace('_', ' ')) * 7
        label_x = x - approx_text_width - text_offset_x

    if 'top' in corner_name:
        label_y = y - text_offset_y // 2
    else:  # 'bottom'
        label_y = y + text_offset_y + 5  # Move a bit lower for bottom labels

    return (label_x, label_y)


# Add logger if not already present at module level
logger = logging.getLogger(__name__)
