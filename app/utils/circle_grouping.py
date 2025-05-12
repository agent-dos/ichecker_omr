# app/utils/circle_grouping.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def group_circles_by_position(circles, image_width, row_threshold, resize_factor):
    """
    Group detected circles by their position (left/right columns and rows).

    Args:
            circles: Array of detected circles
            image_width: Width of the image
            row_threshold: Threshold for row grouping
            resize_factor: Image resize factor

    Returns:
            list: Grouped circles with question numbers
    """
    midpoint_x = image_width // 2
    left_circles = [c for c in circles if c[0] < midpoint_x]
    right_circles = [c for c in circles if c[0] >= midpoint_x]

    # Layout configuration
    items_per_column = 30
    columns = [(left_circles, 1), (right_circles, items_per_column + 1)]

    # Process each column
    grouped_results = []
    for column_circles, start_q_num in columns:
        if not column_circles:
            continue

        # Sort by Y position and group into rows
        rows = _group_into_rows(column_circles, row_threshold, resize_factor)

        # Add question numbers
        for row_idx, row_bubbles in enumerate(rows):
            q_num = start_q_num + row_idx
            grouped_results.append((q_num, row_bubbles))

    return grouped_results


def _group_into_rows(circles, row_threshold, resize_factor):
    """
    Group circles into rows based on their Y-coordinate proximity.

    Args:
            circles: List of circles to group
            row_threshold: Threshold for row grouping
            resize_factor: Image resize factor

    Returns:
            list: Circles grouped into rows
    """
    scaled_row_threshold = int(row_threshold * resize_factor)
    y_sorted = sorted(circles, key=lambda c: c[1])

    rows = []
    current_row = [y_sorted[0]]

    for i in range(1, len(y_sorted)):
        if abs(y_sorted[i][1] - current_row[0][1]) < scaled_row_threshold:
            current_row.append(y_sorted[i])
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [y_sorted[i]]

    if current_row:
        rows.append(sorted(current_row, key=lambda c: c[0]))

    # Sort rows by Y coordinate
    rows.sort(key=lambda r: r[0][1] if r else 0)
    return rows
