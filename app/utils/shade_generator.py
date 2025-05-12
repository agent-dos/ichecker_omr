# app/utils/shade_generator.py
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2


def shade_random_bubbles(
        image,
        num_answers=20,
        shade_intensity=(100, 200),
        offset_range=2,
        size_percent=0.2,
        answers=None
):
    """
    Adds random shading to bubble answers

    Args:
            image: PIL Image or numpy array of answer sheet
            num_answers: Number of bubbles to shade randomly
            shade_intensity: Range of shade darkness (min, max) 0-255
            offset_range: Maximum pixel offset for random positioning
            size_percent: Percentage of circle radius to add/subtract
            answers: Optional predefined answers dict {question: choice_index}

    Returns:
            numpy.ndarray: Image with shaded bubbles
    """
    # Convert PIL image to CV2 format if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_np = image.copy()

    # Detect bubbles in the image
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=18,
        minRadius=10,
        maxRadius=14
    )

    if circles is None:
        return image_np

    # Round circles to integers
    circles = np.round(circles[0, :]).astype(int)

    # Organize bubbles by rows (questions)
    rows = {}
    midpoint_x = image_np.shape[1] // 2

    # Group circles by Y position (within threshold)
    row_threshold = 10
    y_sorted = sorted(circles, key=lambda c: c[1])

    current_row = []
    current_row_y = y_sorted[0][1]

    for circle in y_sorted:
        x, y, r = circle

        # If this circle is far from current row, start a new row
        if abs(y - current_row_y) > row_threshold:
            # Sort current row by x position
            if current_row:
                row_key = f"row_{len(rows)}"
                rows[row_key] = sorted(current_row, key=lambda c: c[0])
                current_row = []
            current_row_y = y

        current_row.append(circle)

    # Add the last row
    if current_row:
        row_key = f"row_{len(rows)}"
        rows[row_key] = sorted(current_row, key=lambda c: c[0])

    # Filter valid rows (should have 5 circles for A-E)
    valid_rows = {k: v for k, v in rows.items() if len(v) >= 5}

    # Create a mapping from row to question number
    # We need to sort them by Y position first
    sorted_rows = sorted(valid_rows.items(), key=lambda x: x[1][0][1])

    # Split into left and right columns
    left_rows = []
    right_rows = []

    for row_key, circles in sorted_rows:
        # Check if the first circle is in the left or right half
        if circles[0][0] < midpoint_x:
            left_rows.append(row_key)
        else:
            right_rows.append(row_key)

    # Create mapping from row key to question number
    row_to_question = {}

    # Limit to 30 questions per column
    for i, row_key in enumerate(left_rows[:30]):
        row_to_question[row_key] = i + 1

    # Limit to 30 questions per column
    for i, row_key in enumerate(right_rows[:30]):
        row_to_question[row_key] = i + 31

    # Generate random answers if not provided
    if answers is None:
        # Select random rows to shade
        rows_to_shade = random.sample(
            list(row_to_question.keys()), min(num_answers, len(row_to_question)))
        answers = {row: random.randint(0, 4) for row in rows_to_shade}

    # Apply shading to selected bubbles
    for row_key, choice_idx in answers.items():
        if row_key in valid_rows and choice_idx < len(valid_rows[row_key]):
            x, y, r = valid_rows[row_key][choice_idx]

            # Apply random offset
            dx = random.randint(-offset_range, offset_range)
            dy = random.randint(-offset_range, offset_range)

            # Apply random size variation
            size_variation = random.uniform(-size_percent, size_percent)
            radius = int(r * (1 + size_variation))

            # Apply random shade intensity
            intensity = random.randint(shade_intensity[0], shade_intensity[1])
            fill_color = (intensity, intensity, intensity)

            # Draw filled circle
            cv2.circle(image_np, (x + dx, y + dy), radius, fill_color, -1)

    return image_np
