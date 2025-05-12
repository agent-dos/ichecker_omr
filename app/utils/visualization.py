# app/utils/visualization.py
import cv2
import numpy as np


def visualize_grayscale(gray_image):
    """
    Convert grayscale image to BGR for visualization.

    Args:
            gray_image: Grayscale image

    Returns:
            numpy.ndarray: BGR image
    """
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def visualize_circles(image, circles):
    """
    Visualize detected circles on the image.

    Args:
            image: The input image
            circles: Detected circles

    Returns:
            numpy.ndarray: Image with circles drawn
    """
    if circles is None:
        return image.copy()

    viz = image.copy()

    # Draw midpoint line for visual reference
    midpoint_x = image.shape[1] // 2
    cv2.line(viz, (midpoint_x, 0), (midpoint_x,
             image.shape[0]), (0, 0, 255), 1)

    # Add circles on different sides with different colors
    for x, y, r in circles:
        # Different colors for left and right columns
        color = (0, 255, 0) if x < midpoint_x else (
            255, 0, 0)  # Green for left, Blue for right

        # Draw the outer circle
        cv2.circle(viz, (x, y), r, color, 2)
        # Draw the center of the circle
        cv2.circle(viz, (x, y), 2, (0, 0, 255), 3)

    return viz


def visualize_threshold(threshold_image):
    """
    Prepare thresholded image for visualization.

    Args:
            threshold_image: Binary thresholded image

    Returns:
            numpy.ndarray: Image ready for display
    """
    # Apply color map for better visualization
    return cv2.applyColorMap(threshold_image, cv2.COLORMAP_HOT)


def visualize_answers(image, answers, coords):
    """
    Visualize detected answers on the image.

    Args:
            image: The input image
            answers: Detected answers (question number, answer)
            coords: Coordinates of the bubbles

    Returns:
            numpy.ndarray: Image with answers visualized
    """
    # Ensure image is BGR for drawing
    if len(image.shape) == 2:
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        viz = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        viz = image.copy()

    # Draw detected bubbles and selections
    for qdata in coords:
        q_num_vis = qdata["question_number"]
        for choice_data in qdata["choices"]:
            x, y, r = choice_data["x"], choice_data["y"], choice_data["r"]
            label_vis = choice_data["label"]
            selected_vis = choice_data["selected"]
            score = choice_data.get("score", 0)

            # Color based on selection status
            color_vis = (0, 255, 0) if selected_vis else (0, 0, 255)

            # Draw bubble outline
            cv2.circle(viz, (x, y), r, color_vis, 1)

            # Add label inside bubble
            cv2.putText(viz, label_vis, (x - r//2 + 2, y + r//2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_vis, 1)

            # Add score near bubble for debugging (only if selected or high score)
            if selected_vis or score > 50:
                cv2.putText(viz, f"{score:.1f}", (x + r + 2, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

            if selected_vis:
                # Thicker circle for selected
                cv2.circle(viz, (x, y), r + 2, (0, 255, 0), 2)
                # Display question number with selected answer more prominently
                cv2.putText(viz, f"{q_num_vis}:{label_vis}", (x - r - 10, y - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Create answer summary overlay
    answers.sort(key=lambda a: a[0])
    img_h, img_w = viz.shape[:2]
    font_scale_overlay = 0.4 * min(1.0, 1000 / max(img_h, img_w))
    line_height_overlay = max(10, int(15 * min(1.0, 1000 / max(img_h, img_w))))

    # Create overlay with detected answers
    max_answers_in_overlay_col = 15
    num_overlay_cols = 1 + \
        (len(answers) - 1) // max_answers_in_overlay_col if answers else 1
    num_overlay_cols = min(num_overlay_cols, 2)

    overlay_box_width = 100 * num_overlay_cols
    overlay_box_height = 5 + line_height_overlay * \
        min(max_answers_in_overlay_col, len(answers) if answers else 1)

    # Draw semi-transparent background
    overlay_bg = viz.copy()
    cv2.rectangle(overlay_bg, (5, 5), (overlay_box_width + 5,
                  overlay_box_height + 5), (255, 255, 255), -1)
    alpha = 0.6
    viz = cv2.addWeighted(overlay_bg, alpha, viz, 1 - alpha, 0)

    # Add text for each answer
    summary_y_start = 20
    for i, (q_num_sum, choice_sum) in enumerate(answers):
        current_overlay_col = i // max_answers_in_overlay_col
        summary_x_pos = 10 + (current_overlay_col * 100)
        summary_y_pos = summary_y_start + \
            (i % max_answers_in_overlay_col) * line_height_overlay

        if summary_x_pos > overlay_box_width - 10:
            break

        text_to_display = f"{q_num_sum}: {choice_sum if choice_sum else '-'}"
        cv2.putText(viz, text_to_display, (summary_x_pos, summary_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_overlay, (50, 50, 50), 1)

    return viz
