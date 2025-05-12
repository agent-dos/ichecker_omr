# app/utils/answer_visualization.py
import cv2
import numpy as np


def visualize_answers(image, answers, coords):
    """
    Visualize detected answers on the image.

    Args:
            image: Original image
            answers: List of (question_number, answer) tuples
            coords: List of bubble coordinate details

    Returns:
            numpy.ndarray: Image with visualizations
    """
    if image is None:
        return None

    # Create a copy to avoid modifying the original
    viz = _prepare_visualization_image(image)

    # Draw bubbles and selections
    _draw_bubbles(viz, coords)

    # Create answer summary overlay
    _draw_answer_summary(viz, answers)

    return viz


def visualize_debug_info(image, debug_info):
    """
    Visualize debug information including score statistics.

    Args:
            image: Original image
            debug_info: Dictionary containing debug information

    Returns:
            numpy.ndarray: Image with debug visualizations
    """
    if image is None:
        return None

    # Create a copy to avoid modifying the original
    viz = _prepare_visualization_image(image)

    # Draw debug information overlay
    _draw_debug_overlay(viz, debug_info)

    return viz


def _prepare_visualization_image(image):
    """
    Prepare the image for visualization by ensuring it's in BGR format.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        return image.copy()


def _draw_bubbles(viz, coords):
    """
    Draw detected bubbles with their labels and selection status.
    """
    for qdata in coords:
        q_num = qdata["question_number"]

        for choice_data in qdata["choices"]:
            x, y, r = choice_data["x"], choice_data["y"], choice_data["r"]
            label = choice_data["label"]
            selected = choice_data["selected"]
            score = choice_data.get("score", 0)

            # Choose color based on selection status
            color = (0, 255, 0) if selected else (
                0, 0, 255)  # Green if selected, Red otherwise

            # Draw bubble outline
            cv2.circle(viz, (x, y), r, color, 1)

            # Add label inside bubble
            cv2.putText(viz, label, (x - r//2 + 2, y + r//2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Show score for all bubbles in debug mode
            cv2.putText(viz, f"{score:.0f}", (x + r + 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 255), 1)

            if selected:
                # Draw thicker circle for selected bubbles
                cv2.circle(viz, (x, y), r + 2, (0, 255, 0), 2)

                # Display question number with selected answer
                cv2.putText(viz, f"{q_num}:{label}", (x - r - 10, y - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def _draw_answer_summary(viz, answers):
    """
    Draw a semi-transparent overlay showing the answer summary.
    """
    if not answers:
        return

    # Sort answers by question number
    answers.sort(key=lambda a: a[0])

    # Calculate dimensions
    img_h, img_w = viz.shape[:2]
    font_scale = 0.4 * min(1.0, 1000 / max(img_h, img_w))
    line_height = max(10, int(15 * min(1.0, 1000 / max(img_h, img_w))))

    # Configuration for answer overlay
    max_answers_per_col = 15
    num_cols = 1 + (len(answers) - 1) // max_answers_per_col if answers else 1
    num_cols = min(num_cols, 2)  # Max 2 columns

    overlay_width = 100 * num_cols
    overlay_height = 5 + line_height * min(max_answers_per_col, len(answers))

    # Create semi-transparent background
    overlay_bg = viz.copy()
    cv2.rectangle(overlay_bg, (5, 5),
                  (overlay_width + 5, overlay_height + 5),
                  (255, 255, 255), -1)

    # Blend with original image
    alpha = 0.6
    cv2.addWeighted(overlay_bg, alpha, viz, 1 - alpha, 0, viz)

    # Draw answer text
    summary_y_start = 20
    for i, (q_num, choice) in enumerate(answers):
        col_idx = i // max_answers_per_col
        x_pos = 10 + (col_idx * 100)
        y_pos = summary_y_start + (i % max_answers_per_col) * line_height

        if x_pos > overlay_width - 10:
            break

        text = f"{q_num}: {choice if choice else '-'}"
        cv2.putText(viz, text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 1)


def _draw_debug_overlay(viz, debug_info):
    """
    Draw debug information overlay on the image.
    """
    if not debug_info:
        return

    # Draw score statistics if available
    if 'score_stats' in debug_info:
        stats = debug_info['score_stats']

        # Create stats overlay in top right corner
        stats_text = [
            f"Score Statistics:",
            f"Min: {stats.get('min', 0):.1f}",
            f"Max: {stats.get('max', 0):.1f}",
            f"Mean: {stats.get('mean', 0):.1f}",
            f"Std: {stats.get('std', 0):.1f}",
            f"Threshold: {stats.get('threshold', 0):.1f}"
        ]

        # Calculate text dimensions
        font_scale = 0.5
        line_height = 25
        max_width = 0
        for text in stats_text:
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            max_width = max(max_width, text_size[0])

        # Position in top right
        start_x = viz.shape[1] - max_width - 20
        start_y = 30

        # Draw background for stats
        overlay_bg = viz.copy()
        cv2.rectangle(overlay_bg,
                      (start_x - 10, start_y - 20),
                      (viz.shape[1] - 10, start_y +
                       len(stats_text) * line_height),
                      (255, 255, 255), -1)
        cv2.addWeighted(overlay_bg, 0.7, viz, 0.3, 0, viz)

        # Draw stats text
        for i, text in enumerate(stats_text):
            cv2.putText(viz, text,
                        (start_x, start_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1)

    # Draw circle count if available
    if 'circles_count' in debug_info:
        count_text = f"Circles detected: {debug_info['circles_count']}"
        cv2.putText(viz, count_text, (10, viz.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
