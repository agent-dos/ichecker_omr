# app/utils/detect_answers.py
import logging
from app.utils.answer_processing import process_answer_sheet
from app.utils.answer_visualization import visualize_answers

logger = logging.getLogger(__name__)


def detect_answers_with_coords(
        image,
        bubble_threshold=100.0,
        param1=50,
        param2=18,
        min_radius=10,
        max_radius=20,
        resize_factor=1.0,
        block_size=31,
        c_value=10,
        row_threshold=8,
        score_multiplier=1.5
):
    """
    Returns detailed answer detection info including bubble coordinates and scores.

    All parameters are configurable for calibration.

    Returns:
            tuple: (answers, coords) where:
                    - answers: List of (question_number, selected_choice)
                    - coords: List of bubble coordinate details
    """
    return process_answer_sheet(
        image,
        bubble_threshold=bubble_threshold,
        param1=param1,
        param2=param2,
        min_radius=min_radius,
        max_radius=max_radius,
        resize_factor=resize_factor,
        block_size=block_size,
        c_value=c_value,
        row_threshold=row_threshold,
        score_multiplier=score_multiplier
    )


def detect_answers(
        image,
        visualize=True,
        bubble_threshold=100.0,
        param1=50,
        param2=18,
        min_radius=10,
        max_radius=20,
        resize_factor=1.0,
        block_size=31,
        c_value=10,
        row_threshold=8,
        score_multiplier=1.5
):
    """
    Detects answers and optionally visualizes them on the image.

    Args:
            image: Input image
            visualize: Whether to create visualization
            Additional parameters for answer detection

    Returns:
            - If visualize: (answers, viz_image)
            - Else: answers only
    """
    # Detect answers
    answers, coords = detect_answers_with_coords(
        image,
        bubble_threshold=bubble_threshold,
        param1=param1,
        param2=param2,
        min_radius=min_radius,
        max_radius=max_radius,
        resize_factor=resize_factor,
        block_size=block_size,
        c_value=c_value,
        row_threshold=row_threshold,
        score_multiplier=score_multiplier
    )

    if not visualize:
        return answers

    # Create visualization
    viz_image = visualize_answers(image, answers, coords)

    return answers, viz_image
