# app/utils/sheet_generator.py
"""
Main answer sheet generation module.
Orchestrates the creation of complete answer sheets using components.
"""
import logging
from PIL import Image, ImageDraw
from app.utils.sheet_layout_constants import *
from app.utils.sheet_components import *

logger = logging.getLogger(__name__)


def create_answer_sheet(
    title="Answer Sheet",
    student_id="000000",
    student_name="",
    page_width=PAGE_WIDTH,
    page_height=PAGE_HEIGHT,
    start_number=1,
    show_corner_markers=True,
    header_fields=None
):
    """
    Creates an answer sheet with advanced features.

    Args:
            title: Title of the exam
            student_id: Student ID for QR code
            student_name: Student name to display
            page_width: Width of the answer sheet
            page_height: Height of the answer sheet
            start_number: Starting question number
            show_corner_markers: Whether to show corner markers
            header_fields: Dictionary of header fields to display

    Returns:
            PIL.Image: Generated answer sheet
    """
    # Create blank image
    img = Image.new('RGB', (page_width, page_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Load fonts
    title_font = load_font(FONT_PATH, TITLE_FONT_SIZE)
    normal_font = load_font(FONT_PATH, DEFAULT_FONT_SIZE)

    # Draw components
    draw_title(draw, title, title_font, page_width)

    if header_fields is None:
        header_fields = get_default_header_fields(student_name)
    draw_header_fields(draw, header_fields, normal_font)

    if show_corner_markers:
        draw_corner_markers(draw, page_width, page_height)

    draw_qr_code(img, draw, student_id, page_width)
    _draw_answer_options(draw, normal_font, start_number)

    return img


def _draw_answer_options(draw, font, start_number):
    """Draw answer bubbles with question numbers."""
    choice_labels = ['A', 'B', 'C', 'D', 'E']

    # Load appropriate font for letters
    letter_font = font
    if LETTER_FONT_BOLD:
        letter_font = load_font(FONT_PATH_BOLD, DEFAULT_FONT_SIZE)

    for item_index in range(ITEMS_PER_SHEET):
        # Calculate position
        if item_index < ITEMS_PER_COLUMN:
            x = LEFT_COL_X
            y = QUESTION_START_Y + item_index * QUESTION_SPACING_Y
            question_num = start_number + item_index
        else:
            x = RIGHT_COL_X
            y = QUESTION_START_Y + \
                (item_index - ITEMS_PER_COLUMN) * QUESTION_SPACING_Y
            question_num = start_number + item_index

        # Draw question number
        draw.text((x, y), f"{question_num}.",
                  font=font, fill=QUESTION_NUMBER_COLOR)

        # Draw bubbles for each choice
        for j, letter in enumerate(choice_labels):
            center_x = x + OPTION_START_OFFSET_X + j * OPTION_SPACING
            center_y = y + DEFAULT_FONT_SIZE // 2

            # Draw bubble with configurable color
            draw_bubble(draw, center_x, center_y)

            # Draw letter with configurable color
            draw_letter_centered(draw, letter, center_x, center_y, letter_font)


def get_default_header_fields(student_name=""):
    """Get default header field configuration."""
    fields = DEFAULT_HEADER_FIELDS.copy()
    # Update name field with provided student name
    if student_name:
        fields[0]['value'] = student_name
    return fields
