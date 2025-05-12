# app/utils/sheet_components.py
"""
Component drawing functions for answer sheet generation.
Handles individual elements like headers, QR codes, and bubbles.
"""
import logging
from PIL import ImageDraw, ImageFont
import qrcode
from app.utils.sheet_layout_constants import *

logger = logging.getLogger(__name__)


def load_font(font_path, size):
    """Load font with fallback to default if not found."""
    try:
        font = ImageFont.truetype(font_path, size)
        logger.debug(f"Loaded font: {font_path} (size {size})")
        return font
    except IOError:
        font = ImageFont.load_default()
        logger.warning(f"Font '{font_path}' not found. Using default.")
        return font


def draw_title(draw, title, font, page_width):
    """Draw centered title at the top of the page."""
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        title_width = bbox[2] - bbox[0]
    except AttributeError:
        title_width = draw.textlength(title, font=font)

    draw.text(((page_width - title_width) // 2, 14),
              title, font=font, fill=TITLE_COLOR)
    logger.debug(f"Drew title: '{title}'")


def draw_header_fields(draw, fields, font):
    """Draw header fields with underlines."""
    for field_info in fields:
        label = field_info['label']
        value = field_info['value']
        x = field_info['x']
        y = field_info['y']
        line_width = field_info.get('line_width', 100)

        # Draw label
        draw.text((x, y), label, font=font, fill=HEADER_TEXT_COLOR)

        # Calculate label width
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            label_width = bbox[2] - bbox[0]
        except AttributeError:
            label_width = draw.textlength(label, font=font)

        # Draw underline
        line_y = y + DEFAULT_FONT_SIZE
        draw.line([(x + label_width + 5, line_y),
                   (x + label_width + line_width, line_y)],
                  fill=HEADER_LINE_COLOR, width=1)

        # Draw value if provided
        if value:
            draw.text((x + label_width + 10, y),
                      value, font=font, fill=HEADER_TEXT_COLOR)


def draw_corner_markers(draw, page_width, page_height):
    """Draw corner markers for sheet alignment."""
    marker_positions = [
        (CORNER_MARGIN, CORNER_MARGIN),
        (page_width - CORNER_MARGIN - CORNER_MARKER_SIZE, CORNER_MARGIN),
        (CORNER_MARGIN, page_height - CORNER_MARGIN - CORNER_MARKER_SIZE),
        (page_width - CORNER_MARGIN - CORNER_MARKER_SIZE,
         page_height - CORNER_MARGIN - CORNER_MARKER_SIZE)
    ]

    for x, y in marker_positions:
        draw.rectangle((x, y, x + CORNER_MARKER_SIZE, y + CORNER_MARKER_SIZE),
                       fill=CORNER_MARKER_COLOR)


def draw_qr_code(img, draw, student_id, page_width):
    """Generate and draw QR code for student ID."""
    qr_x_pos = (page_width - QR_CODE_SIZE) // 2

    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=5,
            border=2,
        )
        qr.add_data(student_id)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color=QR_CODE_COLOR,
                               back_color=QR_CODE_BG_COLOR)
        qr_img = qr_img.resize((QR_CODE_SIZE, QR_CODE_SIZE))
        img.paste(qr_img, (qr_x_pos, QR_CODE_Y_POS))
        logger.debug("QR code drawn successfully.")
    except Exception as e:
        logger.error(f"Error generating QR code for ID {student_id}: {e}")
        draw.rectangle((qr_x_pos, QR_CODE_Y_POS,
                        qr_x_pos + QR_CODE_SIZE, QR_CODE_Y_POS + QR_CODE_SIZE),
                       outline="red")


def draw_bubble(draw, center_x, center_y):
    """Draw a single bubble with configurable color and thickness."""
    draw.ellipse(
        (center_x - CIRCLE_RADIUS, center_y - CIRCLE_RADIUS,
         center_x + CIRCLE_RADIUS, center_y + CIRCLE_RADIUS),
        outline=BUBBLE_COLOR, fill=BUBBLE_FILL_COLOR, width=CIRCLE_THICKNESS
    )


def draw_letter_centered(draw, letter, center_x, center_y, font):
    """Draw a letter centered within given coordinates with offsets."""
    try:
        bbox = draw.textbbox((0, 0), letter, font=font)
        letter_width = bbox[2] - bbox[0]
        letter_height = bbox[3] - bbox[1]

        text_x = center_x - letter_width // 2 + LETTER_OFFSET_X
        text_y = center_y - letter_height // 2 + LETTER_OFFSET_Y

        draw.text((text_x, text_y), letter, font=font, fill=LETTER_COLOR)
    except AttributeError:
        letter_width = draw.textlength(letter, font=font)
        text_x = center_x - letter_width // 2 + LETTER_OFFSET_X
        text_y = center_y - DEFAULT_FONT_SIZE // 2 + LETTER_OFFSET_Y
        draw.text((text_x, text_y), letter, font=font, fill=LETTER_COLOR)
