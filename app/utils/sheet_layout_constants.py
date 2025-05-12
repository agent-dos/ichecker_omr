# app/utils/sheet_layout_constants.py
"""
Configuration constants for answer sheet generation.
This module contains all layout parameters and styling options.
"""

# Page Layout
PAGE_WIDTH, PAGE_HEIGHT = 850, 1202
BG_COLOR = (255, 255, 255)

# Font Configuration
FONT_PATH = "arial.ttf"
FONT_PATH_BOLD = "arialbd.ttf"
DEFAULT_FONT_SIZE = 16
TITLE_FONT_SIZE = 20
LETTER_FONT_BOLD = True

# Bubble Configuration
CIRCLE_RADIUS = 13
CIRCLE_THICKNESS = 2
LETTER_OFFSET_X = 1
LETTER_OFFSET_Y = -2

# Color Configuration
BUBBLE_COLOR = "#282828"  # Color of bubble outline
BUBBLE_FILL_COLOR = None  # Fill color (None for transparent)
LETTER_COLOR = "#282828"  # Color of letters inside bubbles
TITLE_COLOR = "black"  # Color of title text
HEADER_TEXT_COLOR = "black"  # Color of header text
HEADER_LINE_COLOR = "black"  # Color of header underlines
QUESTION_NUMBER_COLOR = "black"  # Color of question numbers

# Header Layout
HEADER_Y = 40

# QR Code Configuration
QR_CODE_Y_POS = 100
QR_CODE_SIZE = 120
QR_CODE_COLOR = "black"
QR_CODE_BG_COLOR = "white"

# Question Layout
QUESTION_START_Y = 115
LEFT_COL_X = 120
RIGHT_COL_X = PAGE_WIDTH // 2 + 100
QUESTION_SPACING_Y = 33
OPTION_SPACING = 35
OPTION_START_OFFSET_X = 50
ITEMS_PER_COLUMN = 30
ITEMS_PER_SHEET = ITEMS_PER_COLUMN * 2

# Corner Markers
CORNER_MARKER_SIZE = 25
CORNER_MARGIN = 80
CORNER_MARKER_COLOR = "black"

# Header Field Configuration
DEFAULT_HEADER_FIELDS = [
    {'label': 'Name:', 'value': '', 'x': 80, 'y': HEADER_Y, 'line_width': 180},
    {'label': 'Year:', 'value': '', 'x': 310,
     'y': HEADER_Y, 'line_width': 100},
    {'label': 'Strand:', 'value': '', 'x': 460,
     'y': HEADER_Y, 'line_width': 100},
    {'label': 'Subject:', 'value': '', 'x': 620,
     'y': HEADER_Y, 'line_width': 100}
]
