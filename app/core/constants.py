# app/core/constants.py
"""
Central constants for the iChecker application.
"""

# Page Layout (from sheet_layout_constants.py)
PAGE_WIDTH, PAGE_HEIGHT = 850, 1202
BG_COLOR = (255, 255, 255)

# Font Configuration (from sheet_layout_constants.py)
FONT_PATH = "fonts/arial.ttf"
FONT_PATH_BOLD = "fonts/arialbd.ttf"
DEFAULT_FONT_SIZE = 16
TITLE_FONT_SIZE = 20
LETTER_FONT_BOLD = True

# Bubble Configuration (from sheet_layout_constants.py)
CIRCLE_RADIUS = 13
CIRCLE_THICKNESS = 2
LETTER_OFFSET_X = 1
LETTER_OFFSET_Y = -2

# Color Configuration (from sheet_layout_constants.py)
BUBBLE_COLOR = "#282828"
BUBBLE_FILL_COLOR = None
LETTER_COLOR = "#282828"
TITLE_COLOR = "black"
HEADER_TEXT_COLOR = "black"
HEADER_LINE_COLOR = "black"
QUESTION_NUMBER_COLOR = "black"

# Header Layout (from sheet_layout_constants.py)
HEADER_Y = 40

# QR Code Configuration (from sheet_layout_constants.py)
QR_CODE_Y_POS = 100
QR_CODE_SIZE = 120
QR_CODE_COLOR = "black"
QR_CODE_BG_COLOR = "white"

# Question Layout (from sheet_layout_constants.py)
QUESTION_START_Y = 115
LEFT_COL_X = 120
RIGHT_COL_X = PAGE_WIDTH // 2 + 100
QUESTION_SPACING_Y = 33
OPTION_SPACING = 35
OPTION_START_OFFSET_X = 50
ITEMS_PER_COLUMN = 30
ITEMS_PER_SHEET = ITEMS_PER_COLUMN * 2

# Corner Markers (from sheet_layout_constants.py)
CORNER_MARKER_SIZE = 25
CORNER_MARGIN = 80
CORNER_MARKER_COLOR = "black"

# Header Field Configuration (from sheet_layout_constants.py)
DEFAULT_HEADER_FIELDS = [
    {'label': 'Name:', 'value': '', 'x': 80, 'y': HEADER_Y, 'line_width': 180},
    {'label': 'Year:', 'value': '', 'x': 310, 'y': HEADER_Y, 'line_width': 100},
    {'label': 'Strand:', 'value': '', 'x': 460, 'y': HEADER_Y, 'line_width': 100},
    {'label': 'Subject:', 'value': '', 'x': 620, 'y': HEADER_Y, 'line_width': 100}
]

# Detection parameters (existing)
DEFAULT_MIN_CORNER_AREA = 300
DEFAULT_MAX_CORNER_AREA = 5000
DEFAULT_CORNER_DISTANCE_THRESHOLD = 30

# Threshold levels for corner detection (existing)
CORNER_THRESHOLD_LEVELS = [30, 50, 70, 90]

# Bubble detection parameters (existing)
DEFAULT_BUBBLE_RADIUS_MIN = 10
DEFAULT_BUBBLE_RADIUS_MAX = 20
DEFAULT_BUBBLE_SPACING = 20

# Image processing (existing)
DEFAULT_BLOCK_SIZE = 31
DEFAULT_C_VALUE = 10

# Visualization colors (existing)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)

# Default parameters for configuration (from original config.py)
DEFAULT_PARAMS = {
    'bubble_threshold': 50.0,
    'param1': 50,
    'param2': 18,
    'min_radius': 10,
    'max_radius': 20,
    'resize_factor': 1.0,
    'block_size': 31,
    'c_value': 10,
    'row_threshold': 8,
    'score_multiplier': 2.0,
    'use_corner_detection': True,
    'debug_detection': False,
    'corner_detection_fallback': True,
    'corner_detection_strict': False,
    'corner_min_area': 300,
    'corner_max_area': 5000,
    'enable_rectification': True,
    'rectification_threshold': 5.0,
    'rectification_margin': 20,
}
