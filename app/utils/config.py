# app/utils/config.py
import json
import os

# Updated default parameters with better calibration
DEFAULT_PARAMS = {
    'bubble_threshold': 50.0,  # Lowered from 100.0
    'param1': 50,
    'param2': 18,
    'min_radius': 10,
    'max_radius': 20,
    'resize_factor': 1.0,
    'block_size': 31,
    'c_value': 10,
    'row_threshold': 8,
    'score_multiplier': 2.0,  # Increased from 1.5
    'use_corner_detection': True,  # Add this parameter
    'debug_detection': False,
    'use_corner_detection': True,
    'corner_detection_fallback': True,  # Process full image if corners not found
    'corner_detection_strict': False,   # Require all 4 corners vs best effort
    'corner_min_area': 300,
    'corner_max_area': 5000,
    'enable_rectification': True,
    'rectification_threshold': 5.0,  # Degrees
    'rectification_margin': 20,  # Pixels
}

# Path to the config file
CONFIG_FILE = "app_config.json"


def get_default_parameters():
    """
    Get default parameters from config file or use built-in defaults

    Returns:
            dict: Parameters for answer detection
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            # If error reading file, return defaults
            return DEFAULT_PARAMS
    else:
        return DEFAULT_PARAMS


def save_parameters(params):
    """
    Save parameters to config file

    Args:
            params (dict): Parameters to save
    """
    with open(CONFIG_FILE, 'w') as f:
        json.dump(params, f)
