# filename: app/core/config.py
import json
import os
import logging
from typing import Dict, Any, Optional
import cv2  # Import cv2 for flag constants

logger = logging.getLogger(__name__)

# Define OpenCV constants mapping (example)
CV2_INTERPOLATION_FLAGS = {
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_AREA': cv2.INTER_AREA,
    'INTER_CUBIC': cv2.INTER_CUBIC,
    'INTER_LANCZOS4': cv2.INTER_LANCZOS4
}

CV2_MORPH_OPS = {
    'MORPH_CLOSE': cv2.MORPH_CLOSE,
    'MORPH_OPEN': cv2.MORPH_OPEN,
    'MORPH_ERODE': cv2.MORPH_ERODE,
    'MORPH_DILATE': cv2.MORPH_DILATE,
    'MORPH_GRADIENT': cv2.MORPH_GRADIENT,
    'MORPH_TOPHAT': cv2.MORPH_TOPHAT,
    'MORPH_BLACKHAT': cv2.MORPH_BLACKHAT,
}

CV2_ADAPTIVE_METHODS = {
    'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
    'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}

CV2_THRESH_TYPES = {
    'THRESH_BINARY': cv2.THRESH_BINARY,
    'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV,
    # Add others if needed (THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)
}


# --- Detailed Default Parameters ---
DEFAULT_PARAMS = {
    # General Analyzer Settings
    'analyzer': {
        'enable_rectification': True,
        # Angle threshold (degrees) to trigger rectification
        'rectification_threshold': 5.0,
    },

    # Rectification Step Parameters
    'rectification': {
        # Interpolation for cv2.warpPerspective
        'warp_interpolation': 'INTER_LINEAR',  # String key for CV2_INTERPOLATION_FLAGS
        # Margin for destination points (relative to standard page size)
        'dst_margin': 0,
        # If corner detection fails, still return original image for processing?
        'fail_safe_return_original': True,
    },

    # QR Detection Step Parameters
    'qr_detection': {
        # Parameters for preprocessing attempts
        'gaussian_blur_ksize': 5,       # Kernel size for Gaussian Blur attempt
        'adaptive_method': 'ADAPTIVE_THRESH_GAUSSIAN_C',  # cv2 adaptive method key
        'adaptive_blocksize': 11,       # Block size for Adaptive Threshold attempt
        'adaptive_c': 2,                # Constant C for Adaptive Threshold attempt
        'equalize_hist': True,          # Enable/disable histogram equalization attempt
    },

    # Corner Detection Step Parameters (Main detection after QR)
    'corner_detection': {
        'min_area': 300,                # Minimum contour area for a candidate
        'max_area': 5000,               # Maximum contour area for a candidate
        'duplicate_threshold': 30,      # Pixel distance to consider corners duplicates
        # Strategy: Simple Thresholding
        'strategy_threshold': {
            'enabled': True,
            'levels': [30, 50, 70, 90],  # List of threshold values to try
            'threshold_type': 'THRESH_BINARY_INV',  # String key for CV2_THRESH_TYPES
            'morph_op1': 'MORPH_CLOSE',  # String key for CV2_MORPH_OPS
            'morph_op2': 'MORPH_OPEN',  # String key for CV2_MORPH_OPS
            'morph_ksize': 5,           # Kernel size for morphology
            # Minimum solidity (area / convex hull area)
            'solidity_min': 0.8,
            # Min aspect ratio (width/height) of bounding box
            'aspect_ratio_min': 0.7,
            'aspect_ratio_max': 1.3,    # Max aspect ratio
            'fill_ratio_min': 0.85,     # Min ratio of white pixels within contour after threshold
        },
        # Strategy: Adaptive Thresholding
        'strategy_adaptive': {
            'enabled': True,
            'adaptive_method': 'ADAPTIVE_THRESH_MEAN_C',  # Key for CV2_ADAPTIVE_METHODS
            'threshold_type': 'THRESH_BINARY_INV',       # Key for CV2_THRESH_TYPES
            'blocksize': 31,            # Block size for adaptive threshold
            'c': 10,                    # Constant C
            'aspect_ratio_min': 0.5,    # Min aspect ratio of bounding box
            'aspect_ratio_max': 2.0,    # Max aspect ratio
        },
        # Strategy: Edge Detection (Canny)
        'strategy_edge': {
            'enabled': True,
            'gaussian_blur_ksize': 5,   # Kernel size for pre-blur
            'canny_threshold1': 50,     # Lower threshold for cv2.Canny
            'canny_threshold2': 150,    # Upper threshold for cv2.Canny
        },
        # Parameters for selecting best corners
        'scoring': {
            'distance_weight': 0.5,
            'area_weight': 0.25,
            'solidity_weight': 0.25,
            'area_norm_factor': 1000.0,  # Value to normalize area score
        },
        'validator': {
            # Parameters for _is_qr_pattern validation
            'qr_filter_enabled': True,
            'qr_canny_threshold1': 50,
            'qr_canny_threshold2': 150,
            'qr_edge_ratio_threshold': 0.15,  # Threshold for edge density
            'qr_complexity_threshold': 0.3,  # Threshold for variance/complexity
        }
    },

    # Bubble Detection Step Parameters
    'bubble_detection': {
        'gaussian_blur_ksize': 5,       # Pre-blur kernel size
        'hough_dp': 1,                  # Inverse ratio of accumulator resolution
        'hough_minDist': 20,            # Minimum distance between centers
        'hough_param1': 50,             # Upper Canny threshold for HoughCircles
        'hough_param2': 18,             # Accumulator threshold for center detection
        'hough_minRadius': 10,          # Minimum bubble radius
        'hough_maxRadius': 20,          # Maximum bubble radius
        # Filtering parameters
        'filter_by_corners': True,      # Enable filtering bubbles outside corner boundary
        # Margin (pixels) inside corner boundary
        'boundary_filter_margin': 5,
        'filter_by_qr': True,           # Enable filtering bubbles inside QR polygon
        'qr_filter_margin_factor': 1.0,  # Margin = radius + factor * radius around QR
    },

    # Bubble Analysis Step Parameters
    'bubble_analysis': {
        # Adaptive threshold for scoring
        'adaptive_method': 'ADAPTIVE_THRESH_MEAN_C',  # Key for CV2_ADAPTIVE_METHODS
        'adaptive_blocksize': 31,       # Block size must be odd
        'adaptive_c': 10,               # Constant C
        # Grouping bubbles into rows/questions
        'grouping_row_threshold': 8,    # Max Y distance for bubbles in same row
        'grouping_items_per_col': 30,   # Expected items per column for numbering
        # Scoring individual bubbles
        'scoring_inner_radius_factor': 0.8,  # Ratio of radius used for fill calculation
        # Min normalized fill score to be considered 'filled'
        'scoring_bubble_threshold': 50.0,
        'scoring_score_multiplier': 2.0,    # Multiplier applied to normalized score
    },

    # Debug/Visualization Options
    'debug_options': {
        'visualize_intermediate_steps': False,  # Master switch
        'log_level': 'INFO',
        # Add more granular controls if needed
        # 'viz_qr_preprocessing': False,
        # 'viz_corner_thresholding': False,
        # ... etc.
    }
}

CONFIG_FILE = "app_config_detailed.json"  # Use a new file name

# --- ConfigManager Class ---
# (Keep the ConfigManager class largely the same, ensure it handles nested dicts)
# Need to handle potential errors if loaded config doesn't match default structure


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self._config = self._load_config()
        # Ensure loaded config has all keys from default (merge missing keys)
        self._config = self._merge_configs(DEFAULT_PARAMS, self._config)

    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config into default config."""
        merged = default.copy()
        for key, value in loaded.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Only update if key exists in default (prevents injection of arbitrary keys)
                # Or allow new keys? For flexibility, let's allow for now.
                merged[key] = value
        return merged

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    logger.info(
                        f"Loaded configuration from {self.config_file}")
                    return loaded_config
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding JSON from {self.config_file}: {e}")
                logger.warning("Using default parameters.")
                return DEFAULT_PARAMS.copy()
            except Exception as e:
                logger.error(
                    f"Error loading config file {self.config_file}: {e}")
                logger.warning("Using default parameters.")
                return DEFAULT_PARAMS.copy()
        else:
            logger.info("Config file not found. Using default parameters.")
            return DEFAULT_PARAMS.copy()

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        config_to_save = config if config is not None else self._config
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            self._config = config_to_save  # Update internal state if saved successfully
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except TypeError as e:
            logger.error(
                f"Error serializing config to JSON: {e}. Check for non-serializable types.")
            return False
        except Exception as e:
            logger.error(f"Error saving config to {self.config_file}: {e}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'bubble.hough_dp')."""
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # logger.warning(f"Config key '{key_path}' not found, returning default.")
            return default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        d = self._config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values (top-level)."""
        # For deep updates, maybe add a deep_update method or use set()
        self._config.update(updates)

    def reset(self) -> None:
        """Reset to default configuration."""
        self._config = DEFAULT_PARAMS.copy()
        self.save_config(self._config)  # Save defaults after reset

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration."""
        # Return a deep copy to prevent external modification
        return json.loads(json.dumps(self._config))


# Singleton instance
config_manager = ConfigManager()

# --- Helper Functions ---


def get_config() -> Dict[str, Any]:
    """Get the current configuration dictionary."""
    return config_manager.config


def save_config(params: Dict[str, Any]) -> None:
    """Save parameters."""
    config_manager.save_config(params)


def get_cv2_flag(flag_name: str, flag_dict: Dict, default_flag: Any) -> Any:
    """Safely get a CV2 flag constant from its string name."""
    if not isinstance(flag_name, str):
        logger.warning(
            f"Invalid flag name type: {type(flag_name)}. Using default.")
        return default_flag
    flag_value = flag_dict.get(flag_name.upper())
    if flag_value is None:
        logger.warning(
            f"Unknown flag name '{flag_name}'. Valid keys: {list(flag_dict.keys())}. Using default.")
        return default_flag
    return flag_value
