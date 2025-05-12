# app/core/config.py
"""Configuration management for iChecker."""
import json
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


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

CONFIG_FILE = "app_config.json"


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return DEFAULT_PARAMS.copy()
        return DEFAULT_PARAMS.copy()

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self._config = config
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self._config.update(updates)

    def reset(self) -> None:
        """Reset to default configuration."""
        self._config = DEFAULT_PARAMS.copy()
        self.save_config(self._config)

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config.copy()


# Singleton instance
config_manager = ConfigManager()


def get_default_parameters() -> Dict[str, Any]:
    """Get default parameters."""
    return config_manager.config


def save_parameters(params: Dict[str, Any]) -> None:
    """Save parameters."""
    config_manager.save_config(params)
