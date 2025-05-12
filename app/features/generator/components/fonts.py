# app/features/generator/components/fonts.py
"""
Font loading utilities for generator.
"""
from PIL import ImageFont
import logging

from app.core.constants import FONT_PATH

logger = logging.getLogger(__name__)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Load font with fallback to default.
    """
    try:
        font = ImageFont.truetype(FONT_PATH, size)
        return font
    except IOError:
        logger.warning(f"Font '{FONT_PATH}' not found. Using default.")
        return ImageFont.load_default()
