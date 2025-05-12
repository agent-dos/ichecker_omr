# app/features/generator/templates/base.py
"""Base template for answer sheet generation."""
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional

from app.core.constants import (
    PAGE_WIDTH, PAGE_HEIGHT, BG_COLOR,
    FONT_PATH, DEFAULT_FONT_SIZE
)


class BaseTemplate(ABC):
    """Abstract base class for answer sheet templates."""
    
    def __init__(self):
        """Initialize base template with common properties."""
        self.page_width = PAGE_WIDTH
        self.page_height = PAGE_HEIGHT
        self.bg_color = BG_COLOR
        self.fonts = self._load_fonts()
    
    @abstractmethod
    def create(self) -> Image.Image:
        """Create the answer sheet. Must be implemented by subclasses."""
        pass
    
    def _load_fonts(self) -> Dict[str, ImageFont.FreeTypeFont]:
        """Load common fonts used in templates."""
        fonts = {}
        try:
            fonts['normal'] = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE)
            fonts['title'] = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE + 4)
            fonts['small'] = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE - 2)
        except IOError:
            # Fallback to default fonts
            default_font = ImageFont.load_default()
            fonts['normal'] = default_font
            fonts['title'] = default_font
            fonts['small'] = default_font
        return fonts
    
    def _create_blank_page(self) -> Tuple[Image.Image, ImageDraw.Draw]:
        """Create a blank page with background color."""
        img = Image.new('RGB', (self.page_width, self.page_height), 
                       color=self.bg_color)
        draw = ImageDraw.Draw(img)
        return img, draw
    
    def _draw_text_centered(
        self,
        draw: ImageDraw.Draw,
        text: str,
        y: int,
        font: ImageFont.FreeTypeFont,
        color: str = 'black'
    ) -> None:
        """Draw text centered horizontally at given y position."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.page_width - text_width) // 2
        draw.text((x, y), text, font=font, fill=color)
    
    def _draw_grid_lines(
        self,
        draw: ImageDraw.Draw,
        start_x: int,
        start_y: int,
        cols: int,
        rows: int,
        cell_width: int,
        cell_height: int,
        color: str = 'lightgray'
    ) -> None:
        """Draw grid lines for bubble placement."""
        # Vertical lines
        for i in range(cols + 1):
            x = start_x + i * cell_width
            draw.line([(x, start_y), (x, start_y + rows * cell_height)], 
                     fill=color, width=1)
        
        # Horizontal lines
        for i in range(rows + 1):
            y = start_y + i * cell_height
            draw.line([(start_x, y), (start_x + cols * cell_width, y)], 
                     fill=color, width=1)
    
    def _calculate_bubble_positions(
        self,
        start_x: int,
        start_y: int,
        rows: int,
        cols: int,
        spacing_x: int,
        spacing_y: int
    ) -> List[List[Tuple[int, int]]]:
        """Calculate bubble center positions for a grid layout."""
        positions = []
        for row in range(rows):
            row_positions = []
            y = start_y + row * spacing_y
            for col in range(cols):
                x = start_x + col * spacing_x
                row_positions.append((x, y))
            positions.append(row_positions)
        return positions