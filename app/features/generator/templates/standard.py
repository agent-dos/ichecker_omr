# app/features/generator/templates/standard.py
"""Standard answer sheet template."""
from PIL import Image, ImageDraw
from typing import Optional, Dict, List

from app.features.generator.templates.base import BaseTemplate
from app.features.generator.components.header import HeaderBuilder
from app.features.generator.components.bubbles import BubbleGrid
from app.features.generator.components.corners import CornerMarkers
from app.features.generator.components.qr_code import QRCodeGenerator


class StandardTemplate(BaseTemplate):
    """Standard iChecker answer sheet template."""
    
    def __init__(
        self,
        title: str = "Answer Sheet",
        student_id: str = "000000",
        student_name: str = "",
        start_number: int = 1,
        show_corner_markers: bool = True,
        header_fields: Optional[Dict] = None
    ):
        """Initialize standard template."""
        super().__init__()
        self.title = title
        self.student_id = student_id
        self.student_name = student_name
        self.start_number = start_number
        self.show_corner_markers = show_corner_markers
        self.header_fields = header_fields
        
        # Initialize components
        self.header_builder = HeaderBuilder()
        self.bubble_grid = BubbleGrid()
        self.corner_markers = CornerMarkers()
        self.qr_generator = QRCodeGenerator()
    
    def create(self) -> Image.Image:
        """Create standard answer sheet."""
        # Create blank page
        img, draw = self._create_blank_page()
        
        # Add components in order
        self._add_title(draw)
        self._add_header(draw)
        self._add_corner_markers(img, draw)
        self._add_qr_code(img)
        self._add_bubbles(draw)
        
        return img
    
    def _add_title(self, draw: ImageDraw.Draw) -> None:
        """Add title to the sheet."""
        self._draw_text_centered(
            draw, self.title, 14, 
            self.fonts['title'], 'black'
        )
    
    def _add_header(self, draw: ImageDraw.Draw) -> None:
        """Add header fields."""
        self.header_builder.draw(
            draw, self.title, self.student_name, 
            self.header_fields
        )
    
    def _add_corner_markers(
        self, 
        img: Image.Image, 
        draw: ImageDraw.Draw
    ) -> None:
        """Add corner markers if enabled."""
        if self.show_corner_markers:
            self.corner_markers.draw(
                draw, self.page_width, self.page_height
            )
    
    def _add_qr_code(self, img: Image.Image) -> None:
        """Add QR code for student ID."""
        self.qr_generator.draw(
            img, self.student_id, self.page_width
        )
    
    def _add_bubbles(self, draw: ImageDraw.Draw) -> None:
        """Add answer bubbles."""
        self.bubble_grid.draw(draw, self.start_number)