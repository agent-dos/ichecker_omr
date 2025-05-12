# app/features/debug/analyzer_debug.py
"""Debug utilities for analyzer."""
import cv2
import numpy as np
from typing import Dict, Optional
import logging

from app.features.corners.detector import CornerDetector
from app.features.corners.visualizer import visualize_corners
from app.features.analyzer.visualizers.debug_viz import visualize_debug

logger = logging.getLogger(__name__)


class AnalyzerDebugger:
    """Debug utilities for answer sheet analysis."""
    
    def __init__(self):
        self.corner_detector = CornerDetector()
    
    def run_corner_debug(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict:
        """Run corner detection debug."""
        results = {}
        
        # Detect corners with multiple strategies
        corners = self.corner_detector.detect(image)
        results['corners'] = corners
        
        # Create visualization
        viz = visualize_corners(image, corners)
        results['visualization'] = viz
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, viz)
            logger.info(f"Saved debug image to {save_path}")
        
        return results
    
    def run_bubble_debug(
        self,
        image: np.ndarray,
        params: Dict
    ) -> Dict:
        """Run bubble detection debug."""
        from app.features.analyzer.service import AnalyzerService
        
        analyzer = AnalyzerService(params)
        results = {}
        
        # Run detection with debug info
        debug_data = {
            'enable_debug': True,
            'save_intermediates': True
        }
        
        analysis_results = analyzer.analyze(image)
        results['analysis'] = analysis_results
        
        # Create debug visualization
        if 'debug_info' in analysis_results:
            viz = visualize_debug(image, analysis_results['debug_info'])
            results['debug_viz'] = viz
        
        return results