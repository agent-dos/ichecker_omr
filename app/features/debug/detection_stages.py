# app/features/debug/detection_stages.py
"""Visualization of detection stages."""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from app.core.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW


class DetectionStageVisualizer:
    """Visualizes different detection stages."""

    def create_stage_comparison(
        self,
        stages: Dict[str, Dict]
    ) -> np.ndarray:
        """Create side-by-side comparison of stages."""
        if not stages:
            return None

        # Get first image to determine size
        first_stage = next(iter(stages.values()))
        if 'image' not in first_stage:
            return None

        h, w = first_stage['image'].shape[:2]

        # Calculate grid size
        num_stages = len(stages)
        cols = min(3, num_stages)
        rows = (num_stages + cols - 1) // cols

        # Create comparison image
        comparison = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Place stage images
        for i, (name, data) in enumerate(stages.items()):
            if 'image' not in data:
                continue

            row = i // cols
            col = i % cols
            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w

            comparison[y1:y2, x1:x2] = data['image']

            # Add label
            cv2.putText(
                comparison, name,
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

        return comparison

    def create_exclusion_map(
        self,
        image: np.ndarray,
        included: List,
        excluded: List,
        boundaries: Optional[List] = None
    ) -> np.ndarray:
        """Create map showing included/excluded regions."""
        viz = image.copy()

        # Draw boundaries if provided
        if boundaries:
            for boundary in boundaries:
                cv2.polylines(
                    viz, [np.array(boundary, np.int32)],
                    True, COLOR_YELLOW, 2
                )

        # Draw excluded items in red
        for item in excluded:
            if len(item) >= 3:  # Circle
                x, y, r = item[:3]
                cv2.circle(viz, (x, y), r, COLOR_RED, 2)
                cv2.line(viz, (x-r, y), (x+r, y), COLOR_RED, 2)
                cv2.line(viz, (x, y-r), (x, y+r), COLOR_RED, 2)

        # Draw included items in green
        for item in included:
            if len(item) >= 3:  # Circle
                x, y, r = item[:3]
                cv2.circle(viz, (x, y), r, COLOR_GREEN, 2)

        return viz
