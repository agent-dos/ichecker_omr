# app/features/corners/detector.py
"""
Main corner detection module.
Limited to coordinating detection strategies.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Import strategies individually to avoid circular imports
from app.features.corners.strategies.threshold import ThresholdStrategy
from app.features.corners.strategies.adaptive import AdaptiveStrategy
from app.features.corners.strategies.edge import EdgeStrategy
from app.features.corners.validators import CornerValidator
from app.core.constants import DEFAULT_MIN_CORNER_AREA, DEFAULT_MAX_CORNER_AREA

logger = logging.getLogger(__name__)


class CornerDetector:
    """
    Detects corner markers using multiple strategies.
    """

    def __init__(
        self,
        min_area: int = DEFAULT_MIN_CORNER_AREA,
        max_area: int = DEFAULT_MAX_CORNER_AREA
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.validator = CornerValidator()
        self.strategies = [
            ThresholdStrategy(),
            AdaptiveStrategy(),
            EdgeStrategy()
        ]

    def detect(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        Detect corner markers in the image.

        Args:
            image: Input image (BGR format)
            qr_polygon: QR code polygon to exclude

        Returns:
            Dictionary of corner positions or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Collect candidates from all strategies
        all_candidates = []
        for strategy in self.strategies:
            candidates = strategy.detect(
                gray, self.min_area, self.max_area, qr_polygon
            )
            all_candidates.extend(candidates)

        # Remove duplicates and validate
        unique_candidates = self._remove_duplicates(all_candidates)
        filtered_candidates = self.validator.filter_qr_patterns(
            unique_candidates, gray
        )

        # Select best corners
        corners = self._select_best_corners(
            filtered_candidates, width, height
        )

        if not self._validate_corners(corners):
            logger.warning("Could not detect all corners")
            return None

        return corners

    def _remove_duplicates(
        self,
        candidates: List[Dict],
        distance_threshold: int = 30
    ) -> List[Dict]:
        """Remove duplicate candidates based on proximity."""
        unique = []

        for candidate in candidates:
            if not self._is_duplicate(candidate, unique, distance_threshold):
                unique.append(candidate)

        return unique

    def _is_duplicate(
        self,
        candidate: Dict,
        existing: List[Dict],
        threshold: int
    ) -> bool:
        """Check if candidate is duplicate of existing ones."""
        cx, cy = candidate['center']

        for exist in existing:
            ex, ey = exist['center']
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if distance < threshold:
                return True

        return False

    def _select_best_corners(
        self,
        candidates: List[Dict],
        width: int,
        height: int
    ) -> Optional[Dict[str, Dict]]:
        """Select the best 4 corners from candidates."""
        if len(candidates) < 4:
            return None

        corner_positions = {
            'top_left': (0, 0),
            'top_right': (width, 0),
            'bottom_left': (0, height),
            'bottom_right': (width, height)
        }

        corners = {}
        used_indices = set()

        for corner_name, ideal_pos in corner_positions.items():
            best_idx, best_candidate = self._find_best_candidate(
                candidates, ideal_pos, used_indices
            )

            if best_idx >= 0:
                corners[corner_name] = best_candidate
                used_indices.add(best_idx)

        return corners

    def _find_best_candidate(
        self,
        candidates: List[Dict],
        ideal_pos: Tuple[int, int],
        used_indices: set
    ) -> Tuple[int, Optional[Dict]]:
        """Find best candidate for a corner position."""
        best_idx = -1
        best_score = -float('inf')
        best_candidate = None

        for i, candidate in enumerate(candidates):
            if i in used_indices:
                continue

            score = self._calculate_corner_score(candidate, ideal_pos)
            if score > best_score:
                best_score = score
                best_idx = i
                best_candidate = candidate

        return best_idx, best_candidate

    def _calculate_corner_score(
        self,
        candidate: Dict,
        ideal_pos: Tuple[int, int]
    ) -> float:
        """Calculate score for corner candidate."""
        cx, cy = candidate['center']
        ix, iy = ideal_pos

        # Distance score
        distance = np.sqrt((cx - ix)**2 + (cy - iy)**2)
        distance_score = 1.0 / (1.0 + distance / 100.0)

        # Property scores
        area_score = min(1.0, candidate.get('area', 0) / 1000.0)
        solidity_score = candidate.get('solidity', 0.7)

        return distance_score * 0.5 + area_score * 0.25 + solidity_score * 0.25

    def _validate_corners(self, corners: Optional[Dict]) -> bool:
        """Validate that all corners were found."""
        if corners is None:
            return False

        required = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        return all(corners.get(name) is not None for name in required)
