# app/features/rectification/enhanced_detector.py
"""
Enhanced corner detection for better handling of skewed and challenging images.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from app.features.corners.strategies.threshold import ThresholdStrategy
from app.features.corners.validators import CornerValidator
from app.features.corners.visualizer import visualize_corners

logger = logging.getLogger(__name__)


class EnhancedCornerDetector:
    """Enhanced corner detection with preprocessing and multiple strategies."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize enhanced corner detector."""
        self.params = params
        self.min_area = params.get('min_area', 200)
        self.max_area = params.get('max_area', 8000)
        self.duplicate_threshold = params.get('duplicate_threshold', 30)
        self.validator = CornerValidator(params.get('validator', {}))

    def detect(
        self,
        image: np.ndarray,
        visualize_steps: bool = False
    ) -> Tuple[Optional[Dict], Dict[str, np.ndarray]]:
        """Detect corners with enhanced preprocessing."""
        viz_steps = {}
        logger.info("Starting enhanced corner detection")

        # Step 1: Preprocess image
        preprocessed, prep_viz = self._preprocess_image(image, visualize_steps)
        viz_steps.update(prep_viz)

        # Step 2: Try multiple detection strategies
        candidates = []

        # Strategy 1: Adaptive threshold
        adaptive_candidates = self._detect_adaptive(preprocessed)
        candidates.extend(adaptive_candidates)

        # Strategy 2: Morphology-based
        morph_candidates = self._detect_morphology(preprocessed)
        candidates.extend(morph_candidates)

        # Strategy 3: Edge-based
        edge_candidates = self._detect_edges(preprocessed)
        candidates.extend(edge_candidates)

        # Step 3: Filter and select best corners
        unique_candidates = self._remove_duplicates(candidates)
        filtered_candidates = self._filter_candidates(unique_candidates)
        final_corners = self._select_best_corners(
            filtered_candidates, image.shape[1], image.shape[0]
        )

        if visualize_steps:
            viz_steps["99_FinalDetection"] = visualize_corners(
                image, final_corners, "Enhanced Detection"
            )

        return final_corners, viz_steps

    def _preprocess_image(
        self,
        image: np.ndarray,
        visualize: bool
    ) -> Tuple[np.ndarray, Dict]:
        """Enhanced preprocessing for difficult images."""
        viz = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        if visualize:
            viz["01_Enhanced"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            viz["02_Denoised"] = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return denoised, viz

    def _detect_adaptive(self, image: np.ndarray) -> List[Dict]:
        """Adaptive threshold detection."""
        candidates = []

        for blocksize in [21, 31, 41]:
            for c in [5, 10, 15]:
                binary = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, blocksize, c
                )

                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    candidate = self._analyze_contour(contour)
                    if candidate:
                        candidates.append(candidate)

        return candidates

    def _detect_morphology(self, image: np.ndarray) -> List[Dict]:
        """Morphology-based detection."""
        candidates = []

        for thresh_val in [50, 100, 150]:
            _, binary = cv2.threshold(
                image, thresh_val, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                candidate = self._analyze_contour(contour)
                if candidate:
                    candidates.append(candidate)

        return candidates

    def _detect_edges(self, image: np.ndarray) -> List[Dict]:
        """Edge-based detection."""
        candidates = []

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            candidate = self._analyze_contour(contour)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _analyze_contour(self, contour: np.ndarray) -> Optional[Dict]:
        """Analyze contour for corner marker properties."""
        area = cv2.contourArea(contour)
        if not (self.min_area < area < self.max_area):
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)

        # Check aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.5 < aspect_ratio < 2.0):
            return None

        # Check solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < 0.8:
            return None

        return {
            'contour': contour,
            'center': center,
            'area': area,
            'bbox': (x, y, w, h),
            'solidity': solidity,
            'aspect_ratio': aspect_ratio
        }

    def _remove_duplicates(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Remove duplicate candidates."""
        unique = []

        for candidate in candidates:
            is_duplicate = False
            cx, cy = candidate['center']

            for existing in unique:
                ex, ey = existing['center']
                distance = np.hypot(cx - ex, cy - ey)

                if distance < self.duplicate_threshold:
                    is_duplicate = True
                    # Keep the one with better solidity
                    if candidate['solidity'] > existing['solidity']:
                        unique.remove(existing)
                        unique.append(candidate)
                    break

            if not is_duplicate:
                unique.append(candidate)

        return unique

    def _filter_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates based on corner marker properties."""
        filtered = []

        for candidate in candidates:
            # Additional filtering based on corner marker characteristics
            bbox = candidate['bbox']
            x, y, w, h = bbox

            # Check if square-like
            squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if squareness < 0.7:
                continue

            # Check fill ratio
            contour = candidate['contour']
            mask = np.zeros((h+2, w+2), dtype=np.uint8)
            cv2.drawContours(
                mask, [contour - np.array([x-1, y-1])], -1, 255, -1)

            fill_ratio = cv2.countNonZero(mask) / (w * h) if (w * h) > 0 else 0
            if fill_ratio < 0.85:
                continue

            filtered.append(candidate)

        return filtered

    def _select_best_corners(
        self,
        candidates: List[Dict],
        width: int,
        height: int
    ) -> Optional[Dict]:
        """Select the best 4 corners from candidates."""
        if len(candidates) < 4:
            logger.warning(f"Only {len(candidates)} candidates found")
            return None

        # Expected corner positions
        corner_positions = {
            'top_left': (0, 0),
            'top_right': (width, 0),
            'bottom_left': (0, height),
            'bottom_right': (width, height)
        }

        corners = {}

        for corner_name, ideal_pos in corner_positions.items():
            best_candidate = None
            best_distance = float('inf')

            for candidate in candidates:
                cx, cy = candidate['center']
                ix, iy = ideal_pos
                distance = np.hypot(cx - ix, cy - iy)

                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

            if best_candidate:
                corners[corner_name] = best_candidate

        return corners if len(corners) == 4 else None
