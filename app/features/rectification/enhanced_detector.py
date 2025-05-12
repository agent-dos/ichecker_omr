# In app/features/rectification/enhanced_detector.py

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
        # Lower default for test markers
        self.min_area = params.get('min_area', 100)
        self.max_area = params.get('max_area', 10000)  # Higher default
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
        logger.debug(f"Image shape: {image.shape}")

        # Step 1: Preprocess image
        preprocessed, prep_viz = self._preprocess_image(image, visualize_steps)
        viz_steps.update(prep_viz)

        # Step 2: Try multiple detection strategies
        candidates = []

        # Strategy 1: Simple thresholding (best for synthetic images)
        simple_candidates = self._detect_simple_threshold(preprocessed)
        candidates.extend(simple_candidates)
        logger.info(
            f"Simple threshold found {len(simple_candidates)} candidates")

        # Strategy 2: Direct on original grayscale (bypass preprocessing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
        direct_candidates = self._detect_simple_threshold(gray)
        candidates.extend(direct_candidates)
        logger.info(
            f"Direct threshold found {len(direct_candidates)} candidates")

        # Strategy 3: Adaptive threshold
        adaptive_candidates = self._detect_adaptive(preprocessed)
        candidates.extend(adaptive_candidates)
        logger.info(
            f"Adaptive strategy found {len(adaptive_candidates)} candidates")

        logger.info(
            f"Total candidates before deduplication: {len(candidates)}")

        # Step 3: Filter and select best corners
        unique_candidates = self._remove_duplicates(candidates)
        logger.info(
            f"Unique candidates after deduplication: {len(unique_candidates)}")

        # Log candidate positions for debugging
        for i, cand in enumerate(unique_candidates):
            center = cand.get('center', (0, 0))
            area = cand.get('area', 0)
            logger.debug(f"Candidate {i}: center={center}, area={area}")

        filtered_candidates = self._filter_candidates(unique_candidates)
        logger.info(f"Filtered candidates: {len(filtered_candidates)}")

        final_corners = self._select_best_corners(
            filtered_candidates, image.shape[1], image.shape[0]
        )

        if visualize_steps and final_corners:
            viz_steps["99_FinalDetection"] = visualize_corners(
                image, final_corners, "Enhanced Detection"
            )

        return final_corners, viz_steps

    def _preprocess_image(
        self,
        image: np.ndarray,
        visualize: bool
    ) -> Tuple[np.ndarray, Dict]:
        """Minimal preprocessing for synthetic images."""
        viz = {}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # For synthetic images, return grayscale directly
        if visualize:
            viz["01_Grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray, viz

    def _detect_simple_threshold(self, image: np.ndarray) -> List[Dict]:
        """Simple threshold detection - works well for synthetic images."""
        candidates = []

        # Try different threshold values
        for thresh_val in [50, 100, 127, 150, 200, 250]:
            try:
                _, binary = cv2.threshold(
                    image, thresh_val, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                logger.debug(
                    f"Threshold {thresh_val}: found {len(contours)} contours")

                for contour in contours:
                    candidate = self._analyze_contour(contour)
                    if candidate:
                        candidates.append(candidate)
            except Exception as e:
                logger.warning(
                    f"Simple threshold with thresh={thresh_val} failed: {e}")

        return candidates

    def _detect_adaptive(self, image: np.ndarray) -> List[Dict]:
        """Adaptive threshold detection."""
        candidates = []

        for blocksize in [11, 21, 31]:
            for c in [2, 5]:
                try:
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
                except Exception as e:
                    logger.warning(
                        f"Adaptive detection with blocksize={blocksize}, c={c} failed: {e}")

        return candidates

    def _analyze_contour(self, contour: np.ndarray) -> Optional[Dict]:
        """Analyze contour for corner marker properties."""
        area = cv2.contourArea(contour)

        # Be more lenient with area constraints for rotated squares
        if not (self.min_area < area < self.max_area):
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)

        # Check aspect ratio - very lenient for rotated squares
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.4 < aspect_ratio < 2.5):
            return None

        # For synthetic markers, we expect high solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < 0.8:  # High solidity for filled rectangles
            return None

        # Compute approximation to check if it's roughly rectangular
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # A rectangle should have 4 vertices (allowing some tolerance)
        num_vertices = len(approx)
        if num_vertices < 4 or num_vertices > 8:
            return None

        return {
            'contour': contour,
            'center': center,
            'area': area,
            'bbox': (x, y, w, h),
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'vertices': num_vertices
        }

    def _remove_duplicates(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Remove duplicate candidates based on proximity."""
        if not candidates:
            return []

        unique = []

        for candidate in candidates:
            is_duplicate = False
            cx, cy = candidate['center']

            for existing in unique:
                ex, ey = existing['center']
                distance = np.hypot(cx - ex, cy - ey)

                if distance < self.duplicate_threshold:
                    is_duplicate = True
                    # Keep the one with larger area (more likely to be the actual marker)
                    if candidate.get('area', 0) > existing.get('area', 0):
                        unique.remove(existing)
                        unique.append(candidate)
                    break

            if not is_duplicate:
                unique.append(candidate)

        return unique

    def _filter_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates - pass through for now."""
        return candidates

    def _select_best_corners(
        self,
        candidates: List[Dict],
        width: int,
        height: int
    ) -> Optional[Dict]:
        """Select the best 4 corners using spatial distribution."""
        if len(candidates) < 4:
            logger.warning(
                f"Only {len(candidates)} candidates found, need at least 4")
            return None

        # If we have exactly 4 candidates, just assign them based on position
        if len(candidates) == 4:
            return self._assign_corners_by_position(candidates)

        # Find the 4 candidates that form the most rectangular shape
        best_corners = self._find_most_rectangular_set(candidates)

        if best_corners:
            return best_corners

        # Fallback: Pick the 4 corners furthest from the center
        center_x, center_y = width // 2, height // 2

        # Sort by distance from center
        candidates_with_dist = []
        for cand in candidates:
            cx, cy = cand['center']
            dist = np.hypot(cx - center_x, cy - center_y)
            candidates_with_dist.append((dist, cand))

        # Sort by distance descending
        candidates_with_dist.sort(key=lambda x: -x[0])

        # Take the 4 furthest
        selected = [cand for _, cand in candidates_with_dist[:4]]

        return self._assign_corners_by_position(selected)

    def _assign_corners_by_position(self, candidates: List[Dict]) -> Dict:
        """Assign corner names based on relative positions."""
        # Sort by x coordinate
        sorted_by_x = sorted(candidates, key=lambda c: c['center'][0])

        # Split into left and right
        left_two = sorted_by_x[:2]
        right_two = sorted_by_x[2:]

        # Sort each pair by y coordinate
        left_two = sorted(left_two, key=lambda c: c['center'][1])
        right_two = sorted(right_two, key=lambda c: c['center'][1])

        return {
            'top_left': left_two[0],
            'bottom_left': left_two[1],
            'top_right': right_two[0],
            'bottom_right': right_two[1]
        }

    def _find_most_rectangular_set(self, candidates: List[Dict]) -> Optional[Dict]:
        """Find the set of 4 candidates that forms the most rectangular shape."""
        from itertools import combinations

        best_score = float('inf')
        best_set = None

        # Try all combinations of 4 candidates
        for combo in combinations(candidates, 4):
            # Calculate how "rectangular" this set is
            centers = [c['center'] for c in combo]

            # Sort to determine corner assignments
            sorted_by_x = sorted(centers, key=lambda p: p[0])
            left_two = sorted(sorted_by_x[:2], key=lambda p: p[1])
            right_two = sorted(sorted_by_x[2:], key=lambda p: p[1])

            tl = left_two[0]
            bl = left_two[1]
            tr = right_two[0]
            br = right_two[1]

            # Calculate distances between corners
            top_edge = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
            bottom_edge = np.hypot(br[0] - bl[0], br[1] - bl[1])
            left_edge = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
            right_edge = np.hypot(br[0] - tr[0], br[1] - tr[1])

            # Calculate diagonals
            diag1 = np.hypot(br[0] - tl[0], br[1] - tl[1])
            diag2 = np.hypot(bl[0] - tr[0], bl[1] - tr[1])

            # A perfect rectangle has equal opposite sides and equal diagonals
            edge_diff = abs(top_edge - bottom_edge) + \
                abs(left_edge - right_edge)
            diag_diff = abs(diag1 - diag2)

            # Also check angles (should be close to 90 degrees)
            # Using dot product to check orthogonality
            top_vec = np.array([tr[0] - tl[0], tr[1] - tl[1]])
            left_vec = np.array([bl[0] - tl[0], bl[1] - tl[1]])

            if np.linalg.norm(top_vec) > 0 and np.linalg.norm(left_vec) > 0:
                cos_angle = np.dot(
                    top_vec, left_vec) / (np.linalg.norm(top_vec) * np.linalg.norm(left_vec))
                # Should be close to 0 for 90 degrees
                angle_score = abs(cos_angle)
            else:
                angle_score = 1.0

            # Combined score (lower is better)
            score = edge_diff + diag_diff + angle_score * 100

            if score < best_score:
                best_score = score
                best_set = combo

        if best_set:
            return self._assign_corners_by_position(list(best_set))

        return None
