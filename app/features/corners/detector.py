# filename: app/features/corners/detector.py
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any  # Added Any

# Import components
from app.features.corners.strategies.enhanced_l_shape import EnhancedLShapeStrategy
from app.features.corners.strategies.pattern_based import PatternBasedStrategy
from app.features.corners.strategies.threshold import ThresholdStrategy
from app.features.corners.strategies.adaptive import AdaptiveStrategy
from app.features.corners.strategies.edge import EdgeStrategy
from app.features.corners.validators import CornerValidator
from app.features.corners.visualizer import visualize_corners  # For final viz fallback

logger = logging.getLogger(__name__)


class CornerDetector:
    """ Detects corner markers using multiple strategies and parameters. """

    def __init__(self, params: Dict[str, Any]):
        """ Initialize with corner_detection parameters. """
        self.params = params
        self.min_area = self.params.get('min_area', 300)
        self.max_area = self.params.get('max_area', 5000)
        self.duplicate_threshold = self.params.get('duplicate_threshold', 30)
        self.scoring_params = self.params.get('scoring', {})

        # Initialize validator
        self.validator = CornerValidator(self.params.get('validator', {}))

        # Initialize strategies with their specific params
        self.strategies = []
        if self.params.get('strategy_threshold', {}).get('enabled', True):
            self.strategies.append(ThresholdStrategy(
                self.params.get('strategy_threshold', {})))
        if self.params.get('strategy_adaptive', {}).get('enabled', True):
            # Ensure AdaptiveStrategy is updated similarly to ThresholdStrategy
            # self.strategies.append(AdaptiveStrategy(self.params.get('strategy_adaptive', {})))
            # Placeholder
            logger.warning(
                "AdaptiveStrategy not fully implemented for viz/params yet.")
        if self.params.get('strategy_edge', {}).get('enabled', True):
            # Ensure EdgeStrategy is updated similarly to ThresholdStrategy
            # self.strategies.append(EdgeStrategy(self.params.get('strategy_edge', {})))
            # Placeholder
            logger.warning(
                "EdgeStrategy not fully implemented for viz/params yet.")
        if self.params.get('strategy_enhanced_l_shape', {}).get('enabled', True):
            self.strategies.append(EnhancedLShapeStrategy(
                self.params.get('strategy_enhanced_l_shape', {})
            ))

        logger.debug(
            f"CornerDetector initialized with {len(self.strategies)} enabled strategies.")

    def detect(
        self,
        image: np.ndarray,
        qr_polygon: Optional[List[Tuple[int, int]]] = None,
        visualize_steps: bool = False
    ) -> Tuple[Optional[Dict[str, Dict]], Dict[str, np.ndarray]]:
        """
        Detect corner markers, returning final corners and visualization steps.
        """
        viz_steps = {}
        final_corners = None

        logger.info("--- Starting Corner Detection ---")
        if image is None:
            logger.error("Corner detection received None image.")
            return None, viz_steps

        # --- Step 1: Preprocessing ---
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            if visualize_steps:
                viz_steps["01_GrayscaleInput"] = cv2.cvtColor(
                    gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.error(
                f"Error during grayscale conversion: {e}", exc_info=True)
            return None, viz_steps

        # --- Step 2: Run Strategies ---
        all_candidates = []
        strategy_base_img = cv2.cvtColor(
            gray, cv2.COLOR_GRAY2BGR) if visualize_steps else None  # For drawing contours

        for i, strategy in enumerate(self.strategies):
            strategy_name = type(strategy).__name__
            logger.info(f"Running strategy: {strategy_name}")
            try:
                # Pass min/max area, qr_polygon, visualize flag
                strategy_candidates, strategy_viz = strategy.detect(
                    gray, self.min_area, self.max_area, qr_polygon, visualize_steps
                )
                all_candidates.extend(strategy_candidates)
                # Merge visualization steps with prefix
                if visualize_steps:
                    for k, v in strategy_viz.items():
                        viz_steps[f"02_{strategy_name}_{k}"] = v
                logger.info(
                    f"{strategy_name} found {len(strategy_candidates)} candidates.")
            except Exception as e:
                logger.error(
                    f"Error running strategy {strategy_name}: {e}", exc_info=True)

        logger.info(
            f"Total raw candidates from strategies: {len(all_candidates)}")
        if visualize_steps and all_candidates:
            viz_raw_cand = strategy_base_img.copy()
            for cand in all_candidates:
                center = tuple(map(int, cand.get('center', (0, 0))))
                if center != (0, 0):
                    cv2.circle(viz_raw_cand, center, 4,
                               (0, 165, 255), 1)  # Orange circles
            viz_steps["03_RawCandidates_All"] = viz_raw_cand

        # --- Step 3: Deduplicate Candidates ---
        unique_candidates = self._remove_duplicates(
            all_candidates, self.duplicate_threshold)
        logger.info(
            f"Candidates after deduplication: {len(unique_candidates)}")
        if visualize_steps and unique_candidates:
            viz_unique_cand = strategy_base_img.copy()
            for cand in unique_candidates:
                center = tuple(map(int, cand.get('center', (0, 0))))
                if center != (0, 0):
                    cv2.circle(viz_unique_cand, center, 5,
                               (255, 255, 0), 1)  # Cyan circles
            viz_steps["04_UniqueCandidates"] = viz_unique_cand

        # --- Step 4: Validate (Filter QR) ---
        try:
            filtered_candidates, validation_viz = self.validator.filter_qr_patterns(
                unique_candidates, gray, visualize_steps
            )
            # Merge viz steps
            if visualize_steps:
                for k, v in validation_viz.items():
                    viz_steps[f"05_Validation_{k}"] = v
            logger.info(
                f"Candidates after QR validation: {len(filtered_candidates)}")
        except Exception as e:
            logger.error(f"Error during corner validation: {e}", exc_info=True)
            # Fallback: use unvalidated if validator fails
            filtered_candidates = unique_candidates

        # --- Step 5: Select Best Corners ---
        if len(filtered_candidates) < 4:
            logger.warning(
                f"Insufficient candidates ({len(filtered_candidates)}) remaining after filtering to select 4 corners.")
            final_corners = None
        else:
            logger.info(
                f"Selecting best 4 corners from {len(filtered_candidates)} candidates...")
            final_corners = self._select_best_corners(
                # Pass width, height
                filtered_candidates, gray.shape[1], gray.shape[0]
            )
            if final_corners and len(final_corners) == 4:
                logger.info("Successfully selected 4 corners.")
            else:
                logger.warning(
                    f"Failed to select 4 distinct corners. Selected: {len(final_corners) if final_corners else 'None'}")
                final_corners = None  # Ensure it's None if selection failed

        # --- Final Visualization ---
        if visualize_steps:
            # Use the standard visualizer for the final output
            final_viz_img = visualize_corners(
                strategy_base_img, final_corners)  # Use BGR base image
            viz_steps["99_FinalDetection"] = final_viz_img

        logger.info(
            f"--- Corner Detection Finished. Found: {len(final_corners) if final_corners else 'None'} corners. ---")
        return final_corners, viz_steps

    def _remove_duplicates(self, candidates: List[Dict], distance_threshold: int) -> List[Dict]:
        """Remove duplicate candidates based on proximity."""
        if not candidates:
            return []

        unique: List[Dict] = []
        for candidate in candidates:
            is_duplicate = False
            cx, cy = candidate.get('center', (None, None))
            if cx is None:
                continue

            for i, existing in enumerate(unique):
                ex, ey = existing.get('center', (None, None))
                if ex is None:
                    continue

                distance = np.hypot(cx - ex, cy - ey)
                if distance < distance_threshold:
                    is_duplicate = True
                    # Keep the one with larger area
                    if candidate.get('area', 0) > existing.get('area', 0):
                        unique[i] = candidate  # Replace with larger area
                    break

            if not is_duplicate:
                unique.append(candidate)

        return unique

    # --- _select_best_corners and _calculate_corner_score need scoring params ---
    def _select_best_corners(
        self, candidates: List[Dict], width: int, height: int
    ) -> Optional[Dict[str, Dict]]:
        """ Select the best 4 corners using scoring based on position and properties. """
        # Ideal corner positions (0,0), (W,0), (0,H), (W,H)
        corner_positions = {
            'top_left': (0, 0),
            'top_right': (width, 0),
            'bottom_left': (0, height),
            'bottom_right': (width, height)
        }
        corners = {}
        # Use a copy to allow removing candidates as they are assigned
        remaining_candidates = list(candidates)

        for corner_name, ideal_pos in corner_positions.items():
            best_candidate = None
            best_score = -float('inf')
            best_idx = -1

            # Find the best match among remaining candidates
            for i, candidate in enumerate(remaining_candidates):
                score = self._calculate_corner_score(candidate, ideal_pos)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_idx = i

            # Assign the best match and remove it from future consideration
            if best_candidate is not None and best_idx != -1:
                corners[corner_name] = best_candidate
                # Remove by index to avoid issues if candidates are identical dicts
                del remaining_candidates[best_idx]
                logger.debug(
                    f"Assigned candidate (idx {best_idx} orig) to {corner_name} with score {best_score:.3f}")
            else:
                logger.warning(
                    f"No suitable candidate found for corner: {corner_name}")
                # If one corner isn't found, should we return None or partial dict?
                # Let's return partial for now, final check in detect() handles None/incomplete
                # Explicitly mark as None if not found
                corners[corner_name] = None

        # Final check if all 4 were assigned
        if all(corners.values()):
            return corners
        else:
            logger.warning("Could not assign all 4 corners.")
            return None  # Return None if not all 4 were found

    def _calculate_corner_score(self, candidate: Dict, ideal_pos: Tuple) -> float:
        """ Calculate score based on distance and potentially other properties. """
        cx, cy = candidate.get('center', (None, None))
        if cx is None:
            return -float('inf')

        ix, iy = ideal_pos

        # 1. Distance Score (Inversely proportional to distance, normalized)
        # Avoid division by zero if distance is huge? Use max distance like diagonal?
        # max_dist = np.hypot(width, height) # Needs width/height passed in
        distance = np.hypot(cx - ix, cy - iy)
        # Simple normalization: score decreases as distance increases
        # Adjust the denominator to control sensitivity
        distance_score = 1.0 / (1.0 + distance / 100.0)

        # 2. Property Scores (if available and weights are set)
        area = candidate.get('area', 0)
        solidity = candidate.get('solidity', 0)  # Default to 0 if missing

        # Normalize area score (e.g., based on expected range or max area param)
        area_norm_factor = self.scoring_params.get('area_norm_factor', 1000.0)
        # Avoid div by zero
        area_score = min(1.0, float(area) / max(1.0, area_norm_factor))

        # Solidity score is already 0-1 roughly
        solidity_score = max(0.0, min(1.0, solidity))

        # Weighted sum
        dist_w = self.scoring_params.get('distance_weight', 0.5)
        area_w = self.scoring_params.get('area_weight', 0.25)
        solid_w = self.scoring_params.get('solidity_weight', 0.25)

        # Normalize weights if they don't sum to 1 (optional but good practice)
        total_weight = dist_w + area_w + solid_w
        if total_weight <= 0:
            total_weight = 1.0  # Avoid division by zero

        final_score = (
            (distance_score * dist_w) +
            (area_score * area_w) +
            (solidity_score * solid_w)
        ) / total_weight

        return final_score
