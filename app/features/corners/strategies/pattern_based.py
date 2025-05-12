# app/features/corners/strategies/pattern_based.py
"""
Pattern-based corner detection strategy for advanced markers.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PatternBasedStrategy:
    """
    Detects corners using pattern matching for advanced markers.
    """

    def __init__(self, params: Dict):
        """Initialize with pattern detection parameters."""
        self.params = params
        self.pattern_type = params.get('pattern_type', 'l_shape')
        self.enabled = params.get('enabled', True)
        self.debug = params.get('debug', False)
        self.overlap_threshold = params.get('overlap_threshold', 30)

    def detect(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]],
        visualize_steps: bool = False
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Detect corner patterns using template matching or feature detection.
        """
        viz_steps = {}
        candidates = []

        if not self.enabled:
            return [], viz_steps

        logger.info(f"Pattern-based detection using: {self.pattern_type}")

        if self.pattern_type == 'l_shape':
            candidates = self._detect_l_shapes(
                gray, min_area, max_area, viz_steps, visualize_steps
            )
        elif self.pattern_type == 'concentric':
            candidates = self._detect_concentric_squares(
                gray, min_area, max_area, viz_steps, visualize_steps
            )

        logger.info(f"Pattern detection found {len(candidates)} candidates")
        return candidates, viz_steps

    def _detect_l_shapes(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        viz_steps: Dict,
        visualize: bool
    ) -> List[Dict]:
        """
        Detect L-shaped corner patterns with corner-specific detection.
        """
        all_candidates = []
        h, w = gray.shape

        # Define regions where each corner type should be found
        corner_regions = {
            'top_left': (0, 0, w//3, h//3),
            'top_right': (2*w//3, 0, w, h//3),
            'bottom_left': (0, 2*h//3, w//3, h),
            'bottom_right': (2*w//3, 2*h//3, w, h)
        }

        # Try multiple threshold values
        for thresh_val in [50, 100, 150, 200]:
            _, binary = cv2.threshold(
                gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # Apply morphology to connect L-shape components
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            if visualize:
                viz_steps[f"l_shape_thresh_{thresh_val}"] = cv2.cvtColor(
                    binary, cv2.COLOR_GRAY2BGR
                )

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if not (min_area < area < max_area):
                    continue

                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(contour)

                # Determine which corner region this contour is in
                corner_type = self._determine_corner_type(
                    x + bw//2, y + bh//2, corner_regions
                )

                if corner_type and self._is_l_shaped(contour, corner_type):
                    center = (x + bw//2, y + bh//2)
                    all_candidates.append({
                        'contour': contour,
                        'center': center,
                        'area': area,
                        'bbox': (x, y, bw, bh),
                        'corner_type': corner_type,
                        'confidence': 0.9,
                        'threshold': thresh_val
                    })

        # Deduplicate candidates from different threshold levels
        final_candidates = self._deduplicate_candidates(all_candidates)

        if visualize:
            self._visualize_candidates(gray, final_candidates, viz_steps)

        return final_candidates

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove duplicate candidates that overlap significantly.
        Keep the one with the best properties (largest area, best confidence).
        """
        if not candidates:
            return []

        # Group candidates by corner type
        corner_groups = {}
        for candidate in candidates:
            corner_type = candidate.get('corner_type', 'unknown')
            if corner_type not in corner_groups:
                corner_groups[corner_type] = []
            corner_groups[corner_type].append(candidate)

        # Select best candidate per corner type
        final_candidates = []
        for corner_type, group in corner_groups.items():
            if not group:
                continue

            # Sort by area (descending) and confidence
            group.sort(key=lambda x: (x.get('confidence', 0), x.get('area', 0)),
                       reverse=True)

            # Take the best candidate for this corner
            best_candidate = group[0]

            # Check if there are other very close candidates we should merge
            cx, cy = best_candidate['center']
            merged_area = best_candidate['area']

            for other in group[1:]:
                ox, oy = other['center']
                distance = np.sqrt((cx - ox)**2 + (cy - oy)**2)

                if distance < self.overlap_threshold:
                    # Merge properties if they're very close
                    merged_area = max(merged_area, other['area'])

            best_candidate['area'] = merged_area
            final_candidates.append(best_candidate)

        return final_candidates

    def _determine_corner_type(
        self,
        x: int,
        y: int,
        corner_regions: Dict
    ) -> Optional[str]:
        """Determine which corner region a point belongs to."""
        for corner_type, (x1, y1, x2, y2) in corner_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return corner_type
        return None

    def _is_l_shaped(self, contour: np.ndarray, corner_type: str) -> bool:
        """
        Check if contour is L-shaped for specific corner type.
        """
        # Get convex hull and analyze shape
        hull = cv2.convexHull(contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # L-shape should have 5-7 vertices when approximated
        if not (5 <= len(approx) <= 7):
            return False

        # Check if the shape has the right orientation for the corner
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return False

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Calculate shape properties
        x, y, w, h = cv2.boundingRect(contour)

        # Check aspect ratio - L shapes aren't square
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.7 <= aspect_ratio <= 1.3):
            # L-shape should be roughly square in bounding box
            return True

        # Additional verification based on corner type
        return self._verify_l_orientation(contour, corner_type)

    def _verify_l_orientation(self, contour: np.ndarray, corner_type: str) -> bool:
        """
        Verify L-shape has correct orientation for corner type.
        """
        # Create a mask to analyze the shape
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Shift contour to origin
        shifted_contour = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

        # Divide into quadrants
        mid_x, mid_y = w // 2, h // 2

        # Count pixels in each quadrant
        q1 = np.sum(mask[0:mid_y, 0:mid_x])  # Top-left
        q2 = np.sum(mask[0:mid_y, mid_x:])   # Top-right
        q3 = np.sum(mask[mid_y:, 0:mid_x])   # Bottom-left
        q4 = np.sum(mask[mid_y:, mid_x:])    # Bottom-right

        # Expected patterns for each corner type
        if corner_type == 'top_left':
            # L should fill top and left
            return q1 > q4 and (q1 + q2 + q3) > 2 * q4
        elif corner_type == 'top_right':
            # L should fill top and right
            return q2 > q3 and (q1 + q2 + q4) > 2 * q3
        elif corner_type == 'bottom_left':
            # L should fill bottom and left
            return q3 > q2 and (q1 + q3 + q4) > 2 * q2
        elif corner_type == 'bottom_right':
            # L should fill bottom and right
            return q4 > q1 and (q2 + q3 + q4) > 2 * q1

        return False

    def _visualize_candidates(
        self,
        gray: np.ndarray,
        candidates: List[Dict],
        viz_steps: Dict
    ) -> None:
        """Create visualization of detected L-shape candidates."""
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw each candidate with color based on corner type
        colors = {
            'top_left': (0, 255, 0),      # Green
            'top_right': (255, 0, 0),     # Blue
            'bottom_left': (0, 255, 255),  # Yellow
            'bottom_right': (255, 0, 255)  # Magenta
        }

        for candidate in candidates:
            contour = candidate['contour']
            corner_type = candidate.get('corner_type', 'unknown')
            color = colors.get(corner_type, (128, 128, 128))

            cv2.drawContours(viz, [contour], -1, color, 2)

            # Draw center point
            cx, cy = candidate['center']
            cv2.circle(viz, (cx, cy), 5, color, -1)

            # Label corner type
            cv2.putText(viz, corner_type, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        viz_steps["l_shape_candidates"] = viz

    def _detect_concentric_squares(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        viz_steps: Dict,
        visualize: bool
    ) -> List[Dict]:
        """Existing concentric detection method."""
        # Implementation remains the same as before
        return []
