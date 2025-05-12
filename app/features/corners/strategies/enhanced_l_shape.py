# app/features/corners/strategies/enhanced_l_shape.py
"""
Enhanced L-shape corner detection strategy that handles any orientation.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedLShapeStrategy:
    """
    Detects L-shaped corners in any orientation using robust shape analysis.
    """

    def __init__(self, params: Dict):
        """Initialize with detection parameters."""
        self.params = params
        self.enabled = params.get('enabled', True)
        self.debug = params.get('debug', False)
        self.min_ratio_threshold = params.get('min_ratio_threshold', 0.3)
        self.max_ratio_threshold = params.get('max_ratio_threshold', 0.7)

    def detect(
        self,
        gray: np.ndarray,
        min_area: int,
        max_area: int,
        qr_polygon: Optional[List[Tuple[int, int]]],
        visualize_steps: bool = False
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Detect L-shaped corners using enhanced shape analysis.
        """
        viz_steps = {}
        if not self.enabled:
            return [], viz_steps

        candidates = []
        h, w = gray.shape

        # Define search regions for corners
        corner_regions = {
            'top_left': (0, 0, int(w * 0.3), int(h * 0.3)),
            'top_right': (int(w * 0.7), 0, w, int(h * 0.3)),
            'bottom_left': (0, int(h * 0.7), int(w * 0.3), h),
            'bottom_right': (int(w * 0.7), int(h * 0.7), w, h)
        }

        # Process each corner region separately
        for corner_name, (x1, y1, x2, y2) in corner_regions.items():
            roi = gray[y1:y2, x1:x2]

            # Detect L-shape in this region
            region_candidates = self._detect_l_in_region(
                roi, corner_name, (x1, y1), min_area, max_area
            )

            candidates.extend(region_candidates)

            if visualize_steps:
                viz_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                # Draw detected shapes
                for cand in region_candidates:
                    contour = cand['contour']
                    cv2.drawContours(viz_roi, [contour], -1, (0, 255, 0), 2)
                viz_steps[f"region_{corner_name}"] = viz_roi

        # Final candidate selection
        final_candidates = self._select_best_corners(candidates)

        if visualize_steps:
            self._create_final_visualization(gray, final_candidates, viz_steps)

        logger.info(
            f"Enhanced L-shape detection found {len(final_candidates)} corners")
        return final_candidates, viz_steps

    def _detect_l_in_region(
        self,
        roi: np.ndarray,
        corner_name: str,
        offset: Tuple[int, int],
        min_area: int,
        max_area: int
    ) -> List[Dict]:
        """
        Detect L-shape in a specific region using multiple techniques.
        """
        candidates = []

        # Method 1: Morphological approach
        morph_candidates = self._morphological_detection(
            roi, corner_name, offset, min_area, max_area
        )
        candidates.extend(morph_candidates)

        # Method 2: Template matching approach
        template_candidates = self._template_matching_detection(
            roi, corner_name, offset, min_area, max_area
        )
        candidates.extend(template_candidates)

        # Method 3: Structural analysis
        struct_candidates = self._structural_analysis_detection(
            roi, corner_name, offset, min_area, max_area
        )
        candidates.extend(struct_candidates)

        # Merge and deduplicate candidates
        return self._merge_candidates(candidates)

    def _morphological_detection(
        self,
        roi: np.ndarray,
        corner_name: str,
        offset: Tuple[int, int],
        min_area: int,
        max_area: int
    ) -> List[Dict]:
        """
        Detect L-shapes using morphological operations.
        """
        candidates = []

        # Apply multiple preprocessing approaches
        for thresh_value in [50, 100, 150]:
            # Threshold
            _, binary = cv2.threshold(
                roi, thresh_value, 255, cv2.THRESH_BINARY_INV)

            # Morphological operations to connect L-shape components
            kernel_sizes = [(3, 3), (5, 5)]
            for ksize in kernel_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Find contours
                contours, _ = cv2.findContours(
                    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    if self._is_l_shaped_contour(contour, min_area, max_area):
                        # Adjust contour coordinates to global space
                        global_contour = contour + \
                            np.array([offset[0], offset[1]])

                        area = cv2.contourArea(contour)
                        x, y, w, h = cv2.boundingRect(contour)

                        candidates.append({
                            'contour': global_contour,
                            'center': (x + w//2 + offset[0], y + h//2 + offset[1]),
                            'area': area,
                            'bbox': (x + offset[0], y + offset[1], w, h),
                            'corner_type': corner_name,
                            'method': 'morphological',
                            'confidence': self._calculate_l_confidence(contour)
                        })

        return candidates

    def _template_matching_detection(
        self,
        roi: np.ndarray,
        corner_name: str,
        offset: Tuple[int, int],
        min_area: int,
        max_area: int
    ) -> List[Dict]:
        """
        Detect L-shapes using template matching.
        """
        candidates = []

        # Generate L-shape templates for different orientations
        templates = self._generate_l_templates(corner_name)

        for template in templates:
            # Resize template to match expected size
            for scale in [0.8, 1.0, 1.2]:
                resized_template = cv2.resize(
                    template, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LINEAR
                )

                # Skip if template is larger than ROI
                if (resized_template.shape[0] > roi.shape[0] or
                        resized_template.shape[1] > roi.shape[1]):
                    continue

                # Match template
                result = cv2.matchTemplate(
                    roi, resized_template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.7
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    w, h = resized_template.shape[::-1]

                    # Create bounding rectangle
                    x, y = pt
                    area = w * h

                    if min_area < area < max_area:
                        # Create contour from rectangle
                        rect_pts = np.array([
                            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                        ], dtype=np.int32)

                        candidates.append({
                            'contour': rect_pts + np.array([offset[0], offset[1]]),
                            'center': (x + w//2 + offset[0], y + h//2 + offset[1]),
                            'area': area,
                            'bbox': (x + offset[0], y + offset[1], w, h),
                            'corner_type': corner_name,
                            'method': 'template',
                            'confidence': result[y, x]
                        })

        return candidates

    def _structural_analysis_detection(
        self,
        roi: np.ndarray,
        corner_name: str,
        offset: Tuple[int, int],
        min_area: int,
        max_area: int
    ) -> List[Dict]:
        """
        Detect L-shapes using structural analysis of edges.
        """
        candidates = []

        # Edge detection
        edges = cv2.Canny(roi, 30, 100)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=20,
            minLineLength=10, maxLineGap=5
        )

        if lines is not None:
            # Group lines that might form L-shapes
            l_shapes = self._group_lines_to_l_shapes(lines)

            for l_shape in l_shapes:
                # Convert line segments to contour
                contour = self._lines_to_contour(l_shape)
                area = cv2.contourArea(contour)

                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    candidates.append({
                        'contour': contour + np.array([offset[0], offset[1]]),
                        'center': (x + w//2 + offset[0], y + h//2 + offset[1]),
                        'area': area,
                        'bbox': (x + offset[0], y + offset[1], w, h),
                        'corner_type': corner_name,
                        'method': 'structural',
                        'confidence': 0.8
                    })

        return candidates

    def _is_l_shaped_contour(
        self,
        contour: np.ndarray,
        min_area: int,
        max_area: int
    ) -> bool:
        """
        Check if a contour represents an L-shape.
        """
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            return False

        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate aspect ratio of bounding box
        width = rect[1][0]
        height = rect[1][1]
        if width == 0 or height == 0:
            return False

        aspect_ratio = max(width, height) / min(width, height)

        # L-shapes typically have aspect ratio close to 1
        if not (0.7 < aspect_ratio < 1.3):
            return False

        # Check solidity (area ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False

        solidity = area / hull_area

        # L-shapes have lower solidity than filled rectangles
        if not (0.4 < solidity < 0.8):
            return False

        # Check for L-shape characteristics
        return self._analyze_shape_structure(contour)

    def _analyze_shape_structure(self, contour: np.ndarray) -> bool:
        """
        Analyze contour structure to determine if it's L-shaped.
        """
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # L-shapes typically have 5-8 vertices when approximated
        num_vertices = len(approx)
        if not (5 <= num_vertices <= 8):
            return False

        # Analyze angles between vertices
        angles = []
        for i in range(num_vertices):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % num_vertices][0]
            p3 = approx[(i + 2) % num_vertices][0]

            # Calculate angle
            v1 = p1 - p2
            v2 = p3 - p2

            angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            angles.append(np.degrees(angle))

        # L-shapes should have at least one ~90 degree angle
        right_angles = sum(1 for angle in angles if 80 < angle < 100)

        return right_angles >= 1

    def _calculate_l_confidence(self, contour: np.ndarray) -> float:
        """
        Calculate confidence score for L-shape detection.
        """
        # Multiple factors contribute to confidence
        confidence = 0.5  # Base confidence

        # Check area consistency
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            # L-shapes have lower circularity
            if 0.3 < circularity < 0.6:
                confidence += 0.2

        # Check convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if 0.4 < solidity < 0.7:
                confidence += 0.2

        # Check polygon approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if 5 <= len(approx) <= 8:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_l_templates(self, corner_name: str) -> List[np.ndarray]:
        """
        Generate L-shape templates for different orientations.
        """
        templates = []
        size = 30  # Template size
        thickness = 8  # Line thickness

        # Create base L-shape
        template = np.zeros((size, size), dtype=np.uint8)

        if corner_name == 'top_left':
            cv2.rectangle(template, (0, 0), (size, thickness), 255, -1)
            cv2.rectangle(template, (0, 0), (thickness, size), 255, -1)
        elif corner_name == 'top_right':
            cv2.rectangle(template, (0, 0), (size, thickness), 255, -1)
            cv2.rectangle(template, (size - thickness, 0),
                          (size, size), 255, -1)
        elif corner_name == 'bottom_left':
            cv2.rectangle(template, (0, size - thickness),
                          (size, size), 255, -1)
            cv2.rectangle(template, (0, 0), (thickness, size), 255, -1)
        else:  # bottom_right
            cv2.rectangle(template, (0, size - thickness),
                          (size, size), 255, -1)
            cv2.rectangle(template, (size - thickness, 0),
                          (size, size), 255, -1)

        # Add rotated versions
        templates.append(template)
        for angle in [5, -5, 10, -10]:  # Small rotations
            M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
            rotated = cv2.warpAffine(template, M, (size, size))
            templates.append(rotated)

        return templates

    def _group_lines_to_l_shapes(
        self,
        lines: np.ndarray
    ) -> List[List[np.ndarray]]:
        """
        Group line segments that might form L-shapes.
        """
        if lines is None or len(lines) < 2:
            return []

        l_shapes = []
        used_lines = set()

        for i, line1 in enumerate(lines):
            if i in used_lines:
                continue

            x1, y1, x2, y2 = line1[0]

            for j, line2 in enumerate(lines):
                if i == j or j in used_lines:
                    continue

                x3, y3, x4, y4 = line2[0]

                # Check if lines are perpendicular and connected
                if self._lines_form_l(line1[0], line2[0]):
                    l_shapes.append([line1, line2])
                    used_lines.add(i)
                    used_lines.add(j)
                    break

        return l_shapes

    def _lines_form_l(self, line1: np.ndarray, line2: np.ndarray) -> bool:
        """
        Check if two lines form an L-shape.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x4 - x3, y4 - y3])

        # Check if perpendicular (dot product close to 0)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            return False

        angle_cos = abs(dot_product / norm_product)

        # Check if angle is close to 90 degrees
        if angle_cos > 0.2:  # cos(90°) = 0, allow some tolerance
            return False

        # Check if lines are connected
        endpoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        for p1 in endpoints[:2]:
            for p2 in endpoints[2:]:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < 10:  # Threshold for connectivity
                    return True

        return False

    def _lines_to_contour(self, lines: List[np.ndarray]) -> np.ndarray:
        """
        Convert line segments to a contour.
        """
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])

        # Remove duplicates and order points
        unique_points = list(set(points))
        if len(unique_points) < 3:
            return np.array(unique_points)

        # Create convex hull to form contour
        hull = cv2.convexHull(np.array(unique_points))
        return hull

    def _merge_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Merge overlapping candidates from different detection methods.
        """
        if not candidates:
            return []

        merged = []
        used = set()

        for i, cand1 in enumerate(candidates):
            if i in used:
                continue

            # Find overlapping candidates
            overlapping = [cand1]
            cx1, cy1 = cand1['center']

            for j, cand2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue

                cx2, cy2 = cand2['center']
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

                if dist < 30:  # Threshold for overlap
                    overlapping.append(cand2)
                    used.add(j)

            # Merge overlapping candidates
            best_cand = max(overlapping, key=lambda x: x['confidence'])
            merged.append(best_cand)
            used.add(i)

        return merged

    def _select_best_corners(self, candidates: List[Dict]) -> List[Dict]:
        """
        Select the best corner for each position.
        """
        corner_groups = {}

        for candidate in candidates:
            corner_type = candidate['corner_type']
            if corner_type not in corner_groups:
                corner_groups[corner_type] = []
            corner_groups[corner_type].append(candidate)

        final_candidates = []

        for corner_type, group in corner_groups.items():
            if group:
                # Select best candidate based on confidence and other factors
                best = max(group, key=lambda x: x['confidence'])
                final_candidates.append(best)

        return final_candidates

    def _create_final_visualization(
        self,
        gray: np.ndarray,
        candidates: List[Dict],
        viz_steps: Dict
    ) -> None:
        """
        Create final visualization of detected corners.
        """
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        colors = {
            'top_left': (0, 255, 0),      # Green
            'top_right': (255, 0, 0),     # Blue
            'bottom_left': (0, 255, 255),  # Yellow
            'bottom_right': (255, 0, 255)  # Magenta
        }

        for candidate in candidates:
            contour = candidate['contour']
            corner_type = candidate['corner_type']
            color = colors.get(corner_type, (128, 128, 128))

            # Draw contour
            cv2.drawContours(viz, [contour], -1, color, 2)

            # Draw center
            cx, cy = candidate['center']
            cv2.circle(viz, (cx, cy), 5, color, -1)

            # Add text
            method = candidate.get('method', 'unknown')
            confidence = candidate.get('confidence', 0)
            text = f"{corner_type} ({method}) {confidence:.2f}"
            cv2.putText(viz, text, (cx - 50, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        viz_steps["final_detection"] = viz
