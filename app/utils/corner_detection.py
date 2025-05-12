# app/utils/corner_detection.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_corner_markers(image, min_area=300, max_area=5000, qr_polygon=None):
    """
    Detect corner markers with enhanced QR code discrimination.

    Args:
        image: Input image (BGR format)
        min_area: Minimum area for corner detection
        max_area: Maximum area for corner detection
        qr_polygon: QR code polygon to exclude

    Returns:
        dict: Corner positions or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Collect candidates from multiple detection methods
    all_candidates = []

    # Method 1: Multiple threshold levels with morphology
    for threshold_value in [30, 50, 70, 90]:
        candidates = _detect_with_threshold(
            gray, threshold_value, min_area, max_area, qr_polygon
        )
        all_candidates.extend(candidates)

    # Method 2: Adaptive threshold
    adaptive_candidates = _detect_with_adaptive(
        gray, min_area, max_area, qr_polygon
    )
    all_candidates.extend(adaptive_candidates)

    # Remove duplicates and QR-like patterns
    unique_candidates = _remove_duplicates(all_candidates)
    filtered_candidates = _filter_qr_patterns(unique_candidates, gray)

    # Select best 4 corners
    corners = _select_best_corners(filtered_candidates, width, height)

    if not corners or not all(corners.values()):
        logger.warning("Could not detect all corners")
        return None

    return corners


def _detect_with_threshold(gray, threshold_value, min_area, max_area, qr_polygon):
    """Detect candidates using binary threshold."""
    _, thresh = cv2.threshold(gray, threshold_value,
                              255, cv2.THRESH_BINARY_INV)

    # Apply morphology to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        candidate = _analyze_contour(
            contour, thresh, min_area, max_area, qr_polygon)
        if candidate:
            candidates.append(candidate)

    return candidates


def _analyze_contour(contour, thresh, min_area, max_area, qr_polygon):
    """Analyze a single contour for corner candidacy."""
    area = cv2.contourArea(contour)
    if not (min_area < area < max_area):
        return None

    x, y, w, h = cv2.boundingRect(contour)
    center = (x + w//2, y + h//2)

    # Check if in QR polygon
    if qr_polygon and _point_in_polygon(center, qr_polygon):
        return None

    # Check basic properties
    solidity = area / cv2.contourArea(cv2.convexHull(contour))
    aspect_ratio = float(w) / h

    # Stricter aspect ratio for corner markers
    if not (0.7 < aspect_ratio < 1.3) or solidity < 0.8:
        return None

    # Check fill uniformity
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked = cv2.bitwise_and(thresh, mask)
    fill_ratio = cv2.countNonZero(masked) / cv2.countNonZero(mask)

    if fill_ratio < 0.85:
        return None

    return {
        'contour': contour,
        'center': center,
        'area': area,
        'bbox': (x, y, w, h),
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'fill_ratio': fill_ratio
    }


def _detect_with_adaptive(gray, min_area, max_area, qr_polygon):
    """Detect candidates using adaptive threshold."""
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    contours, _ = cv2.findContours(
        adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        candidate = _analyze_contour(
            contour, adaptive, min_area, max_area, qr_polygon)
        if candidate:
            candidates.append(candidate)

    return candidates


def _filter_qr_patterns(candidates, gray):
    """Filter out QR code-like patterns."""
    filtered = []

    for candidate in candidates:
        x, y, w, h = candidate['bbox']
        roi = gray[y:y+h, x:x+w]

        # Check for QR pattern characteristics
        if _is_qr_pattern(roi):
            continue

        # Check for internal complexity
        complexity = _calculate_complexity(roi)
        if complexity > 0.3:  # QR codes have high internal complexity
            continue

        filtered.append(candidate)

    return filtered


def _is_qr_pattern(roi):
    """Check if ROI contains QR-like patterns."""
    if roi.size == 0:
        return False

    # Edge detection to find internal structure
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / edges.size

    # QR codes have significant internal edges
    if edge_ratio > 0.15:
        return True

    # Check for finder patterns (corner squares within squares)
    h, w = roi.shape
    corners = [
        roi[0:h//3, 0:w//3],
        roi[0:h//3, -w//3:],
        roi[-h//3:, 0:w//3]
    ]

    for corner in corners:
        if corner.size > 0:
            _, thresh = cv2.threshold(corner, 127, 255, cv2.THRESH_BINARY)
            white_ratio = cv2.countNonZero(thresh) / thresh.size
            if 0.3 < white_ratio < 0.7:  # Mixed pattern
                return True

    return False


def _calculate_complexity(roi):
    """Calculate internal complexity of ROI."""
    if roi.size == 0:
        return 0

    # Calculate variance
    variance = np.var(roi)

    # Normalize by size
    normalized_variance = variance / (roi.size * 255)

    return normalized_variance


def _remove_duplicates(candidates, distance_threshold=30):
    """Remove duplicate candidates based on proximity."""
    unique = []

    for candidate in candidates:
        cx, cy = candidate['center']
        is_duplicate = False

        for existing in unique:
            ex, ey = existing['center']
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

            if distance < distance_threshold:
                # Keep the better candidate
                if _compare_candidates(candidate, existing) > 0:
                    unique.remove(existing)
                    unique.append(candidate)
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(candidate)

    return unique


def _compare_candidates(cand1, cand2):
    """Compare two candidates, return positive if cand1 is better."""
    score1 = (cand1.get('fill_ratio', 0) * 0.4 +
              cand1.get('solidity', 0) * 0.3 +
              (1.0 - abs(1.0 - cand1.get('aspect_ratio', 1))) * 0.3)

    score2 = (cand2.get('fill_ratio', 0) * 0.4 +
              cand2.get('solidity', 0) * 0.3 +
              (1.0 - abs(1.0 - cand2.get('aspect_ratio', 1))) * 0.3)

    return score1 - score2


def _select_best_corners(candidates, width, height):
    """Select the best 4 corners from candidates."""
    if len(candidates) < 4:
        return None

    # Calculate distance-based scores
    corner_positions = {
        'top_left': (0, 0),
        'top_right': (width, 0),
        'bottom_left': (0, height),
        'bottom_right': (width, height)
    }

    corners = {}
    used_indices = set()

    for corner_name, ideal_pos in corner_positions.items():
        best_idx = -1
        best_score = -float('inf')

        for i, candidate in enumerate(candidates):
            if i in used_indices:
                continue

            cx, cy = candidate['center']
            distance = np.sqrt((cx - ideal_pos[0])**2 + (cy - ideal_pos[1])**2)

            # Score based on distance and properties
            distance_score = 1.0 / (1.0 + distance / 100.0)
            property_score = _calculate_property_score(candidate)

            total_score = distance_score * 0.7 + property_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_idx = i

        if best_idx >= 0:
            corners[corner_name] = candidates[best_idx]
            used_indices.add(best_idx)

    return corners if len(corners) == 4 else None


def _calculate_property_score(candidate):
    """Calculate score based on candidate properties."""
    return (candidate.get('fill_ratio', 0) * 0.4 +
            candidate.get('solidity', 0) * 0.3 +
            (1.0 - abs(1.0 - candidate.get('aspect_ratio', 1))) * 0.3)


def _point_in_polygon(point, polygon):
    """Check if point is inside polygon."""
    from app.utils.shape_validation import validate_polygon

    try:
        polygon = validate_polygon(polygon)
        if polygon is None:
            return False
    except Exception as e:
        logger.warning(f"Invalid polygon: {e}")
        return False

    x, y = point
    result = cv2.pointPolygonTest(polygon.astype(
        np.int32), (float(x), float(y)), False)
    return result >= 0


def get_bounding_quadrilateral(corners, margin=10):
    """Get quadrilateral from corners with margin."""
    if not corners or not all(corners.values()):
        return None

    points = []
    for corner_name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
        corner = corners[corner_name]
        cx, cy = corner['center']

        # Apply inward margin
        if 'top' in corner_name:
            cy += margin
        else:
            cy -= margin

        if 'left' in corner_name:
            cx += margin
        else:
            cx -= margin

        points.append([float(cx), float(cy)])

    return np.array(points, dtype=np.float32)


def visualize_corners(image, corners):
    """Visualize detected corners."""
    viz = image.copy()

    if not corners:
        return viz

    colors = {
        'top_left': (0, 255, 0), 'top_right': (255, 0, 0),
        'bottom_left': (0, 255, 255), 'bottom_right': (255, 0, 255)
    }

    for corner_name, corner in corners.items():
        if corner:
            cv2.circle(viz, corner['center'], 10, colors[corner_name], -1)
            cv2.putText(viz, corner_name,
                        (corner['center'][0] - 40, corner['center'][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[corner_name], 2)

    quad = get_bounding_quadrilateral(corners)
    if quad is not None:
        cv2.polylines(viz, [quad.astype(int)], True, (0, 255, 0), 2)

    return viz
