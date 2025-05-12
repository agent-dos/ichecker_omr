# app/utils/robust_corner_detection.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_corner_markers_robust(image, min_area=200, max_area=8000, qr_polygon=None):
    """
    More robust corner detection for tilted answer sheets.
    
    Uses multiple detection strategies:
    1. Multiple threshold levels
    2. Morphological operations
    3. Contour approximation
    4. Area and shape validation
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Collect all potential corner candidates
    all_candidates = []
    
    # Strategy 1: Multiple threshold levels with morphology
    for threshold_value in [30, 50, 70, 90]:
        candidates = _detect_with_threshold(
            gray, threshold_value, min_area, max_area, qr_polygon
        )
        all_candidates.extend(candidates)
    
    # Strategy 2: Adaptive threshold
    adaptive_candidates = _detect_with_adaptive_threshold(
        gray, min_area, max_area, qr_polygon
    )
    all_candidates.extend(adaptive_candidates)
    
    # Strategy 3: Edge detection approach
    edge_candidates = _detect_with_edges(
        gray, min_area, max_area, qr_polygon
    )
    all_candidates.extend(edge_candidates)
    
    # Remove duplicates
    unique_candidates = _remove_duplicate_candidates(all_candidates)
    
    # Select best 4 candidates based on corner properties
    best_corners = _select_best_corners(unique_candidates, width, height)
    
    if len(best_corners) < 4:
        logger.warning(f"Only found {len(best_corners)} corners")
        # Try relaxed detection
        return _fallback_detection(image, min_area // 2, max_area * 2)
    
    return best_corners


def _detect_with_threshold(gray, threshold_value, min_area, max_area, qr_polygon):
    """Detection using binary threshold with morphology."""
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Use contour approximation
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Look for quadrilateral-like shapes
            if 3 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # Skip if in QR area
                if qr_polygon and _point_in_polygon(center, qr_polygon):
                    continue
                
                # Check solidity (filled area)
                solidity = area / cv2.contourArea(cv2.convexHull(contour))
                if solidity > 0.7:  # Relaxed from 0.85
                    candidates.append({
                        'contour': contour,
                        'center': center,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'solidity': solidity,
                        'vertices': len(approx)
                    })
    
    return candidates


def _detect_with_adaptive_threshold(gray, min_area, max_area, qr_polygon):
    """Detection using adaptive threshold."""
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            if qr_polygon and _point_in_polygon(center, qr_polygon):
                continue
            
            # Relax aspect ratio for tilted corners
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:
                candidates.append({
                    'contour': contour,
                    'center': center,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio
                })
    
    return candidates


def _detect_with_edges(gray, min_area, max_area, qr_polygon):
    """Detection using edge detection."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            if qr_polygon and _point_in_polygon(center, qr_polygon):
                continue
            
            candidates.append({
                'contour': contour,
                'center': center,
                'area': area,
                'bbox': (x, y, w, h)
            })
    
    return candidates


def _remove_duplicate_candidates(candidates, distance_threshold=30):
    """Remove duplicate candidates based on center proximity."""
    unique = []
    
    for candidate in candidates:
        is_duplicate = False
        cx, cy = candidate['center']
        
        for existing in unique:
            ex, ey = existing['center']
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if distance < distance_threshold:
                # Keep the one with better properties
                if candidate.get('solidity', 0) > existing.get('solidity', 0):
                    unique.remove(existing)
                    unique.append(candidate)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(candidate)
    
    return unique


def _select_best_corners(candidates, image_width, image_height):
    """Select the best 4 corners from candidates."""
    if len(candidates) <= 4:
        return _assign_corner_positions(candidates, image_width, image_height)
    
    # Score candidates based on corner likelihood
    scored_candidates = []
    for candidate in candidates:
        score = _calculate_corner_score(candidate, image_width, image_height)
        candidate['score'] = score
        scored_candidates.append(candidate)
    
    # Sort by score and take top 4
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    best_four = scored_candidates[:4]
    
    return _assign_corner_positions(best_four, image_width, image_height)


def _calculate_corner_score(candidate, image_width, image_height):
    """Calculate how likely a candidate is to be a corner."""
    cx, cy = candidate['center']
    
    # Distance from image corners
    distances = [
        np.sqrt(cx**2 + cy**2),  # Top-left
        np.sqrt((image_width - cx)**2 + cy**2),  # Top-right
        np.sqrt(cx**2 + (image_height - cy)**2),  # Bottom-left
        np.sqrt((image_width - cx)**2 + (image_height - cy)**2)  # Bottom-right
    ]
    min_distance = min(distances)
    
    # Score based on distance to nearest corner
    distance_score = 1.0 / (1.0 + min_distance / 100.0)
    
    # Score based on shape properties
    area_score = min(1.0, candidate['area'] / 1000.0)
    solidity_score = candidate.get('solidity', 0.7)
    
    # Combined score
    return distance_score * 0.5 + area_score * 0.25 + solidity_score * 0.25


def _assign_corner_positions(candidates, image_width, image_height):
    """Assign candidates to corner positions."""
    if len(candidates) < 4:
        return None
    
    corners = {
        'top_left': None,
        'top_right': None,
        'bottom_left': None,
        'bottom_right': None
    }
    
    # Find corners based on minimum distance to ideal positions
    ideal_positions = {
        'top_left': (0, 0),
        'top_right': (image_width, 0),
        'bottom_left': (0, image_height),
        'bottom_right': (image_width, image_height)
    }
    
    used_candidates = set()
    
    for corner_name, ideal_pos in ideal_positions.items():
        best_candidate = None
        best_distance = float('inf')
        
        for i, candidate in enumerate(candidates):
            if i in used_candidates:
                continue
                
            cx, cy = candidate['center']
            distance = np.sqrt((cx - ideal_pos[0])**2 + (cy - ideal_pos[1])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_candidate = (i, candidate)
        
        if best_candidate:
            used_candidates.add(best_candidate[0])
            corners[corner_name] = best_candidate[1]
    
    return corners


def _fallback_detection(image, min_area, max_area):
    """Fallback detection with relaxed parameters."""
    logger.info("Using fallback corner detection")
    # Implementation would use more aggressive detection strategies
    # This is a placeholder
    return None


def _point_in_polygon(point, polygon):
    """Check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside