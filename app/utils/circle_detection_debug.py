# app/utils/circle_detection_debug.py
import cv2
import numpy as np
import logging
from app.utils.geometry import filter_circles_outside_polygon, filter_circles_inside_quadrilateral

logger = logging.getLogger(__name__)


def visualize_detection_stages(image, params, qr_polygon=None, answer_boundary=None):
    """
    Visualize each stage of circle detection for debugging.

    Args:
        image: Input image
        params: Detection parameters
        qr_polygon: QR code polygon
        answer_boundary: Corner boundary quadrilateral

    Returns:
        dict: Visualization images and statistics for each stage
    """
    if image is None or params is None:
        logger.error("Invalid input to debug visualization")
        return {}

    results = {}

    # Stage 1: Raw circle detection
    raw_circles = _detect_raw_circles(image, params)
    raw_viz, raw_stats = _create_stage_visualization(
        image, raw_circles, "Raw Detection", (0, 255, 0)
    )
    results['raw'] = {'image': raw_viz,
                      'stats': raw_stats, 'circles': raw_circles}

    # Stage 2: After boundary filter (if applicable)
    boundary_circles = raw_circles
    if answer_boundary is not None and raw_circles is not None:
        boundary_circles = filter_circles_inside_quadrilateral(
            raw_circles, answer_boundary, margin=5
        )
        boundary_viz, boundary_stats = _create_stage_visualization(
            image, boundary_circles, "After Boundary Filter", (255, 255, 0)
        )
        results['boundary'] = {
            'image': boundary_viz,
            'stats': boundary_stats,
            'circles': boundary_circles
        }

    # Stage 3: After QR exclusion (if applicable)
    final_circles = boundary_circles
    if qr_polygon is not None and boundary_circles is not None:
        final_circles = filter_circles_outside_polygon(
            boundary_circles, qr_polygon, margin=params.get(
                'max_radius', 20) + 5
        )
        qr_viz, qr_stats = _create_stage_visualization(
            image, final_circles, "After QR Exclusion", (0, 0, 255)
        )
        results['qr_excluded'] = {
            'image': qr_viz,
            'stats': qr_stats,
            'circles': final_circles
        }

    # Stage 4: Comparison view
    comparison_viz = _create_comparison_view(image, results)
    results['comparison'] = {'image': comparison_viz}

    # Stage 5: Exclusion map
    exclusion_viz = _create_exclusion_map(
        image, raw_circles, final_circles, qr_polygon, answer_boundary
    )
    results['exclusion_map'] = {'image': exclusion_viz}

    return results


def _detect_raw_circles(image, params):
    """Detect circles without any filtering."""
    gray = _convert_to_grayscale(image)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=params.get('param1', 50),
        param2=params.get('param2', 18),
        minRadius=params.get('min_radius', 10),
        maxRadius=params.get('max_radius', 20)
    )

    if circles is not None:
        return np.round(circles[0, :]).astype(int)
    return None


def _create_stage_visualization(image, circles, title, color):
    """Create visualization for a detection stage."""
    viz = image.copy()
    stats = {'count': 0, 'left': 0, 'right': 0}

    if circles is not None:
        midpoint_x = image.shape[1] // 2

        for x, y, r in circles:
            cv2.circle(viz, (x, y), r, color, 2)
            cv2.circle(viz, (x, y), 2, color, -1)

            stats['count'] += 1
            if x < midpoint_x:
                stats['left'] += 1
            else:
                stats['right'] += 1

    # Add title and stats
    cv2.putText(viz, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(viz, f"Total: {stats['count']} (L:{stats['left']} R:{stats['right']})",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return viz, stats


def _create_comparison_view(image, results):
    """Create side-by-side comparison of all stages."""
    h, w = image.shape[:2]
    comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # Place images in grid
    stages = ['raw', 'boundary', 'qr_excluded']
    positions = [(0, 0), (0, w), (h, 0)]

    for stage, (y, x) in zip(stages, positions):
        if stage in results and 'image' in results[stage]:
            comparison[y:y+h, x:x+w] = results[stage]['image']

    # Add exclusion map in bottom right
    if 'exclusion_map' in results:
        comparison[h:h*2, w:w*2] = results['exclusion_map']['image']

    return comparison


def _create_exclusion_map(image, raw_circles, final_circles, qr_polygon, answer_boundary):
    """Create visual map showing excluded areas and circles."""
    viz = image.copy()

    # Draw exclusion areas
    if answer_boundary is not None:
        # Draw boundary as filled transparent overlay
        overlay = image.copy()
        cv2.fillPoly(overlay, [answer_boundary.astype(int)], (0, 255, 0))
        viz = cv2.addWeighted(overlay, 0.3, viz, 0.7, 0)

    if qr_polygon is not None:
        # Draw QR exclusion area
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(qr_polygon, dtype=int)], (0, 0, 255))
        viz = cv2.addWeighted(overlay, 0.3, viz, 0.7, 0)

    # Draw excluded circles in red
    if raw_circles is not None and final_circles is not None:
        final_set = set(map(tuple, final_circles))

        for circle in raw_circles:
            if tuple(circle) not in final_set:
                x, y, r = circle
                cv2.circle(viz, (x, y), r, (0, 0, 255), 2)
                cv2.line(viz, (x-r, y), (x+r, y), (0, 0, 255), 2)
                cv2.line(viz, (x, y-r), (x, y+r), (0, 0, 255), 2)

    # Draw accepted circles in green
    if final_circles is not None:
        for x, y, r in final_circles:
            cv2.circle(viz, (x, y), r, (0, 255, 0), 2)

    cv2.putText(viz, "Exclusion Map", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(viz, "Red: Excluded | Green: Accepted", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return viz


def _convert_to_grayscale(image):
    """Convert image to grayscale handling different formats."""
    if len(image.shape) == 2:
        return image
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
