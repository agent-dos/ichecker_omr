# app/utils/corner_debug.py
import cv2
import numpy as np


def visualize_corner_detection_debug(image, candidates, corners):
    """
    Create debug visualization for corner detection.
    """
    viz = image.copy()
    height, width = image.shape[:2]
    
    # Draw all candidates
    for i, candidate in enumerate(candidates):
        cx, cy = candidate['center']
        cv2.circle(viz, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(viz, f"C{i}", (cx + 10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw selected corners
    if corners:
        colors = {
            'top_left': (0, 255, 0),
            'top_right': (255, 0, 0),
            'bottom_left': (0, 255, 255),
            'bottom_right': (255, 0, 255)
        }
        
        for corner_name, corner in corners.items():
            if corner:
                cx, cy = corner['center']
                cv2.circle(viz, (cx, cy), 10, colors[corner_name], 3)
                cv2.putText(viz, corner_name[:2], (cx - 10, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[corner_name], 2)
    
    # Add grid lines for reference
    cv2.line(viz, (width//2, 0), (width//2, height), (128, 128, 128), 1)
    cv2.line(viz, (0, height//2), (width, height//2), (128, 128, 128), 1)
    
    return viz