# app/features/analyzer/visualizers/answer_viz.py
"""
Answer visualization.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple

from app.core.constants import COLOR_GREEN, COLOR_RED


def visualize_scores(
    image: np.ndarray,
    scores: List[Dict]
) -> np.ndarray:
    """
    Visualize bubble fill scores.
    """
    viz = image.copy()
    
    for score_data in scores:
        q_num = score_data['question_number']
        choices = score_data['choices']
        
        for choice in choices:
            x, y, r = choice['x'], choice['y'], choice['r']
            label = choice['label']
            score = choice['score']
            selected = choice['selected']
            
            # Color based on selection
            color = COLOR_GREEN if selected else COLOR_RED
            
            # Draw bubble
            cv2.circle(viz, (x, y), r, color, 1)
            
            # Draw label
            cv2.putText(viz, label, (x - r//2 + 2, y + r//2 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Show score for high scores or selected
            if selected or score > 50:
                cv2.putText(viz, f"{score:.0f}", (x + r + 2, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 255), 1)
            
            if selected:
                cv2.circle(viz, (x, y), r + 2, COLOR_GREEN, 2)
                cv2.putText(viz, f"{q_num}:{label}", (x - r - 10, y - r - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)
    
    return viz


def visualize_answers(
    image: np.ndarray,
    answers: List[Tuple[int, str]],
    coords: List[Dict]
) -> np.ndarray:
    """
    Visualize final answers with summary overlay.
    """
    viz = visualize_scores(image, coords)
    
    # Create answer summary overlay
    _draw_answer_summary(viz, answers)
    
    return viz


def _draw_answer_summary(
    viz: np.ndarray,
    answers: List[Tuple[int, str]]
) -> None:
    """
    Draw semi-transparent answer summary.
    """
    if not answers:
        return
    
    answers.sort(key=lambda a: a[0])
    
    # Calculate overlay dimensions
    img_h, img_w = viz.shape[:2]
    font_scale = 0.4 * min(1.0, 1000 / max(img_h, img_w))
    line_height = max(10, int(15 * min(1.0, 1000 / max(img_h, img_w))))
    
    max_per_col = 15
    num_cols = min(2, 1 + (len(answers) - 1) // max_per_col)
    
    overlay_width = 100 * num_cols
    overlay_height = 5 + line_height * min(max_per_col, len(answers))
    
    # Draw background
    overlay = viz.copy()
    cv2.rectangle(overlay, (5, 5), (overlay_width + 5, overlay_height + 5),
                 (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, viz, 0.4, 0, viz)
    
    # Draw answers
    y_start = 20
    for i, (q_num, choice) in enumerate(answers):
        col_idx = i // max_per_col
        x_pos = 10 + (col_idx * 100)
        y_pos = y_start + (i % max_per_col) * line_height
        
        if x_pos > overlay_width - 10:
            break
        
        text = f"{q_num}: {choice if choice else '-'}"
        cv2.putText(viz, text, (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 1)