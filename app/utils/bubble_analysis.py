# app/utils/bubble_analysis.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def analyze_bubbles(grouped_circles, thresh, bubble_threshold, score_multiplier):
	"""
	Analyze each bubble to determine if it's filled and extract answers.
	
	Args:
		grouped_circles: Circles grouped by question
		thresh: Threshold image
		bubble_threshold: Threshold for bubble detection
		score_multiplier: Score multiplier
		
	Returns:
		tuple: (answers, coords)
	"""
	choice_labels = ['A', 'B', 'C', 'D', 'E']
	answers = []
	all_coords = []
	
	for q_num, row_bubbles in grouped_circles:
		if len(row_bubbles) < 3:  # Need at least 3 bubbles for a valid row
			continue
		
		max_score = 0
		marked_idx = -1
		bubble_details = []
		
		# Analyze each bubble in the row
		for bubble_idx, (x, y, r) in enumerate(row_bubbles[:len(choice_labels)]):
			if bubble_idx >= len(choice_labels):
				break
			
			label = choice_labels[bubble_idx]
			
			# Calculate fill score with enhanced method
			fill_score = _calculate_bubble_fill_score_enhanced(thresh, x, y, r)
			calculated_score = float(fill_score * score_multiplier)
			
			bubble_details.append({
				'label': label,
				'x': int(x),
				'y': int(y),
				'r': int(r),
				'score': calculated_score,
				'selected': False
			})
			
			# Check if this bubble has the highest score
			if calculated_score > bubble_threshold and calculated_score > max_score:
				max_score = calculated_score
				marked_idx = bubble_idx
		
		# Mark the selected answer
		current_answer = ""
		if marked_idx != -1:
			selected_label = choice_labels[marked_idx]
			bubble_details[marked_idx]['selected'] = True
			current_answer = selected_label
		
		answers.append((q_num, current_answer))
		all_coords.append({
			'question_number': q_num,
			'choices': bubble_details
		})
	
	return answers, all_coords


def _calculate_bubble_fill_score_enhanced(thresh, x, y, r):
	"""
	Enhanced fill score calculation using inner circle to avoid edge effects.
	
	Args:
		thresh: Threshold image
		x, y: Circle center coordinates
		r: Circle radius
		
	Returns:
		float: Normalized fill score
	"""
	# Validate inputs
	if r <= 0:
		return 0.0
	
	# Create mask for the bubble (use inner 80% to avoid edge effects)
	bubble_mask = np.zeros_like(thresh)
	inner_radius = int(r * 0.8)
	
	if inner_radius <= 0:
		return 0.0
	
	cv2.circle(bubble_mask, (x, y), inner_radius, 255, -1)
	
	# Calculate fill score
	fill_score = cv2.countNonZero(cv2.bitwise_and(thresh, bubble_mask))
	
	# Normalize by area to make score independent of bubble size
	area = np.pi * inner_radius * inner_radius
	normalized_score = fill_score / area * 100 if area > 0 else 0.0
	
	return normalized_score