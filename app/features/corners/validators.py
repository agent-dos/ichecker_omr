# filename: app/features/corners/validators.py
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple  # Added Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CornerValidator:
    """ Validates corner candidates, including filtering QR-like patterns. """

    def __init__(self, params: Dict[str, Any]):
        """ Initialize with validator parameters from corner_detection['validator']. """
        self.params = params
        self.enabled = self.params.get('qr_filter_enabled', True)
        self.canny_t1 = self.params.get('qr_canny_threshold1', 50)
        self.canny_t2 = self.params.get('qr_canny_threshold2', 150)
        self.edge_ratio_thresh = self.params.get(
            'qr_edge_ratio_threshold', 0.15)
        self.complexity_thresh = self.params.get(
            'qr_complexity_threshold', 0.3)
        logger.debug(f"CornerValidator initialized (Enabled: {self.enabled})")

    def filter_qr_patterns(
        self,
        candidates: List[Dict],
        gray: np.ndarray,
        visualize_steps: bool = False
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Filter out candidates likely belonging to QR codes.
        Returns: (filtered_candidates_list, visualization_steps_dict)
        """
        viz_steps = {}
        if not self.enabled or not candidates:
            logger.debug(
                f"QR Filter for corners skipped (Enabled: {self.enabled}, Candidates: {len(candidates)})")
            return candidates or [], viz_steps  # Return empty list if candidates is None

        filtered = []
        excluded_details = []  # Store info about excluded candidates for viz

        logger.info(
            f"Applying QR filter to {len(candidates)} corner candidates...")
        for idx, candidate in enumerate(candidates):
            candidate_key = f"cand_{idx:03d}"
            # Default to keeping the candidate
            keep_candidate = True
            qr_check_viz = {}  # Viz specific to this candidate check

            try:
                x, y, w, h = candidate['bbox']
                # Ensure ROI is valid before extraction
                y_end, x_end = min(
                    y + h, gray.shape[0]), min(x + w, gray.shape[1])
                if y >= y_end or x >= x_end or w <= 0 or h <= 0:
                    logger.warning(
                        f"Skipping candidate {idx} due to invalid bbox: {candidate['bbox']}")
                    is_qr = False  # Cannot determine, assume not QR
                else:
                    roi = gray[y:y_end, x:x_end]
                    is_qr, qr_check_viz = self._is_qr_pattern(
                        roi, visualize_steps)

                # Add prefix to viz keys from _is_qr_pattern
                if visualize_steps:
                    for k, v in qr_check_viz.items():
                        viz_steps[f"{candidate_key}_qr_check_{k}"] = v

                if is_qr:
                    keep_candidate = False
                    logger.debug(
                        f"Filtered candidate {idx} (bbox: {candidate['bbox']}) as likely QR pattern.")
                    excluded_details.append({
                        'bbox': candidate['bbox'],
                        'center': candidate.get('center', (x+w//2, y+h//2))
                    })

            except Exception as e:
                logger.error(
                    f"Error processing candidate {idx} in QR filter: {e}", exc_info=True)
                # Keep candidate if error occurs during check? Or discard? Safer to keep.
                keep_candidate = True

            if keep_candidate:
                filtered.append(candidate)

        # Add visualization of final filtered candidates (optional)
        if visualize_steps:
            viz_filtered = gray.copy()
            viz_filtered = cv2.cvtColor(viz_filtered, cv2.COLOR_GRAY2BGR)
            # Draw filtered candidates (green)
            for cand in filtered:
                cx, cy = cand.get('center', (0, 0))
                if cx or cy:
                    cv2.circle(viz_filtered, (cx, cy), 5, (0, 255, 0), -1)
            # Draw excluded candidates (red 'x')
            for ex_detail in excluded_details:
                cx, cy = ex_detail.get('center', (0, 0))
                if cx or cy:
                    cv2.line(viz_filtered, (cx-4, cy-4),
                             (cx+4, cy+4), (0, 0, 255), 1)
                    cv2.line(viz_filtered, (cx-4, cy+4),
                             (cx+4, cy-4), (0, 0, 255), 1)
            viz_steps["90_QRFilter_Result"] = viz_filtered

        logger.info(
            f"QR Filter kept {len(filtered)} out of {len(candidates)} candidates.")
        return filtered, viz_steps

    def _is_qr_pattern(self, roi: np.ndarray, visualize_steps: bool = False) -> Tuple[bool, Dict]:
        """ Check ROI for QR-like patterns. Returns: (is_likely_qr, viz_steps_dict) """
        viz_steps = {}
        if roi.size == 0:
            return False, viz_steps

        # --- Check 1: Edge Density ---
        is_likely_qr = False
        try:
            edges = cv2.Canny(roi, self.canny_t1, self.canny_t2)
            edge_ratio = cv2.countNonZero(
                edges) / float(edges.size) if edges.size > 0 else 0.0

            if visualize_steps:
                # Create a combined viz for this check
                h, w = roi.shape[:2]
                combined = np.zeros((h, w*2, 3), dtype=np.uint8)
                combined[:, 0:w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                combined[:, w:w*2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                label = f"Edge Ratio: {edge_ratio:.3f} (>{self.edge_ratio_thresh:.3f}?)"
                cv2.putText(combined, label, (5, h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                viz_steps["01_EdgeCheck"] = combined

            if edge_ratio > self.edge_ratio_thresh:
                is_likely_qr = True
        except cv2.error as e:
            logger.warning(
                f"OpenCV Error during Canny edge detection in _is_qr_pattern: {e}")

        # --- Check 2: Pattern Complexity (Variance) - Only if Edge check passed or failed cleanly ---
        if not is_likely_qr:  # Only run if edge ratio didn't already confirm QR
            try:
                complexity = self._calculate_complexity(roi)
                if visualize_steps:
                    # Add complexity score to ROI image if not already added
                    roi_viz = viz_steps.get("01_EdgeCheck", cv2.cvtColor(
                        roi, cv2.COLOR_GRAY2BGR).copy())
                    label = f"Complex: {complexity:.3f} (>{self.complexity_thresh:.3f}?)"
                    # Find suitable position for second label
                    text_y = roi.shape[0] - \
                        20 if "01_EdgeCheck" in viz_steps else roi.shape[0] - 5
                    cv2.putText(roi_viz, label, (5, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                    # Overwrite or add
                    viz_steps["02_ComplexityCheck"] = roi_viz

                if complexity > self.complexity_thresh:
                    is_likely_qr = True
            except Exception as e:
                logger.warning(
                    f"Error calculating complexity in _is_qr_pattern: {e}")

        return is_likely_qr, viz_steps

    def _calculate_complexity(self, roi: np.ndarray) -> float:
        """ Calculate internal complexity using normalized variance. """
        if roi.size < 4:
            return 0.0  # Need at least a few pixels
        try:
            mean, stddev = cv2.meanStdDev(roi)
            variance = stddev[0][0] ** 2
            # Normalize variance relative to potential intensity range (0-255)
            normalized_variance = variance / (255.0 * 255.0)
            return normalized_variance
        except cv2.error as e:
            logger.warning(
                f"cv2.meanStdDev failed in _calculate_complexity: {e}")
            return 0.0
