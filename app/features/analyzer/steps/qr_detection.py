# app/features/analyzer/steps/qr_detection.py
"""QR code detection step."""
import logging
from typing import Dict, Optional

import cv2
import numpy as np

from app.features.analyzer.steps.base import AnalysisStep
from app.features.qr.detector import QRDetector

logger = logging.getLogger(__name__)


class QRDetectionStep(AnalysisStep):
    """Handles QR code detection."""

    def __init__(self, params: Dict):
        """Initialize with QR detection parameters."""
        super().__init__(params)
        self.detector = QRDetector(params.get('qr_detection', {}))
        self.step_name = "1. QR Code Detection"  # Updated numbering

    def process(self, context: Dict) -> Dict:
        """Process QR code detection on original image."""
        logger.info("=== STEP 1: QR Code Detection ===")

        processing_img = context['processing_img']
        visual_chain = context['visual_chain']
        visualize = context['visualize']

        # Detect QR code on original image
        qr_data, qr_info, viz_steps = self.detector.detect(
            processing_img, visualize)

        # Create visualization showing QR detection results
        output_viz = self._create_visualization(visual_chain, qr_data, qr_info)

        # Create step info
        step_info = self.create_step_info(
            description=f'QR Data: {qr_data if qr_data else "None"}',
            success=qr_data is not None,
            input_image=visual_chain,
            output_image=output_viz,
            viz_steps=viz_steps if visualize else None
        )

        return {
            'step_info': step_info,
            'qr_data': qr_data,
            'context_update': {
                'visual_chain': output_viz,
                'qr_polygon': qr_info.get('polygon') if qr_info else None,
                'qr_info': qr_info  # Pass full QR info to subsequent steps
            }
        }

    def _create_visualization(self, base_image: np.ndarray,
                              qr_data: Optional[str],
                              qr_info: Optional[Dict]) -> np.ndarray:
        """Create visualization showing QR detection results."""
        viz = base_image.copy()

        if qr_data and qr_info:
            # Draw QR polygon if available
            qr_polygon = qr_info.get('polygon')
            if qr_polygon:
                cv2.polylines(viz, [np.array(qr_polygon, dtype=np.int32)],
                              True, (0, 0, 255), 2)

            # Add QR data text
            if 'rect' in qr_info:
                x, y = qr_info['rect']['left'], qr_info['rect']['top'] - 10
                cv2.putText(viz, f"QR: {qr_data}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(viz, "No QR Code Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return viz
