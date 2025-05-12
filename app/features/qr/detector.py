# filename: app/features/qr/detector.py
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import logging
from typing import Tuple, Dict, Optional, List, Any  # Added List, Any

# Import helper to get cv2 flags
from app.core.config import get_cv2_flag, CV2_ADAPTIVE_METHODS

logger = logging.getLogger(__name__)


class QRDetector:
    """Detects and decodes QR codes."""

    # Modified __init__ to accept detailed params
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        logger.debug(f"QRDetector initialized with params: {self.params}")

    def detect(
        self,
        image: np.ndarray,
        visualize_steps: bool = False  # Control flag passed down
        # Modified return type
    ) -> Tuple[Optional[str], Optional[Dict], Dict[str, np.ndarray]]:
        """
        Detect QR code in image.

        Returns:
            tuple: (qr_data, qr_info, intermediate_visualizations)
        """
        intermediate_visualizations = {}

        if image is None:
            return None, None, intermediate_visualizations

        # Ensure image is grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()  # Work on a copy

        if visualize_steps:
            intermediate_visualizations['00_InputGray'] = cv2.cvtColor(
                gray, cv2.COLOR_GRAY2BGR)

        # --- Detection Attempts ---
        qr_codes = None
        attempt_order = ['direct', 'adaptive',
                         'blur', 'equalize']  # Define order

        for attempt_name in attempt_order:
            processed_img = gray  # Start with original gray
            viz_key_prefix = f"01_{attempt_name}"

            try:
                if attempt_name == 'direct':
                    # No preprocessing needed for direct attempt
                    processed_img = gray
                elif attempt_name == 'adaptive':
                    # Use parameters from config
                    method_flag = get_cv2_flag(
                        self.params.get('adaptive_method',
                                        'ADAPTIVE_THRESH_GAUSSIAN_C'),
                        CV2_ADAPTIVE_METHODS,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                    )
                    blocksize = self.params.get('adaptive_blocksize', 11)
                    # Ensure blocksize is odd and >= 3
                    blocksize = max(3, blocksize if blocksize %
                                    2 != 0 else blocksize + 1)
                    c_val = self.params.get('adaptive_c', 2)

                    processed_img = cv2.adaptiveThreshold(
                        gray, 255, method_flag,
                        cv2.THRESH_BINARY, blocksize, c_val
                    )
                    if visualize_steps:
                        intermediate_visualizations[f'{viz_key_prefix}_AdaptiveThresh'] = cv2.cvtColor(
                            processed_img, cv2.COLOR_GRAY2BGR)

                elif attempt_name == 'blur':
                    ksize = self.params.get('gaussian_blur_ksize', 5)
                    # Ensure ksize is odd and positive
                    ksize = max(1, ksize if ksize % 2 != 0 else ksize + 1)
                    processed_img = cv2.GaussianBlur(gray, (ksize, ksize), 0)
                    if visualize_steps:
                        intermediate_visualizations[f'{viz_key_prefix}_Blurred'] = cv2.cvtColor(
                            processed_img, cv2.COLOR_GRAY2BGR)

                elif attempt_name == 'equalize':
                    if self.params.get('equalize_hist', True):
                        processed_img = cv2.equalizeHist(gray)
                        if visualize_steps:
                            intermediate_visualizations[f'{viz_key_prefix}_Equalized'] = cv2.cvtColor(
                                processed_img, cv2.COLOR_GRAY2BGR)
                    else:
                        continue  # Skip if disabled

                # --- Attempt decoding ---
                logger.debug(
                    f"Attempting QR decode using method: {attempt_name}")
                qr_codes = decode(processed_img)

                if qr_codes:
                    logger.info(f"QR code found using method: {attempt_name}")
                    break  # Exit loop if QR code is found

            except cv2.error as e:
                logger.warning(
                    f"OpenCV error during QR detection attempt '{attempt_name}': {e}")
                continue  # Try next method
            except Exception as e:
                logger.error(
                    f"Unexpected error during QR detection attempt '{attempt_name}': {e}")
                continue  # Try next method

        # --- Process Results ---
        if not qr_codes:
            logger.warning("No QR Code detected after all attempts.")
            return None, None, intermediate_visualizations

        # Extract QR information (assuming first one is primary)
        qr_data, qr_info = self._extract_info(qr_codes[0])

        # Add final detection visualization (using the original visualizer logic)
        if visualize_steps:
            # Use the visualizer from the qr feature
            from .visualizer import visualize_qr
            final_viz = visualize_qr(cv2.cvtColor(
                gray, cv2.COLOR_GRAY2BGR), qr_data, qr_info)
            intermediate_visualizations['99_FinalDetection'] = final_viz

        return qr_data, qr_info, intermediate_visualizations

    # _to_grayscale can be removed or simplified as grayscale conversion happens at the start
    # def _to_grayscale(self, image: np.ndarray) -> np.ndarray: ...

    # Internal detection methods can be removed as logic is moved into the main loop
    # def _detect_with_adaptive(self, gray: np.ndarray): ...
    # def _detect_with_blur(self, gray: np.ndarray): ...
    # def _detect_with_equalization(self, gray: np.ndarray): ...

    def _extract_info(self, qr) -> Tuple[str, Dict]:
        """Extract QR data and metadata."""
        # (Keep existing logic)
        try:
            qr_data = qr.data.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning("QR data not UTF-8, trying latin-1.")
            try:
                qr_data = qr.data.decode('latin-1', errors='ignore')
            except Exception as e:
                logger.error(f"Could not decode QR data: {e}")
                qr_data = "DECODING_ERROR"

        qr_info = {
            'polygon': [(p.x, p.y) for p in qr.polygon] if qr.polygon else [],
            'rect': {
                'left': qr.rect.left,
                'top': qr.rect.top,
                'width': qr.rect.width,
                'height': qr.rect.height
            },
            'center': (
                qr.rect.left + qr.rect.width // 2,
                qr.rect.top + qr.rect.height // 2
            ),
            'type': qr.type,
            # If pyzbar version supports it
            'quality': getattr(qr, 'quality', None),
            # If pyzbar version supports it
            'orientation': getattr(qr, 'orientation', None)
        }

        return qr_data, qr_info
