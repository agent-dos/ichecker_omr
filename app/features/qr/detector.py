# app/features/qr/detector.py
"""
QR code detection module.
"""
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class QRDetector:
    """
    Detects and decodes QR codes.
    """

    def detect(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Detect QR code in image.

        Returns:
            tuple: (qr_data, qr_info)
        """
        if image is None:
            return None, None

        # Convert to grayscale
        gray = self._to_grayscale(image)

        # Try multiple detection methods
        methods = [
            lambda: decode(gray),
            lambda: self._detect_with_adaptive(gray),
            lambda: self._detect_with_blur(gray),
            lambda: self._detect_with_equalization(gray)
        ]

        qr_codes = None
        for method in methods:
            try:
                qr_codes = method()
                if qr_codes:
                    break
            except Exception as e:
                logger.debug(f"Detection method failed: {e}")
                continue

        if not qr_codes:
            return None, None

        # Extract QR information
        return self._extract_info(qr_codes[0])

    def visualize(
        self,
        image: np.ndarray,
        qr_data: Optional[str],
        qr_info: Optional[Dict]
    ) -> np.ndarray:
        """
        Visualize QR detection result.
        """
        from app.features.qr.visualizer import visualize_qr
        return visualize_qr(image, qr_data, qr_info)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale if needed.
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_with_adaptive(self, gray: np.ndarray):
        """
        Detect with adaptive threshold.
        """
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return decode(adaptive)

    def _detect_with_blur(self, gray: np.ndarray):
        """
        Detect with Gaussian blur.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return decode(blurred)

    def _detect_with_equalization(self, gray: np.ndarray):
        """
        Detect with histogram equalization.
        """
        equalized = cv2.equalizeHist(gray)
        return decode(equalized)

    def _extract_info(self, qr) -> Tuple[str, Dict]:
        """
        Extract QR data and metadata.
        """
        try:
            qr_data = qr.data.decode('utf-8')
        except UnicodeDecodeError:
            qr_data = qr.data.decode('latin-1', errors='ignore')

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
            'type': qr.type
        }

        return qr_data, qr_info
