# app/utils/qr_processing.py
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import logging

logger = logging.getLogger(__name__)


def detect_qr_code_data(image):
    """
    Detects QR codes in the image using multiple methods.

    Args:
            image: Input image (BGR format)

    Returns:
            tuple: (qr_data, qr_info) where:
                    - qr_data: String content of QR code or None
                    - qr_info: Dictionary with QR code details or None
    """
    if image is None:
        logger.warning("detect_qr_code_data: Input image is None.")
        return None, None

    # Convert to grayscale for processing
    gray = _prepare_grayscale_image(image)

    # Try multiple detection methods
    detection_methods = [
        lambda: decode(gray),
        lambda: _decode_with_adaptive_threshold(gray),
        lambda: _decode_with_gaussian_blur(gray),
        lambda: _decode_with_histogram_equalization(gray),
        lambda: _decode_with_unsharp_mask(gray)
    ]

    qr_codes = None
    for method in detection_methods:
        try:
            qr_codes = method()
            if qr_codes:
                logger.debug(
                    f"QR code detected using method: {method.__name__}")
                break
        except Exception as e:
            logger.debug(f"Detection method failed: {str(e)}")
            continue

    if not qr_codes:
        logger.debug("No QR codes detected with any method.")
        return None, None

    # Extract QR code data and information
    return _extract_qr_information(qr_codes[0])


def _prepare_grayscale_image(image):
    """
    Convert image to grayscale if needed.

    Args:
            image: Input image

    Returns:
            numpy.ndarray: Grayscale image
    """
    if len(image.shape) == 2:
        return image
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _decode_with_adaptive_threshold(gray):
    """
    Decode QR code using adaptive thresholding preprocessing.
    """
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return decode(adaptive)


def _decode_with_gaussian_blur(gray):
    """
    Decode QR code using Gaussian blur preprocessing.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return decode(blurred)


def _decode_with_histogram_equalization(gray):
    """
    Decode QR code using histogram equalization preprocessing.
    """
    equalized = cv2.equalizeHist(gray)
    return decode(equalized)


def _decode_with_unsharp_mask(gray):
    """
    Decode QR code using unsharp masking preprocessing.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return decode(sharpened)


def _extract_qr_information(qr):
    """
    Extract QR code data and metadata from detected QR object.

    Args:
            qr: Decoded QR code object from pyzbar

    Returns:
            tuple: (qr_data, qr_info)
    """
    try:
        qr_data = qr.data.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning("Failed to decode QR data as UTF-8")
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
