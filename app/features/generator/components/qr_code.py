# app/features/generator/components/qr_code.py
"""
QR code component for answer sheets.
"""
import qrcode
from PIL import Image, ImageDraw
import logging

from app.core.constants import (
    QR_CODE_SIZE, QR_CODE_Y_POS,
    QR_CODE_COLOR, QR_CODE_BG_COLOR
)

logger = logging.getLogger(__name__)


class QRCodeGenerator:
    """
    Generates and draws QR codes.
    """

    def draw(
        self,
        img: Image.Image,
        student_id: str,
        page_width: int
    ) -> None:
        """
        Generate and draw QR code on image.
        """
        qr_x_pos = (page_width - QR_CODE_SIZE) // 2

        try:
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=5,
                border=2,
            )
            qr.add_data(student_id)
            qr.make(fit=True)

            # Create QR image
            qr_img = qr.make_image(
                fill_color=QR_CODE_COLOR,
                back_color=QR_CODE_BG_COLOR
            )
            qr_img = qr_img.resize((QR_CODE_SIZE, QR_CODE_SIZE))

            # Paste onto main image
            img.paste(qr_img, (qr_x_pos, QR_CODE_Y_POS))

        except Exception as e:
            logger.error(f"Error generating QR code: {e}")

            # Draw error rectangle
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                (qr_x_pos, QR_CODE_Y_POS,
                 qr_x_pos + QR_CODE_SIZE, QR_CODE_Y_POS + QR_CODE_SIZE),
                outline="red"
            )
