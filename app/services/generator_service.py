# app/services/generator_service.py
import os
import io
import zipfile
import numpy as np
import cv2
from PIL import Image
from app.utils.sheet_generator import create_answer_sheet, get_default_header_fields
from app.utils.shade_generator import shade_random_bubbles


def generate_blank_sheet(
        title="FIRST SEMESTER",
        student_id="000000",
        student_name="",
        exam_part=1,
        items_per_part=60,
        show_corner_markers=True,
        custom_header_fields=None
):
    """
    Generates a blank answer sheet with advanced features.

    Args:
            title: Title of the exam
            student_id: Student ID for QR code
            student_name: Student name to display
            exam_part: Which part of the exam (affects question numbering)
            items_per_part: Number of items per exam part
            show_corner_markers: Whether to include corner markers
            custom_header_fields: Custom header fields configuration

    Returns:
            numpy.ndarray: Generated answer sheet image
    """
    # Calculate starting question number based on exam part
    start_number = (exam_part - 1) * items_per_part + 1

    # Get header fields
    if custom_header_fields is None:
        header_fields = get_default_header_fields(student_name)
    else:
        header_fields = custom_header_fields

    # Create the sheet
    sheet = create_answer_sheet(
        title=title,
        student_id=student_id,
        start_number=start_number,
        show_corner_markers=show_corner_markers,
        header_fields=header_fields
    )

    # Convert PIL image to numpy array for OpenCV
    sheet_np = np.array(sheet)
    return cv2.cvtColor(sheet_np, cv2.COLOR_RGB2BGR)


def generate_shaded_sheet(
        title="FIRST SEMESTER",
        student_id="000000",
        student_name="",
        exam_part=1,
        items_per_part=60,
        show_corner_markers=True,
        custom_header_fields=None,
        num_answers=20,
        shade_intensity=(100, 200),
        offset_range=2,
        size_percent=0.2,
        predefined_answers=None
):
    """
    Generates an answer sheet with randomly shaded bubbles.

    Args:
            All parameters from generate_blank_sheet plus:
            num_answers: Number of bubbles to shade randomly
            shade_intensity: Range of shade darkness (min, max) 0-255
            offset_range: Maximum pixel offset for random positioning
            size_percent: Percentage of circle radius to add/subtract
            predefined_answers: Optional predefined answers

    Returns:
            numpy.ndarray: Generated answer sheet image with shaded bubbles
    """
    # Generate blank sheet
    sheet = generate_blank_sheet(
        title=title,
        student_id=student_id,
        student_name=student_name,
        exam_part=exam_part,
        items_per_part=items_per_part,
        show_corner_markers=show_corner_markers,
        custom_header_fields=custom_header_fields
    )

    # Shade random bubbles
    shaded_sheet = shade_random_bubbles(
        sheet,
        num_answers=num_answers,
        shade_intensity=shade_intensity,
        offset_range=offset_range,
        size_percent=size_percent,
        answers=predefined_answers
    )

    return shaded_sheet


def generate_exam_sheets(
        exam_id,
        exam_title,
        students,
        exam_parts=1,
        items_per_part=60,
        show_corner_markers=True
):
    """
    Generate sheets for multiple students for a specific exam.

    Args:
            exam_id: ID of the exam
            exam_title: Title of the exam
            students: List of student dictionaries with id and name
            exam_parts: Number of parts in the exam
            items_per_part: Number of items per part
            show_corner_markers: Whether to include corner markers

    Returns:
            list: List of tuples (filename, image)
    """
    sheets = []

    for student in students:
        for part in range(1, exam_parts + 1):
            sheet = generate_blank_sheet(
                title=exam_title,
                student_id=student['id'],
                student_name=student['name'],
                exam_part=part,
                items_per_part=items_per_part,
                show_corner_markers=show_corner_markers
            )

            filename = f"sheet_{student['id']}_part{part}.jpg"
            sheets.append((filename, sheet))

    return sheets


def generate_multiple_sheets(
        num_sheets,
        title_prefix="FIRST SEMESTER",
        id_prefix="STUDENT",
        id_start=1,
        exam_parts=1,
        items_per_part=60,
        num_answers=20,
        shade_intensity=(100, 200),
        offset_range=2,
        size_percent=0.2,
        show_corner_markers=True
):
    """
    Generates multiple answer sheets with random shading.

    Args:
            num_sheets: Number of sheets to generate
            title_prefix: Prefix for sheet titles
            id_prefix: Prefix for student IDs
            id_start: Starting number for student IDs
            exam_parts: Number of exam parts
            items_per_part: Number of items per part
            num_answers: Number of bubbles to shade randomly
            shade_intensity: Range of shade darkness
            offset_range: Maximum pixel offset
            size_percent: Percentage of circle radius variation
            show_corner_markers: Whether to include corner markers

    Returns:
            list: List of tuples (filename, image)
    """
    sheets = []

    for i in range(num_sheets):
        sheet_num = id_start + i
        student_id = f"{id_prefix}{sheet_num:06d}"
        student_name = f"Test Student {sheet_num}"

        for part in range(1, exam_parts + 1):
            # Generate shaded sheet for each part
            shaded_sheet = generate_shaded_sheet(
                title=title_prefix,
                student_id=student_id,
                student_name=student_name,
                exam_part=part,
                items_per_part=items_per_part,
                show_corner_markers=show_corner_markers,
                num_answers=num_answers,
                shade_intensity=shade_intensity,
                offset_range=offset_range,
                size_percent=size_percent
            )

            filename = f"sheet_{student_id}_part{part}.jpg"
            sheets.append((filename, shaded_sheet))

    return sheets


def create_download_zip(sheets):
    """
    Creates a ZIP file containing generated sheets.

    Args:
            sheets: List of tuples (filename, image)

    Returns:
            bytes: ZIP file as bytes
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image in sheets:
            success, buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                zip_file.writestr(filename, buffer.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
