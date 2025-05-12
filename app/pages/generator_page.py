# app/pages/generator_page.py
import streamlit as st
import cv2
from app.services.generator_service import (
    generate_blank_sheet,
    generate_shaded_sheet,
    generate_multiple_sheets,
    create_download_zip
)
from app.utils.image_display import safe_display_image


def show_generator_page():
    """
    Displays the answer sheet generator page with advanced features.
    """
    st.header("Answer Sheet Generator")
    st.write("Generate professional answer sheets with QR codes and custom numbering.")

    # Create tabs for different generator options
    generator_tabs = st.tabs([
        "Single Sheet",
        "Multiple Sheets",
        "Exam Batch",
        "Custom Template"
    ])

    with generator_tabs[0]:
        _show_single_sheet_generator()

    with generator_tabs[1]:
        _show_multiple_sheet_generator()

    with generator_tabs[2]:
        _show_exam_batch_generator()

    with generator_tabs[3]:
        _show_custom_template_generator()


def _show_single_sheet_generator():
    """Display UI for generating a single answer sheet."""
    st.subheader("Generate a Single Answer Sheet")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Sheet Title", "FIRST SEMESTER")
        student_id = st.text_input("Student ID", "STUDENT123456")
        student_name = st.text_input("Student Name", "Test Student")

        # Exam configuration
        st.subheader("Exam Configuration")
        exam_part = st.number_input("Exam Part Number", 1, 10, 1)
        items_per_part = st.number_input("Items per Part", 10, 100, 60)
        start_question = (exam_part - 1) * items_per_part + 1
        st.info(f"Questions will start from #{start_question}")

    with col2:
        st.subheader("Sheet Options")
        show_corner_markers = st.checkbox("Show Corner Markers", True)
        add_shading = st.checkbox("Add Random Shading", True)

        # Only show shading options if add_shading is checked
        if add_shading:
            num_answers = st.slider("Number of Answers", 1, items_per_part, 20)

            # Create columns for shade parameters
            shade_col1, shade_col2 = st.columns(2)

            with shade_col1:
                shade_min = st.slider("Min Shade Intensity", 50, 200, 100)
                offset_range = st.slider("Position Offset", 0, 5, 2)

            with shade_col2:
                shade_max = st.slider(
                    "Max Shade Intensity", shade_min, 255, 180)
                size_percent = st.slider("Size Variation", 0.0, 0.5, 0.2, 0.01)

    # Generate button
    if st.button("Generate Sheet"):
        with st.spinner("Generating answer sheet..."):
            try:
                if add_shading:
                    sheet = generate_shaded_sheet(
                        title=title,
                        student_id=student_id,
                        student_name=student_name,
                        exam_part=exam_part,
                        items_per_part=items_per_part,
                        show_corner_markers=show_corner_markers,
                        num_answers=num_answers,
                        shade_intensity=(shade_min, shade_max),
                        offset_range=offset_range,
                        size_percent=size_percent
                    )
                else:
                    sheet = generate_blank_sheet(
                        title=title,
                        student_id=student_id,
                        student_name=student_name,
                        exam_part=exam_part,
                        items_per_part=items_per_part,
                        show_corner_markers=show_corner_markers
                    )

                st.success("Sheet generated successfully!")
                safe_display_image(sheet)

                # Add download button
                success, buffer = cv2.imencode(
                    ".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    sheet_bytes = buffer.tobytes()
                    st.download_button(
                        "Download Sheet",
                        sheet_bytes,
                        f"sheet_{student_id}_part{exam_part}.jpg",
                        "image/jpeg"
                    )
            except Exception as e:
                st.error(f"Error generating sheet: {str(e)}")


def _show_multiple_sheet_generator():
    """Display UI for generating multiple answer sheets."""
    st.subheader("Generate Multiple Answer Sheets")

    col1, col2 = st.columns(2)

    with col1:
        num_sheets = st.number_input("Number of Students", 1, 100, 5)
        title_prefix = st.text_input(
            "Title", "FIRST SEMESTER", key="title_multi")
        id_prefix = st.text_input("ID Prefix", "STUDENT")
        id_start = st.number_input("Starting ID Number", 1, 10000, 1)

        # Exam configuration
        st.subheader("Exam Configuration")
        exam_parts = st.number_input("Number of Exam Parts", 1, 5, 1)
        items_per_part = st.number_input(
            "Items per Part", 10, 100, 60, key="items_multi")

    with col2:
        st.subheader("Sheet Options")
        show_corner_markers = st.checkbox(
            "Show Corner Markers", True, key="corners_multi")
        num_answers = st.slider("Answers Per Sheet", 1,
                                60, 20, key="num_ans_multi")

        # Create columns for shade parameters
        shade_col1, shade_col2 = st.columns(2)

        with shade_col1:
            shade_min = st.slider("Min Shade", 50, 200,
                                  100, key="shade_min_multi")
            offset_range = st.slider("Offset", 0, 5, 2, key="offset_multi")

        with shade_col2:
            shade_max = st.slider("Max Shade", shade_min,
                                  255, 180, key="shade_max_multi")
            size_percent = st.slider(
                "Size Var", 0.0, 0.5, 0.2, 0.01, key="size_multi")

    # Generate button
    if st.button("Generate Sheets"):
        with st.spinner(f"Generating {num_sheets * exam_parts} answer sheets..."):
            try:
                sheets = generate_multiple_sheets(
                    num_sheets=num_sheets,
                    title_prefix=title_prefix,
                    id_prefix=id_prefix,
                    id_start=id_start,
                    exam_parts=exam_parts,
                    items_per_part=items_per_part,
                    num_answers=num_answers,
                    shade_intensity=(shade_min, shade_max),
                    offset_range=offset_range,
                    size_percent=size_percent,
                    show_corner_markers=show_corner_markers
                )

                # Display preview of first sheet
                if sheets:
                    st.success(f"{len(sheets)} sheets generated successfully!")
                    st.subheader("Preview of first sheet:")
                    safe_display_image(sheets[0][1])

                    # Create ZIP file for download
                    zip_bytes = create_download_zip(sheets)

                    st.download_button(
                        "Download All Sheets (ZIP)",
                        zip_bytes,
                        "answer_sheets.zip",
                        "application/zip"
                    )
            except Exception as e:
                st.error(f"Error generating sheets: {str(e)}")


def _show_exam_batch_generator():
    """Display UI for generating sheets for a specific exam."""
    st.subheader("Generate Sheets for Exam")
    st.info("This feature would connect to a database to generate sheets for all students in a class.")

    # Placeholder interface
    exam_id = st.text_input("Exam ID", "EXAM001")
    exam_title = st.text_input("Exam Title", "Final Examination")

    # Student list (in real app, this would come from database)
    st.subheader("Students")
    num_students = st.number_input("Number of Students", 1, 50, 5)

    students = []
    for i in range(num_students):
        student_id = f"STU{i+1:06d}"
        student_name = f"Student {i+1}"
        students.append({'id': student_id, 'name': student_name})

    st.dataframe(students)

    # Additional options
    exam_parts = st.number_input("Exam Parts", 1, 5, 1, key="exam_parts_batch")
    items_per_part = st.number_input(
        "Items per Part", 10, 100, 60, key="items_batch")
    show_corner_markers = st.checkbox(
        "Show Corner Markers", True, key="corners_batch")

    if st.button("Generate Exam Sheets"):
        st.warning("Database integration not implemented in this demo.")
        st.info(
            "In production, this would generate sheets for all students in the exam.")


def _show_custom_template_generator():
    """Display UI for creating custom answer sheets."""
    st.subheader("Custom Answer Pattern")

    st.info(
        "Create answer sheets with specific answers marked (useful for answer keys).")

    # Basic sheet info
    col1, col2 = st.columns(2)

    with col1:
        custom_title = st.text_input(
            "Sheet Title", "ANSWER KEY", key="title_custom")
        custom_id = st.text_input("Sheet ID", "KEY_001", key="id_custom")
        custom_name = st.text_input(
            "Sheet Name", "Answer Key", key="name_custom")

    with col2:
        exam_part = st.number_input("Exam Part", 1, 5, 1, key="part_custom")
        items_per_part = st.number_input(
            "Items per Part", 10, 100, 60, key="items_custom")
        show_corners = st.checkbox(
            "Show Corner Markers", True, key="corners_custom")

    # Create answer selection grid
    st.write("Select answers to shade on the sheet:")

    start_num = (exam_part - 1) * items_per_part + 1
    end_num = start_num + min(60, items_per_part) - \
        1  # Limited to 60 on one sheet

    # Create columns for answer selection
    st.write(f"Questions {start_num}-{end_num}:")

    # Dictionary to store selected answers
    selected_answers = {}

    # Create a 6x10 grid for up to 60 questions
    for row_idx in range(10):  # 10 rows
        cols = st.columns(6)  # 6 columns

        for col_idx in range(6):
            q_num = start_num + row_idx * 6 + col_idx
            if q_num > end_num:
                break

            with cols[col_idx]:
                options = ["None", "A", "B", "C", "D", "E"]
                selected = st.selectbox(
                    f"Q{q_num}",
                    options,
                    key=f"q{q_num}_custom"
                )

                if selected != "None":
                    # Map to row key format used by shade generator
                    row_key = f"row_{q_num - start_num}"
                    selected_answers[row_key] = options.index(selected) - 1

    # Generate button
    if st.button("Generate Custom Sheet"):
        with st.spinner("Generating custom answer sheet..."):
            try:
                sheet = generate_shaded_sheet(
                    title=custom_title,
                    student_id=custom_id,
                    student_name=custom_name,
                    exam_part=exam_part,
                    items_per_part=items_per_part,
                    show_corner_markers=show_corners,
                    # Consistent shading for answer key
                    shade_intensity=(120, 120),
                    offset_range=0,  # No randomness
                    size_percent=0,  # No size variation
                    predefined_answers=selected_answers
                )

                st.success("Custom answer sheet generated!")
                safe_display_image(sheet)

                # Add download button
                success, buffer = cv2.imencode(
                    ".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    sheet_bytes = buffer.tobytes()
                    st.download_button(
                        "Download Custom Sheet",
                        sheet_bytes,
                        f"custom_sheet_{custom_id}.jpg",
                        "image/jpeg"
                    )
            except Exception as e:
                st.error(f"Error generating custom sheet: {str(e)}")
