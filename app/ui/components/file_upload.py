# app/ui/components/file_upload.py
"""
File upload component.
"""
import streamlit as st
from typing import Optional


def handle_file_upload():
    """
    Handle file upload with validation.
    """
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an answer sheet image"
    )

    if uploaded_file is not None:
        # Validate file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File too large. Please upload a file smaller than 10MB.")
            return None

    return uploaded_file
