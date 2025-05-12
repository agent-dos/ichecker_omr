# app/ui/pages/generator.py
"""
Streamlit UI for generator page.
"""
import streamlit as st
import cv2
from typing import Dict

from app.features.generator.service import GeneratorService
from app.ui.components.generator_tabs import render_generator_tabs
from app.common.image.display import display_image


class GeneratorPage:
    """
    Handles the generator page UI.
    """

    def __init__(self):
        self.generator_service = GeneratorService()

    def render(self):
        """
        Render the generator page.
        """
        st.header("Answer Sheet Generator")
        st.write("Generate professional answer sheets.")

        # Render tab selection
        tab_choice = render_generator_tabs()

        # Handle selected tab
        if tab_choice == "single":
            self._render_single_sheet()
        elif tab_choice == "multiple":
            self._render_multiple_sheets()
        elif tab_choice == "batch":
            self._render_exam_batch()
        elif tab_choice == "custom":
            self._render_custom_template()

    def _render_single_sheet(self):
        """
        Render single sheet generator.
        """
        st.subheader("Generate Single Sheet")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Title", "FIRST SEMESTER")
            student_id = st.text_input("Student ID", "STUDENT123456")
            student_name = st.text_input("Student Name", "Test Student")

            exam_part = st.number_input("Exam Part", 1, 10, 1)
            items_per_part = st.number_input("Items", 10, 100, 60)

        with col2:
            show_corners = st.checkbox("Show Corner Markers", True)
            add_shading = st.checkbox("Add Random Shading", True)

            if add_shading:
                num_answers = st.slider("Answers", 1, 60, 20)

        if st.button("Generate"):
            self._generate_single(locals())

    def _generate_single(self, params: Dict):
        """
        Generate single sheet with given parameters.
        """
        try:
            if params.get('add_shading'):
                sheet = self.generator_service.generate_shaded(**params)
            else:
                sheet = self.generator_service.generate_blank(**params)

            st.success("Sheet generated!")
            display_image(sheet)

            # Add download button
            _, buffer = cv2.imencode('.jpg', sheet)
            st.download_button(
                "Download",
                buffer.tobytes(),
                "answer_sheet.jpg",
                "image/jpeg"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")

    def _render_multiple_sheets(self):
        """
        Render multiple sheets generator.
        """
        # Implementation similar to single sheet
        pass

    def _render_exam_batch(self):
        """
        Render exam batch generator.
        """
        # Implementation for batch generation
        pass

    def _render_custom_template(self):
        """
        Render custom template generator.
        """
        # Implementation for custom templates
        pass


def show_generator_page():
    """
    Entry point for generator page.
    """
    page = GeneratorPage()
    page.render()
