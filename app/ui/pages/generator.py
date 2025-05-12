# app/ui/pages/generator.py
"""
Streamlit UI for generator page.
"""
import streamlit as st
import cv2
from typing import Dict
import logging

from app.features.generator.service import GeneratorService
from app.ui.components.generator_tabs import render_generator_tabs
from app.common.image.display import display_image

logger = logging.getLogger(__name__)

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

        # --- Collect Parameters ---
        params_to_pass = {}

        with col1:
            # Store parameters directly in the dictionary
            params_to_pass['title'] = st.text_input("Title", "FIRST SEMESTER")
            params_to_pass['student_id'] = st.text_input(
                "Student ID", "STUDENT123456")
            params_to_pass['student_name'] = st.text_input(
                "Student Name", "Test Student")
            params_to_pass['exam_part'] = st.number_input(
                "Exam Part", 1, 10, 1)
            params_to_pass['items_per_part'] = st.number_input(
                "Items", 10, 100, 60)

        with col2:
            params_to_pass['show_corner_markers'] = st.checkbox(
                "Show Corner Markers", True)
            
            # Add corner pattern selection
            if params_to_pass['show_corner_markers']:
                params_to_pass['corner_pattern'] = st.selectbox(
                    "Corner Pattern",
                    options=["concentric", "l_shape", "cross", "checkerboard"],
                    index=0,
                    help="Select corner marker pattern for better detection"
                )
            
            # Keep this local for conditional input
            add_shading = st.checkbox("Add Random Shading", True)

            # Parameters specific to shading
            if add_shading:
                params_to_pass['num_answers'] = st.slider(
                    "Number of Answers to Shade", 
                    1, params_to_pass['items_per_part'], 20)

        # Advanced options in expander
        with st.expander("Advanced Options"):
            st.markdown("### Corner Pattern Preview")
            pattern_info = {
                "concentric": "Concentric squares - Best for most scanners",
                "l_shape": "L-shaped corners - Good for orientation detection",
                "cross": "Cross pattern - Simple but effective",
                "checkerboard": "Checkerboard - High contrast pattern"
            }
            
            pattern = params_to_pass.get('corner_pattern', 'concentric')
            st.info(pattern_info.get(pattern, ""))
            
            # You could add a preview image here
            # preview_img = self._get_pattern_preview(pattern)
            # st.image(preview_img, width=100)

        # --- Trigger Generation ---
        if st.button("Generate"):
            # Pass the explicitly created dictionary
            self._generate_single(params_to_pass, add_shading)

    def _generate_single(self, generation_params: Dict, add_shading: bool):
        """
        Generate single sheet with given parameters.
        """
        logger.debug(
            f"Generating single sheet with params: {generation_params}, Shading: {add_shading}")
        try:
            if add_shading:
                # generate_shaded handles extracting its specific params from kwargs
                sheet = self.generator_service.generate_shaded(
                    **generation_params)
            else:
                # generate_blank handles extracting its specific params from kwargs
                sheet = self.generator_service.generate_blank(
                    **generation_params)

            st.success("Sheet generated!")
            display_image(sheet)

            # Add download button
            try:
                is_success, buffer = cv2.imencode('.jpg', sheet)
                if is_success:
                    st.download_button(
                        label="Download Sheet (JPG)",
                        data=buffer.tobytes(),
                        file_name="answer_sheet.jpg",
                        mime="image/jpeg"
                    )
                else:
                    st.warning("Could not encode image for download.")
            except Exception as enc_e:
                logger.error(
                    f"Error encoding generated sheet for download: {enc_e}")
                st.error("Failed to prepare image for download.")

        except Exception as e:
            logger.error(f"Error generating sheet: {e}", exc_info=True)
            st.error(f"Error generating sheet: {str(e)}")

    # --- Placeholder methods for other tabs ---
    def _render_multiple_sheets(self):
        """
        Render multiple sheets generator (Placeholder).
        """
        st.info("Multiple sheet generation not yet implemented.")
        pass

    def _render_exam_batch(self):
        """
        Render exam batch generator (Placeholder).
        """
        st.info("Exam batch generation not yet implemented.")
        pass

    def _render_custom_template(self):
        """
        Render custom template generator (Placeholder).
        """
        st.info("Custom template generation not yet implemented.")
        pass


def show_generator_page():
    """
    Entry point for generator page.
    """
    page = GeneratorPage()
    page.render()
