# app/ui/pages/analyzer.py
"""
Streamlit UI for analyzer page.
"""
import streamlit as st
from typing import Optional

from app.features.analyzer.service import AnalyzerService
from app.ui.components.sidebar import render_analyzer_sidebar
from app.ui.components.file_upload import handle_file_upload
from app.ui.pipelines.results import display_pipeline_results
from app.core.config import get_default_parameters


class AnalyzerPage:
    """
    Handles the analyzer page UI.
    """

    def __init__(self):
        self.default_params = get_default_parameters()

    def render(self):
        """
        Render the analyzer page.
        """
        st.header("Answer Sheet Analysis")
        st.write("Upload an answer sheet to analyze.")

        # Render sidebar and get parameters
        params = render_analyzer_sidebar(self.default_params)

        # Handle file upload
        uploaded_file = handle_file_upload()

        if uploaded_file is not None:
            self._process_file(uploaded_file, params)

    def _process_file(
        self,
        uploaded_file,
        params: dict
    ):
        """
        Process uploaded file.
        """
        try:
            # Load image
            from app.common.image.loading import load_uploaded_image
            image = load_uploaded_image(uploaded_file)

            # Create service and analyze
            analyzer = AnalyzerService(params)
            results = analyzer.analyze(image)

            # Display results
            display_pipeline_results(results)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


def show_analyzer_page():
    """
    Entry point for analyzer page.
    """
    page = AnalyzerPage()
    page.render()
