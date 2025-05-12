# filename: app/ui/pages/analyzer.py
import streamlit as st
from typing import Optional, Dict  # Added Dict
import logging  # Import logging

from app.features.analyzer.service import AnalyzerService
from app.ui.components.sidebar import render_analyzer_sidebar
from app.ui.components.file_upload import handle_file_upload
from app.ui.pipelines.results import display_pipeline_results
# No config import needed here anymore


class AnalyzerPage:
    """
    Handles the analyzer page UI.
    """

    # No __init__ needed

    def render(self):
        """
        Render the analyzer page.
        """
        st.header("Answer Sheet Analysis")
        st.write("Upload an answer sheet to analyze.")

        # Render sidebar and get the *current* parameters reflecting UI state
        # Corrected call: No argument needed for render_analyzer_sidebar
        params_from_sidebar = render_analyzer_sidebar()

        # Handle file upload
        uploaded_file = handle_file_upload()

        if uploaded_file is not None:
            # Pass the latest parameters from the sidebar to _process_file
            self._process_file(uploaded_file, params_from_sidebar)

    def _process_file(
        self,
        uploaded_file,
        params: dict  # Parameter dict comes directly from render_analyzer_sidebar
    ):
        """
        Process uploaded file using the provided parameters.
        """
        # Add a check for params
        if not params:
            st.error("Configuration parameters are missing. Cannot process image.")
            # Add logging
            logging.error("Params dictionary missing in _process_file.")
            return

        try:
            # Load image
            from app.common.image.loading import load_uploaded_image
            image = load_uploaded_image(uploaded_file)
            # Log image load
            logging.info(
                f"Image loaded: {uploaded_file.name}, size: {image.shape}")

            # Create service and analyze using the parameters from the sidebar
            # The AnalyzerService __init__ expects the full config dict
            logging.debug("Initializing AnalyzerService...")
            analyzer = AnalyzerService(params)
            logging.debug("AnalyzerService initialized. Starting analysis...")
            # analyze method uses params stored in self.params
            results = analyzer.analyze(image)
            logging.info("Analysis complete.")

            # Display results
            display_pipeline_results(results)

        except ImportError as e:
            # Catch potential import errors if components moved/renamed
            st.error(f"Import error during processing: {e}")
            logging.exception("Import error in _process_file")
        except Exception as e:
            # Catch other errors during analysis
            st.error(f"Error processing image: {str(e)}")
            # Log the full exception traceback for debugging
            logging.exception("Error in _process_file")


def show_analyzer_page():
    """
    Entry point for analyzer page.
    """
    page = AnalyzerPage()
    page.render()
