# app/pages/analyzer_page.py
import streamlit as st
import logging
from app.utils.image_processing import load_and_preprocess_image
from app.utils.config import get_default_parameters
from app.services.pipeline_service import process_full_pipeline
from app.components.analyzer_sidebar import render_analyzer_sidebar
from app.components.pipeline_tabs import display_tabbed_pipeline
from app.utils.image_display import safe_display_image
from app.utils.circle_detection_debug import visualize_detection_stages

logger = logging.getLogger(__name__)


def show_analyzer_page():
    """
    Main entry point for the answer sheet analysis page.
    """
    st.header("Answer Sheet Analysis")
    st.write("Upload an answer sheet to analyze it with our pipeline.")

    # Get default parameters and render sidebar
    default_params = get_default_parameters()
    params = render_analyzer_sidebar(default_params)

    # File upload section
    uploaded_file = _handle_file_upload()

    if uploaded_file is not None:
        _process_uploaded_file(uploaded_file, params)


def _handle_file_upload():
    """
    Handle file upload with validation.

    Returns:
        UploadedFile: The uploaded file or None
    """
    return st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="analyzer_uploader",
        help="Upload an answer sheet image in JPG or PNG format"
    )


def _process_uploaded_file(uploaded_file, params):
    """
    Process the uploaded file through the analysis pipeline.

    Args:
        uploaded_file: StreamLit uploaded file object
        params: Processing parameters dictionary
    """
    try:
        # Load and preprocess the image
        image = load_and_preprocess_image(uploaded_file)

        # Create view tabs
        tabs = st.tabs([
            "Original Image",
            "Analysis Pipeline",
            "Debug Analysis"
        ])

        with tabs[0]:
            _display_original_image(image)

        with tabs[1]:
            _display_analysis_pipeline(image, params)

        with tabs[2]:
            _display_debug_analysis(image, params)

    except ValueError as ve:
        st.error(f"File error: {str(ve)}")
        logger.error(f"File processing error: {str(ve)}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.exception("Unexpected error in file processing")


def _display_original_image(image):
    """
    Display the original uploaded image.

    Args:
        image: OpenCV image array
    """
    st.header("Original Image")
    safe_display_image(image)

    # Display image properties
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Width", f"{image.shape[1]}px")
    with col2:
        st.metric("Height", f"{image.shape[0]}px")
    with col3:
        channels = image.shape[2] if len(image.shape) > 2 else 1
        st.metric("Channels", channels)


def _display_analysis_pipeline(image, params):
    """
    Display the complete analysis pipeline.

    Args:
        image: OpenCV image array
        params: Processing parameters
    """
    st.header("Complete Analysis Pipeline")

    # Add progress indicator
    with st.spinner("Processing image through pipeline..."):
        pipeline_results = process_full_pipeline(image, params)

    # Display pipeline results
    display_tabbed_pipeline(pipeline_results)


def _display_debug_analysis(image, params):
    """
    Display debug analysis for circle detection.

    Args:
        image: OpenCV image array
        params: Processing parameters
    """
    st.header("Detection Debug Analysis")

    # Enable debug mode checkbox
    if not st.checkbox("Enable Debug Mode", value=False):
        st.info("Check the box above to enable debug visualization")
        return

    try:
        with st.spinner("Running debug analysis..."):
            # Get QR polygon and corner boundary for debugging
            from app.services.qr_service import process_qr_code
            from app.utils.corner_detection import (
                detect_corner_markers,
                get_bounding_quadrilateral
            )

            # Detect QR code
            qr_results = process_qr_code(image)
            qr_polygon = None
            if qr_results['qr_info'] and 'polygon' in qr_results['qr_info']:
                qr_polygon = qr_results['qr_info']['polygon']

            # Detect corners
            corners = detect_corner_markers(image, qr_polygon=qr_polygon)
            answer_boundary = None
            if corners:
                answer_boundary = get_bounding_quadrilateral(
                    corners, margin=20)

            # Run debug visualization
            debug_results = visualize_detection_stages(
                image, params, qr_polygon, answer_boundary
            )

            # Display debug results
            _display_debug_results(debug_results)

    except ImportError as ie:
        st.error(f"Missing dependency: {str(ie)}")
        logger.error(f"Import error in debug analysis: {str(ie)}")
    except Exception as e:
        st.error(f"Debug analysis error: {str(e)}")
        logger.exception("Error in debug analysis")


def _display_debug_results(debug_results):
    """
    Display debug visualization results.

    Args:
        debug_results: Dictionary of debug visualization data
    """
    if not debug_results:
        st.warning("No debug results available")
        return

    # Display each debug stage
    stages_to_display = ['raw', 'boundary',
                         'qr_excluded', 'comparison', 'exclusion_map']

    for stage in stages_to_display:
        if stage not in debug_results:
            continue

        data = debug_results[stage]
        st.subheader(f"Stage: {stage.replace('_', ' ').title()}")

        # Display statistics if available
        if 'stats' in data:
            stats = data['stats']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Circles", stats['count'])
            with col2:
                st.metric("Left Column", stats['left'])
            with col3:
                st.metric("Right Column", stats['right'])

        # Display the visualization image
        if 'image' in data:
            safe_display_image(data['image'])

        st.divider()

    # Add download button for debug images
    if st.button("Export Debug Images"):
        _export_debug_images(debug_results)


def _export_debug_images(debug_results):
    """
    Export debug images to files.

    Args:
        debug_results: Dictionary of debug visualization data
    """
    import cv2
    import tempfile
    import zipfile
    import io

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stage, data in debug_results.items():
                    if 'image' in data:
                        filename = f"{stage}.png"
                        filepath = f"{temp_dir}/{filename}"
                        cv2.imwrite(filepath, data['image'])
                        zipf.write(filepath, filename)

            # Provide download
            st.download_button(
                label="Download Debug Images",
                data=zip_buffer.getvalue(),
                file_name="debug_analysis.zip",
                mime="application/zip"
            )

    except Exception as e:
        st.error(f"Error exporting images: {str(e)}")
        logger.exception("Error in debug image export")
