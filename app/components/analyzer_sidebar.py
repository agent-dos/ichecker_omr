# app/components/analyzer_sidebar.py
import streamlit as st
from app.utils.config import save_parameters


def render_analyzer_sidebar(default_params):
    """
    Render the sidebar with enhanced calibration guidance.
    """
    params = default_params.copy()

    st.sidebar.title("Analysis Settings")
    st.sidebar.info(
        "Adjust these parameters if bubbles aren't being detected correctly.")

    # Add calibration tips
    with st.sidebar.expander("🎯 Calibration Tips", expanded=True):
        st.write("""
		**If filled bubbles aren't detected:**
		- Decrease 'Bubble Fill Threshold'
		- Increase 'Score Multiplier'
		- Adjust 'Threshold Constant'
		
		**If empty bubbles are detected as filled:**
		- Increase 'Bubble Fill Threshold'
		- Decrease 'Score Multiplier'
		""")

    # Create parameter sections in sidebar
    with st.sidebar.expander("Circle Detection", expanded=False):
        params['param1'] = st.sidebar.slider(
            "Edge Detection Sensitivity",
            10, 100, default_params['param1'], 1,
            help="Parameter for edge detection (higher values detect fewer circles)"
        )
        params['param2'] = st.sidebar.slider(
            "Circle Accumulator Threshold",
            5, 50, default_params['param2'], 1,
            help="Parameter for circle detection (lower values detect more circles)"
        )
        params['min_radius'] = st.sidebar.slider(
            "Minimum Bubble Radius",
            5, 30, default_params['min_radius'], 1,
            help="Minimum radius of detectable bubbles (pixels)"
        )
        params['max_radius'] = st.sidebar.slider(
            "Maximum Bubble Radius",
            10, 50, default_params['max_radius'], 1,
            help="Maximum radius of detectable bubbles (pixels)"
        )

    with st.sidebar.expander("Bubble Analysis", expanded=True):
        params['bubble_threshold'] = st.sidebar.slider(
            "Bubble Fill Threshold",
            10.0, 300.0, default_params['bubble_threshold'], 5.0,
            help="Threshold to determine if a bubble is filled (lower = more sensitive)"
        )
        params['block_size'] = st.sidebar.slider(
            "Adaptive Threshold Block Size",
            3, 51, default_params['block_size'], 2,
            help="Size of pixel neighborhood for adaptive thresholding (must be odd)"
        )
        params['c_value'] = st.sidebar.slider(
            "Threshold Constant",
            0, 30, default_params['c_value'], 1,
            help="Constant subtracted from mean in adaptive thresholding"
        )
        params['score_multiplier'] = st.sidebar.slider(
            "Score Multiplier",
            0.5, 3.0, default_params['score_multiplier'], 0.1,
            help="Multiplier applied to the fill score"
        )

    with st.sidebar.expander("Layout Parameters", expanded=False):
        params['row_threshold'] = st.sidebar.slider(
            "Row Distance Threshold",
            3, 20, default_params['row_threshold'], 1,
            help="Maximum vertical distance for circles to be considered in the same row"
        )
        params['resize_factor'] = st.sidebar.slider(
            "Resize Factor",
            0.5, 2.0, default_params['resize_factor'], 0.1,
            help="Factor to resize image for processing (1.0 = original size)"
        )

    # Add quick presets
    st.sidebar.subheader("Quick Presets")
    if st.sidebar.button("High Sensitivity"):
        params['bubble_threshold'] = 50.0
        params['score_multiplier'] = 2.0
        params['c_value'] = 5

    if st.sidebar.button("Low Sensitivity"):
        params['bubble_threshold'] = 150.0
        params['score_multiplier'] = 1.0
        params['c_value'] = 15

    # Add button to save parameters
    if st.sidebar.button("Save as Default"):
        try:
            save_parameters(params)
            st.sidebar.success("Parameters saved as default!")
        except Exception as e:
            st.sidebar.error(f"Error saving parameters: {str(e)}")

    return params
