# app/ui/components/sidebar.py
"""
Sidebar components for UI pages.
"""
import streamlit as st
from typing import Dict

from app.core.config import save_parameters


def render_analyzer_sidebar(default_params: Dict) -> Dict:
    """
    Render analyzer sidebar with parameters.
    """
    params = default_params.copy()

    st.sidebar.title("Analysis Settings")

    with st.sidebar.expander("🎯 Calibration Tips", expanded=True):
        st.write("""
        **If bubbles aren't detected:**
        - Decrease 'Bubble Fill Threshold'
        - Increase 'Score Multiplier'
        
        **If empty bubbles are detected:**
        - Increase 'Bubble Fill Threshold'
        - Decrease 'Score Multiplier'
        """)

    # Circle Detection
    with st.sidebar.expander("Circle Detection"):
        params['param1'] = st.slider(
            "Edge Detection",
            10, 100, default_params['param1']
        )
        params['param2'] = st.slider(
            "Circle Threshold",
            5, 50, default_params['param2']
        )
        params['min_radius'] = st.slider(
            "Min Radius",
            5, 30, default_params['min_radius']
        )
        params['max_radius'] = st.slider(
            "Max Radius",
            10, 50, default_params['max_radius']
        )

    # Bubble Analysis
    with st.sidebar.expander("Bubble Analysis", expanded=True):
        params['bubble_threshold'] = st.slider(
            "Fill Threshold",
            10.0, 300.0, default_params['bubble_threshold'], 5.0
        )
        params['score_multiplier'] = st.slider(
            "Score Multiplier",
            0.5, 3.0, default_params['score_multiplier'], 0.1
        )

    # Presets
    st.sidebar.subheader("Quick Presets")
    if st.sidebar.button("High Sensitivity"):
        params['bubble_threshold'] = 50.0
        params['score_multiplier'] = 2.0

    if st.sidebar.button("Low Sensitivity"):
        params['bubble_threshold'] = 150.0
        params['score_multiplier'] = 1.0

    # Save parameters
    if st.sidebar.button("Save as Default"):
        try:
            save_parameters(params)
            st.sidebar.success("Parameters saved!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

    return params
