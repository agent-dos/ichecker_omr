# app/ui/pages/settings.py
"""
Settings page UI.
"""
import streamlit as st
import json
import os

from app.core.config import DEFAULT_PARAMS


def show_settings_page():
    """
    Display settings page.
    """
    st.header("Application Settings")

    # About section
    st.subheader("About iChecker")
    st.write("""
    iChecker is an Answer Sheet Analysis tool that:
    - Generates standard answer sheets
    - Analyzes scanned sheets with computer vision
    - Extracts answers and QR codes
    - Provides detailed calibration options
    """)

    # Configuration section
    st.subheader("Current Configuration")

    config_file = "app_config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
            st.json(config)
    else:
        st.info("Using default configuration.")

    # Reset button
    if st.button("Reset to Defaults"):
        try:
            with open(config_file, "w") as f:
                json.dump(DEFAULT_PARAMS, f, indent=2)
            st.success("Settings reset to defaults!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
