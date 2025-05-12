# app/pages/settings_page.py
import streamlit as st
import json
import os


def show_settings_page():
    """
    Displays the application settings page.

    Allows configuring global application settings.
    """
    st.header("Application Settings")
    st.write("Configure global settings for the application.")

    # Application information
    st.subheader("About iChecker")
    st.write("""
	iChecker is an Answer Sheet Analysis tool that allows you to:
	- Generate standard answer sheets with custom parameters
	- Add random or specific shading for testing purposes
	- Analyze scanned answer sheets with computer vision
	- Extract answers and QR code information
	""")

    # Display current configuration
    st.subheader("Current Configuration")

    try:
        # Get current configuration file if it exists
        if os.path.exists("app_config.json"):
            with open("app_config.json", "r") as f:
                config = json.load(f)
                st.json(config)
        else:
            st.info("No saved configuration found. Using default settings.")
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")

    # Add option to reset to defaults
    if st.button("Reset All Settings to Default"):
        try:
            from app.utils.config import DEFAULT_PARAMS

            with open("app_config.json", "w") as f:
                json.dump(DEFAULT_PARAMS, f)

            st.success("All settings reset to default values!")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting settings: {str(e)}")
