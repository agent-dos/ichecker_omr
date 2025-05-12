# app.py
import streamlit as st
from app.pages.generator_page import show_generator_page
from app.pages.analyzer_page import show_analyzer_page
from app.pages.settings_page import show_settings_page


def main():
    """
    Main entry point for the iChecker Streamlit application.
    Sets up the page configuration and handles tab navigation.
    """
    # Configure page settings
    st.set_page_config(
        page_title="iChecker - Answer Sheet Analyzer",
        page_icon="📝",
        layout="wide",
    )

    # Display application title
    st.title("📝 iChecker - Answer Sheet Analysis")

    # Create main navigation tabs with Generator first
    tab1, tab2, tab3 = st.tabs([
        "🧪 Sheet Generator",
        "📊 Answer Sheet Analysis",
        "⚙️ Settings"
    ])

    # Load appropriate page based on the selected tab
    with tab1:
        show_generator_page()

    with tab2:
        show_analyzer_page()

    with tab3:
        show_settings_page()


if __name__ == "__main__":
    main()
