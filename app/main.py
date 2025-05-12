# app/main.py
"""
Main application entry point.
"""
import streamlit as st

from app.ui.pages.generator import show_generator_page
from app.ui.pages.analyzer import show_analyzer_page
from app.ui.pages.settings import show_settings_page


def main():
    """
    Main application function.
    """
    st.set_page_config(
        page_title="iChecker - Answer Sheet Analyzer",
        page_icon="📝",
        layout="wide",
    )

    st.title("📝 iChecker - Answer Sheet Analysis")

    # Create navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "🧪 Sheet Generator",
        "📊 Answer Sheet Analysis",
        "⚙️ Settings"
    ])

    with tab1:
        show_generator_page()

    with tab2:
        show_analyzer_page()

    with tab3:
        show_settings_page()


if __name__ == "__main__":
    main()
