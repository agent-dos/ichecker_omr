# app/ui/components/generator_tabs.py
"""Tab components for generator page."""
import streamlit as st


def render_generator_tabs() -> str:
    """Render generator tab selection."""
    tab_options = {
        "single": "Single Sheet",
        "multiple": "Multiple Sheets",
        "batch": "Exam Batch",
        "custom": "Custom Template"
    }

    # Create tabs
    tabs = st.tabs(list(tab_options.values()))

    # Return selected tab key
    for key, (tab, label) in enumerate(zip(tabs, tab_options.keys())):
        if tab:
            return label

    return "single"  # Default
