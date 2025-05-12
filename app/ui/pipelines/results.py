# app/ui/pipelines/results.py
"""
Pipeline results display component.
"""
import streamlit as st
import pandas as pd
from typing import Dict

from app.common.image.display import display_image


def display_pipeline_results(results: Dict) -> None:
    """
    Display analysis pipeline results.
    """
    if not results or 'steps' not in results:
        st.warning("No results to display.")
        return

    # Create tabs for each step
    steps = results['steps']
    step_names = [f"{i+1}. {step['name']}" for i, step in enumerate(steps)]
    step_names.append("📊 Results")

    tabs = st.tabs(step_names)

    # Display each step
    for i, (tab, step) in enumerate(zip(tabs[:-1], steps)):
        with tab:
            _display_step(step, i)

    # Display final results
    with tabs[-1]:
        _display_final_results(results)


def _display_step(step: Dict, index: int) -> None:
    """
    Display a single pipeline step.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Input**")
        if 'input_image' in step:
            display_image(step['input_image'])

    with col2:
        st.write("**Output**")
        if 'output_image' in step:
            display_image(step['output_image'])

    if 'description' in step:
        st.write(step['description'])

    if 'success' in step:
        if step['success']:
            st.success("✅ Step completed successfully")
        else:
            st.error("❌ Step failed")


def _display_final_results(results: Dict) -> None:
    """
    Display final analysis results.
    """
    col1, col2 = st.columns(2)

    with col1:
        if 'original_image' in results:
            st.subheader("Original Image")
            display_image(results['original_image'])

    with col2:
        if 'steps' in results and results['steps']:
            last_step = results['steps'][-1]
            if 'output_image' in last_step:
                st.subheader("Final Result")
                display_image(last_step['output_image'])

    # Display QR data
    if 'qr_data' in results and results['qr_data']:
        st.success(f"QR Code: {results['qr_data']}")

    # Display answers
    if 'answers' in results and results['answers']:
        st.subheader("Extracted Answers")

        # Convert to DataFrame
        df = pd.DataFrame(results['answers'], columns=["Question", "Answer"])
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "answers.csv",
            "text/csv"
        )
