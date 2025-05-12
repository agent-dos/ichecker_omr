# filename: app/ui/pipelines/results.py
import streamlit as st
import pandas as pd
from typing import Dict, Any  # Added Any
import numpy as np

from app.common.image.display import display_image  # Assuming this handles BGR


def display_pipeline_results(results: Dict) -> None:
    """Display analysis pipeline results."""
    if not results or 'steps' not in results:
        st.warning("No results to display.")
        return

    # Get global debug flag from results if passed, or assume False
    # This assumes the flag is stored somewhere accessible, e.g., in the results dict itself
    # For simplicity, let's assume it's implicitly handled by presence of intermediate_visualizations
    # show_intermediate = results.get('debug_options', {}).get('visualize_intermediate_steps', False)

    # Create tabs for each step
    steps = results.get('steps', [])
    if not steps:
        st.warning("Analysis pipeline did not produce any steps.")
        # Display final results if available?
        _display_final_summary(results)
        return

    step_names = [
        f"{i+1}. {step.get('name', f'Step {i+1}')}" for i, step in enumerate(steps)]
    # Add a final summary tab
    step_names.append("📊 Summary")

    tabs = st.tabs(step_names)

    # Display each step
    for i, (tab, step) in enumerate(zip(tabs[:-1], steps)):
        with tab:
            _display_step(step, i)  # Pass step data

    # Display final summary in the last tab
    with tabs[-1]:
        _display_final_summary(results)


# app/ui/pipelines/results.py (updated _display_step function)
def _display_step(step: Dict, index: int) -> None:
    """Display a single pipeline step, including intermediate visualizations."""
    step_name = step.get('name', f'Step {index+1}')
    st.subheader(step_name)

    # Display status and description
    success = step.get('success', None)
    if success is True:
        st.success("✅ Step completed successfully")
    elif success is False:
        st.error("❌ Step failed or did not run as expected")
    else:
        st.info("ℹ️ Step status unknown or not applicable")

    if 'description' in step:
        st.write(step.get('description', 'No description available.'))

    # Check if this step has column images (special case for rectification)
    if 'column_images' in step and isinstance(step['column_images'], dict):
        column_images = step['column_images']
        column_labels = step.get('column_labels', {})

        # Create columns based on number of images
        cols = st.columns(len(column_images))

        # Display each image in its column
        for idx, (key, img) in enumerate(column_images.items()):
            with cols[idx]:
                if key in column_labels:
                    st.write(f"**{column_labels[key]}**")
                else:
                    st.write(f"**{key.title()}**")

                if isinstance(img, np.ndarray):
                    display_image(img, use_container_width=True)
                else:
                    st.warning(f"Invalid image format for {key}")
    else:
        # Standard two-column display for other steps
        col1, col2 = st.columns(2)
        with col1:
            if 'input_image' in step and isinstance(step['input_image'], np.ndarray):
                st.write("**Input to Step**")
                display_image(step['input_image'], use_container_width=True)
        with col2:
            if 'output_image' in step and isinstance(step['output_image'], np.ndarray):
                st.write("**Output / Visualization**")
                display_image(step['output_image'], use_container_width=True)

    # Display Intermediate Visualizations (if available and enabled)
    intermediate_viz = step.get('intermediate_visualizations')
    if intermediate_viz and isinstance(intermediate_viz, dict):
        st.markdown("---")  # Separator
        with st.expander("Show Intermediate Steps", expanded=False):
            st.write(f"Detailed visualizations for: **{step_name}**")
            # Sort keys for consistent order (optional)
            sorted_keys = sorted(intermediate_viz.keys())
            for viz_key in sorted_keys:
                viz_img = intermediate_viz[viz_key]
                if isinstance(viz_img, np.ndarray):
                    st.caption(f"`{viz_key}`")  # Display key as caption
                    display_image(viz_img, use_container_width=True)
                else:
                    st.warning(
                        f"Invalid intermediate visualization format for key '{viz_key}': {type(viz_img)}")


def _display_final_summary(results: Dict) -> None:
    """Display final analysis summary (QR, Answers)."""
    st.header("Analysis Summary")

    col1, col2 = st.columns(2)

    with col1:
        if 'original_image' in results:
            st.subheader("Original Image")
            display_image(results['original_image'])

    with col2:
        # Show the final processed image from the last step if possible
        final_processed_image = None
        if results.get('steps'):
            final_processed_image = results['steps'][-1].get('output_image')

        if final_processed_image is not None:
            st.subheader("Final Processed Image")
            display_image(final_processed_image)
        elif 'original_image' in results:  # Fallback if no steps/output
            st.subheader("Input Image (No final output available)")
            display_image(results['original_image'])

    # Display QR data
    st.markdown("---")
    st.subheader("Detected Information")
    qr_data = results.get('qr_data')
    if qr_data:
        st.success(f"**QR Code Data:** {qr_data}")
    else:
        st.warning("**QR Code:** Not Detected")

    # Display transform matrix if available
    transform = results.get('transform_matrix')
    if transform is not None:
        with st.expander("Rectification Transform Matrix"):
            st.code(str(transform))

    # Display answers
    st.markdown("---")
    answers = results.get('final_answers')  # Use the dedicated key
    if answers:
        st.subheader("Extracted Answers")
        # Convert to DataFrame for better display
        try:
            # Ensure answers are sorted? Assume AnswerExtractor sorts them.
            df = pd.DataFrame(answers, columns=["Question", "Answer"])
            # Replace empty answers with a placeholder for clarity
            df['Answer'] = df['Answer'].replace('', '-')
            st.dataframe(df, use_container_width=True, height=(
                min(len(df), 20) * 35 + 3))  # Dynamic height

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Answers (CSV)",
                data=csv,
                file_name="detected_answers.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not display answers table: {e}")
            st.write(answers)  # Display raw list as fallback
    else:
        st.warning("**Answers:** None Extracted")
