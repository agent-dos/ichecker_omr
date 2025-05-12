# app/components/pipeline_visualization.py
import streamlit as st
import pandas as pd
from app.utils.image_display import safe_display_image
from app.components.technical_details import render_technical_details, show_cv2_function_details


def display_pipeline_steps(pipeline_results):
    """
    Display the complete processing pipeline with all steps.

    Args:
            pipeline_results: Dictionary containing pipeline results
    """
    # Display each step in the pipeline
    for i, step in enumerate(pipeline_results['steps']):
        st.subheader(f"Step {i+1}: {step['name']}")

        # Add technical details for this step
        render_technical_details(step['name'])

        # Create two columns for input and output
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Input**")
            if 'input_image' in step and step['input_image'] is not None:
                safe_display_image(step['input_image'])
            else:
                st.info("No input image available")

        with col2:
            st.write("**Output**")
            if 'output_image' in step and step['output_image'] is not None:
                safe_display_image(step['output_image'])
            else:
                st.info("No output image available")

        # Display step description
        if 'description' in step:
            st.write(step['description'])

        # Display additional step information
        if 'stats' in step:
            stats = step['stats']
            stats_cols = st.columns(len(stats))
            for i, (key, value) in enumerate(stats.items()):
                with stats_cols[i]:
                    st.metric(key.replace('_', ' ').title(), value)

        # Display function details if available
        if 'functions' in step and step['functions']:
            st.write("**Key Functions:**")
            for func in step['functions']:
                # Create a button to show function details
                if st.button(f"ℹ️ {func}", key=f"func_{func}_{i}"):
                    show_cv2_function_details(func)

        # Show success/failure indicator
        if 'success' in step:
            if step['success']:
                st.success("✅ Step completed successfully")
            else:
                st.error("❌ Step failed or produced no results")

        # Add a divider between steps
        if i < len(pipeline_results['steps']) - 1:
            st.divider()

    # Display final results
    if 'answers' in pipeline_results and pipeline_results['answers']:
        st.subheader("Extracted Answers")

        # Convert answers to DataFrame for display
        answers_df = pd.DataFrame(pipeline_results['answers'], columns=[
                                  "Question", "Answer"])
        st.dataframe(answers_df, use_container_width=True)

        # Add download button for answers CSV
        csv = answers_df.to_csv(index=False)
        st.download_button(
            "Download Answers as CSV",
            csv,
            "detected_answers.csv",
            "text/csv",
            key="download-csv"
        )

        # Display CSV generation details
        with st.expander("🔍 CSV Generation Details", expanded=False):
            st.markdown("""
			**Functions Used:**
			- `pandas.DataFrame()`: Converts the answer list to a structured DataFrame
			- `pandas.DataFrame.to_csv()`: Serializes the DataFrame to CSV format
			
			**Parameters:**
			- `index=False`: Excludes row indices from the output
			
			**Format:**
			The CSV contains two columns:
			- `Question`: The question number (1-60)
			- `Answer`: The selected option (A-E) or empty string if no answer detected
			""")

        # Display QR code data if available
        if 'qr_data' in pipeline_results and pipeline_results['qr_data']:
            st.success(f"QR Code: {pipeline_results['qr_data']}")
    else:
        st.warning("No answers detected in the image.")
