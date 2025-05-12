# app/components/pipeline_tabs.py
import streamlit as st
import pandas as pd
from app.utils.image_display import safe_display_image
from app.components.technical_details import render_technical_details, show_cv2_function_details


def display_tabbed_pipeline(pipeline_results):
    """
    Display the complete processing pipeline with each step as a tab.

    Args:
            pipeline_results: Dictionary containing pipeline results
    """
    if not pipeline_results or 'steps' not in pipeline_results or not pipeline_results['steps']:
        st.warning("No pipeline steps available to display.")
        return

    # Create tabs for each step in the pipeline plus a results tab
    step_names = [f"{i+1}. {step['name']}" for i,
                  step in enumerate(pipeline_results['steps'])]
    step_names.append("📊 Results")
    tabs = st.tabs(step_names)

    # Display each step in its corresponding tab
    for i, (tab, step) in enumerate(zip(tabs[:-1], pipeline_results['steps'])):
        with tab:
            _display_pipeline_step(step, i)

    # Display results in the last tab
    with tabs[-1]:
        _display_pipeline_results(pipeline_results)


def _display_pipeline_step(step, step_index):
    """
    Display a single pipeline step with technical details.

    Args:
            step: Dictionary containing step information
            step_index: Index of the step in the pipeline
    """
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

    # Show success/failure indicator
    if 'success' in step:
        if step['success']:
            st.success("✅ Step completed successfully")
        else:
            st.error("❌ Step failed or produced no results")

    # Technical details section - using collapsible sections without expanders
    st.write("---")
    st.subheader("🔍 Technical Details")

    # Use checkbox as a collapsible section instead of expander
    show_tech_details = st.checkbox(f"Show Technical Details for {step['name']}",
                                    key=f"tech_details_{step_index}", value=False)

    if show_tech_details:
        # Render technical details without using expanders
        _render_step_technical_details(step['name'])

        # Display function details if available
        if 'functions' in step and step['functions']:
            st.write("**Key Functions:**")
            for func in step['functions']:
                func_col1, func_col2 = st.columns([1, 3])
                with func_col1:
                    if st.button(f"Details", key=f"func_{func}_{step_index}"):
                        with func_col2:
                            _show_function_details_no_expander(func)
                with func_col2:
                    st.code(func, language="python")


def _display_pipeline_results(pipeline_results):
    """
    Display the final results of the pipeline.

    Args:
            pipeline_results: Dictionary containing pipeline results
    """
    col1, col2 = st.columns(2)

    with col1:
        # Display original image
        if 'original_image' in pipeline_results:
            st.subheader("Original Image")
            safe_display_image(pipeline_results['original_image'])

    with col2:
        # Display final processed image (from last step)
        if 'steps' in pipeline_results and pipeline_results['steps']:
            last_step = pipeline_results['steps'][-1]
            if 'output_image' in last_step:
                st.subheader("Final Result")
                safe_display_image(last_step['output_image'])

    # Display QR code data if available
    if 'qr_data' in pipeline_results and pipeline_results['qr_data']:
        st.success(f"QR Code: {pipeline_results['qr_data']}")

    # Display detected answers in a table
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

        # Use checkbox instead of expander for CSV details
        show_csv_details = st.checkbox(
            "🔍 Show CSV Generation Details", value=False)
        if show_csv_details:
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
    else:
        st.warning("No answers detected in the image.")


def _render_step_technical_details(step_name):
    """
    Renders technical details without expanders.

    Args:
            step_name: Name of the pipeline step
    """
    details = {
        "QR Code Detection": {
            "description": """
			Detects and decodes QR codes using multiple image processing techniques:
			1. Grayscale conversion for efficient processing
			2. Multiple detection methods (adaptive threshold, blur, histogram equalization)
			3. Decoding of QR content using the pyzbar library
			""",
            "functions": [
                "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                "pyzbar.decode(image)",
                "cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)",
                "cv2.GaussianBlur(gray, (5, 5), 0)",
                "cv2.equalizeHist(gray)",
                "cv2.polylines(image, [points], True, color)"
            ]
        },
        "Grayscale Conversion": {
            "description": """
			Converts the BGR color image to grayscale (single-channel) using a weighted formula:
			Y = 0.299R + 0.587G + 0.114B
			
			This reduces processing requirements and simplifies subsequent steps by focusing
			on intensity variations rather than color differences.
			""",
            "functions": [
                "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)"
            ]
        },
        "Circle Detection": {
            "description": """
			Uses the Hough Circle Transform to detect circular bubbles in the answer sheet.
			The algorithm works by:
			1. Creating a 3D accumulator space (x, y, radius)
			2. Identifying edge pixels using gradient information
			3. Voting for potential circles in the accumulator space
			4. Finding local maxima that correspond to actual circles
			""",
            "functions": [
                "cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=18, minRadius=10, maxRadius=20)",
                "cv2.circle(image, (x, y), r, color, thickness)",
                "cv2.line(image, (x1, y1), (x2, y2), color, thickness)"
            ]
        },
        "Bubble Analysis": {
            "description": """
			Applies adaptive thresholding to separate filled bubbles from empty ones.
			This technique:
			1. Calculates different thresholds for different regions of the image
			2. Accounts for varying lighting conditions across the sheet
			3. Creates a binary image where filled bubbles appear as white pixels
			""",
            "functions": [
                "cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)",
                "cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)"
            ]
        },
        "Answer Extraction": {
            "description": """
			Analyzes the filled bubbles to determine selected answers:
			1. Creates masks for each detected bubble
			2. Counts filled pixels in each bubble using bitwise operations
			3. Applies score multiplier to get final bubble scores
			4. Selects highest-scoring bubble above threshold as the answer
			5. Groups into questions and generates answer key
			""",
            "functions": [
                "cv2.bitwise_and(thresh, bubble_mask)",
                "cv2.countNonZero(result)",
                "cv2.circle(image, (x, y), r, color, thickness)",
                "cv2.putText(image, text, position, font, scale, color, thickness)",
                "cv2.rectangle(image, pt1, pt2, color, thickness)",
                "cv2.addWeighted(overlay, alpha, image, beta, gamma)"
            ]
        }
    }

    if step_name not in details:
        return

    step_details = details[step_name]
    st.write(step_details["description"])

    st.write("**Key Functions:**")
    for func in step_details["functions"]:
        st.code(func, language="python")


def _show_function_details_no_expander(func_name):
    """
    Shows detailed information about OpenCV functions without using expanders.

    Args:
            func_name: Name of the OpenCV function
    """
    functions = {
        "cv2.cvtColor": {
            "purpose": "Converts an image from one color space to another",
            "common_conversions": [
                "BGR2GRAY - Color to grayscale",
                "BGR2HSV - Color to hue-saturation-value",
                "GRAY2BGR - Grayscale to color (all channels equal)"
            ],
            "performance": "Fast operation, typically O(n) where n is pixel count",
            "usage": "cv2.cvtColor(src, code) → dst"
        },
        "cv2.HoughCircles": {
            "purpose": "Detects circles in a grayscale image using Hough transform",
            "parameters": [
                "dp - Accumulator resolution inverse ratio",
                "minDist - Minimum distance between circle centers",
                "param1 - Upper Canny edge threshold",
                "param2 - Accumulator threshold",
                "minRadius/maxRadius - Size constraints"
            ],
            "complexity": "Computationally intensive O(n³) where n is image dimension",
            "usage": "cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius) → circles"
        },
        "cv2.adaptiveThreshold": {
            "purpose": "Applies adaptive thresholding to grayscale image",
            "methods": [
                "ADAPTIVE_THRESH_MEAN_C - Mean of neighborhood",
                "ADAPTIVE_THRESH_GAUSSIAN_C - Weighted mean using Gaussian window"
            ],
            "advantages": "Handles varying illumination across image",
            "usage": "cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C) → dst"
        },
        "cv2.bitwise_and": {
            "purpose": "Performs bitwise AND operation between two images",
            "application": "Commonly used for masking operations",
            "usage": "cv2.bitwise_and(src1, src2, mask=None) → dst"
        },
        "cv2.countNonZero": {
            "purpose": "Counts non-zero elements in an array",
            "application": "Often used to measure area of detected regions",
            "performance": "Very efficient O(n) operation",
            "usage": "cv2.countNonZero(src) → count"
        }
    }

    # Extract function name without parameters
    base_func_name = func_name.split('(')[0]

    if base_func_name in functions:
        details = functions[base_func_name]

        st.markdown(f"### {base_func_name}")
        st.write(f"**Purpose:** {details['purpose']}")

        for key, value in details.items():
            if key != 'purpose':
                if isinstance(value, list):
                    st.write(f"**{key.title()}:**")
                    for item in value:
                        st.write(f"- {item}")
                else:
                    st.write(f"**{key.title()}:** {value}")
    else:
        st.info(f"Details for {base_func_name} not available")
