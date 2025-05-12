# app/components/technical_details.py
import streamlit as st


def render_technical_details(step_name):
    """
    Renders technical details about the OpenCV functions used in a specific pipeline step.

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

    with st.expander("🔍 Technical Details", expanded=False):
        step_details = details[step_name]
        st.write(step_details["description"])

        st.write("**Key Functions:**")
        for func in step_details["functions"]:
            st.code(func, language="python")


def show_cv2_function_details(func_name):
    """
    Shows detailed information about a specific OpenCV function.

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
