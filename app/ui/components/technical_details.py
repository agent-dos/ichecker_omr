# app/ui/components/technical_details.py
"""Technical details display from original."""
import streamlit as st


def render_technical_details(step_name):
    """Renders technical details about OpenCV functions."""
    details = {
        "QR Code Detection": {
            "description": """
            Detects and decodes QR codes using multiple image processing techniques:
            1. Grayscale conversion for efficient processing
            2. Multiple detection methods
            3. Decoding of QR content using pyzbar
            """,
            "functions": [
                "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                "pyzbar.decode(image)",
                "cv2.adaptiveThreshold()",
                "cv2.GaussianBlur()",
                "cv2.equalizeHist()"
            ]
        },
        "Grayscale Conversion": {
            "description": """
            Converts the BGR color image to grayscale using weighted formula:
            Y = 0.299R + 0.587G + 0.114B
            """,
            "functions": [
                "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)"
            ]
        },
        "Circle Detection": {
            "description": """
            Uses Hough Circle Transform to detect circular bubbles.
            """,
            "functions": [
                "cv2.HoughCircles()",
                "cv2.circle()",
                "cv2.line()"
            ]
        }
    }

    if step_name in details:
        with st.expander("🔍 Technical Details", expanded=False):
            step_details = details[step_name]
            st.write(step_details["description"])
            st.write("**Key Functions:**")
            for func in step_details["functions"]:
                st.code(func, language="python")


def show_cv2_function_details(func_name):
    """Shows detailed information about OpenCV functions."""
    functions = {
        "cv2.cvtColor": {
            "purpose": "Converts an image from one color space to another",
            "common_conversions": [
                "BGR2GRAY - Color to grayscale",
                "BGR2HSV - Color to hue-saturation-value"
            ],
            "performance": "Fast operation, O(n)",
            "usage": "cv2.cvtColor(src, code) → dst"
        },
        "cv2.HoughCircles": {
            "purpose": "Detects circles in grayscale image",
            "parameters": [
                "dp - Accumulator resolution",
                "minDist - Minimum distance between circles",
                "param1 - Upper Canny edge threshold",
                "param2 - Accumulator threshold"
            ],
            "complexity": "O(n³)",
            "usage": "cv2.HoughCircles(...) → circles"
        }
    }

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
