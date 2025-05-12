# app/utils/debug_helper.py
import cv2
from app.utils.circle_detection_debug import visualize_detection_stages


def run_detection_debug(image, params, qr_polygon=None, answer_boundary=None):
    """
    Run detection debug and display results.

    Args:
        image: Input image
        params: Detection parameters
        qr_polygon: QR polygon for exclusion
        answer_boundary: Corner boundary

    Returns:
        dict: Debug visualization results
    """
    debug_results = visualize_detection_stages(
        image, params, qr_polygon, answer_boundary
    )

    # Log statistics
    for stage, data in debug_results.items():
        if 'stats' in data:
            print(f"\nStage: {stage}")
            print(f"  Circles: {data['stats']['count']}")
            print(f"  Left: {data['stats']['left']}")
            print(f"  Right: {data['stats']['right']}")

    return debug_results


def save_debug_visualizations(debug_results, output_dir="debug_output"):
    """Save debug visualizations to files."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    for stage, data in debug_results.items():
        if 'image' in data:
            filename = f"{output_dir}/{stage}.png"
            cv2.imwrite(filename, data['image'])
            print(f"Saved: {filename}")
