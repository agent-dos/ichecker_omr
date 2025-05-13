# filename: app/ui/components/sidebar.py
import streamlit as st
from typing import Dict, List, Tuple, Any
import cv2  # For default flags if needed
import logging  # Import logging

# Import config functions and constants maps
# Import the config_manager instance itself to call its methods
from app.core.config import (
    config_manager, get_config, get_cv2_flag,
    CV2_INTERPOLATION_FLAGS, CV2_MORPH_OPS, CV2_ADAPTIVE_METHODS, CV2_THRESH_TYPES, save_config
)
# DEFAULT_PARAMS might not be needed directly here anymore if reset uses manager
# from app.core.constants import DEFAULT_PARAMS

logger = logging.getLogger(__name__)


# Helper to create selectbox from dict keys
def _selectbox_from_keys(label: str, options_dict: Dict, current_key: str, help_text: str = "") -> str:
    options = list(options_dict.keys())
    if not options:  # Handle empty dictionary case
        st.error(f"No options available for '{label}'.")
        return ""

    try:
        # Handle cases where current_key might not be in options (e.g., after config change)
        if current_key not in options:
            logger.warning(
                f"Invalid current value '{current_key}' for {label}. Using first option '{options[0]}'.")
            current_index = 0
        else:
            current_index = options.index(current_key)
    except ValueError:  # Should not happen if key is in list, but as safeguard
        logger.warning(
            f"Error finding index for '{current_key}' in {label}. Using first option.")
        current_index = 0
    # Handles empty options_dict after check (shouldn't occur)
    except IndexError:
        logger.error(f"Index error for {label} despite options check.")
        return ""

    # Use a unique key for the selectbox based on the label
    widget_key = f"selectbox_{label.lower().replace(' ', '_')}"
    return st.selectbox(label, options, index=current_index, help=help_text, key=widget_key)


def render_analyzer_sidebar() -> Dict:
    """
    Render analyzer sidebar with detailed parameters.
    Returns the current configuration dictionary held by the UI components.
    """
    st.sidebar.title("Analysis Settings")

    # Load current config at the start of rendering
    current_config = get_config()

    # --- Debug Options ---
    st.sidebar.subheader("Debug Options")
    debug_options = current_config.get('debug_options', {})
    debug_options['visualize_intermediate_steps'] = st.sidebar.checkbox(
        "Visualize Intermediate Steps",
        value=debug_options.get('visualize_intermediate_steps', False),
        help="Show detailed visualizations for each processing step (slows processing)."
    )
    current_config['debug_options'] = debug_options

    # --- Main Parameters ---
    st.sidebar.subheader("Processing Parameters")

    # Analyzer Top-Level & Rectification
    with st.sidebar.expander("1. General & Rectification", expanded=False):
        analyzer_params = current_config.get('analyzer', {})
        rect_params = current_config.get('rectification', {})

        analyzer_params['enable_rectification'] = st.checkbox(
            "Enable Rectification", value=analyzer_params.get('enable_rectification', True))
        analyzer_params['rectification_threshold'] = st.slider(
            "Rectification Angle Threshold (°)", 0.0, 20.0,
            # Ensure float
            value=float(analyzer_params.get('rectification_threshold', 5.0)), step=0.5,
            help="Minimum detected angle to trigger rectification.")

        current_interp = rect_params.get('warp_interpolation', 'INTER_LINEAR')
        if current_interp not in CV2_INTERPOLATION_FLAGS:
            current_interp = 'INTER_LINEAR'  # Fallback
        rect_params['warp_interpolation'] = _selectbox_from_keys(
            "Warp Interpolation", CV2_INTERPOLATION_FLAGS, current_interp
        )
        rect_params['dst_margin'] = st.number_input(
            "Rectification Dest Margin", 0, 50,
            value=int(rect_params.get('dst_margin', 0)), step=1)  # Ensure int

        current_config['analyzer'] = analyzer_params
        current_config['rectification'] = rect_params

    # QR Detection
    with st.sidebar.expander("2. QR Code Detection", expanded=False):
        qr_params = current_config.get('qr_detection', {})
        st.caption("Preprocessing attempts before pyzbar:")
        qr_params['gaussian_blur_ksize'] = st.slider(
            "QR Blur Kernel Size", 1, 15,
            # Ensure int
            value=int(qr_params.get('gaussian_blur_ksize', 5)), step=2,
            help="Must be odd.")

        current_qr_adapt_method = qr_params.get(
            'adaptive_method', 'ADAPTIVE_THRESH_GAUSSIAN_C')
        if current_qr_adapt_method not in CV2_ADAPTIVE_METHODS:
            current_qr_adapt_method = 'ADAPTIVE_THRESH_GAUSSIAN_C'
        qr_params['adaptive_method'] = _selectbox_from_keys(
            "QR Adaptive Method", CV2_ADAPTIVE_METHODS, current_qr_adapt_method
        )
        qr_params['adaptive_blocksize'] = st.slider(
            "QR Adaptive Block Size", 3, 51,
            # Ensure int
            value=int(qr_params.get('adaptive_blocksize', 11)), step=2,
            help="Must be odd.")
        qr_params['adaptive_c'] = st.number_input(
            "QR Adaptive Constant C", -10, 10,
            value=int(qr_params.get('adaptive_c', 2)), step=1)  # Ensure int
        qr_params['equalize_hist'] = st.checkbox(
            "QR Use Histogram Equalization",
            value=bool(qr_params.get('equalize_hist', True)))  # Ensure bool

        current_config['qr_detection'] = qr_params

    # Corner Detection
    # Main expander for the whole section
    with st.sidebar.expander("3. Corner Detection", expanded=False):
        corner_params = current_config.get('corner_detection', {})

        # General parameters directly inside the main expander
        # Use bold text instead of subheader here if preferred
        st.write("**General Corner Params**")
        corner_params['min_area'] = st.number_input(
            "Corner Min Area", 50, 2000,
            value=int(corner_params.get('min_area', 300)), step=10, key="corner_min_area")
        corner_params['max_area'] = st.number_input(
            "Corner Max Area", 500, 10000,
            value=int(corner_params.get('max_area', 5000)), step=50, key="corner_max_area")
        corner_params['duplicate_threshold'] = st.slider(
            "Corner Duplicate Threshold (px)", 5, 100,
            value=int(corner_params.get('duplicate_threshold', 30)), step=1, key="corner_dup_thresh")

        # Use Subheaders for grouping strategies, scoring, validator
        # --- Strategy: Threshold ---
        st.subheader("Strategy: Threshold")  # Replaced expander
        strat_thresh = corner_params.get('strategy_threshold', {})
        strat_thresh['enabled'] = st.checkbox("Enable Threshold Strategy", value=bool(
            strat_thresh.get('enabled', True)), key="corner_thresh_enable")
        levels_list = strat_thresh.get('levels', [30, 50, 70, 90])
        if not isinstance(levels_list, list):
            levels_list = [30, 50, 70, 90]  # Fallback
        levels_str = st.text_input(
            "Thresh Levels (comma-sep)",
            value=",".join(map(str, levels_list)),
            key="corner_thresh_levels_input",  # Unique key
            help="Comma-separated threshold values.")
        try:
            parsed_levels = [int(x.strip())
                             for x in levels_str.split(',') if x.strip()]
            if not parsed_levels:
                raise ValueError("Input cannot be empty")
            strat_thresh['levels'] = parsed_levels
        except ValueError:
            st.error(
                "Invalid input for Threshold Levels. Please use comma-separated integers (e.g., 30,50,70).")
            strat_thresh['levels'] = levels_list  # Keep old value

        current_thresh_type = strat_thresh.get(
            'threshold_type', 'THRESH_BINARY_INV')
        if current_thresh_type not in CV2_THRESH_TYPES:
            current_thresh_type = 'THRESH_BINARY_INV'
        strat_thresh['threshold_type'] = _selectbox_from_keys(
            "Thresh Type", CV2_THRESH_TYPES, current_thresh_type)

        current_morph1 = strat_thresh.get('morph_op1', 'MORPH_CLOSE')
        if current_morph1 not in CV2_MORPH_OPS:
            current_morph1 = 'MORPH_CLOSE'
        strat_thresh['morph_op1'] = _selectbox_from_keys(
            "Thresh Morph Op 1", CV2_MORPH_OPS, current_morph1)

        current_morph2 = strat_thresh.get('morph_op2', 'MORPH_OPEN')
        if current_morph2 not in CV2_MORPH_OPS:
            current_morph2 = 'MORPH_OPEN'
        strat_thresh['morph_op2'] = _selectbox_from_keys(
            "Thresh Morph Op 2", CV2_MORPH_OPS, current_morph2)

        strat_thresh['morph_ksize'] = st.slider("Thresh Morph Kernel Size", 1, 11, value=int(
            strat_thresh.get('morph_ksize', 5)), step=2, key="corner_thresh_morph_ksize")
        strat_thresh['solidity_min'] = st.slider("Thresh Min Solidity", 0.5, 1.0, value=float(
            strat_thresh.get('solidity_min', 0.8)), step=0.01, key="corner_thresh_solidity")
        strat_thresh['aspect_ratio_min'] = st.slider("Thresh Min Aspect Ratio", 0.1, 2.0, value=float(
            strat_thresh.get('aspect_ratio_min', 0.7)), step=0.05, key="corner_thresh_ar_min")
        strat_thresh['aspect_ratio_max'] = st.slider("Thresh Max Aspect Ratio", 0.5, 5.0, value=float(
            strat_thresh.get('aspect_ratio_max', 1.3)), step=0.05, key="corner_thresh_ar_max")
        strat_thresh['fill_ratio_min'] = st.slider("Thresh Min Fill Ratio", 0.5, 1.0, value=float(
            strat_thresh.get('fill_ratio_min', 0.85)), step=0.01, key="corner_thresh_fill")
        corner_params['strategy_threshold'] = strat_thresh
        st.divider()  # Add visual separator

        # --- Strategy: Adaptive ---
        st.subheader("Strategy: Adaptive")  # Replaced expander
        strat_adapt = corner_params.get('strategy_adaptive', {})
        strat_adapt['enabled'] = st.checkbox("Enable Adaptive Strategy", value=bool(
            strat_adapt.get('enabled', True)), key="corner_adapt_enable")

        current_adapt_method = strat_adapt.get(
            'adaptive_method', 'ADAPTIVE_THRESH_MEAN_C')
        if current_adapt_method not in CV2_ADAPTIVE_METHODS:
            current_adapt_method = 'ADAPTIVE_THRESH_MEAN_C'
        strat_adapt['adaptive_method'] = _selectbox_from_keys(
            "Adaptive Method", CV2_ADAPTIVE_METHODS, current_adapt_method)

        current_adapt_thresh_type = strat_adapt.get(
            'threshold_type', 'THRESH_BINARY_INV')
        if current_adapt_thresh_type not in CV2_THRESH_TYPES:
            current_adapt_thresh_type = 'THRESH_BINARY_INV'
        strat_adapt['threshold_type'] = _selectbox_from_keys(
            "Adaptive Thresh Type", CV2_THRESH_TYPES, current_adapt_thresh_type)

        strat_adapt['blocksize'] = st.slider("Adaptive Block Size", 3, 61, value=int(
            strat_adapt.get('blocksize', 31)), step=2, key="corner_adapt_blocksize")
        strat_adapt['c'] = st.number_input("Adaptive Constant C", -15, 15, value=int(
            strat_adapt.get('c', 10)), step=1, key="corner_adapt_c")
        strat_adapt['aspect_ratio_min'] = st.slider("Adaptive Min Aspect Ratio", 0.1, 2.0, value=float(
            strat_adapt.get('aspect_ratio_min', 0.5)), step=0.05, key="corner_adapt_ar_min")
        strat_adapt['aspect_ratio_max'] = st.slider("Adaptive Max Aspect Ratio", 0.5, 5.0, value=float(
            strat_adapt.get('aspect_ratio_max', 2.0)), step=0.05, key="corner_adapt_ar_max")
        corner_params['strategy_adaptive'] = strat_adapt
        st.divider()

        # --- Strategy: Edge (Canny) ---
        st.subheader("Strategy: Edge (Canny)")  # Replaced expander
        strat_edge = corner_params.get('strategy_edge', {})
        strat_edge['enabled'] = st.checkbox("Enable Edge Strategy", value=bool(
            strat_edge.get('enabled', True)), key="corner_edge_enable")
        strat_edge['gaussian_blur_ksize'] = st.slider("Edge Blur Kernel Size", 1, 11, value=int(
            strat_edge.get('gaussian_blur_ksize', 5)), step=2, key="corner_edge_blur")
        strat_edge['canny_threshold1'] = st.slider("Edge Canny Thresh 1", 1, 200, value=int(
            strat_edge.get('canny_threshold1', 50)), step=1, key="corner_edge_t1")
        strat_edge['canny_threshold2'] = st.slider("Edge Canny Thresh 2", 1, 400, value=int(
            strat_edge.get('canny_threshold2', 150)), step=1, key="corner_edge_t2")
        corner_params['strategy_edge'] = strat_edge
        st.divider()

        # --- Corner Scoring ---
        st.subheader("Corner Scoring")  # Replaced expander
        scoring = corner_params.get('scoring', {})
        scoring['distance_weight'] = st.slider("Score Distance Weight", 0.0, 1.0, value=float(
            scoring.get('distance_weight', 0.5)), step=0.05, key="corner_score_dist")
        scoring['area_weight'] = st.slider("Score Area Weight", 0.0, 1.0, value=float(
            scoring.get('area_weight', 0.25)), step=0.05, key="corner_score_area")
        scoring['solidity_weight'] = st.slider("Score Solidity Weight", 0.0, 1.0, value=float(
            scoring.get('solidity_weight', 0.25)), step=0.05, key="corner_score_solid")
        scoring['area_norm_factor'] = st.number_input("Score Area Norm Factor", 100.0, 5000.0, value=float(
            scoring.get('area_norm_factor', 1000.0)), step=50.0, key="corner_score_norm")
        corner_params['scoring'] = scoring
        st.divider()

        # --- Corner Validator (QR Filter) ---
        st.subheader("Corner Validator (QR Filter)")  # Replaced expander
        validator = corner_params.get('validator', {})
        validator['qr_filter_enabled'] = st.checkbox("Enable QR Filter for Corners", value=bool(
            validator.get('qr_filter_enabled', True)), key="corner_qrfilt_enable")
        validator['qr_canny_threshold1'] = st.slider("QR Filter Canny Thresh 1", 1, 200, value=int(
            validator.get('qr_canny_threshold1', 50)), step=1, key="corner_qrfilt_t1")
        validator['qr_canny_threshold2'] = st.slider("QR Filter Canny Thresh 2", 1, 400, value=int(
            validator.get('qr_canny_threshold2', 150)), step=1, key="corner_qrfilt_t2")
        validator['qr_edge_ratio_threshold'] = st.slider("QR Filter Edge Ratio Thresh", 0.01, 0.5, value=float(
            validator.get('qr_edge_ratio_threshold', 0.15)), step=0.01, key="corner_qrfilt_edge")
        validator['qr_complexity_threshold'] = st.slider("QR Filter Complexity Thresh", 0.01, 1.0, value=float(
            validator.get('qr_complexity_threshold', 0.3)), step=0.01, key="corner_qrfilt_comp")
        corner_params['validator'] = validator

        # Update main corner config dict
        current_config['corner_detection'] = corner_params

    # Bubble Detection
    with st.sidebar.expander("4. Bubble Detection (Hough Circles)", expanded=False):
        bubble_params = current_config.get('bubble_detection', {})
        # Add keys to prevent duplicate widget errors if sections have same label
        bubble_params['gaussian_blur_ksize'] = st.slider(
            "Bubble Blur Kernel Size", 1, 11,
            value=int(bubble_params.get('gaussian_blur_ksize', 5)), step=2, key="bubble_blur_ksize")
        bubble_params['hough_dp'] = st.number_input(
            "Hough dp (Resolution)", 1.0, 5.0,
            value=float(bubble_params.get('hough_dp', 1.0)), step=0.1, key="bubble_hough_dp")
        bubble_params['hough_minDist'] = st.slider(
            "Hough Min Distance", 5, 50,
            value=int(bubble_params.get('hough_minDist', 20)), step=1, key="bubble_hough_minDist")
        bubble_params['hough_param1'] = st.slider(
            "Hough Param 1 (Canny Edge)", 10, 200,
            value=int(bubble_params.get('hough_param1', 50)), step=5, key="bubble_hough_p1")
        bubble_params['hough_param2'] = st.slider(
            "Hough Param 2 (Accumulator)", 5, 50,
            value=int(bubble_params.get('hough_param2', 18)), step=1, key="bubble_hough_p2")
        bubble_params['hough_minRadius'] = st.slider(
            "Hough Min Radius", 5, 30,
            value=int(bubble_params.get('hough_minRadius', 10)), step=1, key="bubble_hough_minR")
        bubble_params['hough_maxRadius'] = st.slider(
            "Hough Max Radius", 10, 50,
            value=int(bubble_params.get('hough_maxRadius', 20)), step=1, key="bubble_hough_maxR")

        st.subheader("Bubble Filtering")  # Use subheader here too
        bubble_params['filter_by_corners'] = st.checkbox(
            "Filter Bubbles by Corner Boundary",
            value=bool(bubble_params.get('filter_by_corners', True)), key="bubble_filt_corner")
        bubble_params['boundary_filter_margin'] = st.number_input(
            "Corner Boundary Margin (px)", -10, 50,
            value=int(bubble_params.get('boundary_filter_margin', 5)), step=1, key="bubble_filt_corner_margin")
        bubble_params['filter_by_qr'] = st.checkbox(
            "Filter Bubbles by QR Area",
            value=bool(bubble_params.get('filter_by_qr', True)), key="bubble_filt_qr")
        bubble_params['qr_filter_margin_factor'] = st.slider(
            "QR Filter Margin Factor (radius)", 0.0, 3.0,
            value=float(bubble_params.get('qr_filter_margin_factor', 1.0)), step=0.1, key="bubble_filt_qr_margin",
            help="Margin = radius + factor * radius")

        current_config['bubble_detection'] = bubble_params

    # Bubble Analysis
    with st.sidebar.expander("5. Bubble Analysis & Scoring", expanded=False):
        analysis_params = current_config.get('bubble_analysis', {})

        st.subheader("Thresholding for Scoring")  # Use subheader
        current_score_adapt_method = analysis_params.get(
            'adaptive_method', 'ADAPTIVE_THRESH_MEAN_C')
        if current_score_adapt_method not in CV2_ADAPTIVE_METHODS:
            current_score_adapt_method = 'ADAPTIVE_THRESH_MEAN_C'
        analysis_params['adaptive_method'] = _selectbox_from_keys(
            "Score Adaptive Method", CV2_ADAPTIVE_METHODS, current_score_adapt_method
        )
        analysis_params['adaptive_blocksize'] = st.slider(
            "Score Adaptive Block Size", 3, 61,
            value=int(analysis_params.get('adaptive_blocksize', 31)), step=2, key="analysis_adapt_block")
        analysis_params['adaptive_c'] = st.number_input(
            "Score Adaptive Constant C", -15, 15,
            value=int(analysis_params.get('adaptive_c', 10)), step=1, key="analysis_adapt_c")

        st.subheader("Grouping")  # Use subheader
        analysis_params['grouping_row_threshold'] = st.slider(
            "Grouping Row Threshold (px)", 1, 20,
            value=int(analysis_params.get('grouping_row_threshold', 8)), step=1, key="analysis_group_row")
        analysis_params['grouping_items_per_col'] = st.number_input(
            "Grouping Items per Column", 10, 50,
            value=int(analysis_params.get('grouping_items_per_col', 30)), step=1, key="analysis_group_items")

        st.subheader("Scoring")  # Use subheader
        analysis_params['scoring_inner_radius_factor'] = st.slider(
            "Scoring Inner Radius Factor", 0.1, 1.0,
            value=float(analysis_params.get('scoring_inner_radius_factor', 0.8)), step=0.05, key="analysis_score_radius")
        analysis_params['scoring_bubble_threshold'] = st.slider(
            "Scoring Fill Threshold (%)", 10.0, 90.0,
            value=float(analysis_params.get('scoring_bubble_threshold', 50.0)), step=1.0, key="analysis_score_thresh",
            help="Minimum normalized fill score to select bubble.")
        analysis_params['scoring_score_multiplier'] = st.slider(
            "Scoring Multiplier", 0.5, 5.0,
            value=float(analysis_params.get('scoring_score_multiplier', 2.0)), step=0.1, key="analysis_score_mult",
            help="Applied to normalized fill score.")

        current_config['bubble_analysis'] = analysis_params

    # --- Actions ---
    st.sidebar.subheader("Configuration Actions")
    col1, col2 = st.sidebar.columns(2)
    # Add keys to buttons to prevent issues if labels are identical elsewhere
    save_clicked = col1.button("Save Config", key="sidebar_save_button")
    reset_clicked = col2.button("Reset Defaults", key="sidebar_reset_button")

    if save_clicked:
        try:
            if save_config(current_config):  # Pass the modified config dict
                st.sidebar.success("Parameters saved!")
            else:
                st.sidebar.error("Error saving parameters.")
        except Exception as e:
            st.sidebar.error(f"Error saving config: {str(e)}")
            logger.exception("Error during config save.")  # Log full traceback

    if reset_clicked:
        try:
            config_manager.reset()  # Use the reset method on the singleton instance
            st.sidebar.success(
                "Parameters reset to defaults! Rerun triggered.")
            # Need to clear cache or session state related to config if any?
            st.rerun()  # Rerun to reflect defaults in UI and reload config
        except Exception as e:
            st.sidebar.error(f"Error resetting config: {str(e)}")
            # Log full traceback
            logger.exception("Error during config reset.")

    # Return the configuration dictionary reflecting the current UI state
    # This dictionary will be passed to AnalyzerService
    return current_config
