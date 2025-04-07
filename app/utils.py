"""
Utility Functions

Common helper functions for the E-commerce Recommendation System.
"""

import streamlit as st
from styles import apply_custom_css, COLORS, THEME

def setup_page(title, icon=None):
    """Configure Streamlit page settings with professional monochromatic styling."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS from styles module
    apply_custom_css()

def init_session_state():
    """Initialize session state variables."""
    # Data storage
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    
    if 'rfm_data' not in st.session_state:
        st.session_state.rfm_data = None
    
    if 'normalized_data' not in st.session_state:
        st.session_state.normalized_data = None
    
    # Analysis parameters
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 3
    
    if 'ref_date' not in st.session_state:
        st.session_state.ref_date = None
    
    # Processing flags
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    
    # Model storage
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    
    if 'cluster_model' not in st.session_state:
        st.session_state.cluster_model = None
    
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = None
    
    if 'cluster_profiles' not in st.session_state:
        st.session_state.cluster_profiles = None
    
    if 'customer_clusters' not in st.session_state:
        st.session_state.customer_clusters = None
    
    # Application state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

def get_color_palette():
    """Get monochromatic color palette for the application."""
    return COLORS

def get_cluster_colors():
    """Get grayscale colors for clusters."""
    return [
        '#000000',  # Black
        '#333333',  # Dark gray
        '#666666',  # Medium gray
        '#999999',  # Light gray
        '#CCCCCC',  # Very light gray
        '#444444',  # Gray 1
        '#555555',  # Gray 2
        '#777777',  # Gray 3
    ]

def get_monochromatic_color_sequence(n_colors):
    """
    Generate a monochromatic sequence of grayscale colors.
    
    Parameters:
    -----------
    n_colors : int
        Number of colors to generate
    
    Returns:
    --------
    list
        List of hex color codes
    """
    # Start with black, end with light gray
    start = 0  # Black
    end = 150  # Medium-light gray (not too light)
    
    # Generate evenly spaced colors in grayscale
    step = (end - start) / max(1, (n_colors - 1))
    colors = []
    
    for i in range(n_colors):
        val = int(start + i * step)
        hex_color = f'#{val:02x}{val:02x}{val:02x}'
        colors.append(hex_color)
    
    return colors

def display_error(message):
    """Display an error message with professional styling."""
    st.error(f"Error: {message}")

def display_success(message):
    """Display a success message with professional styling."""
    st.success(f"Success: {message}")

def format_currency(value):
    """Format a value as currency with professional styling."""
    return f"${value:,.2f}"

def format_number(value):
    """Format a numeric value with thousand separators."""
    return f"{value:,}"

def display_workflow_steps(steps, current_step):
    """Display a professional workflow progress indicator."""
    from styles import progress_steps
    progress_steps(steps, current_step)

def display_section_header(title, description=None):
    """Display a consistent section header with professional styling."""
    from styles import section_header
    section_header(title, description)

def display_metric_card(title, value, delta=None, prefix="", suffix=""):
    """Display a metric with consistent professional styling."""
    from styles import metric_card
    metric_card(title, value, delta, prefix, suffix)

def create_info_card(title, content):
    """Create a professional info card."""
    from styles import create_card
    return create_card(title, content)

def add_divider():
    """Add a subtle divider between sections."""
    from styles import divider
    divider() 