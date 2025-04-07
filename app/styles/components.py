"""
UI Components Module

Provides reusable UI components with consistent styling for the 
E-commerce Recommendation System application.
"""

import streamlit as st
from .theme import THEME, COLORS, TYPOGRAPHY, SPACING, EFFECTS

def apply_custom_css():
    """Apply custom CSS for professional monochromatic styling."""
    custom_css = """
    <style>
    /* Base styles following design philosophy */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #212121;
        background-color: #FFFFFF;
    }
    
    /* Typography with clear hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: #000000;
        font-weight: 600;
    }
    
    /* Button styling with minimalist aesthetic */
    .stButton > button {
        background-color: #000000;
        color: white;
        border-radius: 4px;
        font-weight: 400;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #616161;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Metric styling with professional look */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 600;
        color: #000000;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: #F5F5F5;
        color: #000000;
        font-weight: 600;
        border: none !important;
        text-align: left;
    }
    
    .dataframe td {
        border: none !important;
        border-bottom: 1px solid #E0E0E0 !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    
    /* Widget labels */
    .stSelectbox label, .stSlider label {
        color: #616161;
        font-weight: 400;
        font-size: 0.9rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def create_card(title, content, padding=SPACING['md'], background=COLORS['light_accent']):
    """
    Create a clean, professional card component.
    
    Parameters:
    -----------
    title : str
        Card title
    content : str
        HTML content to display in the card
    padding : str, optional
        CSS padding value
    background : str, optional
        Background color
        
    Returns:
    --------
    str
        HTML for the card component
    """
    card_html = f"""
    <div style="
        background-color: {background};
        padding: {padding};
        border-radius: {EFFECTS['border_radius']};
        margin-bottom: {SPACING['md']};
        box-shadow: {EFFECTS['shadow_light']};
    ">
        <h3 style="
            margin-top: 0;
            color: {COLORS['highlight']};
            font-weight: {TYPOGRAPHY['heading_weight']};
            font-size: {TYPOGRAPHY['subheading_size']};
        ">{title}</h3>
        <div>{content}</div>
    </div>
    """
    return card_html

def section_header(title, description=None):
    """
    Display a consistent section header.
    
    Parameters:
    -----------
    title : str
        Section title
    description : str, optional
        Section description
    """
    st.markdown(f"## {title}")
    if description:
        st.markdown(f"<p style='color:{COLORS['accent']};'>{description}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0; border: none; height: 1px; background-color: #e0e0e0;'>", unsafe_allow_html=True)

def metric_card(title, value, delta=None, prefix="", suffix=""):
    """
    Display a metric with consistent styling.
    
    Parameters:
    -----------
    title : str
        Metric title
    value : str or numeric
        Metric value
    delta : str or numeric, optional
        Delta value
    prefix : str, optional
        Prefix for the value (e.g., "$")
    suffix : str, optional
        Suffix for the value (e.g., "%")
    """
    formatted_value = f"{prefix}{value}{suffix}"
    if delta is not None:
        st.metric(title, formatted_value, delta)
    else:
        st.metric(title, formatted_value)

def info_box(message, box_type="info"):
    """
    Display a styled information box.
    
    Parameters:
    -----------
    message : str
        Message to display
    box_type : str, optional
        Type of box: 'info', 'success', 'warning', or 'error'
    """
    if box_type == "info":
        bgcolor = "#E3F2FD"
        bordercolor = "#2196F3"
        icon = "ℹ️"
    elif box_type == "success":
        bgcolor = "#E8F5E9"
        bordercolor = COLORS['success']
        icon = "✅"
    elif box_type == "warning":
        bgcolor = "#FFF8E1"
        bordercolor = "#FFC107"
        icon = "⚠️"
    elif box_type == "error":
        bgcolor = "#FFEBEE"
        bordercolor = COLORS['error']
        icon = "❌"
    else:
        bgcolor = COLORS['light_accent']
        bordercolor = COLORS['accent']
        icon = "ℹ️"
        
    st.markdown(
        f"""
        <div style="
            background-color: {bgcolor};
            border-left: 4px solid {bordercolor};
            padding: {SPACING['md']};
            border-radius: {EFFECTS['border_radius']};
            margin-bottom: {SPACING['md']};
        ">
            <p style="margin: 0; display: flex; align-items: center;">
                <span style="margin-right: 8px;">{icon}</span>
                {message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def progress_steps(steps, current_step):
    """
    Display a progress indicator for multi-step processes.
    
    Parameters:
    -----------
    steps : list
        List of step names
    current_step : int
        Current step index (0-based)
    """
    num_steps = len(steps)
    
    step_html = f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: {SPACING['lg']};">
    """
    
    for i, step in enumerate(steps):
        # Determine color based on current step
        if i < current_step:
            color = COLORS['success']
            text_color = COLORS['success']
            weight = TYPOGRAPHY['heading_weight']
        elif i == current_step:
            color = COLORS['highlight']
            text_color = COLORS['highlight']
            weight = TYPOGRAPHY['heading_weight']
        else:
            color = COLORS['light_accent']
            text_color = COLORS['accent']
            weight = TYPOGRAPHY['body_weight']
            
        # Calculate width to create space between steps
        width = f"calc({100/num_steps}% - {SPACING['md']})"
        
        step_html += f"""
        <div style="width: {width}; text-align: center;">
            <div style="
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background-color: {color};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto {SPACING['xs']} auto;
                font-weight: {TYPOGRAPHY['heading_weight']};
            ">{i+1}</div>
            <div style="
                font-size: 0.9rem;
                color: {text_color};
                font-weight: {weight};
            ">{step}</div>
        </div>
        """
    
    step_html += "</div>"
    
    st.markdown(step_html, unsafe_allow_html=True)

def divider(margin_y=SPACING['md']):
    """Add a subtle divider line."""
    st.markdown(
        f'<hr style="height:1px;border:none;background-color:{COLORS["light_accent"]};margin:{margin_y} 0;">',
        unsafe_allow_html=True
    ) 