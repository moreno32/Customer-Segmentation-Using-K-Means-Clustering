"""
Theme Configuration Module

Defines the visual styling for the E-commerce Recommendation System application.
This module provides a consistent visual language through color palette,
typography, and spacing configurations.
"""

# Primary color palette - monochromatic scheme with grayscale exactly as in application_flow.md
COLORS = {
    'background': '#FFFFFF',     # Pure white background
    'text': '#212121',           # Near-black for text
    'accent': '#616161',         # Medium gray for accents
    'light_accent': '#F5F5F5',   # Very light gray for panels/cards
    'highlight': '#000000',      # Pure black for highlights and important elements
    'error': '#B00020',          # Dark red (used sparingly)
    'success': '#2E7D32',        # Dark green (used sparingly)
}

# Typography configuration
TYPOGRAPHY = {
    'font_family': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'heading_weight': '600',
    'body_weight': '400',
    'caption_weight': '300',
    'heading_size': '1.5rem',
    'subheading_size': '1.25rem',
    'body_size': '1rem',
    'caption_size': '0.875rem',
    'line_height': '1.5',
}

# Spacing scale for consistent layout
SPACING = {
    'xs': '0.25rem',
    'sm': '0.5rem',
    'md': '1rem',
    'lg': '1.5rem',
    'xl': '2rem',
    'xxl': '3rem',
}

# Border and shadow styling
EFFECTS = {
    'border_radius': '4px',
    'border_color': '#E0E0E0',
    'border_width': '1px',
    'shadow_light': '0 1px 3px rgba(0,0,0,0.1)',
    'shadow_medium': '0 4px 6px rgba(0,0,0,0.1)',
}

# Animation timing
ANIMATION = {
    'transition_speed': '0.2s',
    'transition_easing': 'ease-in-out',
}

# Combined theme object for easy import
THEME = {
    'colors': COLORS,
    'typography': TYPOGRAPHY,
    'spacing': SPACING,
    'effects': EFFECTS,
    'animation': ANIMATION,
}

def get_theme():
    """Return the complete theme configuration."""
    return THEME

def get_streamlit_config():
    """Return configuration for Streamlit's custom theming."""
    return {
        "primaryColor": COLORS['highlight'],
        "backgroundColor": COLORS['background'],
        "secondaryBackgroundColor": COLORS['light_accent'],
        "textColor": COLORS['text'],
        "font": TYPOGRAPHY['font_family']
    } 