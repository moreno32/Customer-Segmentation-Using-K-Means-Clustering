"""
Styles Package

Provides consistent styling and UI components for the E-commerce Recommendation System.
"""

from .theme import THEME, COLORS, TYPOGRAPHY, SPACING, EFFECTS, get_theme, get_streamlit_config
from .components import (
    apply_custom_css, create_card, section_header, 
    metric_card, info_box, progress_steps, divider
)

__all__ = [
    'THEME', 'COLORS', 'TYPOGRAPHY', 'SPACING', 'EFFECTS',
    'get_theme', 'get_streamlit_config',
    'apply_custom_css', 'create_card', 'section_header',
    'metric_card', 'info_box', 'progress_steps', 'divider'
] 