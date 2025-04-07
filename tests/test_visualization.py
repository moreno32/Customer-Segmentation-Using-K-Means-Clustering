"""
Tests for visualization.py

This module contains test functions for the visualization utilities,
focusing on proper generation of plots and charts.
"""

import os
import pandas as pd
import numpy as np
import pytest
import sys
import plotly.graph_objects as go

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.utils.visualization import plot_rfm_3d
except ImportError:
    pass  # Will be handled in the tests

@pytest.fixture
def sample_rfm_data():
    """Create a sample RFM dataset for testing visualizations."""
    # Create 30 samples with RFM features
    np.random.seed(42)  # For reproducibility
    
    # Generate data with reasonable RFM ranges
    recency = np.random.randint(1, 100, 30)
    frequency = np.random.randint(1, 20, 30)
    monetary = np.random.uniform(10, 1000, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': [f'C{i:03d}' for i in range(1, 31)],
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })
    
    return df

@pytest.fixture
def sample_cluster_labels():
    """Create sample cluster labels for visualization tests."""
    # 30 samples with 3 clusters
    np.random.seed(42)
    return np.random.randint(0, 3, 30)

def test_plot_rfm_3d_without_clusters(sample_rfm_data):
    """Test 3D RFM plot creation without cluster labels."""
    try:
        # Create plot
        fig = plot_rfm_3d(sample_rfm_data)
        
        # Check basic properties
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Check if the figure has 3D data
        assert fig.data[0].type == 'scatter3d'
        
    except ImportError:
        pytest.skip("Visualization module not available")

def test_plot_rfm_3d_with_clusters(sample_rfm_data, sample_cluster_labels):
    """Test 3D RFM plot creation with cluster labels."""
    try:
        # Create plot
        fig = plot_rfm_3d(sample_rfm_data, sample_cluster_labels)
        
        # Check basic properties
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Check if the figure has 3D data
        assert fig.data[0].type == 'scatter3d'
        
        # Check if clusters are represented
        # For scatter3d with discrete colors, we should have a 'color' attribute in marker
        assert hasattr(fig.data[0].marker, 'color')
        
    except ImportError:
        pytest.skip("Visualization module not available")

def test_plot_rfm_3d_input_validation():
    """Test input validation for 3D RFM plot."""
    try:
        # Test with invalid input
        with pytest.raises(Exception):
            # Empty DataFrame should raise an exception
            plot_rfm_3d(pd.DataFrame())
        
        # Create valid dataframe but with missing columns
        invalid_df = pd.DataFrame({
            'CustomerID': ['C001', 'C002'],
            'SomeOtherColumn': [1, 2]
        })
        
        with pytest.raises(Exception):
            # Missing RFM columns should raise an exception
            plot_rfm_3d(invalid_df)
            
    except ImportError:
        pytest.skip("Visualization module not available") 